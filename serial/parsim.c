#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define G 6.67408e-11
#define EPSILON2 (0.005 * 0.005)
#define DELTAT 0.1

#define _USE_MATH_DEFINES

typedef struct particle {
    int cellx, celly;
    double x, y, vx, vy, m;
    struct particle *prev, *next;
} particle_t;

typedef struct {
    double mass_sum;
    double cmx;
    double cmy;
    int adj_cells[8][2];  // Store (x, y) coordinates of 8 neighbors
    particle_t *head;
} cell_t;
long seed;

void init_r4uni(int input_seed) {
    seed = input_seed + 987654321;
}

double rnd_uniform01() {
    int seed_in = seed;
    seed ^= (seed << 13);
    seed ^= (seed >> 17);
    seed ^= (seed << 5);
    return 0.5 + 0.2328306e-09 * (seed_in + (int) seed);
}

double rnd_normal01() {
    double u1, u2, z, result;
    do {
        u1 = rnd_uniform01();
        u2 = rnd_uniform01();
        z = sqrt(-2 * log(u1)) * cos(2 * M_PI * u2);
        result = 0.5 + 0.15 * z;
    } while (result < 0 || result >= 1);
    return result;
}

// Initialize particles
void init_particles(long seed, double side, long ncside, long long n_part, particle_t *par)
{
    double (*rnd01)() = rnd_uniform01;
    long long i;

    if(seed < 0) {
        rnd01 = rnd_normal01;
        seed = -seed;
    }
    
    init_r4uni(seed);

    for(i = 0; i < n_part; i++) {
        par[i].x = rnd01() * side;
        par[i].y = rnd01() * side;
        par[i].vx = (rnd01() - 0.5) * side / ncside / 5.0;
        par[i].vy = (rnd01() - 0.5) * side / ncside / 5.0;

        par[i].m = rnd01() * 0.01 * (ncside * ncside) / n_part / G * EPSILON2;
    }
}

cell_t **init_cells(int grid_size, int space_size, long long number_particles, particle_t *particles) {
    cell_t **cells = (cell_t **)malloc(sizeof(cell_t *) * grid_size);
    if (!cells) return NULL;

    for (int i = 0; i < grid_size; i++) {
        cells[i] = (cell_t *)malloc(sizeof(cell_t) * grid_size);
        if (!cells[i]) return NULL;

        for (int j = 0; j < grid_size; j++) {
            // Initialize cell properties
            cells[i][j].mass_sum = 0;
            cells[i][j].cmx = 0;
            cells[i][j].cmy = 0;
            cells[i][j].head = NULL;

            // Compute adjacent cells with wraparound
            int adj_idx = 0;
            for (int dx = -1; dx <= 1; dx++) {
                for (int dy = -1; dy <= 1; dy++) {
                    if (dx == 0 && dy == 0) continue; // Skip the center cell

                    int ni = (i + dx + grid_size) % grid_size;  // Wrap in x direction
                    int nj = (j + dy + grid_size) % grid_size;  // Wrap in y direction

                    cells[i][j].adj_cells[adj_idx][0] = ni;
                    cells[i][j].adj_cells[adj_idx][1] = nj;
                    adj_idx++;
                }
            }
        }
    }

    for (long long i = 0; i < number_particles; i++) {
        particle_t *particle = &particles[i];
        
        // Determine the cell coordinates
        particle->cellx = (int)(particle->x / ((double)space_size / grid_size));
        particle->celly = (int)(particle->y / ((double)space_size / grid_size));
    
        // Insert particle at the head of the list
        particle->next = cells[particle->cellx][particle->celly].head;
        particle->prev = NULL;  // Since it's the new head, it has no previous element
    
        // Update the previous head's prev pointer to point to the new particle
        if (cells[particle->cellx][particle->celly].head != NULL) {
            cells[particle->cellx][particle->celly].head->prev = particle;
        }
    
        // Update the cell's head to the new particle
        cells[particle->cellx][particle->celly].head = particle;
    }
    
    return cells; 
}

// Compute the center of mass of each cell
void calculate_centers_of_mass(particle_t *particles, cell_t **cells, int grid_size, int space_size, int number_particles) {
    for (int i = 0; i < grid_size; i++) {
        for (int j = 0; j < grid_size; j++) {
            cells[i][j].mass_sum = 0;
            cells[i][j].cmx = 0;
            cells[i][j].cmy = 0;
        }
    }

    for (int i = 0; i < number_particles; i++) {
        printf("miau %d\n", i );
        particle_t *particle = &particles[i];
        cell_t *cell = &cells[particle->cellx][particle->celly];
        
        cell->mass_sum += particle->m;
        cell->cmx += particle->m * particle->x;
        cell->cmy += particle->m * particle->y;
    }

    for (int i = 0; i < grid_size; i++) {
        for (int j = 0; j < grid_size; j++) {
            cell_t *cell = &cells[i][j];
            if (cell->mass_sum != 0) {
                cell->cmx /= cell->mass_sum;
                cell->cmy /= cell->mass_sum;
            }
        }
    }
}

// Check for collisions between particles
int check_collisions(particle_t *particles, int *number_particles, cell_t **cells, int grid_size) {
    int collision_count = 0;
    for (int i = 0; i < grid_size; i++) {
        for (int j = 0; j < grid_size; j++) {
            for (particle_t *particle = cells[i][j].head; particle != NULL; particle = particle->next) {
                for (particle_t *other = particle->next; other != NULL; other = other->next) {
                    double dx = particle->x - other->x;
                    double dy = particle->y - other->y;
                    double dist2 = dx * dx + dy * dy;

                    if (dist2 <= EPSILON2) {
                        collision_count++;
                        // Remove the other particle
                        if (other->prev != NULL) {
                            other->prev->next = other->next;
                        } else {
                            cells[i][j].head = other->next;
                        }

                        if (other->next != NULL) {
                            other->next->prev = other->prev;
                        }

                        (*number_particles)--;

                        // Remove the other particle
                        if (particle->prev != NULL) {
                            particle->prev->next = particle->next;
                        } else {
                            cells[i][j].head = particle->next;
                        }

                        if (particle->next != NULL) {
                            particle->next->prev = particle->prev;
                        }

                        (*number_particles)--;
                    }
                }
            }
        }
    }

    return collision_count;
}



// Compute overall center of mass
void calculate_overall_center_of_mass(particle_t *particles, int number_particles, double *center_of_mass) {
    center_of_mass[0] = 0.0;
    center_of_mass[1] = 0.0;
    double total_mass = 0.0;

    for (int i = 0; i < number_particles; i++) {
        total_mass += particles[i].m;
        center_of_mass[0] += particles[i].m * particles[i].x;
        center_of_mass[1] += particles[i].m * particles[i].y;
    }

    if (total_mass > 0) {
        center_of_mass[0] /= total_mass;
        center_of_mass[1] /= total_mass;
    } else {
        center_of_mass[0] = 0.0;
        center_of_mass[1] = 0.0;
    }
}


void calculate_new_iteration(particle_t *particles, cell_t **cells, int grid_size, int space_size, int number_particles) {
    for (int i = 0; i < number_particles; i++) {
        particle_t *particle = &particles[i];
        printf("Before Particle %d: %f %f %f %f %f %d %d\n", i, particle->x, particle->y, particle->vx, particle->vy, particle->m, particle->cellx, particle->celly);

        // Compute the force acting on the particle
        double fx = 0, fy = 0;
        for (int i = 0; i < 8; i++) {
            int ni = cells[particle->cellx][particle->celly].adj_cells[i][0];
            int nj = cells[particle->cellx][particle->celly].adj_cells[i][1];
            cell_t *cell = &cells[ni][nj];

            double dx = cell->cmx - particle->x;
            double dy = cell->cmy - particle->y;
            double dist2 = dx * dx + dy * dy;

            if (dist2 == 0) continue;

            double f = G * particle->m * cell->mass_sum / dist2;
            fx += f * dx / sqrt(dist2);
            fy += f * dy / sqrt(dist2);
        }
        for (particle_t *other = cells[particle->cellx][particle->celly].head; other != NULL; other = other->next) {
            double dx = other->x - particle->x;
            double dy = other->y - particle->y;
            double dist2 = dx * dx + dy * dy;

            if (dist2 == 0) continue;

            double f = G * particle->m * other->m / dist2;
            fx += f * dx / sqrt(dist2);
            fy += f * dy / sqrt(dist2);
        }

        // Update the particle's position and velocity
        particle->vx += fx / particle->m * DELTAT;
        particle->vy += fy / particle->m * DELTAT;
        particle->x += particle->vx * DELTAT + 0.5 * fx / particle->m * DELTAT * DELTAT;
        particle->y += particle->vy * DELTAT + 0.5 * fy / particle->m * DELTAT * DELTAT;

        particle->x = fmod(fmod(particle->x, space_size) + space_size, space_size);
        particle->y = fmod(fmod(particle->y, space_size) + space_size, space_size);

        // Update cell position after movement
        particle->cellx = (int)(particle->x / ((double)space_size / grid_size));
        particle->celly = (int)(particle->y / ((double)space_size / grid_size));
        
        printf("After Particle %d: %f %f %f %f %f %d %d\n", i, particle->x, particle->y, particle->vx, particle->vy, particle->m, particle->cellx, particle->celly);
    }
}

// Main function
int main(int argc, char *argv[]) {
    if (argc != 6) {
        fprintf(stderr, "Expected 5 arguments, but %d were given\n", argc - 1);
        return 1;
    }

    seed = atol(argv[1]);
    double space_size = atof(argv[2]);
    int grid_size = atoi(argv[3]);
    long long number_particles = atoll(argv[4]);
    int n_time_steps = atoi(argv[5]);

    int collision_count = 0;

    particle_t *particles = (particle_t *)malloc(sizeof(particle_t) * number_particles);
    init_particles(seed, space_size, grid_size, number_particles, particles);
    cell_t **cells = init_cells(grid_size, space_size, number_particles, particles);

    for (int n = 0; n < n_time_steps; n++) {
        calculate_centers_of_mass(particles, cells, grid_size, space_size, number_particles);
        calculate_new_iteration(particles, cells, grid_size, space_size, number_particles);
        collision_count += check_collisions(particles, (int *)&number_particles, cells, grid_size);
        memset(cells[0], 0, sizeof(cell_t) * grid_size * grid_size);
    }
    double center_of_mass[2];  // Use double instead of int
    
    calculate_overall_center_of_mass(particles, number_particles, center_of_mass);
    printf("%.3f %.3f\n", center_of_mass[0], center_of_mass[1]);

    printf("%d\n", collision_count);

    free(particles);
    free(cells[0]);
    free(cells);
    return 0;
}



