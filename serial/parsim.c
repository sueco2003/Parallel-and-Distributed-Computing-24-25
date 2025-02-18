#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>
#include "../init_particles.h"

#define G 6.67408e-11
#define EPSILON2 (0.005 * 0.005)
#define DELTAT 0.1

#define _USE_MATH_DEFINES


cell_t **init_cells(int grid_size, double space_size, long long number_particles, particle_t *particles) {
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
            memset(cells[i][j].adj_cells, 0, sizeof(cells[i][j].adj_cells));

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
            printf("Cell %d %d: mass - %f, cmx - %f, cmy - %f\n", i, j, cell->mass_sum, cell->cmx, cell->cmy);
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

                    printf("Checking collision between %lf and %lf: dist2 - %f\n", particle->m, other->m, dist2);
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
                        
                        // TODO NAO ESTAO A SER REMOVIDAS CORRETAMENTE, FALTA LIMPAR O ARRAY
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

// Ver se só se fazem as alterações no final para todas, ou se é tipo Stochastic, vai-se alterando

void calculate_new_iteration(particle_t *particles, cell_t **cells, int grid_size, double space_size, int number_particles) {
    for (int i = 0; i < number_particles; i++) {
        particle_t *particle = &particles[i];
        printf("Before Particle %d: mass - %f, x - %f, y - %f, vx - %f, vy - %f, cx - %d, cy - %d\n", i, particle->m, particle->x, particle->y, particle->vx, particle->vy, particle->cellx, particle->celly);

        // Compute the force acting on the particle
        double fx = 0, fy = 0;
        bool is_edge_x = (particle->cellx == 0 || particle->cellx == grid_size - 1);
        bool is_edge_y = (particle->celly == 0 || particle->celly == grid_size - 1);

        // Loop through adjacent cells
        for (int i = 0; i < 8; i++) {
            int ni = cells[particle->cellx][particle->celly].adj_cells[i][0];
            int nj = cells[particle->cellx][particle->celly].adj_cells[i][1];
        
            cell_t *cell = &cells[ni][nj];
        
            double dx = cell->cmx - particle->x;
            double dy = cell->cmy - particle->y;

            // Compute shortest wraparound distance
            if (dx > space_size / 2 && is_edge_x) dx -= space_size;
            if (dx < -space_size / 2 && is_edge_x) dx += space_size;
            if (dy > space_size / 2 && is_edge_y) dy -= space_size;
            if (dy < -space_size / 2 && is_edge_y) dy += space_size;
        
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

        int previous_cellx = particle->cellx;
        int previous_celly = particle->celly;
        // Update cell position after movement
        particle->cellx = (int)(particle->x / ((double)space_size / grid_size));
        particle->celly = (int)(particle->y / ((double)space_size / grid_size));

        if (particle->cellx != previous_cellx || particle->celly != previous_celly) {
            // Remove the particle from the current cell
            if (particle->prev != NULL) {
                particle->prev->next = particle->next;
            } else {
                cells[previous_cellx][previous_celly].head = particle->next;
            }

            if (particle->next != NULL) {
                particle->next->prev = particle->prev;
            }

            // Insert the particle in the new cell
            particle->next = cells[particle->cellx][particle->celly].head;
            particle->prev = NULL;  // Since it's the new head, it has no previous element

            // Update the previous head's prev pointer to point to the new particle
            if (cells[particle->cellx][particle->celly].head != NULL) {
                cells[particle->cellx][particle->celly].head->prev = particle;
            }

            // Update the cell's head to the new particle
            cells[particle->cellx][particle->celly].head = particle;
        }
        
        printf("After Particle %d: mass - %f, x - %f, y - %f, vx - %f, vy - %f\n", i, particle->m, particle->x, particle->y, particle->vx, particle->vy);
    }
    printf("\n");
}

// Main function
int main(int argc, char *argv[]) {
    if (argc != 6) {
        fprintf(stderr, "Expected 5 arguments, but %d were given\n", argc - 1);
        return 1;
    }

    int seed = atol(argv[1]);
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
    printf("%.3f %.3f\n", particles[0].x, particles[0].y);

    printf("%d\n", collision_count);

    free(particles);
    for (int i = 0; i < grid_size; i++) {
        free(cells[i]);
    }
    free(cells);
    
    return 0;
}



