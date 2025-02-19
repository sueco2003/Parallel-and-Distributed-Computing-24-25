#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>
// #include <omp.h>
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

            // Initialize head of the list to NULL
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
    int cell_counter = 0;
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
            cell_t *cell = &cells[j][i];
            if (cell->mass_sum != 0) {
                cell->cmx /= cell->mass_sum;
                cell->cmy /= cell->mass_sum;
            }
            //printf("Cell %d x: %.3f, y: %.3f, m: %.3f\n", cell_counter, cell->cmx, cell->cmy, cell->mass_sum);
            cell_counter++;
        }
    }
}


// Check for collisions between particles
int check_collisions(particle_t *particles, cell_t **cells, int grid_size) {

    int collision_count = 0;

    // Detect collisions and mark particles for removal
    for (int i = 0; i < grid_size; i++) {
        for (int j = 0; j < grid_size; j++) {
            for (particle_t *particle = cells[i][j].head; particle != NULL; particle = particle->next) {
                if (particle->m == 0) continue;
                for (particle_t *other = particle->next; other != NULL; other = other->next) {

                    double dx = particle->x - other->x;
                    double dy = particle->y - other->y;
                    double dist2 = dx * dx + dy * dy;

                    // Ensure we only check unique pairs
                    if (dist2 <= EPSILON2) {
                        //printf("Collision detected\n");
                        collision_count++;
                        particle->m = 0;
                        other->m = 0;
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
        double fx = 0.0, fy = 0.0;

        if (particle->m == 0) continue;

        // Forças vindas das partículas na mesma célula
        for (particle_t *other = cells[particle->cellx][particle->celly].head; other != NULL; other = other->next) {
            if (other == particle)
                continue;

            double dx = other->x - particle->x;
            double dy = other->y - particle->y;
            double dist2 = dx * dx + dy * dy;

            if (dist2 == 0.0)
                continue;

            double dist = sqrt(dist2);
            double f = G * particle->m * other->m / dist2;
            double partial_fx = f * (dx / dist);
            double partial_fy = f * (dy / dist);

            if (i == 0) {
                //printf("P0/P[%p] mag: %.3f vecx: %.3f vecy: %.3f fx: %.3f fy: %.3f\n", 
                //(void*)other, f, (dx / dist), (dy / dist), partial_fx, partial_fy);
            }

            fx += partial_fx;
            fy += partial_fy;
        }

        // Forças vindas dos centros de massa das células adjacentes
        for (int c = 0; c < 8; c++) {
            int ni = cells[particle->cellx][particle->celly].adj_cells[c][0];
            int nj = cells[particle->cellx][particle->celly].adj_cells[c][1];
            cell_t *cell = &cells[ni][nj];
            double dx = cell->cmx - particle->x;
            double dy = cell->cmy - particle->y;

            // Ajusta para o wrap-around, se necessário
            if (particle->cellx == 0 && ni == grid_size - 1) dx -= space_size;
            if (particle->cellx == grid_size - 1 && ni == 0) dx += space_size;
            if (particle->celly == 0 && nj == grid_size - 1) dy -= space_size;
            if (particle->celly == grid_size - 1 && nj == 0) dy += space_size;

            double dist2 = dx * dx + dy * dy;
            if (dist2 == 0.0) continue;
            double dist = sqrt(dist2);
            double f = G * particle->m * cell->mass_sum / dist2;
            double partial_fx = f * (dx / dist);
            double partial_fy = f * (dy / dist);

            // Imprime os dados apenas para a partícula 0
            if (i == 0) {
                //printf("P0/C[%d,%d] mag: %.3f vecx: %.3f vecy: %.3f fx: %.3f fy: %.3f\n", ni, nj, f, (dx / dist), (dy / dist), partial_fx, partial_fy);
            }
            fx += partial_fx;
            fy += partial_fy;
        } 

        // Atualiza velocidade e posição da partícula
        particle->vx += (fx / particle->m) * DELTAT;
        particle->vy += (fy / particle->m) * DELTAT;
        particle->x += particle->vx * DELTAT + 0.5 * (fx / particle->m) * DELTAT * DELTAT;
        particle->y += particle->vy * DELTAT + 0.5 * (fy / particle->m) * DELTAT * DELTAT;

        int previous_cellx = particle->cellx;
        int previous_celly = particle->celly;

        // Atualiza a célula da partícula após o movimento
        particle->cellx = (int)(particle->x / (space_size / grid_size));
        particle->celly = (int)(particle->y / (space_size / grid_size));

        if (particle->cellx != previous_cellx || particle->celly != previous_celly) {

            // Remove a partícula da célula anterior
            if (particle->prev != NULL) {
                particle->prev->next = particle->next;
            }
            else {
                cells[previous_cellx][previous_celly].head = particle->next;
            }

            if (particle->next != NULL) {
                particle->next->prev = particle->prev;
            }

            // Insere a partícula na nova célula
            //printf("Particle %d moved from cell (%d, %d) to cell (%d, %d)\n", i, previous_cellx, previous_celly, particle->cellx, particle->celly);
            particle->next = cells[particle->cellx][particle->celly].head;
            particle->prev = NULL;
            if (cells[particle->cellx][particle->celly].head != NULL) {
                cells[particle->cellx][particle->celly].head->prev = particle;
            }
            cells[particle->cellx][particle->celly].head = particle;
        }
        
    }
}


void simulation(particle_t *particles, int grid_size, double space_size, long long number_particles, int n_time_steps) {
    
    int collision_count = 0;

    cell_t **cells = init_cells(grid_size, space_size, number_particles, particles);

    for (int n = 0; n < n_time_steps; n++) {
        printf("Time step %d\n\n", n);
        calculate_centers_of_mass(particles, cells, grid_size, space_size, number_particles);
        calculate_new_iteration(particles, cells, grid_size, space_size, number_particles);
        collision_count += check_collisions(particles, cells, grid_size);
    }
    
    printf("%.3f %.3f\n", particles[0].x, particles[0].y);

    printf("%d\n", collision_count);
}


// Main function
int main(int argc, char *argv[]) {

    if (argc != 6) {
        fprintf(stderr, "Expected 5 arguments, but %d were given\n", argc - 1);
        return 1;
    }

    long seed = atol(argv[1]);
    double space_size = atof(argv[2]);
    int grid_size = atoi(argv[3]);
    long long number_particles = atoll(argv[4]);
    int n_time_steps = atoi(argv[5]);
    int collision_count = 0;

    particle_t *particles = (particle_t *)malloc(sizeof(particle_t) * number_particles);

    if (!particles) {
        fprintf(stderr, "Failed to allocate memory for particles\n");
        return 1;
    }

    init_particles(seed, space_size, grid_size, number_particles, particles);

    // exec_time = -omp_get_wtime();

    simulation(particles, grid_size, space_size, number_particles, n_time_steps);   
    
    // exec_time += omp_get_wtime();
    // fprintf(stderr, "%.1fs\n", exec_time);
    
    // print_result();
}



