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

/**
 * Initializes a grid of cells and assigns particles to their respective cells.
 *
 * This function allocates memory for a 2D grid of cells based on the specified
 * grid size. Each cell is initialized with a list head set to NULL and its
 * adjacent cells are computed with wraparound logic. Particles are then assigned
 * to cells based on their coordinates, and inserted at the head of the cell's
 * particle list.
 *
 * @param grid_size The size of the grid (number of cells along one dimension).
 * @param space_size The physical size of the space being simulated.
 * @param number_particles The total number of particles to be placed in the grid.
 * @param particles An array of particles to be distributed across the grid.
 * @return A pointer to the 2D array of cells, or NULL if memory allocation fails.
 */
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


/**
 * Calculates the centers of mass for each cell in a grid based on particle data.
 *
 * This function iterates over a grid of cells, initializing each cell's mass sum
 * and center of mass coordinates to zero. It then processes each particle, updating
 * the mass sum and weighted position sums for the cell corresponding to each particle's
 * coordinates. Finally, it computes the center of mass for each cell by dividing the
 * weighted sums by the total mass sum, if the mass sum is non-zero.
 *
 * @param particles An array of particles, each containing mass and position data.
 * @param cells A 2D array of cells where each cell will have its center of mass calculated.
 * @param grid_size The number of cells along one dimension of the grid.
 * @param space_size The physical size of the space being simulated.
 * @param number_particles The total number of particles to be processed.
 */
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
            cell_counter++;
        }
    }
}


/**
 * Checks for collisions between particles within a grid of cells.
 *
 * This function iterates over each cell in a 2D grid and examines pairs of particles
 * to detect collisions. A collision is identified when the squared distance between
 * two particles is less than or equal to a predefined threshold (EPSILON2). When a
 * collision is detected, both particles are marked for removal by setting their mass
 * to zero. The function returns the total number of collisions detected.
 *
 * @param particles An array of particles to be checked for collisions.
 * @param cells A 2D array of cells containing particles.
 * @param grid_size The number of cells along one dimension of the grid.
 * @return The total number of collisions detected.
 */
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

/**
 * Updates the state of particles for a new iteration in a grid-based simulation.
 *
 * This function calculates the forces acting on each particle from other particles
 * within the same cell and from the centers of mass of adjacent cells. It updates
 * the velocity and position of each particle based on these forces and moves the
 * particle to a new cell if necessary. The function handles wrap-around logic for
 * particles near the grid boundaries.
 *
 * @param particles An array of particles to be updated.
 * @param cells A 2D array of cells containing particles and their centers of mass.
 * @param grid_size The number of cells along one dimension of the grid.
 * @param space_size The physical size of the space being simulated.
 * @param number_particles The total number of particles to be processed.
 */
void calculate_new_iteration(particle_t *particles, cell_t **cells, int grid_size, double space_size, long long number_particles) {

    for (long long i = 0; i < number_particles; i++) {

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

            fx += partial_fx;
            fy += partial_fy;
        } 

        // Atualiza velocidade e posição da partícula
        particle->vx += (fx / particle->m) * DELTAT;
        particle->vy += (fy / particle->m) * DELTAT;
        particle->x += particle->vx * DELTAT + 0.5 * (fx / particle->m) * DELTAT * DELTAT;
        particle->y += particle->vy * DELTAT + 0.5 * (fy / particle->m) * DELTAT * DELTAT;

        particle->x = fmod(particle->x, space_size);
        particle->y = fmod(particle->y, space_size);

        if (particle->x < 0) particle->x += space_size;
        if (particle->y < 0) particle->y += space_size;

        int previous_cellx = particle->cellx;
        int previous_celly = particle->celly;

        // Atualiza a célula da partícula após o movimento
        particle->cellx = (int)(particle->x / ((double)space_size / grid_size));
        particle->celly = (int)(particle->y / ((double)space_size / grid_size));

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

            particle->next = cells[particle->cellx][particle->celly].head;
            particle->prev = NULL;
            if (cells[particle->cellx][particle->celly].head != NULL) {
                cells[particle->cellx][particle->celly].head->prev = particle;
            }
            cells[particle->cellx][particle->celly].head = particle;
        }
        
    }
}

/**
 * Simulates the dynamics of particles in a grid-based space over a series of time steps.
 *
 * This function initializes a grid of cells and assigns particles to them. It then
 * iteratively calculates the centers of mass, updates particle states, and checks for
 * collisions over the specified number of time steps. The total number of collisions
 * detected during the simulation is returned.
 *
 * @param particles An array of particles to be simulated.
 * @param grid_size The number of cells along one dimension of the grid.
 * @param space_size The physical size of the space being simulated.
 * @param number_particles The total number of particles in the simulation.
 * @param n_time_steps The number of time steps to simulate.
 * @return The total number of collisions detected during the simulation.
 */
int simulation(particle_t *particles, int grid_size, double space_size, long long number_particles, int n_time_steps) {
    
    int collision_count = 0;

    cell_t **cells = init_cells(grid_size, space_size, number_particles, particles);

    for (int n = 0; n < n_time_steps; n++) {
        calculate_centers_of_mass(particles, cells, grid_size, space_size, number_particles);
        calculate_new_iteration(particles, cells, grid_size, space_size, number_particles);
        collision_count += check_collisions(particles, cells, grid_size);
    }

    return collision_count;
}

/**
 * Prints the position of the first particle and the total collision count.
 *
 * This function outputs the x and y coordinates of the first particle in the
 * provided array, formatted to three decimal places, followed by the total
 * number of collisions detected.
 *
 * @param particles An array of particles, where the first particle's position
 *                  will be printed.
 * @param collision_count The total number of collisions to be printed.
 */
void print_result(particle_t *particles, int collision_count) {
    printf("%.3f %.3f\n", particles[0].x, particles[0].y);
    printf("%d\n", collision_count);
}

/**
 * Main function to simulate particle dynamics in a grid-based space.
 *
 * This function initializes particles and simulates their dynamics over a series
 * of time steps. It expects five command-line arguments: a random seed, the physical
 * size of the space, the grid size, the number of particles, and the number of time
 * steps. The function allocates memory for particles, initializes them, and performs
 * the simulation, printing the position of the first particle and the total number
 * of collisions detected.
 *
 * @param argc The number of command-line arguments.
 * @param argv An array of command-line arguments.
 * @return Returns 1 if an error occurs, otherwise 0.
 */
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

    collision_count = simulation(particles, grid_size, space_size, number_particles, n_time_steps);   
    
    // exec_time += omp_get_wtime();
    // fprintf(stderr, "%.1fs\n", exec_time);
    print_result(particles, collision_count);
}



