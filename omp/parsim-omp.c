#include <math.h>
#include <omp.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "init_particles.h"

#define G 6.67408e-11
#define EPSILON2 (0.005 * 0.005)
#define DELTAT 0.1

#define _USE_MATH_DEFINES

/**
 * Initializes a grid of cells and assigns particles to their respective cells.
 *
 * This function allocates memory for a 2D grid of cells based on the specified
 * grid size. Each cell is initialized with a particles_inside head set to NULL and its
 * adjacent cells are computed with wraparound logic. Particles are then assigned
 * to cells based on their coordinates, and inserted at the head of the cell's
 * particle particles_inside.
 *
 * @param grid_size The size of the grid (number of cells along one dimension).
 * @param space_size The physical size of the space being simulated.
 * @param number_particles The total number of particles to be placed in the grid.
 * @param particles An array of particles to be distributed across the grid.
 * @return A pointer to the 2D array of cells, or NULL if memory allocation fails.
 */
cell_t **init_cells(int grid_size, double space_size, long long number_particles, particle_t *particles) {
    // Allocate memory for the grid of cells
    cell_t **cells = (cell_t **)malloc(sizeof(cell_t*) * grid_size);
    if (!cells) return NULL;
    // Allocate memory for each cell
    for (int i = 0; i < grid_size; i++) {
        cells[i] = (cell_t *)malloc(sizeof(cell_t) * grid_size);
        if (!cells[i]) return NULL;
    }

    // Initialize each cell
    #pragma omp parallel
    {
    #pragma omp for collapse(2)
    for (int i = 0; i < grid_size; i++) {
        for (int j = 0; j < grid_size; j++) {
            cell_t *cell = &cells[i][j];
            cell->mass_sum = 0;
            cell->cmx = 0;
            cell->cmy = 0;
            cell->capacity = number_particles * 10 / (grid_size * grid_size);
            cell->particles_inside = (particle_t **)malloc(sizeof(particle_t *) * cell->capacity);
            if (!cell->particles_inside) exit(1);
            cell->current_size = 0;
            omp_init_lock(&cell->lock);  // Initialize the lock

            // Compute adjacent cells with wraparound
            int adj_idx = 0;
            for (int dx = -1; dx <= 1; dx++) {
                for (int dy = -1; dy <= 1; dy++) {
                    // Skip the center cell
                    if (dx == 0 && dy == 0) continue;

                    // Compute the coordinates of the adjacent cell (handling wraparound)
                    int ni = (i + dx + grid_size) % grid_size;
                    int nj = (j + dy + grid_size) % grid_size;

                    // Store the coordinates of the adjacent cell
                    cells[i][j].adj_cells[adj_idx][0] = ni;
                    cells[i][j].adj_cells[adj_idx][1] = nj;
                    adj_idx++;
                }
            }
        }
    }

    #pragma omp for
    // Assign particles to cells based on their coordinates
    for (long long i = 0; i < number_particles; i++) {
        particle_t *particle = &particles[i];

        // Determine the cell coordinates
        int cellx = (particle->x / (space_size / grid_size));
        int celly = (particle->y / (space_size / grid_size));

        cell_t *cell = &cells[cellx][celly];

        // Lock only the specific cell being updated
        omp_set_lock(&cell->lock);

        // Insert particle at the head of the particles_inside
        if (cell->current_size >= cell->capacity) {
            cell->capacity *= 2;
            particle_t **temp = realloc(cell->particles_inside, sizeof(particle_t *) * cell->capacity);
            if (temp == NULL) {
                fprintf(stderr, "Failed to reallocate memory for particles inside cell\n");
                exit(1);
            }
            cell->particles_inside = temp;
        }
        cell->particles_inside[cell->current_size] = particle;
        cell->current_size++;

        omp_unset_lock(&cell->lock);
    }
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
 */
void calculate_centers_of_mass(particle_t *particles, cell_t **cells, int grid_size) {

    // Initialize mass sum and center of mass for each cell
    #pragma omp for collapse(2) schedule(dynamic, 1)
    for (int i = 0; i < grid_size; i++) {
        for (int j = 0; j < grid_size; j++) {
            cell_t *cell = &cells[i][j];
            cell->mass_sum = 0;
            cell->cmx = 0;
            cell->cmy = 0;
            // Calculate mass sum and weighted position sums
            #pragma omp simd
            for (long long idx = 0; idx < cell->current_size; idx++) {
                particle_t *particle = cell->particles_inside[idx];  // Pointer, not an array access
                cell->mass_sum += particle->m;
                cell->cmx += particle->m * particle->x;
                cell->cmy += particle->m * particle->y;
            }
            // Compute center of mass if mass sum
            if (cell->mass_sum != 0) {
                cell->cmx /= cell->mass_sum;
                cell->cmy /= cell->mass_sum;
            }
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
int check_collisions(cell_t **cells, int grid_size, int current_timestamp, int collision_count) {

    // Detect collisions and mark particles for removal
    #pragma omp for collapse(2) schedule(dynamic, 1)
    for (int i = 0; i < grid_size; i++) {
        for (int j = 0; j < grid_size; j++) {
            cell_t *cell = &cells[i][j];

            // Check for collisions between particles in the same cell
            for (long long idx = 0; idx < cell->current_size; idx++) {
                particle_t *particle = cell->particles_inside[idx];

                // Skip particles that are already dead (in previous timestamps)
                if (particle->death_timestamp < current_timestamp) continue;

                // Check for collisions with other particles in the same cell
                #pragma omp simd
                for (long long idx2 = idx + 1; idx2 < cell->current_size; idx2++) {
                    particle_t *other = cell->particles_inside[idx2];

                    // Skip if the other particle is already dead (in previous timestamps)
                    if (other->death_timestamp < current_timestamp) continue;

                    // Compute distance between particles
                    double dx = particle->x - other->x;
                    double dy = particle->y - other->y;
                    double dist2 = dx * dx + dy * dy;

                    // Check if the particles are in collision
                    if (dist2 <= EPSILON2) {
                        // If the collision is not part of a bigger collision, increment the collision count
                        if (particle->m != 0 && other->m != 0) {
                            collision_count++;
                        }
                        // Erase the particles
                        particle->m = 0;
                        other->m = 0;
                        particle->death_timestamp = current_timestamp;
                        other->death_timestamp = current_timestamp;
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

    #pragma omp for simd
    for (long long i = 0; i < number_particles; i++) {
        particle_t *particle = &particles[i];
        particle->fx = 0.0;
        particle->fy = 0.0;
    }

    // Compute forces acting on each particle
    #pragma omp for collapse(2) schedule(dynamic, 1)
    for (int cx = 0; cx < grid_size; cx++) {
        for (int cy = 0; cy < grid_size; cy++) {
            cell_t *cell = &cells[cx][cy];

            // Compute forces between particles in the same cell
            for (long long idx = 0; idx < cell->current_size; idx++) {
                particle_t *particle = cell->particles_inside[idx];

                if (particle->m == 0) continue;

                // Compute forces between particles in the same cell
                #pragma omp simd
                for (int idx2 = idx + 1; idx2 < cell->current_size; idx2++) {
                    particle_t *other = cell->particles_inside[idx2];

                    if (other->m == 0) continue;

                    // Compute force between particles
                    double dx = other->x - particle->x;
                    double dy = other->y - particle->y;
                    double dist2 = dx * dx + dy * dy;
                    double inv_dist = 1.0 / sqrt(dist2);

                    // Compute force components of gravity
                    double f = G * particle->m * other->m * inv_dist * inv_dist;
                    double fx = f * dx * inv_dist;
                    double fy = f * dy * inv_dist;
                    // Update forces of both particles
                    particle->fx += fx;
                    particle->fy += fy;
                    other->fx -= fx;
                    other->fy -= fy;
                }

                // Compute forces from adjacent centers of mass
                for (int c = 0; c < 8; c += 2) {
                    int ni1 = cell->adj_cells[c][0];
                    int nj1 = cell->adj_cells[c][1];
                    int ni2 = cell->adj_cells[c + 1][0];
                    int nj2 = cell->adj_cells[c + 1][1];
                
                    cell_t *adj_cell1 = &cells[ni1][nj1];
                    cell_t *adj_cell2 = &cells[ni2][nj2];
                    
                    // Skip cells with zero mass
                    if (adj_cell1->mass_sum > 0) {
                        // Compute force between particle and center of mass
                        double dx = adj_cell1->cmx - particle->x;
                        double dy = adj_cell1->cmy - particle->y;

                        // Adjusts for wrap-around if necessary
                        if (cx == 0 && ni1 == grid_size - 1)
                            dx -= space_size;
                        else if (cx == grid_size - 1 && ni1 == 0)
                            dx += space_size;
                        if (cy == 0 && nj1 == grid_size - 1)
                            dy -= space_size;
                        else if (cy == grid_size - 1 && nj1 == 0)
                            dy += space_size;

                        double dist2 = dx * dx + dy * dy;
                        double inv_dist = 1.0 / sqrt(dist2);

                        // Add gravitational force due to center of mass to the total force
                        double f = G * particle->m * adj_cell1->mass_sum * inv_dist * inv_dist;
                        particle->fx += f * dx * inv_dist;
                        particle->fy += f * dy * inv_dist;
                    }
                    // Skip cells with zero mass
                    if (adj_cell2->mass_sum > 0) {
                        // Compute force between particle and center of mass
                        double dx = adj_cell2->cmx - particle->x;
                        double dy = adj_cell2->cmy - particle->y;

                        // Adjusts for wrap-around if necessary
                        if (cx == 0 && ni2 == grid_size - 1)
                            dx -= space_size;
                        else if (cx == grid_size - 1 && ni2 == 0)
                            dx += space_size;
                        if (cy == 0 && nj2 == grid_size - 1)
                            dy -= space_size;
                        else if (cy == grid_size - 1 && nj2 == 0)
                            dy += space_size;

                        double dist2 = dx * dx + dy * dy;
                        double inv_dist = 1.0 / sqrt(dist2);

                        // Add gravitational force due to center of mass to the total force
                        double f = G * particle->m * adj_cell2->mass_sum * inv_dist * inv_dist;
                        particle->fx += f * dx * inv_dist;
                        particle->fy += f * dy * inv_dist;
                    }

                    
                }
            }
            // Delete all particles in cell
            cell->current_size = 0;
        }
    }

    // Update particles
    #pragma omp for
    for (long long i = 0; i < number_particles; i++) {
        particle_t *particle = &particles[i];

        // Skip particles that are already dead
        if (particle->m == 0) continue;

        // Compute x and y components of acceleration and update position and velocity
        double ax = particle->fx / particle->m;
        double ay = particle->fy / particle->m;
        particle->x += particle->vx * DELTAT + 0.5 * ax * DELTAT * DELTAT;
        particle->y += particle->vy * DELTAT + 0.5 * ay * DELTAT * DELTAT;
        particle->vx += ax * DELTAT;
        particle->vy += ay * DELTAT;

        // Wrap around logic
        if (particle->x >= space_size) particle->x -= space_size;
        if (particle->x < 0) particle->x += space_size;
        if (particle->y >= space_size) particle->y -= space_size;
        if (particle->y < 0) particle->y += space_size;

        // Compute new cell coordinates
        int cellx = (particle->x / (space_size / grid_size));
        int celly = (particle->y / (space_size / grid_size));

        cell_t *cell = &cells[cellx][celly];

        // Lock only the specific cell being updated
        omp_set_lock(&cell->lock);

        // Insert particle at the head of the particles_inside
        if (cell->current_size >= cell->capacity) {
            cell->capacity *= 2;
            particle_t **temp = realloc(cell->particles_inside, sizeof(particle_t *) * cell->capacity);
            if (temp == NULL) {
                fprintf(stderr, "Failed to reallocate memory for particles inside cell\n");
                exit(1);
            }
            cell->particles_inside = temp;
        }
        cell->particles_inside[cell->current_size] = particle;
        cell->current_size++;

        omp_unset_lock(&cell->lock);
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

    // Initialize the grid of cells and assign particles to them
    cell_t **cells = init_cells(grid_size, space_size, number_particles, particles);

    #pragma omp parallel reduction(+:collision_count)
    {   
        // Simulation loop (compute centers of mass, update particles, check collisions)
        for (int n = 0; n < n_time_steps; n++) {
            calculate_centers_of_mass(particles, cells, grid_size);
            calculate_new_iteration(particles, cells, grid_size, space_size, number_particles);
            collision_count = check_collisions(cells, grid_size, n, collision_count);
        }
    } 

    for (int i = 0; i < grid_size; i++) {
        for (int j = 0; j < grid_size; j++) {
            omp_destroy_lock(&cells[i][j].lock);  // Destroy locks
            free(cells[i][j].particles_inside);
        }
        free(cells[i]);
    }
    free(cells);

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
    // Print the position of the first particle
    printf("%.3f %.3f\n", particles[0].x, particles[0].y);

    // Print the total number of collisions
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
    // Check for the correct number of command-line arguments
    if (argc != 6) {
        fprintf(stderr, "Expected 5 arguments, but %d were given\n", argc - 1);
        return 1;
    }

    // Parse command-line arguments and initialize variables
    long seed = atol(argv[1]);
    double space_size = atof(argv[2]);
    int grid_size = atoi(argv[3]);
    long long number_particles = atoll(argv[4]);
    int n_time_steps = atoi(argv[5]);
    int collision_count = 0;
    double exec_time = 0;
    particle_t *particles = (particle_t *)malloc(sizeof(particle_t) * number_particles);

    if (!particles) {
        fprintf(stderr, "Failed to allocate memory for particles\n");
        return 1;
    }

    // Initialize particles with random positions and velocities
    init_particles(seed, space_size, grid_size, number_particles, particles);

    // Run the simulation and measure the execution time
    exec_time = -omp_get_wtime();
    collision_count = simulation(particles, grid_size, space_size, number_particles, n_time_steps);
    exec_time += omp_get_wtime();

    // Print the results and time of execution
    print_result(particles, collision_count);
    fprintf(stderr, "%.1fs\n", exec_time);

    // Free the memory allocated for the particles
    free(particles);
}