#include <math.h>
#include <omp.h>
#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "init_particles.h"

#define G 6.67408e-11
#define EPSILON2 (0.005 * 0.005)
#define DELTAT 0.1

#define _USE_MATH_DEFINES

// #define BLOCK_LOW(id,p,n) ((id)*(n)/(p))
// #define BLOCK_HIGH(id,p,n) (BLOCK_LOW((id)+1,p,n) - 1)
// #define BLOCK_SIZE(id,p,n) (BLOCK_HIGH(id,p,n) - BLOCK_LOW(id,p,n) + 1)
// #define BLOCK_OWNER(index,p,n) (((p)*((index)+1)-1)/(n))

enum { SEND_DOWN_ROW, SEND_UP_ROW };

MPI_Datatype LEAN_CELL_TYPE;
MPI_Datatype PARTICLE_TYPE;

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
cell_t **init_cells(int grid_size, double space_size, long long number_particles, particle_t *particles, particle_array_t *local_particles, int block_size, int block_low, int block_high) {
    
    // Allocate memory for the grid of cells
    cell_t **cells = (cell_t **)malloc(sizeof(cell_t*) * block_size);
    if (!cells) return NULL;
    // Allocate memory for each cell
    for (int i = 0; i < block_size; i++) {
        cells[i] = (cell_t *)malloc(sizeof(cell_t) * grid_size);
        if (!cells[i]) return NULL;
    }

    // Initialize each cell
    for (int i = 0; i < block_size; i++) {
        for (int j = 0; j < grid_size; j++) {
            cell_t *cell = &cells[i][j];
            cell->mass_sum = 0;
            cell->cmx = 0;
            cell->cmy = 0;
            cell->capacity = number_particles * 10 / (grid_size * grid_size);
            cell->particles_inside = (particle_t **)malloc(sizeof(particle_t *) * cell->capacity);
            if (!cell->particles_inside) exit(1);
            cell->current_size = 0;

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

    // Assign particles to cells and local_particles based on their coordinates
    for (long long i = 0; i < number_particles; i++) {
        particle_t *particle = &particles[i];

        // Determine the cell coordinates
        int cellx = (particle->x / (space_size / grid_size));
        if (cellx < block_low || cellx > block_high) continue;
        int celly = (particle->y / (space_size / grid_size));

        // Add particle to local particles array
        if (local_particles->current_size >= local_particles->capacity) {
            local_particles->capacity *= 2;
            particle_t **temp = realloc(local_particles->particles, sizeof(particle_t*) * local_particles->capacity);
            if (temp == NULL) {
                fprintf(stderr, "Failed to reallocate memory for local particles\n");
                exit(1);
            }
            local_particles->particles = temp;
        }
        local_particles->particles[local_particles->current_size] = particle;
        local_particles->current_size++;

        cell_t *cell = &cells[cellx-block_low][celly];

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
 * @param cells A 2D array of cells where each cell will have its center of mass calculated.
 * @param grid_size The number of cells along one dimension of the grid.
 * @param block_size The number of rows in the block.
 */
void calculate_centers_of_mass(cell_t **cells, int grid_size, int block_size, lean_cell_t **center_of_mass_buffer_recv, lean_cell_t **center_of_mass_buffer_send, int rank, int num_procs, int num_blocks_frontier, MPI_Request *com_requests_send, MPI_Status *com_statuses_send, MPI_Request *com_requests_recv, MPI_Status *com_statuses_recv, int node_before, int node_after) {

    // Initialize mass sum and center of mass for each cell
    for (int i = 0; i < block_size; i++) {
        for (int j = 0; j < grid_size; j++) {
            cell_t *cell = &cells[i][j];
            cell->mass_sum = 0;
            cell->cmx = 0;
            cell->cmy = 0;
            // Calculate mass sum and weighted position sums
            #pragma omp simd
            for (long long idx = 0; idx < cell->current_size; idx++) {
                particle_t *particle = cell->particles_inside[idx];
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

    // Send and receive center of mass data
    int row_idx = 0;

    for (int cx = 0; cx < num_blocks_frontier; cx++) {
        if (cx != 0) row_idx = block_size - 1;
        for (int cy = 0; cy < grid_size; cy++) {
            center_of_mass_buffer_send[cx][cy].mass_sum = cells[row_idx][cy].mass_sum;
            center_of_mass_buffer_send[cx][cy].cmx = cells[row_idx][cy].cmx;
            center_of_mass_buffer_send[cx][cy].cmy = cells[row_idx][cy].cmy;
        }
        if (cx == 0) {
            MPI_Isend(center_of_mass_buffer_send[0], grid_size, LEAN_CELL_TYPE, node_before, SEND_UP_ROW, MPI_COMM_WORLD, &com_requests_send[0]);
            MPI_Irecv(center_of_mass_buffer_recv[0], grid_size, LEAN_CELL_TYPE, node_before, SEND_DOWN_ROW, MPI_COMM_WORLD, &com_requests_recv[0]);
        } else {
            MPI_Isend(center_of_mass_buffer_send[1], grid_size, LEAN_CELL_TYPE, node_after, SEND_DOWN_ROW, MPI_COMM_WORLD, &com_requests_send[1]);
            MPI_Irecv(center_of_mass_buffer_recv[1], grid_size, LEAN_CELL_TYPE, node_after, SEND_UP_ROW, MPI_COMM_WORLD, &com_requests_recv[1]);
        }
    }
    MPI_Waitall(num_blocks_frontier, com_requests_send, com_statuses_send);
    MPI_Waitall(num_blocks_frontier, com_requests_recv, com_statuses_recv);
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
int check_collisions(cell_t **cells, int grid_size, int current_timestamp, int collision_count, int block_size) {

    // Detect collisions and mark particles for removal
    for (int i = 0; i < block_size; i++) {
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
void calculate_new_iteration(particle_array_t *local_particles, cell_t **cells, int grid_size, double space_size, int block_size, lean_cell_t **center_of_mass_buffer_recv, int rank, int num_procs, int block_low, int block_high, int node_before, int node_after, particle_array_t **particles_buffer_send, particle_array_t **particles_buffer_recv, MPI_Request *particles_requests_send, MPI_Status *particles_statuses_send, MPI_Request *particles_requests_recv, MPI_Status *particles_statuses_recv) {

    int reduced_size = block_size - 1;

    #pragma omp for simd
    for (long long i = 0; i < local_particles->current_size; i++) {
        particle_t *particle = local_particles->particles[i];
        particle->fx = 0.0;
        particle->fy = 0.0;
    }

    // Compute forces acting on each particle in cells other than frontier cells
    for (int cx = 1; cx < reduced_size; cx++) {
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

    int iteration_size = (block_size == 1) ? 1 : 2;
    int row_idx = 0;

    // Compute forces acting on each particle in frontier cells
    for (int cx = 0; cx < iteration_size; cx++) {
        if (cx != 0) row_idx = reduced_size;

        int initial_adj = 0;
        int final_adj = 8;

        if(block_size == 1) {
            initial_adj = 3;
            final_adj = 5;
        }

        else if (cx == 0) {
            initial_adj = 3;
        }
        else {
            final_adj = 5;
        }

        for (int cy = 0; cy < grid_size; cy++) {
            cell_t *cell = &cells[row_idx][cy];

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
                // Compute forces from adjacent centers of mass that are inside the same process
                for (int c = initial_adj; c < final_adj; c ++) {
                    int ni = cell->adj_cells[c][0];
                    int nj = cell->adj_cells[c][1];
                
                    cell_t *adj_cell = &cells[ni][nj];
                    
                    // Skip cells with zero mass
                    if (adj_cell->mass_sum > 0) {
                        // Compute force between particle and center of mass
                        double dx = adj_cell->cmx - particle->x;
                        double dy = adj_cell->cmy - particle->y;

                        double dist2 = dx * dx + dy * dy;
                        double inv_dist = 1.0 / sqrt(dist2);

                        // Add gravitational force due to center of mass to the total force
                        double f = G * particle->m * adj_cell->mass_sum * inv_dist * inv_dist;
                        particle->fx += f * dx * inv_dist;
                        particle->fy += f * dy * inv_dist;
                    }
                }

                // Compute forces from adjacent centers of mass that from other process
                for (int c = 0; c < 3; c ++) {
                    int column_index = cy + c - 1;
                    if (column_index < 0) {
                        column_index = grid_size - 1;
                    }
                    else if (column_index == grid_size) {
                        column_index = 0;
                    }

                    lean_cell_t *lean_cell = &center_of_mass_buffer_recv[cx][column_index];
                    
                    // Skip cells with zero mass
                    if (lean_cell->mass_sum > 0) {
                        // Compute force between particle and center of mass
                        double dx = lean_cell->cmx - particle->x;
                        double dy = lean_cell->cmy - particle->y;

                        // Adjusts for wrap-around if necessary
                        if (rank == 0 && cx == 0) {
                            dx -= space_size;
                        }
                        else if (rank == num_procs - 1 && cx == 1) {
                            dx += space_size;
                        }
                        if (cy == 0 && c == 0) {
                            dy -= space_size;
                        }
                        else if (cy == grid_size - 1 && c == 2) {
                            dy += space_size;
                        }

                        // if (cx == 0 && ni1 == grid_size - 1)
                        //     dx -= space_size;
                        // else if (cx == grid_size - 1 && ni1 == 0)
                        //     dx += space_size;
                        // if (cy == 0 && nj1 == grid_size - 1)
                        //     dy -= space_size;
                        // else if (cy == grid_size - 1 && nj1 == 0)
                        //     dy += space_size;

                        double dist2 = dx * dx + dy * dy;
                        double inv_dist = 1.0 / sqrt(dist2);

                        // Add gravitational force due to center of mass to the total force
                        double f = G * particle->m * lean_cell->mass_sum * inv_dist * inv_dist;
                        particle->fx += f * dx * inv_dist;
                        particle->fy += f * dy * inv_dist;
                    }
                }
            }
            // Delete all particles in cell
            cell->current_size = 0;
        }
    }

    // Reset send buffers
    for (int i = 0; i < 2; i++) {
        particles_buffer_send[i]->current_size = 0;
    }

    int receive_size[2];
    bool on_block_before, on_block_after;

    // Update particles
    for (long long i = 0; i < local_particles->current_size; i++) {

        particle_t *particle = local_particles->particles[i];

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

        on_block_before = cellx < block_low || (cellx == grid_size - 1 && block_low == 0);
        on_block_after = cellx > block_high || (cellx == 0 && block_high == grid_size - 1);

        if (on_block_before) {
            if (particles_buffer_send[0]->current_size >= particles_buffer_send[0]->capacity) {
                particles_buffer_send[0]->capacity *= 2;
                particle_t **temp = realloc(particles_buffer_send[0], sizeof(particle_t *) * particles_buffer_send[0]->capacity);
                if (temp == NULL) {
                    fprintf(stderr, "Failed to reallocate memory for particles buffer send 0\n");
                    exit(1);
                }
                particles_buffer_send[0]->particles = temp;
            }
            particles_buffer_send[0]->particles[particles_buffer_send[0]->current_size] = particle;
            particles_buffer_send[0]->current_size++;

            // Remove particle from local particles array
            local_particles->particles[i] = local_particles->particles[local_particles->current_size - 1];
            local_particles->current_size--;
            continue;
        }
        if (on_block_after) {
            if (particles_buffer_send[1]->current_size >= particles_buffer_send[1]->capacity) {
                particles_buffer_send[1]->capacity *= 2;
                particle_t **temp = realloc(particles_buffer_send[1], sizeof(particle_t *) * particles_buffer_send[1]->capacity);
                if (temp == NULL) {
                    fprintf(stderr, "Failed to reallocate memory for particles buffer send 0\n");
                    exit(1);
                }
                particles_buffer_send[1]->particles = temp;
            }
            particles_buffer_send[1]->particles[particles_buffer_send[1]->current_size] = particle;
            particles_buffer_send[1]->current_size++;

            // Remove particle from local particles array
            local_particles->particles[i] = local_particles->particles[local_particles->current_size - 1];
            local_particles->current_size--;
            continue;
        }

        cell_t *cell = &cells[cellx][celly];

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
    }
    printf("Before rank %d\n", rank);
    fflush(stdout);
    MPI_Isend(particles_buffer_send[0]->particles, particles_buffer_send[0]->current_size, PARTICLE_TYPE, node_before, SEND_UP_ROW, MPI_COMM_WORLD, &particles_requests_send[0]);
    MPI_Isend(particles_buffer_send[1]->particles, particles_buffer_send[1]->current_size, PARTICLE_TYPE, node_after, SEND_DOWN_ROW, MPI_COMM_WORLD, &particles_requests_send[1]);
    MPI_Waitall(2, particles_requests_send, particles_statuses_send);
    printf("Rank %d: node_before=%d , node_after=%d\n", rank, node_before, node_after);
    printf("Before 2 rank %d, particles_buffer_send[0]->current_size = %d, particles_buffer_send[1]->current_size = %d\n", rank, particles_buffer_send[0]->current_size, particles_buffer_send[1]->current_size);
    fflush(stdout);
    // SUBSTITUTE WITH A TEST INSTEAD SO WE CAN MAKE THE BUFFER SMALLER
    // CREATE A NEW MPI_TYPE FOR PARTICLE_ARRAY_T
    MPI_Probe(node_before, SEND_DOWN_ROW, MPI_COMM_WORLD, &particles_statuses_recv[0]);
    MPI_Probe(node_after, SEND_UP_ROW, MPI_COMM_WORLD, &particles_statuses_recv[1]);
    MPI_Get_count(&particles_statuses_recv[0], PARTICLE_TYPE, &receive_size[0]);
    MPI_Get_count(&particles_statuses_recv[1], PARTICLE_TYPE, &receive_size[1]);
    printf("Before 3 rank %d, receive_size[0] = %d, receive_size[1] = %d\n", rank, receive_size[0], receive_size[1]);
    fflush(stdout);
    MPI_Irecv(particles_buffer_recv[0]->particles, receive_size[0], PARTICLE_TYPE, node_before, SEND_DOWN_ROW, MPI_COMM_WORLD, &particles_requests_recv[0]);
    MPI_Irecv(particles_buffer_recv[1]->particles, receive_size[1], PARTICLE_TYPE, node_after, SEND_UP_ROW, MPI_COMM_WORLD, &particles_requests_recv[1]);
    MPI_Waitall(2, particles_requests_recv, particles_statuses_recv);

    printf("Rank %d: Particles updated\n", rank);
    fflush(stdout);

    for (int i = 0; i < 2; i ++) {
        for (int j = 0; j < receive_size[i]; j++) {
            particle_t *particle = particles_buffer_recv[i]->particles[j];

            // Add particle to local particles array
            if (local_particles->current_size >= local_particles->capacity) {
                local_particles->capacity *= 2;
                particle_t **temp = realloc(local_particles->particles, sizeof(particle_t *) * local_particles->capacity);
                if (temp == NULL) {
                    fprintf(stderr, "Failed to reallocate memory for local particles\n");
                    exit(1);
                }
                local_particles->particles = temp;
            }
            local_particles->particles[local_particles->current_size] = particle;
            local_particles->current_size++;

            // Compute new cell coordinates
            int cellx = (particle->x / (space_size / grid_size));
            int celly = (particle->y / (space_size / grid_size));

            cell_t *cell = &cells[cellx][celly];

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
        }
    }
}


/**
 * Initializes MPI datatypes for lean_cell_t and particle_t structures.
 *
 * This function creates and commits MPI datatypes for the lean_cell_t and
 * particle_t structures, allowing for efficient communication of these data
 * types between MPI processes.
 */
void init_mpi_datatypes() {

    // Initialize lean_cell_t MPI datatype
    int lean_cell_block_lengths[1] = {3};
    MPI_Aint base_address_lean;
    MPI_Aint lean_cell_displacements[1]; 
    lean_cell_t lean_cell;
    MPI_Get_address(&lean_cell, &base_address_lean);
    MPI_Get_address(&lean_cell.mass_sum, &lean_cell_displacements[0]);
    lean_cell_displacements[0] -= base_address_lean;
    MPI_Datatype lean_cell_types[1] = {MPI_DOUBLE};
    MPI_Type_create_struct(1, lean_cell_block_lengths, lean_cell_displacements, lean_cell_types, &LEAN_CELL_TYPE);
    MPI_Type_commit(&LEAN_CELL_TYPE);

    // Initialize particle_t MPI datatype
    int particle_block_lengths[1] = {8};
    MPI_Aint base_address_particle;
    MPI_Aint particle_displacements[1];
    particle_t particle;
    MPI_Get_address(&particle, &base_address_particle);
    MPI_Get_address(&particle.x, &particle_displacements[0]);
    particle_displacements[0] -= base_address_particle;
    MPI_Datatype particle_types[1] = {MPI_DOUBLE};
    MPI_Type_create_struct(1, particle_block_lengths, particle_displacements, particle_types, &PARTICLE_TYPE);
    MPI_Type_commit(&PARTICLE_TYPE);
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
int simulation(particle_t *particles, int grid_size, double space_size, long long number_particles, int n_time_steps, int argc, char *argv[]) {

    // Initialize MPI variables
    int rank, num_procs;

    // Initialize block size, low, and high
    int block_size, block_low, block_high, num_blocks_frontier;
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Create lean_cell_t and particle_t MPI datatypes
    init_mpi_datatypes();

    // Initialize collision count
    int collision_count = 0;

    // Calculate block size, low, and high
    block_size = BLOCK_SIZE(rank, num_procs, grid_size);
    block_low = BLOCK_LOW(rank, num_procs, grid_size);
    block_high = BLOCK_HIGH(rank, num_procs, grid_size);
    num_blocks_frontier = (block_size == 1) ? 1 : 2;

    // Allocate memory for local particles
    particle_array_t *local_particles = (particle_array_t *)malloc(sizeof(particle_array_t));
    if (!local_particles) {
        fprintf(stderr, "Failed to allocate memory for local particles\n");
        return 1;
    }

    local_particles->current_size = 0;
    local_particles->capacity = number_particles * 2 / num_procs;
    local_particles->particles = (particle_t **)malloc(sizeof(particle_t *) * local_particles->capacity);

    if (!local_particles->particles) {
        fprintf(stderr, "Failed to allocate memory for local particles array\n");
        return 1;
    }

    // Initialize buffers
    lean_cell_t **center_of_mass_buffer_send = (lean_cell_t **)malloc(sizeof(lean_cell_t *) * 2);
    if (center_of_mass_buffer_send == NULL) {
        fprintf(stderr, "Failed to allocate memory for center of mass buffer send\n");
        return 1;
    }

    lean_cell_t **center_of_mass_buffer_recv = (lean_cell_t **)malloc(sizeof(lean_cell_t *) * 2);
    if (center_of_mass_buffer_recv == NULL) {
        fprintf(stderr, "Failed to allocate memory for center of mass buffer recv\n");
        return 1;
    }

    particle_array_t **particles_buffer_send = (particle_array_t**)malloc(sizeof(particle_array_t*) * 2);
    if (particles_buffer_send == NULL) {
        fprintf(stderr, "Failed to allocate memory for particles buffer send\n");
        exit(1);
    }

    particle_array_t **particles_buffer_recv = (particle_array_t**) malloc(sizeof(particle_array_t*) * 2);
    if (particles_buffer_recv == NULL) {
        fprintf(stderr, "Failed to allocate memory for particles buffer send\n");
        exit(1);
    }

    // Allocate memory for each particle_array_t instance
    for (int i = 0; i < 2; i++) {
        particles_buffer_send[i] = malloc(sizeof(particle_array_t));
        if (particles_buffer_send[i] == NULL) {
            fprintf(stderr, "Failed to allocate memory for particles_buffer_send[%d]\n", i);
            exit(1);
        }
        particles_buffer_recv[i] = malloc(sizeof(particle_array_t));
        if (particles_buffer_recv[i] == NULL) {
            fprintf(stderr, "Failed to allocate memory for particles_buffer_recv[%d]\n", i);
            exit(1);
        }
    }

    for (int i = 0; i < 2; i++) {
        center_of_mass_buffer_send[i] = (lean_cell_t *)malloc(sizeof(lean_cell_t) * grid_size);
        if (center_of_mass_buffer_send[i] == NULL) {
            fprintf(stderr, "Failed to allocate memory for center of mass buffer send\n");
            return 1;
        }

        center_of_mass_buffer_recv[i] = (lean_cell_t *)malloc(sizeof(lean_cell_t) * grid_size);
        if (center_of_mass_buffer_recv[i] == NULL) {
            fprintf(stderr, "Failed to allocate memory for center of mass buffer recv\n");
            return 1;
        }

        particles_buffer_send[i]->current_size = 0;
        particles_buffer_send[i]->capacity = local_particles->capacity * num_blocks_frontier / block_size * 2;
        particles_buffer_send[i]->particles = (particle_t **)malloc(sizeof(particle_t *) * particles_buffer_send[i]->capacity);
        if (particles_buffer_send[i]->particles == NULL) {
            fprintf(stderr, "Failed to allocate memory for particles buffer send\n");
            exit(1);
        }

        particles_buffer_recv[i]->current_size = 0;
        particles_buffer_recv[i]->capacity = number_particles;//local_particles->capacity * num_blocks_frontier / block_size * 2;
        particles_buffer_recv[i]->particles = (particle_t **)malloc(sizeof(particle_t *) * particles_buffer_recv[i]->capacity);
        if (particles_buffer_recv[i]->particles == NULL) {
            fprintf(stderr, "Failed to allocate memory for particles buffer send\n");
            exit(1);
        }
    }

    // Calculate previous and next nodes
    int node_before;
    if (rank == 0) {
        node_before = num_procs - 1;
    }
    else {
        node_before = rank - 1;
    }

    int node_after;
    if (rank == num_procs - 1) {
        node_after = 0;
    }
    else {
        node_after = rank + 1;
    }

    // Create requests and statuses for MPI communication
    MPI_Request com_requests_send[num_blocks_frontier];
    MPI_Request com_requests_recv[num_blocks_frontier];
    MPI_Status com_statuses_send[num_blocks_frontier];
    MPI_Status com_statuses_recv[num_blocks_frontier];
    MPI_Request particles_requests_send[num_blocks_frontier];
    MPI_Request particles_requests_recv[num_blocks_frontier];
    MPI_Status particles_statuses_send[num_blocks_frontier];
    MPI_Status particles_statuses_recv[num_blocks_frontier];

    // Initialize the grid of cells and assign particles to them
    cell_t **cells = init_cells(grid_size, space_size, number_particles, particles, local_particles, block_size, block_low, block_high);

    // Simulation loop (compute centers of mass, update particles, check collisions)
    for (int n = 0; n < n_time_steps; n++) {
        printf("Rank %d\n", rank);
        fflush(stdout);
        calculate_centers_of_mass(cells, grid_size, block_size, center_of_mass_buffer_send, center_of_mass_buffer_recv, rank, num_procs, num_blocks_frontier, com_requests_send, com_statuses_send, com_requests_recv, com_statuses_recv, node_before, node_after);
        printf("Rank %d\n", rank);
        fflush(stdout);
        calculate_new_iteration(local_particles, cells, grid_size, space_size, block_size, center_of_mass_buffer_recv, rank, num_procs, block_high, block_low, node_before, node_after, particles_buffer_send, particles_buffer_recv, particles_requests_send, particles_statuses_send, particles_requests_recv, particles_statuses_recv);
        printf("Rank %d\n", rank);
        fflush(stdout);
        collision_count = check_collisions(cells, grid_size, n, collision_count, block_size);
    }

    // Free memory allocated for buffers
    for (int i = 0; i < 2; i++) {
        free(center_of_mass_buffer_send[i]);
        free(center_of_mass_buffer_recv[i]);
        free(particles_buffer_send[i]);
        free(particles_buffer_recv[i]);
    }
    free(center_of_mass_buffer_send);
    free(center_of_mass_buffer_recv);
    free(particles_buffer_send);
    free(particles_buffer_recv);

    // Free memory allocated for local particles
    free(local_particles->particles);
    free(local_particles);

    // Free cells
    for (int i = 0; i < grid_size; i++) {
        for (int j = 0; j < grid_size; j++) {
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
    collision_count = simulation(particles, grid_size, space_size, number_particles, n_time_steps, argc, argv);
    exec_time += omp_get_wtime();

    // Print the results and time of execution
    print_result(particles, collision_count);
    fprintf(stderr, "%.1fs\n", exec_time);

    // Free the memory allocated for the particles
    free(particles);
}