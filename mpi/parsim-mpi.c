#include <mpi.h>
#include <math.h>
#include <omp.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "init_particles.h"
#define NUM_PARTICLES_BUFFER 10000


particle_array_t *local_particles;
particle_t *particle;

enum { TAG_INIT_PARTICLES, TAG_SEND_CENTER_OF_MASS, TAG_SEND_PARTICLES };

node_t* adjacent_processes[8] = {NULL};

particle_array_t* particles;
cell_t** cells;

int myRank, number_processors;
int size_processor_grid[2] = {0, 0};
int my_coordinates[2];
node_t* processes_buffers;

int local_cell_dims[2];
MPI_Comm cart_comm;

void create_cartesian_communicator(int grid_size) {
	int periodic[2] = {1, 1};
	
	// Calculate best grid size for the number of processes
    MPI_Dims_create(number_processors, 2, size_processor_grid);
	if (size_processor_grid[0] > grid_size) size_processor_grid[0] = grid_size;
    if (size_processor_grid[1] > grid_size) size_processor_grid[1] = grid_size;

	MPI_Cart_create(MPI_COMM_WORLD, 2, size_processor_grid, periodic, 1, &cart_comm);

	if (size_processor_grid[0] * size_processor_grid[1] <= myRank) {
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Finalize();
		exit(0);
	}

	MPI_Comm_rank(cart_comm, &myRank);
	MPI_Cart_coords(cart_comm, myRank, 2, my_coordinates);

	// Calculate rank adjacent processes
	int adjacent_processes_rank[8];
	int counter = 0;
	for (int i = -1; i <= 1; i++) {
		for (int j = -1; j <= 1; j++) {
			if (i == 0 && j == 0) {
				continue;
			}
			int coords[2];
			coords[0] = my_coordinates[0] + i;
			coords[1] = my_coordinates[1] + j;
			MPI_Cart_rank(cart_comm, coords, &adjacent_processes_rank[counter]);
			counter++;
		}
	}
	// Set structure for adjacent cells
	for (int i = 0; i < 8; i++) {
		if (adjacent_processes[i] != NULL) {
			continue;
		}
		adjacent_processes[i] = (node_t*)calloc(1, sizeof(node_t));
		adjacent_processes[i]->rank = adjacent_processes_rank[i];

		for (int j = i + 1; j < 8; j++) {
			if (adjacent_processes_rank[j] == adjacent_processes_rank[i]) {
				adjacent_processes[j] = adjacent_processes[i];
			}
		}
	}

	// Populate all processes
	processes_buffers = (node_t*)calloc(size_processor_grid[0] * size_processor_grid[1], sizeof(node_t));
	number_processors = size_processor_grid[0] * size_processor_grid[1];
	int local_nx = BLOCK_SIZE(my_coordinates[0], size_processor_grid[0], grid_size);
    int local_ny = BLOCK_SIZE(my_coordinates[1], size_processor_grid[1], grid_size);
    
    int start_x = BLOCK_LOW(my_coordinates[0], size_processor_grid[0], grid_size);
    int start_y = BLOCK_LOW(my_coordinates[1], size_processor_grid[1], grid_size);

    // Printando para debug
    //printf("Processo %d -> coords=(%d, %d) cuida de %dx%d células\n", myRank, my_coordinates[0], my_coordinates[1], local_nx, local_ny);
           
    //printf("Processo %d -> intervalo de células: x=[%d,%d), y=[%d,%d)\n", myRank, start_x, start_x + local_nx, start_y, start_y + local_ny);
}


void init_cells(int grid_size) {
	local_cell_dims[0] = BLOCK_SIZE(my_coordinates[0], size_processor_grid[0], grid_size) + 2;
	local_cell_dims[1] = BLOCK_SIZE(my_coordinates[1], size_processor_grid[1], grid_size) + 2;
	cells = (cell_t**) malloc(sizeof(cell_t*) * local_cell_dims[0]);
	cell_t* cells_chunk = (cell_t*) calloc(local_cell_dims[0] * local_cell_dims[1], sizeof(cell_t));

	for (int i = 0; i < local_cell_dims[0]; i++) {
		cells[i] = &cells_chunk[i * local_cell_dims[1]];
	}
}

void delegate_particles(int grid_size, double space_size, long long number_particles, particle_t* particles, particle_array_t* local_particles) {
	int number_processors_grid = size_processor_grid[0] * size_processor_grid[1];
	int *counters = (int*)calloc(number_processors_grid, sizeof(int));

	particle_t *buffer_space = (particle_t*)malloc(sizeof(particle_t) * NUM_PARTICLES_BUFFER * (number_processors_grid));
	particle_t **buffers = (particle_t**)malloc(sizeof(particle_t*) * (number_processors_grid));

    for (long long i = 1; i < number_processors_grid; i++) {
        buffers[i - 1] = &buffer_space[NUM_PARTICLES_BUFFER * (i - 1)];
    }

	for (long long i = 0; i < number_particles; i++) {
		particle_t *particle = &particles[i];

        // Determine the cell coordinates
        int cellx = (particle->x / (space_size / grid_size));
        int celly = (particle->y / (space_size / grid_size));
		int coords_proc_grid[2];
		coords_proc_grid[0] = BLOCK_OWNER(cellx, size_processor_grid[0], grid_size);
		coords_proc_grid[1] = BLOCK_OWNER(celly, size_processor_grid[1], grid_size);

		int proc_id_to_send;
		MPI_Cart_rank(cart_comm, coords_proc_grid, &proc_id_to_send);

        if (proc_id_to_send == 0) {
            // Check if realloc is needed and add particle to local_particles
            if (local_particles->size >= local_particles->capacity) {
                local_particles->capacity *= 2;
                local_particles->particles = (particle_t*)realloc(local_particles->particles, sizeof(particle_t) * local_particles->capacity);
            }
            local_particles->particles[local_particles->size] = *particle;
            local_particles->size++;
			continue;
        }

		// Add to process' buffer
		buffers[proc_id_to_send - 1][counters[proc_id_to_send - 1]] = *particle;
		counters[proc_id_to_send - 1]++;

		// Buffer size reach => send
		if (counters[proc_id_to_send - 1] == NUM_PARTICLES_BUFFER) {
			MPI_Send(buffers[proc_id_to_send - 1], SIZEOF_PARTICLE(NUM_PARTICLES_BUFFER), MPI_DOUBLE, proc_id_to_send, TAG_INIT_PARTICLES, cart_comm);
			counters[proc_id_to_send - 1] = 0;
		}
	}

	// Send eveything that is not been sent yet
	for (int i = 1; i < number_processors_grid; i++) {
		if (counters[i - 1] != 0) {
			MPI_Send(buffers[i - 1], SIZEOF_PARTICLE(counters[i - 1]), MPI_DOUBLE, i, TAG_INIT_PARTICLES, cart_comm);
		} else {
			double endFlag[1] = {-1};
			MPI_Send(endFlag, 1, MPI_DOUBLE, i, TAG_INIT_PARTICLES, cart_comm);
		}
	}

	free(counters);
	free(buffer_space);
	free(buffers);
}

void receiveParticles() {
	int number_doubles;
	MPI_Status status;

	while (1) {
		MPI_Probe(0, TAG_INIT_PARTICLES, cart_comm, &status);
		MPI_Get_count(&status, MPI_DOUBLE, &number_doubles);
		
		// Check if it is the end flag
		if (number_doubles == 1) {
			break;
		}
		int num_particles_received = number_doubles / PARTICLE_SIZE;
		// Check if realloc is needed
		if (local_particles->size + num_particles_received > local_particles->capacity) {
			while (local_particles->size + num_particles_received > local_particles->capacity) {
                local_particles->capacity *= 2;
            }
			local_particles->particles = (particle_t*)realloc(local_particles->particles, sizeof(particle_t) * local_particles->capacity);
		}
		// Receive particles
		MPI_Recv(&(local_particles->particles[local_particles->size]), number_doubles, MPI_DOUBLE, 0, TAG_INIT_PARTICLES, cart_comm, &status);
		//printf("Processo %d Recebeu %d particulas\n",myRank, num_particles_received);
		MPI_Get_count(&status, MPI_DOUBLE, &number_doubles);
		local_particles->size += num_particles_received;

		if (number_doubles / 8 != NUM_PARTICLES_BUFFER) {
			break;
		}
	}
}

void calculate_centers_of_mass(int grid_size, double space_size) {
	// Calculate local center of mass
	
	for (int i = 0; i < local_particles->size; i++) {
		particle_t* particle = &local_particles->particles[i];
		//if(myRank == 1 ) printf("Processo %d -> particle[%d] -> x=%f, y=%f, m=%f\n\n", myRank, i, particle->x, particle->y, particle->m);
        int global_cell_index_x = (particle->x / (space_size / grid_size));
        int global_cell_index_y = (particle->y / (space_size / grid_size));
		int local_cell_index_x = CONVERT_TO_LOCAL(my_coordinates[0], size_processor_grid[0], grid_size, global_cell_index_x);
		int local_cell_index_y = CONVERT_TO_LOCAL(my_coordinates[1], size_processor_grid[1], grid_size, global_cell_index_y);
		//if (myRank == 1) printf("global_cell_index_x = %d, global_cell_index_y = %d\n", global_cell_index_x, global_cell_index_y);
		//if (myRank == 1) printf("local_cell_index_x = %d, local_cell_index_y = %d\n\n", local_cell_index_x, local_cell_index_y);
		cell_t* cell = &cells[local_cell_index_x][local_cell_index_y];
		cell->mass_sum += particle->m;
		cell->cmx += particle->m * particle->x;
		cell->cmy += particle->m * particle->y;
	}
	// Calculate global center of mass
	for (int i = 1; i <= local_cell_dims[0] - 2; i++) {
		for (int j = 1; j <= local_cell_dims[1] - 2; j++) {
			cell_t* cell = &cells[i][j];
			if (cell->mass_sum != 0) {
				cell->cmx /= cell->mass_sum;
				cell->cmy /= cell->mass_sum;
			}
			//printf("Processo %d -> cell[%d][%d] -> cmx=%f, cmy=%f, Cmsum= %f\n\n", myRank, i, j, cell->cmx, cell->cmy, cell->mass_sum);
		}
	}
}

void send_recv_centers_of_mass() {
    MPI_Request send_requests[8];
    MPI_Request recv_requests[8];
    MPI_Status status[8];
    
    // Data buffer for sending and receiving cell data
    double send_buffers[8][3];  // [cell_cmx, cell_cmy, cell_mass_sum]
    double recv_buffers[8][3];
    
    int send_count = 0;
    int recv_count = 0;
    
    // Direction names for debugging
    const char* direction_names[8] = {
        "DIAGONAL_UP_LEFT", "UP", "DIAGONAL_UP_RIGHT", 
        "LEFT", "RIGHT", 
        "DIAGONAL_DOWN_LEFT", "DOWN", "DIAGONAL_DOWN_RIGHT"
    };
    
    // Prepare send buffers based on the 8 adjacent processes
    int directions[8][2] = {
        {-1, -1},{-1, 0},{-1, 1},  // Diagonal up left, up, diagonal up right
        {0, -1},         {0, 1},    // Left, right
        {1, -1}, {1, 0}, {1, 1}       // Diagonal down left, down, diagonal down right
    };
    
    //printf("Processo %d: Preparando para enviar centros de massa para vizinhos\n", myRank);
    
    // Sending center of mass data to 8 adjacent processes
    for (int i = 0; i < 8; i++) {
		if (adjacent_processes[i]->rank == myRank) continue;
        int dx = directions[i][0];
        int dy = directions[i][1];
        
        // Determine the boundary cell to send based on the direction
        int boundary_x = (dx == -1) ? 1 : ((dx == 1) ? local_cell_dims[0] - 2 : (local_cell_dims[0] / 2));
        int boundary_y = (dy == -1) ? 1 : ((dy == 1) ? local_cell_dims[1] - 2 : (local_cell_dims[1] / 2));
        
        cell_t* boundary_cell = &cells[boundary_x][boundary_y];
        
        send_buffers[i][0] = boundary_cell->cmx;
        send_buffers[i][1] = boundary_cell->cmy;
        send_buffers[i][2] = boundary_cell->mass_sum;
        
        printf("Processo %d: Enviando para vizinho %s (rank %d): cmx=%f, cmy=%f, mass=%f\n", 
               myRank, direction_names[i], adjacent_processes[i]->rank, 
               send_buffers[i][0], send_buffers[i][1], send_buffers[i][2]);
        
        // Non-blocking send to adjacent process
        MPI_Isend(&send_buffers[i], 3, MPI_DOUBLE, adjacent_processes[i]->rank, TAG_SEND_CENTER_OF_MASS, cart_comm, &send_requests[send_count++]);
    }
    
    //printf("Processo %d: Preparando para receber centros de massa de vizinhos\n", myRank);
    
    // Receive center of mass data from 8 adjacent processes (Non-blocking)
    for (int i = 0; i < 8; i++) {
		if(adjacent_processes[i]->rank == myRank) continue;
        MPI_Irecv(&recv_buffers[i], 3, MPI_DOUBLE, adjacent_processes[i]->rank, TAG_SEND_CENTER_OF_MASS, cart_comm, &recv_requests[recv_count++]);
    }
    // Wait for all sends and receives to complete
    MPI_Waitall(send_count, send_requests, MPI_STATUSES_IGNORE);
    MPI_Waitall(recv_count, recv_requests, MPI_STATUSES_IGNORE);

    //printf("Processo %d: Processando centros de massa recebidos\n", myRank);
    
    // Process received center of mass data
    for (int i = 0; i < 8; i++) {
        int dx = directions[i][0];
        int dy = directions[i][1];
        
        // Determine the ghost cell to update based on the direction
        int ghost_x = (dx == -1) ? 0 : ((dx == 1) ? local_cell_dims[0] - 1 : (local_cell_dims[0] / 2));
        int ghost_y = (dy == -1) ? 0 : ((dy == 1) ? local_cell_dims[1] - 1 : (local_cell_dims[1] / 2));
        
        cell_t* ghost_cell = &cells[ghost_x][ghost_y];
        
        /*printf("Processo %d: Recebido de vizinho %s (rank %d): cmx=%f, cmy=%f, mass=%f\n", 
               myRank, direction_names[i], adjacent_processes[i]->rank, 
               recv_buffers[i][0], recv_buffers[i][1], recv_buffers[i][2]);*/
        
        // Update ghost cell with received center of mass
        if (recv_buffers[i][2] > 0) {  // Only update if mass is non-zero
            ghost_cell->cmx = recv_buffers[i][0];
            ghost_cell->cmy = recv_buffers[i][1];
            ghost_cell->mass_sum = recv_buffers[i][2];
            
            //printf("Processo %d: Atualizando célula fantasma %s com centro de massa\n", myRank, direction_names[i]);
        }
    }
    
    //printf("Processo %d: Comunicação de centros de massa concluída\n", myRank);
}

void calculate_new_iteration(int grid_size, double space_size) {

	//TODO: Implement this function
}

void send_recv_particles() {

	//TODO: Implement this function
}
int main(int argc, char* argv[]) {
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

    // Initialize particles with random positions and velocities
    init_particles(seed, space_size, grid_size, number_particles, particles);

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &number_processors);
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

	// Local particles array
	local_particles = (particle_array_t *)malloc(sizeof(particle_array_t));
	local_particles->size = 0;
	local_particles->capacity = number_particles * 10 / (grid_size * grid_size);
	local_particles->particles = (particle_t *)malloc(sizeof(particle_t) * local_particles->capacity);

	create_cartesian_communicator(grid_size);
	init_cells(grid_size);

	if (myRank == 0) {
		delegate_particles(grid_size, space_size, number_particles, particles, local_particles);
	} else {
		receiveParticles();
	}

	for (int n = 0; n < n_time_steps; n++) {
		calculate_centers_of_mass(grid_size, space_size);
		send_recv_centers_of_mass();
		//calculate_new_iteration(grid_size, space_size);
		//send_recv_particles();

		//memset(cells[0], 0, sizeof(cell_t) * local_cell_dims[0] * local_cell_dims[1]);
	}

	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Finalize();
	return 0;
}
