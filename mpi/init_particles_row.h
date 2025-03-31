#ifndef INIT_PARTICLES_H
#define INIT_PARTICLES_H


typedef struct particle {
    double x, y, vx, vy, m, fx, fy, death_timestamp;
} particle_t;

typedef struct {
    double mass_sum;
    double cmx;
    double cmy;
    int adj_cells[8][2];  // Store (x, y) coordinates of 8 neighbors
    particle_t **particles_inside;
    long long current_size;
    long long capacity;
} cell_t;

typedef struct {
    double mass_sum, cmx, cmy;
} lean_cell_t;

typedef struct {
	long long current_size;
    long long capacity;
	particle_t **particles;
} particle_array_t;

typedef struct {
	int rank;
	int length_send_buffer;
	particle_array_t *particles_buffer_send;
	cell_t* cells_buffer_send;
	cell_t* cells_buffer_recv;
	int index;
	int sent;
	int received;
} node_t;

#define PARTICLE_SIZE 8
#define SIZEOF_PARTICLE(n) (n * PARTICLE_SIZE)
#define GET_NUMBER_PARTICLE(n) (n / PARTICLE_SIZE)

#define CELL_SIZE 3
#define SIZEOF_CELL(n) (n * CELL_SIZE)
#define GET_NUMBER_CELL(n) (n / CELL_SIZE)

// Where n is the number of elements, p the number of processes and id is the rank of the process.
#define BLOCK_LOW(id, p, n) ((id) * (n) / (p))
#define BLOCK_HIGH(id, p, n) (BLOCK_LOW((id) + 1, p, n) - 1)
#define BLOCK_SIZE(id, p, n) (BLOCK_HIGH(id, p, n) - BLOCK_LOW(id, p, n) + 1)
#define BLOCK_OWNER(index, p, n) (((p) * ((index) + 1) - 1) / (n))
#define MIN(a,b) (((a)<(b))?(a):(b))
#define CONVERT_TO_LOCAL(id, p, n, x) (x - BLOCK_LOW(id, p, n) + 1)




void init_r4uni(int input_seed);
double rnd_uniform01();
double rnd_normal01();
void init_particles(long seed, double side, long ncside, long long n_part, particle_t *par);


#endif  // INIT_PARTICLES_H
