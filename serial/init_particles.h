#ifndef INIT_PARTICLES_H
#define INIT_PARTICLES_H

typedef struct particle {
    double x, y, vx, vy, m, fx, fy;
    int death_timestamp;
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


void init_r4uni(int input_seed);
double rnd_uniform01();
double rnd_normal01();
void init_particles(long seed, double side, long ncside, long long n_part, particle_t *par);


#endif  // INIT_PARTICLES_H