#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define G 6.67408e-11
#define EPSILON2 (0.005 * 0.005)
#define DELTAT 0.1

typedef struct {
    double x, y;
    double vx, vy;
    double m;
    int active;
} particle_t;

void init_particles(long seed, double side, long ncside, long long n_part, particle_t *par);
void compute_forces(particle_t *particles, long long n_part, double side, long ncside);
void update_positions(particle_t *particles, long long n_part);
int check_collisions(particle_t *particles, long long n_part);

int main(int argc, char *argv[]) {
    if (argc != 6) {
        fprintf(stderr, "Usage: %s <seed> <side> <grid_size> <num_particles> <num_steps>\n", argv[0]);
        return EXIT_FAILURE;
    }
    
    long seed = atol(argv[1]);
    double side = atof(argv[2]);
    long ncside = atol(argv[3]);
    long long n_part = atoll(argv[4]);
    int n_steps = atoi(argv[5]);

    particle_t *particles = (particle_t *)malloc(n_part * sizeof(particle_t));
    if (!particles) {
        fprintf(stderr, "Memory allocation failed!\n");
        return EXIT_FAILURE;
    }
    
    init_particles(seed, side, ncside, n_part, particles);
    double exec_time = -omp_get_wtime();
    
    int collisions = 0;
    for (int step = 0; step < n_steps; step++) {
        compute_forces(particles, n_part, side, ncside);
        update_positions(particles, n_part);
        collisions += check_collisions(particles, n_part);
    }
    
    exec_time += omp_get_wtime();
    fprintf(stderr, "%.1fs\n", exec_time);
    printf("%.3f %.3f\n", particles[0].x, particles[0].y);
    printf("%d\n", collisions);
    
    free(particles);
    return EXIT_SUCCESS;
}
