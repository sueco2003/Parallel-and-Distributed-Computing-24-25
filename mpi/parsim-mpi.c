#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

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
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 6) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s <seed> <side> <grid_size> <num_particles> <num_steps>\n", argv[0]);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    
    long seed = atol(argv[1]);
    double side = atof(argv[2]);
    long ncside = atol(argv[3]);
    long long n_part = atoll(argv[4]);
    int n_steps = atoi(argv[5]);

    long long local_n_part = n_part / size;
    particle_t *particles = (particle_t *)malloc(local_n_part * sizeof(particle_t));
    if (!particles) {
        fprintf(stderr, "Memory allocation failed on rank %d!\n", rank);
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    
    init_particles(seed, side, ncside, local_n_part, particles);
    double exec_time = -MPI_Wtime();
    
    int local_collisions = 0, global_collisions = 0;
    for (int step = 0; step < n_steps; step++) {
        compute_forces(particles, local_n_part, side, ncside);
        update_positions(particles, local_n_part);
        local_collisions += check_collisions(particles, local_n_part);
        MPI_Allreduce(&local_collisions, &global_collisions, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    }
    
    exec_time += MPI_Wtime();
    if (rank == 0) {
        fprintf(stderr, "%.1fs\n", exec_time);
        printf("%.3f %.3f\n", particles[0].x, particles[0].y);
        printf("%d\n", global_collisions);
    }
    
    free(particles);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
