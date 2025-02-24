# Compiler
CC = gcc

# Libraries
LIBS = -lm
MPI_LIBS = -lmpi
OMP_LIBS = -fopenmp

# Target executables
TARGETS = serial/parsim \
          omp/parsim-omp \
          mpi/parsim-mpi

# Source files
SRCS_C = serial/parsim.c \
          omp/parsim-omp.c \
          mpi/parsim-mpi.c

# Object files
OBJ = init_particles.o

# Protobuf-generated sources (define properly if applicable)
PROTO_C_SRCS = 
PROTO_C_HDRS = 

# Compiler flags
CFLAGS = -O2 -fopenmp -lm
LDFLAGS = $(LIBS)

# Default target
all: $(TARGETS)

# Compile init_particles.c into an object file
init_particles.o: init_particles.c
	$(CC) $(CFLAGS) -c $< -o $@

# Compile each executable, linking with init_particles.o
serial/parsim: serial/parsim.c $(OBJ) $(PROTO_C_SRCS) $(PROTO_C_HDRS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

omp/parsim-omp: omp/parsim-omp.c $(OBJ) $(PROTO_C_SRCS) $(PROTO_C_HDRS)
	$(CC) $(CFLAGS) $(OMP_LIBS) $^ -o $@ $(LDFLAGS)

mpi/parsim-mpi: mpi/parsim-mpi.c $(OBJ) $(PROTO_C_SRCS) $(PROTO_C_HDRS)
	mpicc $(CFLAGS) $^ -o $@ $(LDFLAGS) $(MPI_LIBS)

# Clean rule
clean:
	rm -f $(TARGETS) $(OBJ)
