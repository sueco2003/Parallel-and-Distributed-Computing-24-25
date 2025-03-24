# Compiler
CC = gcc

# Libraries
LIBS = -lm
MPI_LIBS = -lmpi
OMP_LIBS = -fopenmp

# Target executables
TARGETS = parsim-mpi

# Source files
SRCS_C = parsim-mpi.c

# Object files
OBJ = init_particles.o

PROTO_C_SRCS = 
PROTO_C_HDRS = 

# Compiler flags
CFLAGS = -O2 -fopenmp -lm -g
LDFLAGS = $(LIBS)

# Default target
all: $(TARGETS)

# Compile init_particles.c into an object file
init_particles.o: init_particles.c
	$(CC) $(CFLAGS) -c $< -o $@

# Compile each executable, linking with init_particles.o
parsim-mpi: parsim-mpi.c $(OBJ) $(PROTO_C_SRCS) $(PROTO_C_HDRS)
	mpicc $(CFLAGS) $^ -o $@ $(LDFLAGS) $(MPI_LIBS)

# Clean rule
clean:
	rm -f $(TARGETS) $(OBJ)
