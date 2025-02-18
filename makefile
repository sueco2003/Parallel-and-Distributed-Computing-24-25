# Compiler
CC = gcc

# Libraries
LIBS = -lm
MPI_LIBS = -lmpi
OMP_LIBS = -fopenmp

# Target executables
TARGETS = mpi/parsim-mpi \
          serial/parsim \
          omp/parsim-omp

# Source files
SRCS_C = mpi/parsim-mpi.c \
         serial/parsim.c \
         omp/parsim-omp.c 

# Object files
OBJ = init_particles.o

# Protobuf-generated sources (define properly if applicable)
PROTO_C_SRCS = 
PROTO_C_HDRS = 

# Compiler flags
CFLAGS = -O2 -g
LDFLAGS = $(LIBS)

# Default target
all: $(TARGETS)

# Compile init_particles.c into an object file
init_particles.o: init_particles.c
	$(CC) $(CFLAGS) -c $< -o $@

# Compile each executable, linking with init_particles.o
mpi/parsim-mpi: mpi/parsim-mpi.c $(OBJ) $(PROTO_C_SRCS) $(PROTO_C_HDRS)
	mpicc $(CFLAGS) $^ -o $@ $(LDFLAGS) $(MPI_LIBS)

serial/parsim: serial/parsim.c $(OBJ) $(PROTO_C_SRCS) $(PROTO_C_HDRS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

omp/parsim-omp: omp/parsim-omp.c $(OBJ) $(PROTO_C_SRCS) $(PROTO_C_HDRS)
	$(CC) $(CFLAGS) $(OMP_LIBS) $^ -o $@ $(LDFLAGS)

# Clean rule
clean:
	rm -f $(TARGETS) $(OBJ)
