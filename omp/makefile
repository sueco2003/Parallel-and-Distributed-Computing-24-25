# Compiler
CC = gcc

# Libraries
LIBS = -lm
OMP_LIBS = -fopenmp

# Target executables
TARGETS = parsim-omp

# Source files
SRCS_C = parsim-omp.c

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
parsim-omp: parsim-omp.c $(OBJ) $(PROTO_C_SRCS) $(PROTO_C_HDRS)
	$(CC) $(CFLAGS) $(OMP_LIBS) $^ -o $@ $(LDFLAGS)

# Clean rule
clean:
	rm -f $(TARGETS) $(OBJ)
