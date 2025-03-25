int x_size = local_cell_dims[0] - 2;
int y_size = local_cell_dims[1] - 2;
int n_neighbours = x_size * 2 + y_size * 2 + 4;

printf( "Processo %d -> x_size = %d, y_size = %d, n_neighbours = %d\n", myRank, x_size, y_size, n_neighbours);

// Prepare send buffers
for (int i = 0; i < 8; i++) {
    if (adjacent_processes[i]->cells_buffer_send == NULL) {
        adjacent_processes[i]->cells_buffer_send = (cell_t*)calloc(n_neighbours, sizeof(cell_t));
    }
    if (adjacent_processes[i]->cells_buffer_recv == NULL) {
        adjacent_processes[i]->cells_buffer_recv = (cell_t*)calloc(n_neighbours, sizeof(cell_t));
    }
}

// Build buffers to send
for (int i = 7; i >= 0; i--) {
    switch (i) {
        case DIAGONAL_UP_LEFT:
            adjacent_processes[i]->cells_buffer_send[adjacent_processes[i]->length_send_buffer] = cells[1][1];
            adjacent_processes[i]->length_send_buffer++;
            break;
        case DIAGONAL_UP_RIGHT:
            adjacent_processes[i]->cells_buffer_send[adjacent_processes[i]->length_send_buffer] = cells[1][y_size];
            adjacent_processes[i]->length_send_buffer++;
            break;
        case DIAGONAL_DOWN_LEFT:
            adjacent_processes[i]->cells_buffer_send[adjacent_processes[i]->length_send_buffer] = cells[x_size][1];
            adjacent_processes[i]->length_send_buffer++;
            break;
        case DIAGONAL_DOWN_RIGHT:
            adjacent_processes[i]->cells_buffer_send[adjacent_processes[i]->length_send_buffer] = cells[x_size][y_size];
            adjacent_processes[i]->length_send_buffer++;
            break;
        case LEFT:
            for (int j = 1; j <= x_size; j++) {
                adjacent_processes[i]->cells_buffer_send[adjacent_processes[i]->length_send_buffer] = cells[j][1];
                adjacent_processes[i]->length_send_buffer++;
            }
            break;
        case RIGHT:
            for (int j = 1; j <= x_size; j++) {
                adjacent_processes[i]->cells_buffer_send[adjacent_processes[i]->length_send_buffer] = cells[j][y_size];
                adjacent_processes[i]->length_send_buffer++;
            }
            break;
        case UP:
            for (int j = 1; j <= y_size; j++) {
                adjacent_processes[i]->cells_buffer_send[adjacent_processes[i]->length_send_buffer] = cells[1][j];
                adjacent_processes[i]->length_send_buffer++;
            }
            break;
        case DOWN:
            for (int j = 1; j <= y_size; j++) {
                adjacent_processes[i]->cells_buffer_send[adjacent_processes[i]->length_send_buffer] = cells[x_size][j];
                adjacent_processes[i]->length_send_buffer++;
            }
            break;

        default:
            printf("[%d] Default case in send send_recv_centers_of_mass\n", myRank);
            fflush(stdout);
            break;
    }
}

// Receive
MPI_Request request[8];
MPI_Status status[8];
for (int i = 0; i < 8; i++) {
    if (adjacent_processes[i]->received == 1) {
        request[i] = MPI_REQUEST_NULL;
        continue;
    }

    MPI_Irecv(adjacent_processes[i]->cells_buffer_recv, SIZEOF_CELL(n_neighbours), MPI_DOUBLE, adjacent_processes[i]->rank, TAG_SEND_CENTER_OF_MASS, cart_comm, &request[i]);
    adjacent_processes[i]->received = 1;
}

// Send
for (int i = 0; i < 8; i++) {
    if (adjacent_processes[i]->sent == 1) {
        continue;
    }

    MPI_Send(adjacent_processes[i]->cells_buffer_send, SIZEOF_CELL(adjacent_processes[i]->length_send_buffer), MPI_DOUBLE, adjacent_processes[i]->rank, TAG_SEND_CENTER_OF_MASS, cart_comm);
    adjacent_processes[i]->sent = 1;
}
MPI_Waitall(1, request, status);

// Reconstruct data
for (int i = 0; i < 8; i++) {
    switch (i) {
        case DIAGONAL_UP_LEFT:
            cells[0][0] = adjacent_processes[i]->cells_buffer_recv[adjacent_processes[i]->index];
            adjacent_processes[i]->index++;
            break;
        case DIAGONAL_UP_RIGHT:
            cells[0][y_size + 1] = adjacent_processes[i]->cells_buffer_recv[adjacent_processes[i]->index];
            adjacent_processes[i]->index++;
            break;
        case DIAGONAL_DOWN_LEFT:
            cells[x_size + 1][0] = adjacent_processes[i]->cells_buffer_recv[adjacent_processes[i]->index];
            adjacent_processes[i]->index++;
            break;
        case DIAGONAL_DOWN_RIGHT:
            cells[x_size + 1][y_size + 1] = adjacent_processes[i]->cells_buffer_recv[adjacent_processes[i]->index];
            adjacent_processes[i]->index++;
            break;
        case LEFT:
            for (int j = 1; j <= x_size; j++) {
                cells[j][0] = adjacent_processes[i]->cells_buffer_recv[adjacent_processes[i]->index];
                adjacent_processes[i]->index++;
            }
            break;
        case RIGHT:
            for (int j = 1; j <= x_size; j++) {
                cells[j][y_size + 1] = adjacent_processes[i]->cells_buffer_recv[adjacent_processes[i]->index];
                adjacent_processes[i]->index++;
            }
            break;
        case UP:
            for (int j = 1; j <= y_size; j++) {
                cells[0][j] = adjacent_processes[i]->cells_buffer_recv[adjacent_processes[i]->index];
                adjacent_processes[i]->index++;
            }
            break;
        case DOWN:
            for (int j = 1; j <= y_size; j++) {
                cells[x_size + 1][j] = adjacent_processes[i]->cells_buffer_recv[adjacent_processes[i]->index];
                adjacent_processes[i]->index++;
            }
            break;

        default:
            printf("[%d] Default case in send Reconstruct Data\n", myRank);
            fflush(stdout);
            break;
    }
}

// Reset structures
for (int i = 0; i < 8; i++) {
    adjacent_processes[i]->cells_buffer_send = NULL;
    adjacent_processes[i]->cells_buffer_recv = NULL;
    adjacent_processes[i]->length_send_buffer = 0;
    adjacent_processes[i]->sent = 0;
    adjacent_processes[i]->received = 0;
    adjacent_processes[i]->index = 0;
}