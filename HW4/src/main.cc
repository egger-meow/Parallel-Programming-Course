#include <mpi.h>
#include <fstream>
#include <iostream>

// *********************************************
// ** ATTENTION: YOU CANNOT MODIFY THIS FILE. **
// *********************************************

// Read size of matrix_a and matrix_b (n, m, l) and whole data of matrixes from in
//
// in:        input stream of the matrix file
// n_ptr:     pointer to n
// m_ptr:     pointer to m
// l_ptr:     pointer to l
// a_mat_ptr: pointer to matrix a (a should be a continuous memory space for placing n * m elements of int)
// b_mat_ptr: pointer to matrix b (b should be a continuous memory space for placing m * l elements of int)
void construct_matrices(std::ifstream &in, int *n_ptr, int *m_ptr, int *l_ptr,
                        int **a_mat_ptr, int **b_mat_ptr) {
    // Read dimensions n, m, l
    in >> *n_ptr >> *m_ptr >> *l_ptr;
    int n = *n_ptr;
    int m = *m_ptr;
    int l = *l_ptr;

    // Allocate memory for matrices A and B
    *a_mat_ptr = (int*) malloc(n * m * sizeof(int));
    *b_mat_ptr = (int*) malloc(m * l * sizeof(int));

    if (*a_mat_ptr == NULL || *b_mat_ptr == NULL) {
        std::cerr << "Memory allocation failed for matrices A or B.\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Read matrix A
    for(int i = 0; i < n * m; ++i) {
        in >> (*a_mat_ptr)[i];
    }

    // Read matrix B
    for(int i = 0; i < m * l; ++i) {
        in >> (*b_mat_ptr)[i];
    }
}

// Just matrix multiplication (your should output the result in this function)
// 
// n:     row number of matrix a
// m:     col number of matrix a / row number of matrix b
// l:     col number of matrix b
// a_mat: a continuous memory placing n * m elements of int
// b_mat: a continuous memory placing m * l elements of int
void matrix_multiply(const int n, const int m, const int l,
                     const int *a_mat, const int *b_mat) {
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Step 1: Broadcast matrix B to all processes
    int *b_broadcast = NULL;
    if(world_rank != 0){
        b_broadcast = (int*) malloc(m * l * sizeof(int));
        if(b_broadcast == NULL){
            std::cerr << "Memory allocation failed for B in process " << world_rank << ".\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Bcast((world_rank == 0) ? (void*)b_mat : (void*)b_broadcast, m * l, MPI_INT, 0, MPI_COMM_WORLD);

    const int *B = (world_rank == 0) ? b_mat : b_broadcast;

    // Step 2: Scatter rows of matrix A to all processes
    // Calculate the number of rows per process
    int rows_per_proc = n / world_size;
    int remainder = n % world_size;

    int *sendcounts_A = NULL;
    int *displs_A = NULL;

    if(world_rank == 0){
        sendcounts_A = (int*) malloc(world_size * sizeof(int));
        displs_A = (int*) malloc(world_size * sizeof(int));
        if(sendcounts_A == NULL || displs_A == NULL){
            std::cerr << "Memory allocation failed for sendcounts_A or displs_A on master.\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        for(int i = 0; i < world_size; ++i){
            sendcounts_A[i] = (i < remainder) ? (rows_per_proc + 1) * m : rows_per_proc * m;
            displs_A[i] = (i < remainder) ? i * (rows_per_proc + 1) * m : (remainder * (rows_per_proc + 1) + (i - remainder) * rows_per_proc) * m;
        }
    }

    // Each process determines its receive count
    int recvcount_A = (world_rank < remainder) ? (rows_per_proc + 1) * m : rows_per_proc * m;

    // Allocate buffer to receive rows of A
    int *a_recv = (int*) malloc(recvcount_A * sizeof(int));
    if(a_recv == NULL){
        std::cerr << "Memory allocation failed for a_recv in process " << world_rank << ".\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Scatter the rows of A
    MPI_Scatterv(a_mat, sendcounts_A, displs_A, MPI_INT, a_recv, recvcount_A, MPI_INT, 0, MPI_COMM_WORLD);

    // Free sendcounts_A and displs_A on master
    if(world_rank == 0){
        free(sendcounts_A);
        free(displs_A);
    }

    // Step 3: Each process computes its part of matrix C
    int local_rows = recvcount_A / m;
    int *c_local = (int*) malloc(local_rows * l * sizeof(int));
    if(c_local == NULL){
        std::cerr << "Memory allocation failed for c_local in process " << world_rank << ".\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Perform the multiplication
    for(int i = 0; i < local_rows; ++i){
        for(int j = 0; j < l; ++j){
            long long sum = 0; // Use long long to prevent overflow
            for(int k = 0; k < m; ++k){
                sum += a_recv[i * m + k] * B[k * l + j];
            }
            c_local[i * l + j] = (int)sum;
        }
    }

    // Step 4: Gather the results back to the master process
    int *recvcounts_C = NULL;
    int *displs_C = NULL;
    if(world_rank == 0){
        recvcounts_C = (int*) malloc(world_size * sizeof(int));
        displs_C = (int*) malloc(world_size * sizeof(int));
        if(recvcounts_C == NULL || displs_C == NULL){
            std::cerr << "Memory allocation failed for recvcounts_C or displs_C on master.\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        for(int i = 0; i < world_size; i++){
            recvcounts_C[i] = (i < remainder) ? (rows_per_proc + 1) * l : rows_per_proc * l;
            displs_C[i] = (i < remainder) ? i * (rows_per_proc + 1) * l : (remainder * (rows_per_proc + 1) + (i - remainder) * rows_per_proc) * l;
        }
    }

    // Allocate buffer for the final matrix C on master
    int *C = NULL;
    if(world_rank == 0){
        C = (int*) malloc(n * l * sizeof(int));
        if(C == NULL){
            std::cerr << "Memory allocation failed for matrix C on master.\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Gather all parts of C
    MPI_Gatherv(c_local, local_rows * l, MPI_INT, C, recvcounts_C, displs_C, MPI_INT, 0, MPI_COMM_WORLD);

    // Step 5: Master process prints the result
    if(world_rank == 0){
        for(int i = 0; i < n; ++i){
            for(int j = 0; j < l; ++j){
                printf("%d ", C[i * l + j]);
            }
            printf("\n");
        }
        free(C);
        free(recvcounts_C);
        free(displs_C);
    }

    // Step 6: Free allocated memory
    free(a_recv);
    free(c_local);
    if(world_rank != 0){
        free(b_broadcast);
    }

    // Note: All allocations are freed appropriately
}

// Remember to release your allocated memory
void destruct_matrices(int *a_mat, int *b_mat) {
    free(a_mat);
    free(b_mat);
}

int main (int argc, const char **argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " FILE\n";
        return 1;
    }

    MPI_Init(NULL, NULL);
    double start_time = MPI_Wtime();

    int n, m, l;
    int *a_mat, *b_mat;
    std::ifstream in(argv[1]);
    construct_matrices(in, &n, &m, &l, &a_mat, &b_mat);
    matrix_multiply(n, m, l, a_mat, b_mat);
    destruct_matrices(a_mat, b_mat);

    double end_time = MPI_Wtime();
    MPI_Finalize();
    std::cout << "MPI running time: " << end_time - start_time << " Seconds\n";

    return 0;
}
