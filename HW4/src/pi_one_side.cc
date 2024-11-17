#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>
typedef long long int lln;

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    MPI_Win win;

    // TODO: MPI init
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    lln tossesLocal = tosses / world_size;
    lln remainder = tosses % world_size;
    if (world_rank < remainder) 
        tossesLocal += 1;

    unsigned int seed = (unsigned int) time(NULL) + world_rank * 100;

    lln countLocal = 0;

    for (lln i = 0; i < tossesLocal; i++) {
        double x = (double) rand_r(&seed) / RAND_MAX;
        double y = (double) rand_r(&seed) / RAND_MAX;
        if (x * x + y * y <= 1.0) 
            countLocal++;
    }

    lln *counTotal = nullptr; 
    if (world_rank == 0)
    {
        counTotal = (lln *) malloc(sizeof(lln));

        *counTotal = countLocal;
        MPI_Win_create(counTotal, sizeof(lln), sizeof(lln), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
        // Master
    }
    else
    {
        MPI_Win_create(counTotal, 0, sizeof(lln), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
        // Workers
        MPI_Win_fence(0, win);
        MPI_Accumulate(&countLocal, 1, MPI_LONG_LONG_INT, 0, 0, 1, MPI_LONG_LONG_INT, MPI_SUM, win);
    
        MPI_Win_fence(0, win);
    }

    // MPI_Win_fence(0, win);

    // if (world_rank != 0) {
    //     // Workers accumulate their countLocal to master's total_count
    //     MPI_Accumulate(&countLocal, 1, MPI_LONG_LONG_INT, 0, 0, 1, MPI_LONG_LONG_INT, MPI_SUM, win);
    // }

    // // End epoch
    // MPI_Win_fence(0, win);

    // MPI_Win_free(&win);

    if (world_rank == 0)
    {
        // TODO: handle PI result
        pi_result = 4.0 * (double) *counTotal / (double) tosses;
        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }
    
    MPI_Finalize();
    return 0;
}