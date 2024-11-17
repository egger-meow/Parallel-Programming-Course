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



    // TODO: use MPI_Gather
    lln *counts = NULL; 
    if (world_rank == 0) {
        counts = (lln *) malloc(world_size * sizeof(lln));
        if (counts == NULL) 
            MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Gather(&countLocal, 1, MPI_LONG_LONG_INT, counts, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);

    if (world_rank == 0)
    {

        lln counTotal = 0;
        for (int i = 0; i < world_size; i++) 
            counTotal += counts[i];
        
        pi_result = 4.0 * (double) counTotal / (double) tosses;

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }
    
    MPI_Finalize();
    return 0;
}
