#include "mpi.h"
#include <stdio.h>

int main(int argc, char *argv[])
{
    int myrank, world_size, left, right;
    int buffer[524288] = {0};
    int buffer2[524288] = {0};
    int i = 0;
    
    MPI_Request request[2], request2[2];
    MPI_Status status;

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    	
    int id_req = 0;
    double time = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);

    for (i = 0; i < world_size; i++ )
    {
        if (i != myrank)
        {
            MPI_Irecv(buffer, 524288, MPI_INT, i, 0, MPI_COMM_WORLD, &request[id_req]);
            MPI_Isend(buffer2, 524288, MPI_INT, i, 0, MPI_COMM_WORLD, &request2[id_req]);
            id_req++;
        }
    }
    
    for (i = 0; i<2; i++)
    {
	    MPI_Wait(&request[i], MPI_STATUS_IGNORE);
	    MPI_Wait(&request2[i], MPI_STATUS_IGNORE);
    }
 
    time = MPI_Wtime() - time;
	
    printf("Rank = %d, time = %lf\n", myrank, time);	    
    MPI_Finalize();
    return 0;
}

