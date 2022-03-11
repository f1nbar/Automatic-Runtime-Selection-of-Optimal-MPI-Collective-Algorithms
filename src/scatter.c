#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <math.h>
#include <gsl/gsl_heapsort.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_cdf.h>

#define MAX_PROCESSES 10
#define MAX_SIZE 8388608;
#define SEGSIZE 8192 // 8kB
#define STRIDE 32768 // 32 kB

typedef enum scatter_algorithms {
    BASIC_LINEAR,
    BINOMIAL
} scatter_algorithms;

typedef struct time_precision {
    int min_reps;
    int max_reps;
    double cl;
    double eps;
} time_precision;

void default_time_precision(time_precision *precision) {
    *precision = (time_precision){30 /*min_reps*/, 300 /*max_reps*/, 0.97 /*cl*/, 0.015 /*eps*/};
}

const char *coll_tuned_scatter_algorithm_segmentsize= "coll_tuned_scatter_algorithm_segmentsize";
const char *tuned_scatter_algorithm= "coll_tuned_scatter_algorithm";

int set_mca_variable(const char *variable_name, int value) {
    int cidx, nvals, err;
    int val;
    MPI_T_cvar_handle chandle;
    err = MPI_T_cvar_get_index(variable_name, &cidx);
    if (err != MPI_SUCCESS)
        fprintf(stdout, "Error getting %s\n", variable_name);
    err = MPI_T_cvar_handle_alloc(cidx, NULL, &chandle, &nvals);
    err = MPI_T_cvar_write(chandle, &value);
    if (err != MPI_SUCCESS)
        fprintf(stdout, "Error setting %s \n", variable_name);
    MPI_T_cvar_handle_free(&chandle); 
    return EXIT_SUCCESS;
}

void initialize_char_array(int length, char *s) {
	char charset[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,.-#'?!";
	if (length)
	{
		for (int n = 0; n < length; n++)
		{
			int key = rand() % (int)(sizeof(charset) - 1);
			s[n] = charset[key];
		}
	}
}

double time_ci(double cl, int reps, double *T) {
    return fabs(gsl_cdf_tdist_Pinv(cl, reps - 1)) * gsl_stats_sd(T, 1, reps) / sqrt(reps);
}

int scatter(int msg_min, int msg_max, int stride){
    int rank, size, alg;
    int msg_size = SEGSIZE;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );
    MPI_Comm dump_comm;
    char *sendbuf;
    char *recvbuf; 
    double *T = NULL;
    time_precision precision;
    default_time_precision(&precision);
    double time;

    if (size> MAX_PROCESSES || size % 2 != 0) {
        if (rank == 0) {
            printf("You have to use an even number of processes (at most %d)\n", MAX_PROCESSES);
        }
        MPI_Finalize();
        exit(0);
    }

    while (msg_size <= msg_max){

        for (alg = 1; alg <= 2; alg++ ){
            recvbuf = (char *)malloc(msg_size);
            sendbuf = (char *)malloc(msg_size * size);
            initialize_char_array(msg_size, sendbuf);
            set_mca_variable(tuned_scatter_algorithm, alg);
            set_mca_variable(coll_tuned_scatter_algorithm_segmentsize, SEGSIZE);
            MPI_Comm_dup(MPI_COMM_WORLD, &dump_comm);
            if (rank == 0)
                T = (double *)malloc(sizeof(double) * precision.max_reps);
            int reps = 0;
            double totaltime = 0;
            double ci = 0;
            int stop = 0;
            MPI_Barrier(MPI_COMM_WORLD); //Blocks until all processes in the communicator have reached this routine
            while (!stop && reps < precision.max_reps) {
                MPI_Barrier(MPI_COMM_WORLD); 
                if (rank == 0)
                    time = MPI_Wtime();
                MPI_Scatter(sendbuf, msg_size, MPI_CHAR,
                       recvbuf, msg_size, MPI_CHAR, 0, dump_comm);
                MPI_Barrier(MPI_COMM_WORLD);
                if (rank == 0) {
                    T[reps] = MPI_Wtime() - time;
                    totaltime += T[reps];
                }
                reps++;
                if (reps >= precision.min_reps) {
                    if (rank == 0) {
                        ci = time_ci(precision.cl, reps, T);
                        stop = ci * reps / totaltime < precision.eps;
                    }
                    MPI_Bcast(&stop, 1, MPI_INT, 0, dump_comm);
                }
            }
            if (rank == 0) {
                printf("%d %d %d %.10lf \n", size, msg_size, alg, totaltime / reps);
                free(T);
            }
            if (dump_comm != MPI_COMM_NULL)
                MPI_Comm_free(&dump_comm);
        }
        free(sendbuf);
        msg_size += STRIDE; 
    }
    return EXIT_SUCCESS;
}

int main( int argc, char **argv ){
    int provided, err, msg_max, msg_min, stride;
    msg_min = SEGSIZE;
    msg_max = 1048576;
    MPI_Init( &argc, &argv );
    err = MPI_T_init_thread(MPI_THREAD_SINGLE, &provided);
    stride = STRIDE;
    if (err != MPI_SUCCESS)
        MPI_Abort(MPI_COMM_WORLD, err);
    scatter(msg_min, msg_max, stride);
    MPI_T_finalize();
    MPI_Finalize();
}
