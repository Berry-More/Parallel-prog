#include "mpi.h"
#include "omp.h"
#include "math.h"
#include "stdio.h"
#include "stdlib.h"

#define PI	3.14159265358979323846


int errorCheck(double value)
{
	if (fabs(value - PI) < 1e-7) return 0;
	else return 1;
}


int main(int argc, char* argv[])
{
	int rank, size, provided, N, threads, c1, c2;
	double timeStart, timeEnd, sum, result;
	sum = 0;
	
	if (argc < 3)
	{
		N = 1000;
		threads = 4;
	}
	else if (argc == 2)
	{
		c1 = sscanf(argv[1], "%i", &N);
		c2 = sscanf(argv[2], "%i", &threads);
		if (c1 != 1 || c2 != 1)
		{
			fprintf(stderr, "Error: invalid command line argument.\n");
			return EXIT_FAILURE;
		}
	}
	else
	{
		fprintf(stderr, "Error: invalid number of arguments.\n");
		return EXIT_FAILURE;
	}
	
	MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
	if (provided != MPI_THREAD_SERIALIZED)
	{
		fprintf(stderr, "MPI_THREAD_SERIALIZED not available\n");
		return EXIT_FAILURE;
	}
	
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	if (rank == 0) timeStart = MPI_Wtime();
	MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
#pragma omp parallel for num_threads(threads) reduction(+:sum)
	for (int i = rank; i < N; i+=size) sum += pow((-1), i) / (2 * i + 1);
	sum *= 4;
	MPI_Reduce(&sum, &result, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	
	if (rank == 0)
	{
		timeEnd = MPI_Wtime();
		printf("Task 1 output - %i\n", errorCheck(result));
		printf("Result - %f\n", result);
		printf("Time - %f\n", timeEnd - timeStart);
	}
	
	MPI_Finalize();
	return EXIT_SUCCESS;
}