#include "mpi.h"
#include "omp.h"
#include "stdio.h"
#include "stdlib.h"


#define ALLOWABLE_ERROR	1e-7


int taskCheck(double* vectorIn, double a, int vectorSize)
{
	for (int i = 0; i < vectorSize; i++)
		if ((vectorIn[i] - i*a - i) > ALLOWABLE_ERROR) return 1;
	return 0;
}


int main(int argc, char* argv[])
{
	double *v1, *v2, *sub_v1, *sub_v2, *result_v, a, timeStart, timeEnd;
	int vectorSize, threads, provided, rank, size, c1, c2, c3;

	if (argc < 4)
	{
		vectorSize = 40;
		a = 5;
		threads = 4;
	}
	else if (argc == 4)
	{
		c1 = sscanf(argv[1], "%i", &vectorSize);
		c2 = sscanf(argv[2], "%f", &a);
		c3 = sscanf(argv[3], "%f", &threads);
		if (c1 != 1 || c2 != 1 || c3 != 1)
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
	
	if (rank == 0)
	{
		v1 = malloc(vectorSize * sizeof(double));
		v2 = malloc(vectorSize * sizeof(double));
#pragma omp parallel for num_threads(threads)
		for (int i = 0; i < vectorSize; i++)
		{
			v1[i] = i;
			v2[i] = i;
		}
		timeStart = MPI_Wtime();
	}
	
	MPI_Bcast(&vectorSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&a, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	int min_points = vectorSize / size;
	int residue = vectorSize % size;
	
	int *sendcount = malloc(size * sizeof(int));
	int *displa = malloc(size * sizeof(int));
	int k = 0;
	for (int i = 0; i < size; i++)
	{
		if (i < residue) sendcount[i] = min_points + 1;
		else sendcount[i] = min_points;
		displa[i] = k;
		k = k + sendcount[i];
	}
	
	sub_v1 = malloc(sendcount[rank] * sizeof(double));
	sub_v2 = malloc(sendcount[rank] * sizeof(double));
	
	MPI_Scatterv(v1, sendcount, displa, MPI_DOUBLE, sub_v1, 
				sendcount[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Scatterv(v2, sendcount, displa, MPI_DOUBLE, sub_v2, 
				sendcount[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
				
#pragma omp parallel for num_threads(threads)
	for (int i = 0; i < sendcount[rank]; i++) sub_v2[i] += a * sub_v1[i];
	
	result_v = NULL;
	if (rank == 0) result_v = malloc(vectorSize * sizeof(double));
	
	MPI_Gatherv(sub_v2, sendcount[rank], MPI_DOUBLE, result_v, sendcount, 
				displa, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	if (rank == 0)
	{
		timeEnd = MPI_Wtime();
		printf("Task 3 output - %i\n", taskCheck(result_v, a, vectorSize));
		printf("Time - %f\n", timeEnd - timeStart);
		free(v1), free(v2), free(result_v);
	}
	
	free(sub_v1), free(sub_v2), free(sendcount), free(displa);
	MPI_Finalize();
	return 0;
}