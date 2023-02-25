#include "mpi.h"
#include "stdio.h"
#include "stdlib.h"


void print_array(int* array, int arraySize)
{
	for (int i = 0; i < arraySize; i++) printf("%i ", array[i]);
	printf("\n\n");
}


int main(int argc, char* argv[])
{
	int *v, *sub_v, *result_v, *sendcount, *displa, vectorSize,
	provided, min_points, size, rank, residue, c;

	if (argc < 2)
	{
		vectorSize = 40;
	}
	else if (argc == 2)
	{
		c = sscanf(argv[1], "%i", &vectorSize);
		if (c != 1)
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
		v = malloc(vectorSize * sizeof(int));
		for (int i = 0; i < vectorSize; i++) v[i] = i;
		print_array(v, vectorSize);
	}
  
	min_points = vectorSize / size;
	residue = vectorSize % size;
	
	sendcount = malloc(size * sizeof(int));
	displa = malloc(size * sizeof(int));
	int k = 0;
	for (int i = 0; i < size; i++)
	{
		if (i < residue) sendcount[i] = min_points + 1;
		else sendcount[i] = min_points;
		displa[i] = k;
		k = k + sendcount[i];
	}
	sub_v = malloc(sendcount[rank] * sizeof(int));
	MPI_Scatterv(v, sendcount, displa, MPI_INT, sub_v, 
				sendcount[rank], MPI_INT, 0, MPI_COMM_WORLD);
	printf("current rank - %i\n", rank);
	print_array(sub_v, sendcount[rank]);
	
	result_v = NULL;
	if (rank == 0) result_v = malloc(vectorSize * sizeof(int));
	
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Gatherv(sub_v, sendcount[rank], MPI_INT, result_v, sendcount, 
				displa, MPI_INT, 0, MPI_COMM_WORLD);
	if (rank == 0) print_array(result_v, vectorSize);
	
	free(v), free(sub_v), free(result_v), free(sendcount), free(displa);
	MPI_Finalize();
	return 0;
}