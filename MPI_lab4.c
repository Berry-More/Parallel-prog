#include "mpi.h"
#include "omp.h"
#include "math.h"
#include "stdio.h"
#include "stdlib.h"

#define ITTERATIONS_NUM	3
#define ALLOWABLE_ERROR 1e-3


// Инициализация матрицы
void matrixInit(double* matrixArray, int matrixSize, int valueDiag, int valueOther)
{
	for (int i = 0; i < matrixSize; i++)
	{
		for (int j = 0; j < matrixSize; j++)
		{
			if (i == j) matrixArray[i * matrixSize + j] = valueDiag;
			else matrixArray[i * matrixSize + j] = valueOther;
		}
	}
}


// Инициализация вектора, заполнение числом
void vectorSetVal(double* vectorArray, int vectorSize, double fill)
{
	for (int i = 0; i < vectorSize; i++) vectorArray[i] = fill;
}


int taskCheck(double *x, int vectorSize)
{
	for (int i = 0; i < vectorSize; i++)
		if (fabs(x[i] - 1.0) > ALLOWABLE_ERROR) return 1;
	return 0;
}


int main(int argc, char* argv[])
{
	int matrixSize, testVal, threads, c1, c2, provided, rank, size;
	double *x, *b, *A, *y, timeEnd, timeStart;
	
	if (argc < 3)
	{
		matrixSize = 1000;
		threads = 4;
	}
	else if (argc == 3)
	{
		c1 = sscanf(argv[1], "%i", &matrixSize);
		c1 = sscanf(argv[2], "%i", &threads);
		if (c1 != 1 || c2 != 1)
		{
			fprintf(stderr, "Error: invalid command line argument.\n");
			return EXIT_FAILURE;
		}
	}
	else
	{
		fprintf(stderr, "Error: invalid number of arguments. \n");
		return EXIT_FAILURE;
	}
	
	MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
	if (provided != MPI_THREAD_SERIALIZED)
	{
		fprintf(stderr, "MPI_THREAD_SERIALIZRD not available\n");
		return EXIT_FAILURE;
	}
	
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	MPI_Bcast(&matrixSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
	
	b = NULL;
	A = NULL;
	if (rank == 0)
	{
		b = malloc(matrixSize * sizeof(double));
		A = malloc(matrixSize * matrixSize * sizeof(double));
		
		matrixInit(A, matrixSize, 2, 1);
		vectorSetVal(b, matrixSize, matrixSize + 1);
		timeStart = MPI_Wtime();
	}
	
	x = malloc(matrixSize * sizeof(double));
	y = malloc(matrixSize * sizeof(double));
	vectorSetVal(x, matrixSize, 0);
	
	int min_points = matrixSize / size;
	int residue = matrixSize % size;
	
	int *sendcount_v = malloc(size * sizeof(int));
	int *sendcount_m = malloc(size * sizeof(int));
	int *displa_v = malloc(size * sizeof(int));
	int *displa_m = malloc(size * sizeof(int));
	int k = 0;
	
	for (int i = 0; i < size; i++)
	{
		sendcount_v[i] = min_points;
		sendcount_m[i] = min_points * matrixSize;
		if (i < residue)
		{
			sendcount_v[i] += 1;
			sendcount_m[i] += matrixSize;
		}
		displa_v[i] = k;
		displa_m[i] = k * matrixSize;
		k = k + sendcount_v[i];
	}

	// разделение векторов и матриц на куски +
	double *sub_b = malloc(sendcount_v[rank] * sizeof(double));
	double *sub_x = malloc(sendcount_v[rank] * sizeof(double));
	double *sub_y = malloc(sendcount_v[rank] * sizeof(double));
	double *sub_A = malloc(sendcount_m[rank] * sizeof(double));
	
			
	MPI_Scatterv(b, sendcount_v, displa_v, MPI_DOUBLE, sub_b, 
				sendcount_v[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);		
	MPI_Scatterv(A, sendcount_m, displa_m, MPI_DOUBLE, sub_A, 
				sendcount_m[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	// рассчет модуля b +
	double modB = 0;
#pragma omp parallel for num_threads(threads) reduction(+:modB)
	for (int i = 0; i < sendcount_v[rank]; i++) modB += pow(sub_b[i], 2);
	MPI_Allreduce(MPI_IN_PLACE, &modB, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	
	
	int counter = 0;
	while (counter <= ITTERATIONS_NUM)
	{	
		// рассчет модуля y и проверка условие +	
		double modY = 0;
#pragma omp parallel for num_threads(threads) reduction(+:modY)
		for (int i = 0; i < sendcount_v[rank]; i++)
		{
			sub_y[i] = 0;
			for (int j = 0; j < matrixSize; j++)
				sub_y[i] += sub_A[i * sendcount_v[rank] + j] * x[j];
			sub_y[i] -= sub_b[i];
			modY += pow(sub_y[i], 2);
		}
		MPI_Allreduce(MPI_IN_PLACE, &modY, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allgatherv(sub_y, sendcount_v[rank], MPI_DOUBLE, y, sendcount_v,
						displa_v, MPI_DOUBLE, MPI_COMM_WORLD);
		
		
		if (sqrt(modY / modB) < ALLOWABLE_ERROR) break;
	
		// рассчет тау +
		double scalarY = 0;
		double scalarAy = 0;
#pragma omp parallel for num_threads(threads) reduction(+:scalarY, scalarAy)
		for (int i = 0; i < sendcount_v[rank]; i++)
		{
			double Ay = 0;
			for (int j = 0; j < matrixSize; j++)
				Ay += sub_A[i * sendcount_v[rank] + j] * y[j];
			scalarY += y[i] * Ay;
			scalarAy += Ay * Ay;
		}
		MPI_Allreduce(MPI_IN_PLACE, &scalarY, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, &scalarAy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		
#pragma omp parallel for num_threads(threads)
		for (int i = 0; i < matrixSize; i++)		
			x[i] -= y[i] * scalarY / scalarAy;
		
		counter++;
	}
	
	if (rank == 0)
	{	
		timeEnd = MPI_Wtime();
		printf("Task 4 output - %i\n", taskCheck(x, matrixSize));
		printf("Time - %f\n", timeEnd - timeStart);
		free(b), free(A); 
	}
	free(y), free(x), free(sub_b), free(sub_y), free(sub_A), 
	free(sendcount_v), free(displa_v), free(sendcount_m), free(displa_m);
	return 0;
}