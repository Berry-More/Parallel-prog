#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define ITTERATIONS_NUM	3


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

// Поиск решения
void findSolution(double *A, double *x, double *b, int vectorSize, double eps, int threads)
{
	int counter;
	double *y, modY, modB, scalarY, scalarAy;

	y =  malloc(vectorSize * sizeof(double));

	modB = 0;
	for (int i = 0; i < vectorSize; i++)
		modB += pow(b[i], 2);

	counter = 0;
	while (counter <= ITTERATIONS_NUM)
	{
		modY = 0;
#pragma omp parallel for num_threads(threads) reduction(+:modY)
		for (int i = 0; i < vectorSize; i++)
		{	
			y[i] = 0;
			for (int j = 0; j < vectorSize; j++)
				y[i] += A[i * vectorSize + j] * x[j];
			y[i] -= b[i];
			modY += pow(y[i], 2);
		}

		if (sqrt(modY / modB) < eps)
			break;

		scalarY = 0;
		scalarAy = 0;
#pragma omp parallel for num_threads(threads) reduction(+:scalarY, scalarAy)
		for (int i = 0; i < vectorSize; i++)
		{
			double Ay = 0;
			for (int j = 0; j < vectorSize; j++)
				Ay += A[i * vectorSize + j] * y[j];

			scalarY += y[i] * Ay;
			scalarAy += Ay * Ay;
		}

#pragma omp parallel for num_threads(threads)
		for (int i = 0; i < vectorSize; i++)
			{
				x[i] -= y[i] * scalarY / scalarAy;
			}
		counter++;
	}
}

int taskCheck(int matrixSize, double error, int threads)
{
	double *x, *b, *A, t1, t2;

	x = malloc(matrixSize * sizeof(double));
	b = malloc(matrixSize * sizeof(double));
	A = malloc(matrixSize * matrixSize * sizeof(double));

	matrixInit(A, matrixSize, 2, 1);
	vectorSetVal(x, matrixSize, 0);
	vectorSetVal(b, matrixSize, matrixSize + 1);

	t1 = omp_get_wtime();
	findSolution(A, x, b, matrixSize, 1, threads);
	t2 = omp_get_wtime();
	printf("time: %f \n", (t2 - t1));

	for (int i = 0; i < matrixSize; i++)
	{	
		if (fabs(x[i] - 1.0) > error)
		return 1;
	}
	return 0;
}

int main(int argc, char* argv[])
{
	int matrixSize, testVal, threads, c1, c2;
	double error = 1e-7;

	if (argc < 3)
	{
		matrixSize = 1000;
		threads = 4;
	}
	else if (argc == 3)
	{
		c1 = sscanf(argv[1], "%i", &matrixSize);
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
	
	testVal = taskCheck(matrixSize, error, threads);
	printf("Task 4 output: %i\n", testVal);

	return 0;
}