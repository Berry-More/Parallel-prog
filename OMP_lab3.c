#include "omp.h"
#include "math.h"
#include "stdio.h"
#include "stdlib.h"


// сделать проверку по элементам тоже
void matrixInit(double* matrixArray, int matrixSize, int valueDiag, int valueOther, int threads)
{
#pragma omp parallel for num_threads(threads)
    for (int i = 0; i < matrixSize; i++)
    {
        for (int j = 0; j < matrixSize; j++)
        {
            if (i == j)
                matrixArray[i * matrixSize + j] = valueDiag;
            else
                matrixArray[i * matrixSize + j] = valueOther;
        }
    }
}

void BLAS_DGEMM(double* matrixIn1, double* matrixIn2, double* matrixOut, int matrixSize, int threads)
{
    matrixInit(matrixOut, matrixSize, 0, 0, threads);
#pragma omp parallel for num_threads(threads)
    for (int i = 0; i < matrixSize; i++)
    {
        for (int k = 0; k < matrixSize; k++)
        {
            for (int j = 0; j < matrixSize; j++)
                matrixOut[i * matrixSize + j] += matrixIn1[i * matrixSize + k] * matrixIn2[k * matrixSize + j];
        }
    }
}

int taskCheck(double* matrix, int matrixSize)
{
    double error;
    error = 1e-7;
    for (int i = 0; i < matrixSize; i++)
    {
        for (int j = 0; j < matrixSize; j++)
        {
            if ((i == j) && (matrix[i * matrixSize + j] - 1) > error)
                return 1;
            else if ((i != j) && matrix[i * matrixSize + j] > error)
                return 1;
        }
    }
    return 0;
}


int main(int argc, char* argv[])
{
    int matrixSize, threads, testVal, c1, c2;
    double *m1, *m2, *out, t1, t2;
    
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

    m1 = malloc(matrixSize * matrixSize * sizeof(double));
    m2 = malloc(matrixSize * matrixSize * sizeof(double));
    out = malloc(matrixSize * matrixSize * sizeof(double));

    if (m1 == NULL || m2 == NULL || out == NULL)
    {
        return 1;
    }

    matrixInit(m1, matrixSize, 1, 0, threads); 
    matrixInit(m2, matrixSize, 1, 0, threads);

    t1 = omp_get_wtime();
    BLAS_DGEMM(m1, m2, out, matrixSize, threads);
    t2 = omp_get_wtime();
    printf("time: %f \n", (t2 - t1));
    
    testVal = taskCheck(out, matrixSize);
    printf("Task 3 output: %i\n", testVal);

    free(m1);
    free(m2);
    free(out);
    return 0;
}