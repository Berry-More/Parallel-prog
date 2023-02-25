#include "omp.h"
#include "math.h"
#include "stdio.h"
#include "stdlib.h"


void vectorInit(double* vectorArray, int vectorSize, int threads)
{
#pragma omp parallel for num_threads(threads)
	for (int i = 0; i < vectorSize; i++)
		vectorArray[i] = i;
}


void BLAS_DAXPY(double* x, double* y, double a, int vectorSize, int threads)
{
#pragma omp parallel for num_threads(threads)
  for (int i = 0; i < vectorSize; i++)
		y[i] += a * x[i];
}

int taskCheck(double* vectorIn, double a, int vectorSize, double error)
{
	for (int i = 0; i < vectorSize; i++)
	{
		if ((vectorIn[i] - i*a - i) > error)
			return 1;
	}
	return 0;
}


int main(int argc, char* argv[])
{
  double *v1, *v2, error, t1, t2;
  float a;
  int vectorSize, testVal, threads, c1, c2, c3;

	if (argc < 4)
	{
		vectorSize = 100;
		a = 5.0;
		threads = 4;
	}
	else if (argc == 4)
	{
		c1 = sscanf(argv[1], "%i", &vectorSize);
		c2 = sscanf(argv[2], "%f", &a);
		c3 = sscanf(argv[3], "%i", &threads);
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
  
  v1 = malloc(vectorSize * sizeof(double)); v2 = malloc(vectorSize * sizeof(double));
  vectorInit(v1, vectorSize, threads); vectorInit(v2, vectorSize, threads);
  error = 1e-7;
    
  t1 = omp_get_wtime();
  BLAS_DAXPY(v1, v2, a, vectorSize, threads);
  t2 = omp_get_wtime();
  testVal = taskCheck(v2, a, vectorSize, error);
    
  printf("Task 2 output: %i \n", testVal);
  printf("time: %f \n", (t2 - t1));
  free(v1); free(v2);

  return 0;
}