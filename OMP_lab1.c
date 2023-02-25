#include "omp.h"
#include "math.h"
#include "stdio.h"
#include "stdlib.h"

#define PI  3.14159265358979323846


double pi_value(int n, int threads)
{
	double sum = 0;
#pragma omp parallel for num_threads(threads) reduction(+:sum)
	for (int i = 0; i <= n; i++)
	{
		sum += pow((-1), i) / (2 * i + 1);
	}
	sum *= 4;
	return sum;
}

int taskCheck(double calc_pi, double teor_pi, double error)
{
	if (fabs(calc_pi - teor_pi) < error)
		return 0;
	else
		return 1;
}

int main(int argc, char* argv[])
{
    int N, threads, c1, c2;
	double piCalc, error, t1, t2;
	error = 1e-7;

	if (argc < 3)
	{
		N = 1000;
		threads = 4;
	}
	else if (argc == 3)
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

	t1 = omp_get_wtime();
	piCalc = pi_value(N, threads);
	t2 = omp_get_wtime();

    int testVal = taskCheck(piCalc, PI, error);
	printf("Task 1 output: %i \n", testVal);
	printf("time: %f \n", (t2 - t1));

    return 0;
}