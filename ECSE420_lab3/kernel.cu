
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

using namespace std;

#define n 0.0002    //given parameter
#define p 0.5		//given parameter
#define G 0.75		//given parameter

#define N 4			//size

int seqSolver(float u[][N], float u1[][N], float u2[][N]);
int display(float u[][N]);


int main(int argc, char* argv[])
{
	int iterations = atoi(argv[1]);

	float seqDrum_U[N][N] = { 0 };
	float seqDrum_U1[N][N] = { 0 };
	float seqDrum_U2[N][N] = { 0 };

	seqDrum_U1[N / 2][N / 2] += 1.0f;

	for (int i = 0; i < iterations; i++) {
		seqSolver(seqDrum_U, seqDrum_U1, seqDrum_U2);
		printf("U[N/2][N/2] after %d interation: %3.6f\n", i, seqDrum_U[N / 2][N / 2]);
		display(seqDrum_U);
		memcpy(seqDrum_U2, seqDrum_U1, N * N * sizeof(float));
		memcpy(seqDrum_U1, seqDrum_U, N * N * sizeof(float));
	}
	

	return 0;
}

int seqSolver(float u[][N], float u1[][N], float u2[][N]) {
	
	for (int i = 1; i < N - 1; i++) {
		for (int j = 1; j < N - 1; j++) {
			u[i][j] = (p*(u1[i-1][j]+u1[i+1][j]+u1[i][j-1]+u1[i][j+1]-4*u1[i][j])+2*u1[i][j]-(1-n)*u2[i][j]) / (1 + n);
		}
	}

	for (int i = 1; i < N - 1; i++) {
		u[0][i] = G * u[1][i];
		u[N - 1][i] = G * u[N - 2][i];
		u[i][0] = G * u[i][1];
		u[i][N - 1] = G * u[i][N - 2];
	}

	u[0][0] = G * u[1][0];
	u[N - 1][0] = G * u[N - 2][0];
	u[0][N - 1] = G * u[0][N - 2];
	u[N - 1][N - 1] = G * u[N - 1][N - 2];

	return 0;
}


int display(float u[][N]) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			printf("(%d,%d): %3.6f ", i, j, u[i][j]);
		}
		printf("\n");
	}
	printf("\n");

	return 0;
}

