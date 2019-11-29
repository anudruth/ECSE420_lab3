
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
int display(float* u);

__global__ void parallelSolverP1(float* u, float* u1, float* u2)
{
	if ((threadIdx.x > N) && (threadIdx.x < (N * N - N)) && (threadIdx.x % N != 0) && (threadIdx.x % N + 1 != N)) {
		u[threadIdx.x] = (p * (u1[threadIdx.x - N] + u1[threadIdx.x + N] + u1[threadIdx.x - 1] + u1[threadIdx.x + 1] - 4 * u1[threadIdx.x]) + 2 * u1[threadIdx.x] - (1 - n) * u2[threadIdx.x]) / (1 + n);
	}
}

__global__ void parallelSolverP2(float* u) 
{
	if (!((threadIdx.x > N) && (threadIdx.x < (N * N - N)) && (threadIdx.x % N != 0) && (threadIdx.x % N + 1 != N)) &&
		((threadIdx.x != 0) || (threadIdx.x != N * N - 1) || (threadIdx.x != N * N - N) || (threadIdx.x != N - 1))) 
	{
		if (threadIdx.x < N) u[threadIdx.x] = G * u[threadIdx.x + N];
		if ((threadIdx.x > N*N-N) && (threadIdx.x < N*N)) u[threadIdx.x] = G * u[threadIdx.x-N];
		if (threadIdx.x % N == 0) u[threadIdx.x] = G * u[threadIdx.x + 1];
		if (threadIdx.x % N + 1 == N) u[threadIdx.x] = G * u[threadIdx.x - 1];
	}
}

__global__ void parallelSolverP3(float* u) 
{
	if (threadIdx.x == 0) u[threadIdx.x] = G * u[threadIdx.x + N];
	if (threadIdx.x == N * N - 1) u[threadIdx.x] = G * u[threadIdx.x - 1];
	if (threadIdx.x == N * N - N) u[threadIdx.x] = G * u[threadIdx.x - N];
	if (threadIdx.x == N - 1) u[threadIdx.x] = G * u[threadIdx.x - 1];
}

int main(int argc, char* argv[])
{
	int iterations = atoi(argv[1]);


	// sequential implimentation
	float seqDrum_U[N][N] = { 0 };
	float seqDrum_U1[N][N] = { 0 };
	float seqDrum_U2[N][N] = { 0 };

	seqDrum_U1[N / 2][N / 2] += 1.0f;


	printf("\nSequential implementation (part 1):\n");
	for (int i = 0; i < iterations; i++) {
		seqSolver(seqDrum_U, seqDrum_U1, seqDrum_U2);
		printf("U[N/2][N/2] after %d interation: %3.6f\n", i, seqDrum_U[N / 2][N / 2]);
		display(seqDrum_U);
		memcpy(seqDrum_U2, seqDrum_U1, N * N * sizeof(float));
		memcpy(seqDrum_U1, seqDrum_U, N * N * sizeof(float));
	}

	//free(seqDrum_U);
	//free(seqDrum_U1);
	//free(seqDrum_U2);

	//parallel implementation 
	float parDrum_U[N * N] = { 0 };
	float parDrum_U1[N * N] = { 0 };
	float parDrum_U2[N * N] = { 0 };

	float* d_parDrum_U;
	float* d_parDrum_U1;
	float* d_parDrum_U2;

	parDrum_U1[10] += 1;

	cudaMallocManaged((void**)&d_parDrum_U, N * N * sizeof(float));
	cudaMallocManaged((void**)&d_parDrum_U1, N * N * sizeof(float));
	cudaMallocManaged((void**)&d_parDrum_U2, N * N * sizeof(float));

	cudaMemcpy(d_parDrum_U, parDrum_U, N * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_parDrum_U1, parDrum_U1, N * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_parDrum_U2, parDrum_U2, N * N * sizeof(float), cudaMemcpyHostToDevice);

	printf("\nParallel implementation (part 2):\n");
	for (int i = 0; i < iterations; i++) {
		parallelSolverP1 << <1, 16 >> > (d_parDrum_U, d_parDrum_U1, d_parDrum_U2);
		cudaDeviceSynchronize();
		parallelSolverP2 << <1, 16 >> > (d_parDrum_U);
		cudaDeviceSynchronize();
		parallelSolverP3 << <1, 16 >> > (d_parDrum_U);
		cudaDeviceSynchronize();
		printf("U[N/2][N/2] after %d interation: %3.6f\n", i, d_parDrum_U[10]);
		display(d_parDrum_U);
		cudaMemcpy(d_parDrum_U2, d_parDrum_U1, N * N * sizeof(float), cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_parDrum_U1, d_parDrum_U, N * N * sizeof(float), cudaMemcpyDeviceToDevice);
	}
	/*
	free(parDrum_U);
	free(parDrum_U1);
	free(parDrum_U2);
	cudaFree(d_parDrum_U);
	cudaFree(d_parDrum_U1);
	cudaFree(d_parDrum_U2);
	*/
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

int display(float* u) {
	int index = 0;
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			printf("(%d,%d): %3.6f ", i, j, u[index++]);
		}
		printf("\n");
	}
	printf("\n");
	return 0;
}

