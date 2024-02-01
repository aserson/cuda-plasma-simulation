#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "kernels.cuh"

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <cstdlib>
#include <ctime>
#include <iostream> 
#include <fstream> 
#include <iomanip>
#include <string> 
#include <sstream>
#include <direct.h>

#include <cufft.h>

#define nu	1e-4
#define eta	1e-4
#define beta  0.
#define G_K 0.0
#define G_B	0.0
#define k0	10
#define dk	3
#define E_K	0.5
#define E_B 0.5
#define Tend 5.0
#define B0 0.0   
#define Nsteps 2
#define FlagOut 1
#define alpha 0.00
#define M_PI 3.141592653589793238462643

#if	Nsteps==2
const double cfl = 0.2;
#elif Nsteps==3	
const double cfl = 0.5;
#endif

const int N = 512;
const double lamda = 1. / ((double)(N * N));
const double h = 2. * M_PI / (double)N;

double dTout = 0.1;
double Tout = dTout;

const int XYmax = N / 3;

cudaEvent_t start, stop;
float elapsedTime;

double dt;
double dt0 = 0.01;

cufftDoubleComplex* S, * w, * A, * j;
cufftDoubleComplex* wold, * F, * Aold;
cufftDoubleComplex* Jack1;

double* a_d, * b_d, * c_d;
double* ux, * uy;

int* WN;
int Nk = 0;


double phi, phiA;
double* Phi;
int Kxy, KxyA;

double* outH;

const int csize = N * (N / 2 + 1) * sizeof(cufftDoubleComplex);
const int dsize = N * N * sizeof(double);
std::string str;

cufftHandle plan, bplan;

int num_out = 0;


FILE* Fout;
std::ofstream FoutW;
std::ofstream Fout2;
FILE* Ftime = fopen("outputs/time.txt", "a");


const int K1 = 32;
const int K2 = 16;
const int K = 128;


dim3 dimBlock(K1, K2); //размер блока
dim3 dimGrid(N / K1, N / (2 * K2)); //количество блоков
dim3 dimGrid2(N / K1, N / K2); //количество блоков


#include "function.h"



__global__
void RightKer1(cufftDoubleComplex* w, cufftDoubleComplex* S, cufftDoubleComplex* A, cufftDoubleComplex* Jack1, cufftDoubleComplex* F)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int id = (N / 2 + 1) * idx + idy;


	if (idx > N / 2) idx = idx - N;

	double r = (double)(idx * idx + idy * idy);
	double x = (double)idx;


	F[id].x = Jack1[id].x - nu * r * w[id].x - beta * x * S[id].y - B0 * x * r * A[id].y - alpha * w[id].x;
	F[id].y = Jack1[id].y - nu * r * w[id].y + beta * x * S[id].x + B0 * x * r * A[id].x - alpha * w[id].y;


}

__global__
void RightKer2(cufftDoubleComplex* Jack1, cufftDoubleComplex* F)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int id = (N / 2 + 1) * idx + idy;


	if (idx > N / 2) idx = idx - N;

	F[id].x = F[id].x + Jack1[id].x;
	F[id].y = F[id].y + Jack1[id].y;

}

__global__
void RightKer3(cufftDoubleComplex* S, cufftDoubleComplex* A, cufftDoubleComplex* Jack1, cufftDoubleComplex* F)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int id = (N / 2 + 1) * idx + idy;


	if (idx > N / 2) idx = idx - N;

	double r = (double)(idx * idx + idy * idy);
	double x = (double)idx;


	F[id].x = Jack1[id].x - eta * r * A[id].x - B0 * x * S[id].y;
	F[id].y = Jack1[id].y - eta * r * A[id].y + B0 * x * S[id].x;


}

void Right1(cufftDoubleComplex* A, cufftDoubleComplex* j, cufftDoubleComplex* S, cufftDoubleComplex* w, cufftDoubleComplex* F) {
	Jackob(S, w, Jack1);		RightKer1 << <dimGrid, dimBlock >> > (w, S, A, Jack1, F);
	Jackob(A, j, Jack1);		RightKer2 << <dimGrid, dimBlock >> > (Jack1, F);
}

void Right2(cufftDoubleComplex* A, cufftDoubleComplex* j, cufftDoubleComplex* S, cufftDoubleComplex* w, cufftDoubleComplex* F) {

	Jackob(S, A, Jack1);
	RightKer3 << <dimGrid, dimBlock >> > (S, A, Jack1, F);

}


void CreateFolder() {

	char parentfoldername[40];
	char foldername[40];

	time_t seconds = time(NULL);
	strftime(parentfoldername, 40, "%Y%m%d/", localtime(&seconds));
	strftime(foldername, 40, "%Y%m%d_%H%M%S", localtime(&seconds));

	std::cout << foldername << std::endl;
	std::cout << "N = " << std::setw(13) << std::left << N << "Tend=" << std::setw(12) << std::left << Tend << "dTout=" << dTout << std::endl;
	std::cout << "E_u0 = " << std::setw(10) << std::left << E_K << "E_b0 = " << std::setw(10) << std::left << E_B << "F_U = " << std::setw(10) << std::left << G_K << "F_B = " << G_B << std::endl;
	std::cout << "beta = " << std::setw(10) << std::left << beta << "B0   = " << std::setw(10) << std::left << B0 << "nu  = " << std::setw(10) << std::left << nu << "eta = " << eta << std::endl;

	str = "outputs/data/" + std::string(parentfoldername);
	if (mkdir(str.c_str()) != 0)
		std::cout << "error create directory " << str << std::endl;
	str = str + "/" + std::string(foldername);
	if (mkdir(str.c_str()) != 0) 
		std::cout << "error create directory " << str << std::endl;

	Fout = fopen((str + "/out").c_str(), "w+");

	Fout2.open((str + "/out2").c_str());
	Fout2 << "N=\t" << N << "\nTend=\t" << Tend << "\ndTout=\t" << dTout << "\nE_K=\t" << E_K << "\nE_B=\t" << E_B << std::endl;
	Fout2 << "G_K=\t" << G_K << "\nG_B=\t" << G_B << "\nk0=\t" << k0 << "\ndk=\t" << dk << std::endl;
	Fout2 << "beta=\t" << beta << "\nB0=\t" << B0 << "\nnu=\t" << nu << "\neta=\t" << eta << "\ncfl=\t" << cfl << std::endl;
	Fout2.close();

	if (mkdir((str + "/outW").c_str()) != 0) 
		std::cout << "error create directory outW" << std::endl;
	if (mkdir((str + "/outJ").c_str()) != 0) 
		std::cout << "error create directory outJ" << std::endl;

}

void cuda_main() {

	cudaSetDevice(0);


	CreateFolder();

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	ux = (double*)malloc((N * N / K) * sizeof(double));
	uy = (double*)malloc((N * N / K) * sizeof(double));

	cudaHostAlloc((void**)&outH, dsize, cudaHostAllocDefault);

	if (cudaMalloc((void**)&Jack1, csize) != 0)
		fprintf(stderr, "cudaMalloc error Jack1");

	if (cudaMalloc((void**)&a_d, dsize) != 0)
		fprintf(stderr, "cudaMalloc error");

	if (cudaMalloc((void**)&b_d, dsize) != 0)
		fprintf(stderr, "cudaMalloc error");

	if (cudaMalloc((void**)&c_d, dsize) != 0)
		fprintf(stderr, "cudaMalloc error");

	if (cudaMalloc((void**)&S, csize) != 0)
		fprintf(stderr, "cudaMalloc error");

	if (cudaMalloc((void**)&w, csize) != 0)
		fprintf(stderr, "cudaMalloc error");

	if (cudaMalloc((void**)&A, csize) != 0)
		fprintf(stderr, "cudaMalloc error");

	if (cudaMalloc((void**)&j, csize) != 0)
		fprintf(stderr, "cudaMalloc error");

	if (cudaMalloc((void**)&wold, csize) != 0)
		fprintf(stderr, "cudaMalloc error");

	if (cudaMalloc((void**)&F, csize) != 0)
		fprintf(stderr, "cudaMalloc error\n");

	if (cudaMalloc((void**)&Aold, csize) != 0)
		fprintf(stderr, "cudaMalloc error Aold\n");

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	if (cufftPlan2d(&plan, N, N, CUFFT_D2Z) != CUFFT_SUCCESS)
		fprintf(stderr, "CUFFT error: Plan creation failed\n");
	if (cufftPlan2d(&bplan, N, N, CUFFT_Z2D) != CUFFT_SUCCESS)
		fprintf(stderr, "CUFFT error: Plan creation failed\n");

	cudaEventRecord(start, 0);

	//srand ( time(NULL) );

	//Initial Conditions Kinetic Part

	Zero << <dimGrid, dimBlock >> > (S);


	Phi = (double*)malloc(N * (N / 2 + 1) * sizeof(double));
	for (int i = 0; i < N * (N / 2 + 1); i++)
		Phi[i] = 2 * M_PI * (double)rand() / RAND_MAX;
	std::cout << Phi[0] << "\t" << Phi[1] << "\t" << Phi[2] << "\t" << Phi[3] << std::endl;
	cudaMemcpy(a_d, Phi, N * (N / 2 + 1) * sizeof(double), cudaMemcpyHostToDevice);

	ZeroTimeNew << <dimGrid, dimBlock >> > (S, a_d);

	double EnergyS = energy(S);
	Multiplication2 << <dimGrid, dimBlock >> > (S, S, sqrt(E_K / EnergyS));
	MLaplas << <dimGrid, dimBlock >> > (S, w);
	EnergyS = energy(S);

	//Initial Conditions Magnetic Part

	Zero << <dimGrid, dimBlock >> > (A);

	Phi = (double*)malloc(N * (N / 2 + 1) * sizeof(double));
	for (int i = 0; i < N * (N / 2 + 1); i++)
		Phi[i] = 2 * M_PI * (double)rand() / RAND_MAX;
	std::cout << Phi[0] << "\t" << Phi[1] << "\t" << Phi[2] << "\t" << Phi[3] << std::endl;
	cudaMemcpy(a_d, Phi, N * (N / 2 + 1) * sizeof(double), cudaMemcpyHostToDevice);

	ZeroTimeNew << <dimGrid, dimBlock >> > (A, a_d);

	double EnergyA = energy(A);
	Multiplication2 << <dimGrid, dimBlock >> > (A, A, sqrt(E_B / EnergyA));
	Laplas << <dimGrid, dimBlock >> > (A, j);
	EnergyA = energy(A);

	//Forcing Wavenumbers

	WN = (int*)malloc(sizeof(int) * 2 * 4 * (k0 + dk) * (k0 + dk));
	WaveNumbers(WN, Nk, k0 - dk, k0 + dk);

	//Zero Time Step

	double maxS = max(S);
	double maxA = max(A);

	if (maxA > maxS) {
		dt = cfl * h / maxA;
	}
	else {
		dt = cfl * h / maxS;
	}
	if (dt > dt0) dt = dt0;

	//Zero Data Output

	int 	q = 0;	//Time Step Counter
	double 	T = 0.;	//Time

	PRINT(w, num_out, 'k');
	PRINT(j, num_out, 'm');
	//fprintf(Fout,"%f\t%d\t%f\t%f\t%f\t%f\tout0\n",T,q,dt,EnergyS,EnergyA,EnergyS+EnergyA);	
	fprintf(Fout, "%f\t%d\t%f\t%f\t%f\t%f\n", T, q, dt, EnergyS, EnergyA, EnergyS + EnergyA);

	//Zero Consile Output

	num_out++;

	printf("----------------------------------------------------------------------------\n");
	printf("%f\t%d\t%f\t%f\t%f\t%f\tout0\n", T, q, dt, EnergyS, EnergyA, EnergyS + EnergyA);

	//Main Cycle of the Program 

	while (T < Tend) {

		q++;
		T = T + dt;

		//Old Data Saving

		cudaMemcpy(wold, w, csize, cudaMemcpyDeviceToDevice);
		cudaMemcpy(Aold, A, csize, cudaMemcpyDeviceToDevice);

		//Time Integration Scheme

#if Nsteps==2	/* Two-step Scheme */ 		

		Right1(A, j, S, w, F);
		TshemVes << <dimGrid, dimBlock >> > (F, wold, w, dt, 1.);
		Right2(A, j, S, w, F);
		TshemVes << <dimGrid, dimBlock >> > (F, Aold, A, dt, 1.);
		BMLaplas << <dimGrid, dimBlock >> > (w, S);
		Laplas << <dimGrid, dimBlock >> > (A, j);

#elif Nsteps==3	/* Three-step Scheme */ 

		Right1(A, j, S, w, F);
		TshemVes << <dimGrid, dimBlock >> > (F, wold, w, dt, 1. / 3.);
		Right2(A, j, S, w, F);
		TshemVes << <dimGrid, dimBlock >> > (F, Aold, A, dt, 1. / 3.);
		BMLaplas << <dimGrid, dimBlock >> > (w, S);
		Laplas << <dimGrid, dimBlock >> > (A, j);

		Right1(A, j, S, w, F);
		TshemVes << <dimGrid, dimBlock >> > (F, wold, w, dt, 1. / 2.);
		Right2(A, j, S, w, F);
		TshemVes << <dimGrid, dimBlock >> > (F, Aold, A, dt, 1. / 2.);
		BMLaplas << <dimGrid, dimBlock >> > (w, S);
		Laplas << <dimGrid, dimBlock >> > (A, j);

#endif

		//Forcing step

		phi = 2 * M_PI * (double)rand() / RAND_MAX;
		Kxy = rand() % Nk;
		Right1(A, j, S, w, F);
		TshemF << <dimGrid, dimBlock >> > (F, wold, w, WN[2 * Kxy], WN[2 * Kxy + 1], phi, dt, G_K);

		phi = 2 * M_PI * (double)rand() / RAND_MAX;
		Kxy = rand() % Nk;
		Right2(A, j, S, w, F);
		TshemFA << <dimGrid, dimBlock >> > (F, Aold, A, WN[2 * Kxy], WN[2 * Kxy + 1], phi, dt, G_B);

		BMLaplas << <dimGrid, dimBlock >> > (w, S);
		Laplas << <dimGrid, dimBlock >> > (A, j);

		//New Time Step
		maxS = max(S);
		maxA = max(A);

		if (maxA > maxS) {
			dt = cfl * h / maxA;
		}
		else {
			dt = cfl * h / maxS;
		}
		if (dt > dt0) dt = dt0;

		//Actual Eneries
		EnergyS = energy(S);
		EnergyA = energy(A);

		//Data Output
		if (T >= Tout) {

			PRINT(w, num_out, 'k');
			PRINT(j, num_out, 'm');
			//fprintf(Fout,"%f\t%d\t%f\t%f\t%f\t%f\tout%d\n",T,q,dt,EnergyS,EnergyA,EnergyS+EnergyA,num_out);	
			fprintf(Fout, "%f\t%d\t%f\t%f\t%f\t%f\n", T, q, dt, EnergyS, EnergyA, EnergyS + EnergyA);
			printf("%f\t%d\t%f\t%f\t%f\t%f\n", T, q, dt, EnergyS, EnergyA, EnergyS + EnergyA);

			Tout = Tout + dTout;
			num_out++;

		}
		else {

			//fprintf(Fout,"%f\t%d\t%f\t%f\t%f\t%f\tnone\n",T,q,dt,EnergyS,EnergyA,EnergyS+EnergyA);	
			fprintf(Fout, "%f\t%d\t%f\t%f\t%f\t%f\n", T, q, dt, EnergyS, EnergyA, EnergyS + EnergyA);
			printf("%f\t%d\t%f\t%f\t%f\t%f\tnone\n", T, q, dt, EnergyS, EnergyA, EnergyS + EnergyA);

		}

	}

	//ProgramTime Output
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("%3.1f\n", elapsedTime / 1000);
	fprintf(Ftime, "%f	%d		%d=%d*%d	%3.1f\n", T, q, N, K1, N / K1, elapsedTime / 1000);

	//Memory Clearing
	free(Phi);
	free(WN);
	free(ux);
	free(uy);

	cudaFree(a_d);
	cudaFree(b_d);
	cudaFree(c_d);
	cudaFreeHost(outH);

	cudaFree(w);
	cudaFree(S);
	cudaFree(j);
	cudaFree(A);
	cudaFree(F);
	cudaFree(wold);
	cudaFree(Aold);
	cudaFree(Jack1);

	cufftDestroy(plan);
	cufftDestroy(bplan);

	cudaEventDestroy(stop);
	cudaEventDestroy(start);

	fclose(Fout);
	fclose(Ftime);

}
