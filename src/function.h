void WaveNumbers(int* WN, int& NumK, int k_min, int k_max)
{
	int r;

	for (int i1 = -k_max; i1 < k_max + 1; i1++) 	for (int i2 = 1; i2 < k_max + 1; i2++) {

		r = i1 * i1 + i2 * i2;

		if ((r <= k_max * k_max) && (r >= k_min * k_min)) {

			WN[2 * NumK] = i1;
			WN[2 * NumK + 1] = i2;
			if (i1 < 0) WN[2 * NumK] = N + i1;
			NumK++;
		}
	}

	for (int i1 = k_min; i1 < k_max + 1; i1++) {

		WN[2 * NumK] = i1;
		WN[2 * NumK + 1] = 0;

		NumK++;
	}


}


__global__
void Zero(cufftDoubleComplex* f)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int id = (N / 2 + 1) * idx + idy;

	f[id].x = 0.;
	f[id].y = 0.;

}

__global__
void ZeroTime(cufftDoubleComplex* f, double* fi)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int id = (N / 2 + 1) * idx + idy;

	if (idx > N / 2) idx = idx - N;

	double k = sqrt((double)(idx * idx + idy * idy));

	//~ f[id].x=sqrt(k)*exp(-k*k/(2.*k0*k0))*cos(fi[id])*(double)(N*N);
	//~ f[id].y=sqrt(k)*exp(-k*k/(2.*k0*k0))*sin(fi[id])*(double)(N*N);


	//f[id].x=0.;
	//f[id].y=0.;

	if ((k > k0 - dk) && (k < k0 + dk)) {
		f[id].x += cos(fi[id]) * (double)(N * N) / (double)k;
		f[id].y += sin(fi[id]) * (double)(N * N) / (double)k;

	}



}

__global__
void ZeroTimeNew(cufftDoubleComplex* f, double* fi)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int id = (N / 2 + 1) * idx + idy;

	if (idx > N / 2) idx = idx - N;

	double k = sqrt((double)(idx * idx + idy * idy));

	//~ f[id].x=sqrt(k)*exp(-k*k/(2.*k0*k0))*cos(fi[id])*(double)(N*N);
	//~ f[id].y=sqrt(k)*exp(-k*k/(2.*k0*k0))*sin(fi[id])*(double)(N*N);


	//f[id].x=0.;
	//f[id].y=0.;

	if (k > 0) {
		f[id].x = cos(fi[id]) * (double)(N * N) * exp(-(double)(k * k) / (2. * (double)(k0 * k0))) / sqrt((double)k);
		f[id].y = sin(fi[id]) * (double)(N * N) * exp(-(double)(k * k) / (2. * (double)(k0 * k0))) / sqrt((double)k);
	}
	else {
		f[id].x = 0.;
		f[id].y = 0.;

	}


}


__global__
void Laplas(cufftDoubleComplex* f, cufftDoubleComplex* g)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int id = (N / 2 + 1) * idx + idy;

	if (idx > N / 2) idx = idx - N;

	double r = (double)(idx * idx + idy * idy);

	g[id].x = -f[id].x * r;
	g[id].y = -f[id].y * r;

}

__global__
void MLaplas(cufftDoubleComplex* f, cufftDoubleComplex* g)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int id = (N / 2 + 1) * idx + idy;

	if (idx > N / 2) idx = idx - N;

	double r = (double)(idx * idx + idy * idy);

	g[id].x = f[id].x * r;
	g[id].y = f[id].y * r;

}

__global__
void BMLaplas(cufftDoubleComplex* f, cufftDoubleComplex* g)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int id = (N / 2 + 1) * idx + idy;

	if (idx > N / 2) idx = idx - N;

	double r = (double)(idx * idx + idy * idy);

	if (id == 0) {
		g[id].x = 0.;
		g[id].y = 0.;
	}
	else {
		g[id].x = f[id].x / r;
		g[id].y = f[id].y / r;
	}

}



__global__
void Ddx(cufftDoubleComplex* f, cufftDoubleComplex* g)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int id = (N / 2 + 1) * idx + idy;

	if (idx > N / 2) idx = idx - N;

	g[id].x = -(double)idx * f[id].y;
	g[id].y = (double)idx * f[id].x;

}

__global__
void Ddy(cufftDoubleComplex* f, cufftDoubleComplex* g)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int id = (N / 2 + 1) * idx + idy;

	g[id].x = -(double)idy * f[id].y;
	g[id].y = (double)idy * f[id].x;

}

__global__
void Multiplication(double* f, double* g)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int id = N * idx + idy;

	g[id] = f[id] / ((double)(N * N));

}



__global__
void Multiplication2(cufftDoubleComplex* f, cufftDoubleComplex* g, double q)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int id = (N / 2 + 1) * idx + idy;

	g[id].x = q * f[id].x;
	g[id].y = q * f[id].y;
}

__global__
void JackobKer(double* fx, double* fy, double* gx, double* gy, double* Jack)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int id = N * idx + idy;

	Jack[id] = (fx[id] * gy[id] - fy[id] * gx[id]) / ((double)N * N * N * N); // Jack = S_x * w_y - S_y * w_x

}


__global__
void JackobDxDy(cufftDoubleComplex* f, cufftDoubleComplex* g, cufftDoubleComplex* fx, cufftDoubleComplex* fy, cufftDoubleComplex* gx, cufftDoubleComplex* gy)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int id = (N / 2 + 1) * idx + idy;

	if (idx > N / 2) idx = idx - N;

	if ((abs(idx) < XYmax) && (abs(idy) < XYmax)) {
		fx[id].x = -(double)idx * f[id].y;   fx[id].y = (double)idx * f[id].x;
		fy[id].x = -(double)idy * f[id].y;   fy[id].y = (double)idy * f[id].x;
		gx[id].x = -(double)idx * g[id].y;   gx[id].y = (double)idx * g[id].x;
		gy[id].x = -(double)idy * g[id].y;   gy[id].y = (double)idy * g[id].x;
	}
	else {
		fx[id].x = 0.;		fy[id].x = 0.;		gx[id].x = 0.;		gy[id].x = 0.;
		fx[id].y = 0.;		fy[id].y = 0.;		gx[id].y = 0.;		gy[id].y = 0.;
	}

	if ((blockIdx.y == gridDim.y - 1) && (threadIdx.y == blockDim.y - 1)) {
		id = id + 1;
		fx[id].x = 0.;		fy[id].x = 0.;		gx[id].x = 0.;		gy[id].x = 0.;
		fx[id].y = 0.;		fy[id].y = 0.;		gx[id].y = 0.;		gy[id].y = 0.;
	}

}

__global__
void JackobDx(cufftDoubleComplex* f, cufftDoubleComplex* fx)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int id = (N / 2 + 1) * idx + idy;

	if (idx > N / 2) idx = idx - N;

	if ((abs(idx) < XYmax) && (abs(idy) < XYmax)) {
		fx[id].x = -(double)idx * f[id].y;   fx[id].y = (double)idx * f[id].x;
	}
	else {
		fx[id].x = 0.;		fx[id].y = 0.;
	}

	if ((blockIdx.y == gridDim.y - 1) && (threadIdx.y == blockDim.y - 1)) {
		id = id + 1;
		fx[id].x = 0.;		fx[id].y = 0.;
	}

}

__global__
void JackobDy(cufftDoubleComplex* f, cufftDoubleComplex* fy)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int id = (N / 2 + 1) * idx + idy;

	if (idx > N / 2) idx = idx - N;

	if ((abs(idx) < XYmax) && (abs(idy) < XYmax)) {
		fy[id].x = -(double)idy * f[id].y;   fy[id].y = (double)idy * f[id].x;
	}
	else {
		fy[id].x = 0.;		fy[id].y = 0.;
	}

	if ((blockIdx.y == gridDim.y - 1) && (threadIdx.y == blockDim.y - 1)) {
		id = id + 1;
		fy[id].x = 0.;		fy[id].y = 0.;
	}

}



__global__
void JackobDeal(cufftDoubleComplex* f)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int id = (N / 2 + 1) * idx + idy;

	if (idx > N / 2) idx = idx - N;

	if ((abs(idx) >= XYmax) || (abs(idy) >= XYmax)) {
		//	if (idx*idx+idy*idy>XYmax*XYmax) {

		f[id].x = 0.;
		f[id].y = 0.;

	}

	if ((blockIdx.y == gridDim.y - 1) && (threadIdx.y == blockDim.y - 1)) {
		id = id + 1;
		f[id].x = 0.;
		f[id].y = 0.;
	}
}


__global__
void JackobKer1(double* f1, double* f2, double* Jack)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int id = N * idx + idy;

	Jack[id] = (f1[id] * f2[id]) / ((double)N * N * N * N);

}

__global__
void JackobKer2(double* f1, double* f2, double* Jack)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int id = N * idx + idy;

	Jack[id] = Jack[id] - (f1[id] * f2[id]) / ((double)N * N * N * N);

}

void Jackob(cufftDoubleComplex* f1, cufftDoubleComplex* f2, cufftDoubleComplex* Jack) { //f1=S f2=w

	// считаем производные S_x,S_y,w_x,w_y
	JackobDx << <dimGrid, dimBlock >> > (f1, Jack); 		cufftExecZ2D(bplan, Jack, a_d);
	JackobDy << <dimGrid, dimBlock >> > (f2, Jack);		cufftExecZ2D(bplan, Jack, b_d);

	JackobKer1 << <dimGrid2, dimBlock >> > (a_d, b_d, c_d);

	JackobDy << <dimGrid, dimBlock >> > (f1, Jack);		cufftExecZ2D(bplan, Jack, a_d);
	JackobDx << <dimGrid, dimBlock >> > (f2, Jack);		cufftExecZ2D(bplan, Jack, b_d);


	JackobKer2 << <dimGrid2, dimBlock >> > (a_d, b_d, c_d);

	//делаем прямое преобразование фурье
	cufftExecD2Z(plan, c_d, Jack); // c = `(S_x * w_y - S_y * w_x) = `J1

	//делаем деалиасинг
	JackobDeal << <dimGrid, dimBlock >> > (Jack);
}


__global__
void Tshem(cufftDoubleComplex* R, cufftDoubleComplex* Fold, cufftDoubleComplex* Fnew, double dt)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int id = (N / 2 + 1) * idx + idy;


	Fnew[id].x = Fold[id].x + R[id].x * dt;
	Fnew[id].y = Fold[id].y + R[id].y * dt;

}

__global__
void TshemVes(cufftDoubleComplex* R, cufftDoubleComplex* Fold, cufftDoubleComplex* Fnew, double dt, double Ves)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int id = (N / 2 + 1) * idx + idy;


	Fnew[id].x = Fold[id].x + Ves * R[id].x * dt;
	Fnew[id].y = Fold[id].y + Ves * R[id].y * dt;

}


__global__
void TshemF(cufftDoubleComplex* R, cufftDoubleComplex* Fold, cufftDoubleComplex* Fnew, int id1, int id2, double betta, double dt, double G1)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int id = (N / 2 + 1) * idx + idy;

	Fnew[id].x = Fold[id].x + R[id].x * dt;
	Fnew[id].y = Fold[id].y + R[id].y * dt;


	if ((idx == id1) && (idy == id2)) {

		Fnew[id].x += 0.5 * (double)(N * N) * sin(betta) * G1 * sqrt(dt);
		Fnew[id].y -= 0.5 * (double)(N * N) * cos(betta) * G1 * sqrt(dt);

		if (idy == 0) {
			Fnew[(N / 2 + 1) * (N - idx)].x += 0.5 * (double)(N * N) * sin(betta) * G1 * sqrt(dt);
			Fnew[(N / 2 + 1) * (N - idx)].y -= 0.5 * (double)(N * N) * cos(betta) * G1 * sqrt(dt);

		}

	}

}


__global__
void TshemFA(cufftDoubleComplex* R, cufftDoubleComplex* Fold, cufftDoubleComplex* Fnew, int id1, int id2, double betta, double dt, double G1)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int id = (N / 2 + 1) * idx + idy;

	Fnew[id].x = Fold[id].x + R[id].x * dt;
	Fnew[id].y = Fold[id].y + R[id].y * dt;


	if (id == (N / 2 + 1) * id1 + id2) {

		int idx = id1;
		if (idx > N / 2) idx = idx - N;

		double r = (double)(idx * idx + id2 * id2);

		Fnew[id].x += 0.5 * (double)(N * N) * sin(betta) * G1 * sqrt(dt) / r;
		Fnew[id].y -= 0.5 * (double)(N * N) * cos(betta) * G1 * sqrt(dt) / r;


		if (idy == 0) {
			Fnew[(N / 2 + 1) * (N - idx)].x += 0.5 * (double)(N * N) * sin(betta) * G1 * sqrt(dt) / r;
			Fnew[(N / 2 + 1) * (N - idx)].y -= 0.5 * (double)(N * N) * cos(betta) * G1 * sqrt(dt) / r;

		}
	}
}



__global__
void FMax(double* d, double* dout)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;


	__shared__ double a[K];

	a[threadIdx.x] = fabs(d[idx]);


	__syncthreads();

	for (int M = 2; M < K + 1; M = M * 2) {

		if (idx % M == 0) if (a[threadIdx.x + M / 2] > a[threadIdx.x]) a[threadIdx.x] = a[threadIdx.x + M / 2];

		__syncthreads();

	}

	dout[blockIdx.x] = a[0];
}



double max(cufftDoubleComplex* f) {


	Ddx << <dimGrid, dimBlock >> > (f, Jack1); //a=Vy
	cufftExecZ2D(bplan, Jack1, a_d);
	FMax << <N * N / K, K >> > (a_d, b_d);
	cudaMemcpy(ux, b_d, (N * N / K) * sizeof(double), cudaMemcpyDeviceToHost);

	Ddy << <dimGrid, dimBlock >> > (f, Jack1); //b=Vx
	cufftExecZ2D(bplan, Jack1, a_d);
	FMax << <N * N / K, K >> > (a_d, b_d);
	cudaMemcpy(uy, b_d, (N * N / K) * sizeof(double), cudaMemcpyDeviceToHost);

	double Vmax = 0.;
	for (int i = 0; i < N * N / K; i++) {

		if (Vmax < fabs(ux[i])) Vmax = fabs(ux[i]);
		if (Vmax < fabs(uy[i])) Vmax = fabs(uy[i]);

	}


	return Vmax / ((double)(N * N));;

}



__global__
void EnergyKerOne(double* f)
{

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int id = N * idx + idy;

	f[id] = (f[id] / ((double)(N * N))) * (f[id] / ((double)(N * N))) / 2.;

}

__global__
void EnergyKerTwo(double* d, double* dout)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;


	__shared__ double a[K];

	a[threadIdx.x] = d[idx];


	__syncthreads();

	for (int M = 2; M < K + 1; M = M * 2) {

		if (idx % M == 0) a[threadIdx.x] = a[threadIdx.x] + a[threadIdx.x + M / 2];

		__syncthreads();

	}

	dout[blockIdx.x] = a[0];
}



double energy(cufftDoubleComplex* f) {


	Ddx << <dimGrid, dimBlock >> > (f, Jack1); //a=Vy
	cufftExecZ2D(bplan, Jack1, a_d);
	EnergyKerOne << <dimGrid2, dimBlock >> > (a_d);
	EnergyKerTwo << <N * N / K, K >> > (a_d, b_d);
	cudaMemcpy(ux, b_d, (N * N / K) * sizeof(double), cudaMemcpyDeviceToHost);


	Ddy << <dimGrid, dimBlock >> > (f, Jack1); //b=Vx
	cufftExecZ2D(bplan, Jack1, a_d);
	EnergyKerOne << <dimGrid2, dimBlock >> > (a_d);
	EnergyKerTwo << <N * N / K, K >> > (a_d, b_d);
	cudaMemcpy(uy, b_d, (N * N / K) * sizeof(double), cudaMemcpyDeviceToHost);

	double E = 0.;

	for (int i = 0; i < N * N / K; i++)	E = E + ux[i] + uy[i];

	return E * lamda * (4. * M_PI * M_PI);

}


void PRINT(cufftDoubleComplex* f, int iW, char type) {

	cudaMemcpy(Jack1, f, csize, cudaMemcpyDeviceToDevice);
	cufftExecZ2D(bplan, Jack1, a_d);
	Multiplication << <dimGrid2, dimBlock >> > (a_d, b_d);
	cudaMemcpy(outH, b_d, dsize, cudaMemcpyDeviceToHost);
	std::ostringstream num;	num << iW;

	if (type == 'k') 
		FoutW.open((str + "outW/out" + num.str()).c_str(), std::ios::binary | std::ios::out);
	if (type == 'm') 
		FoutW.open((str + "outJ/out" + num.str()).c_str(), std::ios::binary | std::ios::out);

	FoutW.write((char*)outH, dsize);

	FoutW.close();

}
