//########################################################################
//########################################################################
//########################################################################
//########################################################################
//----------------------SPATIAL APD-CA MODEL------------------------------
//------------------------------------------------------------------------
//N-Dimensional set of APD nodes; Each apd node linked to N-Dimensional Ca
//cycling system. APD Dynamics are 'slaved' to Ca Dynamics which themselves
//are coupled to the APD dynamics.  
// APD{X,Y,Z}-->CaC{x,y,z}
//########################################################################
//########################################################################
//########################################################################
//########################################################################
//########################################################################


#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <random>

#define INDEPENDENT_PACING
//#define VARY_PACING
//#define CONTROL

void _1D_ND(int,int,int,int);
__device__ int _ND_1D(int,int,int,int,int,int);
__device__ int cp(int,int,bool);
__global__ void _1Dstencil(double* d_in,double* d_out,int nx,int ny,int nz,int X_,int Y_,int Z_){
	int i = threadIdx.x+ blockIdx.x* blockDim.x ;       
	int i_ = (i/(Z_*Y_))%X_ , 
	    j_ = (i/Z_)%Y_, 
		k_ = i%Z_ ;
	double sum = 0;
	//stencil around i  
	for(int nnx =   2*nx+1; nnx> 0; nnx--){//X		
		int start = i_-(nnx-nx-1);
		int ii = cp(start,X_,false);	    
		for(int nny = 2*ny+1; nny> 0; nny--){//Y
			start = j_-(nny-ny-1);
			int jj = cp(start,Y_,false);
			for(int nnz = 2*nz+1; nnz> 0; nnz--){//Z
					start = k_-(nnz-nz-1);
					int kk = cp(start,Z_,false);
					sum += *(d_in + _ND_1D(ii, jj, kk, X_, Y_, Z_));
			}
		}
	}
	d_out[i] = sum/((2*nx+1)*(2*ny+1)*(2*nz+1));
}

//CHILD LATTICE
#define xl 1
#define yl 64
#define zl 64
//PARENT LATTICE
#define xL 1
#define yL 1
#define zL 960
//CUDA THREAD BLOCK PARAMETERS
#define N (64*64)// number of threads can be as big as 1024
#define THREADS_PER_BLOCK 512// number of threads should be factors of 32. 
#define M N/THREADS_PER_BLOCK //Max # of blocks in single launch=65535

int main(){
	// SYSTEM{IX,IY,IZ}(ix,iy,iz)	
	int IX_=xL,IY_=yL,IZ_=zL,   X_=xl,Y_=yl,Z_=zl;
	const int Itotal = IX_*IY_*IZ_ ;
	const int total = X_*Y_*Z_;
	const int TOTAL = total*Itotal + Itotal; 
	std::cout<<"TOTAL: "<<TOTAL<<"\n"; 
	std::vector<double> v(TOTAL);
	std::vector<double>::iterator itr;
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> noise(0,1);
	std::uniform_real_distribution<> randomIC(0.3,0.9);	    
	std::ofstream fp("pos.dat");
	std::ofstream fl("lpos.dat");
	std::ofstream ff("apos.dat");
	std::ofstream fT("timeVar.dat");
	//Number of Iterations and Record boolean
	int totalIterates = 3;
	int sampleSize = 1;
	int recordCaWhen = totalIterates - sampleSize;
	int recordApdWhen = totalIterates - sampleSize;
	bool recordCa = true;
	bool recordApd = true;
	//diffusion radii 
	int nx= 0, ny = 1, nz = 1;
	int nxx = 0, nyy = 0, nzz = 22;
	//cccccccccccccccccccccccccccccccccuda
	int size = (X_*Y_*Z_)*sizeof(double);   //device memory
	int sizeP = (IX_*IY_*IZ_)*sizeof(double);   //device memory
	double * d_in;							//device memory
	double * d_out;							//device memory
	double * d_inP;							//device memory
	double * d_outP;					    //device memory
	cudaMalloc((void**) &d_in,size);		//allocate space in device
	cudaMalloc((void**) &d_out,size);		//allocate space in device
	cudaMalloc((void**) &d_inP,sizeP);		//allocate space in device 
	cudaMalloc((void**) &d_outP,sizeP);		//allocate space in device 
	std::vector<double>::iterator first;    
	std::vector<double>::iterator last;  
	double* host_vector; host_vector = (double*)malloc(size);
	double* host_vectorP; host_vectorP = (double*)malloc(sizeP);
	//cccccccccccccccccccccccccccccccccuda
//_____________________________________________________________________
//_____________________________________________________________________
//_____________________________________________________________________    

	clock_t st,et;
	double elapse=0;
	st = clock();
	#ifdef INDEPENDENT_PACING    
	fl<<sampleSize<<' '<<Itotal*Y_<<"\n\n\n";
	ff<<sampleSize<<' '<<IZ_<<"\n\n\n";
	std::cout<<"INDEPENDENT PACING\n";
	for(double T = 400; T <= 600 ; T+=10){
	std::cout<<'['<<T<<"]\n";
	#endif
		//INITIAL CONDITIONS	 
		for(itr = v.begin(); itr != v.begin()+Itotal; itr++) *itr = 200;	 
		for(itr = v.begin()+Itotal; itr != v.end(); itr++) *itr = randomIC(gen);
	#ifdef VARY_PACING
	std::cout<<"TUNE PACING\n";
	for(double T = 600 ; T >= 487.5; T += -abs(600-487.5) ){
		(T==600)?totalIterates=100:totalIterates=100;
		recordCaWhen = 0;
		recordApdWhen = 0;
		#endif
		#ifdef CONTROL
		double Tn = T;
		#endif
		//PACE n BEATS_____________________________
		for(int tT = 1; tT <= totalIterates; tT++){ 
			#ifdef CONTROL
			//SET APD CONTROL @ SITE i = 0
			double apd_prev = v.at(0);
			#endif
			//APD block{IX_,IY_,IZ_}
			for(int I = 0 ; I < Itotal ; I++){
				//1D to n-D coordinates (n=2)
				int I_ = (I/(IZ_*IY_))%IX_;
				int J_ = (I/IZ_)%IY_;
				int K_ = I%IZ_; 	
				int OFFSET0 = Itotal + I*total;
				int OFFSET1 = OFFSET0 + total;
				//CRU BLOCK (X_,Y_,Z_)
				#ifdef CONTROL		 
				double DI=(Tn-v.at(I));
				#else
				double DI=(T-v.at(I));
				#endif
				DI=(DI>0)*DI;
				int kk = 0;
				//////// CALCIUM RELEASE ALL (X_,Y_Z_)
				double sumRelease = 0;
				for(itr = v.begin()+OFFSET0; itr != v.begin()+OFFSET1; ++itr){
					double Ps = 1/(1+pow(0.7/(*itr),40));
					double Pc = 1/(1+2*exp(-DI/100));
					double P = Pc*Ps;	
					bool spark= (P>noise(gen));
					double releaseAmnt = spark*0.7*(*itr);
					*itr -= releaseAmnt;
					sumRelease+=releaseAmnt;
				}
				// APD-Ca COUPLING
				v.at(I) = 200./(1+2*exp(-DI/100.)) + 60*(sumRelease/total);
				//////// CA diffusion using CUDA kernel function
				first = v.begin() + OFFSET0;	
				last = v.begin() + OFFSET1;	
				std::copy(first,last,host_vector);
				cudaMemcpy(d_in,host_vector,size,cudaMemcpyHostToDevice);
				_1Dstencil<<<M,THREADS_PER_BLOCK>>>(d_in,d_out,nx,ny,nz,X_,Y_,Z_);
				cudaMemcpy(host_vector,d_out,size,cudaMemcpyDeviceToHost);
				int count = 0;
				while(first!=last){ *(first++) = host_vector[count++];}
				//////// CA UPTAKE ALL (X_,Y_,Z_)
				int k=OFFSET0;
				double sum = 0;   
				for(itr = v.begin()+OFFSET0; itr != v.begin()+OFFSET1; itr++){
					#ifdef CONTROL
					*itr = v.at(k) + (v.at(k)<=1)*(Tn/1000.)*(1-v.at(k));
					#else
					*itr = v.at(k) + (v.at(k)<=1)*(T/1000.)*(1-v.at(k));
					#endif
					sum+=*itr;
				k++;
				}
				////////SAVE CALCIUM DYNAMICS 
				if((recordCaWhen<tT)&&recordCa){
					double avg = sum/total;
					for(int i = OFFSET0 ; i < OFFSET1; i++){
						int i_ = ((i-Itotal)/(Z_*Y_))%X_;
						int j_ = ((i-Itotal)/Z_)%Y_;
						int k_ = (i-Itotal)%Z_;
						fp<<i_<<' '<<j_<<' '<<k_<<' '<<v.at(i)<<' '<<T<<' '<<avg<<"\n";
						(j_==25) ? fl<< v.at(i) <<' '<<'\n' : fl;
					}
					fp<<"\n\n";
				}
			}
			if((recordCaWhen<tT)&&recordCa) {fl<<"\n\n";}
			//APD DIFFUSION using CUDA
			first = v.begin() ;		
			last = v.begin() + Itotal;		
			std::copy(first,last,host_vectorP);
			cudaMemcpy(d_inP,host_vectorP,sizeP,cudaMemcpyHostToDevice);
			_1Dstencil<<<Itotal/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(d_inP,d_outP,nxx,nyy,nzz,IX_,IY_,IZ_);
			cudaMemcpy(host_vectorP,d_outP,sizeP,cudaMemcpyDeviceToHost);
			int count = 0;
			while(first!=last){ *(first++) = host_vectorP[count++];}
			#ifdef CONTROL
			if(tT>100){
				double DT = (0.2/2)*(v.at(0) - apd_prev);
				Tn += DT*(DT<0);}
			#endif
			//SAVE APD STATE------------
			if((recordApdWhen<tT)&&recordApd){
				//APD AVERAGE
				int kkk=0;
				double ssum=0;
				for(itr = v.begin(); itr != v.begin()+Itotal; itr++){
					*itr = v.at(kkk++);
					ssum+=*itr;
				}
				double aavg = ssum/Itotal;
				for(int i = 0 ; i < Itotal; i++){
					int i_ = (i/(IZ_*IY_))%IX_;
					int j_ = (i/IZ_)%IY_;
					int k_ = (i)%IZ_;
					ff<<i_<<' '<<j_<<' '<<k_<<' '<<v.at(i)<<' '<<T<<' '<<aavg<<"\n";
				}
				ff<<"\n\n";
			}
		}
	}
	std::cout<<"==============\n";
	et=clock();
	elapse = ((et - st)/CLOCKS_PER_SEC);
	std::cout<<elapse<<'\n';
	//MEMORY FREEing
	free(host_vectorP); free(host_vector);
	cudaFree(d_in);cudaFree(d_out);cudaFree(d_inP);cudaFree(d_outP);
	return 0;
}




__device__ int _ND_1D(int i_x, int j_y, int k_z, int X, int Y, int Z){
	return i_x*(Z*Y) + j_y*(Z) + k_z;}

void _1D_ND(int i, int X_, int Y_, int Z_){
	std::cout<<"[" <<(i/(Z_*Y_))%X_<<"]["<<(i/Z_)%Y_<<"]["<<i%Z_<<"]"<<std::endl; }

__device__ int cp(int start,int Max,bool fluxB){
	return (start < 0) * (fluxB*Max + pow(-1,(!fluxB))*start + (-1)*(!fluxB)) +
		   (start >= Max) * ((!fluxB)*(Max-1) + pow(-1,!fluxB) * ( start - Max )) +
		   (!(start < 0)*!(start >= Max)) * start;}



//Allasian.com ... scientific software
