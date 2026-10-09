// ΑΛΓΟΡΙΘΜΟΣ 3 : REGISTER BLOCKING / THREAD COARSENING (Shared + Registers)
//
// Κάθε block υπολογίζει μπλοκ C: BM x BN. Κάθε νήμα υπολογίζει υπο-μπλοκ
// TM x TN κρατώντας τους συσσωρευτές σε REGISTERS. Σε κάθε βήμα φορτώνει TM
// τιμές του As και TN τιμές του Bs σε registers και κάνει outer-product
// -> επαναχρησιμοποίηση και από shared σε registers, υψηλότατο arithmetic
// intensity. Πλησιάζει την απόδοση βιβλιοθήκης. Έχει έλεγχο ορίων.
//
// Default: BM=BN=64, BK=8, TM=TN=8 -> 64 νήματα/block, 64 outputs/νήμα.
//
// Μεταγλώττιση (Colab T4 = sm_75):
//     !nvcc -O3 -arch=sm_75 gemm_register.cu -o reg -lcublas
//     !./reg                    # default τετράγωνα μεγέθη (256..4096)
//     !./reg 512 1024 2048      # custom τετράγωνα (N x N)
//     !./reg -r 1024 512 2048   # ορθογώνιο A(M x K)*B(K x N): M=1024 K=512 N=2048
//
//              (RTX 3070 Ti = sm_86):
//     nvcc -O3 -arch=sm_86 gemm_register.cu -o reg  -lcublas
//     ./reg                    # default τετράγωνα μεγέθη (256..4096)
//     ./reg 512 1024 2048      # custom τετράγωνα (N x N)
//     ./reg -r 1024 512 2048   # ορθογώνιο A(M x K)*B(K x N): M=1024 K=512 N=2048


#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <functional>
#include <cuda_runtime.h>

#ifndef USE_CUBLAS
#define USE_CUBLAS 1
#endif
#if USE_CUBLAS
#include <cublas_v2.h>
#endif

#define CUDA_CHECK(call) do { cudaError_t _e = (call); if (_e != cudaSuccess) { fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); exit(EXIT_FAILURE); } } while (0)
#if USE_CUBLAS
#define CUBLAS_CHECK(call) do { cublasStatus_t _s = (call); if (_s != CUBLAS_STATUS_SUCCESS) { fprintf(stderr,"cuBLAS error %s:%d: %d\n",__FILE__,__LINE__,(int)_s); exit(EXIT_FAILURE); } } while (0)
#endif

// KERNEL
template <int BM, int BN, int BK, int TM, int TN>
__global__ void mm_register(int M, int N, int K,
                           const float *A, const float *B, float *C)
{
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    const int blockRow = blockIdx.y, blockCol = blockIdx.x;
    const int threadRow = threadIdx.y;          // 0 .. BM/TM-1
    const int threadCol = threadIdx.x;          // 0 .. BN/TN-1
    const int numThreads = (BM / TM) * (BN / TN);
    const int tid = threadRow * (BN / TN) + threadCol;

    float acc[TM * TN] = {0.0f};                // συσσωρευτές στους REGISTERS
    float regM[TM], regN[TN];

    for (int bk = 0; bk < K; bk += BK) {
        // συνεργατική φόρτωση tile A (BM x BK) στη shared
        for (int i = tid; i < BM * BK; i += numThreads) {
            int r = i / BK, c = i % BK;
            int gRow = blockRow * BM + r, gCol = bk + c;
            As[i] = (gRow < M && gCol < K) ? A[gRow * K + gCol] : 0.0f;
        }
        // συνεργατική φόρτωση tile B (BK x BN) στη shared
        for (int i = tid; i < BK * BN; i += numThreads) {
            int r = i / BN, c = i % BN;
            int gRow = bk + r, gCol = blockCol * BN + c;
            Bs[i] = (gRow < K && gCol < N) ? B[gRow * N + gCol] : 0.0f;
        }
        __syncthreads();

        // υπολογισμός: outer products με δεδομένα από registers
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                regM[i] = As[(threadRow * TM + i) * BK + k];
            #pragma unroll
            for (int j = 0; j < TN; ++j)
                regN[j] = Bs[k * BN + threadCol * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i * TN + j] += regM[i] * regN[j];
        }
        __syncthreads();
    }

    // εγγραφή με έλεγχο ορίων
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int gRow = blockRow * BM + threadRow * TM + i;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int gCol = blockCol * BN + threadCol * TN + j;
            if (gRow < M && gCol < N)
                C[gRow * N + gCol] = acc[i * TN + j];
        }
    }
}

// βοηθητικές
static void cpu_gemm(int M,int N,int K,const float*A,const float*B,float*C){
    for(int i=0;i<M;++i) for(int j=0;j<N;++j){ float s=0.f;
        for(int k=0;k<K;++k) s+=A[i*K+k]*B[k*N+j]; C[i*N+j]=s; }
}
static double max_rel_error(const float*ref,const float*t,int n){
    double m=0; for(int i=0;i<n;++i){ double r=fabs((double)ref[i]);
        double d=fabs((double)ref[i]-(double)t[i]);
        double rel=d/(r>1e-6?r:1e-6); if(rel>m)m=rel;} return m;
}
template<typename F> static float time_ms(F launch,int warmup,int iters){
    for(int i=0;i<warmup;++i) launch();
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t s,e; CUDA_CHECK(cudaEventCreate(&s)); CUDA_CHECK(cudaEventCreate(&e));
    CUDA_CHECK(cudaEventRecord(s));
    for(int i=0;i<iters;++i) launch();
    CUDA_CHECK(cudaEventRecord(e)); CUDA_CHECK(cudaEventSynchronize(e));
    float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms,s,e));
    CUDA_CHECK(cudaEventDestroy(s)); CUDA_CHECK(cudaEventDestroy(e));
    return ms/iters;
}
static inline double gflops(int M,int N,int K,float ms){
    return (2.0*M*N*K)/(ms*1e6);
}


int main(int argc,char**argv){
    // λίστα περιπτώσεων (M,K,N). τετράγωνα by default, ή ορθογώνιο με '-r M K N'.
    std::vector<int> Ms, Ks, Ns;
    if(argc>=5 && strcmp(argv[1],"-r")==0){
        Ms.push_back(atoi(argv[2])); Ks.push_back(atoi(argv[3])); Ns.push_back(atoi(argv[4]));
    } else if(argc>1){
        for(int i=1;i<argc;++i){ int s=atoi(argv[i]); Ms.push_back(s);Ks.push_back(s);Ns.push_back(s);}
    } else {
        int defs[]={256,512,1024,2048,4096};
        for(int s:defs){ Ms.push_back(s);Ks.push_back(s);Ns.push_back(s); }
    }

    int dev=0; cudaDeviceProp prop; CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&prop,dev));
    printf("=== [REGISTER-BLOCK] GPU: %s | SMs: %d | Compute %d.%d ===\n",
           prop.name,prop.multiProcessorCount,prop.major,prop.minor);

#if USE_CUBLAS
    cublasHandle_t handle; CUBLAS_CHECK(cublasCreate(&handle));
    const float alpha=1.f,beta=0.f;
#endif
    const int warmup=2,iters=5;
    const int BM=64,BN=64,BK=8,TM=8,TN=8;     // παράμετροι register blocking

    printf("BM=%d BN=%d BK=%d | TM=%d TN=%d | %d νήματα/block, %d outputs/νήμα\n",
           BM,BN,BK,TM,TN,(BM/TM)*(BN/TN),TM*TN);
    printf("%-16s | %-20s | %-9s | %-20s | %-9s | %-6s\n",
           "M x K x N","Register (ms/GFLOPS)","total ms","cuBLAS (ms/GFLOPS)","speedup","check");
    printf("-----------------|----------------------|-----------|----------------------|-----------|------\n");

    for(size_t c=0;c<Ns.size();++c){
        int M=Ms[c], K=Ks[c], N=Ns[c];
        size_t bytesA=(size_t)M*K*sizeof(float);
        size_t bytesB=(size_t)K*N*sizeof(float);
        size_t bytesC=(size_t)M*N*sizeof(float);
        float *hA=(float*)malloc(bytesA),*hB=(float*)malloc(bytesB);
        float *hRef=(float*)malloc(bytesC),*hC=(float*)malloc(bytesC);
        for(size_t i=0;i<(size_t)M*K;++i) hA[i]=(float)rand()/RAND_MAX;
        for(size_t i=0;i<(size_t)K*N;++i) hB[i]=(float)rand()/RAND_MAX;

        float *dA,*dB,*dC;
        CUDA_CHECK(cudaMalloc(&dA,bytesA)); CUDA_CHECK(cudaMalloc(&dB,bytesB));
        CUDA_CHECK(cudaMalloc(&dC,bytesC));
        CUDA_CHECK(cudaMemcpy(dA,hA,bytesA,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB,hB,bytesB,cudaMemcpyHostToDevice));

        bool haveRef=false;
#if USE_CUBLAS
        CUBLAS_CHECK(cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,
                                 N,M,K,&alpha,dB,N,dA,K,&beta,dC,N));
        CUDA_CHECK(cudaMemcpy(hRef,dC,bytesC,cudaMemcpyDeviceToHost)); haveRef=true;
#endif
        if(!haveRef && (double)M*N*K<=1.2e9){ cpu_gemm(M,N,K,hA,hB,hRef); haveRef=true; }

        // REGISTER BLOCKING
        dim3 block(BN/TN,BM/TM), grid((N+BN-1)/BN,(M+BM-1)/BM);
        float t=time_ms([&]{
            mm_register<BM,BN,BK,TM,TN><<<grid,block>>>(M,N,K,dA,dB,dC);
        },warmup,iters);
        CUDA_CHECK(cudaMemcpy(hC,dC,bytesC,cudaMemcpyDeviceToHost));
        double err = haveRef? max_rel_error(hRef,hC,(int)((size_t)M*N)) : -1;

        float tc=-1;
#if USE_CUBLAS
        tc=time_ms([&]{ cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,
                        N,M,K,&alpha,dB,N,dA,K,&beta,dC,N); },warmup,iters);
#endif
        // end-to-end: H2D(A,B) + kernel + D2H(C)
        float th2d=time_ms([&]{ cudaMemcpy(dA,hA,bytesA,cudaMemcpyHostToDevice);
                                cudaMemcpy(dB,hB,bytesB,cudaMemcpyHostToDevice); },1,3);
        float td2h=time_ms([&]{ cudaMemcpy(hC,dC,bytesC,cudaMemcpyDeviceToHost); },1,3);
        float total=th2d + t + td2h;

        char dim[40]; snprintf(dim,sizeof(dim),"%dx%dx%d",M,K,N);
        printf("%-16s |  %8.3f / %8.1f | %9.3f |  %8.3f / %8.1f | %8.2fx | %-5s\n",
               dim, t, gflops(M,N,K,t), total,
               tc, tc>0?gflops(M,N,K,tc):0.0,
               tc>0? t/tc : 0.0,
               (err>=0 && err<1e-2)?"OK":(err<0?"-":"FAIL"));

        CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
        free(hA); free(hB); free(hRef); free(hC);
    }
#if USE_CUBLAS
    CUBLAS_CHECK(cublasDestroy(handle));
#endif
    printf("\nRegister blocking: επαναχρησιμοποίηση shared->registers, μέγιστο "
           "arithmetic intensity. 'speedup' = πόσες φορές πιο αργό απ' το cuBLAS.\n");
    return 0;
}