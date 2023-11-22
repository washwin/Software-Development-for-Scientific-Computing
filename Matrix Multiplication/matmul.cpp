//CS601: example code to show how matrix multiplication code
//shows orders of magnitude of performance difference when
//using -O3, loop interchange, and parallelization optimizations.

#include<iostream>
#include<cstdlib>
#include<ctime>
#include<chrono>
#include<immintrin.h>
#include<cblas.h>
using namespace std

#ifdef STACKALLOCATED
#define INPUTSIZE 256
#endif
int main(int argc, char* argv[]){
    int n;
#ifdef STACKALLOCATED
    n = INPUTSIZE;
    float A[INPUTSIZE][INPUTSIZE], B[INPUTSIZE][INPUTSIZE], C[INPUTSIZE][INPUTSIZE];
#else
    n=atoi(argv[1]);
    float* A=new float[n*n];
    float* B=new float[n*n];
    float* C=new float[n*n];
    //matrix that is used for verification;
    float *mat_ver=new float[n*n];
#endif

    std::srand(std::time(NULL));
#ifdef STACKALLOCATED
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            A[i][j] = std::rand() / (float)(RAND_MAX);
            B[i][j] = std::rand() / (float)(RAND_MAX);
            C[i][j]=0;
        }  
    }

#else    

    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            A[i*n+j] = std::rand() / (float)(RAND_MAX);
            B[i*n+j] = std::rand() / (float)(RAND_MAX);
            C[i*n+j]=0;
            //intializing verification matrix
            mat_ver[i*n+j]=0;
        }  
    }
#endif

    const auto start=chrono::steady_clock::now();

#ifdef STACKALLOCATED
for(int i=0;i<n;i++)
        for(int k=0;k<n;k++)
            for(int j=0;j<n;j++)
                C[i][j] = C[i][j] + A[i][k] * B[k][j];
//#endif

//1a.There are 6 possible loop orderings
#elif LOOPijk
for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
            for(int k=0;k<n;k++)
                C[i*n+j] = C[i*n+j] + A[i*n+k] * B[k*n+j];
#elif LOOPikj
for(int i=0;i<n;i++)
        for(int k=0;k<n;k++)
            for(int j=0;j<n;j++)
                C[i*n+j] = C[i*n+j] + A[i*n+k] * B[k*n+j];
#elif LOOPjik
for(int j=0;j<n;j++)
        for(int i=0;i<n;i++)
            for(int k=0;k<n;k++)
                C[i*n+j] = C[i*n+j] + A[i*n+k] * B[k*n+j];
#elif LOOPjki
for(int j=0;j<n;j++)
        for(int k=0;k<n;k++)
            for(int i=0;i<n;i++)
                C[i*n+j] = C[i*n+j] + A[i*n+k] * B[k*n+j];
#elif LOOPkij
for(int k=0;k<n;k++)
        for(int i=0;i<n;i++)
            for(int j=0;j<n;j++)
                C[i*n+j] = C[i*n+j] + A[i*n+k] * B[k*n+j];
#elif LOOPkji
for(int k=0;k<n;k++)
        for(int j=0;j<n;j++)
            for(int i=0;i<n;i++)
                C[i*n+j] = C[i*n+j] + A[i*n+k] * B[k*n+j];

// 1c.Used inbuilt function for dot product to implement matmul
#elif SDOT

for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
        C[i * n + j] = cblas_sdot(n,A+i*n,1,B+j,n);
    }
}
// 1c.Used sgemm function implement matmul
#elif SGEMM

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, A, n, B, n, 0.0, C, n);

#else
#ifdef PARALLEL
#pragma omp parallel for
#endif
    for(int i=0;i<n;i++)
    #ifdef LOOPINTERCHANGE
        for(int k=0;k<n;k++)
            for(int j=0;j<n;j++)
    #else
        for(int j=0;j<n;j++)
            for(int k=0;k<n;k++)
    #endif
                C[i*n+j] = C[i*n+j] + A[i*n+k] * B[k*n+j];
#endif
    const auto end=chrono::steady_clock::now();
    const chrono::duration<float> elapsedtime = end-start;

    //calculated verification matrix
    //cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, A, n, B, n, 0.0, mat_ver, n);
    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
            for(int k=0;k<n;k++)
                mat_ver[i*n+j] = mat_ver[i*n+j] + A[i*n+k] * B[k*n+j];
    //calculated the error
    float e_sum=0;
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            e_sum+=abs(C[i*n+j]-mat_ver[i*n+j]);
        }
    }
    if(e_sum < 1E+4;)
        cout<<"Result matched";
    else
        cout<<"Result mismatched";
    cout<<"elapsed seconds:"<<elapsedtime.count()<<endl;
}
