#ifndef CUPDLP_CUDA_KERNALS_H
#define CUPDLP_CUDA_KERNALS_H

#include "cuda_runtime.h"
#define CUPDLP_BLOCK_SIZE 512

#ifndef SFLOAT
#ifdef DLONG
typedef long long cupdlp_int;
#else
typedef int cupdlp_int;
#endif
typedef double cupdlp_float;
#define CudaComputeType CUDA_R_64F
#else
#define CudaComputeType CUDA_R_32F
#endif

#define CHECK_CUDA(func)                                               \
  {                                                                    \
    cudaError_t status = (func);                                       \
    if (status != cudaSuccess) {                                       \
      printf("CUDA API failed at line %d of %s with error: %s (%d)\n", \
             __LINE__, __FILE__, cudaGetErrorString(status), status);  \
      return EXIT_FAILURE;                                             \
    }                                                                  \
  }

#define CHECK_CUSPARSE(func)                                               \
  {                                                                        \
    cusparseStatus_t status = (func);                                      \
    if (status != CUSPARSE_STATUS_SUCCESS) {                               \
      printf("CUSPARSE API failed at line %d of %s with error: %s (%d)\n", \
             __LINE__, __FILE__, cusparseGetErrorString(status), status);  \
      return EXIT_FAILURE;                                                 \
    }                                                                      \
  }

#define CHECK_CUBLAS(func)                                               \
  {                                                                      \
    cublasStatus_t status = (func);                                      \
    if (status != CUBLAS_STATUS_SUCCESS) {                               \
      printf("CUBLAS API failed at line %d of %s with error: %s (%d)\n", \
             __LINE__, __FILE__, cublasGetStatusString(status), status); \
      return EXIT_FAILURE;                                               \
    }                                                                    \
  }

dim3 cuda_gridsize(cupdlp_int n);

__global__ void element_wise_dot_kernel(cupdlp_float *x, const cupdlp_float *y,
                                        const cupdlp_int len);

__global__ void element_wise_div_kernel(cupdlp_float *x, const cupdlp_float *y,
                                        const cupdlp_int len);

__global__ void element_wise_projlb_kernel(cupdlp_float *x,
                                           const cupdlp_float *lb,
                                           const cupdlp_int len);

__global__ void element_wise_projub_kernel(cupdlp_float *x,
                                           const cupdlp_float *ub,
                                           const cupdlp_int len);

__global__ void element_wise_projSamelb_kernel(cupdlp_float *x,
                                               const cupdlp_float lb,
                                               const cupdlp_int len);

__global__ void element_wise_projSameub_kernel(cupdlp_float *x,
                                               const cupdlp_float ub,
                                               const cupdlp_int len);

__global__ void element_wise_initHaslb_kernal(cupdlp_float *haslb,
                                              const cupdlp_float *lb,
                                              const cupdlp_float bound,
                                              const cupdlp_int len);

__global__ void element_wise_initHasub_kernal(cupdlp_float *hasub,
                                              const cupdlp_float *ub,
                                              const cupdlp_float bound,
                                              const cupdlp_int len);

__global__ void element_wise_filterlb_kernal(cupdlp_float *x,
                                             const cupdlp_float *lb,
                                             const cupdlp_float bound,
                                             const cupdlp_int len);

__global__ void element_wise_filterub_kernal(cupdlp_float *x,
                                             const cupdlp_float *ub,
                                             const cupdlp_float bound,
                                             const cupdlp_int len);

__global__ void init_cuda_vec_kernal(cupdlp_float *x, const cupdlp_float val,
                                     const cupdlp_int len);

__global__ void primal_grad_step_kernal(cupdlp_float *xUpdate,
                                        const cupdlp_float *x,
                                        const cupdlp_float *cost,
                                        const cupdlp_float *ATy,
                                        const cupdlp_float dPrimalStep,
                                        const cupdlp_int len);

__global__ void dual_grad_step_kernal(
    cupdlp_float *yUpdate, const cupdlp_float *y, const cupdlp_float *b,
    const cupdlp_float *Ax, const cupdlp_float *AxUpdate,
    const cupdlp_float dDualStep, const cupdlp_int len);

__global__ void naive_sub_kernal(cupdlp_float *z, const cupdlp_float *x,
                                  const cupdlp_float *y, const cupdlp_int len);
#endif