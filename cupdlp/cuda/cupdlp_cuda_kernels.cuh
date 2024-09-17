#ifndef CUPDLP_CUDA_KERNALS_H
#define CUPDLP_CUDA_KERNALS_H

#include <stdio.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <cuda_runtime.h>

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

static inline cudaError_t check_cuda_call(cudaError_t status,
                                          const char *filename, int line)
{
  if (status != cudaSuccess) {
    printf("CUDA API failed at line %d of %s with error: %s (%d)\n",
      line, filename, cudaGetErrorString(status), status);
  }
  return status;
}

static inline cusparseStatus_t check_cusparse_call(cusparseStatus_t status,
                                                   const char *filename, int line)
{
  if (status != CUSPARSE_STATUS_SUCCESS) {
    printf("CUSPARSE API failed at line %d of %s with error: %s (%d)\n",
      line, filename, cusparseGetErrorString(status), status);
  }
  return status;
}

static inline cublasStatus_t check_cublas_call(cublasStatus_t status,
                                               const char *filename, int line)
{
  if (status != CUBLAS_STATUS_SUCCESS) {
    printf("CUBLAS API failed at line %d of %s with error: %s (%d)\n",
      line, filename, cublasGetStatusString(status), status);
  }
  return status;
}

static inline cudaError_t check_cuda_last(const char *filename, int line)
{
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    printf("CUDA API failed at line %d of %s with error: %s (%d)\n",
      line, filename, cudaGetErrorString(status), status);
  }
  return status;
}

#define CHECK_CUDA(res) \
  { if (check_cuda_call(res, __FILE__, __LINE__) != cudaSuccess) \
      return EXIT_FAILURE; }
#define CHECK_CUDA_STRICT(res) \
  { if (check_cuda_call(res, __FILE__, __LINE__) != cudaSuccess) \
      exit(EXIT_FAILURE); }
#define CHECK_CUDA_IGNORE(res) \
  { check_cuda_call(res, __FILE__, __LINE__); }

#define CHECK_CUSPARSE(res) \
  { if (check_cusparse_call(res, __FILE__, __LINE__) != CUSPARSE_STATUS_SUCCESS) \
      return EXIT_FAILURE; }
#define CHECK_CUSPARSE_STRICT(res) \
  { if (check_cusparse_call(res, __FILE__, __LINE__) != CUSPARSE_STATUS_SUCCESS) \
      exit(EXIT_FAILURE); }
#define CHECK_CUSPARSE_IGNORE(res) \
  { check_cusparse_call(res, __FILE__, __LINE__); }

#define CHECK_CUBLAS(res) \
  { if (check_cublas_call(res, __FILE__, __LINE__) != CUBLAS_STATUS_SUCCESS) \
      return EXIT_FAILURE; }
#define CHECK_CUBLAS_STRICT(res) \
  { if (check_cublas_call(res, __FILE__, __LINE__) != CUBLAS_STATUS_SUCCESS) \
      exit(EXIT_FAILURE); }
#define CHECK_CUBLAS_IGNORE(res) \
  { check_cublas_call(res, __FILE__, __LINE__); }

#define CHECK_CUDA_LAST() check_cuda_last(__FILE__, __LINE__)


#define CUPDLP_FREE_VEC(x) \
  { check_cuda_call(cudaFree(x), __FILE__, __LINE__); x = cupdlp_NULL; }

#define CUPDLP_COPY_VEC(dst, src, type, size) \
  check_cuda_call( \
    cudaMemcpy(dst, src, sizeof(type) * (size), cudaMemcpyDefault), \
    __FILE__, __LINE__)

#define CUPDLP_ZERO_VEC(var, type, size) \
  check_cuda_call( \
    cudaMemset(var, 0, sizeof(type) * (size)), __FILE__, __LINE__)

#define CUPDLP_INIT_VEC(var, size)                                             \
  {                                                                            \
    cudaError_t status = cudaMalloc((void **)&var, (size) * sizeof(__typeof__(*var))); \
    check_cuda_call(status, __FILE__, __LINE__);                               \
    if (status != cudaSuccess) goto exit_cleanup;                              \
  }

#define CUPDLP_INIT_ZERO_VEC(var, size)                                         \
  {                                                                            \
    cudaError_t status = cudaMalloc((void **)&var, (size) * sizeof(__typeof__(*var))); \
    check_cuda_call(status, __FILE__, __LINE__);                               \
    if (status != cudaSuccess) goto exit_cleanup;                              \
    status = cudaMemset(var, 0, (size) * sizeof(__typeof__(*var)));            \
    if (status != cudaSuccess) goto exit_cleanup;                              \
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

__global__ void element_wise_initHaslb_kernel(cupdlp_float *haslb,
                                              const cupdlp_float *lb,
                                              const cupdlp_float bound,
                                              const cupdlp_int len);

__global__ void element_wise_initHasub_kernel(cupdlp_float *hasub,
                                              const cupdlp_float *ub,
                                              const cupdlp_float bound,
                                              const cupdlp_int len);

__global__ void element_wise_filterlb_kernel(cupdlp_float *x,
                                             const cupdlp_float *lb,
                                             const cupdlp_float bound,
                                             const cupdlp_int len);

__global__ void element_wise_filterub_kernel(cupdlp_float *x,
                                             const cupdlp_float *ub,
                                             const cupdlp_float bound,
                                             const cupdlp_int len);

__global__ void init_cuda_vec_kernel(cupdlp_float *x, const cupdlp_float val,
                                     const cupdlp_int len);

__global__ void primal_grad_step_kernel(cupdlp_float *xUpdate,
                                        const cupdlp_float *x,
                                        const cupdlp_float *cost,
                                        const cupdlp_float *ATy,
                                        const cupdlp_float dPrimalStep,
                                        const cupdlp_int len);

__global__ void dual_grad_step_kernel(
    cupdlp_float *yUpdate, const cupdlp_float *y, const cupdlp_float *b,
    const cupdlp_float *Ax, const cupdlp_float *AxUpdate,
    const cupdlp_float dDualStep, const cupdlp_int len);

__global__ void naive_sub_kernel(cupdlp_float *z, const cupdlp_float *x,
                                 const cupdlp_float *y, const cupdlp_int len);
#endif