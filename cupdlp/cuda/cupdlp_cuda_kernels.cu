#include "cupdlp_cuda_kernels.cuh"

dim3 cuda_gridsize(cupdlp_int n) {
  cupdlp_int k = (n - 1) / CUPDLP_BLOCK_SIZE + 1;
  cupdlp_int x = k;
  cupdlp_int y = 1;
  if (x > 65535) {
    x = ceil(sqrt(k));
    y = (n - 1) / (x * CUPDLP_BLOCK_SIZE) + 1;
  }
  dim3 d = {(unsigned int)x, (unsigned int)y, 1};
  return d;
}

__global__ void element_wise_dot_kernel(cupdlp_float *x, const cupdlp_float *y,
                                        const cupdlp_int len) {
  cupdlp_int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) x[i] *= y[i];
}

__global__ void element_wise_div_kernel(cupdlp_float *x, const cupdlp_float *y,
                                        const cupdlp_int len) {
  cupdlp_int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) x[i] /= y[i];
}

__global__ void element_wise_projlb_kernel(cupdlp_float *x,
                                           const cupdlp_float *lb,
                                           const cupdlp_int len) {
  cupdlp_int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) x[i] = x[i] < lb[i] ? lb[i] : x[i];
}

__global__ void element_wise_projub_kernel(cupdlp_float *x,
                                           const cupdlp_float *ub,
                                           const cupdlp_int len) {
  cupdlp_int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) x[i] = x[i] > ub[i] ? ub[i] : x[i];
}

__global__ void element_wise_projSamelb_kernel(cupdlp_float *x,
                                               const cupdlp_float lb,
                                               const cupdlp_int len) {
  cupdlp_int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) x[i] = x[i] <= lb ? lb : x[i];
}

__global__ void element_wise_projSameub_kernel(cupdlp_float *x,
                                               const cupdlp_float ub,
                                               const cupdlp_int len) {
  cupdlp_int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) x[i] = x[i] >= ub ? ub : x[i];
}

__global__ void element_wise_initHaslb_kernel(cupdlp_float *haslb,
                                              const cupdlp_float *lb,
                                              const cupdlp_float bound,
                                              const cupdlp_int len) {
  cupdlp_int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) haslb[i] = lb[i] > bound ? 1.0 : 0.0;
}

__global__ void element_wise_initHasub_kernel(cupdlp_float *hasub,
                                              const cupdlp_float *ub,
                                              const cupdlp_float bound,
                                              const cupdlp_int len) {
  cupdlp_int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) hasub[i] = ub[i] < bound ? 1.0 : 0.0;
}

__global__ void element_wise_filterlb_kernel(cupdlp_float *x,
                                             const cupdlp_float *lb,
                                             const cupdlp_float bound,
                                             const cupdlp_int len) {
  cupdlp_int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) x[i] = lb[i] > bound ? lb[i] : 0.0;
}

__global__ void element_wise_filterub_kernel(cupdlp_float *x,
                                             const cupdlp_float *ub,
                                             const cupdlp_float bound,
                                             const cupdlp_int len) {
  cupdlp_int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) x[i] = ub[i] < bound ? ub[i] : 0.0;
}

__global__ void init_cuda_vec_kernel(cupdlp_float *x, const cupdlp_float val,
                                     const cupdlp_int len) {
  cupdlp_int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) x[i] = val;
}

// xUpdate = proj(x - dPrimalStep * (cost - ATy))
__global__ void primal_grad_step_kernel(cupdlp_float *__restrict__ xUpdate,
                                        const cupdlp_float * __restrict__ x,
                                        const cupdlp_float * __restrict__ cost,
                                        const cupdlp_float * __restrict__ ATy,
                                        const cupdlp_float * __restrict__ lb,
                                        const cupdlp_float * __restrict__ ub,
                                        cupdlp_float dPrimalStep, int nCols) {
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < nCols; i += gridDim.x * blockDim.x) {
    xUpdate[i] = min(max(x[i] - dPrimalStep * (cost[i] - ATy[i]),
                         lb[i]),
                     ub[i]);
  }
}

// yUpdate = proj(y + dDualStep * (b - 2 AxUpdate + Ax))
__global__ void dual_grad_step_kernel(cupdlp_float * __restrict__ yUpdate,
                                      const cupdlp_float * __restrict__ y,
                                      const cupdlp_float * __restrict__ b,
                                      const cupdlp_float * __restrict__ Ax,
                                      const cupdlp_float * __restrict__ AxUpdate,
                                      cupdlp_float dDualStep, int nRows, int nEqs) {
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < nRows; i += gridDim.x * blockDim.x) {
    cupdlp_float upd = y[i] + dDualStep * (b[i] - 2 * AxUpdate[i] + Ax[i]);
    yUpdate[i] = i >= nEqs && upd < 0.0 ? 0.0 : upd;
  }
}

// z = x - y
__global__ void naive_sub_kernel(cupdlp_float *z, const cupdlp_float *x,
                                  const cupdlp_float *y, const cupdlp_int len) {
  cupdlp_int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) z[i] = x[i] - y[i];
}