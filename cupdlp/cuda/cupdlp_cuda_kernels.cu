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
    xUpdate[i] = min(max(cupdlp_fma_rn(dPrimalStep, ATy[i] - cost[i], x[i]), lb[i]), ub[i]);
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
    cupdlp_float upd = cupdlp_fma_rn(dDualStep, b[i] - 2 * AxUpdate[i] + Ax[i], y[i]);
    yUpdate[i] = i >= nEqs ? max(upd, 0.0) : upd;
  }
}

// z = x - y
__global__ void naive_sub_kernel(cupdlp_float *z, const cupdlp_float *x,
                                  const cupdlp_float *y, const cupdlp_int len) {
  cupdlp_int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) z[i] = x[i] - y[i];
}


#define QUARTER_WARP_REDUCE_2(val1, val2) { \
  val1 += __shfl_down_sync(0xFFFFFFFF, val1, 4); \
  val2 += __shfl_down_sync(0xFFFFFFFF, val2, 4); \
  val1 += __shfl_down_sync(0xFFFFFFFF, val1, 2); \
  val2 += __shfl_down_sync(0xFFFFFFFF, val2, 2); \
  val1 += __shfl_down_sync(0xFFFFFFFF, val1, 1); \
  val2 += __shfl_down_sync(0xFFFFFFFF, val2, 1); \
}

#define FULL_WARP_REDUCE_2(val1, val2) { \
  val1 += __shfl_down_sync(0xFFFFFFFF, val1, 16); \
  val2 += __shfl_down_sync(0xFFFFFFFF, val2, 16); \
  val1 += __shfl_down_sync(0xFFFFFFFF, val1, 8); \
  val2 += __shfl_down_sync(0xFFFFFFFF, val2, 8); \
  val1 += __shfl_down_sync(0xFFFFFFFF, val1, 4); \
  val2 += __shfl_down_sync(0xFFFFFFFF, val2, 4); \
  val1 += __shfl_down_sync(0xFFFFFFFF, val1, 2); \
  val2 += __shfl_down_sync(0xFFFFFFFF, val2, 2); \
  val1 += __shfl_down_sync(0xFFFFFFFF, val1, 1); \
  val2 += __shfl_down_sync(0xFFFFFFFF, val2, 1); \
}

// assumes block size = 256, warp size = 32
__global__ void movement_1_kernel(cupdlp_float * __restrict__ res_x, cupdlp_float * __restrict__ res_y,
                                  const cupdlp_float * __restrict__ xUpdate, const cupdlp_float * __restrict__ x,
                                  const cupdlp_float * __restrict__ atyUpdate, const cupdlp_float * __restrict__ aty,
                                  int nCols) {

  __shared__ cupdlp_float shared_x[32];
  __shared__ cupdlp_float shared_y[32];
  cupdlp_float val_x = 0.0;
  cupdlp_float val_y = 0.0;
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < nCols; i += blockDim.x * gridDim.x) {
      cupdlp_float dx = xUpdate[i] - x[i];
      cupdlp_float day = atyUpdate[i] - aty[i];
      val_x = cupdlp_fma_rn(dx, dx, val_x);
      val_y = cupdlp_fma_rn(day, dx, val_y);
  }

  int lane = threadIdx.x % 32;
  int wid = threadIdx.x / 32;

  FULL_WARP_REDUCE_2(val_x, val_y)
  if (lane == 0) {
    shared_x[wid] = val_x;
    shared_y[wid] = val_y;
  }
  __syncthreads();

  if (wid == 0) {
    val_x = (threadIdx.x < blockDim.x / 32) ? shared_x[lane] : 0.0;
    val_y = (threadIdx.x < blockDim.x / 32) ? shared_y[lane] : 0.0;
    QUARTER_WARP_REDUCE_2(val_x, val_y)
    if (threadIdx.x == 0) {
      res_x[blockIdx.x] = val_x;
      res_y[blockIdx.x] = val_y;
    }
  }
}

#define QUARTER_WARP_REDUCE(val) { \
  val += __shfl_down_sync(0xFFFFFFFF, val, 4); \
  val += __shfl_down_sync(0xFFFFFFFF, val, 2); \
  val += __shfl_down_sync(0xFFFFFFFF, val, 1); \
}

#define HALF_WARP_REDUCE(val) { \
  val += __shfl_down_sync(0xFFFFFFFF, val, 8); \
  val += __shfl_down_sync(0xFFFFFFFF, val, 4); \
  val += __shfl_down_sync(0xFFFFFFFF, val, 2); \
  val += __shfl_down_sync(0xFFFFFFFF, val, 1); \
}

#define FULL_WARP_REDUCE(val) { \
  val += __shfl_down_sync(0xFFFFFFFF, val, 16); \
  val += __shfl_down_sync(0xFFFFFFFF, val, 8); \
  val += __shfl_down_sync(0xFFFFFFFF, val, 4); \
  val += __shfl_down_sync(0xFFFFFFFF, val, 2); \
  val += __shfl_down_sync(0xFFFFFFFF, val, 1); \
}

// assumes: block size = 256, warp size = 32
__global__ void movement_2_kernel(cupdlp_float * __restrict__ res,
                                  const cupdlp_float * __restrict__ yUpdate, const cupdlp_float * __restrict__ y,
                                  int nRows) {

  __shared__ cupdlp_float shared[32];
  cupdlp_float val = 0.0;
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < nRows; i += blockDim.x * gridDim.x) {
      cupdlp_float d = yUpdate[i] - y[i];
      val = cupdlp_fma_rn(d, d, val);
  }

  int lane = threadIdx.x % 32;
  int wid = threadIdx.x / 32;

  FULL_WARP_REDUCE(val)
  if (lane == 0) {
    shared[wid] = val;
  }
  __syncthreads();

  if (wid == 0) {
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0;
    QUARTER_WARP_REDUCE(val)
    if (threadIdx.x == 0) {
      res[blockIdx.x] = val;
    }
  }
}

// assumes: block size = 512, warp size = 32
__global__ void sum_kernel(cupdlp_float * __restrict__ res, const cupdlp_float * __restrict__ x, int n) {

  __shared__ cupdlp_float shared[32];
  cupdlp_float val = 0.0;
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
      val += x[i];
  }

  int lane = threadIdx.x % 32;
  int wid = threadIdx.x / 32;

  FULL_WARP_REDUCE(val)
  if (lane == 0) {
    shared[wid] = val;
  }
  __syncthreads();

  if (wid == 0) {
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0;
    HALF_WARP_REDUCE(val)
    if (threadIdx.x == 0) {
      res[blockIdx.x] = val;
    }
  }
}
