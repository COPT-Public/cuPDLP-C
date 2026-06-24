/*
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Author: Jeff Daily <jeff.daily@amd.com>
 *
 * AMD GPU (HIP/ROCm) support for the cuPDLP-C CUDA kernels.
 */
#include "cupdlp_cuda_kernels.cuh"
#include "cuda_to_hip.h"



__global__ void element_primal_feas_kernel(cupdlp_float *z,
                                           const cupdlp_float *ax,
                                           const cupdlp_float *rhs,
                                           const cupdlp_float *rowScale,
                                           int ifScaled,
                                           int nEqs, int nRows) {
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < nRows; i += gridDim.x * blockDim.x) {
    cupdlp_float tmp = ax[i] - rhs[i];
    if (i >= nEqs) tmp = min(tmp, 0.0);
    z[i] = tmp * (ifScaled ? rowScale[i] : 1.0);
  }
}

__global__ void element_dual_feas_kernel_1(cupdlp_float *z,
                                           const cupdlp_float *aty,
                                           const cupdlp_float *cost,
                                           int nCols) {
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < nCols; i += gridDim.x * blockDim.x) {
    z[i] = cost[i] - aty[i];
  }
}

__global__ void element_dual_feas_kernel_2(cupdlp_float *z,
                                           const cupdlp_float *dualResidual,
                                           const cupdlp_float *hasLower,
                                           int nCols) {
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < nCols; i += gridDim.x * blockDim.x) {
    z[i] = max(dualResidual[i], 0.0) * hasLower[i];
  }
}

__global__ void element_dual_feas_kernel_3(cupdlp_float *z,
                                           const cupdlp_float *dualResidual,
                                           const cupdlp_float *hasUpper,
                                           int nCols) {
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < nCols; i += gridDim.x * blockDim.x) {
    z[i] = -min(dualResidual[i], 0.0) * hasUpper[i];
  }
}

__global__ void element_primal_infeas_kernel(cupdlp_float *z, const cupdlp_float *aty,
                                             const cupdlp_float *dSlackPos,
                                             const cupdlp_float *dSlackNeg,
                                             const cupdlp_float *colScale,
                                             cupdlp_float alpha, int ifScaled, int nCols) {
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < nCols; i += gridDim.x * blockDim.x) {
    z[i] = alpha * (aty[i] + dSlackPos[i] - dSlackNeg[i]) * (ifScaled ? colScale[i] : 1.0);
  }
}

__global__ void element_dual_infeas_kernel_lb(cupdlp_float *z,
                                              const cupdlp_float *x,
                                              const cupdlp_float *hasLower,
                                              const cupdlp_float *colScale,
                                              cupdlp_float alpha, int ifScaled, int nCols) {
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < nCols; i += gridDim.x * blockDim.x) {
    z[i] = min(alpha * x[i], 0.0) * hasLower[i] / (ifScaled ? colScale[i] : 1.0);
  }
}

__global__ void element_dual_infeas_kernel_ub(cupdlp_float *z,
                                              const cupdlp_float *x,
                                              const cupdlp_float *hasUpper,
                                              const cupdlp_float *colScale,
                                              cupdlp_float alpha, int ifScaled, int nCols) {
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < nCols; i += gridDim.x * blockDim.x) {
    z[i] = max(alpha * x[i], 0.0) * hasUpper[i] / (ifScaled ? colScale[i] : 1.0);
  }
}

__global__ void element_dual_infeas_kernel_constr(cupdlp_float *z,
                                                  const cupdlp_float *ax,
                                                  const cupdlp_float *rowScale,
                                                  cupdlp_float alpha, int ifScaled,
                                                  int nEqs, int nRows) {
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < nRows; i += gridDim.x * blockDim.x) {
    cupdlp_float tmp = alpha * ax[i];
    if (i >= nEqs) tmp = min(tmp, 0.0);
    z[i] = tmp * (ifScaled ? rowScale[i] : 1.0);
  }
}

__global__ void element_wise_dot_kernel(cupdlp_float *x, const cupdlp_float *y, int n) {
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    x[i] *= y[i];
  }
}

__global__ void element_wise_div_kernel(cupdlp_float *x, const cupdlp_float *y, int n) {
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    x[i] /= y[i];
  }
}

__global__ void element_wise_projlb_kernel(cupdlp_float *x,
                                           const cupdlp_float *lb, int n) {
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    x[i] = max(x[i], lb[i]);
  }
}

__global__ void element_wise_projub_kernel(cupdlp_float *x,
                                           const cupdlp_float *ub, int n) {
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    x[i] = min(x[i], ub[i]);
  }
}

__global__ void element_wise_projSamelb_kernel(cupdlp_float *x,
                                               cupdlp_float lb, int n) {
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    x[i] = max(x[i], lb);
  }
}

__global__ void element_wise_projSameub_kernel(cupdlp_float *x,
                                               cupdlp_float ub, int n) {
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    x[i] = min(x[i], ub);
  }
}

__global__ void element_wise_initHaslb_kernel(cupdlp_float *haslb,
                                              const cupdlp_float *lb,
                                              cupdlp_float bound, int n) {
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    haslb[i] = lb[i] > bound ? 1.0 : 0.0;
  }
}

__global__ void element_wise_initHasub_kernel(cupdlp_float *hasub,
                                              const cupdlp_float *ub,
                                              cupdlp_float bound, int n) {
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    hasub[i] = ub[i] < bound ? 1.0 : 0.0;
  }
}

__global__ void element_wise_filterlb_kernel(cupdlp_float *x,
                                             const cupdlp_float *lb,
                                             cupdlp_float bound, int n) {
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    x[i] = lb[i] > bound ? lb[i] : 0.0;
  }
}

__global__ void element_wise_filterub_kernel(cupdlp_float *x,
                                             const cupdlp_float *ub,
                                             cupdlp_float bound, int n) {
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    x[i] = ub[i] < bound ? ub[i] : 0.0;
  }
}

__global__ void init_cuda_vec_kernel(cupdlp_float *x, cupdlp_float val, int n) {
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    x[i] = val;
  }
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

/*
// z = x - y
__global__ void naive_sub_kernel(cupdlp_float *z, const cupdlp_float *x,
                                 const cupdlp_float *y, int n) {
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    z[i] = x[i] - y[i];
  }
}
*/


// Warp size constant for device code
// CDNA (gfx90a, gfx94x): 64-lane wavefront
// RDNA (gfx10xx, gfx11xx): 32-lane wavefront
// CUDA: always 32
#if defined(__HIP_PLATFORM_AMD__)
  #if defined(__GFX9__)
    #define CUPDLP_WARP_SIZE 64
  #else
    #define CUPDLP_WARP_SIZE 32
  #endif
#else
  #define CUPDLP_WARP_SIZE 32
#endif

// Warp reduction macros: arch-unified for wave32 (CUDA/RDNA) and wave64 (CDNA)
// HIP uses __shfl_down (no sync needed, wavefronts execute in lockstep)
// CUDA uses __shfl_down_sync with a 32-bit mask

#if defined(__HIP_PLATFORM_AMD__)
// HIP: All AMD GPUs use __shfl_down (no sync needed)
// Wave64 (gfx9): reduce over 64 lanes
// Wave32 (RDNA): reduce over 32 lanes (offset>32 is no-op)

#if defined(__GFX9__)
// Wave64: CDNA (gfx90a, gfx94x)
#define FULL_WARP_REDUCE_2(val1, val2) { \
  val1 += __shfl_down(val1, 32); \
  val2 += __shfl_down(val2, 32); \
  val1 += __shfl_down(val1, 16); \
  val2 += __shfl_down(val2, 16); \
  val1 += __shfl_down(val1, 8); \
  val2 += __shfl_down(val2, 8); \
  val1 += __shfl_down(val1, 4); \
  val2 += __shfl_down(val2, 4); \
  val1 += __shfl_down(val1, 2); \
  val2 += __shfl_down(val2, 2); \
  val1 += __shfl_down(val1, 1); \
  val2 += __shfl_down(val2, 1); \
}

#define FULL_WARP_REDUCE(val) { \
  val += __shfl_down(val, 32); \
  val += __shfl_down(val, 16); \
  val += __shfl_down(val, 8); \
  val += __shfl_down(val, 4); \
  val += __shfl_down(val, 2); \
  val += __shfl_down(val, 1); \
}

// 256 threads / 64 lanes = 4 warps; final reduction of 4 elements
#define FINAL_REDUCE_2_256(val1, val2) { \
  val1 += __shfl_down(val1, 2); \
  val2 += __shfl_down(val2, 2); \
  val1 += __shfl_down(val1, 1); \
  val2 += __shfl_down(val2, 1); \
}

// 512 threads / 64 lanes = 8 warps; final reduction of 8 elements
#define FINAL_REDUCE_512(val) { \
  val += __shfl_down(val, 4); \
  val += __shfl_down(val, 2); \
  val += __shfl_down(val, 1); \
}

#define FINAL_REDUCE_256(val) { \
  val += __shfl_down(val, 2); \
  val += __shfl_down(val, 1); \
}

#else  // HIP RDNA (wave32)

#define FULL_WARP_REDUCE_2(val1, val2) { \
  val1 += __shfl_down(val1, 16); \
  val2 += __shfl_down(val2, 16); \
  val1 += __shfl_down(val1, 8); \
  val2 += __shfl_down(val2, 8); \
  val1 += __shfl_down(val1, 4); \
  val2 += __shfl_down(val2, 4); \
  val1 += __shfl_down(val1, 2); \
  val2 += __shfl_down(val2, 2); \
  val1 += __shfl_down(val1, 1); \
  val2 += __shfl_down(val2, 1); \
}

#define FULL_WARP_REDUCE(val) { \
  val += __shfl_down(val, 16); \
  val += __shfl_down(val, 8); \
  val += __shfl_down(val, 4); \
  val += __shfl_down(val, 2); \
  val += __shfl_down(val, 1); \
}

// 256 threads / 32 lanes = 8 warps; final reduction of 8 elements
#define FINAL_REDUCE_2_256(val1, val2) { \
  val1 += __shfl_down(val1, 4); \
  val2 += __shfl_down(val2, 4); \
  val1 += __shfl_down(val1, 2); \
  val2 += __shfl_down(val2, 2); \
  val1 += __shfl_down(val1, 1); \
  val2 += __shfl_down(val2, 1); \
}

// 512 threads / 32 lanes = 16 warps; final reduction of 16 elements
#define FINAL_REDUCE_512(val) { \
  val += __shfl_down(val, 8); \
  val += __shfl_down(val, 4); \
  val += __shfl_down(val, 2); \
  val += __shfl_down(val, 1); \
}

// 256 threads / 32 lanes = 8 warps; final reduction of 8 elements
#define FINAL_REDUCE_256(val) { \
  val += __shfl_down(val, 4); \
  val += __shfl_down(val, 2); \
  val += __shfl_down(val, 1); \
}
#endif  // __GFX9__

#else  // CUDA

// CUDA: Wave32, use __shfl_down_sync with 32-bit mask
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

#define FULL_WARP_REDUCE(val) { \
  val += __shfl_down_sync(0xFFFFFFFF, val, 16); \
  val += __shfl_down_sync(0xFFFFFFFF, val, 8); \
  val += __shfl_down_sync(0xFFFFFFFF, val, 4); \
  val += __shfl_down_sync(0xFFFFFFFF, val, 2); \
  val += __shfl_down_sync(0xFFFFFFFF, val, 1); \
}

// 256 threads / 32 lanes = 8 warps; final reduction of 8 elements
#define FINAL_REDUCE_2_256(val1, val2) { \
  val1 += __shfl_down_sync(0xFFFFFFFF, val1, 4); \
  val2 += __shfl_down_sync(0xFFFFFFFF, val2, 4); \
  val1 += __shfl_down_sync(0xFFFFFFFF, val1, 2); \
  val2 += __shfl_down_sync(0xFFFFFFFF, val2, 2); \
  val1 += __shfl_down_sync(0xFFFFFFFF, val1, 1); \
  val2 += __shfl_down_sync(0xFFFFFFFF, val2, 1); \
}

// 512 threads / 32 lanes = 16 warps; final reduction of 16 elements
#define FINAL_REDUCE_512(val) { \
  val += __shfl_down_sync(0xFFFFFFFF, val, 8); \
  val += __shfl_down_sync(0xFFFFFFFF, val, 4); \
  val += __shfl_down_sync(0xFFFFFFFF, val, 2); \
  val += __shfl_down_sync(0xFFFFFFFF, val, 1); \
}

// 256 threads / 32 lanes = 8 warps; final reduction of 8 elements
#define FINAL_REDUCE_256(val) { \
  val += __shfl_down_sync(0xFFFFFFFF, val, 4); \
  val += __shfl_down_sync(0xFFFFFFFF, val, 2); \
  val += __shfl_down_sync(0xFFFFFFFF, val, 1); \
}
#endif  // __HIP_PLATFORM_AMD__

// Max warps per block: 256/32=8 (wave32) or 512/32=16 (wave32), 256/64=4 or 512/64=8 (wave64)
// Upper bound is 16, use that for static shared memory sizing.
static constexpr int kMaxWarpsPerBlock = 16;

// assumes block size = 256, warp size = 32 or 64
__global__ void movement_1_kernel(cupdlp_float * __restrict__ res_x, cupdlp_float * __restrict__ res_y,
                                  const cupdlp_float * __restrict__ xUpdate, const cupdlp_float * __restrict__ x,
                                  const cupdlp_float * __restrict__ atyUpdate, const cupdlp_float * __restrict__ aty,
                                  int nCols) {

  __shared__ cupdlp_float shared_x[kMaxWarpsPerBlock];
  __shared__ cupdlp_float shared_y[kMaxWarpsPerBlock];
  cupdlp_float val_x = 0.0;
  cupdlp_float val_y = 0.0;
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < nCols; i += blockDim.x * gridDim.x) {
      cupdlp_float dx = xUpdate[i] - x[i];
      cupdlp_float day = atyUpdate[i] - aty[i];
      val_x = cupdlp_fma_rn(dx, dx, val_x);
      val_y = cupdlp_fma_rn(day, dx, val_y);
  }

  int lane = threadIdx.x % CUPDLP_WARP_SIZE;
  int wid = threadIdx.x / CUPDLP_WARP_SIZE;
  int nWarps = blockDim.x / CUPDLP_WARP_SIZE;

  FULL_WARP_REDUCE_2(val_x, val_y)
  if (lane == 0) {
    shared_x[wid] = val_x;
    shared_y[wid] = val_y;
  }
  __syncthreads();

  if (wid == 0) {
    val_x = (lane < nWarps) ? shared_x[lane] : 0.0;
    val_y = (lane < nWarps) ? shared_y[lane] : 0.0;
    FINAL_REDUCE_2_256(val_x, val_y)
    if (threadIdx.x == 0) {
      res_x[blockIdx.x] = val_x;
      res_y[blockIdx.x] = val_y;
    }
  }
}

// assumes: block size = 256, warp size = 32 or 64
__global__ void movement_2_kernel(cupdlp_float * __restrict__ res,
                                  const cupdlp_float * __restrict__ yUpdate, const cupdlp_float * __restrict__ y,
                                  int nRows) {

  __shared__ cupdlp_float shared[kMaxWarpsPerBlock];
  cupdlp_float val = 0.0;
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < nRows; i += blockDim.x * gridDim.x) {
      cupdlp_float d = yUpdate[i] - y[i];
      val = cupdlp_fma_rn(d, d, val);
  }

  int lane = threadIdx.x % CUPDLP_WARP_SIZE;
  int wid = threadIdx.x / CUPDLP_WARP_SIZE;
  int nWarps = blockDim.x / CUPDLP_WARP_SIZE;

  FULL_WARP_REDUCE(val)
  if (lane == 0) {
    shared[wid] = val;
  }
  __syncthreads();

  if (wid == 0) {
    val = (lane < nWarps) ? shared[lane] : 0.0;
    FINAL_REDUCE_256(val)
    if (threadIdx.x == 0) {
      res[blockIdx.x] = val;
    }
  }
}

// assumes: block size = 512, warp size = 32 or 64
__global__ void sum_kernel(cupdlp_float * __restrict__ res, const cupdlp_float * __restrict__ x, int n) {

  __shared__ cupdlp_float shared[kMaxWarpsPerBlock];
  cupdlp_float val = 0.0;
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
      val += x[i];
  }

  int lane = threadIdx.x % CUPDLP_WARP_SIZE;
  int wid = threadIdx.x / CUPDLP_WARP_SIZE;
  int nWarps = blockDim.x / CUPDLP_WARP_SIZE;

  FULL_WARP_REDUCE(val)
  if (lane == 0) {
    shared[wid] = val;
  }
  __syncthreads();

  if (wid == 0) {
    val = (lane < nWarps) ? shared[lane] : 0.0;
    FINAL_REDUCE_512(val)
    if (threadIdx.x == 0) {
      res[blockIdx.x] = val;
    }
  }
}
