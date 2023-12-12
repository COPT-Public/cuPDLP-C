#ifndef CUPDLP_CUDA_LINALG_H
#define CUPDLP_CUDA_LINALG_H

#include <cublas_v2.h>         // cublas
#include <cuda_runtime_api.h>  // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>          // cusparseSpMV

#include "cupdlp_cuda_kernels.cuh"

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>   // printf
#include <stdlib.h>  // EXIT_FAILURE

// #include "../cupdlp_defs.h"
// #include "../glbopts.h"
#ifdef __cplusplus
}
#endif
typedef struct CUDA_MATRIX_VECTOR CUDAmv;

struct CUDA_MATRIX_VECTOR {
  cupdlp_int A_num_rows, A_num_cols, A_nnz;

  cusparseHandle_t handle;

  cusparseSpMatDescr_t cuda_csc;
  cupdlp_int *dA_cscOffsets, *dA_rows, *dAcsc_values;

  cusparseSpMatDescr_t cuda_csr;
  cupdlp_int *dA_csrOffsets, *dA_columns, *dAcsr_values;

  void *dBuffer;

  cupdlp_float *dx, *dAx, *dy, *dATy;

  cusparseDnVecDescr_t vecX, vecAx, vecY, vecATy;
};

// int cupdlp_cuda_csr_mul_dv_single_gpu(const int A_num_rows, const int
// A_num_cols, const int A_nnz,
//                                       const int *hA_csrOffsets, const int
//                                       *hA_columns, const float *hA_values,
//                                       const float *hX, float *hY, const float
//                                       alpha, const float beta);
#ifdef __cplusplus
extern "C" cupdlp_int cupdlp_cuda_csr_mul_dv_single_gpu(
    const cupdlp_int A_num_rows, const cupdlp_int A_num_cols,
    const cupdlp_int A_nnz, const cupdlp_int *hA_csrOffsets,
    const cupdlp_int *hA_columns, const cupdlp_float *hA_values,
    const cupdlp_float *hX, cupdlp_float *hY, const cupdlp_float alpha,
    const cupdlp_float beta, cupdlp_float *befor_time, cupdlp_float *mv_time,
    cupdlp_float *after_time);
// #else
// int foo(int,int);
extern "C" CUDAmv *cuda_init_mv(const cupdlp_int A_num_rows,
                                const cupdlp_int A_num_cols,
                                const cupdlp_int A_nnz);

extern "C" cupdlp_int cuda_alloc_csr(CUDAmv *MV,
                                     const cupdlp_int *hA_csrOffsets,
                                     const cupdlp_int *hA_columns,
                                     const cupdlp_float *hAcsr_values);

extern "C" cupdlp_int cuda_alloc_csc(CUDAmv *MV,
                                     const cupdlp_int *hA_cscOffsets,
                                     const cupdlp_int *hA_rows,
                                     const cupdlp_float *hAcsc_values);

extern "C" cupdlp_int cuda_alloc_vectors(CUDAmv *MV,
                                         const cupdlp_int A_num_rows,
                                         const cupdlp_int A_num_cols);

extern "C" cupdlp_int cuda_alloc_dBuffer(CUDAmv *MV);

// cupdlp_int cuda_csc_Ax(CUDAmv *MV, const cupdlp_float *hX, cupdlp_float *hAx,
// const cupdlp_float alpha, const cupdlp_float beta);

// cupdlp_int cuda_csr_Ax(CUDAmv *MV, const cupdlp_float *hX, cupdlp_float *hAx,
// const cupdlp_float alpha, const cupdlp_float beta);

// cupdlp_int cuda_csc_ATy(CUDAmv *MV, const cupdlp_float *hY, cupdlp_float
// *hATy, const cupdlp_float alpha, const cupdlp_float beta);

// cupdlp_int cuda_csr_ATy(CUDAmv *MV, const cupdlp_float *hY, cupdlp_float
// *hATy, const cupdlp_float alpha, const cupdlp_float beta);

extern "C" cupdlp_int cuda_copy_data_from_host_to_device(cupdlp_float *dX,
                                                         const cupdlp_float *hX,
                                                         const cupdlp_int len);

extern "C" cupdlp_int cuda_copy_data_from_device_to_host(cupdlp_float *hX,
                                                         const cupdlp_float *dX,
                                                         const cupdlp_int len);

// cupdlp_int cuda_csc_Ax(CUDAmv *MV, const cupdlp_float alpha, const
// cupdlp_float beta);

// cupdlp_int cuda_csr_Ax(CUDAmv *MV, const cupdlp_float alpha, const
// cupdlp_float beta);

// cupdlp_int cuda_csc_ATy(CUDAmv *MV, const cupdlp_float alpha, const
// cupdlp_float beta);

// cupdlp_int cuda_csr_ATy(CUDAmv *MV, const cupdlp_float alpha, const
// cupdlp_float beta);

extern "C" cupdlp_int cuda_csc_Ax(cusparseHandle_t handle,
                                  cusparseSpMatDescr_t cuda_csc,
                                  cusparseDnVecDescr_t vecX,
                                  cusparseDnVecDescr_t vecAx, void *dBuffer,
                                  const cupdlp_float alpha,
                                  const cupdlp_float beta);
extern "C" cupdlp_int cuda_csr_Ax(cusparseHandle_t handle,
                                  cusparseSpMatDescr_t cuda_csr,
                                  cusparseDnVecDescr_t vecX,
                                  cusparseDnVecDescr_t vecAx, void *dBuffer,
                                  const cupdlp_float alpha,
                                  const cupdlp_float beta);
extern "C" cupdlp_int cuda_csc_ATy(cusparseHandle_t handle,
                                   cusparseSpMatDescr_t cuda_csc,
                                   cusparseDnVecDescr_t vecY,
                                   cusparseDnVecDescr_t vecATy, void *dBuffer,
                                   const cupdlp_float alpha,
                                   const cupdlp_float beta);
extern "C" cupdlp_int cuda_csr_ATy(cusparseHandle_t handle,
                                   cusparseSpMatDescr_t cuda_csr,
                                   cusparseDnVecDescr_t vecY,
                                   cusparseDnVecDescr_t vecATy, void *dBuffer,
                                   const cupdlp_float alpha,
                                   const cupdlp_float beta);

extern "C" cupdlp_int cuda_free_mv(CUDAmv *MV);

extern "C" void cupdlp_projSameub_cuda(cupdlp_float *x, const cupdlp_float ub,
                                       const cupdlp_int len);
extern "C" void cupdlp_projSamelb_cuda(cupdlp_float *x, const cupdlp_float lb,
                                       const cupdlp_int len);
extern "C" void cupdlp_projub_cuda(cupdlp_float *x, const cupdlp_float *ub,
                                   const cupdlp_int len);
extern "C" void cupdlp_projlb_cuda(cupdlp_float *x, const cupdlp_float *lb,
                                   const cupdlp_int len);
extern "C" void cupdlp_ediv_cuda(cupdlp_float *x, const cupdlp_float *y,
                                 const cupdlp_int len);
extern "C" void cupdlp_edot_cuda(cupdlp_float *x, const cupdlp_float *y,
                                 const cupdlp_int len);
extern "C" void cupdlp_haslb_cuda(cupdlp_float *haslb, const cupdlp_float *lb,
                                  const cupdlp_float bound,
                                  const cupdlp_int len);
extern "C" void cupdlp_hasub_cuda(cupdlp_float *hasub, const cupdlp_float *ub,
                                  const cupdlp_float bound,
                                  const cupdlp_int len);
extern "C" void cupdlp_filterlb_cuda(cupdlp_float *x, const cupdlp_float *lb,
                                     const cupdlp_float bound,
                                     const cupdlp_int len);
extern "C" void cupdlp_filterub_cuda(cupdlp_float *x, const cupdlp_float *ub,
                                     const cupdlp_float bound,
                                     const cupdlp_int len);
extern "C" void cupdlp_initvec_cuda(cupdlp_float *x, const cupdlp_float val,
                                    const cupdlp_int len);

extern "C" cupdlp_int cuda_alloc_MVbuffer(
    //        CUPDLP_MATRIX_FORMAT matrix_format,
    cusparseHandle_t handle, cusparseSpMatDescr_t cuda_csc,
    cusparseDnVecDescr_t vecX, cusparseDnVecDescr_t vecAx,
    cusparseSpMatDescr_t cuda_csr, cusparseDnVecDescr_t vecY,
    cusparseDnVecDescr_t vecATy, void **dBuffer);

extern "C" void cupdlp_pgrad_cuda(cupdlp_float *xUpdate, const cupdlp_float *x,
                                  const cupdlp_float *cost,
                                  const cupdlp_float *ATy,
                                  const cupdlp_float dPrimalStep,
                                  const cupdlp_int len);

extern "C" void cupdlp_dgrad_cuda(cupdlp_float *yUpdate, const cupdlp_float *y,
                                  const cupdlp_float *b, const cupdlp_float *Ax,
                                  const cupdlp_float *AxUpdate,
                                  const cupdlp_float dDualStep,
                                  const cupdlp_int len);

extern "C" void cupdlp_sub_cuda(cupdlp_float *z, const cupdlp_float *x,
                                  const cupdlp_float *y, const cupdlp_int len);
#endif
#endif