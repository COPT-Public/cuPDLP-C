#ifndef CUPDLP_CUDA_LINALG_H
#define CUPDLP_CUDA_LINALG_H

#include <cublas_v2.h>         // cublas
#include <cuda_runtime_api.h>  // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>          // cusparseSpMV

#include "cupdlp_cuda_kernels.cuh"

#define PRINT_CUDA_INFO (1)
#define PRINT_DETAILED_CUDA_INFO (0)

#ifdef __cplusplus
extern "C" {
#endif

cupdlp_int cuda_alloc_MVbuffer(
    cusparseHandle_t handle, cusparseSpMatDescr_t cuda_csc,
    cusparseDnVecDescr_t vecX, cusparseDnVecDescr_t vecAx,
    cusparseSpMatDescr_t cuda_csr, cusparseDnVecDescr_t vecY,
    cusparseDnVecDescr_t vecATy, void **dBuffer_csc_ATy, void **dBuffer_csr_Ax);

/*
cupdlp_int cuda_csc_Ax(cusparseHandle_t handle,
                       cusparseSpMatDescr_t cuda_csc,
                       cusparseDnVecDescr_t vecX,
                       cusparseDnVecDescr_t vecAx, void *dBuffer,
                       cupdlp_float alpha, cupdlp_float beta);
*/
cupdlp_int cuda_csr_Ax(cusparseHandle_t handle,
                       cusparseSpMatDescr_t cuda_csr,
                       cusparseDnVecDescr_t vecX,
                       cusparseDnVecDescr_t vecAx, void *dBuffer,
                       cupdlp_float alpha, cupdlp_float beta);
cupdlp_int cuda_csc_ATy(cusparseHandle_t handle,
                        cusparseSpMatDescr_t cuda_csc,
                        cusparseDnVecDescr_t vecY,
                        cusparseDnVecDescr_t vecATy, void *dBuffer,
                        cupdlp_float alpha, cupdlp_float beta);
/*
cupdlp_int cuda_csr_ATy(cusparseHandle_t handle,
                        cusparseSpMatDescr_t cuda_csr,
                        cusparseDnVecDescr_t vecY,
                        cusparseDnVecDescr_t vecATy, void *dBuffer,
                        cupdlp_float alpha, cupdlp_float beta);
*/
void cupdlp_projSameub_cuda(cupdlp_float *x, const cupdlp_float ub, int n);
void cupdlp_projSamelb_cuda(cupdlp_float *x, const cupdlp_float lb, int n);

void cupdlp_projub_cuda(cupdlp_float *x, const cupdlp_float *ub, int n);
void cupdlp_projlb_cuda(cupdlp_float *x, const cupdlp_float *lb, int n);

void cupdlp_ediv_cuda(cupdlp_float *x, const cupdlp_float *y, int n);

void cupdlp_edot_cuda(cupdlp_float *x, const cupdlp_float *y, int n);

void cupdlp_haslb_cuda(cupdlp_float *haslb, const cupdlp_float *lb,
                       cupdlp_float bound, int n);
void cupdlp_hasub_cuda(cupdlp_float *hasub, const cupdlp_float *ub,
                       cupdlp_float bound, int n);

void cupdlp_filterlb_cuda(cupdlp_float *x, const cupdlp_float *lb,
                          cupdlp_float bound, int n);
void cupdlp_filterub_cuda(cupdlp_float *x, const cupdlp_float *ub,
                          cupdlp_float bound, int n);

void cupdlp_initvec_cuda(cupdlp_float *x, cupdlp_float val, int n);

void cupdlp_pgrad_cuda(cupdlp_float *xUpdate, const cupdlp_float *x,
                       const cupdlp_float *cost, const cupdlp_float *ATy,
                       const cupdlp_float *lb, const cupdlp_float *ub,
                       cupdlp_float dPrimalStep, int nCols);

void cupdlp_dgrad_cuda(cupdlp_float *yUpdate,
                       const cupdlp_float *y, const cupdlp_float *b,
                       const cupdlp_float *Ax, const cupdlp_float *AxUpdate,
                       cupdlp_float dDualStep, int nRows, int nEqs);

/*
void cupdlp_sub_cuda(cupdlp_float *z, const cupdlp_float *x,
                                const cupdlp_float *y, const cupdlp_int len);
*/

void cupdlp_movement_interaction_cuda(
    cupdlp_float *dX2, cupdlp_float *dY2, cupdlp_float *dInter, cupdlp_float *buffer,
    const cupdlp_float *xUpdate, const cupdlp_float *x,
    const cupdlp_float *yUpdate, const cupdlp_float *y,
    const cupdlp_float *atyUpdate, const cupdlp_float *aty,
    int nRows, int nCols);

cupdlp_int print_cuda_info(cusparseHandle_t handle);

#ifdef __cplusplus
}
#endif

#endif
