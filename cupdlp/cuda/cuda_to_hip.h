/*
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Author: Jeff Daily <jeff.daily@amd.com>
 *
 * CUDA-to-HIP compatibility header for cuPDLP-C
 *
 * On AMD (USE_HIP / __HIP_PLATFORM_AMD__): aliases CUDA symbols to HIP equivalents.
 * On NVIDIA: no-op include of CUDA headers.
 */
#pragma once

#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)

// Ensure AMD platform is defined before including HIP headers
#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)
#define __HIP_PLATFORM_AMD__
#endif

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hipsparse/hipsparse.h>

// Runtime API
#define cudaMalloc                hipMalloc
#define cudaFree                  hipFree
#define cudaMemcpy                hipMemcpy
#define cudaMemcpyAsync           hipMemcpyAsync
#define cudaMemset                hipMemset
#define cudaDeviceSynchronize     hipDeviceSynchronize
#define cudaGetLastError          hipGetLastError
#define cudaGetErrorString        hipGetErrorString
#define cudaGetDeviceCount        hipGetDeviceCount
#define cudaGetDeviceProperties   hipGetDeviceProperties
#define cudaDeviceGetAttribute    hipDeviceGetAttribute
#define cudaRuntimeGetVersion     hipRuntimeGetVersion
#define cudaDriverGetVersion      hipDriverGetVersion
#define cudaDeviceReset           hipDeviceReset

// Memory copy kinds
#define cudaMemcpyDeviceToHost    hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice    hipMemcpyHostToDevice
#define cudaMemcpyDeviceToDevice  hipMemcpyDeviceToDevice
#define cudaMemcpyDefault         hipMemcpyDefault

// Error types
#define cudaError_t               hipError_t
#define cudaSuccess               hipSuccess

// Device properties
#define cudaDeviceProp            hipDeviceProp_t
#define cudaDevAttrMultiProcessorCount  hipDeviceAttributeMultiprocessorCount
#define cudaDevAttrWarpSize       hipDeviceAttributeWarpSize

// cuBLAS -> hipBLAS
#define cublasHandle_t            hipblasHandle_t
#define cublasStatus_t            hipblasStatus_t
#define CUBLAS_STATUS_SUCCESS     HIPBLAS_STATUS_SUCCESS
#define cublasCreate              hipblasCreate
#define cublasDestroy             hipblasDestroy
#define cublasDaxpy               hipblasDaxpy
#define cublasSaxpy               hipblasSaxpy
#define cublasDdot                hipblasDdot
#define cublasSdot                hipblasSdot
#define cublasDnrm2               hipblasDnrm2
#define cublasSnrm2               hipblasSnrm2
#define cublasDscal               hipblasDscal
#define cublasSscal               hipblasSscal
#define cublasGetStatusString     hipblasStatusToString

// cuSPARSE -> hipSPARSE
#define cusparseHandle_t          hipsparseHandle_t
#define cusparseStatus_t          hipsparseStatus_t
#define CUSPARSE_STATUS_SUCCESS   HIPSPARSE_STATUS_SUCCESS
#define cusparseCreate            hipsparseCreate
#define cusparseDestroy           hipsparseDestroy
#define cusparseGetVersion        hipsparseGetVersion
#define cusparseGetErrorString    hipsparseGetErrorString

// Sparse matrix/vector descriptors
#define cusparseSpMatDescr_t      hipsparseSpMatDescr_t
#define cusparseDnVecDescr_t      hipsparseDnVecDescr_t
#define cusparseCreateCsr         hipsparseCreateCsr
#define cusparseCreateCsc         hipsparseCreateCsc
#define cusparseCreateDnVec       hipsparseCreateDnVec
#define cusparseDestroySpMat      hipsparseDestroySpMat
#define cusparseDestroyDnVec      hipsparseDestroyDnVec

// SpMV operations
#define cusparseSpMV              hipsparseSpMV
#define cusparseSpMV_bufferSize   hipsparseSpMV_bufferSize
#define cusparseSpMVAlg_t         hipsparseSpMVAlg_t
#define cusparseOperation_t       hipsparseOperation_t
#define CUSPARSE_OPERATION_NON_TRANSPOSE  HIPSPARSE_OPERATION_NON_TRANSPOSE
#define CUSPARSE_OPERATION_TRANSPOSE      HIPSPARSE_OPERATION_TRANSPOSE
#define CUSPARSE_SPMV_CSR_ALG2    HIPSPARSE_SPMV_CSR_ALG2

// Compute type
#define CUDA_R_64F                HIP_R_64F
#define CUDA_R_32F                HIP_R_32F

// Index base
#define CUSPARSE_INDEX_BASE_ZERO  HIPSPARSE_INDEX_BASE_ZERO
#define CUSPARSE_INDEX_32I        HIPSPARSE_INDEX_32I

#else  // NVIDIA CUDA

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>

#endif  // USE_HIP
