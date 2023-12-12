#include "cupdlp_cudalinalg.cuh"

// CSR matrix - dense vector multiplication on single GPU
extern "C" cupdlp_int cupdlp_cuda_csr_mul_dv_single_gpu(
    const cupdlp_int A_num_rows, const cupdlp_int A_num_cols,
    const cupdlp_int A_nnz, const cupdlp_int *hA_csrOffsets,
    const cupdlp_int *hA_columns, const cupdlp_float *hA_values,
    const cupdlp_float *hX, cupdlp_float *hY, const cupdlp_float alpha,
    const cupdlp_float beta, cupdlp_float *befor_time, cupdlp_float *mv_time,
    cupdlp_float *after_time) {
  // Y = alpha * A * X + beta * Y

  cupdlp_float dStartTime;

  // Host problem definition
  // const cupdlp_int A_num_rows = 4;
  // const cupdlp_int A_num_cols = 4;
  // const cupdlp_int A_nnz = 9;
  // cupdlp_int hA_csrOffsets[] = {0, 3, 4, 7, 9};
  // cupdlp_int hA_columns[] = {0, 2, 3, 1, 0, 2, 3, 1, 3};
  // cupdlp_float hA_values[] = {1.0, 2.0, 3.0, 4.0, 5.0,
  //                       6.0, 7.0, 8.0, 9.0};
  // cupdlp_float hX[] = {1.0, 2.0, 3.0, 4.0};
  // cupdlp_float hY[] = {0.0, 0.0, 0.0, 0.0};
  // cupdlp_float hY_result[] = {19.0, 8.0, 51.0, 52.0};
  // cupdlp_float alpha = 1.0;
  // cupdlp_float beta = 0.0;

  // dStartTime = getTimeStamp();
  //--------------------------------------------------------------------------
  // Device memory management
  cupdlp_int *dA_csrOffsets, *dA_columns;
  cupdlp_float *dA_values, *dX, *dY;
  CHECK_CUDA(cudaMalloc((void **)&dA_csrOffsets,
                        (A_num_rows + 1) * sizeof(cupdlp_int)))
  CHECK_CUDA(cudaMalloc((void **)&dA_columns, A_nnz * sizeof(cupdlp_int)))
  CHECK_CUDA(cudaMalloc((void **)&dA_values, A_nnz * sizeof(cupdlp_float)))
  // CHECK_CUDA(cudaMalloc((void **)&dX, A_num_cols * sizeof(cupdlp_float)))
  // CHECK_CUDA(cudaMalloc((void **)&dY, A_num_rows * sizeof(cupdlp_float)))
  CHECK_CUDA(cudaMalloc((void **)&dX, A_num_rows * sizeof(cupdlp_float)))
  CHECK_CUDA(cudaMalloc((void **)&dY, A_num_cols * sizeof(cupdlp_float)))

  CHECK_CUDA(cudaMemcpy(dA_csrOffsets, hA_csrOffsets,
                        (A_num_rows + 1) * sizeof(cupdlp_int),
                        cudaMemcpyHostToDevice))
  CHECK_CUDA(cudaMemcpy(dA_columns, hA_columns, A_nnz * sizeof(cupdlp_int),
                        cudaMemcpyHostToDevice))
  CHECK_CUDA(cudaMemcpy(dA_values, hA_values, A_nnz * sizeof(cupdlp_float),
                        cudaMemcpyHostToDevice))
  // CHECK_CUDA(cudaMemcpy(dX, hX, A_num_cols * sizeof(cupdlp_float),
  //                       cudaMemcpyHostToDevice))
  // CHECK_CUDA(cudaMemcpy(dY, hY, A_num_rows * sizeof(cupdlp_float),
  //                       cudaMemcpyHostToDevice))
  CHECK_CUDA(cudaMemcpy(dX, hX, A_num_rows * sizeof(cupdlp_float),
                        cudaMemcpyHostToDevice))
  CHECK_CUDA(cudaMemcpy(dY, hY, A_num_cols * sizeof(cupdlp_float),
                        cudaMemcpyHostToDevice))
  //--------------------------------------------------------------------------
  // CUSPARSE APIs
  cusparseHandle_t handle = NULL;
  cusparseSpMatDescr_t matA;
  cusparseDnVecDescr_t vecX, vecY;
  void *dBuffer = NULL;
  size_t bufferSize = 0;
  // cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseOperation_t op = CUSPARSE_OPERATION_TRANSPOSE;
  cudaDataType computeType = CUDA_R_32F;
#ifndef SFLOAT
  computeType = CUDA_R_64F;
#endif

  CHECK_CUSPARSE(cusparseCreate(&handle))
  // Create sparse matrix A in CSR format
  CHECK_CUSPARSE(cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                   dA_csrOffsets, dA_columns, dA_values,
                                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_BASE_ZERO, computeType))
  // Create dense vector X
  // CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, A_num_cols, dX, computeType))
  // Create dense vector y
  // CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, A_num_rows, dY, computeType))
  CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, A_num_rows, dX, computeType))
  CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, A_num_cols, dY, computeType))

  // allocate an external buffer if needed
  CHECK_CUSPARSE(cusparseSpMV_bufferSize(
      handle, op, &alpha, matA, vecX, &beta, vecY, computeType,
      CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize))
  CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize))

  // *befor_time = getTimeStamp() - dStartTime;
  // dStartTime = getTimeStamp();

  // execute SpMV
  CHECK_CUSPARSE(cusparseSpMV(handle, op, &alpha, matA, vecX, &beta, vecY,
                              computeType, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer))

  // *mv_time = getTimeStamp() - dStartTime;
  // dStartTime = getTimeStamp();

  // destroy matrix/vector descriptors
  CHECK_CUSPARSE(cusparseDestroySpMat(matA))
  CHECK_CUSPARSE(cusparseDestroyDnVec(vecX))
  CHECK_CUSPARSE(cusparseDestroyDnVec(vecY))
  CHECK_CUSPARSE(cusparseDestroy(handle))
  //--------------------------------------------------------------------------
  // device result check
  // CHECK_CUDA(cudaMemcpy(hY, dY, A_num_rows * sizeof(cupdlp_float),
  //                       cudaMemcpyDeviceToHost))
  CHECK_CUDA(cudaMemcpy(hY, dY, A_num_cols * sizeof(cupdlp_float),
                        cudaMemcpyDeviceToHost))

  // cupdlp_int correct = 1;
  // for (cupdlp_int i = 0; i < A_num_cols; i++)
  // // for (cupdlp_int i = 0; i < A_num_rows; i++)
  // {
  //     // if (hY[i] != hY_result[i]) { // direct floating point comparison is
  //     not
  //     //     correct = 0;             // reliable
  //     //     break;
  //     // }
  //     printf("hY[%d]: %f\n", hY[i]);

  //     // printf("hY[%d]: %f, dY[%d]: %f\n", i, hY[i], i, dY[i]);
  // }
  // if (correct)
  //     printf("spmv_csr_example test PASSED\n");
  // else
  //     printf("spmv_csr_example test FAILED: wrong result\n");
  //--------------------------------------------------------------------------
  // device memory deallocation
  CHECK_CUDA(cudaFree(dBuffer))
  CHECK_CUDA(cudaFree(dA_csrOffsets))
  CHECK_CUDA(cudaFree(dA_columns))
  CHECK_CUDA(cudaFree(dA_values))
  CHECK_CUDA(cudaFree(dX))
  CHECK_CUDA(cudaFree(dY))

  // *after_time = getTimeStamp() - dStartTime;
  return EXIT_SUCCESS;
}

extern "C" CUDAmv *cuda_init_mv(const cupdlp_int A_num_rows,
                                const cupdlp_int A_num_cols,
                                const cupdlp_int A_nnz) {
  CUDAmv *MV = (CUDAmv *)malloc(sizeof(CUDAmv));
  MV->A_num_rows = A_num_rows;
  MV->A_num_cols = A_num_cols;
  MV->A_nnz = A_nnz;
  MV->handle = NULL;
  MV->dBuffer = NULL;
  MV->cuda_csc = NULL;
  MV->dAcsc_values = NULL;
  MV->cuda_csr = NULL;
  MV->dAcsr_values = NULL;
  MV->dA_csrOffsets = NULL;
  MV->dA_columns = NULL;
  MV->dx = NULL;
  MV->dAx = NULL;
  MV->dy = NULL;
  MV->dATy = NULL;
  MV->vecX = NULL;
  MV->vecAx = NULL;
  MV->vecY = NULL;
  MV->vecATy = NULL;

  //    CHECK_CUSPARSE(cusparseCreate(&MV->handle))

  return MV;
}

extern "C" cupdlp_int cuda_alloc_csr(CUDAmv *MV,
                                     const cupdlp_int *hA_csrOffsets,
                                     const cupdlp_int *hA_columns,
                                     const cupdlp_float *hAcsr_values) {
  cupdlp_int A_num_rows = MV->A_num_rows;
  cupdlp_int A_num_cols = MV->A_num_cols;
  cupdlp_int A_nnz = MV->A_nnz;
  // Device memory management
  CHECK_CUDA(cudaMalloc((void **)&MV->dA_csrOffsets,
                        (A_num_rows + 1) * sizeof(cupdlp_int)))
  CHECK_CUDA(cudaMalloc((void **)&MV->dA_columns, A_nnz * sizeof(cupdlp_int)))
  CHECK_CUDA(
      cudaMalloc((void **)&MV->dAcsr_values, A_nnz * sizeof(cupdlp_float)))
  CHECK_CUDA(cudaMemcpy(MV->dA_csrOffsets, hA_csrOffsets,
                        (A_num_rows + 1) * sizeof(cupdlp_int),
                        cudaMemcpyHostToDevice))
  CHECK_CUDA(cudaMemcpy(MV->dA_columns, hA_columns, A_nnz * sizeof(cupdlp_int),
                        cudaMemcpyHostToDevice))
  CHECK_CUDA(cudaMemcpy(MV->dAcsr_values, hAcsr_values,
                        A_nnz * sizeof(cupdlp_float), cudaMemcpyHostToDevice))

  cudaDataType computeType = CUDA_R_32F;
#ifndef SFLOAT
  computeType = CUDA_R_64F;
#endif
  // Create sparse matrix A in CSR format
  CHECK_CUSPARSE(cusparseCreateCsr(
      &MV->cuda_csr, A_num_rows, A_num_cols, A_nnz, MV->dA_csrOffsets,
      MV->dA_columns, MV->dAcsr_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ZERO, computeType))

  return EXIT_SUCCESS;
}

extern "C" cupdlp_int cuda_alloc_csc(CUDAmv *MV,
                                     const cupdlp_int *hA_cscOffsets,
                                     const cupdlp_int *hA_rows,
                                     const cupdlp_float *hAcsc_values) {
  cupdlp_int A_num_rows = MV->A_num_rows;
  cupdlp_int A_num_cols = MV->A_num_cols;
  cupdlp_int A_nnz = MV->A_nnz;
  // Device memory management
  CHECK_CUDA(cudaMalloc((void **)&MV->dA_cscOffsets,
                        (A_num_cols + 1) * sizeof(cupdlp_int)))
  CHECK_CUDA(cudaMalloc((void **)&MV->dA_rows, A_nnz * sizeof(cupdlp_int)))
  CHECK_CUDA(
      cudaMalloc((void **)&MV->dAcsc_values, A_nnz * sizeof(cupdlp_float)))

  CHECK_CUDA(cudaMemcpy(MV->dA_cscOffsets, hA_cscOffsets,
                        (A_num_cols + 1) * sizeof(cupdlp_int),
                        cudaMemcpyHostToDevice))
  CHECK_CUDA(cudaMemcpy(MV->dA_rows, hA_rows, A_nnz * sizeof(cupdlp_int),
                        cudaMemcpyHostToDevice))
  CHECK_CUDA(cudaMemcpy(MV->dAcsc_values, hAcsc_values,
                        A_nnz * sizeof(cupdlp_float), cudaMemcpyHostToDevice))

  cudaDataType computeType = CUDA_R_32F;
#ifndef SFLOAT
  computeType = CUDA_R_64F;
#endif
  // Create sparse matrix A in CSC format
  CHECK_CUSPARSE(cusparseCreateCsc(
      &MV->cuda_csc, A_num_rows, A_num_cols, A_nnz, MV->dA_cscOffsets,
      MV->dA_rows, MV->dAcsc_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ZERO, computeType))

  return EXIT_SUCCESS;
}

extern "C" cupdlp_int cuda_alloc_vectors(CUDAmv *MV,
                                         const cupdlp_int A_num_rows,
                                         const cupdlp_int A_num_cols) {
  CHECK_CUDA(cudaMalloc((void **)&MV->dx, A_num_cols * sizeof(cupdlp_float)))
  CHECK_CUDA(cudaMalloc((void **)&MV->dAx, A_num_rows * sizeof(cupdlp_float)))
  CHECK_CUDA(cudaMalloc((void **)&MV->dy, A_num_rows * sizeof(cupdlp_float)))
  CHECK_CUDA(cudaMalloc((void **)&MV->dATy, A_num_cols * sizeof(cupdlp_float)))

  cudaDataType computeType = CUDA_R_32F;
#ifndef SFLOAT
  computeType = CUDA_R_64F;
#endif

  CHECK_CUSPARSE(
      cusparseCreateDnVec(&MV->vecX, A_num_cols, MV->dx, computeType))
  CHECK_CUSPARSE(
      cusparseCreateDnVec(&MV->vecAx, A_num_rows, MV->dAx, computeType))
  CHECK_CUSPARSE(
      cusparseCreateDnVec(&MV->vecY, A_num_rows, MV->dy, computeType))
  CHECK_CUSPARSE(
      cusparseCreateDnVec(&MV->vecATy, A_num_cols, MV->dATy, computeType))

  return EXIT_SUCCESS;
}

// cupdlp_int cuda_alloc_dBuffer(CUDAmv *MV)
// {
//     cudaDataType computeType = CUDA_R_32F;
// #ifndef SFLOAT
//     computeType = CUDA_R_64F;
// #endif

//     size_t AxbufferSize = 0;
//     size_t ATybufferSize = 0;
//     cupdlp_float alpha = 1.0;
//     cupdlp_float beta = 0.0;

//     // get the buffer size needed by Ax
//     CHECK_CUSPARSE(cusparseSpMV_bufferSize(
//         MV->handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
//         &alpha, MV->cuda_csc, MV->vecX, &beta, MV->vecAx, computeType,
//         CUSPARSE_SPMV_ALG_DEFAULT, &AxbufferSize))
//     // get the buffer size needed by ATy
//     CHECK_CUSPARSE(cusparseSpMV_bufferSize(
//         MV->handle, CUSPARSE_OPERATION_TRANSPOSE,
//         &alpha, MV->cuda_csr, MV->vecY, &beta, MV->vecATy, computeType,
//         CUSPARSE_SPMV_ALG_DEFAULT, &ATybufferSize))

//     size_t bufferSize = (AxbufferSize > ATybufferSize) ? AxbufferSize :
//     ATybufferSize;

//     // allocate an external buffer if needed
//     CHECK_CUDA(cudaMalloc(&MV->dBuffer, bufferSize))

//     return EXIT_SUCCESS;
// }

extern "C" cupdlp_int cuda_alloc_MVbuffer(
    //        CUPDLP_MATRIX_FORMAT matrix_format,
    cusparseHandle_t handle, cusparseSpMatDescr_t cuda_csc,
    cusparseDnVecDescr_t vecX, cusparseDnVecDescr_t vecAx,
    cusparseSpMatDescr_t cuda_csr, cusparseDnVecDescr_t vecY,
    cusparseDnVecDescr_t vecATy, void **dBuffer) {
  cudaDataType computeType = CUDA_R_32F;
#ifndef SFLOAT
  computeType = CUDA_R_64F;
#endif

  size_t AxBufferSize = 0;
  size_t ATyBufferSize = 0;
  cupdlp_float alpha = 1.0;
  cupdlp_float beta = 0.0;

  //    switch (matrix_format) {
  //        case CSR_CSC:
  //            // get the buffer size needed by csc Ax
  CHECK_CUSPARSE(cusparseSpMV_bufferSize(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, cuda_csc, vecX, &beta,
      vecAx, computeType, CUSPARSE_SPMV_ALG_DEFAULT, &AxBufferSize))

  // get the buffer size needed by csr ATy
  CHECK_CUSPARSE(cusparseSpMV_bufferSize(
      handle, CUSPARSE_OPERATION_TRANSPOSE, &alpha, cuda_csr, vecY, &beta,
      vecATy, computeType, CUSPARSE_SPMV_ALG_DEFAULT, &ATyBufferSize))
  //            break;
  //        case CSR:
  //            // get the buffer size needed by csr Ax
  //        CHECK_CUSPARSE(cusparseSpMV_bufferSize(
  //                handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, cuda_csr,
  //                vecX, &beta, vecAx, computeType, CUSPARSE_SPMV_ALG_DEFAULT,
  //                &AxBufferSize))

  //            // get the buffer size needed by csr ATy
  //            CHECK_CUSPARSE(cusparseSpMV_bufferSize(
  //                    handle, CUSPARSE_OPERATION_TRANSPOSE, &alpha, cuda_csr,
  //                    vecY, &beta, vecATy, computeType,
  //                    CUSPARSE_SPMV_ALG_DEFAULT, &ATyBufferSize))
  //            break;
  //        case CSC:
  //             get the buffer size needed by csc Ax
  //         CHECK_CUSPARSE(cusparseSpMV_bufferSize(
  //                 handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, cuda_csc,
  //                 vecX, &beta, vecAx, computeType, CUSPARSE_SPMV_ALG_DEFAULT,
  //                 &AxBufferSize))

  //             // get the buffer size needed by csc ATy
  //             CHECK_CUSPARSE(cusparseSpMV_bufferSize(
  //                     handle, CUSPARSE_OPERATION_TRANSPOSE, &alpha, cuda_csc,
  //                     vecY, &beta, vecATy, computeType,
  //                     CUSPARSE_SPMV_ALG_DEFAULT, &ATyBufferSize))
  //            break;
  //        default:
  //            cupdlp_printf("Error: matrix_format is not supported!\n");
  //            exit(EXIT_FAILURE);
  //    }

  size_t bufferSize =
      (AxBufferSize > ATyBufferSize) ? AxBufferSize : ATyBufferSize;

  // allocate an external buffer if needed
  // CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize))
  CHECK_CUDA(cudaMalloc(dBuffer, bufferSize))

  return EXIT_SUCCESS;
}

// cupdlp_int cuda_csc_Ax(CUDAmv *MV, const cupdlp_float *hX, cupdlp_float *hAx,
// const cupdlp_float alpha, const cupdlp_float beta)
// {
//     // hAx = alpha * Acsc * hX + beta * hAx
//     cupdlp_int A_num_rows = MV->A_num_rows;
//     cupdlp_int A_num_cols = MV->A_num_cols;

//     // copy data from host to device
//     CHECK_CUDA(cudaMemcpy(MV->dx, hX, A_num_cols * sizeof(cupdlp_float),
//     cudaMemcpyHostToDevice))
//     // CHECK_CUDA(cusparseDnVecSetValues(MV->vecX, MV->dx)) // no need
//     if (beta != 0.0)
//     {
//         CHECK_CUDA(cudaMemcpy(MV->dAx, hAx, A_num_rows *
//         sizeof(cupdlp_float), cudaMemcpyHostToDevice))
//         // CHECK_CUDA(cusparseDnVecSetValues(MV->vecAx, MV->dAx)) // no need
//     }

//     cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
//     cudaDataType computeType = CUDA_R_32F;
// #ifndef SFLOAT
//     computeType = CUDA_R_64F;
// #endif

//     CHECK_CUSPARSE(cusparseSpMV(MV->handle, op,
//                                 &alpha, MV->cuda_csc, MV->vecX, &beta,
//                                 MV->vecAx, computeType,
//                                 CUSPARSE_SPMV_ALG_DEFAULT, MV->dBuffer))

//     // copy data from device to host
//     CHECK_CUDA(cudaMemcpy(hAx, MV->dAx, A_num_rows * sizeof(cupdlp_float),
//     cudaMemcpyDeviceToHost))

//     return EXIT_SUCCESS;
// }

// cupdlp_int cuda_csr_Ax(CUDAmv *MV, const cupdlp_float *hX, cupdlp_float *hAx,
// const cupdlp_float alpha, const cupdlp_float beta)
// {
//     // hAx = alpha * Acsc * hX + beta * hAx
//     cupdlp_int A_num_rows = MV->A_num_rows;
//     cupdlp_int A_num_cols = MV->A_num_cols;

//     // copy data from host to device
//     CHECK_CUDA(cudaMemcpy(MV->dx, hX, A_num_cols * sizeof(cupdlp_float),
//     cudaMemcpyHostToDevice))
//     // CHECK_CUDA(cusparseDnVecSetValues(MV->vecX, MV->dx)) // no need
//     if (beta != 0.0)
//     {
//         CHECK_CUDA(cudaMemcpy(MV->dAx, hAx, A_num_rows *
//         sizeof(cupdlp_float), cudaMemcpyHostToDevice))
//         // CHECK_CUDA(cusparseDnVecSetValues(MV->vecAx, MV->dAx)) // no need
//     }

//     cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
//     cudaDataType computeType = CUDA_R_32F;
// #ifndef SFLOAT
//     computeType = CUDA_R_64F;
// #endif

//     CHECK_CUSPARSE(cusparseSpMV(MV->handle, op,
//                                 &alpha, MV->cuda_csr, MV->vecX, &beta,
//                                 MV->vecAx, computeType,
//                                 CUSPARSE_SPMV_ALG_DEFAULT, MV->dBuffer))

//     // copy data from device to host
//     CHECK_CUDA(cudaMemcpy(hAx, MV->dAx, A_num_rows * sizeof(cupdlp_float),
//     cudaMemcpyDeviceToHost))

//     return EXIT_SUCCESS;
// }

// cupdlp_int cuda_csc_ATy(CUDAmv *MV, const cupdlp_float *hY, cupdlp_float
// *hATy, const cupdlp_float alpha, const cupdlp_float beta)
// {
//     // hATy = alpha * Acsr^T * hY + beta * hATy
//     cupdlp_int A_num_rows = MV->A_num_rows;
//     cupdlp_int A_num_cols = MV->A_num_cols;

//     // copy data from host to device
//     CHECK_CUDA(cudaMemcpy(MV->dy, hY, A_num_rows * sizeof(cupdlp_float),
//     cudaMemcpyHostToDevice))
//     // CHECK_CUDA(cusparseDnVecSetValues(MV->vecY, MV->dy)) // no need
//     if (beta != 0.0)
//     {
//         CHECK_CUDA(cudaMemcpy(MV->dATy, hATy, A_num_cols *
//         sizeof(cupdlp_float), cudaMemcpyHostToDevice))
//         // CHECK_CUDA(cusparseDnVecSetValues(MV->vecATy, MV->dATy)) // no
//         need
//     }

//     cusparseOperation_t op = CUSPARSE_OPERATION_TRANSPOSE;
//     cudaDataType computeType = CUDA_R_32F;
// #ifndef SFLOAT
//     computeType = CUDA_R_64F;
// #endif

//     CHECK_CUSPARSE(cusparseSpMV(MV->handle, op,
//                                 &alpha, MV->cuda_csc, MV->vecY, &beta,
//                                 MV->vecATy, computeType,
//                                 CUSPARSE_SPMV_ALG_DEFAULT, MV->dBuffer))

//     // copy data from device to host
//     CHECK_CUDA(cudaMemcpy(hATy, MV->dATy, A_num_cols * sizeof(cupdlp_float),
//     cudaMemcpyDeviceToHost))

//     return EXIT_SUCCESS;
// }

// cupdlp_int cuda_csr_ATy(CUDAmv *MV, const cupdlp_float *hY, cupdlp_float
// *hATy, const cupdlp_float alpha, const cupdlp_float beta)
// {
//     // hATy = alpha * Acsr^T * hY + beta * hATy
//     cupdlp_int A_num_rows = MV->A_num_rows;
//     cupdlp_int A_num_cols = MV->A_num_cols;

//     // copy data from host to device
//     CHECK_CUDA(cudaMemcpy(MV->dy, hY, A_num_rows * sizeof(cupdlp_float),
//     cudaMemcpyHostToDevice))
//     // CHECK_CUDA(cusparseDnVecSetValues(MV->vecY, MV->dy)) // no need
//     if (beta != 0.0)
//     {
//         CHECK_CUDA(cudaMemcpy(MV->dATy, hATy, A_num_cols *
//         sizeof(cupdlp_float), cudaMemcpyHostToDevice))
//         // CHECK_CUDA(cusparseDnVecSetValues(MV->vecATy, MV->dATy)) // no
//         need
//     }

//     cusparseOperation_t op = CUSPARSE_OPERATION_TRANSPOSE;
//     cudaDataType computeType = CUDA_R_32F;
// #ifndef SFLOAT
//     computeType = CUDA_R_64F;
// #endif

//     CHECK_CUSPARSE(cusparseSpMV(MV->handle, op,
//                                 &alpha, MV->cuda_csr, MV->vecY, &beta,
//                                 MV->vecATy, computeType,
//                                 CUSPARSE_SPMV_ALG_DEFAULT, MV->dBuffer))

//     // copy data from device to host
//     CHECK_CUDA(cudaMemcpy(hATy, MV->dATy, A_num_cols * sizeof(cupdlp_float),
//     cudaMemcpyDeviceToHost))

//     return EXIT_SUCCESS;
// }

extern "C" cupdlp_int cuda_copy_data_from_host_to_device(cupdlp_float *dX,
                                                         const cupdlp_float *hX,
                                                         const cupdlp_int len) {
  CHECK_CUDA(
      cudaMemcpy(dX, hX, len * sizeof(cupdlp_float), cudaMemcpyHostToDevice))
  return EXIT_SUCCESS;
}

extern "C" cupdlp_int cuda_copy_data_from_device_to_host(cupdlp_float *hX,
                                                         const cupdlp_float *dX,
                                                         const cupdlp_int len) {
  CHECK_CUDA(
      cudaMemcpy(hX, dX, len * sizeof(cupdlp_float), cudaMemcpyDeviceToHost))
  return EXIT_SUCCESS;
}

// cupdlp_int cuda_csc_Ax(CUDAmv *MV, const cupdlp_float alpha, const
// cupdlp_float beta)
// {
//     // hAx = alpha * Acsc * hX + beta * hAx

//     cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
//     cudaDataType computeType = CUDA_R_32F;
// #ifndef SFLOAT
//     computeType = CUDA_R_64F;
// #endif

//     CHECK_CUSPARSE(cusparseSpMV(MV->handle, op,
//                                 &alpha, MV->cuda_csc, MV->vecX, &beta,
//                                 MV->vecAx, computeType,
//                                 CUSPARSE_SPMV_ALG_DEFAULT, MV->dBuffer))

//     return EXIT_SUCCESS;
// }

// cupdlp_int cuda_csr_Ax(CUDAmv *MV, const cupdlp_float alpha, const
// cupdlp_float beta)
// {
//     // hAx = alpha * Acsc * hX + beta * hAx

//     cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
//     cudaDataType computeType = CUDA_R_32F;
// #ifndef SFLOAT
//     computeType = CUDA_R_64F;
// #endif

//     CHECK_CUSPARSE(cusparseSpMV(MV->handle, op,
//                                 &alpha, MV->cuda_csr, MV->vecX, &beta,
//                                 MV->vecAx, computeType,
//                                 CUSPARSE_SPMV_ALG_DEFAULT, MV->dBuffer))

//     return EXIT_SUCCESS;
// }

// cupdlp_int cuda_csc_ATy(CUDAmv *MV, const cupdlp_float alpha, const
// cupdlp_float beta)
// {
//     // hATy = alpha * Acsr^T * hY + beta * hATy
//     cusparseOperation_t op = CUSPARSE_OPERATION_TRANSPOSE;
//     cudaDataType computeType = CUDA_R_32F;
// #ifndef SFLOAT
//     computeType = CUDA_R_64F;
// #endif

//     CHECK_CUSPARSE(cusparseSpMV(MV->handle, op,
//                                 &alpha, MV->cuda_csc, MV->vecY, &beta,
//                                 MV->vecATy, computeType,
//                                 CUSPARSE_SPMV_ALG_DEFAULT, MV->dBuffer))

//     return EXIT_SUCCESS;
// }

// cupdlp_int cuda_csr_ATy(CUDAmv *MV, const cupdlp_float alpha, const
// cupdlp_float beta)
// {
//     // hATy = alpha * Acsr^T * hY + beta * hATy
//     cusparseOperation_t op = CUSPARSE_OPERATION_TRANSPOSE;
//     cudaDataType computeType = CUDA_R_32F;
// #ifndef SFLOAT
//     computeType = CUDA_R_64F;
// #endif

//     CHECK_CUSPARSE(cusparseSpMV(MV->handle, op,
//                                 &alpha, MV->cuda_csr, MV->vecY, &beta,
//                                 MV->vecATy, computeType,
//                                 CUSPARSE_SPMV_ALG_DEFAULT, MV->dBuffer))

//     return EXIT_SUCCESS;
// }

extern "C" cupdlp_int cuda_csc_Ax(cusparseHandle_t handle,
                                  cusparseSpMatDescr_t cuda_csc,
                                  cusparseDnVecDescr_t vecX,
                                  cusparseDnVecDescr_t vecAx, void *dBuffer,
                                  const cupdlp_float alpha,
                                  const cupdlp_float beta) {
  // hAx = alpha * Acsc * hX + beta * hAx

  cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cudaDataType computeType = CUDA_R_32F;
#ifndef SFLOAT
  computeType = CUDA_R_64F;
#endif

  CHECK_CUSPARSE(cusparseSpMV(handle, op, &alpha, cuda_csc, vecX, &beta, vecAx,
                              computeType, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer))

  return EXIT_SUCCESS;
}

extern "C" cupdlp_int cuda_csr_Ax(cusparseHandle_t handle,
                                  cusparseSpMatDescr_t cuda_csr,
                                  cusparseDnVecDescr_t vecX,
                                  cusparseDnVecDescr_t vecAx, void *dBuffer,
                                  const cupdlp_float alpha,
                                  const cupdlp_float beta) {
  // hAx = alpha * Acsc * hX + beta * hAx

  cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cudaDataType computeType = CUDA_R_32F;
#ifndef SFLOAT
  computeType = CUDA_R_64F;
#endif

  CHECK_CUSPARSE(cusparseSpMV(handle, op, &alpha, cuda_csr, vecX, &beta, vecAx,
                              computeType, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer))

  return EXIT_SUCCESS;
}

extern "C" cupdlp_int cuda_csc_ATy(cusparseHandle_t handle,
                                   cusparseSpMatDescr_t cuda_csc,
                                   cusparseDnVecDescr_t vecY,
                                   cusparseDnVecDescr_t vecATy, void *dBuffer,
                                   const cupdlp_float alpha,
                                   const cupdlp_float beta) {
  // hATy = alpha * Acsr^T * hY + beta * hATy
  cusparseOperation_t op = CUSPARSE_OPERATION_TRANSPOSE;
  cudaDataType computeType = CUDA_R_32F;
#ifndef SFLOAT
  computeType = CUDA_R_64F;
#endif

  CHECK_CUSPARSE(cusparseSpMV(handle, op, &alpha, cuda_csc, vecY, &beta, vecATy,
                              computeType, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer))

  return EXIT_SUCCESS;
}

extern "C" cupdlp_int cuda_csr_ATy(cusparseHandle_t handle,
                                   cusparseSpMatDescr_t cuda_csr,
                                   cusparseDnVecDescr_t vecY,
                                   cusparseDnVecDescr_t vecATy, void *dBuffer,
                                   const cupdlp_float alpha,
                                   const cupdlp_float beta) {
  // hATy = alpha * Acsr^T * hY + beta * hATy
  cusparseOperation_t op = CUSPARSE_OPERATION_TRANSPOSE;
  cudaDataType computeType = CUDA_R_32F;
#ifndef SFLOAT
  computeType = CUDA_R_64F;
#endif

  CHECK_CUSPARSE(cusparseSpMV(handle, op, &alpha, cuda_csr, vecY, &beta, vecATy,
                              computeType, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer))

  return EXIT_SUCCESS;
}

extern "C" cupdlp_int cuda_free_mv(CUDAmv *MV) {
  if (MV != NULL) {
    // destroy matrix/vector descriptors
    CHECK_CUSPARSE(cusparseDestroySpMat(MV->cuda_csr))
    CHECK_CUSPARSE(cusparseDestroySpMat(MV->cuda_csc))
    CHECK_CUSPARSE(cusparseDestroyDnVec(MV->vecX))
    CHECK_CUSPARSE(cusparseDestroyDnVec(MV->vecAx))
    CHECK_CUSPARSE(cusparseDestroyDnVec(MV->vecY))
    CHECK_CUSPARSE(cusparseDestroyDnVec(MV->vecATy))
    CHECK_CUSPARSE(cusparseDestroy(MV->handle))

    // device memory deallocation
    CHECK_CUDA(cudaFree(MV->dAcsr_values))
    CHECK_CUDA(cudaFree(MV->dAcsc_values))
    CHECK_CUDA(cudaFree(MV->dA_csrOffsets))
    CHECK_CUDA(cudaFree(MV->dA_columns))
    CHECK_CUDA(cudaFree(MV->dA_cscOffsets))
    CHECK_CUDA(cudaFree(MV->dA_rows))
    CHECK_CUDA(cudaFree(MV->dx))
    CHECK_CUDA(cudaFree(MV->dAx))
    CHECK_CUDA(cudaFree(MV->dy))
    CHECK_CUDA(cudaFree(MV->dATy))
    CHECK_CUDA(cudaFree(MV->dBuffer))

    //        cupdlp_free(MV);
    MV = NULL;
  }

  return EXIT_SUCCESS;
}

extern "C" void cupdlp_projSameub_cuda(cupdlp_float *x, const cupdlp_float ub,
                                       const cupdlp_int len) {
  element_wise_projSameub_kernel<<<cuda_gridsize(len), CUPDLP_BLOCK_SIZE>>>(
      x, ub, len);
}

extern "C" void cupdlp_projSamelb_cuda(cupdlp_float *x, const cupdlp_float lb,
                                       const cupdlp_int len) {
  element_wise_projSamelb_kernel<<<cuda_gridsize(len), CUPDLP_BLOCK_SIZE>>>(
      x, lb, len);
}

extern "C" void cupdlp_projub_cuda(cupdlp_float *x, const cupdlp_float *ub,
                                   const cupdlp_int len) {
  element_wise_projub_kernel<<<cuda_gridsize(len), CUPDLP_BLOCK_SIZE>>>(x, ub,
                                                                        len);
}

extern "C" void cupdlp_projlb_cuda(cupdlp_float *x, const cupdlp_float *lb,
                                   const cupdlp_int len) {
  element_wise_projlb_kernel<<<cuda_gridsize(len), CUPDLP_BLOCK_SIZE>>>(x, lb,
                                                                        len);
}

extern "C" void cupdlp_ediv_cuda(cupdlp_float *x, const cupdlp_float *y,
                                 const cupdlp_int len) {
  element_wise_div_kernel<<<cuda_gridsize(len), CUPDLP_BLOCK_SIZE>>>(x, y, len);
}

extern "C" void cupdlp_edot_cuda(cupdlp_float *x, const cupdlp_float *y,
                                 const cupdlp_int len) {
  element_wise_dot_kernel<<<cuda_gridsize(len), CUPDLP_BLOCK_SIZE>>>(x, y, len);
}

extern "C" void cupdlp_haslb_cuda(cupdlp_float *haslb, const cupdlp_float *lb,
                                  const cupdlp_float bound,
                                  const cupdlp_int len) {
  element_wise_initHaslb_kernal<<<cuda_gridsize(len), CUPDLP_BLOCK_SIZE>>>(
      haslb, lb, bound, len);
}

extern "C" void cupdlp_hasub_cuda(cupdlp_float *hasub, const cupdlp_float *ub,
                                  const cupdlp_float bound,
                                  const cupdlp_int len) {
  element_wise_initHasub_kernal<<<cuda_gridsize(len), CUPDLP_BLOCK_SIZE>>>(
      hasub, ub, bound, len);
}

extern "C" void cupdlp_filterlb_cuda(cupdlp_float *x, const cupdlp_float *lb,
                                     const cupdlp_float bound,
                                     const cupdlp_int len) {
  element_wise_filterlb_kernal<<<cuda_gridsize(len), CUPDLP_BLOCK_SIZE>>>(
      x, lb, bound, len);
}

extern "C" void cupdlp_filterub_cuda(cupdlp_float *x, const cupdlp_float *ub,
                                     const cupdlp_float bound,
                                     const cupdlp_int len) {
  element_wise_filterub_kernal<<<cuda_gridsize(len), CUPDLP_BLOCK_SIZE>>>(
      x, ub, bound, len);
}

extern "C" void cupdlp_initvec_cuda(cupdlp_float *x, const cupdlp_float val,
                                    const cupdlp_int len) {
  init_cuda_vec_kernal<<<cuda_gridsize(len), CUPDLP_BLOCK_SIZE>>>(x, val, len);
}

extern "C" void cupdlp_pgrad_cuda(cupdlp_float *xUpdate,
                                        const cupdlp_float *x,
                                        const cupdlp_float *cost,
                                        const cupdlp_float *ATy,
                                        const cupdlp_float dPrimalStep,
                                        const cupdlp_int len) {
    primal_grad_step_kernal<<<cuda_gridsize(len), CUPDLP_BLOCK_SIZE>>>(
        xUpdate, x, cost, ATy, dPrimalStep, len);
}

extern "C" void cupdlp_dgrad_cuda(cupdlp_float *yUpdate, const cupdlp_float *y, const cupdlp_float *b,
    const cupdlp_float *Ax, const cupdlp_float *AxUpdate,
    const cupdlp_float dDualStep, const cupdlp_int len) {
      dual_grad_step_kernal<<<cuda_gridsize(len), CUPDLP_BLOCK_SIZE>>>(
          yUpdate, y, b, Ax, AxUpdate, dDualStep, len);
}

extern "C" void cupdlp_sub_cuda(cupdlp_float *z, const cupdlp_float *x,
                                  const cupdlp_float *y, const cupdlp_int len)
{
   naive_sub_kernal<<<cuda_gridsize(len), CUPDLP_BLOCK_SIZE>>>(z, x, y, len);
}