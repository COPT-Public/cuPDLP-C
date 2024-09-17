#include "cupdlp_cudalinalg.cuh"

extern "C" cupdlp_int cuda_alloc_MVbuffer(
    cusparseHandle_t handle, cusparseSpMatDescr_t cuda_csc,
    cusparseDnVecDescr_t vecX, cusparseDnVecDescr_t vecAx,
    cusparseSpMatDescr_t cuda_csr, cusparseDnVecDescr_t vecY,
    cusparseDnVecDescr_t vecATy, void **dBuffer_csc_ATy, void **dBuffer_csr_Ax) {

  size_t AxBufferSize = 0;
  size_t ATyBufferSize = 0;
  cupdlp_float alpha = 1.0;
  cupdlp_float beta = 0.0;
  // cusparseSpSVAlg_t alg = CUSPARSE_SPSV_ALG_DEFAULT;
  cusparseSpMVAlg_t alg = CUSPARSE_SPMV_CSR_ALG2; //deterministic

  // get the buffer size needed by csr Ax
  CHECK_CUSPARSE(cusparseSpMV_bufferSize(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, cuda_csr, vecX, &beta,
      vecAx, CudaComputeType, alg, &AxBufferSize))

  // allocate an external buffer if needed
  CHECK_CUDA(cudaMalloc(dBuffer_csr_Ax, AxBufferSize))

  // get the buffer size needed by csc ATy
  CHECK_CUSPARSE(cusparseSpMV_bufferSize(
      handle, CUSPARSE_OPERATION_TRANSPOSE, &alpha, cuda_csc, vecY, &beta,
      vecATy, CudaComputeType, alg, &ATyBufferSize))

  // allocate an external buffer if needed
  CHECK_CUDA(cudaMalloc(dBuffer_csc_ATy, ATyBufferSize))

  return EXIT_SUCCESS;
}

extern "C" cupdlp_int cuda_csc_Ax(cusparseHandle_t handle,
                                  cusparseSpMatDescr_t cuda_csc,
                                  cusparseDnVecDescr_t vecX,
                                  cusparseDnVecDescr_t vecAx, void *dBuffer,
                                  const cupdlp_float alpha,
                                  const cupdlp_float beta) {
  // hAx = alpha * Acsc * hX + beta * hAx

  cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;

  CHECK_CUSPARSE(cusparseSpMV(handle, op, &alpha, cuda_csc, vecX, &beta, vecAx,
                              // CudaComputeType, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer))
                              CudaComputeType, CUSPARSE_SPMV_CSR_ALG2, dBuffer))

  return EXIT_SUCCESS;
}

extern "C" cupdlp_int cuda_csr_Ax(cusparseHandle_t handle,
                                  cusparseSpMatDescr_t cuda_csr,
                                  cusparseDnVecDescr_t vecX,
                                  cusparseDnVecDescr_t vecAx, void *dBuffer,
                                  const cupdlp_float alpha,
                                  const cupdlp_float beta) {
  // hAx = alpha * Acsr * hX + beta * hAx

  cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;

  CHECK_CUSPARSE(cusparseSpMV(handle, op, &alpha, cuda_csr, vecX, &beta, vecAx,
                              // CudaComputeType, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer))
                              CudaComputeType, CUSPARSE_SPMV_CSR_ALG2, dBuffer))

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

  CHECK_CUSPARSE(cusparseSpMV(handle, op, &alpha, cuda_csc, vecY, &beta, vecATy,
                              // CudaComputeType, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer))
                              CudaComputeType, CUSPARSE_SPMV_CSR_ALG2, dBuffer))

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

  CHECK_CUSPARSE(cusparseSpMV(handle, op, &alpha, cuda_csr, vecY, &beta, vecATy,
                              // CudaComputeType, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer))
                              CudaComputeType, CUSPARSE_SPMV_CSR_ALG2, dBuffer))

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
  element_wise_initHaslb_kernel<<<cuda_gridsize(len), CUPDLP_BLOCK_SIZE>>>(
      haslb, lb, bound, len);
}

extern "C" void cupdlp_hasub_cuda(cupdlp_float *hasub, const cupdlp_float *ub,
                                  const cupdlp_float bound,
                                  const cupdlp_int len) {
  element_wise_initHasub_kernel<<<cuda_gridsize(len), CUPDLP_BLOCK_SIZE>>>(
      hasub, ub, bound, len);
}

extern "C" void cupdlp_filterlb_cuda(cupdlp_float *x, const cupdlp_float *lb,
                                     const cupdlp_float bound,
                                     const cupdlp_int len) {
  element_wise_filterlb_kernel<<<cuda_gridsize(len), CUPDLP_BLOCK_SIZE>>>(
      x, lb, bound, len);
}

extern "C" void cupdlp_filterub_cuda(cupdlp_float *x, const cupdlp_float *ub,
                                     const cupdlp_float bound,
                                     const cupdlp_int len) {
  element_wise_filterub_kernel<<<cuda_gridsize(len), CUPDLP_BLOCK_SIZE>>>(
      x, ub, bound, len);
}

extern "C" void cupdlp_initvec_cuda(cupdlp_float *x, const cupdlp_float val,
                                    const cupdlp_int len) {
  init_cuda_vec_kernel<<<cuda_gridsize(len), CUPDLP_BLOCK_SIZE>>>(x, val, len);
}

extern "C" void cupdlp_pgrad_cuda(cupdlp_float *xUpdate,
                                  const cupdlp_float *x,
                                  const cupdlp_float *cost,
                                  const cupdlp_float *ATy,
                                  const cupdlp_float *lb,
                                  const cupdlp_float *ub,
                                  cupdlp_float dPrimalStep, int nCols) {
  constexpr int BLOCK_SIZE = 128;
  constexpr int ELS_PER_THREAD = 16;
  unsigned int nBlocks = (nCols + BLOCK_SIZE * ELS_PER_THREAD - 1) / (BLOCK_SIZE * ELS_PER_THREAD);
  primal_grad_step_kernel<<<nBlocks, BLOCK_SIZE>>>(xUpdate, x, cost, ATy, lb, ub, dPrimalStep, nCols);
}

extern "C" void cupdlp_dgrad_cuda(cupdlp_float *yUpdate,
                                  const cupdlp_float *y,
                                  const cupdlp_float *b,
                                  const cupdlp_float *Ax,
                                  const cupdlp_float *AxUpdate,
                                  cupdlp_float dDualStep, int nRows, int nEqs) {
  constexpr int BLOCK_SIZE = 128;
  constexpr int ELS_PER_THREAD = 16;
  unsigned int nBlocks = (nRows + BLOCK_SIZE * ELS_PER_THREAD - 1) / (BLOCK_SIZE * ELS_PER_THREAD);
  dual_grad_step_kernel<<<nBlocks, BLOCK_SIZE>>>(yUpdate, y, b, Ax, AxUpdate, dDualStep, nRows, nEqs);
}

extern "C" void cupdlp_sub_cuda(cupdlp_float *z, const cupdlp_float *x,
                                  const cupdlp_float *y, const cupdlp_int len)
{
   naive_sub_kernel<<<cuda_gridsize(len), CUPDLP_BLOCK_SIZE>>>(z, x, y, len);
}

extern "C" cupdlp_int print_cuda_info(cusparseHandle_t handle)
{
#if PRINT_CUDA_INFO

  int v_cuda_runtime = 0;
  int v_cuda_driver = 0;
  int v_cusparse = 0;
  CHECK_CUDA(cudaRuntimeGetVersion(&v_cuda_runtime))
  CHECK_CUDA(cudaDriverGetVersion(&v_cuda_driver))
  CHECK_CUSPARSE(cusparseGetVersion(handle, &v_cusparse))

  printf("Cuda runtime %d\n", v_cuda_runtime);
  printf("Cuda driver %d\n", v_cuda_driver);
  printf("cuSparse %d\n", v_cusparse);

  int n_devices = 0;
  CHECK_CUDA(cudaGetDeviceCount(&n_devices))

  for (int i = 0; i < n_devices; i++) {
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, i));

    printf("Cuda device %d: %s\n", i, prop.name);
#if PRINT_DETAILED_CUDA_INFO
    printf("  Clock rate (KHz): %d\n", prop.clockRate);
    printf("  Memory clock rate (KHz): %d\n", prop.memoryClockRate);
    printf("  Memory bus width (bits): %d\n", prop.memoryBusWidth);
    printf("  Peak memory bandwidth (GB/s): %f\n",
            2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    printf("  Global memory available on device (GB): %f\n", prop.totalGlobalMem / 1.0e9);
    printf("  Shared memory available per block (B): %zu\n", prop.sharedMemPerBlock);
    printf("  Warp size in threads: %d\n", prop.warpSize);
    printf("  Maximum number of threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("  Number of multiprocessors on device: %d\n", prop.multiProcessorCount);
#endif
  }
#endif

  return EXIT_SUCCESS;
}
