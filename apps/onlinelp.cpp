//
// Created by C. Zhang on 2023/12/26.
//
#include "onlinelp.h"

#include "../cupdlp/cupdlp.h"
#include "../interface/mps_lp.h"

// consider a large scale online LP instance
//  min     - π'x
//  s.t     - Ax >= -b
//        0 <= x <=using Eigen::MatrixXf;

SpMat getRandomSpMat(size_t nRows, size_t nCols, double p) {
  std::default_random_engine gen;
  std::uniform_real_distribution<double> v(0.0, 1.0);

  std::vector<Eigen::Triplet<double> > tripletList;
  long elems = long(nRows * nCols * p);
  tripletList.reserve(elems);
  SpMat mat(nRows, nCols);
  std::cout << "number of elems: " << elems << std::endl;
  for (int i = 0; i < nRows; ++i)
    for (int j = 0; j < nCols; ++j) {
      double val = v(gen);
      if (val < p) tripletList.emplace_back(i, j, val);
    }
  mat.setFromTriplets(tripletList.begin(), tripletList.end());
  return mat;
}

int main(int argc, char *argv[]) {
  char *fout = "./solution.json";
  int nCols = 0;
  int nRows = 0;
  int nEqs = 0;  // no inequalities.
  double p = 0.1;
  cupdlp_bool ifSaveSol = false;
  for (auto i = 0; i < argc - 1; i++) {
    if (strcmp(argv[i], "-out") == 0) {
      fout = argv[i + 1];
    } else if (strcmp(argv[i], "-m") == 0) {
      nRows = atoi(argv[i + 1]);
    } else if (strcmp(argv[i], "-n") == 0) {
      nCols = atoi(argv[i + 1]);
    } else if (strcmp(argv[i], "-p") == 0) {
      p = atof(argv[i + 1]);
    }
  }

  // set solver parameters
  cupdlp_bool ifChangeIntParam[N_INT_USER_PARAM] = {false};
  cupdlp_int intParam[N_INT_USER_PARAM] = {0};
  cupdlp_bool ifChangeFloatParam[N_FLOAT_USER_PARAM] = {false};
  cupdlp_float floatParam[N_FLOAT_USER_PARAM] = {0.0};
  getUserParam(argc, argv, ifChangeIntParam, intParam, ifChangeFloatParam,
               floatParam);
  if ((nCols == 0) || (nRows == 0)) {
    return -1;
  }
  // ---------------------------------------------------
  // generate a random online LP instance
  eigen_array pi = ArrayXd::Random(nCols).abs() * -1.0;
  std::cout << "generating sparse matrix" << std::endl;
  SpMat A = getRandomSpMat(nRows, nCols, p) * -1.0;
  eigen_array b = ArrayXd::Random(nRows).abs() * -nCols / 10;
  long nnz = A.nonZeros();
  // new idx is unchanged since it is not permuted;
  eigen_array_int constraint_new_arr(nRows);
  for (auto i = 0; i < nRows; ++i) constraint_new_arr[i] = i;
  cupdlp_int *constraint_new_idx = constraint_new_arr.data();
  // lower = 0; upper = 1
  eigen_array lower(nCols);
  eigen_array upper(nCols);
  lower.setZero();
  upper.setOnes();
  std::cout << "generating sparse matrix finished" << std::endl;

#if DBG_ONLINE_LP
  std::cout << pi.transpose() << std::endl;
  std::cout << A << std::endl;
  std::cout << lower.transpose() << std::endl;
  std::cout << upper.transpose() << std::endl;
#endif

  auto *scaling = (CUPDLPscaling *)cupdlp_malloc(sizeof(CUPDLPscaling));

  // claim solvers variables
  // prepare pointers
  CUPDLP_MATRIX_FORMAT src_matrix_format = CSC;
  CUPDLP_MATRIX_FORMAT dst_matrix_format = CSR_CSC;
  CUPDLPcsc *csc_cpu = cupdlp_NULL;
  CUPDLPproblem *prob = cupdlp_NULL;
  CUPDLPwork *w = cupdlp_NULL;
  w = (CUPDLPwork *)calloc(1, sizeof(CUPDLPwork));
#if !(CUPDLP_CPU)
  cupdlp_float cuda_prepare_time = getTimeStamp();
  CHECK_CUSPARSE(cusparseCreate(&w->cusparsehandle));
  CHECK_CUBLAS(cublasCreate(&w->cublashandle));
  cuda_prepare_time = getTimeStamp() - cuda_prepare_time;
#endif

  problem_create(&prob);
  csc_create(&csc_cpu);
  csc_cpu->nRows = nRows;
  csc_cpu->nCols = nCols;
  csc_cpu->nMatElem = nnz;
  csc_cpu->colMatBeg = A.outerIndexPtr();
  csc_cpu->colMatIdx = A.innerIndexPtr();
  csc_cpu->colMatElem = A.valuePtr();

#if !(CUPDLP_CPU)
  csc_cpu->cuda_csc = NULL;
#endif

  cupdlp_float scaling_time = getTimeStamp();
  Init_Scaling(scaling, nCols, nRows, pi.data(), b.data());
  PDHG_Scale_Data(csc_cpu, 1, scaling, pi.data(), lower.data(), upper.data(), b.data());
  scaling_time = getTimeStamp() - scaling_time;

  cupdlp_float alloc_matrix_time = 0.0;
  cupdlp_float copy_vec_time = 0.0;

  problem_alloc(prob, nRows, nCols, nEqs, pi.data(), 0.0, 1.0, csc_cpu,
                src_matrix_format, dst_matrix_format, b.data(), lower.data(),
                upper.data(), &alloc_matrix_time, &copy_vec_time);

  w->problem = prob;
  w->scaling = scaling;

#if DBG_ONLINE_LP
  // checkout
  eigen_buff rb(scaling->rowScale, nRows);
  eigen_buff cb(scaling->colScale, nCols);
  std::cout << "row scaler: " << rb.transpose() << std::endl;
  std::cout << "col scaler: " << cb.transpose() << std::endl;
#endif

  PDHG_Alloc(w);
  w->timers->dScalingTime = scaling_time;
  w->timers->dPresolveTime = 0.0;
  CUPDLP_COPY_VEC(w->rowScale, scaling->rowScale, cupdlp_float, nRows);
  CUPDLP_COPY_VEC(w->colScale, scaling->colScale, cupdlp_float, nCols);

#if !(CUPDLP_CPU)
  w->timers->AllocMem_CopyMatToDeviceTime += alloc_matrix_time;
  w->timers->CopyVecToDeviceTime += copy_vec_time;
  w->timers->CudaPrepareTime = cuda_prepare_time;
#endif

  cupdlp_printf("--------------------------------------------------\n");
  cupdlp_printf("enter main solve loop\n");
  cupdlp_printf("--------------------------------------------------\n");

  eigen_array x_origin(nCols);
  eigen_array y_origin(nRows);
  x_origin.setOnes();
  y_origin.setOnes();

  LP_SolvePDHG(w, ifChangeIntParam, intParam, ifChangeFloatParam, floatParam,
               fout, nCols, x_origin.data(), NULL, NULL, y_origin.data(), NULL,
               NULL, ifSaveSol, NULL, constraint_new_idx, NULL);

  // print result
  // TODO: implement after adding IO

exit_cleanup:

  if (scaling) {
    scaling_clear(scaling);
  }
  // free memory
  problem_clear(prob);
  #if !(CUPDLP_CPU)
    CHECK_CUDA(cudaDeviceReset())
  #endif
  return 0;
}
