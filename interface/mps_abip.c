#include "mps_lp.h"
/*
    min cTx
    s.t. Aeq x = b
         Aineq x <= bineq
         ub >= x >= 0
         colUbIdx: index of columns with upper bound (not all columns have upper
   bound)
*/
cupdlp_retcode main(int argc, char **argv) {
  cupdlp_retcode retcode = RETCODE_OK;

  char *fname = "./example/s_afiro.mps";
  char *fout = "./solution.json";

  char probname[128] = "?";
  int *Aeqp = NULL;
  int *Aeqi = NULL;
  double *Aeqx = NULL;

  int *Aineqp = NULL;
  int *Aineqi = NULL;
  double *Aineqx = NULL;

  int *Afullp = NULL;
  int *Afulli = NULL;
  double *Afullx = NULL;

  int *colUbIdx = NULL;
  double *colUbElem = NULL;

  int nCols;
  int nRows;
  int nEqs;
  int nIneqRow;
  int nColUb;

  int nnz = 0;
  double *rhs = NULL;
  double *cost = NULL;

  double *x = NULL;
  double *s = NULL;
  double *t = NULL;
  double *sx = NULL;
  double *ss = NULL;
  double *st = NULL;
  double *y = NULL;
  double *dense_data = NULL;
  int *row_ptr = NULL;
  int *col_ind = NULL;
  double *val_csr = NULL;
  int *col_ptr = NULL;
  int *row_ind = NULL;
  double *val_csc = NULL;

  cupdlp_float *lower = NULL;
  cupdlp_float *upper = NULL;

  int niters = 0;
  // load parameters
  for (cupdlp_int i = 0; i < argc - 1; i++) {
    if (strcmp(argv[i], "-niter") == 0) {
      niters = atof(argv[i + 1]);
    } else if (strcmp(argv[i], "-fname") == 0) {
      fname = argv[i + 1];
    } else if (strcmp(argv[i], "-out") == 0) {
      fout = argv[i + 1];
    } else if (strcmp(argv[i], "-h") == 0) {
      print_script_usage();
    }
  }
  if (strcmp(argv[argc - 1], "-h") == 0) {
    print_script_usage();
  }
  // claim solvers variables
  // prepare pointers
  CUPDLP_MATRIX_FORMAT src_matrix_format = CSC;
  CUPDLP_MATRIX_FORMAT dst_matrix_format = CSR_CSC;
  CUPDLPdense *dense = cupdlp_NULL;
  // CUPDLPcsr *csr = cupdlp_NULL;
  CUPDLPcsc *csc_cpu = cupdlp_NULL;
  CUPDLPproblem *prob = cupdlp_NULL;

  // set solver parameters
  cupdlp_bool ifChangeIntParam[N_INT_USER_PARAM] = {false};
  cupdlp_int intParam[N_INT_USER_PARAM] = {0};
  cupdlp_bool ifChangeFloatParam[N_FLOAT_USER_PARAM] = {false};
  cupdlp_float floatParam[N_FLOAT_USER_PARAM] = {0.0};
  CUPDLP_CALL(getUserParam(argc, argv, ifChangeIntParam, intParam,
                           ifChangeFloatParam, floatParam));

  /*
      min cTx
      s.t. Aeq x = b
           Aineq x <= bineq
           ub >= x >= 0
           colUbIdx: index of columns with upper bound (not all columns have
     upper bound)
  */

  /* Reading the standard mps file */
  retcode = cupdlpMpsRead(fname, probname, &nRows, &nEqs, &nIneqRow, &nCols,
                          &nnz, &Afullp, &Afulli, &Afullx, &Aeqp, &Aeqi, &Aeqx,
                          &Aineqp, &Aineqi, &Aineqx, &rhs, &cost, &nColUb,
                          &colUbIdx, &colUbElem);

  if (retcode != RETCODE_OK) {
    cupdlp_printf("Error reading MPS file\n");
    retcode = RETCODE_FAILED;
    goto exit_cleanup;
  }

  cupdlp_printf("prob: %s\n", probname);
  cupdlp_printf(
      "nRows: %d, nEqs: %d, nIneqRow: %d, nCols: %d, nnz: %d, nColUb: %d\n",
      nRows, nEqs, nIneqRow, nCols, nnz, nColUb);

  /* Create the full lower and upper bounds*/
  CUPDLP_INIT_ZERO(lower, nCols);
  CUPDLP_INIT_ZERO(upper, nCols);

  for (int i = 0; i < nCols; i++) {
    upper[i] = INFINITY;
  }
  for (int i = 0; i < nColUb; i++) {
    upper[colUbIdx[i]] = colUbElem[i];
  }

  // translate MPS standard from to PDLP form,
  //    min cTx
  //        s.t. - Aeq x    = -b
  //             - Aineq x >= -bineq
  //             ub >= x >= 0
  for (int i = 0; i < nRows; i++) {
    rhs[i] *= -1.0;
  }
  for (int i = 0; i < nnz; i++) {
    Afullx[i] *= -1.0;
  }
#if CUPDLP_DEBUG
  vecPrint("rhs: \n", rhs, nRows);
  vecPrint("lb: \n", lower, nCols);
  vecPrint("ub: \n", upper, nCols);
  cupdlp_int brief = 1;
  cupdlp_dcs *Afull = (cupdlp_dcs *)cupdlp_malloc(sizeof(cupdlp_dcs));
  Afull->nzmax = Afullp[nCols];
  Afull->m = nRows;
  Afull->n = nCols;
  Afull->p = Afullp;
  Afull->i = Afulli;
  Afull->x = Afullx;
  Afull->nz = -1;
  cupdlp_printf("Afull:\n");
  cupdlp_dcs_print(Afull, brief);
  cupdlp_dcs *AfullT = cupdlp_dcs_transpose(Afull, 1);
  cupdlp_printf("AfullT:\n");
  cupdlp_dcs_print(AfullT, brief);
#endif
  col_ptr = Afullp;
  row_ind = Afulli;
  val_csc = Afullx;

  CUPDLPscaling *scaling =
      (CUPDLPscaling *)cupdlp_malloc(sizeof(CUPDLPscaling));
  CUPDLP_CALL(Init_Scaling(scaling, nCols, nRows, cost, rhs));
  cupdlp_int ifScaling = 1;

  if (ifChangeIntParam[IF_SCALING]) {
    ifScaling = intParam[IF_SCALING];
  }

  if (ifChangeIntParam[IF_RUIZ_SCALING]) {
    scaling->ifRuizScaling = intParam[IF_RUIZ_SCALING];
  }

  if (ifChangeIntParam[IF_L2_SCALING]) {
    scaling->ifL2Scaling = intParam[IF_L2_SCALING];
  }

  if (ifChangeIntParam[IF_PC_SCALING]) {
    scaling->ifPcScaling = intParam[IF_PC_SCALING];
  }

  // these two handles need to be established first
  CUPDLPwork *w = cupdlp_NULL;
  CUPDLP_INIT_ZERO(w, 1);
#if !(CUPDLP_CPU)
  CHECK_CUSPARSE(cusparseCreate(&w->cusparsehandle));
  CHECK_CUBLAS(cublasCreate(&w->cublashandle));
#endif

  CUPDLP_CALL(problem_create(&prob));

  // currently, only supprot that input matrix is CSC, and store both CSC and
  // CSR
  CUPDLP_CALL(csc_create(&csc_cpu));
  csc_cpu->nRows = nRows;
  csc_cpu->nCols = nCols;
  csc_cpu->nMatElem = nnz;
  csc_cpu->colMatBeg = col_ptr;
  csc_cpu->colMatIdx = row_ind;
  csc_cpu->colMatElem = val_csc;
#if !(CUPDLP_CPU)
  csc_cpu->cuda_csc = NULL;
#endif

  CUPDLP_CALL(PDHG_Scale_Data(csc_cpu, ifScaling, scaling, cost, lower, upper, rhs));

  cupdlp_float alloc_matrix_time = 0.0;
  cupdlp_float copy_vec_time = 0.0;
  CUPDLP_CALL(problem_alloc(prob, nRows, nCols, nEqs, cost, 0.0, 1.0, csc_cpu,
                            src_matrix_format, dst_matrix_format, rhs, lower,
                            upper, &alloc_matrix_time, &copy_vec_time));
  // solve
  cupdlp_printf("--------------------------------------------------\n");
  cupdlp_printf("enter main solve loop\n");
  cupdlp_printf("--------------------------------------------------\n");
  // CUPDLP_CALL(LP_SolvePDHG(prob, cupdlp_NULL, cupdlp_NULL, cupdlp_NULL,
  // cupdlp_NULL));

  // vecPrint("csc cpu data", csc_cpu->colMatElem, csc_cpu->nMatElem);
  // cupdlp_float *tem = (cupdlp_float *)cupdlp_malloc(sizeof(cupdlp_float) *
  // csc_cpu->nMatElem); cupdlp_copy_vec(tem,
  // prob->data->csc_matrix->colMatElem, cupdlp_float, csc_cpu->nMatElem);
  // vecPrint("csc gpu data", tem, csc_cpu->nMatElem);
  // cupdlp_copy_vec(tem, prob->data->csr_matrix->rowMatElem, cupdlp_float,
  // csc_cpu->nMatElem); vecPrint("csr gpu data", tem, csc_cpu->nMatElem);

  w->problem = prob;
  w->scaling = scaling;
  PDHG_Alloc(w);
#if !(CUPDLP_CPU)
  w->timers->AllocMem_CopyMatToDeviceTime += alloc_matrix_time;
  w->timers->CopyVecToDeviceTime += copy_vec_time;
#endif
  //   CUPDLP_CALL(LP_SolvePDHG(prob, scaling, ifChangeIntParam, intParam,
  //                               ifChangeFloatParam, floatParam, fout));
  CUPDLP_CALL(LP_SolvePDHG(w, ifChangeIntParam, intParam, ifChangeFloatParam,
                           floatParam, fout));

  // print result
  // TODO: implement after adding IO

exit_cleanup:
  // free memory
  if (scaling) {
    scaling_clear(scaling);
  }
  dense_clear(dense);
  // csr_clear(csr);
  // csc_clear(csc);
  csc_clear_host(csc_cpu);
  problem_clear(prob);
  freealldata(Aeqp, Aeqi, Aeqx, Aineqp, Aineqi, Aineqx, colUbIdx, colUbElem,
              rhs, cost, x, s, t, sx, ss, st, y, lower, upper);
  #if !(CUPDLP_CPU)
    CHECK_CUDA(cudaDeviceReset())
  #endif

  return retcode;
}