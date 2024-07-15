#include "jlcupdlp.h"

cupdlp_retcode cupdlp_presolve() {
  cupdlp_retcode retcode = RETCODE_OK;

exit_cleanup:
  return retcode;
}

/*
    min  cT x
    s.t. lhs <= Ax <= rhs
         lower <= x <= upper
*/
cupdlp_retcode cupdlp_solve(
    // problem data
    cupdlp_int nRows, cupdlp_int nCols, cupdlp_float *cost,
    cupdlp_int *A_csc_beg, cupdlp_int *A_csc_idx, cupdlp_float *A_csc_val,
    cupdlp_float *lhs, cupdlp_float *rhs, cupdlp_float *lower,
    cupdlp_float *upper, cupdlp_int sense, cupdlp_float offset,
    // parameters
    cupdlp_bool *ifChangeIntParam, cupdlp_int *intParam,
    cupdlp_bool *ifChangeFloatParam, cupdlp_float *floatParam,
    // solutions and informations
    cupdlp_int *status_pdlp, cupdlp_int *value_valid, cupdlp_int *dual_valid,
    cupdlp_float *col_value, cupdlp_float *col_dual, cupdlp_float *row_value,
    cupdlp_float *row_dual, cupdlp_float *primal_obj, cupdlp_float *dual_obj,
    cupdlp_float *duality_gap, cupdlp_float *comp, cupdlp_float *primal_feas,
    cupdlp_float *dual_feas, cupdlp_float *primal_obj_avg,
    cupdlp_float *dual_obj_avg, cupdlp_float *duality_gap_avg,
    cupdlp_float *comp_avg, cupdlp_float *primal_feas_avg,
    cupdlp_float *dual_feas_avg, cupdlp_int *niter, cupdlp_float *runtime,
    cupdlp_float *presolve_time, cupdlp_float *scaling_time) {
  cupdlp_retcode retcode = RETCODE_OK;

  printf("Calling C from Julia\n");

  // for cuPDLP
  int nCols_pdlp = 0;
  int nRows_pdlp = 0;
  int nEqs_pdlp = 0;
  int nnz_pdlp = 0;
  *status_pdlp = -1;

  cupdlp_float *rhs_pdlp = NULL;
  cupdlp_float *cost_pdlp = NULL;
  cupdlp_float *lower_pdlp = NULL;
  cupdlp_float *upper_pdlp = NULL;
  int *csc_beg_pdlp = NULL, *csc_idx_pdlp = NULL;
  double *csc_val_pdlp = NULL;

  //   double offset =
  //   0.0;  // true objVal = sig * c'x + offset, sig = 1 (min) or -1 (max)
  //   double sense = 1;  // 1 (min) or -1 (max)
  int *constraint_new_idx = NULL;
  int *constraint_type = NULL;

  cupdlp_float *col_value_pdlp = cupdlp_NULL;
  cupdlp_float *col_dual_pdlp = cupdlp_NULL;
  cupdlp_float *row_value_pdlp = cupdlp_NULL;
  cupdlp_float *row_dual_pdlp = cupdlp_NULL;

  CUPDLPscaling *scaling =
      (CUPDLPscaling *)cupdlp_malloc(sizeof(CUPDLPscaling));

  // claim solvers variables
  // prepare pointers
  CUPDLP_MATRIX_FORMAT src_matrix_format = CSC;
  CUPDLP_MATRIX_FORMAT dst_matrix_format = CSR_CSC;
  CUPDLPcsc *csc_cpu = cupdlp_NULL;
  CUPDLPproblem *prob = cupdlp_NULL;

  // the work object needs to be established first
  // free inside cuPDLP
  CUPDLPwork *w = cupdlp_NULL;
  CUPDLP_INIT_ZERO(w, 1);

  //  presolve
  *presolve_time = getTimeStamp();

  *presolve_time = getTimeStamp() - *presolve_time;

  // formulate LP
  CUPDLP_CALL(formulateLP(
      nRows, nCols, cost, A_csc_beg, A_csc_idx, A_csc_val, lhs, rhs, lower,
      upper, sense, offset, &nRows_pdlp, &nCols_pdlp, &nnz_pdlp, &nEqs_pdlp,
      &cost_pdlp, &csc_beg_pdlp, &csc_idx_pdlp, &csc_val_pdlp, &rhs_pdlp,
      &lower_pdlp, &upper_pdlp, &constraint_new_idx, &constraint_type));

  //   scaling
  CUPDLP_CALL(
      Init_Scaling(scaling, nCols_pdlp, nRows_pdlp, cost_pdlp, rhs_pdlp));
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
#if !(CUPDLP_CPU)
  cupdlp_float cuda_prepare_time = getTimeStamp();
  CHECK_CUSPARSE(cusparseCreate(&w->cusparsehandle));
  CHECK_CUBLAS(cublasCreate(&w->cublashandle));
  cuda_prepare_time = getTimeStamp() - cuda_prepare_time;
#endif

  CUPDLP_CALL(problem_create(&prob));
  // currently, only supprot that input matrix is CSC, and store both CSC and
  // CSR
  CUPDLP_CALL(csc_create(&csc_cpu));
  csc_cpu->nRows = nRows_pdlp;
  csc_cpu->nCols = nCols_pdlp;
  csc_cpu->nMatElem = nnz_pdlp;
  csc_cpu->colMatBeg = (int *)malloc((1 + nCols_pdlp) * sizeof(int));
  csc_cpu->colMatIdx = (int *)malloc(nnz_pdlp * sizeof(int));
  csc_cpu->colMatElem = (double *)malloc(nnz_pdlp * sizeof(double));
  memcpy(csc_cpu->colMatBeg, csc_beg_pdlp, (nCols_pdlp + 1) * sizeof(int));
  memcpy(csc_cpu->colMatIdx, csc_idx_pdlp, nnz_pdlp * sizeof(int));
  memcpy(csc_cpu->colMatElem, csc_val_pdlp, nnz_pdlp * sizeof(double));
#if !(CUPDLP_CPU)
  csc_cpu->cuda_csc = NULL;
#endif

  *scaling_time = getTimeStamp();
  CUPDLP_CALL(PDHG_Scale_Data_cuda(csc_cpu, ifScaling, scaling, cost_pdlp,
                                   lower_pdlp, upper_pdlp, rhs_pdlp));
  *scaling_time = getTimeStamp() - *scaling_time;

  // problem alloc
  cupdlp_float alloc_matrix_time = 0.0;
  cupdlp_float copy_vec_time = 0.0;

  CUPDLP_CALL(problem_alloc(prob, nRows_pdlp, nCols_pdlp, nEqs_pdlp, cost_pdlp,
                            offset, sense, csc_cpu, src_matrix_format,
                            dst_matrix_format, rhs_pdlp, lower_pdlp, upper_pdlp,
                            &alloc_matrix_time, &copy_vec_time));

  // solve
  w->problem = prob;
  w->scaling = scaling;
  PDHG_Alloc(w);
  w->timers->dScalingTime = *scaling_time;
  w->timers->dPresolveTime = *presolve_time;
  CUPDLP_COPY_VEC(w->rowScale, scaling->rowScale, cupdlp_float, nRows_pdlp);
  CUPDLP_COPY_VEC(w->colScale, scaling->colScale, cupdlp_float, nCols_pdlp);

#if !(CUPDLP_CPU)
  w->timers->AllocMem_CopyMatToDeviceTime += alloc_matrix_time;
  w->timers->CopyVecToDeviceTime += copy_vec_time;
  w->timers->CudaPrepareTime = cuda_prepare_time;
#endif

  cupdlp_printf("--------------------------------------------------\n");
  cupdlp_printf("enter main solve loop\n");
  cupdlp_printf("--------------------------------------------------\n");

  //   CUPDLP_CALL(LP_SolvePDHG(
  //   w, ifChangeIntParam, intParam, ifChangeFloatParam, floatParam, NULL,
  //   nCols_pdlp, col_value, col_dual, row_value, row_dual, value_valid,
  //   dual_valid, 0, NULL, constraint_new_idx, constraint_type, status_pdlp));

  PDHG_PrintHugeCUPDHG();

  CUPDLP_CALL(PDHG_SetUserParam(w, ifChangeIntParam, intParam,
                                ifChangeFloatParam, floatParam));

  CUPDLP_CALL(PDHG_Solve(w));

  *status_pdlp = (cupdlp_int)w->resobj->termCode;

  CUPDLP_CALL(PDHG_PostSolve(w, nCols, constraint_new_idx, constraint_type,
                             col_value, col_dual, row_value, row_dual,
                             value_valid, dual_valid));
  // get results
  *primal_obj = w->resobj->dPrimalObj;
  *dual_obj = w->resobj->dDualObj;
  *duality_gap = w->resobj->dDualityGap;
  *comp = w->resobj->dComplementarity;
  *primal_feas = w->resobj->dPrimalFeas;
  *dual_feas = w->resobj->dDualFeas;
  *primal_obj_avg = w->resobj->dPrimalObjAverage;
  *dual_obj_avg = w->resobj->dDualObjAverage;
  *duality_gap_avg = w->resobj->dDualityGapAverage;
  *comp_avg = w->resobj->dComplementarityAverage;
  *primal_feas_avg = w->resobj->dPrimalFeasAverage;
  *dual_feas_avg = w->resobj->dDualFeasAverage;
  *niter = w->timers->nIter;
  *runtime = w->timers->dSolvingTime;

exit_cleanup:
  PDHG_Destroy(&w);

  // free problem
  if (scaling) {
    scaling_clear(scaling);
  }

  if (cost_pdlp != NULL) cupdlp_free(cost_pdlp);
  if (csc_beg_pdlp != NULL) cupdlp_free(csc_beg_pdlp);
  if (csc_idx_pdlp != NULL) cupdlp_free(csc_idx_pdlp);
  if (csc_val_pdlp != NULL) cupdlp_free(csc_val_pdlp);
  if (rhs_pdlp != NULL) cupdlp_free(rhs_pdlp);
  if (lower_pdlp != NULL) cupdlp_free(lower_pdlp);
  if (upper_pdlp != NULL) cupdlp_free(upper_pdlp);
  if (constraint_new_idx != NULL) cupdlp_free(constraint_new_idx);
  if (constraint_type != NULL) cupdlp_free(constraint_type);

  // free memory
  csc_clear(csc_cpu);
  problem_clear(prob);

  return retcode;
}

cupdlp_retcode formulateLP(
    // problem data
    cupdlp_int nRows, cupdlp_int nCols, cupdlp_float *cost,
    cupdlp_int *A_csc_beg, cupdlp_int *A_csc_idx, cupdlp_float *A_csc_val,
    cupdlp_float *lhs, cupdlp_float *rhs, cupdlp_float *lower,
    cupdlp_float *upper, cupdlp_int sense, cupdlp_float offset,
    // reformulated problem data
    cupdlp_int *nRows_pdlp, cupdlp_int *nCols_pdlp, cupdlp_int *nnz_pdlp,
    cupdlp_int *nEqs_pdlp, cupdlp_float **cost_pdlp, cupdlp_int **csc_beg_pdlp,
    cupdlp_int **csc_idx_pdlp, cupdlp_float **csc_val_pdlp,
    cupdlp_float **rhs_pdlp, cupdlp_float **lower_pdlp,
    cupdlp_float **upper_pdlp, cupdlp_int **constraint_new_idx,
    cupdlp_int **constraint_type) {
  cupdlp_retcode retcode = RETCODE_OK;

  int has_lower, has_upper;

  *nRows_pdlp = nRows;
  *nCols_pdlp = nCols;
  *nEqs_pdlp = 0;
  *nnz_pdlp = A_csc_beg[nCols];

  CUPDLP_INIT(*constraint_type, nRows);
  CUPDLP_INIT(*constraint_new_idx, *nRows_pdlp);

  // recalculate nRows and nnz for Ax - z = 0
  for (int i = 0; i < nRows; i++) {
    has_lower = lhs[i] > -1e20;
    has_upper = rhs[i] < 1e20;

    // count number of equations and rows
    if (has_lower && has_upper && lhs[i] == rhs[i]) {
      (*constraint_type)[i] = EQ;
      (*nEqs_pdlp)++;
    } else if (has_lower && !has_upper) {
      (*constraint_type)[i] = GEQ;
    } else if (!has_lower && has_upper) {
      (*constraint_type)[i] = LEQ;
    } else if (has_lower && has_upper) {
      (*constraint_type)[i] = BOUND;
      (*nCols_pdlp)++;
      (*nnz_pdlp)++;
      (*nEqs_pdlp)++;
    } else {
      // printf("Error: constraint %d has no lower and upper bound\n", i);
      // retcode = 1;
      // goto exit_cleanup;

      // what if regard free as bounded
      printf("Warning: constraint %d has no lower and upper bound\n", i);
      (*constraint_type)[i] = BOUND;
      (*nCols_pdlp)++;
      (*nnz_pdlp)++;
      (*nEqs_pdlp)++;
    }
  }

  // allocate memory
  CUPDLP_INIT(*cost_pdlp, *nCols_pdlp);
  CUPDLP_INIT(*lower_pdlp, *nCols_pdlp);
  CUPDLP_INIT(*upper_pdlp, *nCols_pdlp);
  CUPDLP_INIT(*csc_beg_pdlp, *nCols_pdlp + 1);
  CUPDLP_INIT(*csc_idx_pdlp, *nnz_pdlp);
  CUPDLP_INIT(*csc_val_pdlp, *nnz_pdlp);
  CUPDLP_INIT(*rhs_pdlp, *nRows_pdlp);

  // cost, lower, upper
  for (int i = 0; i < nCols; i++) {
    (*cost_pdlp)[i] = cost[i] * (sense);
    (*lower_pdlp)[i] = lower[i];
    (*upper_pdlp)[i] = upper[i];
  }
  // slack costs
  for (int i = nCols; i < *nCols_pdlp; i++) {
    (*cost_pdlp)[i] = 0.0;
  }
  // slack bounds
  for (int i = 0, j = nCols; i < *nRows_pdlp; i++) {
    if ((*constraint_type)[i] == BOUND) {
      (*lower_pdlp)[j] = lhs[i];
      (*upper_pdlp)[j] = rhs[i];
      j++;
    }
  }

  for (int i = 0; i < *nCols_pdlp; i++) {
    if ((*lower_pdlp)[i] < -1e20) (*lower_pdlp)[i] = -INFINITY;
    if ((*upper_pdlp)[i] > 1e20) (*upper_pdlp)[i] = INFINITY;
  }

  // permute LP rhs
  // EQ or BOUND first
  for (int i = 0, j = 0; i < *nRows_pdlp; i++) {
    if ((*constraint_type)[i] == EQ) {
      (*rhs_pdlp)[j] = lhs[i];
      (*constraint_new_idx)[i] = j;
      j++;
    } else if ((*constraint_type)[i] == BOUND) {
      (*rhs_pdlp)[j] = 0.0;
      (*constraint_new_idx)[i] = j;
      j++;
    }
  }
  // then LEQ or GEQ
  for (int i = 0, j = *nEqs_pdlp; i < *nRows_pdlp; i++) {
    if ((*constraint_type)[i] == LEQ) {
      (*rhs_pdlp)[j] = -rhs[i];  // multiply -1
      (*constraint_new_idx)[i] = j;
      j++;
    } else if ((*constraint_type)[i] == GEQ) {
      (*rhs_pdlp)[j] = lhs[i];
      (*constraint_new_idx)[i] = j;
      j++;
    }
  }

  // formulate and permute LP matrix
  // beg remains the same
  for (int i = 0; i < nCols + 1; i++) (*csc_beg_pdlp)[i] = A_csc_beg[i];
  for (int i = nCols + 1; i < *nCols_pdlp + 1; i++)
    (*csc_beg_pdlp)[i] = (*csc_beg_pdlp)[i - 1] + 1;

  // row idx changes
  for (int i = 0, k = 0; i < nCols; i++) {
    // same order as in rhs
    // EQ or BOUND first
    for (int j = (*csc_beg_pdlp)[i]; j < (*csc_beg_pdlp)[i + 1]; j++) {
      if ((*constraint_type)[A_csc_idx[j]] == EQ ||
          (*constraint_type)[A_csc_idx[j]] == BOUND) {
        (*csc_idx_pdlp)[k] = (*constraint_new_idx)[A_csc_idx[j]];
        (*csc_val_pdlp)[k] = A_csc_val[j];
        k++;
      }
    }
    // then LEQ or GEQ
    for (int j = (*csc_beg_pdlp)[i]; j < (*csc_beg_pdlp)[i + 1]; j++) {
      if ((*constraint_type)[A_csc_idx[j]] == LEQ) {
        (*csc_idx_pdlp)[k] = (*constraint_new_idx)[A_csc_idx[j]];
        (*csc_val_pdlp)[k] = -A_csc_val[j];  // multiply -1
        k++;
      } else if ((*constraint_type)[A_csc_idx[j]] == GEQ) {
        (*csc_idx_pdlp)[k] = (*constraint_new_idx)[A_csc_idx[j]];
        (*csc_val_pdlp)[k] = A_csc_val[j];
        k++;
      }
    }
  }

  // slacks for BOUND
  for (int i = 0, j = nCols; i < *nRows_pdlp; i++) {
    if ((*constraint_type)[i] == BOUND) {
      (*csc_idx_pdlp)[(*csc_beg_pdlp)[j]] = (*constraint_new_idx)[i];
      (*csc_val_pdlp)[(*csc_beg_pdlp)[j]] = -1.0;
      j++;
    }
  }

exit_cleanup:
  return retcode;
}