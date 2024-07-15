#ifndef JL_CUPDLP_H
#define JL_CUPDLP_H

#include "../cupdlp/cupdlp.h"
#include "../interface/mps_lp.h"
#include "../interface/wrapper_highs.h"

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
    cupdlp_float *presolve_time, cupdlp_float *scaling_time);

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
    cupdlp_int **constraint_type);

#endif