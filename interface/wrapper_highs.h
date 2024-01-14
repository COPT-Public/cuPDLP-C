//
// Created by sky on 24-01-06.
//

#ifndef CUPDLP_WRAPPER_HiGHS_H
#define CUPDLP_WRAPPER_HiGHS_H

#ifdef __cplusplus
extern "C" {
#endif

#define CUPDLP_INIT(var, size)                                  \
  {                                                             \
    (var) = (typeof(var))malloc((size) * sizeof(typeof(*var))); \
    if ((var) == NULL) {                                        \
      retcode = 1;                                              \
      goto exit_cleanup;                                        \
    }                                                           \
  }

typedef enum CONSTRAINT_TYPE { EQ = 0, LEQ, GEQ, BOUND } ConstraintType;

int formulateLP_highs(void *model, double **cost, int *nCols, int *nRows,
                      int *nnz, int *nEqs, int **csc_beg, int **csc_idx,
                      double **csc_val, double **rhs, double **lower,
                      double **upper, double *offset, double *sense_origin,
                      int *nCols_origin, int **constraint_new_idx,
                      int **constraint_type);

int loadMps_highs(void *model, const char *filename);  // ok 0, fail 1

void deleteModel_highs(void *model);
void *createModel_highs();

void *presolvedModel_highs(void *presolve, void *model);
void *postsolvedModel_highs(void *model, int nCols_pre, int nRows_pre,
                            double *col_value_pre, double *col_dual_pre,
                            double *row_value_pre, double *row_dual_pre,
                            int value_valid_pre, int dual_valid_pre,
                            int nCols_org, int nRows_org, double *col_value_org,
                            double *col_dual_org, double *row_value_org,
                            double *row_dual_org);

void getModelSize_highs(void *model, int *nCols, int *nRows, int *nnz);

#ifdef __cplusplus
}
#endif
#endif  // CUPDLP_WRAPPER_HiGHS_H
