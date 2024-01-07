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

typedef enum CONSTRAINT_TYPE { EQ = 0, LEQ, GEQ, BOUND } constraint_type;

int formulateLP_highs(void *model, double **cost, int *nCols, int *nRows,
                      int *nnz, int *nEqs, int **csc_beg, int **csc_idx,
                      double **csc_val, double **rhs, double **lower,
                      double **upper, double *offset, double *sign_origin,
                      int *nCols_origin, int **constraint_new_idx);

void loadMps_highs(void *model, const char *filename);

void deleteModel_highs(void *model);
void *createModel_highs();

void *presolvedModel_highs(void *presolve, void *model);

#ifdef __cplusplus
}
#endif
#endif  // CUPDLP_WRAPPER_HiGHS_H
