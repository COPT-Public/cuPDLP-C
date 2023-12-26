//
// Created by chuwen on 23-12-4.
//

#ifndef CUPDLP_WRAPPER_CLP_H
#define CUPDLP_WRAPPER_CLP_H
#ifdef __cplusplus
extern "C" {
#endif
// #include "../cupdlp/cupdlp.h"

#ifdef DLONG
typedef long long cupdlp_int;
#else
typedef int cupdlp_int;
#endif

#ifndef SFLOAT
typedef double cupdlp_float;
#else
typedef float cupdlp_float;
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

cupdlp_int formulateLP(void *model, cupdlp_float **cost, cupdlp_int *nCols, cupdlp_int *nRows, cupdlp_int *nnz,
                cupdlp_int *nEqs, cupdlp_int **csc_beg, cupdlp_int **csc_idx, cupdlp_float **csc_val,
                cupdlp_float **rhs, cupdlp_float **lower, cupdlp_float **upper, cupdlp_float *offset,
                cupdlp_int *nCols_origin);

cupdlp_int formulateLP_new(void *model, cupdlp_float **cost, cupdlp_int *nCols, cupdlp_int *nRows,
                    cupdlp_int *nnz, cupdlp_int *nEqs, cupdlp_int **csc_beg, cupdlp_int **csc_idx,
                    cupdlp_float **csc_val, cupdlp_float **rhs, cupdlp_float **lower,
                    cupdlp_float **upper, cupdlp_float *offset, cupdlp_int *nCols_origin,
                    cupdlp_int **constraint_new_idx);

void loadMps(void *model, const char *filename);

void deleteModel(void *model);
void *createModel();

void *createPresolve();
void deletePresolve(void *presolve);
void *presolvedModel(void *presolve, void *model);

#ifdef __cplusplus
}
#endif
#endif  // CUPDLP_WRAPPER_CLP_H
