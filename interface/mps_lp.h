#ifndef mps_lp_cuda_h
#define mps_lp_cuda_h

#include "../cupdlp/cupdlp.h"
#ifdef __cplusplus
extern "C" {
#endif

cupdlp_retcode data_alloc(CUPDLPdata *data, cupdlp_int nRows, cupdlp_int nCols,
                          void *matrix, CUPDLP_MATRIX_FORMAT src_matrix_format,
                          CUPDLP_MATRIX_FORMAT dst_matrix_format);

cupdlp_retcode problem_alloc(
    CUPDLPproblem *prob, cupdlp_int nRows, cupdlp_int nCols, cupdlp_int nEqs,
    cupdlp_float *cost, cupdlp_float offset, cupdlp_float sense_origin,
    void *matrix, CUPDLP_MATRIX_FORMAT src_matrix_format,
    CUPDLP_MATRIX_FORMAT dst_matrix_format, cupdlp_float *rhs,
    cupdlp_float *lower, cupdlp_float *upper, cupdlp_float *alloc_matrix_time,
    cupdlp_float *copy_vec_time);
// problem and data part
cupdlp_retcode problem_create(CUPDLPproblem **prob);
void freealldata(int *Aeqp, int *Aeqi, double *Aeqx, int *Aineqp, int *Aineqi,
                 double *Aineqx, int *colUbIdx, double *colUbElem, double *rhs,
                 double *cost, double *x, double *s, double *t, double *sx,
                 double *ss, double *st, double *y, double *lower,
                 double *upper);

void print_script_usage();
#ifdef __cplusplus
}
#endif
#endif