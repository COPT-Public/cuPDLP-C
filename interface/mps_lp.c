#include "mps_lp.h"

void print_script_usage() {
  printf("Test Script User Parameters:\n");
  printf("\n");

  printf("  -h: print this helper\n");
  printf("\n");
  printf("  -fname <str>    : path for .mps or .mps.gz LP file\n");
  printf("\n");
  printf("  -out <str>      : path for .json output file\n");
  printf("\n");
  printf("  -outSol <str>   : path for .json output solution file\n");
  printf("\n");
  printf(
      "  -savesol <bool> : whether to write solution to .json solution file\n");
  printf("\n");
  // printf(
  //     "  -ifPre <bool>   : whether to use presolver (and thus
  //     postsolver)\n");
  // printf("\n");
}

void freealldata(int *Aeqp, int *Aeqi, double *Aeqx, int *Aineqp, int *Aineqi,
                 double *Aineqx, int *colUbIdx, double *colUbElem, double *rhs,
                 double *cost, double *x, double *s, double *t, double *sx,
                 double *ss, double *st, double *y, double *lower,
                 double *upper) {
  if (Aeqp) {
    cupdlp_free(Aeqp);
  }

  if (Aeqi) {
    cupdlp_free(Aeqi);
  }

  if (Aeqx) {
    cupdlp_free(Aeqx);
  }

  if (Aineqp) {
    cupdlp_free(Aineqp);
  }

  if (Aineqi) {
    cupdlp_free(Aineqi);
  }

  if (Aineqx) {
    cupdlp_free(Aineqx);
  }

  if (colUbIdx) {
    cupdlp_free(colUbIdx);
  }

  if (colUbElem) {
    cupdlp_free(colUbElem);
  }

  if (rhs) {
    cupdlp_free(rhs);
  }

  if (cost) {
    cupdlp_free(cost);
  }

  if (x) {
    cupdlp_free(x);
  }

  if (s) {
    cupdlp_free(s);
  }

  if (t) {
    cupdlp_free(t);
  }

  if (sx) {
    cupdlp_free(sx);
  }

  if (ss) {
    cupdlp_free(ss);
  }

  if (st) {
    cupdlp_free(st);
  }

  if (y) {
    cupdlp_free(y);
  }

  if (lower) {
    cupdlp_free(lower);
  }

  if (upper) {
    cupdlp_free(upper);
  }
}

cupdlp_retcode problem_create(CUPDLPproblem **prob) {
  cupdlp_retcode retcode = RETCODE_OK;

  CUPDLP_INIT(*prob, 1);

exit_cleanup:
  return retcode;
}

cupdlp_retcode data_alloc(CUPDLPdata *data, cupdlp_int nRows, cupdlp_int nCols,
                          void *matrix, CUPDLP_MATRIX_FORMAT src_matrix_format,
                          CUPDLP_MATRIX_FORMAT dst_matrix_format) {
  cupdlp_retcode retcode = RETCODE_OK;

  data->nRows = nRows;
  data->nCols = nCols;
  data->matrix_format = dst_matrix_format;
  data->dense_matrix = cupdlp_NULL;
  data->csr_matrix = cupdlp_NULL;
  data->csc_matrix = cupdlp_NULL;
#if CUPDLP_CPU
  data->device = CPU;
#else
  data->device = SINGLE_GPU;
#endif

  switch (dst_matrix_format) {
    case DENSE:
      CUPDLP_CALL(dense_create(&data->dense_matrix));
      CUPDLP_CALL(dense_alloc_matrix(data->dense_matrix, nRows, nCols, matrix,
                                     src_matrix_format));
      break;
    case CSR:
      CUPDLP_CALL(csr_create(&data->csr_matrix));
      CUPDLP_CALL(csr_alloc_matrix(data->csr_matrix, nRows, nCols, matrix,
                                   src_matrix_format));
      break;
    case CSC:
      CUPDLP_CALL(csc_create(&data->csc_matrix));
      CUPDLP_CALL(csc_alloc_matrix(data->csc_matrix, nRows, nCols, matrix,
                                   src_matrix_format));
      break;
    case CSR_CSC:
      CUPDLP_CALL(csc_create(&data->csc_matrix));
      CUPDLP_CALL(csc_alloc_matrix(data->csc_matrix, nRows, nCols, matrix,
                                   src_matrix_format));
      CUPDLP_CALL(csr_create(&data->csr_matrix));
      CUPDLP_CALL(csr_alloc_matrix(data->csr_matrix, nRows, nCols, matrix,
                                   src_matrix_format));
      break;
    default:
      break;
  }
  // currently, only supprot that input matrix is CSC, and store both CSC and
  // CSR data->csc_matrix = matrix;

exit_cleanup:
  return retcode;
}

cupdlp_retcode problem_alloc(
    CUPDLPproblem *prob, cupdlp_int nRows, cupdlp_int nCols, cupdlp_int nEqs,
    cupdlp_float *cost, cupdlp_float offset, cupdlp_float sense_origin,
    void *matrix, CUPDLP_MATRIX_FORMAT src_matrix_format,
    CUPDLP_MATRIX_FORMAT dst_matrix_format, cupdlp_float *rhs,
    cupdlp_float *lower, cupdlp_float *upper, cupdlp_float *alloc_matrix_time,
    cupdlp_float *copy_vec_time) {
  cupdlp_retcode retcode = RETCODE_OK;
  prob->nRows = nRows;
  prob->nCols = nCols;
  prob->nEqs = nEqs;
  prob->data = cupdlp_NULL;
  prob->cost = cupdlp_NULL;
  prob->offset = offset;
  prob->sense_origin = sense_origin;
  prob->rhs = cupdlp_NULL;
  prob->lower = cupdlp_NULL;
  prob->upper = cupdlp_NULL;

  cupdlp_float begin = getTimeStamp();

  CUPDLP_INIT(prob->data, 1);
  CUPDLP_INIT_VEC(prob->cost, nCols);
  CUPDLP_INIT_VEC(prob->rhs, nRows);
  CUPDLP_INIT_VEC(prob->lower, nCols);
  CUPDLP_INIT_VEC(prob->upper, nCols);
  CUPDLP_INIT_ZERO_VEC(prob->hasLower, nCols);
  CUPDLP_INIT_ZERO_VEC(prob->hasUpper, nCols);

  CUPDLP_CALL(data_alloc(prob->data, nRows, nCols, matrix, src_matrix_format,
                         dst_matrix_format));
  *alloc_matrix_time = getTimeStamp() - begin;

  prob->data->csc_matrix->MatElemNormInf = infNorm(
      ((CUPDLPcsc *)matrix)->colMatElem, ((CUPDLPcsc *)matrix)->nMatElem);

  begin = getTimeStamp();
  CUPDLP_COPY_VEC(prob->cost, cost, cupdlp_float, nCols);
  CUPDLP_COPY_VEC(prob->rhs, rhs, cupdlp_float, nRows);
  CUPDLP_COPY_VEC(prob->lower, lower, cupdlp_float, nCols);
  CUPDLP_COPY_VEC(prob->upper, upper, cupdlp_float, nCols);
  *copy_vec_time = getTimeStamp() - begin;

  // todo, translate to cuda
  // for (int i = 0; i < nCols; i++)
  // {
  //     prob->hasLower[i] = (lower[i] > -INFINITY);
  //     prob->hasUpper[i] = (upper[i] < +INFINITY);
  // }
  // cupdlp_haslb(prob->hasLower, lower, -INFINITY, nCols);
  // cupdlp_hasub(prob->hasUpper, upper, +INFINITY, nCols);

  cupdlp_haslb(prob->hasLower, prob->lower, -INFINITY, nCols);
  cupdlp_hasub(prob->hasUpper, prob->upper, +INFINITY, nCols);

  // TODO: cal dMaxCost, dMaxRhs, dMaxRowBound

exit_cleanup:
  return retcode;
}
