#ifndef CUPDLP_CS_H
#define CUPDLP_CS_H

#include "cupdlp_defs.h"

/* sparse matrix in column-oriented form used in reading mps*/
typedef struct cupdlp_cs_sparse {
  cupdlp_int nzmax;
  cupdlp_int m;     /* number of rows */
  cupdlp_int n;     /* number of columns */
  cupdlp_int *p;    /* column pointers (size n+1) or col indices (size nzmax) */
  cupdlp_int *i;    /* row indices, size nzmax */
  cupdlp_float *x; /* numerical values, size nzmax */
  cupdlp_int nz;    /* # of entries in triplet matrix, -1 for compressed-col */
} cupdlp_dcs;

cupdlp_int cupdlp_dcs_entry(cupdlp_dcs *T, cupdlp_int i, cupdlp_int j, cupdlp_float x);
cupdlp_dcs *cupdlp_dcs_compress(const cupdlp_dcs *T);
cupdlp_float cupdlp_dcs_norm(const cupdlp_dcs *A);
cupdlp_int cupdlp_dcs_print(const cupdlp_dcs *A, cupdlp_int brief);

/* utilities */
void *_dcs_calloc(cupdlp_int n, size_t size);
void *cupdlp_dcs_free(void *p);
void *cupdlp_dcs_realloc(void *p, cupdlp_int n, size_t size, cupdlp_int *ok);
cupdlp_dcs *cupdlp_dcs_spalloc(cupdlp_int m, cupdlp_int n, cupdlp_int nzmax, cupdlp_int values, cupdlp_int t);
cupdlp_dcs *cupdlp_dcs_spfree(cupdlp_dcs *A);
cupdlp_int cupdlp_dcs_sprealloc(cupdlp_dcs *A, cupdlp_int nzmax);
void *cupdlp_dcs_malloc(cupdlp_int n, size_t size);

/* utilities */
cupdlp_float cupdlp_dcs_cumsum(cupdlp_int *p, cupdlp_int *c, cupdlp_int n);
cupdlp_dcs *cupdlp_dcs_done(cupdlp_dcs *C, void *w, void *x, cupdlp_int ok);
cupdlp_int *cupdlp_dcs_idone(cupdlp_int *p, cupdlp_dcs *C, void *w, cupdlp_int ok);
cupdlp_dcs *cupdlp_dcs_transpose(const cupdlp_dcs *A, cupdlp_int values);

#define IS_CSC(A) (A && (A->nz == -1))
#define IS_TRIPLET(A) (A && (A->nz >= 0))
/*--------------------------------------------------------------------------*/

#endif
