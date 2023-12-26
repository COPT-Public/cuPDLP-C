#ifndef lp_mps_h
#define lp_mps_h

#include "cupdlp_defs.h"

/* Implement an LP mps file reader */
cupdlp_int cupdlpMpsRead(char *fname, char *name, int *pnRow, int *pnEqRow,
                         int *pnInEqRow, int *pnCol, int *pnElem,
                         int **pfullMatBeg, int **pfullMatIdx,
                         cupdlp_float **pfullMatElem, int **peqMatBeg,
                         int **peqMatIdx, cupdlp_float **peqMatElem,
                         int **pIneqMatBeg, int **pIneqMatIdx,
                         cupdlp_float **pIneqMatElem, cupdlp_float **prowRHS,
                         cupdlp_float **pcolObj, int *pnColUb, int **pcolUbIdx,
                         cupdlp_float **pcolUbElem);

#endif /* lp_mps_h */
