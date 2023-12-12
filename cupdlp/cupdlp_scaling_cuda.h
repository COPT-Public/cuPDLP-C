//
// Created by LJS on 23-11-30.
//

#ifndef CUPDLP_SCALING_CUDA_H
#define CUPDLP_SCALING_CUDA_H

#include "cupdlp_defs.h"
#include "glbopts.h"

cupdlp_retcode PDHG_Scale_Data_cuda(CUPDLPcsc *csc, cupdlp_int ifScaling,
                                    CUPDLPscaling *scaling, cupdlp_float *cost,
                                    cupdlp_float *lower, cupdlp_float *upper,
                                    cupdlp_float *rhs);

cupdlp_retcode Init_Scaling(CUPDLPscaling *scaling, cupdlp_int ncols,
                            cupdlp_int nrows, cupdlp_float *cost,
                            cupdlp_float *rhs);

#endif  // CUPDLP_CUPDLP_SCALING_H
