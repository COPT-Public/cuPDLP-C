
#ifndef CUPDLP_H
#define CUPDLP_H

#include "cupdlp_cs.h"
#include "cupdlp_defs.h"
#include "cupdlp_linalg.h"
#include "cupdlp_mps.h"
#include "cupdlp_proj.h"
#include "cupdlp_restart.h"
// #include "cupdlp_scaling.h"
#include "cupdlp_solver.h"
#include "cupdlp_step.h"
#include "cupdlp_utils.h"
#include "glbopts.h"
// #include "cupdlp_scaling_new.h"
#include "cupdlp_scaling_cuda.h"

#if !(CUPDLP_CPU)
#include "cuda/cupdlp_cudalinalg.cuh"
#endif
#endif
