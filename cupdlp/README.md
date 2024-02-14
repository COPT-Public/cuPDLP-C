# cuPDLP-C observations

This directory contains files from [cuPDLP-C v0.3.0](https://github.com/COPT-Public/cuPDLP-C/tree/v0.3.0). Below are some issues experienced when integrating them into HiGHS.

## Preprocessing issue

The following line is not recognised by g++, 

> #if !(CUPDLP_CPU)

so I've had to replace all ocurrences by

> #ifndef CUPDLP_CPU

This yields a compiler warning about "extra tokens at end of #ifndef
directive" in the case of the following, but it's not a problem for
now, as CUPDLP_CPU is set

> #ifndef CUPDLP_CPU & USE_KERNELS

## cmake issues

CUPDLP_CPU and CUPDLP_DEBUG should both set when building. However, they are not recognised so are forced by the following lines in cupdlp_defs.h

#define CUPDLP_CPU

#define CUPDLP_DEBUG (1)

## Use of macro definitions within C

Although the macro definitions in [glbopts.h](https://github.com/ERGO-Code/HiGHS/blob/add-pdlp/src/pdlp/cupdlp/glbopts.h) are fine when used in C under Linux, they cause the following compiler errors on Windows.

> error C2146: syntax error: missing ';' before identifier 'calloc' (or 'malloc')

In HiGHS, all the macros using `typeof` have been replaced by multiple type-specific macros

## Problem with sys/time.h

The HiGHS branch add-pdlp compiles and runs fine on @jajhall's Linux machine, but CI tests on GitHub fail utterly due to `sys/time.h` not being found. Until this is fixed, or HiGHS passes its own timer for use within `cuPDLP-c`, timing within `cuPDLP-c` can be disabled using the compiler directive `CUPDLP_TIMER`. By default this is defined, so the `cuPDLP-c` is retained.

## Termination of cuPDLP-C

cuPDLP-C terminates when either the current or averaged iterates satisfy primal/dual feasibility (and a duality gap criterion), using a 2-norm measure relative to the size of the RHS/costs. HiGHS assesses primal/dual feasibility using a infinity-norm absolute measure. Thus the cuPDLP-C result frequently fails to satisfy HiGHS primal/dual feasibility. To get around this, `iInfNormAbsLocalTermination` has been introduced into cuPDLP-C. 

By default, `iInfNormAbsLocalTermination` is false, so that the original cuPDLP-C termination criteria are used.

When `iInfNormAbsLocalTermination` is true, cuPDLP-C terminates only when primal/dual feasibility is satisfied for the infinity-norm absolute measure of the current iterate, so that HiGHS primal/dual feasibility is satisfied. 

## Controlling the `cuPDLP-c` logging

As a research code, `cuPDLP-c` naturally produces a lot of logging output. HiGHS must be able to run with less logging output, or completely silently. This is achieved using the `nLogLevel` parameter in `cuPDLP-c`. 

By default, `nLogLevel` is 2, so all the original `cuPDLP-c` logging is produced.

* If `nLogLevel` is 1, then the `cuPDLP-c` logging is less verbose 
* If `nLogLevel` is 0, then there is no `cuPDLP-c` logging

A related issue is the use of `fp` and `fp_sol`. HiGHS won't be using these, so sets them to null pointers. `cuPDLP-c` already doesn't print the solution if `fp_sol` is a null pointer, so the call to `writeJson(fp, pdhg);` is now conditional on `if (fp)`. 

## Returning the iteration count

The `cuPDLP-c` iteration count is held in `pdhg->timers->nIter`, but `pdhg` is destroyed in `LP_SolvePDHG`, so `cupdlp_int* num_iter` has been added to the parameter list of this method.




