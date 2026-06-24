# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Author: Jeff Daily <jeff.daily@amd.com>
#
# Locate the ROCm/HIP toolchain and the hipBLAS/hipSPARSE (rocBLAS/rocSPARSE)
# libraries used by the AMD GPU build.

message(NOTICE "Finding HIP/ROCm environment")
message(NOTICE "    - ROCM_PATH: $ENV{ROCM_PATH}")

# Default ROCM_PATH if not set
if(NOT DEFINED ENV{ROCM_PATH})
  set(ENV{ROCM_PATH} "/opt/rocm")
endif()

set(ROCM_PATH $ENV{ROCM_PATH})

# enable_language(HIP) auto-detects the host GPU arch (and errors on a
# no-GPU build host); pass -DCMAKE_HIP_ARCHITECTURES=... to override.
enable_language(HIP)
message(NOTICE "    - CMAKE_HIP_ARCHITECTURES: ${CMAKE_HIP_ARCHITECTURES}")

# Find hipBLAS
find_library(HIP_LIBRARY_BLAS
    NAMES hipblas
    HINTS "${ROCM_PATH}/lib"
    REQUIRED
)

# Find hipSPARSE
find_library(HIP_LIBRARY_SPARSE
    NAMES hipsparse
    HINTS "${ROCM_PATH}/lib"
    REQUIRED
)

# Find amdhip64 runtime
find_library(HIP_LIBRARY_RT
    NAMES amdhip64
    HINTS "${ROCM_PATH}/lib"
    REQUIRED
)

# Find rocBLAS (hipBLAS backend)
find_library(HIP_LIBRARY_ROCBLAS
    NAMES rocblas
    HINTS "${ROCM_PATH}/lib"
    REQUIRED
)

# Find rocSPARSE (hipSPARSE backend)
find_library(HIP_LIBRARY_ROCSPARSE
    NAMES rocsparse
    HINTS "${ROCM_PATH}/lib"
    REQUIRED
)

set(HIP_LIBRARY ${HIP_LIBRARY_RT} ${HIP_LIBRARY_BLAS} ${HIP_LIBRARY_SPARSE} ${HIP_LIBRARY_ROCBLAS} ${HIP_LIBRARY_ROCSPARSE})
message(NOTICE "    - HIP Libraries: ${HIP_LIBRARY}")

# Set include directories
set(HIP_INCLUDE_DIRS
    "${ROCM_PATH}/include"
    "${ROCM_PATH}/include/hipblas"
    "${ROCM_PATH}/include/hipsparse"
)
