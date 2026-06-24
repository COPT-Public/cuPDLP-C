set(CUDA_SOURCES
    ${CUPDLP_INCLUDE_DIR}/cuda/cupdlp_cuda_kernels.cu
    ${CUPDLP_INCLUDE_DIR}/cuda/cupdlp_cudalinalg.cu
)

set(CUDA_HEADERS
    ${CUPDLP_INCLUDE_DIR}/cuda/cupdlp_cuda_kernels.cuh
    ${CUPDLP_INCLUDE_DIR}/cuda/cupdlp_cudalinalg.cuh
    ${CUPDLP_INCLUDE_DIR}/cuda/cuda_to_hip.h
)

if (${USE_HIP} STREQUAL "ON")
    # HIP/ROCm build
    enable_language(HIP)
    set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE HIP)

    add_library(cudalin SHARED
        ${CUDA_SOURCES}
        ${CUDA_HEADERS}
    )
    target_compile_definitions(cudalin PUBLIC USE_HIP)
    set_target_properties(cudalin PROPERTIES
        HIP_SEPARABLE_COMPILATION ON
        HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}"
        WINDOWS_EXPORT_ALL_SYMBOLS ON
    )
    target_include_directories(cudalin PUBLIC ${HIP_INCLUDE_DIRS})
    target_compile_definitions(cudalin
            PUBLIC
            $<$<CONFIG:Debug>:-DCUPDLP_DEBUG=1>
    )
    target_link_libraries(cudalin ${HIP_LIBRARY} $<$<NOT:$<PLATFORM_ID:Windows>>:m>)
else()
    # CUDA build
    enable_language(CXX CUDA)

    add_library(cudalin SHARED
        ${CUDA_SOURCES}
        ${CUDA_HEADERS}
    )
    set_target_properties(cudalin PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_include_directories(cudalin PUBLIC "/usr/local/cuda/include")
    target_compile_definitions(cudalin
            PUBLIC
            $<$<CONFIG:Debug>:-DCUPDLP_DEBUG=1>
    )
    target_link_libraries(cudalin ${CUDA_LIBRARY} $<$<NOT:$<PLATFORM_ID:Windows>>:m>)
endif()

# add tests
add_executable(testcudalin test_cuda_linalg.c)
add_executable(testcublas test_cublas.c)

if (${USE_HIP} STREQUAL "ON")
    set_target_properties(testcudalin PROPERTIES HIP_SEPARABLE_COMPILATION ON)
    target_link_libraries(testcudalin PRIVATE cudalin ${HIP_LIBRARY})
    set_target_properties(testcublas PROPERTIES HIP_SEPARABLE_COMPILATION ON)
    target_link_libraries(testcublas PRIVATE cudalin ${HIP_LIBRARY})
else()
    set_target_properties(testcudalin PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(testcudalin PRIVATE cudalin ${CUDA_LIBRARY})
    set_target_properties(testcublas PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(testcublas PRIVATE cudalin ${CUDA_LIBRARY})
endif()
