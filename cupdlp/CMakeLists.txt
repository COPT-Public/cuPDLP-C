

set(CUPDLP_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/cupdlp/")
file(GLOB CUPDLP_INCLUDE_HEADERS
        "${CUPDLP_INCLUDE_DIR}/*.h"
)
file(GLOB CUPDLP_SOURCE_FILES
        "${CUPDLP_INCLUDE_DIR}/*.c"
)

add_library(cupdlp SHARED
        ${CUPDLP_INCLUDE_HEADERS}
        ${CUPDLP_SOURCE_FILES}
)
target_compile_definitions(cupdlp
        PUBLIC
        # If the debug configuration pass the DEBUG define to the compiler
        $<$<CONFIG:Debug>:-DCUPDLP_DEBUG=1>
)

if (${CUDA_LIBRARY-NOTFOUND})
    message(NOTICE "- CPU version PDLP")
    target_compile_definitions(cupdlp
            PUBLIC
            -DCUPDLP_CPU=1
    )
    target_link_libraries(cupdlp m)
else()
    add_subdirectory(cuda)
    message(NOTICE "- GPU version PDLP")
    target_include_directories(cupdlp PUBLIC "/usr/local/cuda/include")
    target_link_libraries(cupdlp PRIVATE cudalin ${CUDA_LIBRARY} m)
    set_target_properties(cupdlp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif ()
