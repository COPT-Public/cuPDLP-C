enable_language(CXX)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})


message(NOTICE "--------------------- cuPDLP Applications -----------------------")
message("building applications using cuPDLP")


add_executable(onlinelp
        onlinelp.cpp onlinelp.h
        ../interface/mps_lp.h
        ../interface/mps_lp.c
)
set_target_properties(onlinelp
        PROPERTIES
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED ON
)
if (${CUDA_LIBRARY-NOTFOUND})
    target_link_libraries(onlinelp PUBLIC cupdlp m)
else ()
    target_compile_definitions(wrapper_clp
            PUBLIC
            -DCUPDLP_CPU=1
    )
    target_link_libraries(onlinelp PUBLIC cupdlp ${CUDA_LIBRARY} m)
endif ()