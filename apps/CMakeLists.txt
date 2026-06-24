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
    target_compile_definitions(onlinelp PUBLIC -DCUPDLP_CPU=1)
elseif (${USE_HIP} STREQUAL "ON")
    target_link_libraries(onlinelp PUBLIC cupdlp ${HIP_LIBRARY} m)
    target_compile_definitions(onlinelp PUBLIC USE_HIP)
    target_include_directories(onlinelp PUBLIC ${HIP_INCLUDE_DIRS})
else ()
    target_link_libraries(onlinelp PUBLIC cupdlp ${CUDA_LIBRARY} m)
endif ()