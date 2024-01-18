enable_language(CXX)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})


message(NOTICE "--------------------- cuPDLP Interface -----------------------")
message("building python interface for cuPDLP")


add_subdirectory(pybind11)

pybind11_add_module(pycupdlp cupdlp_bindings.cpp)

set_target_properties(pycupdlp
        PROPERTIES
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED ON
)
if (${CUDA_LIBRARY-NOTFOUND})
    target_link_libraries(pycupdlp PUBLIC cupdlp wrapper_highs m)
else ()
    target_link_libraries(pycupdlp PUBLIC cupdlp wrapper_highs ${CUDA_LIBRARY} m)
endif ()