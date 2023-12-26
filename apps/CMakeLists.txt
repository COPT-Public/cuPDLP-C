find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})


message(NOTICE "--------------------- cuPDLP Applications -----------------------")
message("building applications using cuPDLP")


add_executable(onlinelp onlinelp.cpp onlinelp.h ../interface/mps_lp.h ../interface/mps_lp.c)
target_link_libraries(
        onlinelp PUBLIC
        cupdlp
)