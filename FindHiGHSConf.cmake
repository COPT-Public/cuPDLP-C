# (c) Tianhao Liu
set(HiGHS_LIBRARY-NOTFOUND, OFF)
message(NOTICE "Finding HiGHS environment")
message(NOTICE "    - HiGHS Home detected at $ENV{HiGHS_HOME}")
set(CMAKE_HiGHS_PATH "$ENV{HiGHS_HOME}")
set(HiGHS_HEADER_LIB "$ENV{HiGHS_HOME}/build/lib")
set(HiGHS_INCLUDE_DIR "$ENV{HiGHS_HOME}/src")

file(GLOB_RECURSE 
        HiGHS_HEADER_FILES 
        "${HiGHS_INCLUDE_DIR}/*.h" 
        "${HiGHS_INCLUDE_DIR}/*.hpp" 
        "$ENV{HiGHS_HOME}/build/*.h"
        "$ENV{HiGHS_HOME}/build/*.hpp"
)

find_library(HiGHS_LIBRARY
        NAMES highs
        PATHS "${HiGHS_HEADER_LIB}"
        REQUIRED
        NO_DEFAULT_PATH
)

message(NOTICE 
        "    - HiGHS Libraries detected at ${HiGHS_LIBRARY}")
message(NOTICE
        "    - HiGHS include dir at ${HiGHS_INCLUDE_DIR}")
