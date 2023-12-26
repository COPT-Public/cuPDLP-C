# (c) Chuwen Zhang
set(CLP_LIBRARY-NOTFOUND, OFF)
message(NOTICE "Finding Coin environment")
message(NOTICE "    - Coin Home detected at $ENV{COIN_HOME}")
message(NOTICE "    - Clp  Home detected at $ENV{CLP_HOME}")
set(CMAKE_CLP_PATH "$ENV{CLP_HOME}")
set(CLP_HEADER_LIB "$ENV{CLP_HOME}/lib/")
set(CLP_INCLUDE_DIR "$ENV{CLP_HOME}/include/coin-or")
set(CMAKE_COIN_PATH "$ENV{COIN_HOME}")
set(COIN_HEADER_LIB "$ENV{COIN_HOME}/lib/")
set(COIN_INCLUDE_DIR "$ENV{COIN_HOME}/include/coin-or")

file(GLOB 
        CLP_HEADER_FILES 
        "${CLP_INCLUDE_DIR}/*.h" 
        "${CLP_INCLUDE_DIR}/*.hpp" 
        "$ENV{CLP_HOME}/include/clp/coin/*.h"
        "$ENV{CLP_HOME}/include/clp/coin/*.hpp"
        "${COIN_INCLUDE_DIR}/*.h"
        "${COIN_INCLUDE_DIR}/*.hpp"
        "$ENV{COIN_HOME}/include/coinutils/coin/*.h"
        "$ENV{COIN_HOME}/include/coinutils/coin/*.hpp"
)

find_library(CLP_LIBRARY
        NAMES Clp
        PATHS "${CLP_HEADER_LIB}"
        REQUIRED
        NO_DEFAULT_PATH
)
find_library(CLP_LIBRARY_COIN
        NAMES CoinUtils
        PATHS "${COIN_HEADER_LIB}"
        REQUIRED
        NO_DEFAULT_PATH
)
find_library(CLP_LIBRARY_OSICLP
        NAMES OsiClp
        PATHS "${CLP_HEADER_LIB}"
        REQUIRED
        NO_DEFAULT_PATH
)

message(        NOTICE 
        "    - CLP Libraries detected at
        ${CLP_LIBRARY}
        ${CLP_LIBRARY_COIN}
        ${CLP_LIBRARY_OSICLP}")
message(NOTICE "    - CLP include dir at ${CLP_INCLUDE_DIR}")
