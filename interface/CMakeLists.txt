enable_language(CXX)
#############################################
# enable LP parsers
#############################################
add_library(wrapper_lp SHARED
        mps_lp.h mps_lp.c
)
target_link_libraries(
        wrapper_lp PUBLIC cupdlp
)
if (${CUDA_LIBRARY-NOTFOUND})
    message(NOTICE "- CPU version PDLP")
    target_compile_definitions(wrapper_lp
            PUBLIC
            -DCUPDLP_CPU
    )
endif()
#############################################
# enable Coin-OR CLP input targets
#############################################

message(NOTICE "--------------------- cuPDLP   Coin  CONFIG -----------------------")
include(../FindCLPConf.cmake)
add_library(wrapper_clp SHARED
        mps_lp.h mps_lp.c
        wrapper_clp.cpp ${CLP_HEADER_FILES}
        wrapper_clp.h)
target_include_directories(
        wrapper_clp PRIVATE
        ${CLP_INCLUDE_DIR}
        ${COIN_INCLUDE_DIR}
        "$ENV{CLP_HOME}/include/clp/coin/"
        "$ENV{COIN_HOME}/include/coinutils/coin/"
        )
target_link_libraries(
        wrapper_clp PUBLIC cupdlp
        ${CLP_LIBRARY} ${CLP_LIBRARY_OSICLP} ${CLP_LIBRARY_COIN}
)
if (${CUDA_LIBRARY-NOTFOUND})
    message(NOTICE "- CPU version PDLP")
    target_compile_definitions(wrapper_clp
            PUBLIC
            -DCUPDLP_CPU
    )
endif()


#############################################
# cuPDLP-C binary using CLP as parsers
#############################################
add_executable(plc 
        ${CLP_HEADER_FILES}
        mps_clp.c
)
target_include_directories(
        plc PRIVATE 
        ${CLP_INCLUDE_DIR} 
        ${COIN_INCLUDE_DIR}
        "$ENV{CLP_HOME}/include/clp/coin/"
        "$ENV{COIN_HOME}/include/coinutils/coin/"
        )
target_link_libraries(
        plc PUBLIC
        wrapper_clp
        ${CLP_LIBRARY}
        ${CLP_LIBRARY_OSICLP}
        ${CLP_LIBRARY_COIN}
)


if (${CUDA_LIBRARY-NOTFOUND})
    target_link_libraries(plc PRIVATE cupdlp m)
else ()
    target_link_libraries(plc PRIVATE cupdlp ${CUDA_LIBRARY} m)
endif ()
if (${CUDA_LIBRARY-NOTFOUND})
    add_custom_target(
            deploy
            COMMAND mv "${CMAKE_BINARY_DIR}/bin/plc" "${CMAKE_BINARY_DIR}/bin/plccpu"
            DEPENDS plc plc
    )
else()
    add_custom_target(
            deploy
            COMMAND mv "${CMAKE_BINARY_DIR}/bin/plc" "${CMAKE_BINARY_DIR}/bin/plcgpu"
            DEPENDS plc plc
    )
endif ()