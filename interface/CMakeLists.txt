enable_language(CXX)
#############################################
# enable LP parsers
#############################################
add_library(wrapper_lp SHARED
        mps_lp.h mps_lp.c
)
set_target_properties(wrapper_lp PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
target_link_libraries(
        wrapper_lp PUBLIC cupdlp
)
if (${CUDA_LIBRARY-NOTFOUND})
    message(NOTICE "- CPU version PDLP")
    target_compile_definitions(wrapper_lp
            PUBLIC
            -DCUPDLP_CPU=1
    )
elseif (${USE_HIP} STREQUAL "ON")
    target_compile_definitions(wrapper_lp PUBLIC USE_HIP)
    target_include_directories(wrapper_lp PUBLIC ${HIP_INCLUDE_DIRS})
    target_link_libraries(wrapper_lp PRIVATE ${HIP_LIBRARY})
endif()

#############################################
# enable HiGHS input targets
#############################################

message(NOTICE "--------------------- cuPDLP  HiGHS  CONFIG -----------------------")
include(../FindHiGHSConf.cmake)
add_library(wrapper_highs SHARED
        mps_lp.h mps_lp.c
        wrapper_highs.cpp ${HiGHS_HEADER_FILES}
        wrapper_highs.h)
set_target_properties(wrapper_highs PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

target_include_directories(
        wrapper_highs PUBLIC
        ${HiGHS_INCLUDE_DIR}
)
target_link_libraries(
        wrapper_highs PUBLIC cupdlp
        ${HiGHS_LIBRARY}
)
if (${CUDA_LIBRARY-NOTFOUND})
    message(NOTICE "- CPU version PDLP")
    target_compile_definitions(wrapper_highs
            PUBLIC
            -DCUPDLP_CPU=1
    )
elseif (${USE_HIP} STREQUAL "ON")
    target_compile_definitions(wrapper_highs PUBLIC USE_HIP)
    target_include_directories(wrapper_highs PUBLIC ${HIP_INCLUDE_DIRS})
    target_link_libraries(wrapper_highs PRIVATE ${HIP_LIBRARY})
endif()


#############################################
# cuPDLP-C binary using HiGHS as parsers
#############################################
add_executable(plc 
        ${HiGHS_HEADER_FILES}
        mps_highs.c
)
target_include_directories(
        plc PRIVATE
        ${HiGHS_INCLUDE_DIR}
        )
target_link_libraries(
        plc PUBLIC
        wrapper_highs
        ${HiGHS_LIBRARY}
)


if (${CUDA_LIBRARY-NOTFOUND})
    target_link_libraries(plc PRIVATE cupdlp $<$<NOT:$<PLATFORM_ID:Windows>>:m>)
elseif (${USE_HIP} STREQUAL "ON")
    target_link_libraries(plc PRIVATE cupdlp ${HIP_LIBRARY} $<$<NOT:$<PLATFORM_ID:Windows>>:m>)
else ()
    target_link_libraries(plc PRIVATE cupdlp ${CUDA_LIBRARY} $<$<NOT:$<PLATFORM_ID:Windows>>:m>)
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