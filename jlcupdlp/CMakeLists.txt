enable_language(CXX)

message(NOTICE "--------------------- cuPDLP Interface -----------------------")
message("building julia interface for cuPDLP")

add_library(jlcupdlp SHARED
        jlcupdlp.h jlcupdlp.c
)

if (${CUDA_LIBRARY-NOTFOUND})
    target_link_libraries(jlcupdlp PUBLIC cupdlp wrapper_highs m)
else ()
    target_link_libraries(jlcupdlp PUBLIC cupdlp wrapper_highs ${CUDA_LIBRARY} m)
endif ()
