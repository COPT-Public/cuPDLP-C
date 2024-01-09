# (c) Tianhao Liu
set(HiGHS_LIBRARY-NOTFOUND, OFF)
message(NOTICE "Finding HiGHS environment")
message(NOTICE "    - HiGHS Home detected at $ENV{HIGHS_HOME}")

set(HIGHS_DIR "$ENV{HIGHS_HOME}/lib/cmake/highs")

find_package(HIGHS REQUIRED)
find_package(Threads REQUIRED)