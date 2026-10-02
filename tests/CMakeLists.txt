find_package(Python3 3.8 REQUIRED COMPONENTS Interpreter)
foreach(_case bounded equality mixed_bounds compressed_mps)
    add_test(NAME installation_${_case}
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/run_tests.py"
            --solver "$<TARGET_FILE:plc>" --case "${_case}")
    set_tests_properties(installation_${_case} PROPERTIES TIMEOUT 60 LABELS "installation")
endforeach()
