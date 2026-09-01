add_executable(cudaee_unit unit/test_core.cpp)
target_link_libraries(cudaee_unit PRIVATE cudaee_core)
target_compile_definitions(cudaee_unit PRIVATE CUDAEE_TEST_TMP_DIR="${CMAKE_CURRENT_BINARY_DIR}/tmp")
target_compile_options(cudaee_unit PRIVATE -Wall -Wextra -Wconversion -Wshadow -Wformat=2)

add_test(NAME unit.core COMMAND cudaee_unit)

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
  add_test(
    NAME exhaustive.small_instances
    COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/exhaustive/check_small_instances.py"
      "$<TARGET_FILE:cudaee>" "${CMAKE_CURRENT_BINARY_DIR}/tmp/exhaustive")
  set_tests_properties(exhaustive.small_instances PROPERTIES LABELS "exhaustive;correctness")
endif()

set(PR299_TSP "${PROJECT_SOURCE_DIR}/third_party/ElimTSP/data/pr299.tsp")
set(PR299_EDGES "${PROJECT_SOURCE_DIR}/third_party/ElimTSP/data/pr299.edg")
if(EXISTS "${PR299_TSP}" AND EXISTS "${PR299_EDGES}")
  add_test(
    NAME integration.pr299_cpu_eliminate
    COMMAND cudaee gpu-eliminate
      --tsp "${PR299_TSP}"
      --edges "${PR299_EDGES}"
      --output "${CMAKE_CURRENT_BINARY_DIR}/tmp/pr299.cpu.edg"
      --proof "${CMAKE_CURRENT_BINARY_DIR}/tmp/pr299.cpu.proof"
      --backend cpu
      --max-rounds 100)
  set_tests_properties(integration.pr299_cpu_eliminate PROPERTIES LABELS "integration")

  add_test(
    NAME integration.pr299_cpu_verify
    COMMAND cudaee verify
      --tsp "${PR299_TSP}"
      --edges "${PR299_EDGES}"
      --proof "${CMAKE_CURRENT_BINARY_DIR}/tmp/pr299.cpu.proof")
  set_tests_properties(integration.pr299_cpu_verify PROPERTIES
    DEPENDS integration.pr299_cpu_eliminate
    LABELS "integration")

  if(CUDAEE_ENABLE_CUDA)
    if(Python3_Interpreter_FOUND)
      add_test(
        NAME differential.pr299_cpu_cuda
        COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/differential/compare_backends.py"
          "$<TARGET_FILE:cudaee>" "${PR299_TSP}" "${PR299_EDGES}"
          "${CMAKE_CURRENT_BINARY_DIR}/tmp/differential")
      set_tests_properties(differential.pr299_cpu_cuda PROPERTIES LABELS "gpu;differential")
    endif()
  endif()
endif()
