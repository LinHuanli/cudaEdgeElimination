add_executable(cudaee_unit unit/test_core.cpp)
target_link_libraries(cudaee_unit PRIVATE cudaee_core)
target_compile_definitions(cudaee_unit PRIVATE CUDAEE_TEST_TMP_DIR="${CMAKE_CURRENT_BINARY_DIR}/tmp")
target_compile_options(cudaee_unit PRIVATE -Wall -Wextra -Wconversion -Wshadow -Wformat=2)

add_test(NAME unit.core COMMAND cudaee_unit)

add_executable(cudaee_hamilton_tutte_unit unit/test_hamilton_tutte.cpp)
target_link_libraries(cudaee_hamilton_tutte_unit PRIVATE cudaee_core)
target_compile_options(cudaee_hamilton_tutte_unit PRIVATE
  -Wall -Wextra -Wconversion -Wshadow -Wformat=2)
add_test(NAME unit.hamilton_tutte COMMAND cudaee_hamilton_tutte_unit)
if(CUDAEE_ENABLE_CUDA)
  set_tests_properties(unit.hamilton_tutte PROPERTIES LABELS "correctness;hamilton-tutte;gpu")
else()
  set_tests_properties(unit.hamilton_tutte PROPERTIES LABELS "correctness;hamilton-tutte")
endif()

add_library(elimtsp_swap_oracle OBJECT "${PROJECT_SOURCE_DIR}/third_party/ElimTSP/src/swap.c")
target_include_directories(elimtsp_swap_oracle PRIVATE
  "${PROJECT_SOURCE_DIR}/third_party/ElimTSP/src")
# ccutil.h 自行定义 NDEBUG；先撤销 Release 默认值，避免测试 oracle 重复定义告警。
target_compile_options(elimtsp_swap_oracle PRIVATE -UNDEBUG)

add_executable(cudaee_kopt_unit
  unit/test_kopt.cpp
  $<TARGET_OBJECTS:elimtsp_swap_oracle>)
target_link_libraries(cudaee_kopt_unit PRIVATE cudaee_core)
target_compile_definitions(cudaee_kopt_unit PRIVATE
  CUDAEE_KOPT_TEST_TMP_DIR="${CMAKE_CURRENT_BINARY_DIR}/tmp")
target_compile_options(cudaee_kopt_unit PRIVATE
  -Wall -Wextra -Wconversion -Wshadow -Wformat=2)
add_test(NAME unit.kopt COMMAND cudaee_kopt_unit)
if(CUDAEE_ENABLE_CUDA)
  set_tests_properties(unit.kopt PROPERTIES LABELS "correctness;path-system;gpu")
else()
  set_tests_properties(unit.kopt PROPERTIES LABELS "correctness;path-system")
endif()

add_executable(cudaee_path_system_unit unit/test_path_system.cpp)
target_link_libraries(cudaee_path_system_unit PRIVATE cudaee_core)
target_compile_options(cudaee_path_system_unit PRIVATE
  -Wall -Wextra -Wconversion -Wshadow -Wformat=2)
add_test(NAME unit.path_system COMMAND cudaee_path_system_unit)
if(CUDAEE_ENABLE_CUDA)
  set_tests_properties(unit.path_system PROPERTIES LABELS "correctness;path-system;gpu")
else()
  set_tests_properties(unit.path_system PROPERTIES LABELS "correctness;path-system")
endif()

add_test(
  NAME integration.path_table_manifest
  COMMAND cudaee path-table
    --paths 5
    --backend cpu
    --output "${CMAKE_CURRENT_BINARY_DIR}/tmp/path-table-m5.cpu.manifest")
set_tests_properties(integration.path_table_manifest PROPERTIES
  LABELS "integration;path-system")

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
