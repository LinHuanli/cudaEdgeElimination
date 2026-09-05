find_package(Python3 COMPONENTS Interpreter QUIET)

if(Python3_Interpreter_FOUND)
  add_test(NAME unit.hybrid_protocol
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/unit/test_hybrid_protocol.py")
  set_tests_properties(unit.hybrid_protocol PROPERTIES
    LABELS "correctness;reproducibility" ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
  add_test(NAME unit.hs2014_data
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/unit/test_hs2014_data.py")
  set_tests_properties(unit.hs2014_data PROPERTIES
    LABELS "correctness;reproducibility" ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
  add_test(NAME unit.fgpu_benchmark
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/unit/test_benchmark_fgpu.py")
  set_tests_properties(unit.fgpu_benchmark PROPERTIES
    LABELS "correctness;reproducibility" ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
  add_test(NAME unit.optimal_tour_oracle
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/unit/test_optimal_tour_oracle.py")
  set_tests_properties(unit.optimal_tour_oracle PROPERTIES
    LABELS "correctness;exhaustive" ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
  add_test(
    NAME negative.fgpu_output_boundary
    COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/differential/reject_fgpu_output_escape.py"
      --exe "$<TARGET_FILE:fgpu-elim>"
      --instance "${CMAKE_CURRENT_SOURCE_DIR}/data/recursive-point.tsp"
      --tmp "${CMAKE_CURRENT_BINARY_DIR}/tmp/fgpu-output-boundary")
  set_tests_properties(negative.fgpu_output_boundary PROPERTIES
    PASS_REGULAR_EXPRESSION "FGPU output boundary violations rejected"
    LABELS "negative;fgpu;filesystem;correctness")
endif()

add_executable(cudaee_unit unit/test_core.cpp)
target_link_libraries(cudaee_unit PRIVATE cudaee_core)
target_compile_definitions(cudaee_unit PRIVATE CUDAEE_TEST_TMP_DIR="${CMAKE_CURRENT_BINARY_DIR}/tmp")
target_compile_options(cudaee_unit PRIVATE -Wall -Wextra -Wconversion -Wshadow -Wformat=2)

add_test(NAME unit.core COMMAND cudaee_unit)
add_executable(cudaee_optimal_filters_unit exhaustive/test_fgpu_optimal_filters.cpp)
target_link_libraries(cudaee_optimal_filters_unit PRIVATE cudaee_core)
target_compile_options(cudaee_optimal_filters_unit PRIVATE -Wall -Wextra -Wconversion -Wshadow)
add_test(NAME exhaustive.fgpu_optimal_filters COMMAND cudaee_optimal_filters_unit)
set_tests_properties(exhaustive.fgpu_optimal_filters PROPERTIES LABELS "correctness;exhaustive;fgpu")
add_executable(cudaee_build_identity_unit unit/test_build_identity.cpp)
target_link_libraries(cudaee_build_identity_unit PRIVATE cudaee_core)
target_compile_definitions(cudaee_build_identity_unit PRIVATE
  CUDAEE_TEST_TMP_DIR="${CMAKE_CURRENT_BINARY_DIR}/tmp/build-identity")
add_test(NAME unit.build_identity COMMAND cudaee_build_identity_unit)
set_tests_properties(unit.build_identity PROPERTIES LABELS "correctness;reproducibility")
add_executable(cudaee_transaction_unit unit/test_resident_transaction.cpp)
target_link_libraries(cudaee_transaction_unit PRIVATE cudaee_core)
target_compile_options(cudaee_transaction_unit PRIVATE -Wall -Wextra -Wconversion -Wshadow)
add_test(NAME unit.resident_transaction COMMAND cudaee_transaction_unit)
set_tests_properties(unit.resident_transaction PROPERTIES LABELS "correctness;transaction")
if(CUDAEE_ENABLE_CUDA)
  target_compile_definitions(cudaee_optimal_filters_unit PRIVATE CUDAEE_TEST_REQUIRE_GPU=1)
  set_property(TEST exhaustive.fgpu_optimal_filters APPEND PROPERTY LABELS "gpu")
  set_tests_properties(exhaustive.fgpu_optimal_filters PROPERTIES SKIP_RETURN_CODE 77)
  add_executable(cudaee_main_metric_unit unit/test_main_metric.cu)
  target_include_directories(cudaee_main_metric_unit PRIVATE "${PROJECT_SOURCE_DIR}/include")
  set_target_properties(cudaee_main_metric_unit PROPERTIES
    CUDA_ARCHITECTURES "${CUDAEE_CUDA_ARCHITECTURES}" CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
  add_test(NAME unit.main_metric COMMAND cudaee_main_metric_unit)
  set_tests_properties(unit.main_metric PROPERTIES LABELS "correctness;gpu;metric" SKIP_RETURN_CODE 77)
  add_executable(cudaee_gpu_bootstrap_unit unit/test_gpu_bootstrap.cpp)
  add_executable(cudaee_quick_reply_cache_unit unit/test_quick_reply_cache.cu)
  target_link_libraries(cudaee_quick_reply_cache_unit PRIVATE cudaee_core)
  set_target_properties(cudaee_quick_reply_cache_unit PROPERTIES
    CUDA_ARCHITECTURES "${CUDAEE_CUDA_ARCHITECTURES}" CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
  add_test(NAME unit.quick_reply_cache COMMAND cudaee_quick_reply_cache_unit)
  set_tests_properties(unit.quick_reply_cache PROPERTIES LABELS "correctness;gpu;reply" SKIP_RETURN_CODE 77)
  target_include_directories(cudaee_gpu_bootstrap_unit PRIVATE "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
  target_link_libraries(cudaee_gpu_bootstrap_unit PRIVATE cudaee_core)
  add_test(NAME unit.gpu_bootstrap COMMAND cudaee_gpu_bootstrap_unit)
  set_tests_properties(unit.gpu_bootstrap PROPERTIES LABELS "correctness;gpu;bootstrap" SKIP_RETURN_CODE 77)
  set_tests_properties(unit.core PROPERTIES LABELS "correctness;gpu")
  add_executable(cudaee_transaction_cuda_unit unit/test_resident_transaction_cuda.cu)
  target_link_libraries(cudaee_transaction_cuda_unit PRIVATE cudaee_core)
  set_target_properties(cudaee_transaction_cuda_unit PROPERTIES
    CUDA_ARCHITECTURES "${CUDAEE_CUDA_ARCHITECTURES}" CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
  add_test(NAME unit.resident_transaction_cuda COMMAND cudaee_transaction_cuda_unit)
  add_executable(cudaee_sparse_pdhg_unit unit/test_sparse_pdhg.cu)
  target_link_libraries(cudaee_sparse_pdhg_unit PRIVATE cudaee_core)
  set_target_properties(cudaee_sparse_pdhg_unit PROPERTIES
    CUDA_ARCHITECTURES "${CUDAEE_CUDA_ARCHITECTURES}" CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
  add_test(NAME unit.sparse_pdhg COMMAND cudaee_sparse_pdhg_unit)
  add_executable(cudaee_sec_replay_unit unit/test_resident_sec_replay.cu)
  target_link_libraries(cudaee_sec_replay_unit PRIVATE cudaee_core)
  set_target_properties(cudaee_sec_replay_unit PROPERTIES
    CUDA_ARCHITECTURES "${CUDAEE_CUDA_ARCHITECTURES}" CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
  add_test(NAME unit.sec_replay COMMAND cudaee_sec_replay_unit)
  set_tests_properties(unit.sec_replay PROPERTIES LABELS "correctness;gpu;lp;replay" SKIP_RETURN_CODE 77)
  if(Python3_Interpreter_FOUND)
    add_test(NAME exhaustive.fgpu_hybrid
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/exhaustive/check_fgpu_hybrid.py"
        "$<TARGET_FILE:fgpu-elim>" "${CMAKE_CURRENT_BINARY_DIR}/tmp/fgpu-hybrid")
    set_tests_properties(exhaustive.fgpu_hybrid PROPERTIES
      LABELS "correctness;gpu;fgpu;exhaustive" ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
    add_test(NAME exhaustive.fgpu_point_schedule
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/exhaustive/check_fgpu_point_schedule.py"
        "$<TARGET_FILE:fgpu-elim>" "${CMAKE_CURRENT_BINARY_DIR}/tmp/fgpu-point-schedule")
    set_tests_properties(exhaustive.fgpu_point_schedule PROPERTIES
      LABELS "correctness;gpu;fgpu;exhaustive;schedule" ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
    add_test(NAME exhaustive.fgpu_variants
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/exhaustive/check_fgpu_variants.py"
        "$<TARGET_FILE:fgpu-elim>" "${CMAKE_CURRENT_BINARY_DIR}/tmp/fgpu-variants")
    set_tests_properties(exhaustive.fgpu_variants PROPERTIES
      LABELS "correctness;gpu;fgpu;exhaustive" ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
  endif()
  set_tests_properties(unit.sparse_pdhg PROPERTIES LABELS "correctness;gpu;lp;pdhg" SKIP_RETURN_CODE 77)
  set_tests_properties(unit.resident_transaction_cuda PROPERTIES
    LABELS "correctness;gpu;transaction" SKIP_RETURN_CODE 77)
else()
  set_tests_properties(unit.core PROPERTIES LABELS "correctness")
endif()

add_test(
  NAME integration.protected_tour
  COMMAND cudaee tour-check
    --tsp "${CMAKE_CURRENT_SOURCE_DIR}/data/recursive-point.tsp"
    --edges "${CMAKE_CURRENT_SOURCE_DIR}/data/recursive-point.edg"
    --tour "${CMAKE_CURRENT_SOURCE_DIR}/data/recursive-point.tour"
    --expected-cost 395)
set_tests_properties(integration.protected_tour PROPERTIES LABELS "integration;correctness")

set(COMPLETE_GRAPH_EDGES "${CMAKE_CURRENT_BINARY_DIR}/tmp/recursive-point.complete.cli.edg")
add_test(
  NAME integration.complete_graph
  COMMAND cudaee complete-graph
    --tsp "${CMAKE_CURRENT_SOURCE_DIR}/data/recursive-point.tsp"
    --output "${COMPLETE_GRAPH_EDGES}")
set_tests_properties(integration.complete_graph PROPERTIES
  PASS_REGULAR_EXPRESSION "status=OK dimension=8 active_edges=28"
  LABELS "integration;graph")

add_test(
  NAME integration.complete_graph_tour
  COMMAND cudaee tour-check
    --tsp "${CMAKE_CURRENT_SOURCE_DIR}/data/recursive-point.tsp"
    --edges "${COMPLETE_GRAPH_EDGES}"
    --tour "${CMAKE_CURRENT_SOURCE_DIR}/data/recursive-point.tour"
    --expected-cost 395)
set_tests_properties(integration.complete_graph_tour PROPERTIES
  DEPENDS integration.complete_graph
  LABELS "integration;correctness;graph")

if(CUDAEE_ENABLE_CUDA)
  set(FGPU_TEST_DIR "${CMAKE_CURRENT_BINARY_DIR}/tmp/fgpu")
  set(FGPU_TEST_EDGES "${FGPU_TEST_DIR}/recursive-point.edg")
  set(FGPU_TEST_CERT "${FGPU_TEST_DIR}/recursive-point.fgcert")
  add_test(
    NAME integration.fgpu_geometry_run
    COMMAND fgpu-elim run
      --instance "${CMAKE_CURRENT_SOURCE_DIR}/data/recursive-point.tsp"
      --tour "${CMAKE_CURRENT_SOURCE_DIR}/data/recursive-point.opt.tour"
      --tour-role known-optimum
      --expected-cost 251
      --device 0
      --numeric mixed-safe
      --verification epoch
      --pdlp off
      --potential-candidates 6
      --geometry-witnesses 4
      --enable-jv 0
      --enable-ht 0
      --output-edges "${FGPU_TEST_EDGES}"
      --fixed "${FGPU_TEST_DIR}/recursive-point.fix"
      --nonpairs "${FGPU_TEST_DIR}/recursive-point.nonpairs"
      --certificate "${FGPU_TEST_CERT}"
      --manifest "${FGPU_TEST_DIR}/recursive-point.manifest")
  set_tests_properties(integration.fgpu_geometry_run PROPERTIES
    PASS_REGULAR_EXPRESSION "status=OK.*geometry_deleted=[1-9]"
    LABELS "integration;fgpu;geometry;gpu;correctness")

  add_test(
    NAME integration.fgpu_manifest_v2
    COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/differential/check_fgpu_manifest.py"
      --manifest "${FGPU_TEST_DIR}/recursive-point.manifest"
      --expect "input_edges=-"
      --expect "tour_role=known-optimum"
      --expect "enable_geometry=1"
      --expect "enable_jv=0"
      --expect "enable_ht=0"
      --expect "potential_candidates=6"
      --expect "geometry_witnesses_per_edge=4"
      --expect "pdlp=off")
  set_tests_properties(integration.fgpu_manifest_v2 PROPERTIES
    DEPENDS integration.fgpu_geometry_run
    PASS_REGULAR_EXPRESSION "FGPU manifest V2 verified"
    LABELS "integration;fgpu;manifest;correctness")

  add_test(
    NAME integration.fgpu_geometry_verify
    COMMAND fgpu-elim verify
      --instance "${CMAKE_CURRENT_SOURCE_DIR}/data/recursive-point.tsp"
      --tour "${CMAKE_CURRENT_SOURCE_DIR}/data/recursive-point.opt.tour"
      --tour-role known-optimum
      --expected-cost 251
      --output-edges "${FGPU_TEST_EDGES}"
      --fixed "${FGPU_TEST_DIR}/recursive-point.fix"
      --nonpairs "${FGPU_TEST_DIR}/recursive-point.nonpairs"
      --certificate "${FGPU_TEST_CERT}")
  set_tests_properties(integration.fgpu_geometry_verify PROPERTIES
    DEPENDS integration.fgpu_geometry_run
    PASS_REGULAR_EXPRESSION "status=VERIFIED.*geometry=[1-9]"
    LABELS "integration;fgpu;geometry;certificate;correctness")

  add_test(
    NAME negative.fgpu_geometry_tamper
    COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/differential/tamper_fgpu_certificate.py"
      --exe "$<TARGET_FILE:fgpu-elim>"
      --instance "${CMAKE_CURRENT_SOURCE_DIR}/data/recursive-point.tsp"
      --edges "${FGPU_TEST_EDGES}"
      --certificate "${FGPU_TEST_CERT}"
      --output "${FGPU_TEST_DIR}/recursive-point.tampered.fgcert")
  set_tests_properties(negative.fgpu_geometry_tamper PROPERTIES
    DEPENDS integration.fgpu_geometry_run
    PASS_REGULAR_EXPRESSION "tampered FGPU geometry certificate rejected"
    LABELS "negative;fgpu;geometry;certificate;correctness")

  set(FGPU_PDLP_TEST_DIR "${CMAKE_CURRENT_BINARY_DIR}/tmp/fgpu-pdlp")
  set(FGPU_PDLP_TEST_EDGES "${FGPU_PDLP_TEST_DIR}/recursive-point.edg")
  set(FGPU_PDLP_TEST_CERT "${FGPU_PDLP_TEST_DIR}/recursive-point.fgcert")
  add_test(
    NAME integration.fgpu_pdlp_run
    COMMAND fgpu-elim run
      --instance "${CMAKE_CURRENT_SOURCE_DIR}/data/recursive-point.tsp"
      --tour "${CMAKE_CURRENT_SOURCE_DIR}/data/recursive-point.opt.tour"
      --tour-role known-optimum
      --expected-cost 251
      --device 0
      --numeric mixed-safe
      --verification epoch
      --pdlp native
      --pdlp-iterations 2000
      --enable-geometry 0
      --enable-jv 0
      --enable-ht 0
      --output-edges "${FGPU_PDLP_TEST_EDGES}"
      --fixed "${FGPU_PDLP_TEST_DIR}/recursive-point.fix"
      --nonpairs "${FGPU_PDLP_TEST_DIR}/recursive-point.nonpairs"
      --certificate "${FGPU_PDLP_TEST_CERT}"
      --manifest "${FGPU_PDLP_TEST_DIR}/recursive-point.manifest")
  set_tests_properties(integration.fgpu_pdlp_run PROPERTIES
    PASS_REGULAR_EXPRESSION "status=OK.*lp_deleted=[1-9]"
    LABELS "integration;fgpu;pdlp;gpu;correctness")

  add_test(
    NAME integration.fgpu_pdlp_verify
    COMMAND fgpu-elim verify
      --instance "${CMAKE_CURRENT_SOURCE_DIR}/data/recursive-point.tsp"
      --tour "${CMAKE_CURRENT_SOURCE_DIR}/data/recursive-point.opt.tour"
      --tour-role known-optimum
      --expected-cost 251
      --output-edges "${FGPU_PDLP_TEST_EDGES}"
      --fixed "${FGPU_PDLP_TEST_DIR}/recursive-point.fix"
      --nonpairs "${FGPU_PDLP_TEST_DIR}/recursive-point.nonpairs"
      --certificate "${FGPU_PDLP_TEST_CERT}")
  set_tests_properties(integration.fgpu_pdlp_verify PROPERTIES
    DEPENDS integration.fgpu_pdlp_run
    PASS_REGULAR_EXPRESSION "status=VERIFIED.*lp=[1-9]"
    LABELS "integration;fgpu;pdlp;certificate;correctness")

  add_test(
    NAME negative.fgpu_pdlp_tamper
    COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/differential/tamper_fgpu_lp_certificate.py"
      --exe "$<TARGET_FILE:fgpu-elim>"
      --instance "${CMAKE_CURRENT_SOURCE_DIR}/data/recursive-point.tsp"
      --edges "${FGPU_PDLP_TEST_EDGES}"
      --certificate "${FGPU_PDLP_TEST_CERT}"
      --output "${FGPU_PDLP_TEST_DIR}/recursive-point.tampered.fgcert")
  set_tests_properties(negative.fgpu_pdlp_tamper PROPERTIES
    DEPENDS integration.fgpu_pdlp_run
    PASS_REGULAR_EXPRESSION "tampered FGPU LP certificate rejected"
    LABELS "negative;fgpu;pdlp;certificate;correctness")

  add_test(
    NAME negative.fgpu_fixed_tamper
    COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/differential/tamper_fgpu_fixed_output.py"
      --exe "$<TARGET_FILE:fgpu-elim>"
      --instance "${CMAKE_CURRENT_SOURCE_DIR}/data/recursive-point.tsp"
      --edges "${FGPU_PDLP_TEST_EDGES}"
      --fixed "${FGPU_PDLP_TEST_DIR}/recursive-point.fix"
      --nonpairs "${FGPU_PDLP_TEST_DIR}/recursive-point.nonpairs"
      --certificate "${FGPU_PDLP_TEST_CERT}"
      --output "${FGPU_PDLP_TEST_DIR}/recursive-point.tampered.fix")
  set_tests_properties(negative.fgpu_fixed_tamper PROPERTIES
    DEPENDS integration.fgpu_pdlp_run
    PASS_REGULAR_EXPRESSION "tampered FGPU fixed output rejected"
    LABELS "negative;fgpu;fixed;certificate;correctness")

  set(FGPU_RESIDENT_TEST_DIR "${CMAKE_CURRENT_BINARY_DIR}/tmp/fgpu-resident")
  set(FGPU_RESIDENT_TEST_EDGES "${FGPU_RESIDENT_TEST_DIR}/recursive-point.edg")
  set(FGPU_RESIDENT_TEST_CERT "${FGPU_RESIDENT_TEST_DIR}/recursive-point.fgcert")
  add_test(
    NAME integration.fgpu_resident_run
    COMMAND fgpu-elim resident
      --instance "${CMAKE_CURRENT_SOURCE_DIR}/data/recursive-point.tsp"
      --tour "${CMAKE_CURRENT_SOURCE_DIR}/data/recursive-point.opt.tour"
      --tour-role known-optimum
      --expected-cost 251
      --device 0
      --potential-candidates 6
      --pdlp-iterations 2000
      --max-pdlp-epochs 2
      --max-hs-epochs 10
      --max-jv-rounds 10
      --enable-geometry 1
      --enable-pdlp 1
      --enable-quick-hs 1
      --enable-jv 1
      --cpu-audit 1
      --output-edges "${FGPU_RESIDENT_TEST_EDGES}"
      --fixed "${FGPU_RESIDENT_TEST_DIR}/recursive-point.fix"
      --nonpairs "${FGPU_RESIDENT_TEST_DIR}/recursive-point.nonpairs"
      --certificate "${FGPU_RESIDENT_TEST_CERT}"
      --manifest "${FGPU_RESIDENT_TEST_DIR}/recursive-point.manifest")
  set_tests_properties(integration.fgpu_resident_run PROPERTIES
    PASS_REGULAR_EXPRESSION
      "status=OK mode=resident.*geometry_deleted=[1-9].*lp_deleted=[1-9].*final_edges=8.*converged=1"
    LABELS "integration;fgpu;resident;geometry;pdlp;gpu;correctness")

  add_test(
    NAME integration.fgpu_resident_verify
    COMMAND fgpu-elim verify
      --instance "${CMAKE_CURRENT_SOURCE_DIR}/data/recursive-point.tsp"
      --tour "${CMAKE_CURRENT_SOURCE_DIR}/data/recursive-point.opt.tour"
      --tour-role known-optimum
      --expected-cost 251
      --output-edges "${FGPU_RESIDENT_TEST_EDGES}"
      --fixed "${FGPU_RESIDENT_TEST_DIR}/recursive-point.fix"
      --nonpairs "${FGPU_RESIDENT_TEST_DIR}/recursive-point.nonpairs"
      --certificate "${FGPU_RESIDENT_TEST_CERT}")
  set_tests_properties(integration.fgpu_resident_verify PROPERTIES
    DEPENDS integration.fgpu_resident_run
    PASS_REGULAR_EXPRESSION "status=VERIFIED.*geometry=[1-9].*lp=[1-9].*active_edges=8"
    LABELS "integration;fgpu;resident;certificate;gpu;correctness")

  set(FGPU_RESIDENT_RAW_TEST_DIR "${CMAKE_CURRENT_BINARY_DIR}/tmp/fgpu-resident-raw")
  add_test(
    NAME integration.fgpu_resident_raw_run
    COMMAND fgpu-elim resident
      --instance "${CMAKE_CURRENT_SOURCE_DIR}/data/recursive-point.tsp"
      --tour "${CMAKE_CURRENT_SOURCE_DIR}/data/recursive-point.opt.tour"
      --tour-role known-optimum
      --expected-cost 251
      --device 0
      --potential-candidates 6
      --pdlp-iterations 2000
      --max-pdlp-epochs 2
      --max-hs-epochs 10
      --max-jv-rounds 10
      --enable-geometry 1
      --enable-pdlp 1
      --enable-quick-hs 1
      --enable-jv 1
      --cpu-audit 0
      --output-edges "${FGPU_RESIDENT_RAW_TEST_DIR}/recursive-point.edg"
      --fixed "${FGPU_RESIDENT_RAW_TEST_DIR}/recursive-point.fix"
      --nonpairs "${FGPU_RESIDENT_RAW_TEST_DIR}/recursive-point.nonpairs"
      --manifest "${FGPU_RESIDENT_RAW_TEST_DIR}/recursive-point.manifest")
  set_tests_properties(integration.fgpu_resident_raw_run PROPERTIES
    PASS_REGULAR_EXPRESSION
      "status=OK mode=resident.*final_edges=8.*converged=1.*cpu_audited=0.*certificate_bytes=0"
    LABELS "integration;fgpu;resident;raw;gpu;performance")

  if(Python3_Interpreter_FOUND)
    add_test(
      NAME exhaustive.fgpu_solve_gpu_safe
      COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/exhaustive/check_fgpu_solve.py"
        "$<TARGET_FILE:fgpu-elim>"
        "${CMAKE_CURRENT_SOURCE_DIR}/data/recursive-point.tsp"
        "${CMAKE_CURRENT_SOURCE_DIR}/data/recursive-point.edg"
        "${CMAKE_CURRENT_SOURCE_DIR}/data/recursive-point.opt.tour"
        "${CMAKE_CURRENT_BINARY_DIR}/tmp/fgpu-solve-exhaustive")
    set_tests_properties(exhaustive.fgpu_solve_gpu_safe PROPERTIES
      PASS_REGULAR_EXPRESSION "FGPU solve exhaustive check passed"
      LABELS "exhaustive;integration;fgpu;gpu;gpu-replay;correctness;solve")

    add_test(
      NAME exhaustive.fgpu_solve_e2_overlap
      COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/exhaustive/check_fgpu_solve.py"
        "$<TARGET_FILE:fgpu-elim>"
        "${CMAKE_CURRENT_SOURCE_DIR}/data/e2-overlap-regression.tsp"
        "${CMAKE_CURRENT_SOURCE_DIR}/data/e2-overlap-regression.edg"
        "${CMAKE_CURRENT_SOURCE_DIR}/data/e2-overlap-regression.opt.tour"
        "${CMAKE_CURRENT_BINARY_DIR}/tmp/fgpu-solve-e2-overlap"
        "ht_extra_edge_deleted"
        "point_path_end_closed_replies")
    set_tests_properties(exhaustive.fgpu_solve_e2_overlap PROPERTIES
      PASS_REGULAR_EXPRESSION "FGPU solve exhaustive check passed"
      LABELS "exhaustive;integration;fgpu;gpu;gpu-replay;correctness;solve;e2;point-path-end")

    add_test(
      NAME exhaustive.fgpu_e2_distinct
      COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/exhaustive/check_fgpu_e2.py"
        "$<TARGET_FILE:fgpu-elim>"
        "${CMAKE_CURRENT_SOURCE_DIR}/data/e2-distinct-regression.tsp"
        "${CMAKE_CURRENT_SOURCE_DIR}/data/e2-distinct-regression.opt.tour"
        "${CMAKE_CURRENT_BINARY_DIR}/tmp/fgpu-e2-distinct")
    set_tests_properties(exhaustive.fgpu_e2_distinct PROPERTIES
      PASS_REGULAR_EXPRESSION "FGPU e2 exhaustive check passed"
      LABELS "exhaustive;integration;fgpu;gpu;correctness;e2;path-system")

    add_test(
      NAME exhaustive.fgpu_oneshot_unprotected
      COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/exhaustive/check_fgpu_oneshot.py"
        "$<TARGET_FILE:fgpu-elim>"
        "${CMAKE_CURRENT_SOURCE_DIR}/data/recursive-point.tsp"
        "${CMAKE_CURRENT_SOURCE_DIR}/data/recursive-point.opt.tour"
        "${CMAKE_CURRENT_BINARY_DIR}/tmp/fgpu-oneshot-exhaustive")
    set_tests_properties(exhaustive.fgpu_oneshot_unprotected PROPERTIES
      PASS_REGULAR_EXPRESSION "FGPU one-shot exhaustive optimal-edge check passed"
      LABELS "exhaustive;integration;fgpu;resident;gpu;correctness;oneshot")

    add_test(
      NAME exhaustive.fgpu_nonpair_persistence
      COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/exhaustive/check_fgpu_nonpair_persistence.py"
        "$<TARGET_FILE:fgpu-elim>"
        "${CMAKE_CURRENT_SOURCE_DIR}/data/recursive-point.tsp"
        "${CMAKE_CURRENT_SOURCE_DIR}/data/recursive-point.opt.tour"
        "${CMAKE_CURRENT_BINARY_DIR}/tmp/fgpu-nonpair-persistence")
    set_tests_properties(exhaustive.fgpu_nonpair_persistence PROPERTIES
      PASS_REGULAR_EXPRESSION "FGPU non-pair persistence check passed"
      LABELS "exhaustive;integration;fgpu;resident;gpu;correctness;lp;nonpair")
  endif()

  if(EXISTS "${PROJECT_SOURCE_DIR}/third_party/ElimTSP/KH-elim/pr299.tsp")
    set(FGPU_RESIDENT_HS_TEST_DIR "${CMAKE_CURRENT_BINARY_DIR}/tmp/fgpu-resident-hs")
    set(FGPU_RESIDENT_HS_TEST_EDGES "${FGPU_RESIDENT_HS_TEST_DIR}/pr299.edg")
    set(FGPU_RESIDENT_HS_TEST_CERT "${FGPU_RESIDENT_HS_TEST_DIR}/pr299.fgcert")
    add_test(
      NAME integration.fgpu_resident_quick_hs_run
      COMMAND fgpu-elim resident-local
        --instance "${PROJECT_SOURCE_DIR}/third_party/ElimTSP/KH-elim/pr299.tsp"
        --input-edges "${PROJECT_SOURCE_DIR}/third_party/ElimTSP/KH-elim/pr299.edg"
        --device 0
        --max-hs-epochs 10
        --max-jv-rounds 10
        --enable-quick-hs 1
        --enable-jv 1
        --cpu-audit 1
        --output-edges "${FGPU_RESIDENT_HS_TEST_EDGES}"
        --fixed "${FGPU_RESIDENT_HS_TEST_DIR}/pr299.fix"
        --nonpairs "${FGPU_RESIDENT_HS_TEST_DIR}/pr299.nonpairs"
        --certificate "${FGPU_RESIDENT_HS_TEST_CERT}"
        --manifest "${FGPU_RESIDENT_HS_TEST_DIR}/pr299.manifest")
    set_tests_properties(integration.fgpu_resident_quick_hs_run PROPERTIES
      PASS_REGULAR_EXPRESSION
        "status=OK mode=resident-local.*jv_deleted=[1-9].*quick_hs_deleted=[1-9].*final_edges=860.*converged=1.*cpu_audited=1"
      LABELS "integration;fgpu;resident;quick-hs;gpu;correctness;strength")

    add_test(
      NAME integration.fgpu_resident_quick_hs_verify
      COMMAND fgpu-elim verify
        --instance "${PROJECT_SOURCE_DIR}/third_party/ElimTSP/KH-elim/pr299.tsp"
        --input-edges "${PROJECT_SOURCE_DIR}/third_party/ElimTSP/KH-elim/pr299.edg"
        --output-edges "${FGPU_RESIDENT_HS_TEST_EDGES}"
        --fixed "${FGPU_RESIDENT_HS_TEST_DIR}/pr299.fix"
        --nonpairs "${FGPU_RESIDENT_HS_TEST_DIR}/pr299.nonpairs"
        --certificate "${FGPU_RESIDENT_HS_TEST_CERT}")
    set_tests_properties(integration.fgpu_resident_quick_hs_verify PROPERTIES
      DEPENDS integration.fgpu_resident_quick_hs_run
      PASS_REGULAR_EXPRESSION "status=VERIFIED.*quick_hs=[1-9].*active_edges=860"
      LABELS "integration;fgpu;resident;quick-hs;certificate;gpu;correctness")

    if(Python3_Interpreter_FOUND)
      add_test(
        NAME integration.fgpu_direct_fix_pr299
        COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/exhaustive/check_fgpu_direct_fix.py"
          "$<TARGET_FILE:fgpu-elim>"
          "${PROJECT_SOURCE_DIR}/third_party/ElimTSP/KH-elim/pr299.tsp"
          "${PROJECT_SOURCE_DIR}/third_party/ElimTSP/KH-elim/pr299.edg"
          "${CMAKE_CURRENT_SOURCE_DIR}/data/pr299.opt.tour"
          "${CMAKE_CURRENT_BINARY_DIR}/tmp/fgpu-direct-fix-pr299")
      set_tests_properties(integration.fgpu_direct_fix_pr299 PROPERTIES
        PASS_REGULAR_EXPRESSION "FGPU Direct Fix check passed"
        LABELS "integration;fgpu;gpu;gpu-replay;correctness;fixing;strength")

      add_test(
        NAME negative.fgpu_resident_quick_hs_tamper
        COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/differential/tamper_fgpu_quick_hs_certificate.py"
          --exe "$<TARGET_FILE:fgpu-elim>"
          --instance "${PROJECT_SOURCE_DIR}/third_party/ElimTSP/KH-elim/pr299.tsp"
          --input-edges "${PROJECT_SOURCE_DIR}/third_party/ElimTSP/KH-elim/pr299.edg"
          --edges "${FGPU_RESIDENT_HS_TEST_EDGES}"
          --certificate "${FGPU_RESIDENT_HS_TEST_CERT}"
          --output "${FGPU_RESIDENT_HS_TEST_DIR}/pr299.tampered.fgcert")
      set_tests_properties(negative.fgpu_resident_quick_hs_tamper PROPERTIES
        DEPENDS integration.fgpu_resident_quick_hs_run
        PASS_REGULAR_EXPRESSION "tampered FGPU Quick-HS certificate rejected"
        LABELS "negative;fgpu;resident;quick-hs;certificate;correctness")
    endif()
  endif()
endif()

if(TARGET cudaee_kh_elim)
  add_test(
    NAME integration.author_kh_jq_pr299
    COMMAND cudaee_kh_elim
      -Jq
      -o "${CMAKE_CURRENT_BINARY_DIR}/tmp/pr299.kh-jq.edg"
      -T "${PROJECT_SOURCE_DIR}/third_party/ElimTSP/KH-elim/pr299.tsp"
      "${PROJECT_SOURCE_DIR}/third_party/ElimTSP/KH-elim/pr299.edg")
  set_tests_properties(integration.author_kh_jq_pr299 PROPERTIES
    PASS_REGULAR_EXPRESSION "Remaining: 872"
    LABELS "integration;author-reference;strength")
endif()

add_executable(cudaee_hamilton_tutte_unit unit/test_hamilton_tutte.cpp)
target_link_libraries(cudaee_hamilton_tutte_unit PRIVATE cudaee_core)
target_compile_definitions(cudaee_hamilton_tutte_unit PRIVATE
  CUDAEE_HT_TEST_TMP_DIR="${CMAKE_CURRENT_BINARY_DIR}/tmp")
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

set(HT_RECURSIVE_TSP "${CMAKE_CURRENT_SOURCE_DIR}/data/recursive-point.tsp")
set(HT_RECURSIVE_EDGES "${CMAKE_CURRENT_SOURCE_DIR}/data/recursive-point.edg")
set(HT_RECURSIVE_PROOF "${CMAKE_CURRENT_BINARY_DIR}/tmp/recursive-point.cli.ht-proof")
add_test(
  NAME integration.ht_recursive_prove
  COMMAND cudaee ht-prove
    --tsp "${HT_RECURSIVE_TSP}"
    --edges "${HT_RECURSIVE_EDGES}"
    --u 2 --v 4
    --proof "${HT_RECURSIVE_PROOF}"
    --scheduler dfs
    --backend cpu
    --max-neighborhood 0
    --max-cd-candidates 8
    --max-candidate-degree 0
    --max-root-replies 1000
    --max-k 3
    --max-deletion-sets 1
    --max-depth 1
    --max-states 20000
    --max-total-replies 200000
    --max-replies-per-move 1000
    --max-point-candidates 5
    --max-end-candidates 5)
set_tests_properties(integration.ht_recursive_prove PROPERTIES
  LABELS "integration;hamilton-tutte")

add_test(
  NAME integration.ht_recursive_verify
  COMMAND cudaee ht-verify
    --tsp "${HT_RECURSIVE_TSP}"
    --edges "${HT_RECURSIVE_EDGES}"
    --proof "${HT_RECURSIVE_PROOF}")
set_tests_properties(integration.ht_recursive_verify PROPERTIES
  DEPENDS integration.ht_recursive_prove
  LABELS "integration;hamilton-tutte")

set(HT_WAVEFRONT_PROOF "${CMAKE_CURRENT_BINARY_DIR}/tmp/recursive-point.wavefront.ht-proof")
set(HT_WAVEFRONT_TRACE "${CMAKE_CURRENT_BINARY_DIR}/tmp/recursive-point.wavefront.ht-trace")
set(HT_WAVEFRONT_TRACE_REPORT
  "${CMAKE_CURRENT_BINARY_DIR}/tmp/recursive-point.wavefront.ht-trace-report")
add_test(
  NAME integration.ht_wavefront_prove
  COMMAND cudaee ht-prove
    --tsp "${HT_RECURSIVE_TSP}"
    --edges "${HT_RECURSIVE_EDGES}"
    --u 2 --v 4
    --proof "${HT_WAVEFRONT_PROOF}"
    --trace-output "${HT_WAVEFRONT_TRACE}"
    --scheduler wavefront
    --backend cpu
    --reply-backend cpu
    --reply-frontier-batch-states 256
    --path-append-backend cpu
    --propagation-backend cpu
    --max-neighborhood 0
    --max-cd-candidates 8
    --max-candidate-degree 0
    --max-root-replies 1000
    --max-k 3
    --max-deletion-sets 1
    --max-depth 1
    --max-states 20000
    --max-total-replies 200000
    --max-replies-per-move 1000
    --max-point-candidates 5
    --max-end-candidates 5)
set_tests_properties(integration.ht_wavefront_prove PROPERTIES
  PASS_REGULAR_EXPRESSION "fuse_leaf_buckets=1"
  LABELS "integration;hamilton-tutte;wavefront")

add_test(
  NAME integration.ht_wavefront_verify
  COMMAND cudaee ht-verify
    --tsp "${HT_RECURSIVE_TSP}"
    --edges "${HT_RECURSIVE_EDGES}"
    --proof "${HT_WAVEFRONT_PROOF}")
set_tests_properties(integration.ht_wavefront_verify PROPERTIES
  DEPENDS integration.ht_wavefront_prove
  LABELS "integration;hamilton-tutte;wavefront")

add_test(
  NAME integration.ht_trace_replay
  COMMAND cudaee ht-trace-replay
    --input "${HT_WAVEFRONT_TRACE}"
    --output "${HT_WAVEFRONT_TRACE_REPORT}"
    --speculation-widths 1,2,4,8,all)
set_tests_properties(integration.ht_trace_replay PROPERTIES
  DEPENDS integration.ht_wavefront_prove
  PASS_REGULAR_EXPRESSION "speculation_width=all"
  LABELS "integration;hamilton-tutte;wavefront;trace")

set(HT_TRANSPOSED_PROOF "${CMAKE_CURRENT_BINARY_DIR}/tmp/recursive-point.transposed.ht-proof")
add_test(
  NAME integration.ht_transposed_prove
  COMMAND cudaee ht-prove
    --tsp "${HT_RECURSIVE_TSP}"
    --edges "${HT_RECURSIVE_EDGES}"
    --u 2 --v 4
    --proof "${HT_TRANSPOSED_PROOF}"
    --scheduler transposed
    --speculation 1
    --backend cpu
    --leaf-backend cpu
    --reply-backend cpu
    --path-append-backend cpu
    --propagation-backend cpu
    --max-neighborhood 0
    --max-cd-candidates 8
    --max-candidate-degree 0
    --max-root-replies 1000
    --max-k 3
    --max-deletion-sets 1
    --max-depth 1
    --max-states 20000
    --max-total-replies 200000
    --max-replies-per-move 1000
    --max-point-candidates 5
    --max-end-candidates 5)
set_tests_properties(integration.ht_transposed_prove PROPERTIES
  PASS_REGULAR_EXPRESSION "status=PROVEN scheduler=transposed"
  LABELS "integration;hamilton-tutte;transposed")

add_test(
  NAME integration.ht_transposed_verify
  COMMAND cudaee ht-verify
    --tsp "${HT_RECURSIVE_TSP}"
    --edges "${HT_RECURSIVE_EDGES}"
    --proof "${HT_TRANSPOSED_PROOF}")
set_tests_properties(integration.ht_transposed_verify PROPERTIES
  DEPENDS integration.ht_transposed_prove
  LABELS "integration;hamilton-tutte;transposed")

if(CUDAEE_ENABLE_CUDA)
  set(HT_WAVEFRONT_HYBRID_PROOF
    "${CMAKE_CURRENT_BINARY_DIR}/tmp/recursive-point.wavefront.hybrid.ht-proof")
  add_test(
    NAME integration.ht_wavefront_hybrid_prove
    COMMAND cudaee ht-prove
      --tsp "${HT_RECURSIVE_TSP}"
      --edges "${HT_RECURSIVE_EDGES}"
      --u 2 --v 4
      --proof "${HT_WAVEFRONT_HYBRID_PROOF}"
      --scheduler wavefront
      --backend cpu
      --leaf-backend cuda
      --fuse-leaf-buckets 0
      --reply-backend cpu
      --path-append-backend cpu
      --propagation-backend cpu
      --max-neighborhood 0
      --max-cd-candidates 8
      --max-candidate-degree 0
      --max-root-replies 1000
      --max-k 3
      --max-deletion-sets 1
      --max-depth 1
      --max-states 20000
      --max-total-replies 200000
      --max-replies-per-move 1000
      --max-point-candidates 5
      --max-end-candidates 5)
  set_tests_properties(integration.ht_wavefront_hybrid_prove PROPERTIES
    PASS_REGULAR_EXPRESSION "fuse_leaf_buckets=0 leaf_cost_backend=cuda"
    LABELS "integration;hamilton-tutte;wavefront;gpu")

  add_test(
    NAME integration.ht_wavefront_hybrid_verify
    COMMAND cudaee ht-verify
      --tsp "${HT_RECURSIVE_TSP}"
      --edges "${HT_RECURSIVE_EDGES}"
      --proof "${HT_WAVEFRONT_HYBRID_PROOF}")
  set_tests_properties(integration.ht_wavefront_hybrid_verify PROPERTIES
    DEPENDS integration.ht_wavefront_hybrid_prove
    LABELS "integration;hamilton-tutte;wavefront;gpu")
endif()

set(HT_EPOCH_EDGES "${CMAKE_CURRENT_BINARY_DIR}/tmp/recursive-point.epoch.edg")
set(HT_EPOCH_PROOF "${CMAKE_CURRENT_BINARY_DIR}/tmp/recursive-point.epoch.proof")
add_test(
  NAME integration.ht_epoch_commit
  COMMAND cudaee ht-commit
    --tsp "${HT_RECURSIVE_TSP}"
    --edges "${HT_RECURSIVE_EDGES}"
    --output "${HT_EPOCH_EDGES}"
    --proof "${HT_EPOCH_PROOF}"
    --ht-proof "${HT_WAVEFRONT_PROOF}"
    --ht-proof "${HT_WAVEFRONT_PROOF}")
set_tests_properties(integration.ht_epoch_commit PROPERTIES
  DEPENDS integration.ht_wavefront_prove
  LABELS "integration;hamilton-tutte;epoch")

add_test(
  NAME integration.ht_epoch_verify
  COMMAND cudaee verify
    --tsp "${HT_RECURSIVE_TSP}"
    --edges "${HT_RECURSIVE_EDGES}"
    --proof "${HT_EPOCH_PROOF}")
set_tests_properties(integration.ht_epoch_verify PROPERTIES
  DEPENDS integration.ht_epoch_commit
  LABELS "integration;hamilton-tutte;epoch")

if(ZLIB_FOUND AND Python3_Interpreter_FOUND)
  add_test(
    NAME negative.ht_epoch_zlib_tamper
    COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/differential/tamper_ht_zlib_certificate.py"
      --exe "$<TARGET_FILE:cudaee>"
      --tsp "${HT_RECURSIVE_TSP}"
      --edges "${HT_RECURSIVE_EDGES}"
      --proof "${HT_EPOCH_PROOF}"
      --output "${CMAKE_CURRENT_BINARY_DIR}/tmp/recursive-point.epoch.tampered.proof")
  set_tests_properties(negative.ht_epoch_zlib_tamper PROPERTIES
    DEPENDS integration.ht_epoch_commit
    PASS_REGULAR_EXPRESSION "tampered compressed HT certificate rejected"
    LABELS "negative;certificate;hamilton-tutte;epoch")
endif()

set(HT_SCAN_EDGES "${CMAKE_CURRENT_BINARY_DIR}/tmp/recursive-point.scan.edg")
set(HT_SCAN_PROOF "${CMAKE_CURRENT_BINARY_DIR}/tmp/recursive-point.scan.proof")
set(HT_SCAN_REPORT "${CMAKE_CURRENT_BINARY_DIR}/tmp/recursive-point.scan.report")
add_test(
  NAME integration.ht_scan
  COMMAND cudaee ht-scan
    --tsp "${HT_RECURSIVE_TSP}"
    --edges "${HT_RECURSIVE_EDGES}"
    --output "${HT_SCAN_EDGES}"
    --proof "${HT_SCAN_PROOF}"
    --report "${HT_SCAN_REPORT}"
    --protected-tour "${CMAKE_CURRENT_SOURCE_DIR}/data/recursive-point.tour"
    --expected-cost 395
    --max-targets 1
    --target-offset 14
    --target-order canonical
    --backend cpu
    --reply-backend cpu
    --path-append-backend cpu
    --propagation-backend cpu
    --max-neighborhood 0
    --max-cd-candidates 8
    --max-candidate-degree 0
    --max-root-replies 1000
    --max-k 3
    --max-deletion-sets 1
    --max-depth 1
    --max-states 20000
    --max-total-replies 200000
    --max-replies-per-move 1000
    --max-point-candidates 5
    --max-end-candidates 5)
set_tests_properties(integration.ht_scan PROPERTIES
  PASS_REGULAR_EXPRESSION "fuse_leaf_buckets=1"
  LABELS "integration;hamilton-tutte;scan")

add_test(
  NAME integration.ht_scan_verify
  COMMAND cudaee verify
    --tsp "${HT_RECURSIVE_TSP}"
    --edges "${HT_RECURSIVE_EDGES}"
    --proof "${HT_SCAN_PROOF}")
set_tests_properties(integration.ht_scan_verify PROPERTIES
  DEPENDS integration.ht_scan
  LABELS "integration;hamilton-tutte;scan")

set(LOCAL_ELIMINATION_EDGES "${CMAKE_CURRENT_BINARY_DIR}/tmp/recursive-point.local.edg")
set(LOCAL_ELIMINATION_PROOF "${CMAKE_CURRENT_BINARY_DIR}/tmp/recursive-point.local.proof")
set(LOCAL_ELIMINATION_REPORT "${CMAKE_CURRENT_BINARY_DIR}/tmp/recursive-point.local.report")
add_test(
  NAME integration.local_elimination
  COMMAND cudaee local-eliminate
    --tsp "${HT_RECURSIVE_TSP}"
    --edges "${HT_RECURSIVE_EDGES}"
    --output "${LOCAL_ELIMINATION_EDGES}"
    --proof "${LOCAL_ELIMINATION_PROOF}"
    --report "${LOCAL_ELIMINATION_REPORT}"
    --protected-tour "${CMAKE_CURRENT_SOURCE_DIR}/data/recursive-point.tour"
    --expected-cost 395
    --max-targets 1
    --max-ht-epochs 1
    --max-jv-rounds 100
    --target-order weight-desc
    --backend cpu
    --reply-backend cpu
    --path-append-backend cpu
    --propagation-backend cpu
    --max-neighborhood 0
    --max-cd-candidates 8
    --max-candidate-degree 0
    --max-root-replies 1000
    --max-k 3
    --max-deletion-sets 1
    --max-depth 1
    --max-states 20000
    --max-total-replies 200000
    --max-replies-per-move 1000
    --max-point-candidates 5
    --max-end-candidates 5)
set_tests_properties(integration.local_elimination PROPERTIES
  PASS_REGULAR_EXPRESSION
    "local_status=OK termination=ht-epoch-limit complete_sweep=0 profile=none stages=3 committed=8 protected_tour_checked=1"
  LABELS "integration;hamilton-tutte;local-elimination")

add_test(
  NAME integration.local_elimination_verify
  COMMAND cudaee verify
    --tsp "${HT_RECURSIVE_TSP}"
    --edges "${HT_RECURSIVE_EDGES}"
    --proof "${LOCAL_ELIMINATION_PROOF}")
set_tests_properties(integration.local_elimination_verify PROPERTIES
  DEPENDS integration.local_elimination
  LABELS "integration;hamilton-tutte;local-elimination")

add_test(
  NAME integration.local_elimination_tour
  COMMAND cudaee tour-check
    --tsp "${HT_RECURSIVE_TSP}"
    --edges "${LOCAL_ELIMINATION_EDGES}"
    --tour "${CMAKE_CURRENT_SOURCE_DIR}/data/recursive-point.tour"
    --expected-cost 395)
set_tests_properties(integration.local_elimination_tour PROPERTIES
  DEPENDS integration.local_elimination
  LABELS "integration;correctness;local-elimination")

set(LOCAL_COMPLETE_EDGES "${CMAKE_CURRENT_BINARY_DIR}/tmp/recursive-point.local-complete.edg")
set(LOCAL_COMPLETE_PROOF "${CMAKE_CURRENT_BINARY_DIR}/tmp/recursive-point.local-complete.proof")
set(LOCAL_COMPLETE_REPORT "${CMAKE_CURRENT_BINARY_DIR}/tmp/recursive-point.local-complete.report")
add_test(
  NAME integration.local_complete_sweep
  COMMAND cudaee local-eliminate
    --tsp "${HT_RECURSIVE_TSP}"
    --edges "${HT_RECURSIVE_EDGES}"
    --output "${LOCAL_COMPLETE_EDGES}"
    --proof "${LOCAL_COMPLETE_PROOF}"
    --report "${LOCAL_COMPLETE_REPORT}"
    --complete-sweep 1
    --target-batch-size 8
    --profile kh-jq
    --backend cpu
    --leaf-backend cpu
    --reply-backend cpu
    --path-append-backend cpu
    --propagation-backend cpu
    --target-workers 1
    --require-initial-edges 28
    --max-final-edges 28)
set_tests_properties(integration.local_complete_sweep PROPERTIES
  PASS_REGULAR_EXPRESSION "local_status=OK termination=converged complete_sweep=1 profile=kh-jq"
  LABELS "integration;hamilton-tutte;local-elimination;complete-sweep")

add_test(
  NAME integration.local_complete_sweep_verify
  COMMAND cudaee verify
    --tsp "${HT_RECURSIVE_TSP}"
    --edges "${HT_RECURSIVE_EDGES}"
    --proof "${LOCAL_COMPLETE_PROOF}")
set_tests_properties(integration.local_complete_sweep_verify PROPERTIES
  DEPENDS integration.local_complete_sweep
  LABELS "integration;correctness;local-elimination;complete-sweep")

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
