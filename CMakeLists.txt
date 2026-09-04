cmake_minimum_required(VERSION 3.27)

project(cudaEdgeElimination VERSION 0.1.0 LANGUAGES CXX C)

option(CUDAEE_ENABLE_CUDA "编译 CUDA JV 候选后端" ON)
option(CUDAEE_ENABLE_SANITIZERS "启用 AddressSanitizer/UndefinedBehaviorSanitizer" OFF)
option(CUDAEE_ENABLE_OPENMP "启用 CPU 精确成本矩阵并行" ON)
option(CUDAEE_BUILD_TESTS "编译测试" ON)
option(CUDAEE_BUILD_BENCHMARKS "编译研究基准" OFF)
option(CUDAEE_BUILD_AUTHOR_TOOLS "编译论文作者的 KH-ElimTSP 对照工具" ON)
set(CUDAEE_CUDA_ARCHITECTURES "89" CACHE STRING "CUDA 实码架构列表")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(ZLIB QUIET)
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(MPFR REQUIRED IMPORTED_TARGET mpfr)
if(CUDAEE_ENABLE_OPENMP)
  find_package(OpenMP QUIET COMPONENTS C CXX)
endif()

set(CUDAEE_SOURCES
    src/common/types.cpp
    src/common/distance.cpp
    src/common/graph.cpp
    src/common/tour.cpp
    src/cpu/elimination_commit.cpp
    src/cpu/hamilton_tutte.cpp
    src/cpu/ht_epoch.cpp
    src/cpu/ht_trace.cpp
    src/cpu/ht_scan.cpp
    src/cpu/local_elimination.cpp
    src/cpu/hamilton_tutte_proof.cpp
    src/cpu/hamilton_tutte_recursive.cpp
    src/cpu/hamilton_tutte_transposed.cpp
    src/cpu/hamilton_tutte_wavefront.cpp
    src/cpu/transposed_leaf_broker.cpp
    src/cpu/ht_path_append.cpp
    src/cpu/jv_elimination.cpp
    src/cpu/kopt_search.cpp
    src/cpu/path_system.cpp
    src/fgpu/geometry.cpp
    src/fgpu/pdlp.cpp
    src/fgpu/quick_hs_verifier.cpp
    src/fgpu/resident_pipeline.cpp
    src/fgpu/pipeline.cpp
    src/lp/lp_epoch.cpp
    src/lp/cuopt_sidecar.cpp)

if(CUDAEE_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "CUDAEE_ENABLE_CUDA=ON，但没有找到 CUDA 编译器")
  endif()
  enable_language(CUDA)
  list(APPEND CUDAEE_SOURCES
      src/cuda/device_affinity.cu
      src/cuda/exact_tour_dp.cu
      src/cuda/ht_cd.cu
      src/cuda/ht_path_append.cu
      src/cuda/ht_replies.cu
      src/cuda/jv_cuda.cu
      src/cuda/fgpu_geometry.cu
      src/cuda/fgpu_pdlp.cu
      src/cuda/fgpu_resident.cu
      src/cuda/kopt_cost.cu
      src/cuda/path_compatibility.cu)
else()
  list(APPEND CUDAEE_SOURCES
      src/cuda/device_affinity_stub.cpp
      src/cuda/exact_tour_dp_stub.cpp
      src/cuda/ht_cd_stub.cpp
      src/cuda/ht_path_append_stub.cpp
      src/cuda/ht_replies_stub.cpp
      src/cuda/jv_cuda_stub.cpp
      src/cuda/fgpu_geometry_stub.cpp
      src/cuda/fgpu_pdlp_stub.cpp
      src/cuda/fgpu_resident_stub.cpp
      src/cuda/kopt_cost_stub.cpp
      src/cuda/path_compatibility_stub.cpp)
endif()

add_library(cudaee_core STATIC ${CUDAEE_SOURCES})
target_include_directories(cudaee_core PUBLIC include)
target_compile_features(cudaee_core PUBLIC cxx_std_20)
target_compile_definitions(cudaee_core PUBLIC CUDAEE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_link_libraries(cudaee_core PUBLIC ${CMAKE_DL_LIBS} PRIVATE Threads::Threads PkgConfig::MPFR)

if(CUDAEE_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
  target_link_libraries(cudaee_core PRIVATE OpenMP::OpenMP_CXX)
  target_compile_definitions(cudaee_core PRIVATE CUDAEE_HAS_OPENMP=1)
endif()

if(ZLIB_FOUND)
  target_link_libraries(cudaee_core PRIVATE ZLIB::ZLIB)
  target_compile_definitions(cudaee_core PRIVATE CUDAEE_HAVE_ZLIB=1)
endif()

target_compile_options(cudaee_core PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wconversion;-Wshadow;-Wformat=2>)

if(CUDAEE_ENABLE_CUDA)
  set_target_properties(cudaee_core PROPERTIES
    CUDA_ARCHITECTURES "${CUDAEE_CUDA_ARCHITECTURES}"
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION OFF)
  # 不传 --use_fast_math；默认精度路径是整数距离证明链的一部分。
  target_compile_options(cudaee_core PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  target_compile_definitions(cudaee_core PUBLIC CUDAEE_HAS_CUDA=1)
endif()

if(CUDAEE_ENABLE_SANITIZERS)
  if(CUDAEE_ENABLE_CUDA)
    message(FATAL_ERROR "Sanitizer 配置应使用 CUDAEE_ENABLE_CUDA=OFF")
  endif()
  target_compile_options(cudaee_core PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(cudaee_core PUBLIC -fsanitize=address,undefined)
endif()

add_executable(cudaee src/cli/main.cpp)
target_link_libraries(cudaee PRIVATE cudaee_core)
target_compile_options(cudaee PRIVATE -Wall -Wextra -Wconversion -Wshadow -Wformat=2)

add_executable(fgpu-elim src/cli/fgpu_main.cpp)
target_link_libraries(fgpu-elim PRIVATE cudaee_core)
target_compile_options(fgpu-elim PRIVATE -Wall -Wextra -Wconversion -Wshadow -Wformat=2)

set(CUDAEE_KH_ELIM_DIR "${CMAKE_SOURCE_DIR}/third_party/ElimTSP/KH-elim")
if(CUDAEE_BUILD_AUTHOR_TOOLS AND EXISTS "${CUDAEE_KH_ELIM_DIR}/kh-elim.c")
  # 保持作者算法为隔离的对照可执行文件；GPU 代码不直接取得其删除授权。
  add_executable(cudaee_kh_elim "${CUDAEE_KH_ELIM_DIR}/kh-elim.c")
  target_include_directories(cudaee_kh_elim PRIVATE "${CUDAEE_KH_ELIM_DIR}")
  target_link_libraries(cudaee_kh_elim PRIVATE m)
  # 作者代码的 Release 行为不定义 NDEBUG；保持其 Makefile 语义，避免改变断言路径。
  target_compile_options(cudaee_kh_elim PRIVATE -UNDEBUG -Wno-unused-result)

  if(OpenMP_C_FOUND)
    add_executable(cudaee_kh_elim_omp "${CUDAEE_KH_ELIM_DIR}/kh-elim_omp.c")
    target_include_directories(cudaee_kh_elim_omp PRIVATE "${CUDAEE_KH_ELIM_DIR}")
    target_link_libraries(cudaee_kh_elim_omp PRIVATE OpenMP::OpenMP_C m)
    target_compile_options(cudaee_kh_elim_omp PRIVATE -UNDEBUG -Wno-unused-result)
  endif()
endif()

include(GNUInstallDirs)
install(TARGETS cudaee fgpu-elim RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

if(CUDAEE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(CUDAEE_BUILD_BENCHMARKS)
  add_executable(cudaee_kopt_cost_benchmark benchmarks/kopt_cost_benchmark.cpp)
  target_link_libraries(cudaee_kopt_cost_benchmark PRIVATE cudaee_core)
  target_compile_options(cudaee_kopt_cost_benchmark PRIVATE
    -Wall -Wextra -Wconversion -Wshadow -Wformat=2)

  add_executable(cudaee_jv_benchmark benchmarks/jv_benchmark.cpp)
  target_link_libraries(cudaee_jv_benchmark PRIVATE cudaee_core)
  target_compile_options(cudaee_jv_benchmark PRIVATE
    -Wall -Wextra -Wconversion -Wshadow -Wformat=2)
endif()
