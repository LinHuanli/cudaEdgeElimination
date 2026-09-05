# 每次 build 都运行，但只有内容变化才更新生成头文件，避免无意义重编译。
if(NOT DEFINED CUDAEE_IDENTITY_ROOT OR NOT DEFINED CUDAEE_IDENTITY_OUTPUT)
  message(FATAL_ERROR "缺少构建身份的输入或输出目录")
endif()
execute_process(COMMAND git rev-parse HEAD WORKING_DIRECTORY "${CUDAEE_IDENTITY_ROOT}"
  OUTPUT_VARIABLE CUDAEE_IDENTITY_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE git_status)
if(NOT git_status EQUAL 0)
  set(CUDAEE_IDENTITY_COMMIT "unversioned")
endif()
execute_process(COMMAND git diff --binary HEAD -- WORKING_DIRECTORY "${CUDAEE_IDENTITY_ROOT}"
  OUTPUT_VARIABLE identity_diff RESULT_VARIABLE diff_status)
if(NOT diff_status EQUAL 0)
  set(identity_diff "git-diff-unavailable")
endif()
string(SHA256 CUDAEE_IDENTITY_DIFF_SHA256 "${identity_diff}")
execute_process(COMMAND git status --porcelain --untracked-files=normal
  WORKING_DIRECTORY "${CUDAEE_IDENTITY_ROOT}" OUTPUT_VARIABLE identity_status)
if(identity_status STREQUAL "")
  set(CUDAEE_IDENTITY_DIRTY "false")
else()
  set(CUDAEE_IDENTITY_DIRTY "true")
endif()

# 同时覆盖尚未 git add 的自有源码。只扫描白名单目录，绝不读数据集、
# 外部 references、依赖、缓存或实验结果。
file(GLOB_RECURSE identity_files RELATIVE "${CUDAEE_IDENTITY_ROOT}"
  "${CUDAEE_IDENTITY_ROOT}/src/*" "${CUDAEE_IDENTITY_ROOT}/include/*"
  "${CUDAEE_IDENTITY_ROOT}/tests/*" "${CUDAEE_IDENTITY_ROOT}/cmake/*")
list(APPEND identity_files CMakeLists.txt CMakePresets.json)
list(SORT identity_files)
set(identity_tree "")
foreach(identity_path IN LISTS identity_files)
  if(NOT IS_DIRECTORY "${CUDAEE_IDENTITY_ROOT}/${identity_path}" AND
     NOT identity_path MATCHES "(__pycache__|\\.pyc$)")
    file(SHA256 "${CUDAEE_IDENTITY_ROOT}/${identity_path}" identity_hash)
    string(APPEND identity_tree "${identity_path}:${identity_hash}\n")
  endif()
endforeach()
string(SHA256 CUDAEE_IDENTITY_SOURCE_SHA256 "${identity_tree}")
configure_file("${CUDAEE_IDENTITY_ROOT}/cmake/build_identity.hpp.in"
  "${CUDAEE_IDENTITY_OUTPUT}" @ONLY)
