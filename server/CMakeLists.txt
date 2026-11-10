# SPDX-FileCopyrightText: 2026 yuunnn-w
# SPDX-License-Identifier: GPL-3.0-only
cmake_minimum_required(VERSION 3.16)
project(rknn_spark_server C CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ---------------------------------------------------------------- RKNN3 SDK
# Rockchip 的 RKNN3 SDK（头文件 + librknn3_api.so / librknn3_api_rkcp.so）不在本仓库里，
# 它受 Rockchip 自己的 RKNN SDK License 约束，需要你自己下载安装（见 scripts/get_sdk.sh 与 docs/build.md）。
# 三种给法任选其一：
#   cmake -DRKNN3_SDK_ROOT=/path/to/rknn3-toolkit ..
#   export RKNN3_SDK_ROOT=/path/to/rknn3-toolkit && cmake ..
#   或者先跑 scripts/get_sdk.sh，它会克隆到仓库根的 .sdk/
if(NOT RKNN3_SDK_ROOT AND DEFINED ENV{RKNN3_SDK_ROOT})
  set(RKNN3_SDK_ROOT "$ENV{RKNN3_SDK_ROOT}")
endif()
if(NOT RKNN3_SDK_ROOT)
  set(RKNN3_SDK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.sdk")
endif()

set(RKNN3_HINTS
    "${RKNN3_SDK_ROOT}/rknn3-runtime/rknn3-api/include"
    "${RKNN3_SDK_ROOT}/include"
    "${RKNN3_SDK_ROOT}/rknn3-api/include"
    "/usr/include"
    "/usr/local/include")
set(RKNN3_LIB_HINTS
    "${RKNN3_SDK_ROOT}/rknn3-runtime/rknn3-api/lib/Linux/aarch64"
    "${RKNN3_SDK_ROOT}/lib"
    "${RKNN3_SDK_ROOT}/rknn3-runtime/rknn3-api/lib"
    "/usr/lib"
    "/usr/local/lib")

find_path(RKNN3_INCLUDE_DIR rknn3_api.h HINTS ${RKNN3_HINTS} REQUIRED)
find_library(RKNN3_API_LIB NAMES rknn3_api HINTS ${RKNN3_LIB_HINTS} REQUIRED)
# 协处理器模式（RK1820/RK1828 经 PCIe/USB/Ethernet）需要 rkcp 后端；RK3572 本机 NPU 用的是 native 后端。
find_library(RKNN3_CP_LIB NAMES rknn3_api_rkcp HINTS ${RKNN3_LIB_HINTS})
if(NOT RKNN3_CP_LIB)
  find_library(RKNN3_CP_LIB NAMES rknn3_api_native HINTS ${RKNN3_LIB_HINTS})
endif()
if(NOT RKNN3_CP_LIB)
  message(FATAL_ERROR "找不到 librknn3_api_rkcp.so（或 rknn3_api_native.so）：请设置 RKNN3_SDK_ROOT，"
                      "或先执行 scripts/get_sdk.sh 把 RKNN3 SDK 取到 .sdk/")
endif()

message(STATUS "RKNN3 include : ${RKNN3_INCLUDE_DIR}")
message(STATUS "RKNN3 lib     : ${RKNN3_API_LIB}")
message(STATUS "RKNN3 cp lib  : ${RKNN3_CP_LIB}")

add_subdirectory(thirdparty/utils)

add_executable(rknn_spark_server
    cpp/main.cpp
    cpp/engine.cpp
    cpp/httpd.cpp
)

target_include_directories(rknn_spark_server PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/tokenizer
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/float16
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/json
    ${RKNN3_INCLUDE_DIR}
)

target_link_libraries(rknn_spark_server PRIVATE
    timeutils
    fileutils
    cnpy
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/tokenizer/lib/Linux/aarch64/libtokenizer.a
    ${RKNN3_API_LIB}
    ${RKNN3_CP_LIB}
    dl
    pthread
)

target_compile_options(rknn_spark_server PRIVATE -O2 -Wall)

set_target_properties(rknn_spark_server PROPERTIES INSTALL_RPATH "$ORIGIN/lib")
install(TARGETS rknn_spark_server DESTINATION bin)
