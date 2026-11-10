cmake_minimum_required(VERSION 3.15)

# 本项目只需要 model-zoo utils 里的三个小工具库（图像/音频相关的部分与推理服务无关，未包含）。
# 这些源文件来自 airockchip/rknn3-model-zoo（Apache-2.0），见 ../NOTICE。

add_library(fileutils STATIC file_utils.c)
target_include_directories(fileutils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(cnpy STATIC cnpy.cpp)
target_include_directories(cnpy PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(timeutils STATIC time_utils.c)
target_include_directories(timeutils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
