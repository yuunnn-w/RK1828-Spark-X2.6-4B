/*
The MIT License

Copyright (c) Carl Rogers, 2011

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
 */

#ifndef __CNPY_H__
#define __CNPY_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define EMMC_WRITE_CHUNK_SIZE 32768

// numpy数据类型枚举
typedef enum
{
  CNPY_TYPE_FLOAT32,
  CNPY_TYPE_FLOAT64,
  CNPY_TYPE_INT8,
  CNPY_TYPE_INT16,
  CNPY_TYPE_INT32,
  CNPY_TYPE_INT64,
  CNPY_TYPE_UINT8,
  CNPY_TYPE_UINT16,
  CNPY_TYPE_UINT32,
  CNPY_TYPE_UINT64,
  CNPY_TYPE_BOOLEAN,
} cnpy_type;

// numpy数组结构体
typedef struct
{
  void* raw_data;      // 原始数据指针
  size_t    raw_data_size; // 原始数据大小
  size_t    data_begin;    // 实际数据开始位置
  size_t* shape;         // 数组形状
  size_t    ndim;          // 维度数
  cnpy_type dtype;         // 数据类型
  bool      fortran_order; // 是否是Fortran顺序
} npy_array;

// 打开numpy文件
int npy_open(const char* filepath, npy_array* arr);

// 关闭numpy数组，释放资源
void npy_close(npy_array* arr);

// 保存float数据到numpy文件
int npy_save_float_buffer_to_file(const char* filepath, float* data, uint64_t size, uint32_t* shape, uint32_t ndim);

// 保存任意类型数据到numpy文件
int npy_save_buffer_to_file(const char* filepath, void* data, uint32_t* shape, uint32_t ndim, cnpy_type dtype);

int parse_npy_header_from_mem(void* data, size_t size, npy_array* arr);

// 分段写入数据到文件的通用函数
int write_data_in_chunks(FILE* fp, const void* data, size_t total_bytes);

#ifdef __cplusplus
}
#endif
#endif /* __CNPY_H__ */