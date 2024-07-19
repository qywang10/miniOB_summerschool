/* Copyright (c) 2021 OceanBase and/or its affiliates. All rights reserved.
miniob is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details. */

#include <stdint.h>
#include "common/math/simd_util.h"

#if defined(USE_SIMD)

int mm256_extract_epi32_var_indx(const __m256i vec, const unsigned int i)
{
  __m128i idx = _mm_cvtsi32_si128(i);
  __m256i val = _mm256_permutevar8x32_epi32(vec, _mm256_castsi128_si256(idx));
  return _mm_cvtsi128_si32(_mm256_castsi256_si128(val));
}

int mm256_sum_epi32(const int *values, int size)
{
  // your code here
  __m256i sum_vec = _mm256_setzero_si256(); // 初始化一个全0的向量

  int i;
  for (i = 0; i <= size - 8; i += 8) {
      __m256i vec = _mm256_loadu_si256((__m256i*)&values[i]); // 加载8个32位整数
      sum_vec = _mm256_add_epi32(sum_vec, vec); // 向量加法
  }

  // 将向量中的元素相加
  int temp[8];
  _mm256_storeu_si256((__m256i*)temp, sum_vec); // 存储到临时数组中

  int sum = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];

  // 处理剩余的元素
  for (; i < size; i++) {
      sum += values[i];
  }
  return sum;
}

float mm256_sum_ps(const float *values, int size)
{
  // your code here
  __m256 sum_vec = _mm256_setzero_ps(); // 初始化一个全0的向量

  int i;
  for (i = 0; i <= size - 8; i += 8) {
      __m256 vec = _mm256_loadu_ps(&values[i]); // 加载8个32位浮点数
      sum_vec = _mm256_add_ps(sum_vec, vec); // 向量加法
  }

  // 将向量中的元素相加
  float temp[8];
  _mm256_storeu_ps(temp, sum_vec); // 存储到临时数组中

  float sum = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];

  // 处理剩余的元素
  for (; i < size; i++) {
      sum += values[i];
  }
  return sum;
}

template <typename V>
void selective_load(V *memory, int offset, V *vec, __m256i &inv)
{
  int *inv_ptr = reinterpret_cast<int *>(&inv);
  for (int i = 0; i < SIMD_WIDTH; i++) {
    if (inv_ptr[i] == -1) {
      vec[i] = memory[offset++];
    }
  }
}
template void selective_load<uint32_t>(uint32_t *memory, int offset, uint32_t *vec, __m256i &inv);
template void selective_load<int>(int *memory, int offset, int *vec, __m256i &inv);
template void selective_load<float>(float *memory, int offset, float *vec, __m256i &inv);

#endif