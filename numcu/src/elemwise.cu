/**
 * Elementwise operations
 */
#include "elemwise.h"

#ifndef CUVEC_DISABLE_CUDA

// dst = src_num / src_div
__global__ void div(float *dst, const float *src_num, const float *src_div, const size_t N,
                    float zeroDivDefault) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  dst[i] = (src_div[i] || zeroDivDefault == FLOAT_MAX) ? src_num[i] / src_div[i] : zeroDivDefault;
}
// dst = src_a * src_b
__global__ void mul(float *dst, const float *src_a, const float *src_b, const size_t N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  dst[i] = src_a[i] * src_b[i];
}
// dst = src_a + src_b
__global__ void add(float *dst, const float *src_a, const float *src_b, const size_t N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  dst[i] = src_a[i] + src_b[i];
}

#endif // CUVEC_DISABLE_CUDA

/// dst = src_num / src_div
void d_div(float *dst, const float *src_num, const float *src_div, const size_t N,
           float zeroDivDefault) {
#ifdef CUVEC_DISABLE_CUDA
  for (size_t i = 0; i < N; ++i)
    dst[i] =
        (src_div[i] || zeroDivDefault == FLOAT_MAX) ? src_num[i] / src_div[i] : zeroDivDefault;
#else
  dim3 thrds(NUMCU_THREADS, 1, 1);
  dim3 blcks((N + NUMCU_THREADS - 1) / NUMCU_THREADS, 1, 1);
  div<<<blcks, thrds>>>(dst, src_num, src_div, N, zeroDivDefault);
#endif
}
/// dst = src_a * src_b
void d_mul(float *dst, const float *src_a, const float *src_b, const size_t N) {
#ifdef CUVEC_DISABLE_CUDA
  for (size_t i = 0; i < N; ++i) dst[i] = src_a[i] * src_b[i];
#else
  dim3 thrds(NUMCU_THREADS, 1, 1);
  dim3 blcks((N + NUMCU_THREADS - 1) / NUMCU_THREADS, 1, 1);
  mul<<<blcks, thrds>>>(dst, src_a, src_b, N);
#endif
}
/// dst = src_a + src_b
void d_add(float *dst, const float *src_a, const float *src_b, const size_t N) {
#ifdef CUVEC_DISABLE_CUDA
  for (size_t i = 0; i < N; ++i) dst[i] = src_a[i] + src_b[i];
#else
  dim3 thrds(NUMCU_THREADS, 1, 1);
  dim3 blcks((N + NUMCU_THREADS - 1) / NUMCU_THREADS, 1, 1);
  add<<<blcks, thrds>>>(dst, src_a, src_b, N);
#endif
}
