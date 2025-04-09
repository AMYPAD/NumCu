/**
 * Elementwise operations
 */
#include "elemwise.h"
#include <stdexcept> // std::invalid_argument

#ifndef CUVEC_DISABLE_CUDA

// dst = src_num / src_div
__global__ void knlDiv(float *dst, const float *src_num, const float *src_div, const size_t N,
                       float zeroDivDefault) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  dst[i] = (src_div[i] || zeroDivDefault == FLOAT_MAX) ? src_num[i] / src_div[i] : zeroDivDefault;
}
// dst = src_a * src_b
__global__ void knlMul(float *dst, const float *src_a, const float *src_b, const size_t N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  dst[i] = src_a[i] * src_b[i];
}
// dst = src_a + src_b
__global__ void knlAdd(float *dst, const float *src_a, const float *src_b, const size_t N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  dst[i] = src_a[i] + src_b[i];
}

#endif // CUVEC_DISABLE_CUDA

/// dst = src_num / src_div
void div(float *dst, const float *src_num, const float *src_div, const size_t N,
         float zeroDivDefault) {
#ifndef CUVEC_DISABLE_CUDA
  cudaPointerAttributes attr;
  cudaPointerGetAttributes(&attr, dst);
  dim3 thrds(NUMCU_THREADS, 1, 1);
  dim3 blcks((N + NUMCU_THREADS - 1) / NUMCU_THREADS, 1, 1);
  switch (attr.type) {
  case cudaMemoryTypeHost:
  case cudaMemoryTypeUnregistered:
#endif
    for (size_t i = 0; i < N; ++i)
      dst[i] =
          (src_div[i] || zeroDivDefault == FLOAT_MAX) ? src_num[i] / src_div[i] : zeroDivDefault;
#ifndef CUVEC_DISABLE_CUDA
    break;
  case cudaMemoryTypeDevice:
  case cudaMemoryTypeManaged:
    knlDiv<<<blcks, thrds>>>(dst, src_num, src_div, N, zeroDivDefault);
    break;
  default:
    throw std::invalid_argument("unknown memory type");
    break;
  }
#endif
}
/// dst = src_a * src_b
void mul(float *dst, const float *src_a, const float *src_b, const size_t N) {
#ifndef CUVEC_DISABLE_CUDA
  cudaPointerAttributes attr;
  cudaPointerGetAttributes(&attr, dst);
  dim3 thrds(NUMCU_THREADS, 1, 1);
  dim3 blcks((N + NUMCU_THREADS - 1) / NUMCU_THREADS, 1, 1);
  switch (attr.type) {
  case cudaMemoryTypeHost:
  case cudaMemoryTypeUnregistered:
#endif
    for (size_t i = 0; i < N; ++i) dst[i] = src_a[i] * src_b[i];
#ifndef CUVEC_DISABLE_CUDA
    break;
  case cudaMemoryTypeDevice:
  case cudaMemoryTypeManaged:
    knlMul<<<blcks, thrds>>>(dst, src_a, src_b, N);
    break;
  default:
    throw std::runtime_error("unknown memory type");
    break;
  }
#endif
}
/// dst = src_a + src_b
void add(float *dst, const float *src_a, const float *src_b, const size_t N) {
#ifndef CUVEC_DISABLE_CUDA
  cudaPointerAttributes attr;
  cudaPointerGetAttributes(&attr, dst);
  dim3 thrds(NUMCU_THREADS, 1, 1);
  dim3 blcks((N + NUMCU_THREADS - 1) / NUMCU_THREADS, 1, 1);
  switch (attr.type) {
  case cudaMemoryTypeHost:
  case cudaMemoryTypeUnregistered:
#endif
    for (size_t i = 0; i < N; ++i) dst[i] = src_a[i] + src_b[i];
#ifndef CUVEC_DISABLE_CUDA
    break;
  case cudaMemoryTypeDevice:
  case cudaMemoryTypeManaged:
    knlAdd<<<blcks, thrds>>>(dst, src_a, src_b, N);
    break;
  default:
    throw std::runtime_error("unknown memory type");
    break;
  }
#endif
}
