#ifndef _NUMCU_ELEMWISE_H_
#define _NUMCU_ELEMWISE_H_

#include <cstddef> // size_t
#ifndef FLOAT_MAX
#include <limits>
const float FLOAT_MAX = std::numeric_limits<float>::infinity();
#endif // FLOAT_MAX

/// dst = src_num / src_div
void d_div(float *dst, const float *src_num, const float *src_div, const size_t N,
           float zeroDivDefault = FLOAT_MAX);
/// dst = src_a * src_b
void d_mul(float *dst, const float *src_a, const float *src_b, const size_t N);
/// dst = src_a + src_b
void d_add(float *dst, const float *src_a, const float *src_b, const size_t N);

#endif // _NUMCU_ELEMWISE_H_
