/*
 * @Author: yangcheng 
 * @Date: 2022-09-21 14:37:21 
 * @Last Modified by: yangcheng
 * @Last Modified time: 2022-10-13 14:08:52
 */
#pragma once
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <chrono>

namespace apollo {
namespace perception {
namespace camera {

static float Sigmoid(float x) {
    if (x >= 0) {
        return 1.0f / (1.0f + std::exp(-x));
    } else {
        return std::exp(x) / (1.0f + std::exp(x));    /* to aovid overflow */
    }
}

static float Logit(float x) {
    if (x == 0) {
        return static_cast<float>(INT32_MIN);
    } else  if (x == 1) {
        return static_cast<float>(INT32_MAX);
    } else {
        return std::log(x / (1.0f - x));
    }
}


static inline float fast_exp(float x) {
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = static_cast<int32_t>((1 << 23) * (1.4426950409 * x + 126.93490512f));
    return v.f;
}

static float* SoftMaxFast(const float* src, int32_t length) {
    const float alpha = *std::max_element(src, src + length);
    float denominator{ 0 };

    float* dst;
    dst = new float[length];
    for (int32_t i = 0; i < length; ++i) {
        dst[i] = fast_exp(src[i] - alpha);
        denominator += dst[i];
    }

    for (int32_t i = 0; i < length; ++i) {
        dst[i] /= denominator;
    }
    return dst;
}

// static bool cmp(const base::TrafficLight &a, const base::TrafficLight &b) {
//     return a.detect_score > b.detect_score;
// }

static float diou(float lbox[4], float rbox[4]) {
   float interBox[] = {
        (std::max)(lbox[0] - lbox[2] / 2.f , rbox[0] - rbox[2] / 2.f), //left
        (std::min)(lbox[0] + lbox[2] / 2.f , rbox[0] + rbox[2] / 2.f), //right
        (std::max)(lbox[1] - lbox[3] / 2.f , rbox[1] - rbox[3] / 2.f), //top
        (std::min)(lbox[1] + lbox[3] / 2.f , rbox[1] + rbox[3] / 2.f), //bottom
    };

    if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS = (interBox[1] - interBox[0])*(interBox[3] - interBox[2]);
    return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
}
}  // namespace camera
}  // namespace perception
}  // namespace apollo