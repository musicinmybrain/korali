#include <cmath>
#include <algorithm>
#include <cstdio>
#include <numeric>
#include <functional>
#include "l2.hpp"
// #include <execution>

namespace korali
{
  namespace regularizer {

    float L2::penalty(const std::vector<float>& weights) {
      float penalty = 0;
      #pragma omp parallel for simd reduction(+:penalty)
      for (size_t i = 0; i < weights.size(); i++)
        penalty += weights[i]*weights[i];
      return 0.5f*_lambda*penalty;
    }

    std::vector<float> L2::d_penalty(const std::vector<float>& weights) {
      std:: vector<float> d_penalty(weights.size(), _lambda);
      std::transform(
        // std::execution::par_unseq,
        std::begin(weights),
        std::end(weights),
        std::begin(d_penalty),
        std::begin(d_penalty),
        std::multiplies<float>());
      return d_penalty;
    }
  } // namespace regularizer
} // namespace korali
