#include <cmath>
#include <algorithm>
#include <cstdio>
#include <numeric>
#include <functional>
#include "l1.hpp"
// #include <execution>

namespace korali
{
  namespace regularizer {

    float L1::penalty(const std::vector<float>& weights) {
      float penalty = 0;
      #pragma omp parallel for simd reduction(+:penalty)
      for (size_t i = 0; i < weights.size(); i++)
        penalty += std::abs(weights[i]);
      return _lambda*penalty;
    }

    std::vector<float> L1::d_penalty(const std::vector<float>& weights) {
      auto d_penalty(weights);
      std::transform(
        // std::execution::par_unseq,
        std::begin(d_penalty),
        std::end(d_penalty),
        std::begin(d_penalty),
        [this] (auto w) {
          if(w > 0)
            return _lambda;
          else if (w < 0)
            return -_lambda;
          else
            return 0.0f;
        } );
      return d_penalty;
    }
  } // namespace regularizer
} // namespace korali
