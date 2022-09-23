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

    float L1::penality(const std::vector<float>& weights) {
      float penality = 0;
      #pragma omp parallel for simd reduction(+:penality)
      for (size_t i = 0; i < weights.size(); i++)
        penality += std::abs(weights[i]);
      return _lambda*penality;
    }

    std::vector<float> L1::d_penality(const std::vector<float>& weights) {
      auto d_penality(weights);
      std::transform(
        // std::execution::par_unseq,
        std::begin(d_penality),
        std::end(d_penality),
        std::begin(d_penality),
        [this] (auto w) {
          if(w > 0)
            return _lambda;
          else if (w < 0)
            return -_lambda;
          else
            return 0.0f;
        } );
      return d_penality;
    }
  } // namespace regularizer
} // namespace korali
