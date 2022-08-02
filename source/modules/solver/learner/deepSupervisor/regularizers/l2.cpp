#include <cmath>
#include <algorithm>
#include <cstdio>
#include <numeric>
#include <functional>
#include "l2.hpp"
#include <execution>

namespace korali
{
  namespace regularizer {

    float L2::penality(const std::vector<float>& weights) {
      float penality = 0;
      #pragma omp parallel for simd reduction(+:penality)
      for (size_t i = 0; i < weights.size(); i++)
        penality += weights[i]*weights[i];
      return 0.5f*_lambda*penality;
    }

    std::vector<float> L2::d_penality(const std::vector<float>& weights) {
      std:: vector<float> d_penality(weights.size(), _lambda);
      std::transform(
        std::execution::par_unseq,
        std::begin(weights),
        std::end(weights),
        std::begin(d_penality),
        std::begin(d_penality),
        std::multiplies<float>());
      return d_penality;
    }
  } // namespace regularizer
} // namespace korali
