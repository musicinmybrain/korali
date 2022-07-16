#include <cmath>
#include <algorithm>
#include <cstdio>
#include <numeric>
#include "mse.hpp"

namespace korali
{
  namespace loss {

    float MSE::loss(const std::vector<float>& y_true, const std::vector<float>& y_pred){
      return std::transform_reduce(
        // std::execution::par_unseq,
        std::begin(y_true),
        std::end(y_true),
        std::begin(y_pred),
        float{},
        std::plus<>{},
        [] (auto y, auto yhat) { return std::pow(y - yhat, 2); }) / (2.0f*y_true.size());
    }

    void MSE::dloss(const std::vector<float>& y_true, std::vector<float>& y_pred){
      std::transform(
        // std::execution::par_unseq,
        std::begin(y_true),
        std::end(y_true),
        std::begin(y_pred),
        std::begin(y_pred),
        [] (auto y, auto yhat) { return y-yhat; });
    }

  } // namespace loss
} // namespace korali
