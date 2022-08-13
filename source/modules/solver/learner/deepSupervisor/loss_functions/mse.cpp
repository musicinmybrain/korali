#include <cmath>
#include <algorithm>
#include <cstdio>
#include <numeric>
#include "mse.hpp"
#include <execution>

namespace korali
{
  namespace loss {

    float MSE::loss(const std::vector<float>& y_true, const std::vector<float>& y_pred){
      float loss =  std::transform_reduce(
        std::execution::par_unseq,
        std::begin(y_true),
        std::end(y_true),
        std::begin(y_pred),
        float{},
        std::plus<>{},
        [] (auto y, auto yhat) { return std::pow(y - yhat, 2); });
      // float loss{};
      // for(size_t i = 0; i < y_true.size(); i++){
      //   loss += std::pow(y_true[i]- y_pred[i], 2);
      // }
      return loss / (float) y_true.size();
    }

    std::vector<float> MSE::dloss(const std::vector<float>& y_true, const std::vector<float>& y_pred){
      std::vector<float> dloss;
      dloss.reserve(y_true.size());
      std::transform(
        std::execution::par_unseq,
        std::begin(y_true),
        std::end(y_true),
        std::begin(y_pred),
        std::begin(dloss),
        [] (auto y, auto yhat) { return y-yhat; });
      return dloss;
    }

  } // namespace loss
} // namespace korali
