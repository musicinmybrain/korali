#include <cmath>
#include <algorithm>
#include <cstdio>
#include <numeric>
#include <cmath>
#include "cross_entropy.hpp"
// #include <execution>

namespace korali
{
  namespace loss {
    float CrossEntropy::loss(const std::vector<float>& y_true, const std::vector<float>& y_pred){
      float loss{};
      size_t classes  = y_true.size();
      for(size_t k = 0; k < classes; k++){
        loss = std::fma(y_true[k], std::log(y_pred[k]), loss);
      }
      return -loss;
    }

    std::vector<float> CrossEntropy::dloss(const std::vector<float>& y_true, const std::vector<float>& y_pred){
      /*
       * for(size_t k = 0; k < classes; k++){
       * dloss[k] = y_pred[k] - y_true[k];
       * */
      size_t classes  = y_true.size();
      std::vector<float> dloss;
      dloss.reserve(classes);
      std::transform(
        // std::execution::par_unseq,
        std::begin(y_true),
        std::end(y_true),
        std::begin(y_pred),
        std::begin(dloss),
        [] (auto y, auto yhat) { return yhat-y; });
      return dloss;
    }

  } // namespace loss
} // namespace korali
