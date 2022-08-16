#include <cmath>
#include <algorithm>
#include <cstdio>
#include <numeric>
#include "loss.hpp"
#include <execution>



namespace korali
{
  namespace loss {

    float Loss::loss(const std::vector<std::vector<float>>& y_true, const std::vector<std::vector<float>>& y_pred){
      // return std::transform_reduce(
      //   std::execution::par_unseq,
      //   std::begin(y_true),
      //   std::end(y_true),
      //   std::begin(y_pred),
      //   float{},
      //   std::plus<>{},
      //   [this] (auto y, auto yhat) { return loss(y, yhat);} ) / (float) y_true.size();
      float loss{};
      for(size_t b = 0; b < y_true.size(); b++){
        loss += this->loss(y_true[b], y_pred[b]);
      }
      return loss / (float) y_true.size();
    }
    std::vector<std::vector<float>> Loss::dloss(const std::vector<std::vector<float>>& y_true, const std::vector<std::vector<float>>& y_pred){
        auto grad_loss = std::vector<std::vector<float>>(y_true.size(), std::vector<float>(y_true[0].size()));
        std::transform(
          std::execution::par_unseq,
          std::begin(y_true),
          std::end(y_true),
          std::begin(y_pred),
          std::begin(grad_loss),
          [this] (auto y, auto yhat) { return dloss(y, yhat);} );
        return grad_loss;
    }
  } // namespace loss
} // namespace korali
