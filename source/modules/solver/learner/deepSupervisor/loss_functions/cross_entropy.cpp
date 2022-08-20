#include <cmath>
#include <algorithm>
#include <cstdio>
#include <numeric>
#include <cmath>
#include "cross_entropy.hpp"
#include <execution>

namespace korali
{
  namespace loss {
    float CrossEntropy::loss(const std::vector<float>& y_true, const std::vector<float>& y_pred){
      float loss{};
      size_t classes  = y_true.size();
      for(size_t k = 0; k < classes; k++){
        loss = std::fma(y_true[k], std::log(y_pred[k]), loss);
      }
      return loss;
    }

    std::vector<float> CrossEntropy::dloss(const std::vector<float>& y_true, const std::vector<float>& y_pred){
      std::vector<float> dloss{y_pred};
      size_t classes  = dloss.size();
      for(size_t k = 0; k < classes; k++){
        dloss[k] -= y_true[k];
      }
      return dloss;
    }

  } // namespace loss
} // namespace korali
