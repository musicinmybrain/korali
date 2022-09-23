#include <cmath>
#include <algorithm>
#include <cstdio>
#include <numeric>
#include <cmath>
#include "negative_log_likelihood.hpp"
// #include <execution>

namespace korali
{
  namespace loss {
    float NLL::loss(const std::vector<float>& y_true, const std::vector<float>& y_pred){
      float loss{};
      size_t classes  = y_true.size();
      for(size_t k = 0; k < classes; k++){
        loss += y_true[k]*y_pred[k];
      }
      return -loss;
    }

    std::vector<float> NLL::dloss(const std::vector<float>& y_true, const std::vector<float>& y_pred){
      std::vector<float> dloss{y_pred};
      size_t classes  = dloss.size();
      for(size_t k = 0; k < classes; k++){
        dloss[k] = std::exp(dloss[k])-y_true[k];
      }
      return dloss;
    }

  } // namespace loss
} // namespace korali
