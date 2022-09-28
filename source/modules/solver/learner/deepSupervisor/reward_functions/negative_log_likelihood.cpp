#include <cmath>
#include <algorithm>
#include <cstdio>
#include <numeric>
#include <cmath>
#include "negative_log_likelihood.hpp"
// #include <execution>

namespace korali
{
  namespace reward {
    float NLL::reward(const std::vector<float>& y_true, const std::vector<float>& y_pred){
      float loss{};
      size_t classes  = y_true.size();
      for(size_t k = 0; k < classes; k++){
        loss += y_true[k]*y_pred[k];
      }
      // loss would be -loss but we want the reward
      return loss;
    }

    std::vector<float> NLL::dreward(const std::vector<float>& y_true, const std::vector<float>& y_pred){
      std::vector<float> dreward{y_pred};
      size_t classes  = dreward.size();
      for(size_t k = 0; k < classes; k++){
        // want the reward => need negative sign
        dreward[k] = -std::exp(dreward[k])-y_true[k];
      }
      return dreward;
    }

  } // namespace reward
} // namespace korali
