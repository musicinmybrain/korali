#include <cmath>
#include <algorithm>
#include <cstdio>
#include <numeric>
#include "mse.hpp"
// #include <execution>

namespace korali
{
  namespace reward {

    float MSE::reward(const std::vector<float>& y_true, const std::vector<float>& y_pred){
      float loss{};
      for(size_t i = 0; i < y_true.size(); i++){
        loss += (y_true[i] - y_pred[i])*(y_true[i] - y_pred[i]);
      }
      // We calculate an reward => need negative loss
      return -(loss / (float) y_true.size());
    }

    std::vector<float> MSE::dreward(const std::vector<float>& y_true, const std::vector<float>& y_pred){
      size_t size = y_true.size();
      std::vector<float> dreward;
      dreward.reserve(size);
      // We calculate an reward => ytue-yhat
      for(size_t i=0; i < size; i++){
        dreward.push_back( (2*(y_true[i]-y_pred[i])) );
      }
      return dreward;
    }

  } // namespace reward
} // namespace korali
