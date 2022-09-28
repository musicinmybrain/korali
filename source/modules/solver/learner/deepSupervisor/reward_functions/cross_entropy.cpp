#include <cmath>
#include <algorithm>
#include <cstdio>
#include <numeric>
#include <cmath>
#include "cross_entropy.hpp"
// #include <execution>

namespace korali
{
  namespace reward {
    float CrossEntropy::reward(const std::vector<float>& y_true, const std::vector<float>& y_pred){
      float reward{};
      size_t classes  = y_true.size();
      for(size_t k = 0; k < classes; k++){
        reward = std::fma(y_true[k], std::log(y_pred[k]), reward);
      }
      // loss would be -reward
      return reward;
    }

    std::vector<float> CrossEntropy::dreward(const std::vector<float>& y_true, const std::vector<float>& y_pred){
      size_t classes  = y_true.size();
      std::vector<float> dreward;
      // We calculate an reward => ytue-yhat
      for(size_t i=0; i < classes; i++){
        dreward.push_back( (2*(y_true[i]-y_pred[i])) );
      }
      return dreward;
    }

  } // namespace reward
} // namespace korali
