#include <cmath>
#include <algorithm>
#include <cstdio>
#include <numeric>
#include "accuracy.hpp"
#include <execution>

namespace korali
{
  namespace metrics {
    float Accuracy::compute(const std::vector<float>& y_true, const std::vector<float>& y_pred){
      auto max = max_element(y_pred.begin(), y_pred.end());
      int argmax = distance(y_pred.begin(), max);
      if(y_true[argmax] != 0.0)
        return 1;
      else
        return 0;
    }
  } // namespace metrics
} // namespace korali
