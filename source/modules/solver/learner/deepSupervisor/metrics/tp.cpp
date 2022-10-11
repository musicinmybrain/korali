#include <cmath>
#include <algorithm>
#include <cstdio>
#include <numeric>
#include "tp.hpp"

namespace korali
{
  namespace metrics {
    float TruePositive::compute(const std::vector<std::vector<float>>& y_true, const std::vector<std::vector<float>>& y_pred){
      float metrics{};
      for(size_t b = 0; b < y_true.size(); b++){
        metrics += this->compute(y_true[b], y_pred[b]);
      }
      return (float) metrics;
    }

    float TruePositive::compute(const std::vector<float>& y_true, const std::vector<float>& y_pred){
      auto max = max_element(y_pred.begin(), y_pred.end());
      int argmax = distance(y_pred.begin(), max);
      if(y_true[argmax] != 0.0)
        return 1;
      else
        return 0;
    }
  } // namespace metrics
} // namespace korali
