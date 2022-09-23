#include <cmath>
#include <algorithm>
#include <cstdio>
#include <numeric>
#include "metrics.hpp"



namespace korali
{
  namespace metrics {
    float Metrics::compute(const std::vector<std::vector<float>>& y_true, const std::vector<std::vector<float>>& y_pred){
      float metrics{};
      for(size_t b = 0; b < y_true.size(); b++){
        metrics += this->compute(y_true[b], y_pred[b]);
      }
      return (float) metrics / y_true.size();
    }
  } // namespace metrics
} // namespace korali
