/**************************************************************
 * Implementation of different metrics functions
 **************************************************************/

#ifndef _KORALI_METRICS_HPP_
#define _KORALI_METRICS_HPP_

#include <algorithm>
#include <vector>
#include <numeric>

namespace korali
{
  namespace metrics {

    class Metrics
    {
      public:

      /**
      * @brief Default destructor to avoid warnings
      */
      virtual ~Metrics() = default;
      /**
      * @brief calculates the metrics as an average over a batch.
      * @param y_true true prediction of size [BS, OC].
      * @param y_pred prediction of our model of size size [BS, OC].
      * @return metrics over the whole batch of size float.
      */
      float compute(const std::vector<std::vector<float>>& y_true, const std::vector<std::vector<float>>& y_pred);
      /**
      * @brief calculates the metrics for a sample.
      * @param y_true true prediction of size OC.
      * @param y_pred prediction of our model of size size OC.
      * @return metrics for the sample of size float.
      */
      virtual float compute(const std::vector<float>& y_true, const std::vector<float>& y_pred) = 0;
    };

  } // namespace metrics
} // namespace korali

#endif
