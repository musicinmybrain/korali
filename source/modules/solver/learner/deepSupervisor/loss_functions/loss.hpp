/**************************************************************
 * Implementation of different loss functions
 **************************************************************/

#ifndef _KORALI_LOSS_HPP_
#define _KORALI_LOSS_HPP_

#include <algorithm>
#include <ranges>
#include <vector>
#include <numeric>

namespace korali
{
  namespace loss {

    class Loss
    {
      public:

      /**
      * @brief Default destructor to avoid warnings
      */
      virtual ~Loss() = default;
      /**
      * @brief Takes a sample evaluation and its gradient and calculates the next set of parameters
      * @param evaluation The value of the objective function at the current set of parameters
      * @param gradient The gradient of the objective function at the current set of parameters
      */
      virtual float loss(const std::vector<float>& y_true, const std::vector<float>& y_pred) = 0;
      virtual std::vector<float> dloss(const std::vector<float>& y_true, const std::vector<float>& y_pred) = 0;
    };

  } // namespace loss
} // namespace korali

#endif
