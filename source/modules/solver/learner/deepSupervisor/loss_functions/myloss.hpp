/**************************************************************
 * Implementation of different loss functions
 **************************************************************/

#ifndef _KORALI_LOSSES_HPP_
#define _KORALI_LOSSES_HPP_

#include <algorithm>
#include <ranges>
#include <vector>
#include <numeric>
// #include <execution>

namespace korali
{

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
  // virtual void loss(std::vector<float> &output) = 0;
  virtual void loss(const std::vector<float>& y_true, const std::vector<float>& y_pred) = 0;
};

}

#endif
