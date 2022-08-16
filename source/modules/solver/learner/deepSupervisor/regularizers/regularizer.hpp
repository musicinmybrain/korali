/**************************************************************
 * Implementation of different regularizer functions
 **************************************************************/

#ifndef _KORALI_REGULARIZER_HPP_
#define _KORALI_REGULARIZER_HPP_

#include <algorithm>
#include <range/v3/all.hpp>
#include <vector>
#include <numeric>

namespace korali
{
  namespace regularizer {

    class Regularizer
    {
      public:

      /**
      * @brief Default destructor to avoid warnings
      */
      virtual ~Regularizer() = default;
      /**
      * @brief Takes the nn weights and calculates a penality term to be added to the loss function.
      * @param neural network weights of size weights(nn).
      * @return penality term.
      */
      virtual float penality(const std::vector<float>& weights) = 0;
      /**
      * @brief Takes the gradients of the nn weights and calculates the gradient of the penality term.
      * @param neural network weights of size weights(nn).
      * @return gradient of the penality term to be added to the jaccobian of the loss function.
      */
      virtual std::vector<float> d_penality(const std::vector<float>& d_weights) = 0;
    };

  } // namespace regularizer
} // namespace korali

#endif
