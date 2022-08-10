/**************************************************************
 * Implementation of Mean Squared Regularizer Function
 **************************************************************/
#ifndef _KORALI_L2_HPP_
#define _KORALI_L2_HPP_
#include <vector>
#include "regularizer.hpp"

namespace korali
{
    namespace regularizer{
        class L2: public Regularizer
        {
          public:
            L2(float lambda) : _lambda{lambda}{};
            /**
            * @brief importance weight
            */
            float _lambda{};
            virtual float penality(const std::vector<float>& weights) override;
            virtual std::vector<float> d_penality(const std::vector<float>& weights) override;
        };
    } // namespace loss
} // namespace korali

#endif // _KORALI_MSE_HPP_
