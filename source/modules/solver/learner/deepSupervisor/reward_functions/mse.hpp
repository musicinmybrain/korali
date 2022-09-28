/**************************************************************
 * Implementation of Mean Squared Loss Function
 **************************************************************/
#ifndef _KORALI_MSE_HPP_
#define _KORALI_MSE_HPP_
#include <vector>
#include "reward.hpp"

namespace korali
{
    namespace reward{
        class MSE: public Reward
        {
            virtual float reward(const std::vector<float>& y_true, const std::vector<float>& y_pred) override;
            virtual std::vector<float> dreward(const std::vector<float>& y_true, const std::vector<float>& y_pred) override;
        };
    } // namespace reward
} // namespace korali

#endif // _KORALI_MSE_HPP_
