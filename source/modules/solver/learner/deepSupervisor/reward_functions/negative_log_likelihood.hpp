/**************************************************************
 * Implementation of the Negative Log Likelihood Reward Function
 **************************************************************/
#ifndef _KORALI_NLL_HPP_
#define _KORALI_NLL_HPP_
#include <vector>
#include "reward.hpp"

namespace korali
{
    namespace reward{
        class NLL: public Reward
        {
            virtual float reward(const std::vector<float>& y_true, const std::vector<float>& y_pred) override;
            virtual std::vector<float> dreward(const std::vector<float>& y_true, const std::vector<float>& y_pred) override;
        };
    } // namespace reward
} // namespace korali

#endif // _KORALI_NLL_HPP_
