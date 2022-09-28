/**************************************************************
 * Implementation of the Cross Entropy Reward Function
 **************************************************************/
#ifndef _KORALI_CROSS_ENTROPY_HPP_
#define _KORALI_CROSS_ENTROPY_HPP_
#include <vector>
#include "reward.hpp"

namespace korali
{
    namespace reward{
        class CrossEntropy: public Reward
        {
            public:
                CrossEntropy(bool isInputLogits = false) : _isInputLogits{isInputLogits} {};
                virtual float reward(const std::vector<float>& y_true, const std::vector<float>& y_pred) override;
                virtual std::vector<float> dreward(const std::vector<float>& y_true, const std::vector<float>& y_pred) override;
            private:
                const bool _isInputLogits;
        };
    } // namespace reward
} // namespace korali

#endif // _KORALI_CROSS_ENTROPY_HPP_
