/**************************************************************
 * Implementation of the Cross Entropy Loss Function
 **************************************************************/
#ifndef _KORALI_CROSS_ENTROPY_HPP_
#define _KORALI_CROSS_ENTROPY_HPP_
#include <vector>
#include "loss.hpp"

namespace korali
{
    namespace loss{
        class CrossEntropy: public Loss
        {
            public:
                CrossEntropy(bool isInputLogits = false) : _isInputLogits{isInputLogits} {};
                virtual float loss(const std::vector<float>& y_true, const std::vector<float>& y_pred) override;
                virtual std::vector<float> dloss(const std::vector<float>& y_true, const std::vector<float>& y_pred) override;
            private:
                const bool _isInputLogits;
        };
    } // namespace loss
} // namespace korali

#endif // _KORALI_CROSS_ENTROPY_HPP_
