/**************************************************************
 * Implementation of the Negative Log Likelihood Loss Function
 **************************************************************/
#ifndef _KORALI_NLL_HPP_
#define _KORALI_NLL_HPP_
#include <vector>
#include "loss.hpp"

namespace korali
{
    namespace loss{
        class NLL: public Loss
        {
            virtual float loss(const std::vector<float>& y_true, const std::vector<float>& y_pred) override;
            virtual std::vector<float> dloss(const std::vector<float>& y_true, const std::vector<float>& y_pred) override;
        };
    } // namespace loss
} // namespace korali

#endif // _KORALI_NLL_HPP_
