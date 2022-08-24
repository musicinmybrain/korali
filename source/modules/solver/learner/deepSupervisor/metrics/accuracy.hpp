/**************************************************************
 * Implementation of Accuracy Metrics Function
 **************************************************************/
#ifndef _KORALI_ACCURACY_HPP_
#define _KORALI_ACCURACY_HPP_
#include <vector>
#include "metrics.hpp"

namespace korali
{
    namespace metrics{
        class Accuracy: public Metrics
        {
            virtual float compute(const std::vector<float>& y_true, const std::vector<float>& y_pred) override;
        };
    } // namespace metrics
} // namespace korali

#endif // _KORALI_ACCURACY_HPP_
