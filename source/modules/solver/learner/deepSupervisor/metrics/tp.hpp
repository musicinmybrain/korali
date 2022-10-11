/**************************************************************
 * Implementation of True Positive Metrics Function
 **************************************************************/
#pragma once
#include <vector>
#include "metrics.hpp"

namespace korali
{
    namespace metrics{
        class TruePositive : public Metrics
        {
            float compute(const std::vector<std::vector<float>>& y_true, const std::vector<std::vector<float>>& y_pred) override;
            float compute(const std::vector<float>& y_true, const std::vector<float>& y_pred) override;
        };
    } // namespace metrics
} // namespace korali

