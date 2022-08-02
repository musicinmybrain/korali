/**************************************************************
 * Implementation of different learning_rate functions
 **************************************************************/
#ifndef _KORALI_LEARNING_RATE_HPP_
#define _KORALI_LEARNING_RATE_HPP_

#include <algorithm>
#include <ranges>
#include <vector>
#include <numeric>

namespace korali
{
  namespace learning_rate {
    class LearningRate
    {
      public:
        /**
         * @brief Default destructor to avoid warnings
         */
        virtual ~LearningRate() = default;
        LearningRate(float initial) : _initial{initial} {};
        /**
          * @brief inital learning rate.
          */
        /**
         * @brief Takes a solver as input and returns the new learning rate.
         * @return learning rate, usually 0 < learning rate <= 1.
         */
        virtual float get(float epoch) const;
        // virtual float operator()(const solver::learner::DeepSupervisor& solver) = 0;
      protected:
        const float _initial;
    };
  } // namespace learning_rate
} // namespace korali

#endif
