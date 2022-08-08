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
  namespace solver {
    namespace learner {
      class DeepSupervisor;
    }
  }
  // Need forward decleration to pass deepSupervisor to get function
  namespace learning_rate {
    using korali::solver::learner::DeepSupervisor;
    class LearningRate
    {
      public:
        /**
         * @brief Default destructor to avoid warnings
         */
        virtual ~LearningRate() = default;
        LearningRate(float initial_learning_rate) : _ilr{initial_learning_rate} {};
        /**
          * @brief inital learning rate.
          */
        /**
         * @brief Takes a solver as input and returns the new learning rate.
         * @return learning rate, usually 0 < learning rate <= 1.
         */
        virtual float get(DeepSupervisor const *const solver);
      protected:
        const float _ilr;
    };
  } // namespace learning_rate
} // namespace korali

#endif
