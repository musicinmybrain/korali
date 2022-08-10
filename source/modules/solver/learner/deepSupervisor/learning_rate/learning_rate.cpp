#include "learning_rate.hpp"
#include "../deepSupervisor.hpp"

namespace korali
{
  namespace learning_rate {
    float LearningRate::get(DeepSupervisor const *const solver) {
      return _ilr;
    };

  } // namespace learning_rate
} // namespace korali
