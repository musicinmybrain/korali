#include "step_based_decay.hpp"
#include <algorithm>
#include "../deepSupervisor.hpp"

namespace korali
{
  namespace learning_rate{
    float StepDecay::get(DeepSupervisor const *const solver) {
      return _ilr*std::pow(_decay_factor, std::floor(solver->_epochCount/_epochs));
    }
  } // namespace learning_rate
} // namespace korali


