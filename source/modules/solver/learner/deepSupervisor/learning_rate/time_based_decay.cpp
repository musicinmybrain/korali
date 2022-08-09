#include "time_based_decay.hpp"
#include <algorithm>
#include "../deepSupervisor.hpp"

namespace korali
{
  namespace learning_rate{
    float TimeDecay::get(DeepSupervisor const *const solver) {
      _prev_lr = _prev_lr*(1/(1 + _decay_factor*solver->_epochCount));
      return _prev_lr;
    }
  } // namespace learning_rate
} // namespace korali


