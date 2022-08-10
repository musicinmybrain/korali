#include "decay.hpp"
#include <algorithm>
#include "../deepSupervisor.hpp"

namespace korali
{
    namespace learning_rate{
      float Decay::get(DeepSupervisor const *const solver) {
        auto m = std::min(_ilr, (_ilr*_decay_factor) / (float) solver->_epochCount);
        if(_lower_bound >= _ilr)
          return m;
        else
          return std::max({m, _lower_bound});
      }
    } // namespace learning_rate
} // namespace korali


