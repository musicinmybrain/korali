#include "decay.hpp"
#include <algorithm>

namespace korali
{
    namespace learning_rate{
      float Decay::get(float epoch) const {
        return std::min({_initial, _decay_factor/epoch});
      }
    } // namespace learning_rate
} // namespace korali


