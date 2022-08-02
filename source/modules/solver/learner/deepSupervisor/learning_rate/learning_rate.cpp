#include "learning_rate.hpp"

namespace korali
{
  namespace learning_rate {
    float LearningRate::get(float epoch) const {
      return _initial;
    };

  } // namespace learning_rate
} // namespace korali
