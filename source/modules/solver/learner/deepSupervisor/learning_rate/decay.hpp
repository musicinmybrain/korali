/**************************************************************
 * Implementation of a simple learning rate decay
 **************************************************************/
#ifndef _KORALI_DECAY_HPP_
#define _KORALI_DECAY_HPP_
#include "learning_rate.hpp"
#include <limits>

namespace korali
{
    namespace learning_rate{
        class Decay: public LearningRate
        {
          public:
                Decay(float initial, float _decay, float lower_bound) : LearningRate{initial}, _decay_factor{_decay}, _lower_bound{lower_bound} {};
                Decay(float initial, float _decay) : LearningRate{initial}, _decay_factor{_decay} {};
                /**
                 * @brief learning rate is prop to decay_factor/epoch as soon as decay_factor/epoch < inital.
                 */
                const float _decay_factor{};
                /**
                 * @brief minimum value that the learning rate can shring to, if given.
                 */
                const float _lower_bound{std::numeric_limits<float>::max()};
                virtual float get(DeepSupervisor const *const solver) override;
        };
    } // namespace learning_rate
} // namespace korali

#endif // _KORALI_DECAY_HPP_
