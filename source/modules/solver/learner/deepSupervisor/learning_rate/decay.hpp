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
                Decay(float initial, float _decay, float min_value) : LearningRate{initial}, _decay_factor{_decay}, _min_value{min_value} {};
                Decay(float initial, float _decay) : LearningRate{initial}, _decay_factor{_decay} {};
                /**
                 * @brief learning rate is prop to decay_factor/epoch as soon as decay_factor/epoch < inital.
                 */
                const float _decay_factor{};
                const float _min_value{std::numeric_limits<float>::max()};
                virtual float get(float epoch) const override;
                // virtual float operator()(const solver::learner::DeepSupervisor& solver) override;
        };
    } // namespace learning_rate
} // namespace korali

#endif // _KORALI_DECAY_HPP_
