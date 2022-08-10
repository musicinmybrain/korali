/**************************************************************
 * Implementation of a simple learning rate decay
 **************************************************************/
#ifndef _KORALI_STEP_DECAY_HPP_
#define _KORALI_STEP_DECAY_HPP_
#include "learning_rate.hpp"
#include <limits>

namespace korali
{
    namespace learning_rate{
        class StepDecay: public LearningRate
        {
            public:
                StepDecay(float initial, float _decay, int epochs) : LearningRate{initial}, _decay_factor{_decay}, _epochs{epochs} {};
                /**
                 * @brief factor by which we divide the learning rate every epoch steps.
                 */
                const float _decay_factor{};
                const int _epochs{};
                virtual float get(DeepSupervisor const *const solver) override;
        };
    } // namespace learning_rate
} // namespace korali

#endif // _KORALI_STEP_DECAY_HPP_
