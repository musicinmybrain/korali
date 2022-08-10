/**************************************************************
 * Implementation of a simple learning rate decay
 **************************************************************/
#ifndef _KORALI_TIME_DECAY_HPP_
#define _KORALI_TIME_DECAY_HPP_
#include "learning_rate.hpp"
#include <limits>

namespace korali
{
    namespace learning_rate{
        class TimeDecay: public LearningRate
        {
            public:
                TimeDecay(float initial, float _decay) : LearningRate{initial}, _prev_lr{initial}, _decay_factor{_decay} {};
                /**
                 * @brief factor in the denuminator
                 */
                float _prev_lr{};
                const float _decay_factor{};
                virtual float get(DeepSupervisor const *const solver) override;
        };
    } // namespace learning_rate
} // namespace korali

#endif // _KORALI_TIME_DECAY_HPP_
