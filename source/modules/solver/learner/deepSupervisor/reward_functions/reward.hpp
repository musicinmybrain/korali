/**************************************************************
 * Implementation of different reward functions
 **************************************************************/

#ifndef _KORALI_REWARD_HPP_
#define _KORALI_REWARD_HPP_

#include <algorithm>
#include <range/v3/all.hpp>
#include <vector>
#include <numeric>

namespace korali
{
  namespace reward {

    class Reward
    {
      public:

      /**
      * @brief Default destructor to avoid warnings
      */
      virtual ~Reward() = default;
      /**
      * @brief calculates the reward as an average over a batch.
      * @param y_true true prediction of size [BS, OC].
      * @param y_pred prediction of our model of size size [BS, OC].
      * @return reward over the whole batch of size float.
      */
      float reward(const std::vector<std::vector<float>>& y_true, const std::vector<std::vector<float>>& y_pred);
      /**
      * @brief calculates the derivative of the reward for each sample in a batch.
      * @param y_true true prediction of size [BS, OC].
      * @param y_pred prediction of our model of size size [BS, OC].
      * @return gradient of the reward over the whole batch of size [BS, OC].
      */
      std::vector<std::vector<float>> dreward(const std::vector<std::vector<float>>& y_true, const std::vector<std::vector<float>>& y_pred);
      /**
      * @brief calculates the reward for a sample.
      * @param y_true true prediction of size OC.
      * @param y_pred prediction of our model of size size OC.
      * @return reward for the sample of size float.
      */
      virtual float reward(const std::vector<float>& y_true, const std::vector<float>& y_pred) = 0;
      /**
      * @brief calculates the derivativ of the reward for a sample.
      * @param y_true true prediction of size OC.
      * @param y_pred prediction of our model of size size OC.
      * @return gradient of the reward for a sample of size OC.
      */
      virtual std::vector<float> dreward(const std::vector<float>& y_true, const std::vector<float>& y_pred) = 0;
    };

  } // namespace reward
} // namespace korali

#endif
