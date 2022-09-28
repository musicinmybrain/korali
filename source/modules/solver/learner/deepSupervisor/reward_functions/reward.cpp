#include <cmath>
#include <algorithm>
#include <cstdio>
#include <numeric>
#include "reward.hpp"
// #include <execution>



namespace korali
{
  namespace reward {
    float Reward::reward(const std::vector<std::vector<float>>& y_true, const std::vector<std::vector<float>>& y_pred){
      float reward{};
      for(size_t b = 0; b < y_true.size(); b++){
        reward += this->reward(y_true[b], y_pred[b]);
      }
      return reward / (float) y_true.size();
    }
    std::vector<std::vector<float>> Reward::dreward(const std::vector<std::vector<float>>& y_true, const std::vector<std::vector<float>>& y_pred){
        auto grad_reward = std::vector<std::vector<float>>(y_true.size(), std::vector<float>(y_true[0].size()));

        for(size_t b = 0; b < y_true.size(); b++){
          auto reward_per_batch = dreward(y_true[b], y_pred[b]);
          grad_reward[b] = reward_per_batch;
        }
        return grad_reward;
    }
  } // namespace reward
} // namespace korali
