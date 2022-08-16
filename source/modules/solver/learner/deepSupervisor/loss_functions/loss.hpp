/**************************************************************
 * Implementation of different loss functions
 **************************************************************/

#ifndef _KORALI_LOSS_HPP_
#define _KORALI_LOSS_HPP_

#include <algorithm>
#include <range/v3/all.hpp>
#include <vector>
#include <numeric>

namespace korali
{
  namespace loss {

    class Loss
    {
      public:

      /**
      * @brief Default destructor to avoid warnings
      */
      virtual ~Loss() = default;
      /**
      * @brief calculates the loss as an average over a batch.
      * @param y_true true prediction of size [BS, OC].
      * @param y_pred prediction of our model of size size [BS, OC].
      * @return loss over the whole batch of size float.
      */
      float loss(const std::vector<std::vector<float>>& y_true, const std::vector<std::vector<float>>& y_pred);
      /**
      * @brief calculates the derivative of the loss for each sample in a batch.
      * @param y_true true prediction of size [BS, OC].
      * @param y_pred prediction of our model of size size [BS, OC].
      * @return gradient of the loss over the whole batch of size [BS, OC].
      */
      std::vector<std::vector<float>> dloss(const std::vector<std::vector<float>>& y_true, const std::vector<std::vector<float>>& y_pred);
      /**
      * @brief calculates the loss for a sample.
      * @param y_true true prediction of size OC.
      * @param y_pred prediction of our model of size size OC.
      * @return loss for the sample of size float.
      */
      virtual float loss(const std::vector<float>& y_true, const std::vector<float>& y_pred) = 0;
      /**
      * @brief calculates the derivativ of the loss for a sample.
      * @param y_true true prediction of size OC.
      * @param y_pred prediction of our model of size size OC.
      * @return gradient of the loss for a sample of size OC.
      */
      virtual std::vector<float> dloss(const std::vector<float>& y_true, const std::vector<float>& y_pred) = 0;
    };

  } // namespace loss
} // namespace korali

#endif
