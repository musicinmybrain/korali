/** \namespace continuous
* @brief Namespace declaration for modules of type: continuous.
*/

/** \file
* @brief Header file for module: VRACER.
*/

/** \dir solver/agent/continuous/VRACER
* @brief Contains code, documentation, and scripts for module: VRACER.
*/

#pragma once

#include "modules/distribution/univariate/normal/normal.hpp"
#include "modules/problem/reinforcementLearning/continuous/continuous.hpp"
#include "modules/solver/agent/continuous/continuous.hpp"

namespace korali
{
namespace solver
{
namespace agent
{
namespace continuous
{
;

/**
* @brief Class declaration for module: VRACER.
*/
class VRACER : public Continuous
{
  public: 
  
 
  /**
  * @brief Determines whether the module can trigger termination of an experiment run.
  * @return True, if it should trigger termination; false, otherwise.
  */
  bool checkTermination() override;
  /**
  * @brief Obtains the entire current state and configuration of the module.
  * @param js JSON object onto which to save the serialized state of the module.
  */
  void getConfiguration(knlohmann::json& js) override;
  /**
  * @brief Sets the entire state and configuration of the module, given a JSON object.
  * @param js JSON object from which to deserialize the state of the module.
  */
  void setConfiguration(knlohmann::json& js) override;
  /**
  * @brief Applies the module's default configuration upon its creation.
  * @param js JSON object containing user configuration. The defaults will not override any currently defined settings.
  */
  void applyModuleDefaults(knlohmann::json& js) override;
  /**
  * @brief Applies the module's default variable configuration to each variable in the Experiment upon creation.
  */
  void applyVariableDefaults() override;
  

  /**
   * @brief Update the V-target or current and previous experiences in the episode
   * @param expId Current Experience Id
   */
  void updateVtbc(size_t expId);

  /**
   * @brief Calculates the gradients for the policy/critic neural network
   * @param miniBatch The indexes of the experience mini batch
   * @param policyIdx The indexes of the policy to compute the gradient for
   */
  void calculatePolicyGradients(const std::vector<std::pair<size_t, size_t>> &miniBatch, const size_t policyIdx);

  /**
   * @brief [Statistics] Keeps track of the mu of the current minibatch for each action variable
   */
  std::vector<float> _miniBatchPolicyMean;

  /**
   * @brief [Statistics] Keeps track of the sigma of the current minibatch for each action variable
   */
  std::vector<float> _miniBatchPolicyStdDev;

  /**
   * @brief [Statistics] Keeps track of the mu of the current minibatch for each action variable
   */
  std::vector<float> _miniBatchCurrentPolicyMean;

  /**
   * @brief [Statistics] Keeps track of the sigma of the individual policies in Bayesian learning current minibatch for each action variable
   */
  std::vector<float> _miniBatchCurrentPolicyStdDev;

  /**
   * @brief [Statistics] Keeps track of the mean of the policy gradients
   */
  std::vector<float> _miniBatchPolicyGradientMean;

  /**
   * @brief [Statistics] Keeps track of the sigma of the policy gradients
   */
  std::vector<float> _miniBatchPolicyGradientStdDev;

  /**
   * @brief [Statistics] Keeps track of the mean of the policy gradients
   */
  std::vector<float> _miniBatchKLGradientMean;

  /**
   * @brief [Statistics] Keeps track of the sigma of the policy gradients
   */
  std::vector<float> _miniBatchKLGradientStdDev;


  knlohmann::json getPolicy() override;
  void setPolicy(const knlohmann::json &hyperparameters) override;
  float calculateStateValue(const std::vector<std::vector<float>> &stateSequence, const size_t policyIdx = 0) override;
  void runPolicy(const std::vector<std::vector<std::vector<float>>> &stateSequenceBatch, std::vector<policy_t> &policy, const size_t policyIdx = 0) override;
  void gaussianPredictivePosteriorDistribution(const std::vector<std::vector<std::vector<float>>> &stateSequenceBatch, std::vector<policy_t> &curPolicy) override;
  void calculatePredictivePosteriorProbabilities(std::vector<float> &action, const std::vector<std::pair<size_t, size_t>> &miniBatch, const std::vector<std::vector<std::vector<float>>> &stateSequenceBatch, std::vector<policy_t> &curPolicy) override;
  void finalizePredictivePosterior(const std::vector<std::vector<std::vector<float>>> &stateSequenceBatch, std::vector<policy_t> &curPolicy, const size_t policyIdx) override;
  void trainPolicy() override;
  void printInformation() override;
  void initializeAgent() override;
};

} //continuous
} //agent
} //solver
} //korali
;
