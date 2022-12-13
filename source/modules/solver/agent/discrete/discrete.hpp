/** \namespace agent
* @brief Namespace declaration for modules of type: agent.
*/

/** \file
* @brief Header file for module: Discrete.
*/

/** \dir solver/agent/discrete
* @brief Contains code, documentation, and scripts for module: Discrete.
*/

#pragma once

#include "modules/problem/reinforcementLearning/discrete/discrete.hpp"
#include "modules/solver/agent/agent.hpp"

namespace korali
{
namespace solver
{
namespace agent
{
;

/**
* @brief Class declaration for module: Discrete.
*/
class Discrete : public Agent
{
  public: 
  /**
  * @brief [Internal Use] Observed action probabilities.
  */
   std::vector<float> _observationsApproximatorActionProbabilities;
  
 
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
   * @brief Storage for the pointer to the (discrete) learning problem
   */
  problem::reinforcementLearning::Discrete *_problem;

  float calculateImportanceWeight(const std::vector<float> &action, const policy_t &curPolicy, const policy_t &oldPolicy) override;

  /**
   * @brief Calculates the gradient of importance weight wrt to NN output
   * @param curPolicy current policy object
   * @param oldPolicy old policy object from RM
   * @return gradient of importance weight wrt NN output (q_i's and inverse temperature)
   */
  std::vector<float> calculateImportanceWeightGradient(const policy_t &curPolicy, const policy_t &oldPolicy);

  /**
   * @brief Calculates the gradient of KL(p_old, p_cur) wrt to the NN output.
   * @param oldPolicy current policy object
   * @param curPolicy old policy object from RM
   * @return gradient of KL wrt curent distribution parameter (q_i's and inverse temperature)
   */
  std::vector<float> calculateKLDivergenceGradient(const policy_t &oldPolicy, const policy_t &curPolicy);

  /**
  * @brief Evaluates the log probability of a trajectory given policy hyperparameter
  * @param states Vector of states in the trajectory.
  * @param actions Vector of actions in the trajectory.
  * @param policyHyperparameter The neural network policy hyperparameter.
  * @return The log probability of the trajectory.
  */
  virtual float evaluateTrajectoryLogProbability(const std::vector<std::vector<std::vector<float>>> &states, const std::vector<std::vector<std::vector<float>>> &actions, const std::vector<float> &policyHyperparameter) override;

  /**
  * @brief Evaluates the log probability of a trajectory given observed action frequencies.
  * @param states Vector of states in the trajectory.
  * @param actions Vector of actions in the trajectory.
  * @return The log probability of the trajectory.
  */
  virtual float evaluateTrajectoryLogProbabilityWithObservedPolicy(const std::vector<std::vector<std::vector<float>>> &states, const std::vector<std::vector<std::vector<float>>> &actions) override;

  void getAction(korali::Sample &sample) override;

  virtual void initializeAgent() override;
};

} //agent
} //solver
} //korali
;
