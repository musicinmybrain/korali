/** \namespace learner
* @brief Namespace declaration for modules of type: learner.
*/

/** \file
* @brief Header file for module: DeepSupervisor.
*/

/** \dir solver/learner/deepSupervisor
* @brief Contains code, documentation, and scripts for module: DeepSupervisor.
*/

#pragma once

#include "modules/experiment/experiment.hpp"
#include "modules/neuralNetwork/neuralNetwork.hpp"
#include "modules/problem/supervisedLearning/supervisedLearning.hpp"
#include "modules/solver/learner/deepSupervisor/optimizers/fAdaBelief.hpp"
#include "modules/solver/learner/deepSupervisor/optimizers/fAdagrad.hpp"
#include "modules/solver/learner/deepSupervisor/optimizers/fAdam.hpp"
#include "modules/solver/learner/deepSupervisor/optimizers/fGradientBasedOptimizer.hpp"
#include "modules/solver/learner/deepSupervisor/optimizers/fMadGrad.hpp"
#include "modules/solver/learner/deepSupervisor/optimizers/fRMSProp.hpp"
#include "modules/solver/learner/learner.hpp"

namespace korali
{
namespace solver
{
namespace learner
{
;

/**
* @brief Class declaration for module: DeepSupervisor.
*/
class DeepSupervisor : public Learner
{
  public: 
  /**
  * @brief Specifies the operation mode for the learner.
  */
   std::string _mode;
  /**
  * @brief Sets the configuration of the hidden layers for the neural network.
  */
   knlohmann::json _neuralNetworkHiddenLayers;
  /**
  * @brief Allows setting an aditional activation for the output layer.
  */
   knlohmann::json _neuralNetworkOutputActivation;
  /**
  * @brief Sets any additional configuration (e.g., masks) for the output NN layer.
  */
   knlohmann::json _neuralNetworkOutputLayer;
  /**
  * @brief Specifies which Neural Network backend engine to use.
  */
   std::string _neuralNetworkEngine;
  /**
  * @brief Determines which optimizer algorithm to use to apply the gradients on the neural network's hyperparameters.
  */
   std::string _neuralNetworkOptimizer;
  /**
  * @brief Stores the training neural network hyperparameters (weights and biases).
  */
   std::vector<float> _hyperparameters;
  /**
  * @brief Function to calculate the difference (loss) between the NN inference and the exact solution and its gradients for optimization.
  */
   std::string _lossFunction;
  /**
  * @brief Learning rate for the underlying ADAM optimizer.
  */
   float _learningRate;
  /**
  * @brief Regulates if l2 regularization will be applied to the neural network.
  */
   int _l2RegularizationEnabled;
  /**
  * @brief Importance weight of l2 regularization.
  */
   int _l2RegularizationImportance;
  /**
  * @brief Specified by how much will the weights of the last linear transformation of the NN be scaled. A value of < 1.0 is useful for a more deterministic start.
  */
   float _outputWeightsScaling;
  /**
  * @brief Specifies in how many parts will the mini batch be split for concurrent processing. It must divide the training mini batch size perfectly.
  */
   size_t _batchConcurrency;
  /**
  * @brief [Internal Use] The output of the neural network if running on testing mode.
  */
   std::vector<std::vector<float>> _evaluation;
  /**
  * @brief [Internal Use] Current value of the loss function.
  */
   float _currentLoss;
  /**
  * @brief [Internal Use] Stores the gradient of the neural network parameters.
  */
   std::vector<float> _hyperparameterGradients;
  /**
  * @brief [Termination Criteria] Specifies the maximum number of suboptimal generations.
  */
   float _targetLoss;
  
 
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
  * @brief Runs the operation specified on the given sample. It checks recursively whether the function was found by the current module or its parents.
  * @param sample Sample to operate on. Should contain in the 'Operation' field an operation accepted by this module or its parents.
  * @param operation Should specify an operation type accepted by this module or its parents.
  * @return True, if operation found and executed; false, otherwise.
  */
  bool runOperation(std::string operation, korali::Sample& sample) override;
  

  /**
   * @brief Korali Problem for optimizing NN weights and biases
   */
  problem::SupervisedLearning *_problem;

  /**
   * @brief Korali Experiment for optimizing the NN's weights and biases
   */
  korali::Experiment _optExperiment;

  /**
   * @brief Gradient-based solver pointer to access directly (for performance)
   */
  korali::fGradientBasedOptimizer *_optimizer;

  /**
   * @brief A neural network to be trained based on inputs and solutions
   */
  NeuralNetwork *_neuralNetwork;

  std::vector<std::vector<float>> &getEvaluation(const std::vector<std::vector<std::vector<float>>> &input) override;
  std::vector<float> getHyperparameters() override;
  void setHyperparameters(const std::vector<float> &hyperparameters) override;

  void initialize() override;
  void runGeneration() override;
  void runTrainingGeneration();
  void runTestingGeneration();
  void printGenerationAfter() override;

  std::vector<float> backwardGradients(const std::vector<std::vector<float>> &gradients);

  /**
   * @brief Run the training pipeline of the network given an input and return the output.
   * @param sample A sample containing the NN's input BxTxIC (B: Batch Size, T: Time steps, IC: Input channels) and solution BxOC data (B: Batch Size, OC: Output channels)
   */
  void runTrainingOnWorker(korali::Sample &sample);

  /**
   * @brief Run the forward evaluation pipeline of the network given an input and return the output.
   * @param sample A sample containing the NN's input BxTxIC (B: Batch Size, T: Time steps, IC: Input channels)
   */
  void runEvaluationOnWorker(korali::Sample &sample);

  /**
   * @brief Update the hyperparameters for the neural network after an update for every worker.
   * @param sample A sample containing the new NN's hyperparameters
   */
  void updateHyperparametersOnWorker(korali::Sample &sample);
};

} //learner
} //solver
} //korali
;
