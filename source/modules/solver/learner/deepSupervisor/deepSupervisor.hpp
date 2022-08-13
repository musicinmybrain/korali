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

#include <random>
#include "modules/experiment/experiment.hpp"
#include "modules/neuralNetwork/neuralNetwork.hpp"
#include "modules/problem/supervisedLearning/supervisedLearning.hpp"
#include "modules/solver/learner/learner.hpp"
#include "modules/solver/learner/deepSupervisor/optimizers/fAdaBelief.hpp"
#include "modules/solver/learner/deepSupervisor/optimizers/fAdagrad.hpp"
#include "modules/solver/learner/deepSupervisor/optimizers/fAdam.hpp"
#include "modules/solver/learner/deepSupervisor/optimizers/fGradientBasedOptimizer.hpp"
#include "modules/solver/learner/deepSupervisor/optimizers/fMadGrad.hpp"
#include "modules/solver/learner/deepSupervisor/optimizers/fRMSProp.hpp"
#include "modules/solver/learner/deepSupervisor/loss_functions/loss.hpp"
#include "modules/solver/learner/deepSupervisor/loss_functions/mse.hpp"
#include "modules/solver/learner/deepSupervisor/regularizers/regularizer.hpp"
#include "modules/solver/learner/deepSupervisor/regularizers/l2.hpp"
#include "modules/solver/learner/deepSupervisor/regularizers/l1.hpp"
#include "modules/solver/learner/deepSupervisor/learning_rate/learning_rate.hpp"
#include "modules/solver/learner/deepSupervisor/learning_rate/decay.hpp"
#include "modules/solver/learner/deepSupervisor/learning_rate/step_based_decay.hpp"
#include "modules/solver/learner/deepSupervisor/learning_rate/time_based_decay.hpp"

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
  * @brief Function to calculate the difference (loss) between the NN inference and the exact solution and its gradients for optimization.
  */
   std::string _regularizerType;
  /**
  * @brief Coefficient of the regularizer
  */
   float _regularizerCoefficient;
  /**
  * @brief Learning rate for the underlying gradient based optimizers.
  */
   float _learningRate;
  /**
  * @brief Type of the learning rate.
  */
   std::string _learningRateType;
  /**
  * @brief Factor how fast the learning rate decays.
  */
   float _learningRateDecayFactor;
  /**
  * @brief Smallest value the learning rate can obtain.
  */
   float _learningRateLowerBound;
  /**
  * @brief At what point to divide the 'Step Based' learning rate.
  */
   float _learningRateSteps;
  /**
  * @brief Whether to save the learning rate in the Results.
  */
   int _learningRateSave;
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
  * @brief Specifies whether to shuffel the input data during initialization.
  */
   int _dataInputShuffel;
  /**
  * @brief Specifies whether to shuffel the training data for each epoch.
  */
   int _dataTrainingShuffel;
  /**
  * @brief [Internal Use] The output of the neural network if running on testing mode (Npred x OC).
  */
   std::vector<std::vector<float>> _evaluation;
  /**
  * @brief [Internal Use] If given indicates to split the training data into training and validation data. 0<given<1: percentage for validation data; given>1: nb. of samples for validation split.
  */
   float _dataValidationSplit;
  /**
  * @brief [Internal Use] Provides the solution for one-step ahead prediction with layout NV*OC, where N is the batch size and OC is the vector size of the output.
  */
   std::vector<float> _validationSetSolution;
  /**
  * @brief [Internal Use] Provides the number of samples of the validation set NV.
  */
   size_t _validationSetSize;
  /**
  * @brief [Internal Use] Current value of the training loss.
  */
   float _currentTrainingLoss;
  /**
  * @brief [Internal Use] Current value of the loss on the validation set.
  */
   float _currentValidationLoss;
  /**
  * @brief [Internal Use] Value of the testing loss.
  */
   float _testingLoss;
  /**
  * @brief [Internal Use] Stores the current neural network normalization mean parameters.
  */
   std::vector<float> _normalizationMeans;
  /**
  * @brief [Internal Use] Stores the current neural network normalization variance parameters.
  */
   std::vector<float> _normalizationVariances;
  /**
  * @brief [Internal Use] Stores the current epoch number.
  */
   size_t _epochCount;
  /**
  * @brief [Termination Criteria] Specifies the maximum number of epochs to run when in training mode
  */
   size_t _epochs;
  /**
  * @brief [Termination Criteria] Specifies the maximum number of epochs to run when in training mode
  */
   float _targetLoss;
  /**
  * @brief [Termination Criteria] Stops prediction after the first generation by setting isPredictionDone to bool
  */
   int _isPredictionFinished;
  
 
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
    korali::problem::SupervisedLearning *_problem;
    /**
     * @brief Korali Experiment for optimizing the NN's weights and biases
     */
    korali::Experiment _optExperiment;
    /**
     * @brief Gradient-based solver pointer to access directly (for performance)
     */
    korali::fGradientBasedOptimizer *_optimizer;
    /**
     * @brief loss function object.
     */
    korali::loss::Loss *_loss{};
    /**
     * @brief regularizer function object.
     */
    korali::regularizer::Regularizer *_regularizer{};
    /**
     * @brief learning rate function object.
     */
    korali::learning_rate::LearningRate *_learning_rate{};
    /**
     * @brief A neural network to be trained based on inputs and solutions
     */
    korali::NeuralNetwork *_neuralNetwork;
    /**
     * @brief [Internal Use] if validation set is given.
     */
    bool _hasValidationSet{false};
    /**
     * @brief [Internal Use] number of total training/testing samples.
     */
    size_t N{};
    /**
     * @brief [Internal Use] number of timesteps.
     */
    size_t T{};
    /**
     * @brief [Internal Use] number of input channels.
     */
    size_t IC{};
    /**
     * @brief [Internal Use] number training/testing set batch size.
     */
    size_t BS{};
    /**
     * @brief [Internal Use] total work size per worker.
     */
    size_t NW{};
    /**
     * @brief [Internal Use] mini-batch size per worker.
     */
    size_t BW{};
    /**
     * @brief [Internal Use] number of output channels.
     */
    size_t OC{};
    /**
     * @brief [Internal Use] number of validation samples.
     */
    size_t NV{};
    /**
     * @brief [Internal Use] random number engine for input data.
     */
    std::mt19937 input_reng;
    /**
     * @brief [Internal Use] random number engine for solution data.
     */
    std::mt19937 solution_reng;
    /**
    * @brief nn wrapper function.
    * @details
    * 1. Forwards input through neural network.
    * 2. Obtains output values of the nn.
    * @param input 3D vector of size [N, T, IC]
    * @return returns 2D output vector of size [N, OC]
    */
    std::vector<std::vector<float>> &getEvaluation(const std::vector<std::vector<std::vector<float>>> &input) override;
    std::vector<float> getHyperparameters() override;
    void setHyperparameters(const std::vector<float> &hyperparameters) override;

    void initialize() override;
    void runGeneration() override;
    /**
    * @brief runs an epoch
    * @details Runs samples/batch_size iterations of forward/backward pass
    */
    void runEpoch();
    /**
    * @brief runs an epoch
    * @details Runs samples/batch_size iterations of forward/backward pass
    */
    void runPrediction();
    /**
    * @brief splits up the training input - a minibatch further into superminibatches and runs a forward/backward pass.
    * @details
    * 1. Runs the forward pass of the neural network to get the output
    * 2. Calculates the derivative of the output loss function given its input value
    * 3. Runs the backward pass of the neural network
    * 4. Runs the given optimizer to optimize the hyperparameters of the neural network
    * 5. Updates the hyperparameters of the neural network
    */
    void runTrainingGeneration();
    /**
    * @brief splits up the test input - further into minibatches and runs just the forward pass.
    */
    void runTestingGeneration();
    /**
    * @brief backpropagates the jaccobian of the loss function
    * @param input 2D vector of size [BS, OC]
    * @return gradients of the neural network weights for all layers (size = nn hyperparameter count).
    */
    std::vector<float> backwardGradients(const std::vector<std::vector<float>> &dloss);
    /**
    * @brief flattens 2d solution vector
    * @param 2d vector
    * @return flattend vector.
    */
    std::vector<float> flatten(const std::vector<std::vector<float>> &vec) const;
    /**
    * @brief flattens 3d input vector
    * @param 3d vector
    * @return flattend vector.
    */
    std::vector<float> flatten(const std::vector<std::vector<std::vector<float>>> &vec) const;
    /**
    * @brief de-flattens 1d input vector into a 2d vector.
    * @param 2d vector
    * @return flattend vector.
    */
    std::vector<std::vector<float>> deflatten(const std::vector<float> &vec_flat, size_t BS, size_t OC) const;
    /**
    * @brief de-flattens 1d input vector into a 3d vector.
    * @param 3d vector
    * @return flattend vector.
    */
    std::vector<std::vector<std::vector<float>>> deflatten(const std::vector<float> &vec_flat, size_t BS, size_t T, size_t IC) const;
    /**
    * @brief Run one iteration of forward backward loop on a worker
    * @param sample A sample containing the NN's input BxTxIC (B: Batch Size, T: Time steps, IC: Input channels)
    */
    void runTrainingOnWorker(korali::Sample &sample);
    /**
    * @brief Run one iteration of forward backward pass.
    * @param sample A sample containing the NN's input BxTxIC (B: Batch Size, T: Time steps, IC: Input channels)
    */
    void runForwardData(korali::Sample &sample);
    /**
    * @brief function to check whether to run more generations.
    */
    void finalize() override;
    /**
    *
    * @brief function that can be used to print after a generation.
    */
    void printGenerationBefore() override;
    /**
    * @brief function that can be used to print after a generation.
    */
    void printGenerationAfter() override;
    /**
    * @brief function that can be used to print after a run.
    */
    void printRunAfter() override;
};

} //learner
} //solver
} //korali
;
