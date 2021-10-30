#include "engine.hpp"
#include "modules/experiment/experiment.hpp"
#include "modules/solver/learner/deepSupervisor/deepSupervisor.hpp"
#include "sample/sample.hpp"
#include <omp.h>

namespace korali
{
namespace solver
{
namespace learner
{
;

void DeepSupervisor::initialize()
{
  // Getting problem pointer
  _problem = dynamic_cast<problem::SupervisedLearning *>(_k->_problem);

  // Don't reinitialize if experiment was already initialized
  if (_k->_isInitialized == true) return;

  // Check whether the minibatch size (N) can be divided by the requested concurrency
  if (_problem->_trainingBatchSize % _trainingConcurrency > 0) KORALI_LOG_ERROR("The training concurrency requested (%lu) does not divide the training mini batch size (%lu) perfectly.", _trainingConcurrency, _problem->_trainingBatchSize);
  if (_problem->_inferenceBatchSize % _inferenceConcurrency > 0) KORALI_LOG_ERROR("The inference concurrency requested (%lu) does not divide the inference mini batch size (%lu) perfectly.", _inferenceConcurrency, _problem->_inferenceBatchSize);

  /*****************************************************************
   * Setting up Neural Networks
   *****************************************************************/

  // Configuring neural network's inputs
  knlohmann::json neuralNetworkConfig;
  neuralNetworkConfig["Type"] = "Neural Network";
  neuralNetworkConfig["Engine"] = _neuralNetworkEngine;
  neuralNetworkConfig["Timestep Count"] = _problem->_maxTimesteps;

  // Iterator for the current layer id
  size_t curLayer = 0;

  // Setting the number of input layer nodes as number of input vector size
  neuralNetworkConfig["Layers"][curLayer]["Type"] = "Layer/Input";
  neuralNetworkConfig["Layers"][curLayer]["Output Channels"] = _problem->_inputSize;
  curLayer++;

  // Adding user-defined hidden layers
  for (size_t i = 0; i < _neuralNetworkHiddenLayers.size(); i++)
  {
    neuralNetworkConfig["Layers"][curLayer]["Weight Scaling"] = _outputWeightsScaling;
    neuralNetworkConfig["Layers"][curLayer] = _neuralNetworkHiddenLayers[i];
    curLayer++;
  }

  // Adding linear transformation layer to convert hidden state to match output channels
  neuralNetworkConfig["Layers"][curLayer]["Type"] = "Layer/Linear";
  neuralNetworkConfig["Layers"][curLayer]["Output Channels"] = _problem->_solutionSize;
  neuralNetworkConfig["Layers"][curLayer]["Weight Scaling"] = _outputWeightsScaling;
  curLayer++;

  // Applying a user-defined pre-activation function
  if (_neuralNetworkOutputActivation != "Identity")
  {
    neuralNetworkConfig["Layers"][curLayer]["Type"] = "Layer/Activation";
    neuralNetworkConfig["Layers"][curLayer]["Function"] = _neuralNetworkOutputActivation;
    curLayer++;
  }

  // Applying output layer configuration
  neuralNetworkConfig["Layers"][curLayer] = _neuralNetworkOutputLayer;
  neuralNetworkConfig["Layers"][curLayer]["Type"] = "Layer/Output";

  // Instancing training neural network
  auto trainingNeuralNetworkConfig = neuralNetworkConfig;
  trainingNeuralNetworkConfig["Batch Sizes"] = {_problem->_trainingBatchSize, _problem->_inferenceBatchSize};
  trainingNeuralNetworkConfig["Mode"] = "Training";
  _neuralNetwork = dynamic_cast<NeuralNetwork *>(getModule(trainingNeuralNetworkConfig, _k));
  _neuralNetwork->applyModuleDefaults(trainingNeuralNetworkConfig);
  _neuralNetwork->setConfiguration(trainingNeuralNetworkConfig);
  _neuralNetwork->initialize();

  /*****************************************************************
   * Initializing NN hyperparameters
   *****************************************************************/

  // If the hyperparameters have not been specified, produce new initial ones
  if (_hyperparameters.size() == 0) _hyperparameters = _neuralNetwork->generateInitialHyperparameters();

  /*****************************************************************
   * Setting up weight and bias optimization experiment
   *****************************************************************/

  if (_neuralNetworkOptimizer == "Adam") _optimizer = new korali::fAdam(_hyperparameters.size());
  if (_neuralNetworkOptimizer == "AdaBelief") _optimizer = new korali::fAdaBelief(_hyperparameters.size());
  if (_neuralNetworkOptimizer == "MADGRAD") _optimizer = new korali::fMadGrad(_hyperparameters.size());
  if (_neuralNetworkOptimizer == "RMSProp") _optimizer = new korali::fRMSProp(_hyperparameters.size());
  if (_neuralNetworkOptimizer == "Adagrad") _optimizer = new korali::fAdagrad(_hyperparameters.size());

  // Setting hyperparameter structures in the neural network and optmizer
  setHyperparameters(_hyperparameters);

  // Resetting Optimizer
  _optimizer->reset();

  // Setting current loss
  _currentLoss = 0.0f;
}

void DeepSupervisor::runGeneration()
{
  // Grabbing constants
  const size_t N = _problem->_trainingBatchSize;
  const size_t OC = _problem->_solutionSize;

  _optimizer->_eta = _learningRate;

  // Creating gradient vector
  auto gradientVector = &_problem->_solutionData;

  // If we use an MSE loss function, we need to update the gradient vector with its difference with each of batch's last timestep of the NN output
  if (_lossFunction == "Mean Squared Error")
  {
    // Making a copy of the solution data for MSE calculation
    _MSEVector = _problem->_solutionData;

    // Getting a reference to the neural network output
    const auto &results = getEvaluation(_problem->_inputData);

    // Calculating gradients via the loss function
//#pragma omp parallel for simd
    for (size_t b = 0; b < N; b++)
      for (size_t i = 0; i < OC; i++)
       _MSEVector[b][i] = _MSEVector[b][i] - results[b][i];

    // Calculating loss across the batch size
    _currentLoss = 0.0;
//#pragma omp parallel for simd
    for (size_t b = 0; b < N; b++)
      for (size_t i = 0; i < OC; i++)
        _currentLoss += _MSEVector[b][i] * _MSEVector[b][i];
    _currentLoss = _currentLoss / ((float)N * 2.0f);

    // Setting gradient vector as target for gradient backward propagation
    gradientVector = &_MSEVector;
  }

  // Getting hyperparameter gradients
  auto nnHyperparameterGradients = backwardGradients(*gradientVector);

  // Passing hyperparameter gradients through a gradient descent update
  _optimizer->processResult(0.0f, nnHyperparameterGradients);

  // Getting new set of hyperparameters from Adam
  _neuralNetwork->setHyperparameters(_optimizer->_currentValue);
}

std::vector<float> DeepSupervisor::getHyperparameters()
{
  return _neuralNetwork->getHyperparameters();
}

void DeepSupervisor::setHyperparameters(const std::vector<float> &hyperparameters)
{
  // Update evaluation network
  _neuralNetwork->setHyperparameters(hyperparameters);

  // Updating optimizer's current value
  _optimizer->_currentValue = hyperparameters;
}

std::vector<std::vector<float>> &DeepSupervisor::getEvaluation(const std::vector<std::vector<std::vector<float>>> &input)
{
  // Grabbing constants
  const size_t N = input.size();

  // Checking that incoming data has a correct format
  _problem->verifyData();

  // Running the input values through the neural network
  _neuralNetwork->forward(input);

  // Returning the output values for the last given timestep
  return _neuralNetwork->getOutputValues(N);
}

std::vector<float> &DeepSupervisor::backwardGradients(const std::vector<std::vector<float>> &gradients)
{
  // Grabbing constants
  const size_t N = gradients.size();

  // Running the input values through the neural network
  _neuralNetwork->backward(gradients);

  // Getting NN hyperparameter gradients
  auto& nnHyperparameterGradients = _neuralNetwork->getHyperparameterGradients(N);

  // If required, apply L2 Normalization to the network's hyperparameters
  if (_l2RegularizationEnabled)
  {
    const auto nnHyperparameters = _neuralNetwork->getHyperparameters();
    #pragma omp parallel for simd
    for (size_t i = 0; i < nnHyperparameterGradients.size(); i++)
      nnHyperparameterGradients[i] -= _l2RegularizationImportance * nnHyperparameters[i];
  }

  // Returning the hyperparameter gradients
  return nnHyperparameterGradients;
}

void DeepSupervisor::runInferenceOnWorker(korali::Sample &sample)
{
 printf("Running Inference on Worker!\n");
 exit(0);
}

void DeepSupervisor::runTrainingStepOnWorker(korali::Sample &sample)
{

}

void DeepSupervisor::updateHyperparametersOnWorker(korali::Sample &sample)
{

}

void DeepSupervisor::printGenerationAfter()
{
  // Printing results so far
  if (_lossFunction == "Mean Squared Error") _k->_logger->logInfo("Normal", " + Training Loss: %.15f\n", _currentLoss);
  if (_lossFunction == "Direct Gradient") _k->_logger->logInfo("Normal", " + Gradient L2-Norm: %.15f\n", std::sqrt(_currentLoss));
}

void DeepSupervisor::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Current Loss"))
 {
 try { _currentLoss = js["Current Loss"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Current Loss']\n%s", e.what()); } 
   eraseValue(js, "Current Loss");
 }

 if (isDefined(js, "Normalization Means"))
 {
 try { _normalizationMeans = js["Normalization Means"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Normalization Means']\n%s", e.what()); } 
   eraseValue(js, "Normalization Means");
 }

 if (isDefined(js, "Normalization Variances"))
 {
 try { _normalizationVariances = js["Normalization Variances"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Normalization Variances']\n%s", e.what()); } 
   eraseValue(js, "Normalization Variances");
 }

 if (isDefined(js, "Neural Network", "Hidden Layers"))
 {
 _neuralNetworkHiddenLayers = js["Neural Network"]["Hidden Layers"].get<knlohmann::json>();

   eraseValue(js, "Neural Network", "Hidden Layers");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Neural Network']['Hidden Layers'] required by deepSupervisor.\n"); 

 if (isDefined(js, "Neural Network", "Output Activation"))
 {
 _neuralNetworkOutputActivation = js["Neural Network"]["Output Activation"].get<knlohmann::json>();

   eraseValue(js, "Neural Network", "Output Activation");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Neural Network']['Output Activation'] required by deepSupervisor.\n"); 

 if (isDefined(js, "Neural Network", "Output Layer"))
 {
 _neuralNetworkOutputLayer = js["Neural Network"]["Output Layer"].get<knlohmann::json>();

   eraseValue(js, "Neural Network", "Output Layer");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Neural Network']['Output Layer'] required by deepSupervisor.\n"); 

 if (isDefined(js, "Neural Network", "Engine"))
 {
 try { _neuralNetworkEngine = js["Neural Network"]["Engine"].get<std::string>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Neural Network']['Engine']\n%s", e.what()); } 
   eraseValue(js, "Neural Network", "Engine");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Neural Network']['Engine'] required by deepSupervisor.\n"); 

 if (isDefined(js, "Neural Network", "Optimizer"))
 {
 try { _neuralNetworkOptimizer = js["Neural Network"]["Optimizer"].get<std::string>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Neural Network']['Optimizer']\n%s", e.what()); } 
{
 bool validOption = false; 
 if (_neuralNetworkOptimizer == "Adam") validOption = true; 
 if (_neuralNetworkOptimizer == "AdaBelief") validOption = true; 
 if (_neuralNetworkOptimizer == "MADGRAD") validOption = true; 
 if (_neuralNetworkOptimizer == "RMSProp") validOption = true; 
 if (_neuralNetworkOptimizer == "Adagrad") validOption = true; 
 if (validOption == false) KORALI_LOG_ERROR(" + Unrecognized value (%s) provided for mandatory setting: ['Neural Network']['Optimizer'] required by deepSupervisor.\n", _neuralNetworkOptimizer.c_str()); 
}
   eraseValue(js, "Neural Network", "Optimizer");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Neural Network']['Optimizer'] required by deepSupervisor.\n"); 

 if (isDefined(js, "Hyperparameters"))
 {
 try { _hyperparameters = js["Hyperparameters"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Hyperparameters']\n%s", e.what()); } 
   eraseValue(js, "Hyperparameters");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Hyperparameters'] required by deepSupervisor.\n"); 

 if (isDefined(js, "Loss Function"))
 {
 try { _lossFunction = js["Loss Function"].get<std::string>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Loss Function']\n%s", e.what()); } 
{
 bool validOption = false; 
 if (_lossFunction == "Direct Gradient") validOption = true; 
 if (_lossFunction == "Mean Squared Error") validOption = true; 
 if (validOption == false) KORALI_LOG_ERROR(" + Unrecognized value (%s) provided for mandatory setting: ['Loss Function'] required by deepSupervisor.\n", _lossFunction.c_str()); 
}
   eraseValue(js, "Loss Function");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Loss Function'] required by deepSupervisor.\n"); 

 if (isDefined(js, "Learning Rate"))
 {
 try { _learningRate = js["Learning Rate"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Learning Rate']\n%s", e.what()); } 
   eraseValue(js, "Learning Rate");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Learning Rate'] required by deepSupervisor.\n"); 

 if (isDefined(js, "L2 Regularization", "Enabled"))
 {
 try { _l2RegularizationEnabled = js["L2 Regularization"]["Enabled"].get<int>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['L2 Regularization']['Enabled']\n%s", e.what()); } 
   eraseValue(js, "L2 Regularization", "Enabled");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['L2 Regularization']['Enabled'] required by deepSupervisor.\n"); 

 if (isDefined(js, "L2 Regularization", "Importance"))
 {
 try { _l2RegularizationImportance = js["L2 Regularization"]["Importance"].get<int>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['L2 Regularization']['Importance']\n%s", e.what()); } 
   eraseValue(js, "L2 Regularization", "Importance");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['L2 Regularization']['Importance'] required by deepSupervisor.\n"); 

 if (isDefined(js, "Output Weights Scaling"))
 {
 try { _outputWeightsScaling = js["Output Weights Scaling"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Output Weights Scaling']\n%s", e.what()); } 
   eraseValue(js, "Output Weights Scaling");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Output Weights Scaling'] required by deepSupervisor.\n"); 

 if (isDefined(js, "Training Concurrency"))
 {
 try { _trainingConcurrency = js["Training Concurrency"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Training Concurrency']\n%s", e.what()); } 
   eraseValue(js, "Training Concurrency");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Training Concurrency'] required by deepSupervisor.\n"); 

 if (isDefined(js, "Inference Concurrency"))
 {
 try { _inferenceConcurrency = js["Inference Concurrency"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Inference Concurrency']\n%s", e.what()); } 
   eraseValue(js, "Inference Concurrency");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Inference Concurrency'] required by deepSupervisor.\n"); 

 if (isDefined(js, "Termination Criteria", "Target Loss"))
 {
 try { _targetLoss = js["Termination Criteria"]["Target Loss"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Termination Criteria']['Target Loss']\n%s", e.what()); } 
   eraseValue(js, "Termination Criteria", "Target Loss");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Termination Criteria']['Target Loss'] required by deepSupervisor.\n"); 

 if (isDefined(_k->_js.getJson(), "Variables"))
 for (size_t i = 0; i < _k->_js["Variables"].size(); i++) { 
 } 
 Learner::setConfiguration(js);
 _type = "learner/deepSupervisor";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: deepSupervisor: \n%s\n", js.dump(2).c_str());
} 

void DeepSupervisor::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Neural Network"]["Hidden Layers"] = _neuralNetworkHiddenLayers;
   js["Neural Network"]["Output Activation"] = _neuralNetworkOutputActivation;
   js["Neural Network"]["Output Layer"] = _neuralNetworkOutputLayer;
   js["Neural Network"]["Engine"] = _neuralNetworkEngine;
   js["Neural Network"]["Optimizer"] = _neuralNetworkOptimizer;
   js["Hyperparameters"] = _hyperparameters;
   js["Loss Function"] = _lossFunction;
   js["Learning Rate"] = _learningRate;
   js["L2 Regularization"]["Enabled"] = _l2RegularizationEnabled;
   js["L2 Regularization"]["Importance"] = _l2RegularizationImportance;
   js["Output Weights Scaling"] = _outputWeightsScaling;
   js["Training Concurrency"] = _trainingConcurrency;
   js["Inference Concurrency"] = _inferenceConcurrency;
   js["Termination Criteria"]["Target Loss"] = _targetLoss;
   js["Current Loss"] = _currentLoss;
   js["Normalization Means"] = _normalizationMeans;
   js["Normalization Variances"] = _normalizationVariances;
 for (size_t i = 0; i <  _k->_variables.size(); i++) { 
 } 
 Learner::getConfiguration(js);
} 

void DeepSupervisor::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"L2 Regularization\": {\"Enabled\": false, \"Importance\": 0.0001}, \"Neural Network\": {\"Output Activation\": \"Identity\", \"Output Layer\": {}}, \"Termination Criteria\": {\"Target Loss\": -1.0}, \"Hyperparameters\": [], \"Output Weights Scaling\": 1.0, \"Training Concurrency\": 1, \"Inference Concurrency\": 1}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Learner::applyModuleDefaults(js);
} 

void DeepSupervisor::applyVariableDefaults() 
{

 std::string defaultString = "{}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 if (isDefined(_k->_js.getJson(), "Variables"))
  for (size_t i = 0; i < _k->_js["Variables"].size(); i++) 
   mergeJson(_k->_js["Variables"][i], defaultJs); 
 Learner::applyVariableDefaults();
} 

bool DeepSupervisor::checkTermination()
{
 bool hasFinished = false;

 if ((_k->_currentGeneration > 1) && (_targetLoss > 0.0) && (_currentLoss <= _targetLoss))
 {
  _terminationCriteria.push_back("deepSupervisor['Target Loss'] = " + std::to_string(_targetLoss) + ".");
  hasFinished = true;
 }

 hasFinished = hasFinished || Learner::checkTermination();
 return hasFinished;
}

bool DeepSupervisor::runOperation(std::string operation, korali::Sample& sample)
{
 bool operationDetected = false;

 if (operation == "Run Inference On Worker")
 {
  runInferenceOnWorker(sample);
  return true;
 }

 if (operation == "Run Training Step On Worker")
 {
  runTrainingStepOnWorker(sample);
  return true;
 }

 if (operation == "Update Hyperparameters On Worker")
 {
  updateHyperparametersOnWorker(sample);
  return true;
 }

 operationDetected = operationDetected || Learner::runOperation(operation, sample);
 if (operationDetected == false) KORALI_LOG_ERROR(" + Operation %s not recognized for problem DeepSupervisor.\n", operation.c_str());
 return operationDetected;
}

;

} //learner
} //solver
} //korali
;
