#include "engine.hpp"
#include "modules/experiment/experiment.hpp"
#include "modules/solver/learner/deepSupervisor/deepSupervisor.hpp"
#include "sample/sample.hpp"
#include <random>
#include <cmath>
#include <cstdio>
#include <omp.h>
#include <execution>
#include <range/v3/view/join.hpp>
#include <range/v3/range/conversion.hpp>

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

  if(_dataInputShuffel){
    std::random_device r;
    std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
    // create two random engines with the same state
    input_reng = std::mt19937(seed);
    solution_reng = input_reng;
    shuffle(_problem->_inputData.begin(), _problem->_inputData.end(), input_reng);
    shuffle(_problem->_solutionData.begin(), _problem->_solutionData.end(), solution_reng);
  }
  // Data ========================================================================
  std::vector<size_t> batchSizes;
  // VALIDATION SET ======
  _hasValidationSet = (_problem->_dataValidationInput.size() || _dataValidationSplit ) ? true : false;
  if(_hasValidationSet){
    NV = _problem->_dataValidationInput.size();
    if((_problem->_dataValidationInput.size() || _problem->_dataValidationSolution.size()) && _dataValidationSplit )
      KORALI_LOG_ERROR("You cannot have a validation set as input as well as a validation split to split the training data.");
    // Validation Split is given ===========================================================================
    if(_dataValidationSplit) {
      if(0 < _dataValidationSplit &&  _dataValidationSplit < 1){
        // Choose fraction from training data for validation split
        NV = (int) _problem->_inputData.size()*_dataValidationSplit;
        N = _problem->_inputData.size()-NV;
        _problem->_dataValidationInput = {_problem->_inputData.begin()+N, _problem->_inputData.end()};
        _problem->_dataValidationSolution = {_problem->_solutionData.begin()+N, _problem->_solutionData.end()};
        _problem->_inputData = {_problem->_inputData.begin(), _problem->_inputData.begin()+N};
        _problem->_solutionData = {_problem->_solutionData.begin(), _problem->_solutionData.begin()+N};
      } else if(1 < _dataValidationSplit &&  _problem->_inputData.size() > _dataValidationSplit){
        // Choose samples from training data for validation split
        NV = _dataValidationSplit;
        N = _problem->_inputData.size()-NV;
        if(N < 1)
          KORALI_LOG_ERROR("Validation Split to large, no training samples left.");
        _problem->_dataValidationInput = {_problem->_inputData.begin()+N, _problem->_inputData.end()};
        _problem->_dataValidationSolution = {_problem->_solutionData.begin()+N, _problem->_solutionData.end()};
        _problem->_inputData = {_problem->_inputData.begin(), _problem->_inputData.begin()+N};
        _problem->_solutionData = {_problem->_solutionData.begin(), _problem->_solutionData.begin()+N};
      }
      assert(NV == _problem->_dataValidationInput.size());
      assert(NV == _problem->_dataValidationSolution.size());
      assert(N == _problem->_inputData.size());
      assert(N == _problem->_solutionData.size());
    }
    // =====================================================================================================
    batchSizes.push_back(NV);
    if (_batchConcurrency > 1){
      // If we parallize by _batchConcurrency workers, we need to support the split up batch size as well
      if (NV % _batchConcurrency > 0) KORALI_LOG_ERROR("The batch concurrency requested (%lu) does not divide the validation set size (%lu) perfectly.", _batchConcurrency, NV);
      batchSizes.push_back(NV / _batchConcurrency);
    }
    (*_k)["Results"]["Validation Loss"] = true;
  }
  // =============================================================================
  if (!_problem->_trainingBatchSize) KORALI_LOG_ERROR("Training Batch Size is not set.");
  // Check whether the minibatch size (N) can be divided by the requested concurrency TODO make this a warning and add batch size for reminder Need to also adapt verifyData() of supervisedLearning problem.
  if (_problem->_trainingBatchSize % _batchConcurrency > 0) KORALI_LOG_ERROR("The training concurrency requested (%lu) does not divide the training mini batch size (%lu) perfectly.", _batchConcurrency, _problem->_trainingBatchSize);
  // BATCH SIZES needed for the neual network architecture ==========================================
  // TRAINING ========
  batchSizes.push_back(_problem->_trainingBatchSize);
  // If we parallize by _batchConcurrency workers, we need to support the split up batch size as well
  if (_batchConcurrency > 1) batchSizes.push_back(_problem->_trainingBatchSize / _batchConcurrency);
  // TESTING =========
  if (!_problem->_testingBatchSizes.empty()){
    // Check whether the minibatch size (N) can be divided by the requested concurrency TODO make this a warning and add batch size for reminder
    for (auto bs : _problem->_testingBatchSizes){
      if (bs % _batchConcurrency > 0)
        KORALI_LOG_ERROR("The Testing concurrency requested (%lu) does not divide the training mini batch size (%lu) perfectly.", _batchConcurrency, bs);
      batchSizes.push_back(bs);
      // If we parallize by _batchConcurrency workers, we need to support the split up batch size as well
      if (_batchConcurrency > 1)
        batchSizes.push_back(bs / _batchConcurrency);
    }
  } else {
    if (_problem->_testingBatchSize){
      // Check whether the minibatch size (N) can be divided by the requested concurrency TODO make this a warning and add batch size for reminder
      if (_problem->_testingBatchSize % _batchConcurrency > 0)
        KORALI_LOG_ERROR("The Testing concurrency requested (%lu) does not divide the training mini batch size (%lu) perfectly.", _batchConcurrency, _problem->_testingBatchSize);
      batchSizes.push_back(_problem->_testingBatchSize);
      // If we parallize by _batchConcurrency workers, we need to support the split up batch size as well
      if (_batchConcurrency > 1) batchSizes.push_back(_problem->_testingBatchSize / _batchConcurrency);
    }
  }
  if(_learningRateSave)
    (*_k)["Results"]["Learning Rate"] = true;
  // ================================================================================================

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

  if (!_neuralNetworkOutputLayer.empty() && isDefined(_neuralNetworkOutputLayer, "Type")){
    neuralNetworkConfig["Layers"][curLayer] = _neuralNetworkOutputLayer;
    curLayer++;
  } else if(_neuralNetworkOutputLayer.empty() && !isDefined(_neuralNetworkOutputLayer, "Type")){
    // If no output layer is defined add a linear transformation layer to convert hidden state to match output channels
    neuralNetworkConfig["Layers"][curLayer]["Type"] = "Layer/Linear";
    neuralNetworkConfig["Layers"][curLayer]["Output Channels"] = _problem->_solutionSize;
    neuralNetworkConfig["Layers"][curLayer]["Weight Scaling"] = _outputWeightsScaling;
    curLayer++;
  }
  // Applying a user-defined pre-activation function
  if (_neuralNetworkOutputActivation != "Identity")
  {
    // If and output activation is defined add it
    neuralNetworkConfig["Layers"][curLayer]["Type"] = "Layer/Activation";
    neuralNetworkConfig["Layers"][curLayer]["Function"] = _neuralNetworkOutputActivation;
    curLayer++;
  }
  if (!_neuralNetworkOutputLayer.empty() && !isDefined(_neuralNetworkOutputLayer, "Type")){
    // Applying output layer configuration in case of transformation masks
    neuralNetworkConfig["Layers"][curLayer] = _neuralNetworkOutputLayer;
  }
  neuralNetworkConfig["Layers"][curLayer]["Type"] = "Layer/Output";

  // Instancing training neural network
  auto trainingNeuralNetworkConfig = neuralNetworkConfig;
  trainingNeuralNetworkConfig["Batch Sizes"] = batchSizes;
  trainingNeuralNetworkConfig["Mode"] = "Training";
  _neuralNetwork = dynamic_cast<NeuralNetwork *>(getModule(trainingNeuralNetworkConfig, _k));
  _neuralNetwork->applyModuleDefaults(trainingNeuralNetworkConfig);
  _neuralNetwork->setConfiguration(trainingNeuralNetworkConfig);
  _neuralNetwork->initialize();
  (*_k)["Results"]["Description"] = _problem->_description;
  /*****************************************************************************
   * Setting up the LOSS FUNCTION
   *****************************************************************************/
  if (_lossFunction == "Direct Gradient" || _lossFunction.empty())
    _loss = NULL;
  else if (_lossFunction == "Mean Squared Error")
    _loss = new korali::loss::MSE();
  else
    KORALI_LOG_ERROR("Unkown Loss Function %s", _lossFunction.c_str());
  if(_loss)
    (*_k)["Results"]["Loss Function"] = _lossFunction;
  /*****************************************************************************
   * Setting up possible REGULARIZERS
   *****************************************************************************/
  if(_regularizerType == "None" || _regularizerType.empty())
    _regularizer = NULL;
  else if (_regularizerType == "L1")
    _regularizer = new korali::regularizer::L1(_regularizerCoefficient);
  else if (_regularizerType == "L2")
    _regularizer = new korali::regularizer::L2(_regularizerCoefficient);
  else
    KORALI_LOG_ERROR("Unkown Regularizer Type %s", _regularizerType.c_str());
  if(_regularizer)
    (*_k)["Results"]["Regularizer"]["Type"] = _regularizerType;
  /*****************************************************************************
   * Setting up the LEARNING RATE (function)
   *****************************************************************************/
  if (_learningRateType == "Const" || _learningRateType.empty()){
    _learning_rate = new korali::learning_rate::LearningRate(_learningRate);
  } else if (_learningRateType == "Step Based"){
    _learning_rate = new korali::learning_rate::StepDecay(_learningRate, _learningRateDecayFactor, _learningRateSteps);
  } else if (_learningRateType == "Time Based"){
    _learning_rate = new korali::learning_rate::TimeDecay(_learningRate, _learningRateDecayFactor);
  }
    // else if (_learningRateType == "Decay"){
    //   if(_learningRateLowerBound)
    //     _learning_rate = new korali::learning_rate::Decay(_learningRate, _learningRateDecayFactor, _learningRateLowerBound);
    //   else
      // _learning_rate = new korali::learning_rate::Decay(_learningRate, _learningRateDecayFactor);
    // } else if (_learningRateType == "Custom")
    // TODO possibility to add custom functions
  else
    KORALI_LOG_ERROR("Unkown learning rate type provided %s", _learningRateType.c_str());
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
  _currentTrainingLoss = 0.0f;
  _currentValidationLoss = 0.0f;
}

void DeepSupervisor::runGeneration()
{
  if (_mode == "Training" || _mode == "Automatic Training") runEpoch();
  if (_mode == "Predict") runPrediction();
  if (_mode == "Testing") runPrediction();
  if(_mode == "Predict" or _mode == "Testing" or _mode == "Training")
    _isOneEpochFinished = true;
}

void DeepSupervisor::runEpoch()
{
    // Check whether training concurrency exceeds the number of workers
    if (_batchConcurrency > _k->_engine->_conduit->getWorkerCount()) KORALI_LOG_ERROR("The batch concurrency requested (%lu) exceeds the number of Korali workers defined in the conduit type/configuration (%lu).", _batchConcurrency, _k->_engine->_conduit->getWorkerCount());
    // Updating solver's learning rate, if changed
    _optimizer->_eta = std::max(_learning_rate->get(this), _learningRateLowerBound);
    if(_learningRateSave)
      (*_k)["Results"]["Learning Rate"] = _optimizer->_eta;
    // Checking that incoming data has a correct format
    _problem->verifyData();
    // Data ========================================================================
    BS = _problem->_trainingBatchSize;
    NW = BS / _batchConcurrency;
    N = _problem->_inputData.size();
    T = _problem->_inputData[0].size();
    IC = _problem->_inputData[0][0].size();
    OC = _problem->_solutionData[0].size();
    // Remainder for unequal batch sizes
    // size_t remainder = N % BS;
    // Iterations for epoch (without remainder)
    // =============================================================================
    size_t IforE = N / BS;
    const auto nnHyperparameters = _neuralNetwork->getHyperparameters();
    if(_dataTrainingShuffel){
      shuffle(_problem->_inputData.begin(), _problem->_inputData.end(), input_reng);
      shuffle(_problem->_solutionData.begin(), _problem->_solutionData.end(), solution_reng);
    }
    auto inputDataFlat = flatten(_problem->_inputData);
    auto solutionDataFlat = flatten(_problem->_solutionData);
#ifdef DEBUG
    if(std::any_of(inputDataFlat.begin(), inputDataFlat.end(), [](const float v) { return !std::isfinite(v);}))
      KORALI_LOG_ERROR("Non finite input training values");
    if(std::any_of(solutionDataFlat.begin(), solutionDataFlat.end(), [](const float v) { return !std::isfinite(v);}))
      KORALI_LOG_ERROR("Non finite input solution values");
#endif
    // Running epochs
    size_t bId; // batch id
    size_t wId; // worker id
    std::vector<Sample> samples(_batchConcurrency);
    auto nnHyperparameterGradients = std::vector<float>(_neuralNetwork->_hyperparameterCount, 0.0f);
    _currentTrainingLoss = 0.0f;
    size_t input_size_per_BS = T*OC;
    size_t solution_size_per_BS = T*IC;
    for (bId = 0; bId < IforE; bId++)
    {
      for (wId = 0; wId < _batchConcurrency; wId++)
      {
        // TODO: add different distribution strategies for workers
        // ==========================================================================================================
        samples[wId]["Sample Id"] = wId;
        samples[wId]["Module"] = "Solver";
        samples[wId]["Operation"] = "Run Training On Worker";
        // Problem: wrong sizes here!!!
        // wId*NW*input_size_per_BS, (wId+1)*NW*input_size_per_BS
        // samples[wId]["Input Data"] = std::vector<float>(inputDataFlat.begin()+bId*BS*input_size_per_BS+wId*NW*input_size_per_BS, inputDataFlat.begin()+bId*BS*input_size_per_BS+(wId+1)*NW*input_size_per_BS);
        // samples[wId]["Solution Data"] = std::vector<float>(solutionDataFlat.begin()+bId*BS*solution_size_per_BS+wId*NW*solution_size_per_BS, solutionDataFlat.begin()+bId*BS*solution_size_per_BS+(wId+1)*NW*solution_size_per_BS);
        samples[wId]["Hyperparameters"] = nnHyperparameters;
        if(_batchConcurrency==1){
          samples[wId]["Input Dims"] = std::vector<size_t> {BS, T, IC};
          samples[wId]["Solution Dims"] = std::vector<size_t> {BS, OC};
          samples[wId]["Input Data"] = std::vector<float>(inputDataFlat.begin()+bId*BS*input_size_per_BS, inputDataFlat.begin()+(bId+1)*BS*input_size_per_BS);
          samples[wId]["Solution Data"] = std::vector<float>(solutionDataFlat.begin()+bId*BS*solution_size_per_BS, solutionDataFlat.begin()+(bId+1)*BS*solution_size_per_BS);
        } else{
          samples[wId]["Input Dims"] = std::vector<size_t> {NW, T, IC};
          samples[wId]["Solution Dims"] = std::vector<size_t> {NW, OC};
          samples[wId]["Input Data"] = std::vector<float>(inputDataFlat.begin()+(bId*BS+wId*NW)*input_size_per_BS, inputDataFlat.begin()+(bId*BS+(wId+1)*NW)*input_size_per_BS);
          samples[wId]["Solution Data"] = std::vector<float>(solutionDataFlat.begin()+(bId*BS + wId*NW)*solution_size_per_BS, solutionDataFlat.begin()+(bId*BS + (wId+1)*NW)*solution_size_per_BS);
        }
      }
      if(_batchConcurrency > 1){
        for (wId = 0; wId < _batchConcurrency; wId++) KORALI_START(samples[wId]);
        // Waiting for samples to finish
        KORALI_WAITALL(samples);
      } else{
        // Do not run on other workers
        runTrainingOnWorker(samples[0]);
      }
      for (wId = 0; wId < _batchConcurrency; wId++){
        if(_loss)
          _currentTrainingLoss += KORALI_GET(float, samples[wId], "Training Loss");
        const auto dloss = KORALI_GET(std::vector<float>, samples[wId], "Hyperparameter Gradients");
        assert(dloss.size() ==  nnHyperparameterGradients.size());
        // Calculate the sum of the gradient batches/mean would only change the learning rate.
        std::transform(nnHyperparameterGradients.begin(), nnHyperparameterGradients.end(), dloss.begin(), nnHyperparameterGradients.begin(), std::plus<float>());
      }
    }
    (*_k)["Results"]["Epoch"] = _epochCount;
    // TODO: take care of remainder ==========================================================================================
    // =======================================================================================================================
    // Calculate the average validation loss
    if(_hasValidationSet && _loss){
      _currentValidationLoss = 0.0f;
      auto y_val = getEvaluation(_problem->_dataValidationInput);
      _currentValidationLoss = _loss->loss(y_val, _problem->_dataValidationSolution);
      (*_k)["Results"]["Validation Loss"] = _currentValidationLoss;
    }
    if(_regularizer){
      _currentTrainingLoss += _regularizer->penality(_neuralNetwork->getHyperparameters());
      auto d_penalty = _regularizer->d_penality(_neuralNetwork->getHyperparameters());
      std::transform(std::execution::par_unseq, std::begin(nnHyperparameterGradients), std::end(nnHyperparameterGradients), std::begin(d_penalty), std::begin(nnHyperparameterGradients), std::minus<float>());
    }
    if(_loss){
      _currentTrainingLoss = _currentTrainingLoss/ (float)(_batchConcurrency*IforE);
      (*_k)["Results"]["Training Loss"] = _currentTrainingLoss;
    }
    // // Passing hyperparameter gradients through a gradient descent update
    _optimizer->processResult(0.0f, nnHyperparameterGradients);
    // // Getting new set of hyperparameters from the gradient descent algorithm
    auto &new_hyperparameters = _optimizer->_currentValue;
    _hyperparameters = new_hyperparameters;
    _neuralNetwork->setHyperparameters(new_hyperparameters);
    ++_epochCount;
    (*_k)["Results"]["Mode"] = _mode;
}

void DeepSupervisor::runPrediction()
{
    // Check whether training concurrency exceeds the number of workers
    if (_batchConcurrency > _k->_engine->_conduit->getWorkerCount()) KORALI_LOG_ERROR("The batch concurrency requested (%lu) exceeds the number of Korali workers defined in the conduit type/configuration (%lu).", _batchConcurrency, _k->_engine->_conduit->getWorkerCount());
    // Checking that incoming data has a correct format

    if(_problem->_testingBatchSize && !_problem->_testingBatchSizes.empty()){
      if(std::find(_problem->_testingBatchSizes.begin(), _problem->_testingBatchSizes.end(), _problem->_testingBatchSize) == _problem->_testingBatchSizes.end())
        KORALI_LOG_ERROR("Testing Batch size (%lu) different than that of any of the confiure testing batch sizes.\n", _problem->_testingBatchSize);
    }
    if(!_problem->_testingBatchSize){
      _k->_logger->logWarning("Normal","'Testing Batch Size' has not been defined.\n");
      if(!_problem->_inputData.size()){
        KORALI_LOG_ERROR("No input supplied for prediction.");
      }
      BS = _problem->_inputData.size();
    } else{
      // Checking that incoming data has a correct format
      if(std::find(_problem->_testingBatchSizes.begin(), _problem->_testingBatchSizes.end(), _problem->_inputData.size()) == _problem->_testingBatchSizes.end())
        KORALI_LOG_ERROR("Testing Batch sizes different than that of input data (%lu).\n", _problem->_inputData.size());
        // KORALI_LOG_ERROR("Testing Batch size %lu different than that of input data (%lu).\n", _problem->_testingBatchSize, _problem->_inputData.size());
      BS = _problem->_inputData.size();
    }
    // Data ========================================================================
    N = _problem->_inputData.size();
    NW = BS / _batchConcurrency;
    T = _problem->_inputData[0].size();
    IC = _problem->_inputData[0][0].size();
    // =============================================================================
    const auto nnHyperparameters = _neuralNetwork->getHyperparameters();
    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    // TODO: shuffel testing data
    // shuffle(_problem->_inputData.begin(), _problem->_inputData.end(), std::default_random_engine(seed));
    // shuffle(_problem->_solutionData.begin(), _problem._solutionData.end(), std::default_random_engine(seed));
    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Calculating per worker dimensions
    auto inputDataFlat = flatten(_problem->_inputData);
    // Running epochs
    size_t wId; // worker id
    std::vector<Sample> samples(_batchConcurrency);
    for (wId = 0; wId < _batchConcurrency; wId++)
    {
      samples[wId]["Sample Id"] = wId;
      samples[wId]["Module"] = "Solver";
      samples[wId]["Operation"] = "Forward Data";
      samples[wId]["Input Dims"] = std::vector<size_t> {NW, T, IC};
      samples[wId]["Input Data"] = std::vector<float>(inputDataFlat.begin()+wId*NW*T*IC, inputDataFlat.begin()+(wId+1)*NW*T*IC);
      samples[wId]["Hyperparameters"] = nnHyperparameters;
    }
    if(_batchConcurrency > 1){
      for (wId = 0; wId < _batchConcurrency; wId++) KORALI_START(samples[wId]);
      // Waiting for samples to finish
      KORALI_WAITALL(samples);
    } else{
      runForwardData(samples[0]);
    }
    _evaluation.clear();
    for (wId = 0; wId < _batchConcurrency; wId++){
      const auto ypred = KORALI_GET(std::vector<std::vector<float>>, samples[wId], "Evaluation");
      _evaluation.insert(_evaluation.end(), ypred.begin(), ypred.end());
    }
    if(_mode == "Testing" && _loss){
      _testingLoss = 0.0f;
      auto y_val = getEvaluation(_problem->_inputData);
      _testingLoss = _loss->loss(y_val, _problem->_solutionData);
      (*_k)["Results"]["Testing Loss"] = _testingLoss;
    }
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

  // Running the input values through the neural network
  _neuralNetwork->forward(input);

  // Returning the output values for the last given timestep
  return _neuralNetwork->getOutputValues(N);
}

std::vector<float> DeepSupervisor::backwardGradients(const std::vector<std::vector<float>> &dloss)
{
  // Grabbing constants
  const size_t N = dloss.size();

  // Running the input values through the neural network
  _neuralNetwork->backward(dloss);
  // Getting NN hyperparameter gradients
  auto hyperparameterGradients = _neuralNetwork->getHyperparameterGradients(N);

  return hyperparameterGradients;
}


void DeepSupervisor::runTrainingOnWorker(korali::Sample &sample)
{
  // Copy hyperparameters to workers neural network
  auto nnHyperparameters = KORALI_GET(std::vector<float>, sample, "Hyperparameters");
  // TODO use std move here
  _neuralNetwork->setHyperparameters(nnHyperparameters);
  sample._js.getJson().erase("Hyperparameters");
  // Getting input batch from sample
  auto inputDataFlat = KORALI_GET(std::vector<float>, sample, "Input Data");
  sample._js.getJson().erase("Input Data");
  // Getting solution from sample
  auto solutionDataFlat = KORALI_GET(std::vector<float>, sample, "Solution Data");
  sample._js.getJson().erase("Solution Data");
  // Getting input/solution dimensions
  auto inputDims = KORALI_GET(std::vector<size_t>, sample, "Input Dims");
  auto solutionDims = KORALI_GET(std::vector<size_t>, sample, "Solution Dims");
  sample._js.getJson().erase("Input Dims");
  sample._js.getJson().erase("Solution Dims");
  size_t BS = inputDims[0];
  size_t T = inputDims[1];
  size_t IC = inputDims[2];
  size_t OC = solutionDims[1];
  // De-flattening input and solution vectors
  auto input = deflatten(inputDataFlat, BS, T, IC);
  auto y = deflatten(solutionDataFlat, BS, OC);
  // FORWARD neural network on input data
  const auto yhat = getEvaluation(input);
  // TODO maybe add loss rather to problem than as part of learner ?
  // Making a copy of the solution data where we will store the derivative of the output data
  auto& dloss = y;
  _currentTrainingLoss = 0.0;
  if(_loss){
    _currentTrainingLoss = _loss->loss(y, yhat);
    dloss = _loss->dloss(y, yhat);
  }
  // BACKPROPAGATE the derivative of the output loss
  auto hyperparameterGradients = backwardGradients(dloss);
  sample["Hyperparameter Gradients"] = hyperparameterGradients;
  if(_loss)
    sample["Training Loss"] = _currentTrainingLoss;
}

void DeepSupervisor::runForwardData(korali::Sample &sample)
{
  // Copy hyperparameters to workers neural network
  auto nnHyperparameters = KORALI_GET(std::vector<float>, sample, "Hyperparameters");
  _neuralNetwork->setHyperparameters(nnHyperparameters);
  sample._js.getJson().erase("Hyperparameters");
  auto inputDataFlat = KORALI_GET(std::vector<float>, sample, "Input Data");
  sample._js.getJson().erase("Input Data");
  // Getting input dimensions
  auto inputDims = KORALI_GET(std::vector<size_t>, sample, "Input Dims");
  sample._js.getJson().erase("Input Dims");
  size_t BS = inputDims[0];
  size_t T = inputDims[1];
  size_t IC = inputDims[2];
  // De-flattening input
  auto input = deflatten(inputDataFlat, BS, T, IC);
  sample["Evaluation"] = getEvaluation(input);
}

void DeepSupervisor::finalize() {
  if(_mode == "Predict" or _mode == "Testing" or _mode == "Training"){
    // Variable to check if run only one epoch
    _isOneEpochFinished = false;
  }
}


void DeepSupervisor::printGenerationBefore(){

}


void DeepSupervisor::printGenerationAfter()
{
  if (_mode == "Automatic Training")
  {
    // Printing results so far
    size_t width = 60;
    char bar[width];
    _k->_logger->progressBar(_epochCount/(float)_epochs, bar, width);
    if(_hasValidationSet)
      _k->_logger->logInfo("Normal", "\r[Korali] Epoch %zu/%zu %s Train Loss: %f | Val. Loss: %f | Learning Rate: %f\r", _epochCount, _epochs, bar, _currentTrainingLoss, _currentValidationLoss, _optimizer->_eta);
    else
      _k->_logger->logInfo("Normal", "\r[Korali] Epoch %zu/%zu %s Train Loss: %f | Learning Rate: %f\r", _epochCount, _epochs, bar, _currentTrainingLoss, _optimizer->_eta);
    if(_epochCount>=_epochs){
      _k->_logger->logInfo("Normal", "\n");
    }
  }
  if (_mode == "Predict")
  {
    // TODO: do if metric like accuracy or somethign like that is set
  }
}

void DeepSupervisor::printRunAfter(){
  if(_mode == "Automatic Training"){
    Solver::printRunAfter();
  } else if(_mode == "Training" && _epochCount>=_epochs){
    Solver::printRunAfter();
  }
}

std::vector<float> DeepSupervisor::flatten(const std::vector<std::vector<float>> &vec) const{
  auto N = vec.size();
  auto OC = vec[0].size();
  std::vector<float> vec_flat;
  vec_flat.reserve(N*OC);
  #pragma omp parallel for simd collapse(2)
  for (size_t i = 0; i < N; i++)
    for (size_t j = 0; j < OC; j++)
      vec_flat[i * OC + j] = vec[i][j];
  return vec_flat;
}

std::vector<float> DeepSupervisor::flatten(const std::vector<std::vector<std::vector<float>>> &vec) const{
  auto N = vec.size();
  auto T = vec[0].size();
  auto IC = vec[0][0].size();
  std::vector<float> vec_flat;
  vec_flat.reserve(N*T*IC);
  #pragma omp parallel for simd collapse(3)
  for (size_t i = 0; i < N; i++){
    for (size_t j = 0; j < T; j++){
      for (size_t k = 0; k < IC; k++){
        vec_flat[i * T * IC + j * IC + k] = vec[i][j][k];
      }
    }
  }
  return vec_flat;
}

std::vector<std::vector<float>> DeepSupervisor::deflatten(const std::vector<float> &vec_flat, size_t BS, size_t OC) const{
  auto vec = std::vector<std::vector<float>>(BS, std::vector<float>(OC));
  #pragma omp parallel for simd collapse(2)
  for (size_t i = 0; i < BS; i++)
    for (size_t j = 0; j < OC; j++)
      vec[i][j] = vec_flat[i * OC + j];
  return vec;
}

std::vector<std::vector<std::vector<float>>> DeepSupervisor::deflatten(const std::vector<float> &vec_flat, size_t BS, size_t T, size_t IC) const{
  auto vec = std::vector<std::vector<std::vector<float>>>(BS, std::vector<std::vector<float>>(T, std::vector<float>(IC)));
  #pragma omp parallel for simd collapse(3)
  for (size_t i = 0; i < BS; i++)
    for (size_t j = 0; j < T; j++)
      for (size_t k = 0; k < IC; k++)
        vec[i][j][k] = vec_flat[i * T * IC + j * IC + k];
  return vec;
}

void DeepSupervisor::setConfiguration(knlohmann::json& js) 
{
  if (isDefined(js, "Results"))  eraseValue(js, "Results");

  if (isDefined(js, "Evaluation"))
  {
    try
    {
      _evaluation = js["Evaluation"].get<std::vector<std::vector<float>>>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Evaluation']\n%s", e.what());
    }
    eraseValue(js, "Evaluation");
  }
  if (isDefined(js, "Data", "Validation", "Split"))
  {
    try
    {
      _dataValidationSplit = js["Data"]["Validation"]["Split"].get<float>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Data']['Validation']['Split']\n%s", e.what());
    }
    eraseValue(js, "Data", "Validation", "Split");
  }
  if (isDefined(js, "Validation Set", "Solution"))
  {
    try
    {
      _validationSetSolution = js["Validation Set"]["Solution"].get<std::vector<float>>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Validation Set']['Solution']\n%s", e.what());
    }
    eraseValue(js, "Validation Set", "Solution");
  }
  if (isDefined(js, "Validation Set", "Size"))
  {
    try
    {
      _validationSetSize = js["Validation Set"]["Size"].get<size_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Validation Set']['Size']\n%s", e.what());
    }
    eraseValue(js, "Validation Set", "Size");
  }
  if (isDefined(js, "Current Training Loss"))
  {
    try
    {
      _currentTrainingLoss = js["Current Training Loss"].get<float>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Current Training Loss']\n%s", e.what());
    }
    eraseValue(js, "Current Training Loss");
  }
  if (isDefined(js, "Current Validation Loss"))
  {
    try
    {
      _currentValidationLoss = js["Current Validation Loss"].get<float>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Current Validation Loss']\n%s", e.what());
    }
    eraseValue(js, "Current Validation Loss");
  }
  if (isDefined(js, "Testing Loss"))
  {
    try
    {
      _testingLoss = js["Testing Loss"].get<float>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Testing Loss']\n%s", e.what());
    }
    eraseValue(js, "Testing Loss");
  }
  if (isDefined(js, "Normalization Means"))
  {
    try
    {
      _normalizationMeans = js["Normalization Means"].get<std::vector<float>>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Normalization Means']\n%s", e.what());
    }
    eraseValue(js, "Normalization Means");
  }
  if (isDefined(js, "Normalization Variances"))
  {
    try
    {
      _normalizationVariances = js["Normalization Variances"].get<std::vector<float>>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Normalization Variances']\n%s", e.what());
    }
    eraseValue(js, "Normalization Variances");
  }
  if (isDefined(js, "Epoch Count"))
  {
    try
    {
      _epochCount = js["Epoch Count"].get<size_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Epoch Count']\n%s", e.what());
    }
    eraseValue(js, "Epoch Count");
  }
  if (isDefined(js, "Mode"))
  {
    try
    {
      _mode = js["Mode"].get<std::string>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Mode']\n%s", e.what());
    }
      {
        bool validOption = false; 
        if (_mode == "Automatic Training") validOption = true; 
        if (_mode == "Training") validOption = true; 
        if (_mode == "Predict") validOption = true; 
        if (_mode == "Testing") validOption = true; 
        if (validOption == false) KORALI_LOG_ERROR("Unrecognized value (%s) provided for mandatory setting: ['Mode'] required by deepSupervisor.\n Valid Options are:\n  - Automatic Training\n  - Training\n  - Predict\n  - Testing\n",_mode.c_str()); 
      }
    eraseValue(js, "Mode");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Mode'] required by deepSupervisor.\n"); 

  if (isDefined(js, "Neural Network", "Hidden Layers"))
  {
  _neuralNetworkHiddenLayers = js["Neural Network"]["Hidden Layers"].get<knlohmann::json>();

    eraseValue(js, "Neural Network", "Hidden Layers");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Neural Network']['Hidden Layers'] required by deepSupervisor.\n"); 

  if (isDefined(js, "Neural Network", "Output Activation"))
  {
  _neuralNetworkOutputActivation = js["Neural Network"]["Output Activation"].get<knlohmann::json>();

    eraseValue(js, "Neural Network", "Output Activation");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Neural Network']['Output Activation'] required by deepSupervisor.\n"); 

  if (isDefined(js, "Neural Network", "Output Layer"))
  {
  _neuralNetworkOutputLayer = js["Neural Network"]["Output Layer"].get<knlohmann::json>();

    eraseValue(js, "Neural Network", "Output Layer");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Neural Network']['Output Layer'] required by deepSupervisor.\n"); 

  if (isDefined(js, "Neural Network", "Engine"))
  {
    try
    {
      _neuralNetworkEngine = js["Neural Network"]["Engine"].get<std::string>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Neural Network']['Engine']\n%s", e.what());
    }
    eraseValue(js, "Neural Network", "Engine");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Neural Network']['Engine'] required by deepSupervisor.\n"); 

  if (isDefined(js, "Neural Network", "Optimizer"))
  {
    try
    {
      _neuralNetworkOptimizer = js["Neural Network"]["Optimizer"].get<std::string>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Neural Network']['Optimizer']\n%s", e.what());
    }
      {
        bool validOption = false; 
        if (_neuralNetworkOptimizer == "Adam") validOption = true; 
        if (_neuralNetworkOptimizer == "AdaBelief") validOption = true; 
        if (_neuralNetworkOptimizer == "MADGRAD") validOption = true; 
        if (_neuralNetworkOptimizer == "RMSProp") validOption = true; 
        if (_neuralNetworkOptimizer == "Adagrad") validOption = true; 
        if (validOption == false) KORALI_LOG_ERROR("Unrecognized value (%s) provided for mandatory setting: ['Neural Network']['Optimizer'] required by deepSupervisor.\n Valid Options are:\n  - Adam\n  - AdaBelief\n  - MADGRAD\n  - RMSProp\n  - Adagrad\n",_neuralNetworkOptimizer.c_str()); 
      }
    eraseValue(js, "Neural Network", "Optimizer");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Neural Network']['Optimizer'] required by deepSupervisor.\n"); 

  if (isDefined(js, "Hyperparameters"))
  {
    try
    {
      _hyperparameters = js["Hyperparameters"].get<std::vector<float>>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Hyperparameters']\n%s", e.what());
    }
    eraseValue(js, "Hyperparameters");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Hyperparameters'] required by deepSupervisor.\n"); 

  if (isDefined(js, "Loss Function"))
  {
    try
    {
      _lossFunction = js["Loss Function"].get<std::string>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Loss Function']\n%s", e.what());
    }
      {
        bool validOption = false; 
        if (_lossFunction == "Direct Gradient") validOption = true; 
        if (_lossFunction == "Mean Squared Error") validOption = true; 
        if (validOption == false) KORALI_LOG_ERROR("Unrecognized value (%s) provided for mandatory setting: ['Loss Function'] required by deepSupervisor.\n Valid Options are:\n  - Direct Gradient\n  - Mean Squared Error\n",_lossFunction.c_str()); 
      }
    eraseValue(js, "Loss Function");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Loss Function'] required by deepSupervisor.\n"); 

  if (isDefined(js, "Regularizer", "Type"))
  {
    try
    {
      _regularizerType = js["Regularizer"]["Type"].get<std::string>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Regularizer']['Type']\n%s", e.what());
    }
      {
        bool validOption = false; 
        if (_regularizerType == "") validOption = true; 
        if (_regularizerType == "None") validOption = true; 
        if (_regularizerType == "L1") validOption = true; 
        if (_regularizerType == "L2") validOption = true; 
        if (validOption == false) KORALI_LOG_ERROR("Unrecognized value (%s) provided for mandatory setting: ['Regularizer']['Type'] required by deepSupervisor.\n Valid Options are:\n  - \n  - None\n  - L1\n  - L2\n",_regularizerType.c_str()); 
      }
    eraseValue(js, "Regularizer", "Type");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Regularizer']['Type'] required by deepSupervisor.\n"); 

  if (isDefined(js, "Regularizer", "Coefficient"))
  {
    try
    {
      _regularizerCoefficient = js["Regularizer"]["Coefficient"].get<float>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Regularizer']['Coefficient']\n%s", e.what());
    }
    eraseValue(js, "Regularizer", "Coefficient");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Regularizer']['Coefficient'] required by deepSupervisor.\n"); 

  if (isDefined(js, "Learning Rate"))
  {
    try
    {
      _learningRate = js["Learning Rate"].get<float>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Learning Rate']\n%s", e.what());
    }
    eraseValue(js, "Learning Rate");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Learning Rate'] required by deepSupervisor.\n"); 

  if (isDefined(js, "Learning Rate Type"))
  {
    try
    {
      _learningRateType = js["Learning Rate Type"].get<std::string>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Learning Rate Type']\n%s", e.what());
    }
      {
        bool validOption = false; 
        if (_learningRateType == "") validOption = true; 
        if (_learningRateType == "Const") validOption = true; 
        if (_learningRateType == "Step Based") validOption = true; 
        if (_learningRateType == "Time Based") validOption = true; 
        if (validOption == false) KORALI_LOG_ERROR("Unrecognized value (%s) provided for mandatory setting: ['Learning Rate Type'] required by deepSupervisor.\n Valid Options are:\n  - \n  - Const\n  - Step Based\n  - Time Based\n",_learningRateType.c_str()); 
      }
    eraseValue(js, "Learning Rate Type");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Learning Rate Type'] required by deepSupervisor.\n"); 

  if (isDefined(js, "Learning Rate Decay Factor"))
  {
    try
    {
      _learningRateDecayFactor = js["Learning Rate Decay Factor"].get<float>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Learning Rate Decay Factor']\n%s", e.what());
    }
    eraseValue(js, "Learning Rate Decay Factor");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Learning Rate Decay Factor'] required by deepSupervisor.\n"); 

  if (isDefined(js, "Learning Rate Lower Bound"))
  {
    try
    {
      _learningRateLowerBound = js["Learning Rate Lower Bound"].get<float>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Learning Rate Lower Bound']\n%s", e.what());
    }
    eraseValue(js, "Learning Rate Lower Bound");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Learning Rate Lower Bound'] required by deepSupervisor.\n"); 

  if (isDefined(js, "Learning Rate Steps"))
  {
    try
    {
      _learningRateSteps = js["Learning Rate Steps"].get<float>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Learning Rate Steps']\n%s", e.what());
    }
    eraseValue(js, "Learning Rate Steps");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Learning Rate Steps'] required by deepSupervisor.\n"); 

  if (isDefined(js, "Learning Rate Save"))
  {
    try
    {
      _learningRateSave = js["Learning Rate Save"].get<int>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Learning Rate Save']\n%s", e.what());
    }
    eraseValue(js, "Learning Rate Save");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Learning Rate Save'] required by deepSupervisor.\n"); 

  if (isDefined(js, "L2 Regularization", "Enabled"))
  {
    try
    {
      _l2RegularizationEnabled = js["L2 Regularization"]["Enabled"].get<int>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['L2 Regularization']['Enabled']\n%s", e.what());
    }
    eraseValue(js, "L2 Regularization", "Enabled");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['L2 Regularization']['Enabled'] required by deepSupervisor.\n"); 

  if (isDefined(js, "L2 Regularization", "Importance"))
  {
    try
    {
      _l2RegularizationImportance = js["L2 Regularization"]["Importance"].get<int>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['L2 Regularization']['Importance']\n%s", e.what());
    }
    eraseValue(js, "L2 Regularization", "Importance");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['L2 Regularization']['Importance'] required by deepSupervisor.\n"); 

  if (isDefined(js, "Output Weights Scaling"))
  {
    try
    {
      _outputWeightsScaling = js["Output Weights Scaling"].get<float>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Output Weights Scaling']\n%s", e.what());
    }
    eraseValue(js, "Output Weights Scaling");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Output Weights Scaling'] required by deepSupervisor.\n"); 

  if (isDefined(js, "Batch Concurrency"))
  {
    try
    {
      _batchConcurrency = js["Batch Concurrency"].get<size_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Batch Concurrency']\n%s", e.what());
    }
    eraseValue(js, "Batch Concurrency");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Batch Concurrency'] required by deepSupervisor.\n"); 

  if (isDefined(js, "Data", "Input", "Shuffel"))
  {
    try
    {
      _dataInputShuffel = js["Data"]["Input"]["Shuffel"].get<int>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Data']['Input']['Shuffel']\n%s", e.what());
    }
    eraseValue(js, "Data", "Input", "Shuffel");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Data']['Input']['Shuffel'] required by deepSupervisor.\n"); 

  if (isDefined(js, "Data", "Training", "Shuffel"))
  {
    try
    {
      _dataTrainingShuffel = js["Data"]["Training"]["Shuffel"].get<int>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Data']['Training']['Shuffel']\n%s", e.what());
    }
    eraseValue(js, "Data", "Training", "Shuffel");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Data']['Training']['Shuffel'] required by deepSupervisor.\n"); 

  if (isDefined(js, "Termination Criteria", "Epochs"))
  {
    try
    {
      _epochs = js["Termination Criteria"]["Epochs"].get<size_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Termination Criteria']['Epochs']\n%s", e.what());
    }
    eraseValue(js, "Termination Criteria", "Epochs");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Termination Criteria']['Epochs'] required by deepSupervisor.\n"); 

  if (isDefined(js, "Termination Criteria", "Target Loss"))
  {
    try
    {
      _targetLoss = js["Termination Criteria"]["Target Loss"].get<float>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Termination Criteria']['Target Loss']\n%s", e.what());
    }
    eraseValue(js, "Termination Criteria", "Target Loss");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Termination Criteria']['Target Loss'] required by deepSupervisor.\n"); 

  if (isDefined(js, "Termination Criteria", "Is One Epoch Finished"))
  {
    try
    {
      _isOneEpochFinished = js["Termination Criteria"]["Is One Epoch Finished"].get<int>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ deepSupervisor ] \n + Key:    ['Termination Criteria']['Is One Epoch Finished']\n%s", e.what());
    }
    eraseValue(js, "Termination Criteria", "Is One Epoch Finished");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Termination Criteria']['Is One Epoch Finished'] required by deepSupervisor.\n"); 

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
   js["Mode"] = _mode;
   js["Neural Network"]["Hidden Layers"] = _neuralNetworkHiddenLayers;
   js["Neural Network"]["Output Activation"] = _neuralNetworkOutputActivation;
   js["Neural Network"]["Output Layer"] = _neuralNetworkOutputLayer;
   js["Neural Network"]["Engine"] = _neuralNetworkEngine;
   js["Neural Network"]["Optimizer"] = _neuralNetworkOptimizer;
   js["Hyperparameters"] = _hyperparameters;
   js["Loss Function"] = _lossFunction;
   js["Regularizer"]["Type"] = _regularizerType;
   js["Regularizer"]["Coefficient"] = _regularizerCoefficient;
   js["Learning Rate"] = _learningRate;
   js["Learning Rate Type"] = _learningRateType;
   js["Learning Rate Decay Factor"] = _learningRateDecayFactor;
   js["Learning Rate Lower Bound"] = _learningRateLowerBound;
   js["Learning Rate Steps"] = _learningRateSteps;
   js["Learning Rate Save"] = _learningRateSave;
   js["L2 Regularization"]["Enabled"] = _l2RegularizationEnabled;
   js["L2 Regularization"]["Importance"] = _l2RegularizationImportance;
   js["Output Weights Scaling"] = _outputWeightsScaling;
   js["Batch Concurrency"] = _batchConcurrency;
   js["Data"]["Input"]["Shuffel"] = _dataInputShuffel;
   js["Data"]["Training"]["Shuffel"] = _dataTrainingShuffel;
   js["Termination Criteria"]["Epochs"] = _epochs;
   js["Termination Criteria"]["Target Loss"] = _targetLoss;
   js["Termination Criteria"]["Is One Epoch Finished"] = _isOneEpochFinished;
   js["Evaluation"] = _evaluation;
   js["Data"]["Validation"]["Split"] = _dataValidationSplit;
   js["Validation Set"]["Solution"] = _validationSetSolution;
   js["Validation Set"]["Size"] = _validationSetSize;
   js["Current Training Loss"] = _currentTrainingLoss;
   js["Current Validation Loss"] = _currentValidationLoss;
   js["Testing Loss"] = _testingLoss;
   js["Normalization Means"] = _normalizationMeans;
   js["Normalization Variances"] = _normalizationVariances;
   js["Epoch Count"] = _epochCount;
 Learner::getConfiguration(js);
} 

void DeepSupervisor::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"L2 Regularization\": {\"Enabled\": false, \"Importance\": 0.0001}, \"Regularizer\": {\"Coefficient\": 0.0001, \"Type\": \"None\"}, \"Loss Function\": \"Direct Gradient\", \"Learning Rate Type\": \"Const\", \"Learning Rate Save\": true, \"Learning Rate Decay Factor\": 100, \"Learning Rate Steps\": 0, \"Learning Rate Lower Bound\": -10000000000, \"Neural Network\": {\"Output Activation\": \"Identity\", \"Output Layer\": {}}, \"Termination Criteria\": {\"Epochs\": 10000000000, \"Is One Epoch Finished\": false, \"Target Loss\": -1.0, \"Max Generations\": 10000000000}, \"Hyperparameters\": [], \"Output Weights Scaling\": 1.0, \"Batch Concurrency\": 1, \"Epoch Count\": 1, \"Data\": {\"Validation\": {\"Split\": 0.0}, \"Training\": {\"Shuffel\": true}, \"Input\": {\"Shuffel\": true}}}";
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

  if ((_epochCount >= _epochs) && (_mode == "Automatic Training"))
  {
    _terminationCriteria.push_back("deepSupervisor['Epochs'] = " + std::to_string(_epochs) + ".");
    hasFinished = true;
  }

  if ((_epochCount > 1) && (_targetLoss > 0.0) && (_currentValidationLoss <= _targetLoss) && (_mode == "Training"))
  {
    _terminationCriteria.push_back("deepSupervisor['Target Loss'] = " + std::to_string(_targetLoss) + ".");
    hasFinished = true;
  }

  if (_isOneEpochFinished && (_mode == "Predict" || _mode == "Testing" || _mode == "Training"))
  {
    _terminationCriteria.push_back("deepSupervisor['Is One Epoch Finished'] = " + std::to_string(_isOneEpochFinished) + ".");
    hasFinished = true;
  }

  hasFinished = hasFinished || Learner::checkTermination();
  return hasFinished;
}

bool DeepSupervisor::runOperation(std::string operation, korali::Sample& sample)
{
 bool operationDetected = false;

 if (operation == "Run Training On Worker")
 {
  runTrainingOnWorker(sample);
  return true;
 }

 if (operation == "Forward Data")
 {
  runForwardData(sample);
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
