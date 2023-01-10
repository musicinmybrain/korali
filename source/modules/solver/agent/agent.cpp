#include "auxiliar/fs.hpp"
#include "engine.hpp"
#include "modules/solver/agent/agent.hpp"
#include "sample/sample.hpp"
#include <algorithm>
#include <chrono>

namespace korali
{
namespace solver
{
;

#pragma omp declare reduction(vec_int_plus        \
                              : std::vector <int> \
                              : std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus <int>())) initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

void Agent::initialize()
{
  _variableCount = _k->_variables.size();

  // Getting problem pointer
  _problem = dynamic_cast<problem::ReinforcementLearning *>(_k->_problem);

  // Getting number of agents
  const size_t numAgents = _problem->_agentsPerEnvironment;

  // Allocating and obtaining action bounds information
  _actionLowerBounds.resize(_problem->_actionVectorSize);
  _actionUpperBounds.resize(_problem->_actionVectorSize);

  for (size_t i = 0; i < _problem->_actionVectorSize; i++)
  {
    auto varIdx = _problem->_actionVectorIndexes[i];
    _actionLowerBounds[i] = _k->_variables[varIdx]->_lowerBound;
    _actionUpperBounds[i] = _k->_variables[varIdx]->_upperBound;

    if (_actionUpperBounds[i] - _actionLowerBounds[i] <= 0.0) KORALI_LOG_ERROR("Upper (%f) and Lower Bound (%f) of action variable %lu invalid.\n", _actionUpperBounds[i], _actionLowerBounds[i], i);
  }

  if (_episodesPerGeneration < 1)
    KORALI_LOG_ERROR("Episodes Per Generation must be larger equal 1 (is %zu)", _episodesPerGeneration);

  // Initializing selected policy
  initializeAgent();

  // Initializing random seed for the shuffle operation
  mt = new std::mt19937(rd());
  mt->seed(_k->_randomSeed++);

  // If not set, using heurisitc for maximum size
  if (_experienceReplayMaximumSize == 0)
    _experienceReplayMaximumSize = std::pow(2, 14) * std::sqrt(_problem->_stateVectorSize + _problem->_actionVectorSize);

  // If not set, filling ER before learning
  if (_experienceReplayStartSize == 0)
    _experienceReplayStartSize = _experienceReplayMaximumSize;

  // Initialize current beta for all agents
  _experienceReplayOffPolicyREFERCurrentBeta = std::vector<float>(numAgents, _experienceReplayOffPolicyREFERBeta);

  // Formatting reward history for each agent
  _trainingRewardHistory.resize(numAgents);

  //  Pre-allocating space for the experience replay memory
  _stateBuffer.resize(_experienceReplayMaximumSize);
  _featureBuffer.resize(_experienceReplayMaximumSize);
  _actionBuffer.resize(_experienceReplayMaximumSize);
  _rewardUpdateBuffer.resize(_experienceReplayMaximumSize);
  _retraceValueBufferContiguous.resize(_experienceReplayMaximumSize * numAgents);
  _rewardBufferContiguous.resize(_experienceReplayMaximumSize * numAgents);
  _stateValueBufferContiguous.resize(_experienceReplayMaximumSize * numAgents);
  _importanceWeightBuffer.resize(_experienceReplayMaximumSize);
  _truncatedImportanceWeightBufferContiguous.resize(_experienceReplayMaximumSize * numAgents);
  _productImportanceWeightBuffer.resize(_experienceReplayMaximumSize);
  _truncatedStateValueBuffer.resize(_experienceReplayMaximumSize);
  _truncatedStateBuffer.resize(_experienceReplayMaximumSize);
  _terminationBuffer.resize(_experienceReplayMaximumSize);
  _expPolicyBuffer.resize(_experienceReplayMaximumSize);
  _curPolicyBuffer.resize(_experienceReplayMaximumSize);
  _isOnPolicyBuffer.resize(_experienceReplayMaximumSize);
  _episodePosBuffer.resize(_experienceReplayMaximumSize);
  _episodeIdBuffer.resize(_experienceReplayMaximumSize);

  // Pre-allocate space for policies
  _policyBuffer.resize(_experienceReplayMaximumSize);

  _experienceReplayOffPolicyHistory.resize(0);

  // Initialize background samples
  _backgroundTrajectoryCount = 0;
  _demonstrationFeatureReward.resize(0);
  _demonstrationLogProbability.resize(0);
  _maxEntropyObjective.resize(0);
  _demonstrationBatchImportanceWeight.resize(0);
  _backgroundBatchImportanceWeight.resize(0);
  _effectiveSampleSize.resize(0);
  _backgroundTrajectoryLogProbabilities.resize(_backgroundSampleSize);

  //  Pre-allocating space for state time sequence
  _stateTimeSequence.resize(numAgents);
  for (size_t a = 0; a < numAgents; ++a)
    _stateTimeSequence[a].resize(_timeSequenceLength);

  /*********************************************************************
   * If initial generation, set initial agent configuration
   *********************************************************************/
  if (_k->_currentGeneration == 0)
  {
    _currentEpisode = 0;
    _policyUpdateCount = 0;
    _rewardUpdateCount = 0;
    _experienceCount = 0;

    // Initializing training statistics
    _trainingAverageFeatureReward.resize(numAgents, 0.0);

    // Initializing training and episode statistics
    /*
    _testingAverageReward.resize(numAgents, -korali::Inf);
    _testingBestReward.resize(numAgents,-korali::Inf);
    _testingWorstReward.resize(numAgents,-korali::Inf);
    _testingBestAverageReward.resize(numAgents, -korali::Inf);
    */
    _experienceCount = 0;

    _testingBestEpisodeId = 0;
    _trainingBestReward.resize(numAgents, -korali::Inf);
    _trainingBestEpisodeId.resize(numAgents, 0);
    _trainingAverageReward.resize(numAgents, -korali::Inf);

    /* Initializing REFER information */

    // If cutoff scale is not defined, use a heuristic value [defaults to 4.0]
    if (_experienceReplayOffPolicyCutoffScale <= 0.0f)
      KORALI_LOG_ERROR("Experience Replay Cutoff Scale must be larger 0.0");

    _experienceReplayOffPolicyCount.resize(numAgents, 0);
    _experienceReplayOffPolicyRatio.resize(numAgents, 0.0f);
    if (_learningRate <= 0.0f)
      KORALI_LOG_ERROR("Learning rate must be larger 0.0");

    _currentLearningRate = _learningRate;
    _experienceReplayOffPolicyCurrentCutoff = _experienceReplayOffPolicyCutoffScale;

    _experienceReplayOffPolicyCurrentCutoff = _experienceReplayOffPolicyCutoffScale;

    // State & Feature Rescaling information
    _featureRescalingMeans = std::vector<std::vector<float>>(numAgents, std::vector<float>(_problem->_featureVectorSize, 0.0));
    _featureRescalingSigmas = std::vector<std::vector<float>>(numAgents, std::vector<float>(_problem->_featureVectorSize, 1.0));
    _stateRescalingMeans = std::vector<std::vector<float>>(numAgents, std::vector<float>(_problem->_stateVectorSize, 0.0f));
    _stateRescalingSigmas = std::vector<std::vector<float>>(numAgents, std::vector<float>(_problem->_stateVectorSize, 1.0f));

    // Reward Rescaling information
    _rewardRescalingSigma = 1.;
    _rewardRescalingSumSquaredRewards = 0.;

    // If not given, get agent's initial policy
    if (not isDefined(_trainingCurrentPolicies, "Policy Hyperparameters"))
      _trainingCurrentPolicies = getPolicy();
  }

  // Setting current agent's training state
  setPolicy(_trainingCurrentPolicies["Policy Hyperparameters"]);

  // If this continues a previous training run, deserialize previous input experience replay. Only for the root (engine) rank
  if (_k->_currentGeneration > 0)
    if (_mode == "Training")
      if (_k->_engine->_conduit != NULL)
        deserializeExperienceReplay();

  // Initializing session-wise profiling timers
  _sessionRunningTime = 0.0;
  _sessionSerializationTime = 0.0;
  _sessionWorkerComputationTime = 0.0;
  _sessionWorkerCommunicationTime = 0.0;
  _sessionPolicyEvaluationTime = 0.0;
  _sessionPolicyUpdateTime = 0.0;
  _sessionRewardUpdateTime = 0.0;
  _sessionPartitionFunctionStatUpdateTime = 0.0;
  _sessionWorkerAttendingTime = 0.0;
  _sessionTrajectoryLogProbabilityUpdateTime = 0.0;

  // Initializing session-specific counters
  _sessionGeneration = 1;
  _sessionEpisodeCount = 0;
  _sessionExperienceCount = 0;
  _sessionPolicyUpdateCount = 0;
  _sessionRewardUpdateCount = 0;

  // Calculating how many more experiences do we need in this session to reach the starting size
  _sessionExperiencesUntilStartSize = _stateBuffer.size() > _experienceReplayStartSize ? 0 : _experienceReplayStartSize - _stateBuffer.size();

  if (_mode == "Training")
  {
    // Creating storate for _agents and their status
    _workers.resize(_concurrentWorkers);
    _isWorkerRunning.resize(_concurrentWorkers, false);

    // In case the agent was tested before, remove _testingCurrentPolicies
    _testingCurrentPolicies.clear();
  }

  if (_mode == "Testing")
  {
    // Fixing termination criteria for testing mode
    _maxGenerations = _k->_currentGeneration + 1;

    // Setting testing policy to best testing hyperparameters if not custom-set by the user
    if (_testingCurrentPolicies.empty())
    {
      // Checking if testing policies have been generated
      if (_testingBestPolicies.empty())
      {
        _k->_logger->logInfo("Minimal", "Using current training policy for testing.\n");
        _testingCurrentPolicies = _trainingCurrentPolicies;
      }
      else
      {
        _k->_logger->logInfo("Minimal", "Using best testing policy for test-run.\n");
        _testingCurrentPolicies = _testingBestPolicies;
      }
    }

    // Checking if there's testing samples defined
    if (_testingSampleIds.size() == 0)
      KORALI_LOG_ERROR("For testing, you need to indicate the sample ids to run in the ['Testing']['Sample Ids'] field.\n");

    // Prepare storage for rewards from tested samples
    _testingReward.resize(_testingSampleIds.size());
  }

  if (_problem->_numberObservedTrajectories < _demonstrationBatchSize) KORALI_LOG_ERROR("Demonstration Batch Size (%zu) must be smaller than total number of observed trajectories (%zu).\n", _demonstrationBatchSize, _problem->_numberObservedTrajectories);

  if (_backgroundSampleSize <= _backgroundBatchSize) KORALI_LOG_ERROR("Bachground Sample Size too small, must be greater than Background Batch Size");

  if ((_demonstrationPolicy == "Constant" || _demonstrationPolicy == "Linear" || _demonstrationPolicy == "Quadratic") == false) KORALI_LOG_ERROR("Demonstration Policy is invalid, choose 'Constant', 'Linear' or 'Quadratic'");

  _backgroundTrajectoryStates.resize(_backgroundSampleSize);
  _backgroundTrajectoryActions.resize(_backgroundSampleSize);
  _backgroundTrajectoryFeatures.resize(_backgroundSampleSize);
  _backgroundPolicyHyperparameter.resize(_backgroundSampleSize);

  _rewardFunctionExperiment["Problem"]["Type"] = "Supervised Learning";
  _rewardFunctionExperiment["Problem"]["Max Timesteps"] = 1;
  _rewardFunctionExperiment["Problem"]["Training Batch Size"] = _rewardFunctionBatchSize;
  _rewardFunctionExperiment["Problem"]["Testing Batch Size"] = 1;
  _rewardFunctionExperiment["Problem"]["Input"]["Size"] = _problem->_featureVectorSize;
  _rewardFunctionExperiment["Problem"]["Solution"]["Size"] = 1;

  _rewardFunctionExperiment["Solver"]["Type"] = "DeepSupervisor";
  _rewardFunctionExperiment["Solver"]["Mode"] = "Training";
  _rewardFunctionExperiment["Solver"]["L2 Regularization"]["Enabled"] = _rewardFunctionL2RegularizationEnabled;
  _rewardFunctionExperiment["Solver"]["L2 Regularization"]["Importance"] = _rewardFunctionL2RegularizationImportance;
  _rewardFunctionExperiment["Solver"]["Loss Function"] = "Direct Gradient";
  _rewardFunctionExperiment["Solver"]["Learning Rate"] = _rewardFunctionLearningRate;
  _rewardFunctionExperiment["Solver"]["Neural Network"]["Engine"] = _neuralNetworkEngine;
  _rewardFunctionExperiment["Solver"]["Neural Network"]["Optimizer"] = _neuralNetworkOptimizer;
  _rewardFunctionExperiment["Solver"]["Neural Network"]["Hidden Layers"] = _rewardFunctionNeuralNetworkHiddenLayers;
  _rewardFunctionExperiment["Solver"]["Output Weights Scaling"] = 0.001;

  _rewardFunctionExperiment["Solver"]["Neural Network"]["Output Layer"]["Scale"][0] = 1.0f;
  _rewardFunctionExperiment["Solver"]["Neural Network"]["Output Layer"]["Shift"][0] = -0.5f;
  _rewardFunctionExperiment["Solver"]["Neural Network"]["Output Layer"]["Transformation Mask"][0] = "Sigmoid";

  // Running initialization to verify that the configuration is correct
  _rewardFunctionExperiment.initialize();
  _rewardFunctionProblem = dynamic_cast<problem::SupervisedLearning *>(_rewardFunctionExperiment._problem);
  _rewardFunctionLearner = dynamic_cast<solver::DeepSupervisor *>(_rewardFunctionExperiment._solver);
    
  // Preallocating space in the underlying supervised problem's input and solution data structures 
  _rewardFunctionProblem->_inputData.resize(_rewardFunctionBatchSize);
  _rewardFunctionProblem->_solutionData.resize(_rewardFunctionBatchSize);

  // Init gradient
  _maxEntropyGradient.resize(_rewardFunctionLearner->_neuralNetwork->_hyperparameterCount,0);
}

void Agent::runGeneration()
{
  if (_mode == "Training") trainingGeneration();
  if (_mode == "Testing") testingGeneration();
}

void Agent::trainingGeneration()
{
  auto beginTime = std::chrono::steady_clock::now(); // Profiling

  // Setting generation-specific timers
  _generationRunningTime = 0.0;
  _generationSerializationTime = 0.0;
  _generationWorkerComputationTime = 0.0;
  _generationWorkerCommunicationTime = 0.0;
  _generationPolicyEvaluationTime = 0.0;
  _generationPolicyUpdateTime = 0.0;
  _generationRewardUpdateTime = 0.0;
  _generationPartitionFunctionStatUpdateTime = 0.0;
  _generationWorkerAttendingTime = 0.0;
  _generationTrajectoryLogProbabilityUpdateTime = 0.0;

  // Running until all _workers have finished
  while (_sessionEpisodeCount < _episodesPerGeneration * _sessionGeneration)
  {
    // Launching (or re-launching) agents
    for (size_t workerId = 0; workerId < _concurrentWorkers; workerId++)
      if (_isWorkerRunning[workerId] == false)
      {
        _workers[workerId]["Sample Id"] = _currentEpisode++;
        _workers[workerId]["Module"] = "Problem";
        _workers[workerId]["Operation"] = "Run Training Episode";
        for (size_t p = 0; p < _problem->_policiesPerEnvironment; p++)
          _workers[workerId]["Policy Hyperparameters"][p] = _trainingCurrentPolicies["Policy Hyperparameters"][p];
        _workers[workerId]["State Rescaling"]["Means"] = _stateRescalingMeans;
        _workers[workerId]["State Rescaling"]["Standard Deviations"] = _stateRescalingSigmas;
        _workers[workerId]["Feature Rescaling"]["Means"] = _featureRescalingMeans;
        _workers[workerId]["Feature Rescaling"]["Standard Deviations"] = _featureRescalingSigmas;

        KORALI_START(_workers[workerId]);

        _isWorkerRunning[workerId] = true;
      }

    // Listening to _workers for incoming experiences
    KORALI_LISTEN(_workers);

    // Attending to running agents, checking if any experience has been received
    for (size_t workerId = 0; workerId < _concurrentWorkers; workerId++)
      if (_isWorkerRunning[workerId] == true)
        attendWorker(workerId);

    // Perform optimization steps on the critic/policy, if reached the minimum replay memory size
    if (_experienceCount >= _experienceReplayStartSize)
    {
      // Update background and demonstration batch after initial RM is full and then every 10 episodes
      if (_optimizeMaxEntropyObjective == true)
        if (_sessionExperienceCount > (_experiencesBetweenRewardUpdates * _sessionRewardUpdateCount + _sessionExperiencesUntilStartSize))
        {
          const auto startTime = std::chrono::steady_clock::now(); // Profiling

          // Sample trajectory to replace
          //const float u = _uniformGenerator->getRandomNumber();
          //const size_t bckIdx = std::floor(u * (float)(_rewardUpdateCount + 1)); // TODO: fix statistic
          const size_t bckIdx = _backgroundTrajectoryCount % _backgroundSampleSize;
          updateBackgroundBatch(bckIdx);
          updateDemonstrationBatch(bckIdx);

          const auto endTime = std::chrono::steady_clock::now();                                                                              // Profiling
          _sessionTrajectoryLogProbabilityUpdateTime += std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count();    // Profiling
          _generationTrajectoryLogProbabilityUpdateTime += std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count(); // Profiling

          _k->_logger->logInfo("Detailed", "Done!\n");
        }

      // If we accumulated enough experiences, we rescale the states (once)
      if (_stateRescalingEnabled == true)
        if (_policyUpdateCount == 0)
          rescaleStates();

      // If we accumulated enough experiences, we rescale the features (once)
      if (_featureRescalingEnabled == true)
        if (_policyUpdateCount == 0)
          rescaleFeatures();

      if (_optimizeMaxEntropyObjective == false)
      {
        while (_rewardUpdateCount * _experiencesBetweenRewardUpdates <= _experienceReplayStartSize)
        {
          updateRewardFunction();
          _rewardUpdateCount++;
        }
      }
      else
      {
        // Update the reward function based on guided cost learning
        while (_sessionExperienceCount > (_experiencesBetweenRewardUpdates * _sessionRewardUpdateCount + _sessionExperiencesUntilStartSize))
        {
          auto beginTime = std::chrono::steady_clock::now(); // Profiling

          updateRewardFunction();

          auto endTime = std::chrono::steady_clock::now();                                                                  // Profiling
          _sessionRewardUpdateTime += std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - beginTime).count();    // Profiling
          _generationRewardUpdateTime += std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - beginTime).count(); // Profiling

          _rewardUpdateCount++;
          _sessionRewardUpdateCount++;
        }
      }

      // If we accumulated enough experiences between updates in this session, update now
      while (_sessionExperienceCount > (_experiencesBetweenPolicyUpdates * _sessionPolicyUpdateCount + _sessionExperiencesUntilStartSize))
      {
        auto beginTime = std::chrono::steady_clock::now(); // Profiling

        // Calling the algorithm specific policy training algorithm
        trainPolicy();

        auto endTime = std::chrono::steady_clock::now();                                                                  // Profiling
        _sessionPolicyUpdateTime += std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - beginTime).count();    // Profiling
        _generationPolicyUpdateTime += std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - beginTime).count(); // Profiling

        // Increasing policy update counters
        _policyUpdateCount++;
        _sessionPolicyUpdateCount++;

        // Updating the off policy cutoff
        _experienceReplayOffPolicyCurrentCutoff = _experienceReplayOffPolicyCutoffScale / (1.0f + _experienceReplayOffPolicyAnnealingRate * (float)_policyUpdateCount);

        for (size_t a = 0; a < _problem->_agentsPerEnvironment; a++)
        {
          // Updating REFER learning rate and beta parameters
          _currentLearningRate = _learningRate / (1.0f + _experienceReplayOffPolicyAnnealingRate * (float)_policyUpdateCount);
          if (_experienceReplayOffPolicyRatio[a] > _experienceReplayOffPolicyTarget)
            _experienceReplayOffPolicyREFERCurrentBeta[a] = (1.0f - _currentLearningRate) * _experienceReplayOffPolicyREFERCurrentBeta[a];
          else
            _experienceReplayOffPolicyREFERCurrentBeta[a] = (1.0f - _currentLearningRate) * _experienceReplayOffPolicyREFERCurrentBeta[a] + _currentLearningRate;
        }
      }

      // Getting new policy hyperparameters (for agents to generate actions)
      _trainingCurrentPolicies = getPolicy();
    }
  }

  // Now serializing experience replay database
  if (_experienceReplaySerialize == true)
    if (_k->_fileOutputEnabled)
      if (_k->_fileOutputFrequency > 0)
        if (_k->_currentGeneration % _k->_fileOutputFrequency == 0)
          serializeExperienceReplay();

  // Measuring generation time
  auto endTime = std::chrono::steady_clock::now();                                                             // Profiling
  _sessionRunningTime += std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - beginTime).count();    // Profiling
  _generationRunningTime += std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - beginTime).count(); // Profiling

  /*********************************************************************
   * Updating statistics/bookkeeping
   *********************************************************************/

  // Updating average cumulative reward statistics
  _trainingAverageReward = std::vector<float>(_problem->_agentsPerEnvironment, 0.0f);
  _trainingAverageFeatureReward = std::vector<float>(_problem->_agentsPerEnvironment, 0.0f);

  for (size_t a = 0; a < _problem->_agentsPerEnvironment; a++)
  {
    ssize_t startEpisodeId = _trainingRewardHistory[a].size() - _trainingAverageDepth;
    ssize_t endEpisodeId = _trainingRewardHistory[a].size() - 1;

    if (startEpisodeId < 0) startEpisodeId = 0;
    for (ssize_t e = startEpisodeId; e <= endEpisodeId; e++)
      _trainingAverageReward[a] += _trainingRewardHistory[a][e];
    _trainingAverageReward[a] /= (float)(endEpisodeId - startEpisodeId + 1);

    for (ssize_t e = startEpisodeId; e < endEpisodeId; e++)
      _trainingAverageFeatureReward[a] += _trainingFeatureRewardHistory[a][e];
    _trainingAverageFeatureReward[a] /= (float)(endEpisodeId - startEpisodeId);
  }

  // Increasing session's generation count
  _sessionGeneration++;
}

void Agent::testingGeneration()
{
  // Allocating testing agents
  std::vector<Sample> testingWorkers(_testingSampleIds.size());

  // Launching  agents
  for (size_t workerId = 0; workerId < _testingSampleIds.size(); workerId++)
  {
    testingWorkers[workerId]["Sample Id"] = _testingSampleIds[workerId];
    testingWorkers[workerId]["Module"] = "Problem";
    testingWorkers[workerId]["Operation"] = "Run Testing Episode";

    for (size_t p = 0; p < _problem->_policiesPerEnvironment; p++)
      testingWorkers[workerId]["Policy Hyperparameters"][p] = _testingCurrentPolicies["Policy Hyperparameters"][p];
    testingWorkers[workerId]["State Rescaling"]["Means"] = _stateRescalingMeans;
    testingWorkers[workerId]["State Rescaling"]["Standard Deviations"] = _stateRescalingSigmas;
    testingWorkers[workerId]["Feature Rescaling"]["Means"] = _featureRescalingMeans;
    testingWorkers[workerId]["Feature Rescaling"]["Standard Deviations"] = _featureRescalingSigmas;

    KORALI_START(testingWorkers[workerId]);
  }

  KORALI_WAITALL(testingWorkers);

  for (size_t workerId = 0; workerId < _testingSampleIds.size(); workerId++)
    _testingReward[workerId] = KORALI_GET(float, testingWorkers[workerId], "Testing Reward");
}

void Agent::updateBackgroundBatch(const size_t replacementIdx)
{
  // Initialize background batch
  if (_optimizeMaxEntropyObjective == false) return;
  if (_backgroundTrajectoryCount == 0)
  {
    _k->_logger->logInfo("Detailed", "Initializing background batch..\n");
    
    // Getting index of last experience
    size_t expId = _terminationBuffer.size() - 1;

    // Add trajectories from exploration phase to background batch
    for (size_t m = 0; m < _backgroundBatchSize; ++m)
    {
      const size_t episodeLength = _episodePosBuffer[expId];
      const size_t episodeStartIdx = expId - episodeLength;

      // Safety checks
      if (_terminationBuffer[expId] == e_nonTerminal)
        KORALI_LOG_ERROR("Experience %zu is not the start of a trajectory.", expId);
      if (episodeStartIdx == 0 && m < _backgroundBatchSize - 1)
        KORALI_LOG_ERROR("Increase exploration phase, not enough trajectories sampled (%zu/%zu) to fill a single back ground batch.", _backgroundTrajectoryCount, _backgroundBatchSize);

      // Allocate memory for new trajectory
      _backgroundTrajectoryStates[_backgroundTrajectoryCount].resize(episodeLength);
      _backgroundTrajectoryActions[_backgroundTrajectoryCount].resize(episodeLength);
      _backgroundTrajectoryFeatures[_backgroundTrajectoryCount].resize(episodeLength);

      // Store background trajectory
      for (size_t i = 0; i < episodeLength; ++i)
      {
        _backgroundPolicyHyperparameter[_backgroundTrajectoryCount] = _policyBuffer[episodeStartIdx];
        _backgroundTrajectoryStates[_backgroundTrajectoryCount][i] = _stateBuffer[episodeStartIdx + i];
        _backgroundTrajectoryActions[_backgroundTrajectoryCount][i] = _actionBuffer[episodeStartIdx + i];
        _backgroundTrajectoryFeatures[_backgroundTrajectoryCount][i] = _featureBuffer[episodeStartIdx + i];
      }

      // Increase background sample counter
      _backgroundTrajectoryCount++;

      // Decrease counter to move to the next previous last experience of a trajectory
      expId = episodeStartIdx - 1;
    }

    if (_backgroundTrajectoryCount != _backgroundBatchSize)
      KORALI_LOG_ERROR("Error during background batch intialization. Size is %zu but should be %zu.", _backgroundTrajectoryCount, _backgroundBatchSize);

    // Evaluate all trajectory logprobabilities, at the beginning all trajectories sampled from same arbitrary policy
    for (size_t i = 0; i < _backgroundTrajectoryCount; ++i)
    {
      _backgroundTrajectoryLogProbabilities[i].resize(_backgroundSampleSize + 1);
      const float trajectoryLogP = evaluateTrajectoryLogProbability(_backgroundTrajectoryStates[i], _backgroundTrajectoryActions[i], _backgroundPolicyHyperparameter[i]);
      if (_useFusionDistribution)
      {
        // Insert probability from linear policy first
        _backgroundTrajectoryLogProbabilities[i][0] = evaluateTrajectoryLogProbabilityWithObservedPolicy(_backgroundTrajectoryStates[i], _backgroundTrajectoryActions[i]);
        for (size_t j = 0; j < _backgroundTrajectoryCount; ++j)
          _backgroundTrajectoryLogProbabilities[i][j + 1] = trajectoryLogP;
      }
      else
      {
        _backgroundTrajectoryLogProbabilities[i][i + 1] = trajectoryLogP;
      }
    }
  }
  // Insert latest trajectory to background batch until full
  else if (_backgroundTrajectoryCount < _backgroundSampleSize)
  {
    _k->_logger->logInfo("Detailed", "Updating background batch (%zu trajectories stored)..\n", _backgroundTrajectoryCount);
    const size_t expId = _terminationBuffer.size() - 1;
    const size_t episodeLength = _episodePosBuffer[expId];
    const size_t episodeStartIdx = expId - episodeLength;

    // Safety check
    if (_episodeIdBuffer[episodeStartIdx] != _episodeIdBuffer[expId])
      KORALI_LOG_ERROR("Experience %zu is not the start of same trajectory.", episodeStartIdx);

    // Allocate memory for new trajectory
    _backgroundTrajectoryStates[_backgroundTrajectoryCount].resize(episodeLength);
    _backgroundTrajectoryActions[_backgroundTrajectoryCount].resize(episodeLength);
    _backgroundTrajectoryFeatures[_backgroundTrajectoryCount].resize(episodeLength);

    // Store background trajectory
    for (size_t i = 0; i < episodeLength; ++i)
    {
      _backgroundPolicyHyperparameter[_backgroundTrajectoryCount] = _policyBuffer[episodeStartIdx];
      _backgroundTrajectoryStates[_backgroundTrajectoryCount][i] = _stateBuffer[episodeStartIdx + i];
      _backgroundTrajectoryActions[_backgroundTrajectoryCount][i] = _actionBuffer[episodeStartIdx + i];
      _backgroundTrajectoryFeatures[_backgroundTrajectoryCount][i] = _featureBuffer[episodeStartIdx + i];
    }

    // Increase background sample counter
    _backgroundTrajectoryCount++;
    _backgroundTrajectoryLogProbabilities[_backgroundTrajectoryCount - 1].resize(_backgroundSampleSize + 1);

    if (_useFusionDistribution)
    {
      // For all previous background trajectories evaluate log probability with newest policy
      for (size_t i = 0; i < _backgroundTrajectoryCount - 1; ++i)
        _backgroundTrajectoryLogProbabilities[i][_backgroundTrajectoryCount] = evaluateTrajectoryLogProbability(_backgroundTrajectoryStates[i], _backgroundTrajectoryActions[i], _backgroundPolicyHyperparameter[_backgroundTrajectoryCount - 1]);

      // For newest policy evaluate trajectory log probability with observed policy, all previous policies, and the current one
      _backgroundTrajectoryLogProbabilities[_backgroundTrajectoryCount - 1][0] = evaluateTrajectoryLogProbabilityWithObservedPolicy(_backgroundTrajectoryStates[_backgroundTrajectoryCount - 1], _backgroundTrajectoryActions[_backgroundTrajectoryCount - 1]);
      for (size_t i = 0; i < _backgroundTrajectoryCount; ++i)
        _backgroundTrajectoryLogProbabilities[_backgroundTrajectoryCount - 1][i + 1] = evaluateTrajectoryLogProbability(_backgroundTrajectoryStates[_backgroundTrajectoryCount - 1], _backgroundTrajectoryActions[_backgroundTrajectoryCount - 1], _backgroundPolicyHyperparameter[i]);
    }
    else
    {
      // Evaluate trajectory log probability with own policy
      _backgroundTrajectoryLogProbabilities[_backgroundTrajectoryCount - 1][_backgroundTrajectoryCount] = evaluateTrajectoryLogProbability(_backgroundTrajectoryStates[_backgroundTrajectoryCount - 1], _backgroundTrajectoryActions[_backgroundTrajectoryCount - 1], _backgroundPolicyHyperparameter[_backgroundTrajectoryCount - 1]);
    }
  }
  // replace background trajectories
  //else if (replacementIdx < _backgroundSampleSize)
  else
  {
    _k->_logger->logInfo("Detailed", "Replacing background trajectories (%zu trajectories stored)..\n", _backgroundTrajectoryCount);
    const size_t expId = _terminationBuffer.size() - 1;
    const size_t episodeLength = _episodePosBuffer[expId];
    const size_t episodeStartIdx = expId - episodeLength;

    // Safety checks
    if (_episodeIdBuffer[episodeStartIdx] != _episodeIdBuffer[expId])
      KORALI_LOG_ERROR("Experience %zu is not the start of same trajectory.", episodeStartIdx);

    // Allocate memory for new trajectory
    _backgroundTrajectoryStates[replacementIdx].resize(episodeLength);
    _backgroundTrajectoryActions[replacementIdx].resize(episodeLength);
    _backgroundTrajectoryFeatures[replacementIdx].resize(episodeLength);

    // Replace background trajectory
#pragma omp parallel for
    for (size_t i = 0; i < episodeLength; ++i)
    {
      _backgroundTrajectoryStates[replacementIdx][i] = _stateBuffer[episodeStartIdx + i];
      _backgroundTrajectoryActions[replacementIdx][i] = _actionBuffer[episodeStartIdx + i];
      _backgroundTrajectoryFeatures[replacementIdx][i] = _featureBuffer[episodeStartIdx + i];
    }

    _backgroundPolicyHyperparameter[replacementIdx] = _policyBuffer[episodeStartIdx];
    _backgroundTrajectoryCount++;

    if (_useFusionDistribution)
    {
      // For all previous background trajectories evaluate log probability with newest policy
      for (size_t i = 0; i < _backgroundSampleSize; ++i)
        if (i != replacementIdx)
          _backgroundTrajectoryLogProbabilities[i][replacementIdx + 1] = evaluateTrajectoryLogProbability(_backgroundTrajectoryStates[i], _backgroundTrajectoryActions[i], _backgroundPolicyHyperparameter[replacementIdx]);

      // For newest policy evaluate trajectory log probability with observed policy, all previous policies, and the current one
      _backgroundTrajectoryLogProbabilities[replacementIdx][0] = evaluateTrajectoryLogProbabilityWithObservedPolicy(_backgroundTrajectoryStates[replacementIdx], _backgroundTrajectoryActions[replacementIdx]);
      for (size_t i = 0; i < _backgroundSampleSize; ++i)
        _backgroundTrajectoryLogProbabilities[replacementIdx][i + 1] = evaluateTrajectoryLogProbability(_backgroundTrajectoryStates[replacementIdx], _backgroundTrajectoryActions[replacementIdx], _backgroundPolicyHyperparameter[i]);
    }
    else
    {
      _backgroundTrajectoryLogProbabilities[replacementIdx][replacementIdx + 1] = evaluateTrajectoryLogProbability(_backgroundTrajectoryStates[replacementIdx], _backgroundTrajectoryActions[replacementIdx], _backgroundPolicyHyperparameter[replacementIdx]);
    }
  }
  //else
  //{
  //  _k->_logger->logInfo("Detailed", "Skipping background trajectory update.\n");
  //}
}

void Agent::updateDemonstrationBatch(const size_t replacementIdx)
{
  if (_optimizeMaxEntropyObjective == false) return;
  if (_demonstrationTrajectoryLogProbabilities.size() == 0)
  {
    _k->_logger->logInfo("Detailed", "Initializing demonstration batch..\n");
    _demonstrationTrajectoryLogProbabilities.resize(_problem->_numberObservedTrajectories);
    for (size_t m = 0; m < _problem->_numberObservedTrajectories; ++m)
    {
      _demonstrationTrajectoryLogProbabilities[m].resize(_backgroundSampleSize + 1);
      _demonstrationTrajectoryLogProbabilities[m][0] = evaluateTrajectoryLogProbabilityWithObservedPolicy(_problem->_observationsStates[m], _problem->_observationsActions[m]);

      if (_useFusionDistribution)
        for (size_t i = 0; i < _backgroundTrajectoryCount; ++i)
        {
          _demonstrationTrajectoryLogProbabilities[m][i + 1] = evaluateTrajectoryLogProbability(_problem->_observationsStates[m], _problem->_observationsActions[m], _backgroundPolicyHyperparameter[i]);
        }
    }
  }
  else if (_backgroundTrajectoryCount < _backgroundSampleSize && _useFusionDistribution)
  // Evaluate demonstrations with latest policy
  {
    _k->_logger->logInfo("Detailed", "Updating demonstration batch with new trajectory..\n");
    for (size_t m = 0; m < _problem->_numberObservedTrajectories; ++m)
    {
      _demonstrationTrajectoryLogProbabilities[m][_backgroundTrajectoryCount] = evaluateTrajectoryLogProbability(_problem->_observationsStates[m], _problem->_observationsActions[m], _backgroundPolicyHyperparameter[_backgroundTrajectoryCount - 1]);
    }
  }
  else
  // Evaluate demonstrations with latest policy
  {
    _k->_logger->logInfo("Detailed", "Updating demonstration batch with replaced trajectory..\n");
    for (size_t m = 0; m < _problem->_numberObservedTrajectories; ++m)
    {
      _demonstrationTrajectoryLogProbabilities[m][replacementIdx + 1] = evaluateTrajectoryLogProbability(_problem->_observationsStates[m], _problem->_observationsActions[m], _backgroundPolicyHyperparameter[replacementIdx]);
    }
  }
}

std::vector<float> Agent::calculateReward(const std::vector<std::vector<std::vector<float>>> &featuresBatch) const
{
  auto output = _rewardFunctionLearner->getEvaluation(featuresBatch);
  std::vector<float> rewards(output.size());
  for (size_t b = 0; b < output.size(); ++b)
  {
    rewards[b] = output[b][0];
    if (std::isfinite(rewards[b]) == false) KORALI_LOG_ERROR("Calculate reward returned nonfinite number!.");
  }
  return rewards;
}

void Agent::updateRewardFunction()
{
  const size_t stepsPerUpdate = 1;
  const size_t totalBatchSize = _backgroundBatchSize + _demonstrationBatchSize;

  const unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::minstd_rand0 generator(seed);
  _k->_logger->logInfo("Detailed", "Update Reward Function..\n");

  for (size_t stepNum = 0; stepNum < stepsPerUpdate; ++stepNum)
  {
    // Randomize demonstration batch
    std::vector<size_t> randomDemonstrationIndexes(_problem->_numberObservedTrajectories);
    std::iota(std::begin(randomDemonstrationIndexes), std::end(randomDemonstrationIndexes), 0);
    std::shuffle(randomDemonstrationIndexes.begin(), randomDemonstrationIndexes.end(), generator);

    // Randomize background batch
    const size_t maxRand = std::min(_backgroundTrajectoryCount, _backgroundSampleSize);
    std::vector<size_t> randomBackgroundIndexes(maxRand);
    std::iota(std::begin(randomBackgroundIndexes), std::end(randomBackgroundIndexes), 0);
    std::shuffle(randomBackgroundIndexes.begin(), randomBackgroundIndexes.end(), generator);

    // For statistics
    float demoLogP = 0.;

    // Calculate cumulative rewards for demonstration batch and extract trajectory probabilities
    float cumDemoReward = 0.;
    std::vector<float> cumulativeRewardsDemonstrationBatch(_demonstrationBatchSize, 0.0);
    std::vector<std::vector<float>> gradientCumulativeRewardFunctionDemonstrationBatch(_demonstrationBatchSize, std::vector<float>(_rewardFunctionLearner->_neuralNetwork->_hyperparameterCount, 0.));
    std::vector<std::vector<float>> demonstrationTrajectoryLogProbabilities(_demonstrationBatchSize, std::vector<float>(totalBatchSize));
    for (size_t n = 0; n < _demonstrationBatchSize; ++n)
    {
      const size_t demIdx = randomDemonstrationIndexes[n];
      const size_t observedTrajectoryLength = _problem->_observationsFeatures[demIdx].size();

      size_t t = 0;
      float cumReward = 0.0;

      while (t < observedTrajectoryLength)
      {
        std::vector<std::vector<std::vector<float>>> featuresBatch(_rewardFunctionBatchSize, std::vector<std::vector<float>>(1, std::vector<float>(_problem->_featureVectorSize, 0.)));
        std::vector<std::vector<float>> backwardMultiplier(_rewardFunctionBatchSize, std::vector<float>(1, 0.));

        const size_t batchSize = std::min(_rewardFunctionBatchSize, observedTrajectoryLength - t);
        for (size_t b = 0; b < batchSize; ++b)
        {
          featuresBatch[b][0] = _problem->_observationsFeatures[demIdx][b][0];
          backwardMultiplier[b] = std::vector<float>(1, 1.);
        }

        const auto rewards = calculateReward(featuresBatch);

        // Accumulate cumulative reward
        for (size_t b = 0; b < batchSize; ++b)
        {
          cumReward += rewards[b];
        }

        // Backward dummy
        _rewardFunctionLearner->_neuralNetwork->backward(backwardMultiplier);

        // Accumulate gradients from demonstrations
        const auto rewardGradients = _rewardFunctionLearner->_neuralNetwork->getHyperparameterGradients(_rewardFunctionBatchSize);

#pragma omp parallel for
        for (size_t i = 0; i < rewardGradients.size(); ++i)
        {
          gradientCumulativeRewardFunctionDemonstrationBatch[n][i] += rewardGradients[i];
        }

        t += batchSize;
      }

      cumulativeRewardsDemonstrationBatch[n] = cumReward;
      cumDemoReward += cumReward;

      if (_optimizeMaxEntropyObjective == true)
      {
        for (size_t i = 0; i < _demonstrationBatchSize; ++i)
        {
          demonstrationTrajectoryLogProbabilities[n][i] = _demonstrationTrajectoryLogProbabilities[demIdx][0];
          if (std::isfinite(demonstrationTrajectoryLogProbabilities[n][i]) == false) KORALI_LOG_ERROR("Demonstration trajectory log probability is not finite");
          demoLogP += demonstrationTrajectoryLogProbabilities[n][i];
        }

        for (size_t m = 0; m < _backgroundBatchSize; ++m)
        {
          const size_t bckIdx = randomBackgroundIndexes[m];
          demonstrationTrajectoryLogProbabilities[n][_demonstrationBatchSize + m] = _demonstrationTrajectoryLogProbabilities[demIdx][bckIdx + 1];
          if (std::isfinite(demonstrationTrajectoryLogProbabilities[n][_demonstrationBatchSize + m]) == false) KORALI_LOG_ERROR("Demonstration trajectory log probability is not finite");
          demoLogP += demonstrationTrajectoryLogProbabilities[n][_demonstrationBatchSize + m];
        }
      }
    }
      
    // Record history of average feature reward of demonstrations
    _demonstrationFeatureReward.push_back(cumDemoReward / (float)_demonstrationBatchSize);

    if (_optimizeMaxEntropyObjective == true)
    {
      // Record history of logprobability of demonstrations
      _demonstrationLogProbability.push_back(demoLogP / (float)(_demonstrationBatchSize + _backgroundBatchSize));

      // Calculate cumulative rewards for randomized background batch and extract trajectory probabilities
      std::vector<float> cumulativeRewardsBackgroundBatch(_backgroundBatchSize, 0.0);
      std::vector<std::vector<float>> gradientCumulativeRewardFunctionBackgroundBatch(_backgroundBatchSize, std::vector<float>(_rewardFunctionLearner->_neuralNetwork->_hyperparameterCount, 0.));
      std::vector<std::vector<float>> backgroundTrajectoryLogProbabilities(_backgroundBatchSize, std::vector<float>(totalBatchSize));

      for (size_t m = 0; m < _backgroundBatchSize; ++m)
      {
        const size_t bckIdx = randomBackgroundIndexes[m];
        const size_t backgroundTrajectoryLength = _backgroundTrajectoryFeatures[bckIdx].size();

        size_t t = 0;
        float cumReward = 0.0;

        while (t < backgroundTrajectoryLength)
        {
          std::vector<std::vector<std::vector<float>>> featuresBatch(_rewardFunctionBatchSize, std::vector<std::vector<float>>(1, std::vector<float>(_problem->_featureVectorSize, 0.)));
          std::vector<std::vector<float>> backwardMultiplier(_rewardFunctionBatchSize, std::vector<float>(1, 0.));

          const size_t batchSize = std::min(_rewardFunctionBatchSize, backgroundTrajectoryLength - t);
          for (size_t b = 0; b < batchSize; ++b)
          {
            featuresBatch[b] = {_backgroundTrajectoryFeatures[bckIdx][b]};
            backwardMultiplier[b] = std::vector<float>(1, 1.);
          }

          const auto rewards = calculateReward(featuresBatch);

          // Accumulate cumulative reward
          for (size_t b = 0; b < batchSize; ++b)
          {
            cumReward += rewards[b];
          }

          // Backward dummy
          _rewardFunctionLearner->_neuralNetwork->backward(backwardMultiplier);

          // Accumulate gradients from demonstrations
          const auto rewardGradients = _rewardFunctionLearner->_neuralNetwork->getHyperparameterGradients(_rewardFunctionBatchSize);

#pragma omp parallel for
          for (size_t i = 0; i < rewardGradients.size(); ++i)
          {
            gradientCumulativeRewardFunctionBackgroundBatch[m][i] += rewardGradients[i];
          }

          t += batchSize;
        }

        cumulativeRewardsBackgroundBatch[m] = cumReward;

        for (size_t n = 0; n < _demonstrationBatchSize; ++n)
        {
          backgroundTrajectoryLogProbabilities[m][n] = _backgroundTrajectoryLogProbabilities[bckIdx][0]; // probability from observed policy
          if (std::isfinite(backgroundTrajectoryLogProbabilities[m][n]) == false) KORALI_LOG_ERROR("Background trajectory log probability is not finite");
        }

        if (_useFusionDistribution)
          for (size_t i = 0; i < _backgroundBatchSize; ++i)
          {
            const size_t bckIdx2 = randomBackgroundIndexes[i];
            backgroundTrajectoryLogProbabilities[m][_demonstrationBatchSize + i] = _backgroundTrajectoryLogProbabilities[bckIdx][bckIdx2 + 1];
            if (std::isfinite(backgroundTrajectoryLogProbabilities[m][_demonstrationBatchSize + 1]) == false) KORALI_LOG_ERROR("Background trajectory log probability is not finite");
          }
        else
        {
          KORALI_LOG_ERROR("Not yet implemneted.");
        }
      }

      // Calculate importance weights of background batch
      std::vector<float> backgroundBatchLogImportanceWeights(_backgroundBatchSize);
      for (size_t m = 0; m < _backgroundBatchSize; ++m)
      {
        // Caclculate importance weight (1/K sum_k q_k(T))^-1
        if (_useFusionDistribution)
          backgroundBatchLogImportanceWeights[m] = std::log((float)totalBatchSize) - logSumExp(backgroundTrajectoryLogProbabilities[m]);
        else
          backgroundBatchLogImportanceWeights[m] = -backgroundTrajectoryLogProbabilities[m][m + 1];
      }

      // Calculate importance weights of demonstration batch
      std::vector<float> demonstrationBatchLogImportanceWeights(_demonstrationBatchSize);
      for (size_t n = 0; n < _demonstrationBatchSize; ++n)
      {
        // Caclculate importance weight (1/K sum_k q_k(T))^-1
        if (_useFusionDistribution)
          demonstrationBatchLogImportanceWeights[n] = std::log((float)totalBatchSize) - logSumExp(demonstrationTrajectoryLogProbabilities[n]);
        else
          demonstrationBatchLogImportanceWeights[n] = -demonstrationTrajectoryLogProbabilities[n][0];
      }

      // Preparation for calculation of log partition function with log-sum-exp trick
      float maxExp = -Inf;
      float maxExpEss = -Inf;
      for (size_t m = 0; m < _backgroundBatchSize; ++m)
      {
        const float exp = backgroundBatchLogImportanceWeights[m] + cumulativeRewardsBackgroundBatch[m];
        if (exp > maxExp) maxExp = exp;
        if (backgroundBatchLogImportanceWeights[m] > maxExpEss) maxExpEss = backgroundBatchLogImportanceWeights[m];
      }

      for (size_t n = 0; n < _demonstrationBatchSize; ++n)
      {
        const float exp = demonstrationBatchLogImportanceWeights[n] + cumulativeRewardsDemonstrationBatch[n];
        if (exp > maxExp) maxExp = exp;
        if (demonstrationBatchLogImportanceWeights[n] > maxExpEss) maxExpEss = demonstrationBatchLogImportanceWeights[n];
      }

      float sumExpNoMax = 0.0;
      float sumExpNoMaxEss = 0.0;
      float sum2ExpNoMaxEss = 0.0;

      for (size_t m = 0; m < _backgroundBatchSize; ++m)
      {
        const float exp = backgroundBatchLogImportanceWeights[m] + cumulativeRewardsBackgroundBatch[m];
        sumExpNoMax += std::exp(exp - maxExp);
        sumExpNoMaxEss += std::exp(backgroundBatchLogImportanceWeights[m] - maxExpEss);
        sum2ExpNoMaxEss += std::exp(2. * backgroundBatchLogImportanceWeights[m] - 2. * maxExpEss);
      }

      for (size_t n = 0; n < _demonstrationBatchSize; ++n)
      {
        const float exp = demonstrationBatchLogImportanceWeights[n] + cumulativeRewardsDemonstrationBatch[n];
        sumExpNoMax += std::exp(exp - maxExp);
        sumExpNoMaxEss += std::exp(demonstrationBatchLogImportanceWeights[n] - maxExpEss);
        sum2ExpNoMaxEss += std::exp(2. * demonstrationBatchLogImportanceWeights[n] - 2. * maxExpEss);
      }

      // Calculate log of partition function
      _logPartitionFunction = std::log(sumExpNoMax) + maxExp - std::log((float)totalBatchSize);

      // Calculate Effective Sample Size
      const float lEss1 = 2. * (std::log(sumExpNoMaxEss) + maxExpEss);
      const float lEss2 = std::log(sum2ExpNoMaxEss) + 2 * maxExpEss;
      const float ess = std::exp(lEss1 - lEss2);

      // Reset gradient
      std::fill(_maxEntropyGradient.begin(), _maxEntropyGradient.end(), 0.);

      // Calculate gradient of loglikelihood (contribution from partition function & background batch)
      const float invTotalBatchSize = 1. / (float)totalBatchSize;

      for (size_t m = 0; m < _backgroundBatchSize; ++m)
      {
        const float mult = std::exp(backgroundBatchLogImportanceWeights[m] + cumulativeRewardsBackgroundBatch[m] - _logPartitionFunction) * invTotalBatchSize;
        if (mult < 0.) KORALI_LOG_ERROR("Mult negative! is %f", mult);
        if (mult > 1.1) KORALI_LOG_ERROR("Mult larger one! is %f", mult);

#pragma omp parallel for
        for (size_t k = 0; k < _maxEntropyGradient.size(); ++k)
        {
          _maxEntropyGradient[k] -= _demonstrationBatchSize * mult * gradientCumulativeRewardFunctionBackgroundBatch[m][k];
        }
      }
      size_t negBracket = 0.;
      // Calculate gradient of loglikelihood wrt. feature weights (contribution from partition function, demonstration return & demonstration batch)
      for (size_t n = 0; n < _demonstrationBatchSize; ++n)
      {
        const float mult = std::exp(demonstrationBatchLogImportanceWeights[n] + cumulativeRewardsDemonstrationBatch[n] - _logPartitionFunction) * invTotalBatchSize;
        if (mult < 0.) KORALI_LOG_ERROR("Mult negative! is %f", mult);
        if (mult > 1.01) KORALI_LOG_ERROR("Mult larger one! is %f", mult);

        if ((1. - _demonstrationBatchSize * mult) < 0.) negBracket++;

#pragma omp parallel for
        for (size_t k = 0; k < _maxEntropyGradient.size(); ++k)
        {
          _maxEntropyGradient[k] += (1. - _demonstrationBatchSize * mult) * gradientCumulativeRewardFunctionDemonstrationBatch[n][k];
          if (std::isfinite(_maxEntropyGradient[k]) == false) KORALI_LOG_ERROR("Reward gradient not finite!");
        }
        _k->_logger->logInfo("Detailed", "Neg Bracket (%zu/%zu)!\n", negBracket, _demonstrationBatchSize);
        _k->_logger->logInfo("Detailed", "Effective Sample Size (%f/%f)!\n", ess, (float)(_demonstrationBatchSize + _backgroundBatchSize));
      }

      // Record history of importance weights
      _demonstrationBatchImportanceWeight.push_back(logSumExp(demonstrationBatchLogImportanceWeights) - std::log((float)_demonstrationBatchSize));
      _backgroundBatchImportanceWeight.push_back(logSumExp(demonstrationBatchLogImportanceWeights) - std::log((float)_backgroundBatchSize));
      _effectiveSampleSize.push_back(ess);

      // Record history of max entropy objective
      _maxEntropyObjective.push_back(cumDemoReward - _demonstrationBatchSize * _logPartitionFunction);
    }
    else
    {
      for (size_t n = 0; n < _demonstrationBatchSize; ++n)
      {
#pragma omp parallel for
        for (size_t k = 0; k < _maxEntropyGradient.size(); ++k)
        {
          // Contribution from demonstration return
          _maxEntropyGradient[k] += gradientCumulativeRewardFunctionDemonstrationBatch[n][k];
          if (std::isfinite(_maxEntropyGradient[k]) == false) KORALI_LOG_ERROR("Reward gradient not finite!");
        }
      }
    }

    // Passing hyperparameter gradients through an Adam update
    _rewardFunctionLearner->_optimizer->processResult(_maxEntropyGradient);

    // Getting new set of hyperparameters from Adam
    _rewardFunctionLearner->_neuralNetwork->setHyperparameters(_rewardFunctionLearner->_optimizer->_currentValue);
  }
  _k->_logger->logInfo("Detailed", "Done!\n");
}

void Agent::rescaleStates()
{
  // Calculation of state moments
  std::vector<std::vector<float>> sumStates(_problem->_agentsPerEnvironment, std::vector<float>(_problem->_stateVectorSize, 0.0f));
  std::vector<std::vector<float>> squaredSumStates(_problem->_agentsPerEnvironment, std::vector<float>(_problem->_stateVectorSize, 0.0f));

  for (size_t i = 0; i < _stateBuffer.size(); ++i)
    for (size_t a = 0; a < _problem->_agentsPerEnvironment; ++a)
      for (size_t d = 0; d < _problem->_stateVectorSize; ++d)
      {
        sumStates[a][d] += _stateBuffer[i][a][d];
        squaredSumStates[a][d] += _stateBuffer[i][a][d] * _stateBuffer[i][a][d];
      }

  _k->_logger->logInfo("Detailed", " + Using State Normalization N(Mean, Sigma):\n");

  for (size_t a = 0; a < _problem->_agentsPerEnvironment; ++a)
    for (size_t d = 0; d < _problem->_stateVectorSize; ++d)
    {
      _stateRescalingMeans[a][d] = sumStates[a][d] / (float)_stateBuffer.size();
      if (std::isfinite(_stateRescalingMeans[a][d]) == false) _stateRescalingMeans[a][d] = 0.0f;

      _stateRescalingSigmas[a][d] = std::sqrt(squaredSumStates[a][d] / (float)_stateBuffer.size() - _stateRescalingMeans[a][d] * _stateRescalingMeans[a][d]);
      if (std::isfinite(_stateRescalingSigmas[a][d]) == false) KORALI_LOG_ERROR("State sdev not finite. Cannot scale states.");
      if (std::isfinite(_stateRescalingSigmas[a][d]) <= 1e-9) KORALI_LOG_ERROR("State sdev below threshold. Cannot scale states.");

      _k->_logger->logInfo("Detailed", " + State [%zu]: N(%f, %f)\n", d, _stateRescalingMeans[a][d], _stateRescalingSigmas[a][d]);
    }

  // Actual rescaling of initial states
  for (size_t i = 0; i < _stateBuffer.size(); ++i)
    for (size_t a = 0; a < _problem->_agentsPerEnvironment; ++a)
      for (size_t d = 0; d < _problem->_stateVectorSize; ++d)
        _stateBuffer[i][a][d] = (_stateBuffer[i][a][d] - _stateRescalingMeans[a][d]) / _stateRescalingSigmas[a][d];
}

void Agent::rescaleFeatures()
{
  // Calculation of state moments
  std::vector<std::vector<float>> sumFeatures(_problem->_agentsPerEnvironment, std::vector<float>(_problem->_featureVectorSize, 0.0));
  std::vector<std::vector<float>> squaredSumFeatures(_problem->_agentsPerEnvironment, std::vector<float>(_problem->_featureVectorSize, 0.0));

  for (size_t i = 0; i < _featureBuffer.size(); ++i)
    for (size_t a = 0; a < _problem->_agentsPerEnvironment; ++a)
      for (size_t d = 0; d < _problem->_featureVectorSize; ++d)
      {
        sumFeatures[a][d] += _featureBuffer[i][a][d];
        squaredSumFeatures[a][d] += _featureBuffer[i][a][d] * _featureBuffer[i][a][d];
      }

  _k->_logger->logInfo("Detailed", " + Using State Normalization N(Mean, Sigma):\n");

  for (size_t a = 0; a < _problem->_agentsPerEnvironment; ++a)
    for (size_t d = 0; d < _problem->_featureVectorSize; ++d)
    {
      _featureRescalingMeans[a][d] = sumFeatures[a][d] / (float)_featureBuffer.size();
      if (std::isfinite(_featureRescalingMeans[a][d]) == false) KORALI_LOG_ERROR("Feature mean not finite. Cannot shift features.");

      _featureRescalingSigmas[a][d] = std::sqrt(squaredSumFeatures[a][d] / (float)_featureBuffer.size() - _featureRescalingMeans[a][d] * _featureRescalingMeans[a][d]);
      if (std::isfinite(_featureRescalingSigmas[a][d]) == false) KORALI_LOG_ERROR("Feature sdev not finite. Cannot scale features.");

      _k->_logger->logInfo("Detailed", " + Feature [%zu]: N(%f, %f)\n", d, _featureRescalingMeans[a][d], _featureRescalingSigmas[a][d]);
    }

  // Actual rescaling of initial features
  for (size_t i = 0; i < _featureBuffer.size(); ++i)
    for (size_t a = 0; a < _problem->_agentsPerEnvironment; ++a)
    {
      for (size_t d = 0; d < _problem->_featureVectorSize; ++d)
        _featureBuffer[i][a][d] = (_featureBuffer[i][a][d] - _featureRescalingMeans[a][d]) / _featureRescalingSigmas[a][d];
    }

}

void Agent::attendWorker(size_t workerId)
{
  auto beginTime = std::chrono::steady_clock::now(); // Profiling

  // Storage for the incoming message
  knlohmann::json message;

  // Retrieving the experience, if any has arrived for the current agent.
  if (_workers[workerId].retrievePendingMessage(message))
  {
    // Getting episode Id
    size_t episodeId = message["Sample Id"];
    message["Episodes"]["Sample Id"] = episodeId;

    // If agent requested new policy, send the new hyperparameters
    if (message["Action"] == "Request New Policy")
    {
      KORALI_SEND_MSG_TO_SAMPLE(_workers[workerId], _trainingCurrentPolicies["Policy Hyperparameters"]);
    }

    // Process episode(s) incoming from the agent(s)
    if (message["Action"] == "Send Episodes")
    {
      // Process every episode received and its experiences (add them to replay memory)
      processEpisode(message["Episodes"]);

      // Waiting for the agent to come back with all the information
      KORALI_WAIT(_workers[workerId]);

      // Getting the training reward of the latest episodes
      _trainingLastReward = KORALI_GET(std::vector<float>, _workers[workerId], "Training Rewards");

      // Keeping training statistics. Updating if exceeded best training policy so far.
      for (size_t a = 0; a < _problem->_agentsPerEnvironment; a++)
      {
        _experienceReplayOffPolicyHistory.push_back(_experienceReplayOffPolicyRatio);
        if (_trainingLastReward[a] > _trainingBestReward[a])
        {
          _trainingBestReward[a] = _trainingLastReward[a];
          _trainingBestEpisodeId[a] = episodeId;
        }
        _trainingRewardHistory[a].push_back(_trainingLastReward[a]);
      }

      // Storing bookkeeping information
      _trainingExperienceHistory.push_back(message["Episodes"]["Experiences"].size());

      // If the policy has exceeded the threshold during training, we gather its statistics
      if (_workers[workerId]["Tested Policy"] == true)
      {
        _testingCandidateCount++;
        _testingBestReward = KORALI_GET(float, _workers[workerId], "Best Testing Reward");
        _testingWorstReward = KORALI_GET(float, _workers[workerId], "Worst Testing Reward");
        _testingAverageReward = KORALI_GET(float, _workers[workerId], "Average Testing Reward");
        _testingAverageRewardHistory.push_back(_testingAverageReward);

        // If the average testing reward is better than the previous best, replace it
        // and store hyperparameters as best so far.
        if (_testingAverageReward > _testingBestAverageReward)
        {
          _testingBestAverageReward = _testingAverageReward;
          _testingBestEpisodeId = episodeId;
          for (size_t d = 0; d < _problem->_policiesPerEnvironment; ++d)
            _testingBestPolicies["Policy Hyperparameters"][d] = _workers[workerId]["Policy Hyperparameters"][d];
        }
      }

      // Obtaining profiling information
      _sessionWorkerComputationTime += KORALI_GET(double, _workers[workerId], "Computation Time");
      _sessionWorkerCommunicationTime += KORALI_GET(double, _workers[workerId], "Communication Time");
      _sessionPolicyEvaluationTime += KORALI_GET(double, _workers[workerId], "Policy Evaluation Time");
      _generationWorkerComputationTime += KORALI_GET(double, _workers[workerId], "Computation Time");
      _generationWorkerCommunicationTime += KORALI_GET(double, _workers[workerId], "Communication Time");
      _generationPolicyEvaluationTime += KORALI_GET(double, _workers[workerId], "Policy Evaluation Time");

      // Set agent as finished
      _isWorkerRunning[workerId] = false;
    }
  }

  auto endTime = std::chrono::steady_clock::now();                                                                     // Profiling
  _sessionWorkerAttendingTime += std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - beginTime).count();    // Profiling
  _generationWorkerAttendingTime += std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - beginTime).count(); // Profiling
}

void Agent::processEpisode(knlohmann::json &episode)
{
  /*********************************************************************
   * Adding episode's experiences into the replay memory
   *********************************************************************/
  const size_t episodeId = episode["Sample Id"];
  const size_t numAgents = _problem->_agentsPerEnvironment;

  // Storage for the episode's cumulative feature reward
  std::vector<float> cumulativeFeatureReward(numAgents, 0.0f);

  // Go over experiences in episode
  const size_t episodeExperienceCount = episode["Experiences"].size();
  for (size_t expId = 0; expId < episodeExperienceCount; expId++)
  {
    // Put state to replay memory
    _stateBuffer.add(episode["Experiences"][expId]["State"].get<std::vector<std::vector<float>>>());

    // Get action and put it to replay memory
    _actionBuffer.add(episode["Experiences"][expId]["Action"].get<std::vector<std::vector<float>>>());

    // Getting features
    _featureBuffer.add(episode["Experiences"][expId]["Features"].get<std::vector<std::vector<float>>>());

    // Calculate reward
    std::vector<float> featureReward(numAgents, 0.0f);
    for (size_t a = 0; a < _problem->_agentsPerEnvironment; ++a)
    {
      // Getting reward (TODO: batch forwarding)
      featureReward[a] = calculateReward({{_featureBuffer[_featureBuffer.size() - 1][a]}})[0];

      // Accumulate feature reward
      cumulativeFeatureReward[a] += featureReward[a];
    }

    // For cooporative multi-agent model rewards are averaged
    if (_multiAgentRelationship == "Cooperation")
    {
      float avgReward = std::accumulate(featureReward.begin(), featureReward.end(), 0.);
      avgReward /= numAgents;
      featureReward = std::vector<float>(numAgents, avgReward);
    }

    // Update reward rescaling moments
    if (_rewardRescalingEnabled)
    {
      KORALI_LOG_ERROR("IRL does not allow reward rescaling");
    }

    // Put reward to replay memory
    for (size_t a = 0; a < numAgents; a++)
      _rewardBufferContiguous.add(featureReward[a]);

    _rewardUpdateBuffer.add(_rewardUpdateCount);

    // Checking and adding experience termination status and truncated state to replay memory
    termination_t termination;
    std::vector<std::vector<float>> truncatedState;
    std::vector<float> truncatedStateValue;

    if (episode["Experiences"][expId]["Termination"] == "Non Terminal") termination = e_nonTerminal;
    if (episode["Experiences"][expId]["Termination"] == "Terminal") termination = e_terminal;
    if (episode["Experiences"][expId]["Termination"] == "Truncated")
    {
      termination = e_truncated;
      truncatedState = episode["Experiences"][expId]["Truncated State"].get<std::vector<std::vector<float>>>();
    }

    _terminationBuffer.add(termination);
    _truncatedStateBuffer.add(truncatedState);
    _truncatedStateValueBuffer.add(truncatedStateValue);

    // Storing policy on episode start
    if (expId == 0)
    {
      assert(_problem->_policiesPerEnvironment == 1); // TODO: this works only for unique policy
      _policyBuffer.add(episode["Policy Hyperparameters"][0].get<std::vector<float>>());
    }
    else
    {
      _policyBuffer.add({}); // Placeholder
    }

    // Getting policy information and state value
    std::vector<policy_t> expPolicy(numAgents);
    std::vector<float> stateValue(numAgents);

    if (isDefined(episode["Experiences"][expId], "Policy", "State Value"))
    {
      stateValue = episode["Experiences"][expId]["Policy"]["State Value"].get<std::vector<float>>();
      for (size_t a = 0; a < numAgents; a++)
      {
        expPolicy[a].stateValue = stateValue[a];
      }
    }
    else
    {
      KORALI_LOG_ERROR("Policy has not produced state value for the current experience.\n");
    }
    for (size_t a = 0; a < numAgents; a++)
      _stateValueBufferContiguous.add(stateValue[a]);

    /* Story policy information for continuous action spaces */
    if (isDefined(episode["Experiences"][expId], "Policy", "Distribution Parameters"))
    {
      const auto distParams = episode["Experiences"][expId]["Policy"]["Distribution Parameters"].get<std::vector<std::vector<float>>>();
      for (size_t a = 0; a < numAgents; a++)
        expPolicy[a].distributionParameters = distParams[a];
    }

    if (isDefined(episode["Experiences"][expId], "Policy", "Unbounded Action"))
    {
      const auto unboundedAc = episode["Experiences"][expId]["Policy"]["Unbounded Action"].get<std::vector<std::vector<float>>>();
      for (size_t a = 0; a < numAgents; a++)
        expPolicy[a].unboundedAction = unboundedAc[a];
    }

    /* Story policy information for discrete action spaces */
    if (isDefined(episode["Experiences"][expId], "Policy", "Action Index"))
    {
      const auto actIdx = episode["Experiences"][expId]["Policy"]["Action Index"].get<std::vector<size_t>>();
      for (size_t a = 0; a < numAgents; a++)
        expPolicy[a].actionIndex = actIdx[a];
    }

    if (isDefined(episode["Experiences"][expId], "Policy", "Action Probabilities"))
    {
      const auto actProb = episode["Experiences"][expId]["Policy"]["Action Probabilities"].get<std::vector<std::vector<float>>>();
      for (size_t a = 0; a < numAgents; a++)
        expPolicy[a].actionProbabilities = actProb[a];
    }

    if (isDefined(episode["Experiences"][expId], "Policy", "Available Actions"))
    {
      const auto availAct = episode["Experiences"][expId]["Policy"]["Available Actions"].get<std::vector<std::vector<size_t>>>();
      for (size_t a = 0; a < numAgents; a++)
      {
        expPolicy[a].availableActions = availAct[a];
        if (std::accumulate(expPolicy[a].availableActions.begin(), expPolicy[a].availableActions.end(), 0) == 0)
          KORALI_LOG_ERROR("State with experience id %zu for agent %zu detected with no available actions.", expId, a);
      }
    }

    // Storing policy information in replay memory
    _expPolicyBuffer.add(expPolicy);
    _curPolicyBuffer.add(expPolicy);

    // Storing Episode information in replay memory
    _episodeIdBuffer.add(episodeId);
    _episodePosBuffer.add(expId);

    // Adding placeholder for retrace value
    for (size_t a = 0; a < numAgents; a++)
      _retraceValueBufferContiguous.add(0.0f);

    // If outgoing experience is off policy, subtract off policy counter
    if (_isOnPolicyBuffer.size() == _experienceReplayMaximumSize)
    {
      const auto &onPolicyBuffer = _isOnPolicyBuffer[0];

      size_t count = 1;
      // Consider all observation for the off-policy statistics
      if (_problem->_policiesPerEnvironment == 1)
        count = std::count(onPolicyBuffer.begin(), onPolicyBuffer.end(), false);

      // Update offPolicyCount
      for (size_t a = 0; a < numAgents; a++)
        if ((onPolicyBuffer[a] == false) || (_problem->_policiesPerEnvironment == 1))
        {
          // Safety check for overflow
          if (_experienceReplayOffPolicyCount[a] < count)
          {
            KORALI_LOG_ERROR("Agent %ld: _experienceReplayOffPolicyCount=%ld smaller than decrement %ld.\n", a, _experienceReplayOffPolicyCount[a], count);
            // count = _experienceReplayOffPolicyCount[d];
          }

          // Update off-policy count
          _experienceReplayOffPolicyCount[a] -= count;
        }
    }

    // Adding new experience's on policiness (by default is true when adding it to the ER)
    _isOnPolicyBuffer.add(std::vector<char>(numAgents, true));

    // Initialize experience's importance weight (1.0 because its freshly produced)
    _importanceWeightBuffer.add(std::vector<float>(numAgents, 1.0f));
    for (size_t a = 0; a < numAgents; a++)
      _truncatedImportanceWeightBufferContiguous.add(1.0f);
    _productImportanceWeightBuffer.add(1.0f);
  }

  _trainingFeatureRewardHistory.push_back(cumulativeFeatureReward);

  /*********************************************************************
   * Computing initial retrace value for the newly added experiences
   *********************************************************************/

  // Getting position of the final experience of the episode in the replay memory
  const ssize_t endId = (ssize_t)_stateBuffer.size() - 1;

  // Getting the starting ID of the initial experience of the episode in the replay memory
  ssize_t startId = endId - episodeExperienceCount + 1;

  // Storage for the retrace value
  std::vector<float> retV(numAgents, 0.0f);

  // If it was a truncated episode, add the value function for the terminal state to retV
  if (_terminationBuffer[endId] == e_truncated)
  {
    for (size_t a = 0; a < numAgents; a++)
    {
      // Get truncated state
      auto expTruncatedStateSequence = getTruncatedStateSequence(endId, a);

      // Forward tuncated state. Take policy d if there is multiple policies, otherwise policy 0
      std::vector<policy_t> truncatedPolicy;
      if (_problem->_policiesPerEnvironment == 1)
        retV[a] = calculateStateValue(expTruncatedStateSequence);
      else
        retV[a] = calculateStateValue(expTruncatedStateSequence, a);

      // Get value of trucated state
      if (std::isfinite(retV[a]) == false)
        KORALI_LOG_ERROR("Calculated state value for truncated state returned an invalid value: %f\n", retV[a]);
    }

    // For cooporative multi-agent model truncated state-values are averaged
    if (_multiAgentRelationship == "Cooperation")
    {
      float avgRetV = std::accumulate(retV.begin(), retV.end(), 0.);
      avgRetV /= numAgents;
      retV = std::vector<float>(numAgents, avgRetV);
    }

    // The value of the truncated state equals initial retrace Value
    _truncatedStateValueBuffer[endId] = retV;
  }

  // Now going backwards, setting the retrace value of every experience
  for (ssize_t expId = endId; expId >= startId; expId--)
    for (size_t a = 0; a < numAgents; a++)
    {
      // Calculating retrace value. Importance weight is 1.0f because the policy is current.
      retV[a] = _rewardBufferContiguous[expId * numAgents + a] + _discountFactor * retV[a];
      _retraceValueBufferContiguous[expId * numAgents + a] = retV[a];
    }

  // Update reward rescaling sigma
  if (_rewardRescalingEnabled)
  {
    // get environment Id vector
    // finalize computation of standard deviation for reward rescaling
    KORALI_LOG_ERROR("IRL does not allow reward rescaling");
    _rewardRescalingSigma = std::sqrt(_rewardRescalingSumSquaredRewards / ((float)_rewardBufferContiguous.size()) + 1e-9);
  }

  // Increasing episode counters
  _sessionEpisodeCount++;

  // Increasing total experience counters
  _experienceCount += episodeExperienceCount;
  _sessionExperienceCount += episodeExperienceCount;
}

std::vector<std::pair<size_t, size_t>> Agent::generateMiniBatch()
{
  // Get number of agents
  const size_t numAgents = _problem->_agentsPerEnvironment;

  // Allocating storage for mini batch experiecne indexes
  std::vector<std::pair<size_t, size_t>> miniBatch(_miniBatchSize * numAgents);

  // Fill minibatch
  for (size_t b = 0; b < _miniBatchSize; b++)
  {
    // Producing random (uniform) number for the selection of the experience
    const float x = _uniformGenerator->getRandomNumber();

    // Selecting experience
    const size_t expId = std::floor(x * (float)(_stateBuffer.size() - 1));

    for (size_t a = 0; a < numAgents; a++)
    {
      // Setting experience
      miniBatch[b * numAgents + a].first = expId;
      miniBatch[b * numAgents + a].second = a;
    }
  }

  // clang-format off
  // Sorting minibatch: first by expId, second by agentId
  // to quickly detect duplicates when updating metadata
  std::sort(miniBatch.begin(), miniBatch.end(), [numAgents](const std::pair<size_t, size_t> &exp0, const std::pair<size_t, size_t> &exp1) -> bool {
    return exp0.first * numAgents + exp0.second < exp1.first * numAgents + exp1.second;
  });
  // clang-format on

  // Returning generated minibatch
  return miniBatch;
}

std::vector<std::vector<std::vector<float>>> Agent::getMiniBatchStateSequence(const std::vector<std::pair<size_t, size_t>> &miniBatch)
{
  // Get number of experiences in minibatch
  const size_t numExperiences = miniBatch.size();

  // Allocating state sequence vector
  std::vector<std::vector<std::vector<float>>> stateSequence(numExperiences);

#pragma omp parallel for
  for (size_t b = 0; b < numExperiences; b++)
  {
    // Getting current expId and agentId
    const size_t expId = miniBatch[b].first;
    const size_t agentId = miniBatch[b].second;

    // Getting starting expId
    const size_t startId = getTimeSequenceStartExpId(expId);

    // Calculating time sequence length
    const size_t T = expId - startId + 1;

    // Resizing state sequence vector to the correct time sequence length
    stateSequence[b].resize(T);
    for (size_t t = 0; t < T; t++)
    {
      // Now adding states
      const size_t sequenceId = startId + t;
      stateSequence[b][t].reserve(_problem->_stateVectorSize);
      stateSequence[b][t].insert(stateSequence[b][t].begin(), _stateBuffer[sequenceId][agentId].begin(), _stateBuffer[sequenceId][agentId].end());
    }
  }

  return stateSequence;
}

void Agent::updateExperienceMetadata(const std::vector<std::pair<size_t, size_t>> &miniBatch, const std::vector<policy_t> &policyData)
{
  const size_t miniBatchSize = miniBatch.size();
  const size_t numAgents = _problem->_agentsPerEnvironment;

  /* Creating a selection of unique expIds, agentIds
   * Important: this assumes the minibatch ids are sorted.
   */

  // Create Buffers
  std::vector<size_t> updateBatch;
  std::vector<std::pair<size_t, size_t>> updateMinibatch;
  std::vector<policy_t> updatePolicyData;

  // Fill updateMinibatch and updatePolicyData
  size_t b = 0;
  while (b < miniBatchSize)
  {
    // Add new unique combination
    updateBatch.push_back(b);
    updateMinibatch.push_back(miniBatch[b]);
    updatePolicyData.push_back(policyData[b]);

    size_t a = 1;

    // Iterate over experiences with same expId
    while ((miniBatch[b + a].first == miniBatch[b + a - 1].first) && (b + a < miniBatchSize))
    {
      // Add unique experiences from agents
      if (miniBatch[b + a].second != miniBatch[b + a - 1].second)
      {
        updateMinibatch.push_back(miniBatch[b + a]);
        updatePolicyData.push_back(policyData[b + a]);
      }
      a++;
    }

    // Increment batch counter by the number of same expIds
    b += a;
  }

  // Container to compute offpolicy count difference in minibatch
  std::vector<int> offPolicyCountDelta(numAgents, 0);

#pragma omp parallel for reduction(vec_int_plus \
                                   : offPolicyCountDelta)
  for (size_t i = 0; i < updateMinibatch.size(); i++)
  {
    // Get current expId and agentId
    const size_t expId = updateMinibatch[i].first;
    const size_t agentId = updateMinibatch[i].second;

    // Get and set current policy
    const auto &curPolicy = updatePolicyData[i];
    _curPolicyBuffer[expId][agentId] = curPolicy;

    // Get state value
    _stateValueBufferContiguous[expId * numAgents + agentId] = curPolicy.stateValue;
    if (std::isfinite(curPolicy.stateValue) == false)
      KORALI_LOG_ERROR("Calculated state value returned an invalid value: %f\n", curPolicy.stateValue);

    // Get action and policy for this experience
    const auto &expAction = _actionBuffer[expId][agentId];
    const auto &expPolicy = _expPolicyBuffer[expId][agentId];

    // Compute importance weight
    const float importanceWeight = calculateImportanceWeight(expAction, curPolicy, expPolicy);
    if (std::isfinite(importanceWeight) == false)
      KORALI_LOG_ERROR("Calculated value of importanceWeight returned an invalid value: %f\n", importanceWeight);

    // Set importance weight and truncated importance weight
    _importanceWeightBuffer[expId][agentId] = importanceWeight;
    _truncatedImportanceWeightBufferContiguous[expId * numAgents + agentId] = std::min(_importanceWeightTruncationLevel, importanceWeight);

    // Keep track of off-policyness (in principle only necessary for agentId==policyId)
    if (not _multiAgentCorrelation)
    {
      // Checking if experience is on policy
      const bool isOnPolicy = (importanceWeight > (1.0f / _experienceReplayOffPolicyCurrentCutoff)) && (importanceWeight < _experienceReplayOffPolicyCurrentCutoff);

      // Updating off policy count if a change is detected
      if (_isOnPolicyBuffer[expId][agentId] == true && isOnPolicy == false)
        offPolicyCountDelta[agentId]++;

      if (_isOnPolicyBuffer[expId][agentId] == false && isOnPolicy == true)
        offPolicyCountDelta[agentId]--;

      // Write to onPolicy vector
      _isOnPolicyBuffer[expId][agentId] = isOnPolicy;
    }

    // Update truncated state value
    if (_terminationBuffer[expId] == e_truncated)
    {
      // Get truncated state
      auto expTruncatedStateSequence = getTruncatedStateSequence(expId, agentId);

      // Forward tuncated state
      // TODO: other policy for exp-sharing in multi-policy case??
      float truncatedStateValue;
      if (_problem->_policiesPerEnvironment == 1)
        truncatedStateValue = calculateStateValue(expTruncatedStateSequence);
      else
        truncatedStateValue = calculateStateValue(expTruncatedStateSequence, agentId);

      // Check value of trucated state
      if (std::isfinite(truncatedStateValue) == false)
        KORALI_LOG_ERROR("Calculated state value for truncated state returned an invalid value: %f\n", truncatedStateValue);

      // Write truncated state value
      _truncatedStateValueBuffer[expId][agentId] = truncatedStateValue;
    }
  }

  /* Now taking care of advanced correlation features for MARL */

  if (_multiAgentCorrelation)
  {
#pragma omp parallel for reduction(vec_int_plus \
                                   : offPolicyCountDelta)
    for (size_t i = 0; i < updateBatch.size(); i++)
    {
      const size_t batchId = updateBatch[i];
      const size_t expId = miniBatch[batchId].first;

      // Load importance weight for expId
      const auto &importanceWeight = _importanceWeightBuffer[expId];

      // Compute product of importance weights
      float logProdImportanceWeight = 0.0f;
      for (size_t a = 0; a < numAgents; a++)
      {
        // Numerical safety [ ln(10)=2.30258509299 ]
        if (importanceWeight[a] == 0)
          logProdImportanceWeight += 2.30258509299 * std::numeric_limits<float>::min_exponent10;
        else
          logProdImportanceWeight += std::log(importanceWeight[a]);
      }

      // Compute cut-off in log-space
      const float logCutOff = (float)numAgents * std::log(_experienceReplayOffPolicyCurrentCutoff);

      // Check whether experience is onPolicy
      const bool onPolicy = (logProdImportanceWeight > (-1. * logCutOff)) && (logProdImportanceWeight < logCutOff);

      // Load isOnPolicy
      auto &isOnPolicy = _isOnPolicyBuffer[expId];

      // Write to prodImportanceWeight vector
      _productImportanceWeightBuffer[expId] = std::exp(logProdImportanceWeight);

      // Truncate product of importance weight
      const float truncatedProdImportanceWeight = std::min(_importanceWeightTruncationLevel, _productImportanceWeightBuffer[expId]);

      // Overwrite truncated importance weights
      for (size_t a = 0; a < numAgents; a++)
        _truncatedImportanceWeightBufferContiguous[expId * numAgents + a] = truncatedProdImportanceWeight;

      // Record change of on-policyness
      for (size_t d = 0; d < numAgents; d++)
      {
        if (isOnPolicy[d] == true && onPolicy == false)
          offPolicyCountDelta[d]++;

        if (isOnPolicy[d] == false && onPolicy == true)
          offPolicyCountDelta[d]--;
      }

      // Overwrite onPolicyVector
      std::fill(isOnPolicy.begin(), isOnPolicy.end(), onPolicy);
    }
  }

  // Average state values for cooperative MA
  if (_multiAgentRelationship == "Cooperation")
  {
#pragma omp parallel for
    for (size_t i = 0; i < updateBatch.size(); i++)
    {
      const size_t batchId = updateBatch[i];
      const size_t expId = miniBatch[batchId].first;

      // Average state-value
      float averageStateValue = 0.0f;
      for (size_t a = 0; a < numAgents; a++)
        averageStateValue += _stateValueBufferContiguous[expId * numAgents + a];
      averageStateValue /= numAgents;

      // Overwrite state value with average
      for (size_t a = 0; a < numAgents; a++)
        _stateValueBufferContiguous[expId * numAgents + a] = averageStateValue;

      // Same for truncated state-value
      if (_terminationBuffer[expId] == e_truncated)
      {
        // Load truncated state value
        auto &truncatedStateValue = _truncatedStateValueBuffer[expId];

        // Average truncated state value
        float averageTruncatedStateValue = std::accumulate(truncatedStateValue.begin(), truncatedStateValue.end(), 0.);
        averageTruncatedStateValue /= numAgents;

        // Overwrite truncated state value with average
        std::fill(truncatedStateValue.begin(), truncatedStateValue.end(), averageTruncatedStateValue);
      }
    }
  }

  /* Taking care of off-policy statistics */

  if (_problem->_policiesPerEnvironment == 1)
  {
    // Consider all observation for the off-policy statistics
    int sumOffPolicyCountDelta = std::accumulate(offPolicyCountDelta.begin(), offPolicyCountDelta.end(), 0.);
    offPolicyCountDelta = std::vector<int>(numAgents, sumOffPolicyCountDelta);
  }

  // Updating the off policy count and ratio
  for (size_t a = 0; a < numAgents; a++)
  {
    // Safety check for overflow
    if ((int)_experienceReplayOffPolicyCount[a] < -offPolicyCountDelta[a])
      KORALI_LOG_ERROR("Agent %ld: offPolicyCountDelta=%d bigger than _experienceReplayOffPolicyCount=%ld.\n", a, offPolicyCountDelta[a], _experienceReplayOffPolicyCount[a]);

    // Update off policy count
    _experienceReplayOffPolicyCount[a] += offPolicyCountDelta[a];
    _experienceReplayOffPolicyRatio[a] = (float)_experienceReplayOffPolicyCount[a] / (float)_isOnPolicyBuffer.size();

    // Normalize off policy Ratio
    if (_problem->_policiesPerEnvironment == 1)
      _experienceReplayOffPolicyRatio[a] /= (float)numAgents;
  }

  /* Update Retrace value */

  // Now filtering experiences from the same episode
  std::vector<size_t> retraceMiniBatch;

  // Adding last experience from the sorted minibatch
  retraceMiniBatch.push_back(miniBatch[updateBatch.back()].first);

  // Adding experiences so long as they do not repeat episodes
  for (ssize_t i = updateBatch.size() - 2; i >= 0; i--)
  {
    const size_t currExpId = miniBatch[updateBatch[i]].first;
    const size_t nextExpId = miniBatch[updateBatch[i + 1]].first;
    const size_t curEpisode = _episodeIdBuffer[currExpId];
    const size_t nextEpisode = _episodeIdBuffer[nextExpId];
    if (curEpisode != nextEpisode) retraceMiniBatch.push_back(currExpId);
  }

  size_t t = 0;
  std::vector<ssize_t> featureMiniBatch(_rewardFunctionBatchSize);
  std::vector<std::vector<std::vector<float>>>
    featureBatch(_rewardFunctionBatchSize, std::vector<std::vector<float>>(_problem->_agentsPerEnvironment, std::vector<float>(_problem->_featureVectorSize)));

  // Update rewards backward
  for (size_t a = 0; a < numAgents; a++)
  {
  for (size_t i = 0; i < retraceMiniBatch.size(); i++)
  {
    // Finding the earliest experience corresponding to the same episode as this experience
    const ssize_t endId = retraceMiniBatch[i];

    // If the starting experience has already been discarded, take the earliest one that still remains
    const ssize_t startId = (ssize_t)_episodePosBuffer[endId] < endId ? endId - (ssize_t)_episodePosBuffer[endId] : 0;

    // Now iterating backwards to find the beginning
    for (ssize_t curId = endId; curId >= startId; curId--)
    {
      // Update reward if old
      if (_rewardUpdateBuffer[curId] < _rewardUpdateCount)
      {
        // Add features and experience id
        featureMiniBatch[t] = curId;
        featureBatch[t++] = {{_featureBuffer[curId][a]}};
        if (t == _rewardFunctionBatchSize)
        {
          // Feature batch is full, forward features
          auto featureReward = calculateReward(featureBatch);

          // Update rewards
          for (size_t b = 0; b < _rewardFunctionBatchSize; ++b)
          {
            const size_t expId = featureMiniBatch[b];
            _rewardBufferContiguous[expId * numAgents + a] = featureReward[b];
            _rewardUpdateBuffer[expId] = _rewardUpdateCount;
          }

          // Reset count
          t = 0;
        }
      }
    }
  }

  // Update the rest
  if (t > 0)
  {
    auto featureReward = calculateReward(featureBatch);
    for (size_t b = 0; b < t; ++b)
    {
      const ssize_t expId = featureMiniBatch[b];
      for (size_t a = 0; a < numAgents; a++)
        _rewardBufferContiguous[expId * numAgents + a] = featureReward[b];
      _rewardUpdateBuffer[expId] = _rewardUpdateCount;
    }
  }
  }

// Calculating retrace value for the oldest experiences of unique episodes
#pragma omp parallel for schedule(guided, 1)
  for (size_t i = 0; i < retraceMiniBatch.size(); i++)
  {
    // Determine start of the episode
    const ssize_t endId = retraceMiniBatch[i];
    ssize_t startId = endId - _episodePosBuffer[endId];

    // If start of episode has been discarded, take earliest one
    if (startId < 0) startId = 0;

    // Storage for the retrace value
    std::vector<float> retV(numAgents, 0.0f);

    // For truncated episode, set truncated state value function
    if (_terminationBuffer[endId] == e_truncated)
      retV = _truncatedStateValueBuffer[endId];

    // If non-terminal state, set next retrace value
    if (_terminationBuffer[endId] == e_nonTerminal)
      for (size_t a = 0; a < numAgents; a++)
        retV[a] = _retraceValueBufferContiguous[(endId + 1) * numAgents + a];

    // Now iterating backwards and compute retrace value
    for (ssize_t curId = endId; curId >= startId; curId--)
    {
      // Getting current reward, action, and state
      for (size_t a = 0; a < numAgents; a++)
      {
        // Load truncated importance weight
        const float truncatedImportanceWeight = _truncatedImportanceWeightBufferContiguous[curId * numAgents + a];

        // Load state value
        const float stateValue = _stateValueBufferContiguous[curId * numAgents + a];

        // Getting current reward, action, and state
        const float curReward = _rewardBufferContiguous[curId * numAgents + a];

        // Apply recursion
        retV[a] = stateValue + truncatedImportanceWeight * (curReward + _discountFactor * retV[a] - stateValue);

        // Store retrace value
        _retraceValueBufferContiguous[curId * numAgents + a] = retV[a];
      }
    }
  }
}

size_t Agent::getTimeSequenceStartExpId(size_t expId)
{
  const size_t episodePos = _episodePosBuffer[expId];

  // Determine actual length of time sequence
  const size_t lookBack = std::min(_timeSequenceLength - 1, episodePos);

  if (lookBack > expId)
    // Return start of buffer if expId is part of a cut episode
    return 0;
  else
    // Return time sequence start
    return expId - lookBack;
}

void Agent::resetTimeSequence()
{
  for (size_t a = 0; a < _problem->_agentsPerEnvironment; ++a)
    _stateTimeSequence[a].clear();
}

std::vector<std::vector<std::vector<float>>> Agent::getMiniBatchFeatureSequence(const std::vector<size_t> &miniBatch)
{
  // Allocating feature sequence vector
  std::vector<std::vector<std::vector<float>>> featureSequence(_rewardFunctionBatchSize, std::vector<std::vector<float>>(_problem->_agentsPerEnvironment, std::vector<float>(_problem->_featureVectorSize)));

#pragma omp parallel for
  for (size_t b = 0; b < miniBatch.size(); b++)
  {
    // Getting current expId
    const size_t expId = miniBatch[b];

    for (size_t a = 0; a < _problem->_agentsPerEnvironment; ++a)
    {
      featureSequence[b][a] = _featureBuffer[expId][a];
    }
  }

  return featureSequence;
}

std::vector<std::vector<float>> Agent::getTruncatedStateSequence(size_t expId, size_t agentId)
{
  // Getting starting expId
  size_t startId = getTimeSequenceStartExpId(expId);

  // Creating storage for the time sequence
  std::vector<std::vector<float>> timeSequence;

  // Now adding states, except for the initial one
  for (size_t e = startId + 1; e <= expId; e++)
    timeSequence.push_back(_stateBuffer[e][agentId]);

  // Lastly, adding truncated state
  timeSequence.push_back(_truncatedStateBuffer[expId][agentId]);

  return timeSequence;
}

void Agent::finalize()
{
  if (_mode != "Training") return;

  if (_experienceReplaySerialize == true)
    if (_k->_fileOutputEnabled)
      serializeExperienceReplay();

  _k->_logger->logInfo("Normal", "Waiting for pending agents to finish...\n");

  // Waiting for pending agents to finish
  bool agentsRemain = true;
  do
  {
    agentsRemain = false;
    for (size_t workerId = 0; workerId < _concurrentWorkers; workerId++)
      if (_isWorkerRunning[workerId] == true)
      {
        attendWorker(workerId);
        agentsRemain = true;
      }

    if (agentsRemain) KORALI_LISTEN(_workers);
  } while (agentsRemain == true);
}

void Agent::serializeExperienceReplay()
{
  _k->_logger->logInfo("Detailed", "Serializing Training State...\n");
  auto beginTime = std::chrono::steady_clock::now(); // Profiling

  // Creating JSON storage variable
  knlohmann::json stateJson;

  // Get number of agents
  const size_t numAgents = _problem->_agentsPerEnvironment;

  // Serializing agent's database into the JSON storage
  for (size_t i = 0; i < _stateBuffer.size(); i++)
  {
    stateJson["Experience Replay"][i]["Episode Id"] = _episodeIdBuffer[i];
    stateJson["Experience Replay"][i]["Episode Pos"] = _episodePosBuffer[i];
    stateJson["Experience Replay"][i]["State"] = _stateBuffer[i];
    stateJson["Experience Replay"][i]["Action"] = _actionBuffer[i];
    stateJson["Experience Replay"][i]["Importance Weight"] = _importanceWeightBuffer[i];
    stateJson["Experience Replay"][i]["Product Importance Weight"] = _productImportanceWeightBuffer[i];
    stateJson["Experience Replay"][i]["Is On Policy"] = _isOnPolicyBuffer[i];
    stateJson["Experience Replay"][i]["Truncated State"] = _truncatedStateBuffer[i];
    stateJson["Experience Replay"][i]["Truncated State Value"] = _truncatedStateValueBuffer[i];
    stateJson["Experience Replay"][i]["Termination"] = _terminationBuffer[i];

    stateJson["Experience Replay"][i]["Feature"] = _featureBuffer[i];
    stateJson["Experience Replay"][i]["Reward Update"] = _rewardUpdateBuffer[i];
    stateJson["Experience Replay"][i]["Policy"] = _policyBuffer[i];

    for (size_t a = 0; a < numAgents; a++)
    {
      stateJson["Experience Replay"][i]["Reward"][a] = _rewardBufferContiguous[i * numAgents + a];
      stateJson["Experience Replay"][i]["State Value"][a] = _stateValueBufferContiguous[i * numAgents + a];
      stateJson["Experience Replay"][i]["Retrace Value"][a] = _retraceValueBufferContiguous[i * numAgents + a];
      stateJson["Experience Replay"][i]["Truncated Importance Weight"][a] = _truncatedImportanceWeightBufferContiguous[i * numAgents + a];
    }

    std::vector<float> expStateValue(numAgents, 0.0f);
    std::vector<std::vector<float>> expDistributionParameter(numAgents, std::vector<float>(_expPolicyBuffer[0][0].distributionParameters.size()));
    std::vector<size_t> expActionIdx(numAgents, 0);
    std::vector<std::vector<float>> expUnboundedAct(numAgents, std::vector<float>(_expPolicyBuffer[0][0].unboundedAction.size()));
    std::vector<std::vector<float>> expActProb(numAgents, std::vector<float>(_expPolicyBuffer[0][0].actionProbabilities.size()));
    std::vector<std::vector<size_t>> expAvailAct(numAgents, std::vector<size_t>(_expPolicyBuffer[0][0].availableActions.size()));

    std::vector<float> curStateValue(numAgents, 0.0f);
    std::vector<std::vector<float>> curDistributionParameter(numAgents, std::vector<float>(_curPolicyBuffer[0][0].distributionParameters.size()));
    std::vector<size_t> curActionIdx(numAgents, 0);
    std::vector<std::vector<float>> curUnboundedAct(numAgents, std::vector<float>(_curPolicyBuffer[0][0].unboundedAction.size()));
    std::vector<std::vector<float>> curActProb(numAgents, std::vector<float>(_curPolicyBuffer[0][0].actionProbabilities.size()));
    std::vector<std::vector<size_t>> curAvailAct(numAgents, std::vector<size_t>(_curPolicyBuffer[0][0].availableActions.size()));

    for (size_t a = 0; a < numAgents; a++)
    {
      expStateValue[a] = _expPolicyBuffer[i][a].stateValue;
      expDistributionParameter[a] = _expPolicyBuffer[i][a].distributionParameters;
      expActionIdx[a] = _expPolicyBuffer[i][a].actionIndex;
      expUnboundedAct[a] = _expPolicyBuffer[i][a].unboundedAction;
      expActProb[a] = _expPolicyBuffer[i][a].actionProbabilities;
      expAvailAct[a] = _expPolicyBuffer[i][a].availableActions;

      curStateValue[a] = _curPolicyBuffer[i][a].stateValue;
      curDistributionParameter[a] = _curPolicyBuffer[i][a].distributionParameters;
      curActionIdx[a] = _curPolicyBuffer[i][a].actionIndex;
      curUnboundedAct[a] = _curPolicyBuffer[i][a].unboundedAction;
      curActProb[a] = _curPolicyBuffer[i][a].actionProbabilities;
      curAvailAct[a] = _curPolicyBuffer[i][a].availableActions;
    }
    stateJson["Experience Replay"][i]["Experience Policy"]["State Value"] = expStateValue;
    stateJson["Experience Replay"][i]["Experience Policy"]["Distribution Parameters"] = expDistributionParameter;
    stateJson["Experience Replay"][i]["Experience Policy"]["Action Index"] = expActionIdx;
    stateJson["Experience Replay"][i]["Experience Policy"]["Unbounded Action"] = expUnboundedAct;
    stateJson["Experience Replay"][i]["Experience Policy"]["Action Probabilities"] = expActProb;
    stateJson["Experience Replay"][i]["Experience Policy"]["Available Actions"] = expAvailAct;

    stateJson["Experience Replay"][i]["Current Policy"]["State Value"] = curStateValue;
    stateJson["Experience Replay"][i]["Current Policy"]["Distribution Parameters"] = curDistributionParameter;
    stateJson["Experience Replay"][i]["Current Policy"]["Action Index"] = curActionIdx;
    stateJson["Experience Replay"][i]["Current Policy"]["Unbounded Action"] = curUnboundedAct;
    stateJson["Experience Replay"][i]["Current Policy"]["Action Probabilities"] = curActProb;
    stateJson["Experience Replay"][i]["Current Policy"]["Available Actions"] = curAvailAct;
  }

  // Serialize the optimizer
  for (size_t p = 0; p < _problem->_policiesPerEnvironment; p++)
    _criticPolicyLearner[p]->_optimizer->getConfiguration(stateJson["Optimizer"][p]);

  // If results directory doesn't exist, create it
  if (!dirExists(_k->_fileOutputPath)) mkdir(_k->_fileOutputPath);

  // Resolving file path
  std::string statePath = _k->_fileOutputPath + "/state.json";

  // Storing database to file
  _k->_logger->logInfo("Detailed", "Saving json..\n");
  if (saveJsonToFile(statePath.c_str(), stateJson) != 0)
    KORALI_LOG_ERROR("Could not serialize training state into file %s\n", statePath.c_str());
  _k->_logger->logInfo("Detailed", "Agent's Training State serialized\n");

  auto endTime = std::chrono::steady_clock::now();                                                                   // Profiling
  _sessionSerializationTime += std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - beginTime).count();    // Profiling
  _generationSerializationTime += std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - beginTime).count(); // Profiling
}

void Agent::deserializeExperienceReplay()
{
  auto beginTime = std::chrono::steady_clock::now(); // Profiling

  // Creating JSON storage variable
  knlohmann::json stateJson;

  // Resolving file path
  std::string statePath = _k->_fileOutputPath + "/state.json";

  // Get number of agents
  const size_t numAgents = _problem->_agentsPerEnvironment;

  // Loading database from file
  _k->_logger->logInfo("Normal", "Loading previous run training state from file %s...\n", statePath.c_str());
  if (loadJsonFromFile(stateJson, statePath.c_str()) == false)
    KORALI_LOG_ERROR("Trying to resume training or test policy but could not find or deserialize agent's state from file %s...\n", statePath.c_str());

  // Clearing existing database
  _stateBuffer.clear();
  _actionBuffer.clear();
  _rewardUpdateBuffer.clear();
  _featureBuffer.clear();
  _policyBuffer.clear();

  _retraceValueBufferContiguous.clear();
  _rewardBufferContiguous.clear();
  _stateValueBufferContiguous.clear();
  _importanceWeightBuffer.clear();
  _truncatedImportanceWeightBufferContiguous.clear();
  _truncatedStateValueBuffer.clear();
  _productImportanceWeightBuffer.clear();
  _truncatedStateBuffer.clear();
  _terminationBuffer.clear();
  _expPolicyBuffer.clear();
  _curPolicyBuffer.clear();
  _isOnPolicyBuffer.clear();
  _episodePosBuffer.clear();
  _episodeIdBuffer.clear();

  // Deserializing database from JSON to the agent's state
  for (size_t i = 0; i < stateJson["Experience Replay"].size(); i++)
  {
    _episodeIdBuffer.add(stateJson["Experience Replay"][i]["Episode Id"].get<size_t>());
    _episodePosBuffer.add(stateJson["Experience Replay"][i]["Episode Pos"].get<size_t>());
    _rewardUpdateBuffer.add(stateJson["Experience Replay"][i]["Reward Update"].get<float>());
    _featureBuffer.add(stateJson["Experience Replay"][i]["Feature"].get<std::vector<std::vector<float>>>());
    _policyBuffer.add(stateJson["Experience Replay"][i]["Policy"].get<std::vector<float>>());

    _stateBuffer.add(stateJson["Experience Replay"][i]["State"].get<std::vector<std::vector<float>>>());
    _actionBuffer.add(stateJson["Experience Replay"][i]["Action"].get<std::vector<std::vector<float>>>());
    _importanceWeightBuffer.add(stateJson["Experience Replay"][i]["Importance Weight"].get<std::vector<float>>());

    _productImportanceWeightBuffer.add(stateJson["Experience Replay"][i]["Product Importance Weight"].get<float>());
    _isOnPolicyBuffer.add(stateJson["Experience Replay"][i]["Is On Policy"].get<std::vector<char>>());
    _truncatedStateBuffer.add(stateJson["Experience Replay"][i]["Truncated State"].get<std::vector<std::vector<float>>>());
    _truncatedStateValueBuffer.add(stateJson["Experience Replay"][i]["Truncated State Value"].get<std::vector<float>>());
    _terminationBuffer.add(stateJson["Experience Replay"][i]["Termination"].get<termination_t>());

    for (size_t a = 0; a < numAgents; a++)
    {
      _rewardBufferContiguous.add(stateJson["Experience Replay"][i]["Reward"][a].get<float>());
      _stateValueBufferContiguous.add(stateJson["Experience Replay"][i]["State Value"][a].get<float>());
      _retraceValueBufferContiguous.add(stateJson["Experience Replay"][i]["Retrace Value"][a].get<float>());
      _truncatedImportanceWeightBufferContiguous.add(stateJson["Experience Replay"][i]["Truncated Importance Weight"][a].get<float>());
    }

    std::vector<policy_t> expPolicy(numAgents);
    std::vector<policy_t> curPolicy(numAgents);
    for (size_t a = 0; a < numAgents; a++)
    {
      expPolicy[a].stateValue = stateJson["Experience Replay"][i]["Experience Policy"]["State Value"][a].get<float>();
      expPolicy[a].distributionParameters = stateJson["Experience Replay"][i]["Experience Policy"]["Distribution Parameters"][a].get<std::vector<float>>();
      expPolicy[a].unboundedAction = stateJson["Experience Replay"][i]["Experience Policy"]["Unbounded Action"][a].get<std::vector<float>>();
      expPolicy[a].actionIndex = stateJson["Experience Replay"][i]["Experience Policy"]["Action Index"][a].get<size_t>();
      expPolicy[a].actionProbabilities = stateJson["Experience Replay"][i]["Experience Policy"]["Action Probabilities"][a].get<std::vector<float>>();
      expPolicy[a].availableActions = stateJson["Experience Replay"][i]["Experience Policy"]["Available Actions"][a].get<std::vector<size_t>>();

      curPolicy[a].stateValue = stateJson["Experience Replay"][i]["Current Policy"]["State Value"][a].get<float>();
      curPolicy[a].distributionParameters = stateJson["Experience Replay"][i]["Current Policy"]["Distribution Parameters"][a].get<std::vector<float>>();
      curPolicy[a].actionIndex = stateJson["Experience Replay"][i]["Current Policy"]["Action Index"][a].get<size_t>();
      curPolicy[a].unboundedAction = stateJson["Experience Replay"][i]["Current Policy"]["Unbounded Action"][a].get<std::vector<float>>();
      curPolicy[a].actionProbabilities = stateJson["Experience Replay"][i]["Current Policy"]["Action Probabilities"][a].get<std::vector<float>>();
      curPolicy[a].availableActions = stateJson["Experience Replay"][i]["Current Policy"]["Available Actions"][a].get<std::vector<size_t>>();
    }
    _expPolicyBuffer.add(expPolicy);
    _curPolicyBuffer.add(curPolicy);
  }

  // Deserialize the optimizer
  for (size_t p = 0; p < _problem->_policiesPerEnvironment; p++)
    _criticPolicyLearner[p]->_optimizer->setConfiguration(stateJson["Optimizer"][p]);

  auto endTime = std::chrono::steady_clock::now();                                                                         // Profiling
  double deserializationTime = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - beginTime).count() / 1.0e+9; // Profiling
  _k->_logger->logInfo("Normal", "Took %fs to deserialize training state.\n", deserializationTime);
}

void Agent::printGenerationAfter()
{
  if (_mode == "Training")
  {
    _k->_logger->logInfo("Normal", "Experience Replay Statistics:\n");
    _k->_logger->logInfo("Normal", " + Experience Memory Size:      %lu/%lu\n", _stateBuffer.size(), _experienceReplayMaximumSize);
    if (_maxEpisodes > 0)
      _k->_logger->logInfo("Normal", " + Total Episodes Count:        %lu/%lu\n", _currentEpisode, _maxEpisodes);
    else
      _k->_logger->logInfo("Normal", " + Total Episodes Count:        %lu\n", _currentEpisode);

    if (_maxExperiences > 0)
      _k->_logger->logInfo("Normal", " + Total Experience Count:      %lu/%lu\n", _experienceCount, _maxExperiences);
    else
      _k->_logger->logInfo("Normal", " + Total Experience Count:      %lu\n", _experienceCount);

    _k->_logger->logInfo("Normal", "Training Statistics:\n");
    if (_maxPolicyUpdates > 0)
      _k->_logger->logInfo("Normal", " + Policy Update Count:         %lu/%lu\n", _policyUpdateCount, _maxPolicyUpdates);
    else
      _k->_logger->logInfo("Normal", " + Policy Update Count:         %lu\n", _policyUpdateCount);

    size_t numPolicies = _problem->_policiesPerEnvironment;
    for (size_t a = 0; a < _problem->_agentsPerEnvironment; a++)
    {
      _k->_logger->logInfo("Normal", "Off-Policy Statistics for policy %lu: \n", a);
      _k->_logger->logInfo("Normal", " + Count (Ratio/Target):        %lu/%lu (%.3f/%.3f)\n", numPolicies > 1 ? _experienceReplayOffPolicyCount[a] : _experienceReplayOffPolicyCount[a] / _problem->_agentsPerEnvironment, _stateBuffer.size(), _experienceReplayOffPolicyRatio[a], _experienceReplayOffPolicyTarget);
      _k->_logger->logInfo("Normal", " + Importance Weight Cutoff:    [%.3f, %.3f]\n", 1.0f / _experienceReplayOffPolicyCurrentCutoff, _experienceReplayOffPolicyCurrentCutoff);
      _k->_logger->logInfo("Normal", " + REFER Beta Factor:           %f\n", _experienceReplayOffPolicyREFERCurrentBeta[a]);
      _k->_logger->logInfo("Normal", " + Latest Reward for agent %lu:               %f\n", a, _trainingLastReward[a]);
      _k->_logger->logInfo("Normal", " + %lu-Episode Average Reward for agent %lu:  %f\n", _trainingAverageDepth, a, _trainingAverageReward[a]);
      _k->_logger->logInfo("Normal", " + Best Reward for agent %lu:                 %f (%lu)\n", a, _trainingBestReward[a], _trainingBestEpisodeId[a]);
    }

    if (_rewardRescalingEnabled)
      _k->_logger->logInfo("Normal", " + Reward Rescaling: N(0.0, %.3e)\n", _rewardRescalingSigma);

    if (_problem->_testingFrequency > 0)
    {
      _k->_logger->logInfo("Normal", "Testing Statistics:\n");
      _k->_logger->logInfo("Normal", " + Best Average Reward: %f (%lu)\n", _testingBestAverageReward, _testingBestEpisodeId);
      _k->_logger->logInfo("Normal", " + Latest Average (Worst / Best) Reward: %f (%f / %f)\n", _testingAverageReward, _testingWorstReward, _testingBestReward);
    }

    if (_policyUpdateCount != 0)
    {
      printInformation();
      _k->_logger->logInfo("Normal", " + Current Learning Rate:           %.3e\n", _currentLearningRate);
    }

    if (_stateRescalingEnabled)
      _k->_logger->logInfo("Normal", " + Using State Rescaling\n");

    if (_featureRescalingEnabled)
      _k->_logger->logInfo("Normal", " + Using Feature Rescaling\n");

    _k->_logger->logInfo("Normal", "Background Trajectory Count:      %lu/%lu\n", std::min(_backgroundTrajectoryCount, _backgroundSampleSize), _backgroundSampleSize);
    _k->_logger->logInfo("Normal", "Total Number Background Samples:  %zu\n", _backgroundTrajectoryCount);
    _k->_logger->logInfo("Normal", "Log Partition Function:           %f (%f)\n", _logPartitionFunction, _logSdevPartitionFunction);

    _k->_logger->logInfo("Detailed", "Profiling Information:                    [Generation] - [Session]\n");
    _k->_logger->logInfo("Detailed", " + Experience Serialization Time:         [%5.3fs] - [%3.3fs]\n", _generationSerializationTime / 1.0e+9, _sessionSerializationTime / 1.0e+9);
    _k->_logger->logInfo("Detailed", " + Worker Attending Time:                 [%5.3fs] - [%3.3fs]\n", _generationWorkerAttendingTime / 1.0e+9, _sessionWorkerAttendingTime / 1.0e+9);
    _k->_logger->logInfo("Detailed", " + Avg Worker Computation Time:           [%5.3fs] - [%3.3fs]\n", _generationWorkerComputationTime / 1.0e+9, _sessionWorkerComputationTime / 1.0e+9);
    _k->_logger->logInfo("Detailed", " + Avg Worker Communication/Wait Time:    [%5.3fs] - [%3.3fs]\n", _generationWorkerCommunicationTime / 1.0e+9, _sessionWorkerCommunicationTime / 1.0e+9);
    _k->_logger->logInfo("Detailed", " + Avg Policy Evaluation Time:            [%5.3fs] - [%3.3fs]\n", _generationPolicyEvaluationTime / 1.0e+9, _sessionPolicyEvaluationTime / 1.0e+9);
    _k->_logger->logInfo("Detailed", " + Policy Update Time:                    [%5.3fs] - [%3.3fs]\n", _generationPolicyUpdateTime / 1.0e+9, _sessionPolicyUpdateTime / 1.0e+9);
    _k->_logger->logInfo("Detailed", " + Reward Update Time:                    [%5.3fs] - [%3.3fs]\n", _generationRewardUpdateTime / 1.0e+9, _sessionRewardUpdateTime / 1.0e+9);
    _k->_logger->logInfo("Detailed", " + Trajectory Probability Update Time:    [%5.3fs] - [%3.3fs]\n", _generationTrajectoryLogProbabilityUpdateTime / 1.0e+9, _sessionTrajectoryLogProbabilityUpdateTime / 1.0e+9);
    _k->_logger->logInfo("Detailed", " + [I/O] Result File Saving Time:         [%5.3fs]\n", _k->_resultSavingTime / 1.0e+9);
    _k->_logger->logInfo("Detailed", " + Running Time:                          [%5.3fs] - [%3.3fs]\n", _generationRunningTime / 1.0e+9, _sessionRunningTime / 1.0e+9);
  }

  if (_mode == "Testing")
  {
    _k->_logger->logInfo("Normal", "Testing Results:\n");
    for (size_t sampleId = 0; sampleId < _testingSampleIds.size(); sampleId++)
    {
      _k->_logger->logInfo("Normal", " + Sample %lu:\n", _testingSampleIds[sampleId]);
      _k->_logger->logInfo("Normal", "   + (Average) Cumulative Reward:           %f\n", _testingReward[sampleId]);
    }
  }
}

void Agent::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Policy", "Parameter Count"))
 {
 try { _policyParameterCount = js["Policy"]["Parameter Count"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Policy']['Parameter Count']\n%s", e.what()); } 
   eraseValue(js, "Policy", "Parameter Count");
 }

 if (isDefined(js, "Action Lower Bounds"))
 {
 try { _actionLowerBounds = js["Action Lower Bounds"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Action Lower Bounds']\n%s", e.what()); } 
   eraseValue(js, "Action Lower Bounds");
 }

 if (isDefined(js, "Action Upper Bounds"))
 {
 try { _actionUpperBounds = js["Action Upper Bounds"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Action Upper Bounds']\n%s", e.what()); } 
   eraseValue(js, "Action Upper Bounds");
 }

 if (isDefined(js, "Current Episode"))
 {
 try { _currentEpisode = js["Current Episode"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Current Episode']\n%s", e.what()); } 
   eraseValue(js, "Current Episode");
 }

 if (isDefined(js, "Training", "Reward History"))
 {
 try { _trainingRewardHistory = js["Training"]["Reward History"].get<std::vector<std::vector<float>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Training']['Reward History']\n%s", e.what()); } 
   eraseValue(js, "Training", "Reward History");
 }

 if (isDefined(js, "Training", "Feature Reward History"))
 {
 try { _trainingFeatureRewardHistory = js["Training"]["Feature Reward History"].get<std::vector<std::vector<float>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Training']['Feature Reward History']\n%s", e.what()); } 
   eraseValue(js, "Training", "Feature Reward History");
 }

 if (isDefined(js, "Training", "Experience History"))
 {
 try { _trainingExperienceHistory = js["Training"]["Experience History"].get<std::vector<size_t>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Training']['Experience History']\n%s", e.what()); } 
   eraseValue(js, "Training", "Experience History");
 }

 if (isDefined(js, "Testing", "Average Reward History"))
 {
 try { _testingAverageRewardHistory = js["Testing"]["Average Reward History"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Testing']['Average Reward History']\n%s", e.what()); } 
   eraseValue(js, "Testing", "Average Reward History");
 }

 if (isDefined(js, "Training", "Average Reward"))
 {
 try { _trainingAverageReward = js["Training"]["Average Reward"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Training']['Average Reward']\n%s", e.what()); } 
   eraseValue(js, "Training", "Average Reward");
 }

 if (isDefined(js, "Training", "Average Feature Reward"))
 {
 try { _trainingAverageFeatureReward = js["Training"]["Average Feature Reward"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Training']['Average Feature Reward']\n%s", e.what()); } 
   eraseValue(js, "Training", "Average Feature Reward");
 }

 if (isDefined(js, "Training", "Last Reward"))
 {
 try { _trainingLastReward = js["Training"]["Last Reward"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Training']['Last Reward']\n%s", e.what()); } 
   eraseValue(js, "Training", "Last Reward");
 }

 if (isDefined(js, "Training", "Best Reward"))
 {
 try { _trainingBestReward = js["Training"]["Best Reward"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Training']['Best Reward']\n%s", e.what()); } 
   eraseValue(js, "Training", "Best Reward");
 }

 if (isDefined(js, "Training", "Best Episode Id"))
 {
 try { _trainingBestEpisodeId = js["Training"]["Best Episode Id"].get<std::vector<size_t>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Training']['Best Episode Id']\n%s", e.what()); } 
   eraseValue(js, "Training", "Best Episode Id");
 }

 if (isDefined(js, "Training", "Current Policies"))
 {
 _trainingCurrentPolicies = js["Training"]["Current Policies"].get<knlohmann::json>();

   eraseValue(js, "Training", "Current Policies");
 }

 if (isDefined(js, "Training", "Best Policies"))
 {
 _trainingBestPolicies = js["Training"]["Best Policies"].get<knlohmann::json>();

   eraseValue(js, "Training", "Best Policies");
 }

 if (isDefined(js, "Testing", "Reward"))
 {
 try { _testingReward = js["Testing"]["Reward"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Testing']['Reward']\n%s", e.what()); } 
   eraseValue(js, "Testing", "Reward");
 }

 if (isDefined(js, "Testing", "Best Reward"))
 {
 try { _testingBestReward = js["Testing"]["Best Reward"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Testing']['Best Reward']\n%s", e.what()); } 
   eraseValue(js, "Testing", "Best Reward");
 }

 if (isDefined(js, "Testing", "Worst Reward"))
 {
 try { _testingWorstReward = js["Testing"]["Worst Reward"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Testing']['Worst Reward']\n%s", e.what()); } 
   eraseValue(js, "Testing", "Worst Reward");
 }

 if (isDefined(js, "Testing", "Best Episode Id"))
 {
 try { _testingBestEpisodeId = js["Testing"]["Best Episode Id"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Testing']['Best Episode Id']\n%s", e.what()); } 
   eraseValue(js, "Testing", "Best Episode Id");
 }

 if (isDefined(js, "Testing", "Candidate Count"))
 {
 try { _testingCandidateCount = js["Testing"]["Candidate Count"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Testing']['Candidate Count']\n%s", e.what()); } 
   eraseValue(js, "Testing", "Candidate Count");
 }

 if (isDefined(js, "Testing", "Average Reward"))
 {
 try { _testingAverageReward = js["Testing"]["Average Reward"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Testing']['Average Reward']\n%s", e.what()); } 
   eraseValue(js, "Testing", "Average Reward");
 }

 if (isDefined(js, "Testing", "Best Average Reward"))
 {
 try { _testingBestAverageReward = js["Testing"]["Best Average Reward"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Testing']['Best Average Reward']\n%s", e.what()); } 
   eraseValue(js, "Testing", "Best Average Reward");
 }

 if (isDefined(js, "Testing", "Best Policies"))
 {
 _testingBestPolicies = js["Testing"]["Best Policies"].get<knlohmann::json>();

   eraseValue(js, "Testing", "Best Policies");
 }

 if (isDefined(js, "Experience Replay", "Off Policy", "Count"))
 {
 try { _experienceReplayOffPolicyCount = js["Experience Replay"]["Off Policy"]["Count"].get<std::vector<size_t>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Experience Replay']['Off Policy']['Count']\n%s", e.what()); } 
   eraseValue(js, "Experience Replay", "Off Policy", "Count");
 }

 if (isDefined(js, "Experience Replay", "Off Policy", "Ratio"))
 {
 try { _experienceReplayOffPolicyRatio = js["Experience Replay"]["Off Policy"]["Ratio"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Experience Replay']['Off Policy']['Ratio']\n%s", e.what()); } 
   eraseValue(js, "Experience Replay", "Off Policy", "Ratio");
 }

 if (isDefined(js, "Experience Replay", "Off Policy", "History"))
 {
 try { _experienceReplayOffPolicyHistory = js["Experience Replay"]["Off Policy"]["History"].get<std::vector<std::vector<float>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Experience Replay']['Off Policy']['History']\n%s", e.what()); } 
   eraseValue(js, "Experience Replay", "Off Policy", "History");
 }

 if (isDefined(js, "Experience Replay", "Off Policy", "Current Cutoff"))
 {
 try { _experienceReplayOffPolicyCurrentCutoff = js["Experience Replay"]["Off Policy"]["Current Cutoff"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Experience Replay']['Off Policy']['Current Cutoff']\n%s", e.what()); } 
   eraseValue(js, "Experience Replay", "Off Policy", "Current Cutoff");
 }

 if (isDefined(js, "Experience Replay", "Off Policy", "REFER Current Beta"))
 {
 try { _experienceReplayOffPolicyREFERCurrentBeta = js["Experience Replay"]["Off Policy"]["REFER Current Beta"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Experience Replay']['Off Policy']['REFER Current Beta']\n%s", e.what()); } 
   eraseValue(js, "Experience Replay", "Off Policy", "REFER Current Beta");
 }

 if (isDefined(js, "Current Learning Rate"))
 {
 try { _currentLearningRate = js["Current Learning Rate"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Current Learning Rate']\n%s", e.what()); } 
   eraseValue(js, "Current Learning Rate");
 }

 if (isDefined(js, "Policy Update Count"))
 {
 try { _policyUpdateCount = js["Policy Update Count"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Policy Update Count']\n%s", e.what()); } 
   eraseValue(js, "Policy Update Count");
 }

 if (isDefined(js, "Uniform Generator"))
 {
 _uniformGenerator = dynamic_cast<korali::distribution::univariate::Uniform*>(korali::Module::getModule(js["Uniform Generator"], _k));
 _uniformGenerator->applyVariableDefaults();
 _uniformGenerator->applyModuleDefaults(js["Uniform Generator"]);
 _uniformGenerator->setConfiguration(js["Uniform Generator"]);
   eraseValue(js, "Uniform Generator");
 }

 if (isDefined(js, "Experience Count"))
 {
 try { _experienceCount = js["Experience Count"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Experience Count']\n%s", e.what()); } 
   eraseValue(js, "Experience Count");
 }

 if (isDefined(js, "Reward", "Rescaling", "Sigma"))
 {
 try { _rewardRescalingSigma = js["Reward"]["Rescaling"]["Sigma"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Reward']['Rescaling']['Sigma']\n%s", e.what()); } 
   eraseValue(js, "Reward", "Rescaling", "Sigma");
 }

 if (isDefined(js, "Reward", "Rescaling", "Sum Squared Rewards"))
 {
 try { _rewardRescalingSumSquaredRewards = js["Reward"]["Rescaling"]["Sum Squared Rewards"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Reward']['Rescaling']['Sum Squared Rewards']\n%s", e.what()); } 
   eraseValue(js, "Reward", "Rescaling", "Sum Squared Rewards");
 }

 if (isDefined(js, "Log Partition Function"))
 {
 try { _logPartitionFunction = js["Log Partition Function"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Log Partition Function']\n%s", e.what()); } 
   eraseValue(js, "Log Partition Function");
 }

 if (isDefined(js, "Log Sdev Partition Function"))
 {
 try { _logSdevPartitionFunction = js["Log Sdev Partition Function"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Log Sdev Partition Function']\n%s", e.what()); } 
   eraseValue(js, "Log Sdev Partition Function");
 }

 if (isDefined(js, "Demonstration Policy"))
 {
 try { _demonstrationPolicy = js["Demonstration Policy"].get<std::string>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Demonstration Policy']\n%s", e.what()); } 
   eraseValue(js, "Demonstration Policy");
 }

 if (isDefined(js, "Feature Rescaling", "Means"))
 {
 try { _featureRescalingMeans = js["Feature Rescaling"]["Means"].get<std::vector<std::vector<float>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Feature Rescaling']['Means']\n%s", e.what()); } 
   eraseValue(js, "Feature Rescaling", "Means");
 }

 if (isDefined(js, "Feature Rescaling", "Sigmas"))
 {
 try { _featureRescalingSigmas = js["Feature Rescaling"]["Sigmas"].get<std::vector<std::vector<float>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Feature Rescaling']['Sigmas']\n%s", e.what()); } 
   eraseValue(js, "Feature Rescaling", "Sigmas");
 }

 if (isDefined(js, "Reward Update Count"))
 {
 try { _rewardUpdateCount = js["Reward Update Count"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Reward Update Count']\n%s", e.what()); } 
   eraseValue(js, "Reward Update Count");
 }

 if (isDefined(js, "Background Trajectory Count"))
 {
 try { _backgroundTrajectoryCount = js["Background Trajectory Count"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Background Trajectory Count']\n%s", e.what()); } 
   eraseValue(js, "Background Trajectory Count");
 }

 if (isDefined(js, "Demonstration Log Probability"))
 {
 try { _demonstrationLogProbability = js["Demonstration Log Probability"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Demonstration Log Probability']\n%s", e.what()); } 
   eraseValue(js, "Demonstration Log Probability");
 }

 if (isDefined(js, "Demonstration Feature Reward"))
 {
 try { _demonstrationFeatureReward = js["Demonstration Feature Reward"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Demonstration Feature Reward']\n%s", e.what()); } 
   eraseValue(js, "Demonstration Feature Reward");
 }

 if (isDefined(js, "Max Entropy Objective"))
 {
 try { _maxEntropyObjective = js["Max Entropy Objective"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Max Entropy Objective']\n%s", e.what()); } 
   eraseValue(js, "Max Entropy Objective");
 }

 if (isDefined(js, "Demonstration Batch Importance Weight"))
 {
 try { _demonstrationBatchImportanceWeight = js["Demonstration Batch Importance Weight"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Demonstration Batch Importance Weight']\n%s", e.what()); } 
   eraseValue(js, "Demonstration Batch Importance Weight");
 }

 if (isDefined(js, "Background Batch Importance Weight"))
 {
 try { _backgroundBatchImportanceWeight = js["Background Batch Importance Weight"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Background Batch Importance Weight']\n%s", e.what()); } 
   eraseValue(js, "Background Batch Importance Weight");
 }

 if (isDefined(js, "Effective Sample Size"))
 {
 try { _effectiveSampleSize = js["Effective Sample Size"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Effective Sample Size']\n%s", e.what()); } 
   eraseValue(js, "Effective Sample Size");
 }

 if (isDefined(js, "Statistic Log Partition Function"))
 {
 try { _statisticLogPartitionFunction = js["Statistic Log Partition Function"].get<std::vector<std::vector<float>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Statistic Log Partition Function']\n%s", e.what()); } 
   eraseValue(js, "Statistic Log Partition Function");
 }

 if (isDefined(js, "State Rescaling", "Means"))
 {
 try { _stateRescalingMeans = js["State Rescaling"]["Means"].get<std::vector<std::vector<float>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['State Rescaling']['Means']\n%s", e.what()); } 
   eraseValue(js, "State Rescaling", "Means");
 }

 if (isDefined(js, "State Rescaling", "Sigmas"))
 {
 try { _stateRescalingSigmas = js["State Rescaling"]["Sigmas"].get<std::vector<std::vector<float>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['State Rescaling']['Sigmas']\n%s", e.what()); } 
   eraseValue(js, "State Rescaling", "Sigmas");
 }

 if (isDefined(js, "Effective Minibatch Size"))
 {
 try { _effectiveMinibatchSize = js["Effective Minibatch Size"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Effective Minibatch Size']\n%s", e.what()); } 
   eraseValue(js, "Effective Minibatch Size");
 }

 if (isDefined(js, "Mode"))
 {
 try { _mode = js["Mode"].get<std::string>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Mode']\n%s", e.what()); } 
{
 bool validOption = false; 
 if (_mode == "Training") validOption = true; 
 if (_mode == "Testing") validOption = true; 
 if (validOption == false) KORALI_LOG_ERROR(" + Unrecognized value (%s) provided for mandatory setting: ['Mode'] required by agent.\n", _mode.c_str()); 
}
   eraseValue(js, "Mode");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Mode'] required by agent.\n"); 

 if (isDefined(js, "Testing", "Sample Ids"))
 {
 try { _testingSampleIds = js["Testing"]["Sample Ids"].get<std::vector<size_t>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Testing']['Sample Ids']\n%s", e.what()); } 
   eraseValue(js, "Testing", "Sample Ids");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Testing']['Sample Ids'] required by agent.\n"); 

 if (isDefined(js, "Testing", "Current Policies"))
 {
 _testingCurrentPolicies = js["Testing"]["Current Policies"].get<knlohmann::json>();

   eraseValue(js, "Testing", "Current Policies");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Testing']['Current Policies'] required by agent.\n"); 

 if (isDefined(js, "Training", "Average Depth"))
 {
 try { _trainingAverageDepth = js["Training"]["Average Depth"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Training']['Average Depth']\n%s", e.what()); } 
   eraseValue(js, "Training", "Average Depth");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Training']['Average Depth'] required by agent.\n"); 

 if (isDefined(js, "Concurrent Workers"))
 {
 try { _concurrentWorkers = js["Concurrent Workers"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Concurrent Workers']\n%s", e.what()); } 
   eraseValue(js, "Concurrent Workers");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Concurrent Workers'] required by agent.\n"); 

 if (isDefined(js, "Episodes Per Generation"))
 {
 try { _episodesPerGeneration = js["Episodes Per Generation"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Episodes Per Generation']\n%s", e.what()); } 
   eraseValue(js, "Episodes Per Generation");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Episodes Per Generation'] required by agent.\n"); 

 if (isDefined(js, "Mini Batch", "Size"))
 {
 try { _miniBatchSize = js["Mini Batch"]["Size"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Mini Batch']['Size']\n%s", e.what()); } 
   eraseValue(js, "Mini Batch", "Size");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Mini Batch']['Size'] required by agent.\n"); 

 if (isDefined(js, "Time Sequence Length"))
 {
 try { _timeSequenceLength = js["Time Sequence Length"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Time Sequence Length']\n%s", e.what()); } 
   eraseValue(js, "Time Sequence Length");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Time Sequence Length'] required by agent.\n"); 

 if (isDefined(js, "Learning Rate"))
 {
 try { _learningRate = js["Learning Rate"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Learning Rate']\n%s", e.what()); } 
   eraseValue(js, "Learning Rate");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Learning Rate'] required by agent.\n"); 

 if (isDefined(js, "L2 Regularization", "Enabled"))
 {
 try { _l2RegularizationEnabled = js["L2 Regularization"]["Enabled"].get<int>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['L2 Regularization']['Enabled']\n%s", e.what()); } 
   eraseValue(js, "L2 Regularization", "Enabled");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['L2 Regularization']['Enabled'] required by agent.\n"); 

 if (isDefined(js, "L2 Regularization", "Importance"))
 {
 try { _l2RegularizationImportance = js["L2 Regularization"]["Importance"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['L2 Regularization']['Importance']\n%s", e.what()); } 
   eraseValue(js, "L2 Regularization", "Importance");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['L2 Regularization']['Importance'] required by agent.\n"); 

 if (isDefined(js, "Neural Network", "Hidden Layers"))
 {
 _neuralNetworkHiddenLayers = js["Neural Network"]["Hidden Layers"].get<knlohmann::json>();

   eraseValue(js, "Neural Network", "Hidden Layers");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Neural Network']['Hidden Layers'] required by agent.\n"); 

 if (isDefined(js, "Neural Network", "Optimizer"))
 {
 try { _neuralNetworkOptimizer = js["Neural Network"]["Optimizer"].get<std::string>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Neural Network']['Optimizer']\n%s", e.what()); } 
   eraseValue(js, "Neural Network", "Optimizer");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Neural Network']['Optimizer'] required by agent.\n"); 

 if (isDefined(js, "Neural Network", "Engine"))
 {
 try { _neuralNetworkEngine = js["Neural Network"]["Engine"].get<std::string>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Neural Network']['Engine']\n%s", e.what()); } 
   eraseValue(js, "Neural Network", "Engine");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Neural Network']['Engine'] required by agent.\n"); 

 if (isDefined(js, "Discount Factor"))
 {
 try { _discountFactor = js["Discount Factor"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Discount Factor']\n%s", e.what()); } 
   eraseValue(js, "Discount Factor");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Discount Factor'] required by agent.\n"); 

 if (isDefined(js, "Importance Weight Truncation Level"))
 {
 try { _importanceWeightTruncationLevel = js["Importance Weight Truncation Level"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Importance Weight Truncation Level']\n%s", e.what()); } 
   eraseValue(js, "Importance Weight Truncation Level");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Importance Weight Truncation Level'] required by agent.\n"); 

 if (isDefined(js, "State Rescaling", "Enabled"))
 {
 try { _stateRescalingEnabled = js["State Rescaling"]["Enabled"].get<int>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['State Rescaling']['Enabled']\n%s", e.what()); } 
   eraseValue(js, "State Rescaling", "Enabled");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['State Rescaling']['Enabled'] required by agent.\n"); 

 if (isDefined(js, "Reward", "Rescaling", "Enabled"))
 {
 try { _rewardRescalingEnabled = js["Reward"]["Rescaling"]["Enabled"].get<int>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Reward']['Rescaling']['Enabled']\n%s", e.what()); } 
   eraseValue(js, "Reward", "Rescaling", "Enabled");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Reward']['Rescaling']['Enabled'] required by agent.\n"); 

 if (isDefined(js, "Feature Rescaling", "Enabled"))
 {
 try { _featureRescalingEnabled = js["Feature Rescaling"]["Enabled"].get<int>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Feature Rescaling']['Enabled']\n%s", e.what()); } 
   eraseValue(js, "Feature Rescaling", "Enabled");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Feature Rescaling']['Enabled'] required by agent.\n"); 

 if (isDefined(js, "Experience Replay", "Serialize"))
 {
 try { _experienceReplaySerialize = js["Experience Replay"]["Serialize"].get<int>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Experience Replay']['Serialize']\n%s", e.what()); } 
   eraseValue(js, "Experience Replay", "Serialize");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Experience Replay']['Serialize'] required by agent.\n"); 

 if (isDefined(js, "Experience Replay", "Start Size"))
 {
 try { _experienceReplayStartSize = js["Experience Replay"]["Start Size"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Experience Replay']['Start Size']\n%s", e.what()); } 
   eraseValue(js, "Experience Replay", "Start Size");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Experience Replay']['Start Size'] required by agent.\n"); 

 if (isDefined(js, "Experience Replay", "Maximum Size"))
 {
 try { _experienceReplayMaximumSize = js["Experience Replay"]["Maximum Size"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Experience Replay']['Maximum Size']\n%s", e.what()); } 
   eraseValue(js, "Experience Replay", "Maximum Size");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Experience Replay']['Maximum Size'] required by agent.\n"); 

 if (isDefined(js, "Experience Replay", "Off Policy", "Cutoff Scale"))
 {
 try { _experienceReplayOffPolicyCutoffScale = js["Experience Replay"]["Off Policy"]["Cutoff Scale"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Experience Replay']['Off Policy']['Cutoff Scale']\n%s", e.what()); } 
   eraseValue(js, "Experience Replay", "Off Policy", "Cutoff Scale");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Experience Replay']['Off Policy']['Cutoff Scale'] required by agent.\n"); 

 if (isDefined(js, "Experience Replay", "Off Policy", "Target"))
 {
 try { _experienceReplayOffPolicyTarget = js["Experience Replay"]["Off Policy"]["Target"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Experience Replay']['Off Policy']['Target']\n%s", e.what()); } 
   eraseValue(js, "Experience Replay", "Off Policy", "Target");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Experience Replay']['Off Policy']['Target'] required by agent.\n"); 

 if (isDefined(js, "Experience Replay", "Off Policy", "Annealing Rate"))
 {
 try { _experienceReplayOffPolicyAnnealingRate = js["Experience Replay"]["Off Policy"]["Annealing Rate"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Experience Replay']['Off Policy']['Annealing Rate']\n%s", e.what()); } 
   eraseValue(js, "Experience Replay", "Off Policy", "Annealing Rate");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Experience Replay']['Off Policy']['Annealing Rate'] required by agent.\n"); 

 if (isDefined(js, "Experience Replay", "Off Policy", "REFER Beta"))
 {
 try { _experienceReplayOffPolicyREFERBeta = js["Experience Replay"]["Off Policy"]["REFER Beta"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Experience Replay']['Off Policy']['REFER Beta']\n%s", e.what()); } 
   eraseValue(js, "Experience Replay", "Off Policy", "REFER Beta");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Experience Replay']['Off Policy']['REFER Beta'] required by agent.\n"); 

 if (isDefined(js, "Experiences Between Policy Updates"))
 {
 try { _experiencesBetweenPolicyUpdates = js["Experiences Between Policy Updates"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Experiences Between Policy Updates']\n%s", e.what()); } 
   eraseValue(js, "Experiences Between Policy Updates");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Experiences Between Policy Updates'] required by agent.\n"); 

 if (isDefined(js, "Experiences Between Reward Updates"))
 {
 try { _experiencesBetweenRewardUpdates = js["Experiences Between Reward Updates"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Experiences Between Reward Updates']\n%s", e.what()); } 
   eraseValue(js, "Experiences Between Reward Updates");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Experiences Between Reward Updates'] required by agent.\n"); 

 if (isDefined(js, "Optimize Max Entropy Objective"))
 {
 try { _optimizeMaxEntropyObjective = js["Optimize Max Entropy Objective"].get<int>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Optimize Max Entropy Objective']\n%s", e.what()); } 
   eraseValue(js, "Optimize Max Entropy Objective");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Optimize Max Entropy Objective'] required by agent.\n"); 

 if (isDefined(js, "Use Fusion Distribution"))
 {
 try { _useFusionDistribution = js["Use Fusion Distribution"].get<int>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Use Fusion Distribution']\n%s", e.what()); } 
   eraseValue(js, "Use Fusion Distribution");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Use Fusion Distribution'] required by agent.\n"); 

 if (isDefined(js, "Demonstration Batch Size"))
 {
 try { _demonstrationBatchSize = js["Demonstration Batch Size"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Demonstration Batch Size']\n%s", e.what()); } 
   eraseValue(js, "Demonstration Batch Size");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Demonstration Batch Size'] required by agent.\n"); 

 if (isDefined(js, "Background Batch Size"))
 {
 try { _backgroundBatchSize = js["Background Batch Size"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Background Batch Size']\n%s", e.what()); } 
   eraseValue(js, "Background Batch Size");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Background Batch Size'] required by agent.\n"); 

 if (isDefined(js, "Background Sample Size"))
 {
 try { _backgroundSampleSize = js["Background Sample Size"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Background Sample Size']\n%s", e.what()); } 
   eraseValue(js, "Background Sample Size");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Background Sample Size'] required by agent.\n"); 

 if (isDefined(js, "Reward Function", "Neural Network", "Hidden Layers"))
 {
 _rewardFunctionNeuralNetworkHiddenLayers = js["Reward Function"]["Neural Network"]["Hidden Layers"].get<knlohmann::json>();

   eraseValue(js, "Reward Function", "Neural Network", "Hidden Layers");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Reward Function']['Neural Network']['Hidden Layers'] required by agent.\n"); 

 if (isDefined(js, "Reward Function", "Learning Rate"))
 {
 try { _rewardFunctionLearningRate = js["Reward Function"]["Learning Rate"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Reward Function']['Learning Rate']\n%s", e.what()); } 
   eraseValue(js, "Reward Function", "Learning Rate");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Reward Function']['Learning Rate'] required by agent.\n"); 

 if (isDefined(js, "Reward Function", "Batch Size"))
 {
 try { _rewardFunctionBatchSize = js["Reward Function"]["Batch Size"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Reward Function']['Batch Size']\n%s", e.what()); } 
   eraseValue(js, "Reward Function", "Batch Size");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Reward Function']['Batch Size'] required by agent.\n"); 

 if (isDefined(js, "Reward Function", "L2 Regularization", "Enabled"))
 {
 try { _rewardFunctionL2RegularizationEnabled = js["Reward Function"]["L2 Regularization"]["Enabled"].get<int>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Reward Function']['L2 Regularization']['Enabled']\n%s", e.what()); } 
   eraseValue(js, "Reward Function", "L2 Regularization", "Enabled");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Reward Function']['L2 Regularization']['Enabled'] required by agent.\n"); 

 if (isDefined(js, "Reward Function", "L2 Regularization", "Importance"))
 {
 try { _rewardFunctionL2RegularizationImportance = js["Reward Function"]["L2 Regularization"]["Importance"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Reward Function']['L2 Regularization']['Importance']\n%s", e.what()); } 
   eraseValue(js, "Reward Function", "L2 Regularization", "Importance");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Reward Function']['L2 Regularization']['Importance'] required by agent.\n"); 

 if (isDefined(js, "Multi Agent Relationship"))
 {
 try { _multiAgentRelationship = js["Multi Agent Relationship"].get<std::string>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Multi Agent Relationship']\n%s", e.what()); } 
{
 bool validOption = false; 
 if (_multiAgentRelationship == "Individual") validOption = true; 
 if (_multiAgentRelationship == "Cooperation") validOption = true; 
 if (_multiAgentRelationship == "Competition") validOption = true; 
 if (validOption == false) KORALI_LOG_ERROR(" + Unrecognized value (%s) provided for mandatory setting: ['Multi Agent Relationship'] required by agent.\n", _multiAgentRelationship.c_str()); 
}
   eraseValue(js, "Multi Agent Relationship");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Multi Agent Relationship'] required by agent.\n"); 

 if (isDefined(js, "Multi Agent Correlation"))
 {
 try { _multiAgentCorrelation = js["Multi Agent Correlation"].get<int>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Multi Agent Correlation']\n%s", e.what()); } 
   eraseValue(js, "Multi Agent Correlation");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Multi Agent Correlation'] required by agent.\n"); 

 if (isDefined(js, "Multi Agent Sampling"))
 {
 try { _multiAgentSampling = js["Multi Agent Sampling"].get<std::string>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Multi Agent Sampling']\n%s", e.what()); } 
{
 bool validOption = false; 
 if (_multiAgentSampling == "Tuple") validOption = true; 
 if (_multiAgentSampling == "Baseline") validOption = true; 
 if (_multiAgentSampling == "Experience") validOption = true; 
 if (validOption == false) KORALI_LOG_ERROR(" + Unrecognized value (%s) provided for mandatory setting: ['Multi Agent Sampling'] required by agent.\n", _multiAgentSampling.c_str()); 
}
   eraseValue(js, "Multi Agent Sampling");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Multi Agent Sampling'] required by agent.\n"); 

 if (isDefined(js, "Termination Criteria", "Max Episodes"))
 {
 try { _maxEpisodes = js["Termination Criteria"]["Max Episodes"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Termination Criteria']['Max Episodes']\n%s", e.what()); } 
   eraseValue(js, "Termination Criteria", "Max Episodes");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Termination Criteria']['Max Episodes'] required by agent.\n"); 

 if (isDefined(js, "Termination Criteria", "Max Experiences"))
 {
 try { _maxExperiences = js["Termination Criteria"]["Max Experiences"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Termination Criteria']['Max Experiences']\n%s", e.what()); } 
   eraseValue(js, "Termination Criteria", "Max Experiences");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Termination Criteria']['Max Experiences'] required by agent.\n"); 

 if (isDefined(js, "Termination Criteria", "Max Policy Updates"))
 {
 try { _maxPolicyUpdates = js["Termination Criteria"]["Max Policy Updates"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Termination Criteria']['Max Policy Updates']\n%s", e.what()); } 
   eraseValue(js, "Termination Criteria", "Max Policy Updates");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Termination Criteria']['Max Policy Updates'] required by agent.\n"); 

 if (isDefined(_k->_js.getJson(), "Variables"))
 for (size_t i = 0; i < _k->_js["Variables"].size(); i++) { 
 } 
 Solver::setConfiguration(js);
 _type = "agent";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: agent: \n%s\n", js.dump(2).c_str());
} 

void Agent::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Mode"] = _mode;
   js["Testing"]["Sample Ids"] = _testingSampleIds;
   js["Testing"]["Current Policies"] = _testingCurrentPolicies;
   js["Training"]["Average Depth"] = _trainingAverageDepth;
   js["Concurrent Workers"] = _concurrentWorkers;
   js["Episodes Per Generation"] = _episodesPerGeneration;
   js["Mini Batch"]["Size"] = _miniBatchSize;
   js["Time Sequence Length"] = _timeSequenceLength;
   js["Learning Rate"] = _learningRate;
   js["L2 Regularization"]["Enabled"] = _l2RegularizationEnabled;
   js["L2 Regularization"]["Importance"] = _l2RegularizationImportance;
   js["Neural Network"]["Hidden Layers"] = _neuralNetworkHiddenLayers;
   js["Neural Network"]["Optimizer"] = _neuralNetworkOptimizer;
   js["Neural Network"]["Engine"] = _neuralNetworkEngine;
   js["Discount Factor"] = _discountFactor;
   js["Importance Weight Truncation Level"] = _importanceWeightTruncationLevel;
   js["State Rescaling"]["Enabled"] = _stateRescalingEnabled;
   js["Reward"]["Rescaling"]["Enabled"] = _rewardRescalingEnabled;
   js["Feature Rescaling"]["Enabled"] = _featureRescalingEnabled;
   js["Experience Replay"]["Serialize"] = _experienceReplaySerialize;
   js["Experience Replay"]["Start Size"] = _experienceReplayStartSize;
   js["Experience Replay"]["Maximum Size"] = _experienceReplayMaximumSize;
   js["Experience Replay"]["Off Policy"]["Cutoff Scale"] = _experienceReplayOffPolicyCutoffScale;
   js["Experience Replay"]["Off Policy"]["Target"] = _experienceReplayOffPolicyTarget;
   js["Experience Replay"]["Off Policy"]["Annealing Rate"] = _experienceReplayOffPolicyAnnealingRate;
   js["Experience Replay"]["Off Policy"]["REFER Beta"] = _experienceReplayOffPolicyREFERBeta;
   js["Experiences Between Policy Updates"] = _experiencesBetweenPolicyUpdates;
   js["Experiences Between Reward Updates"] = _experiencesBetweenRewardUpdates;
   js["Optimize Max Entropy Objective"] = _optimizeMaxEntropyObjective;
   js["Use Fusion Distribution"] = _useFusionDistribution;
   js["Demonstration Batch Size"] = _demonstrationBatchSize;
   js["Background Batch Size"] = _backgroundBatchSize;
   js["Background Sample Size"] = _backgroundSampleSize;
   js["Reward Function"]["Neural Network"]["Hidden Layers"] = _rewardFunctionNeuralNetworkHiddenLayers;
   js["Reward Function"]["Learning Rate"] = _rewardFunctionLearningRate;
   js["Reward Function"]["Batch Size"] = _rewardFunctionBatchSize;
   js["Reward Function"]["L2 Regularization"]["Enabled"] = _rewardFunctionL2RegularizationEnabled;
   js["Reward Function"]["L2 Regularization"]["Importance"] = _rewardFunctionL2RegularizationImportance;
   js["Multi Agent Relationship"] = _multiAgentRelationship;
   js["Multi Agent Correlation"] = _multiAgentCorrelation;
   js["Multi Agent Sampling"] = _multiAgentSampling;
   js["Termination Criteria"]["Max Episodes"] = _maxEpisodes;
   js["Termination Criteria"]["Max Experiences"] = _maxExperiences;
   js["Termination Criteria"]["Max Policy Updates"] = _maxPolicyUpdates;
   js["Policy"]["Parameter Count"] = _policyParameterCount;
   js["Action Lower Bounds"] = _actionLowerBounds;
   js["Action Upper Bounds"] = _actionUpperBounds;
   js["Current Episode"] = _currentEpisode;
   js["Training"]["Reward History"] = _trainingRewardHistory;
   js["Training"]["Feature Reward History"] = _trainingFeatureRewardHistory;
   js["Training"]["Experience History"] = _trainingExperienceHistory;
   js["Testing"]["Average Reward History"] = _testingAverageRewardHistory;
   js["Training"]["Average Reward"] = _trainingAverageReward;
   js["Training"]["Average Feature Reward"] = _trainingAverageFeatureReward;
   js["Training"]["Last Reward"] = _trainingLastReward;
   js["Training"]["Best Reward"] = _trainingBestReward;
   js["Training"]["Best Episode Id"] = _trainingBestEpisodeId;
   js["Training"]["Current Policies"] = _trainingCurrentPolicies;
   js["Training"]["Best Policies"] = _trainingBestPolicies;
   js["Testing"]["Reward"] = _testingReward;
   js["Testing"]["Best Reward"] = _testingBestReward;
   js["Testing"]["Worst Reward"] = _testingWorstReward;
   js["Testing"]["Best Episode Id"] = _testingBestEpisodeId;
   js["Testing"]["Candidate Count"] = _testingCandidateCount;
   js["Testing"]["Average Reward"] = _testingAverageReward;
   js["Testing"]["Best Average Reward"] = _testingBestAverageReward;
   js["Testing"]["Best Policies"] = _testingBestPolicies;
   js["Experience Replay"]["Off Policy"]["Count"] = _experienceReplayOffPolicyCount;
   js["Experience Replay"]["Off Policy"]["Ratio"] = _experienceReplayOffPolicyRatio;
   js["Experience Replay"]["Off Policy"]["History"] = _experienceReplayOffPolicyHistory;
   js["Experience Replay"]["Off Policy"]["Current Cutoff"] = _experienceReplayOffPolicyCurrentCutoff;
   js["Experience Replay"]["Off Policy"]["REFER Current Beta"] = _experienceReplayOffPolicyREFERCurrentBeta;
   js["Current Learning Rate"] = _currentLearningRate;
   js["Policy Update Count"] = _policyUpdateCount;
 if(_uniformGenerator != NULL) _uniformGenerator->getConfiguration(js["Uniform Generator"]);
   js["Experience Count"] = _experienceCount;
   js["Reward"]["Rescaling"]["Sigma"] = _rewardRescalingSigma;
   js["Reward"]["Rescaling"]["Sum Squared Rewards"] = _rewardRescalingSumSquaredRewards;
   js["Log Partition Function"] = _logPartitionFunction;
   js["Log Sdev Partition Function"] = _logSdevPartitionFunction;
   js["Demonstration Policy"] = _demonstrationPolicy;
   js["Feature Rescaling"]["Means"] = _featureRescalingMeans;
   js["Feature Rescaling"]["Sigmas"] = _featureRescalingSigmas;
   js["Reward Update Count"] = _rewardUpdateCount;
   js["Background Trajectory Count"] = _backgroundTrajectoryCount;
   js["Demonstration Log Probability"] = _demonstrationLogProbability;
   js["Demonstration Feature Reward"] = _demonstrationFeatureReward;
   js["Max Entropy Objective"] = _maxEntropyObjective;
   js["Demonstration Batch Importance Weight"] = _demonstrationBatchImportanceWeight;
   js["Background Batch Importance Weight"] = _backgroundBatchImportanceWeight;
   js["Effective Sample Size"] = _effectiveSampleSize;
   js["Statistic Log Partition Function"] = _statisticLogPartitionFunction;
   js["State Rescaling"]["Means"] = _stateRescalingMeans;
   js["State Rescaling"]["Sigmas"] = _stateRescalingSigmas;
   js["Effective Minibatch Size"] = _effectiveMinibatchSize;
 for (size_t i = 0; i <  _k->_variables.size(); i++) { 
 } 
 Solver::getConfiguration(js);
} 

void Agent::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Episodes Per Generation\": 1, \"Concurrent Workers\": 1, \"Discount Factor\": 0.995, \"Time Sequence Length\": 1, \"Importance Weight Truncation Level\": 1.0, \"Multi Agent Relationship\": \"Individual\", \"Multi Agent Correlation\": false, \"Multi Agent Sampling\": \"Tuple\", \"State Rescaling\": {\"Enabled\": false}, \"Feature Rescaling\": {\"Enabled\": false}, \"Reward\": {\"Rescaling\": {\"Enabled\": false}}, \"Mini Batch\": {\"Size\": 256}, \"L2 Regularization\": {\"Enabled\": false, \"Importance\": 0.0001}, \"Training\": {\"Average Depth\": 100, \"Current Policies\": {}, \"Best Policies\": {}}, \"Testing\": {\"Sample Ids\": [], \"Current Policies\": {}, \"Best Policies\": {}}, \"Termination Criteria\": {\"Max Episodes\": 0, \"Max Experiences\": 0, \"Max Policy Updates\": 0}, \"Experience Replay\": {\"Serialize\": true, \"Off Policy\": {\"Cutoff Scale\": 4.0, \"Target\": 0.1, \"REFER Beta\": 0.3, \"Annealing Rate\": 0.0}}, \"Uniform Generator\": {\"Name\": \"Agent / Uniform Generator\", \"Type\": \"Univariate/Uniform\", \"Minimum\": 0.0, \"Maximum\": 1.0}}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Solver::applyModuleDefaults(js);
} 

void Agent::applyVariableDefaults() 
{

 std::string defaultString = "{}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 if (isDefined(_k->_js.getJson(), "Variables"))
  for (size_t i = 0; i < _k->_js["Variables"].size(); i++) 
   mergeJson(_k->_js["Variables"][i], defaultJs); 
 Solver::applyVariableDefaults();
} 

bool Agent::checkTermination()
{
 bool hasFinished = false;

 if ((_mode == "Training") && (_maxEpisodes > 0) && (_currentEpisode >= _maxEpisodes))
 {
  _terminationCriteria.push_back("agent['Max Episodes'] = " + std::to_string(_maxEpisodes) + ".");
  hasFinished = true;
 }

 if ((_mode == "Training") && (_maxExperiences > 0) && (_experienceCount >= _maxExperiences))
 {
  _terminationCriteria.push_back("agent['Max Experiences'] = " + std::to_string(_maxExperiences) + ".");
  hasFinished = true;
 }

 if ((_mode == "Training") && (_maxPolicyUpdates > 0) && (_policyUpdateCount >= _maxPolicyUpdates))
 {
  _terminationCriteria.push_back("agent['Max Policy Updates'] = " + std::to_string(_maxPolicyUpdates) + ".");
  hasFinished = true;
 }

 hasFinished = hasFinished || Solver::checkTermination();
 return hasFinished;
}

;

} //solver
} //korali
;
