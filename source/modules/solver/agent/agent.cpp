#include "auxiliar/fs.hpp"
#include "engine.hpp"
#include "modules/solver/agent/agent.hpp"
#include "sample/sample.hpp"
#include <chrono>

namespace korali
{
namespace solver
{


void Agent::initialize()
{
  _variableCount = _k->_variables.size();

  // Getting problem pointer
  _problem = dynamic_cast<problem::ReinforcementLearning *>(_k->_problem);

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

  //  Pre-allocating space for the experience replay memory
  _stateVector.resize(_experienceReplayMaximumSize);
  _actionVector.resize(_experienceReplayMaximumSize);
  _retraceValueVector.resize(_experienceReplayMaximumSize);
  _rewardVector.resize(_experienceReplayMaximumSize);
  _stateValueVector.resize(_experienceReplayMaximumSize);
  _importanceWeightVector.resize(_experienceReplayMaximumSize);
  _truncatedImportanceWeightVector.resize(_experienceReplayMaximumSize);
  _truncatedStateValueVector.resize(_experienceReplayMaximumSize);
  _truncatedStateVector.resize(_experienceReplayMaximumSize);
  _terminationVector.resize(_experienceReplayMaximumSize);
  _expPolicyVector.resize(_experienceReplayMaximumSize);
  _curPolicyVector.resize(_experienceReplayMaximumSize);
  _isOnPolicyVector.resize(_experienceReplayMaximumSize);
  _episodePosVector.resize(_experienceReplayMaximumSize);
  _episodeIdVector.resize(_experienceReplayMaximumSize);

  //  Pre-allocating space for state time sequence
  _stateTimeSequence.resize(_timeSequenceLength);

  /*********************************************************************
   *   // If initial generation, set initial agent configuration
   *********************************************************************/

  if (_k->_currentGeneration == 0)
  {
    _currentEpisode = 0;
    _policyUpdateCount = 0;
    _testingCandidateCount = 0;
    _currentSampleID = 0;
    _experienceCount = 0;

    // Initializing training and episode statistics
    _testingAverageReward = -korali::Inf;
    _testingStdevReward = +korali::Inf;
    _testingBestReward = -korali::Inf;
    _testingWorstReward = +korali::Inf;
    _trainingBestReward = -korali::Inf;
    _trainingBestEpisodeId = 0;
    _trainingAverageReward = 0.0f;
    _testingPreviousAverageReward = -korali::Inf;
    _testingBestAverageReward = -korali::Inf;
    _testingBestEpisodeId = 0;

    // Initializing REFER information

    // If cutoff scale is not defined, use a heuristic value
    if (_experienceReplayOffPolicyCutoffScale < 0.0f)
      KORALI_LOG_ERROR("Expericne Replay Cutoff Scale must be larger 0.0");

    _experienceReplayOffPolicyCount = 0;
    _experienceReplayOffPolicyRatio = 0.0f;
    _experienceReplayOffPolicyCurrentCutoff = _experienceReplayOffPolicyCutoffScale;
    _currentLearningRate = _learningRate;

    // Rescaling information
    _stateRescalingMeans = std::vector<float>(_problem->_stateVectorSize, 0.0);
    _stateRescalingSigmas = std::vector<float>(_problem->_stateVectorSize, 1.0);

    _rewardRescalingMean = 0.0f;
    _rewardRescalingSigma = 1.0f;
    _rewardRescalingCount = 0;
    _rewardOutboundPenalizationCount = 0;
  }

  // If this continues a previous training run, deserialize previous input experience replay
  if (_k->_currentGeneration > 0)
    if (_mode == "Training" || _testingBestPolicy.empty())
      deserializeExperienceReplay();

  // Getting agent's initial policy
  _trainingCurrentPolicy = getAgentPolicy();

  // Initializing session-wise profiling timers
  _sessionRunningTime = 0.0;
  _sessionSerializationTime = 0.0;
  _sessionAgentComputationTime = 0.0;
  _sessionAgentCommunicationTime = 0.0;
  _sessionAgentPolicyEvaluationTime = 0.0;
  _sessionPolicyUpdateTime = 0.0;
  _sessionAgentAttendingTime = 0.0;

  // Initializing session-specific counters
  _sessionExperienceCount = 0;
  _sessionEpisodeCount = 0;
  _sessionGeneration = 1;
  _sessionPolicyUpdateCount = 0;

  // Calculating how many more experiences do we need in this session to reach the starting size
  _sessionExperiencesUntilStartSize = _stateVector.size() > _experienceReplayStartSize ? 0 : _experienceReplayStartSize - _stateVector.size();

  if (_mode == "Training")
  {
    // Creating storate for _agents and their status
    _agents.resize(_concurrentEnvironments);
    _isAgentRunning.resize(_concurrentEnvironments, false);
  }

  if (_mode == "Testing")
  {
    // Fixing termination criteria for testing mode
    _maxGenerations = _k->_currentGeneration + 1;

    // Setting testing policy to best testing hyperparameters if not custom-set by the user
    if (_testingPolicy.empty())
    {
      // Checking if testing policies have been generated
      if (_testingBestPolicy.empty())
      {
        _k->_logger->logWarning("Minimal", "Trying to test policy, but no testing policies have been generated during training yet or given in the configuration. Using current training policy instead.\n");
        _testingPolicy = _trainingCurrentPolicy;
      }
      else
      {
        _testingPolicy = _testingBestPolicy;
      }
    }

    // Checking if there's testing samples defined
    if (_testingSampleIds.size() == 0)
      KORALI_LOG_ERROR("For testing, you need to indicate the sample ids to run in the ['Testing']['Sample Ids'] field.\n");

    // Prepare storage for rewards from tested samples
    _testingReward.resize(_testingSampleIds.size());
  }
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
  _generationAgentComputationTime = 0.0;
  _generationAgentCommunicationTime = 0.0;
  _generationAgentPolicyEvaluationTime = 0.0;
  _generationPolicyUpdateTime = 0.0;
  _generationAgentAttendingTime = 0.0;

  // Running until all _agents have finished
  while (_sessionEpisodeCount < _episodesPerGeneration * _sessionGeneration)
  {
    // Launching (or re-launching) agents
    for (size_t agentId = 0; agentId < _concurrentEnvironments; agentId++)
      if (_isAgentRunning[agentId] == false)
      {
        _agents[agentId]["Sample Id"] = _currentEpisode++;
        _agents[agentId]["Module"] = "Problem";
        _agents[agentId]["Operation"] = "Run Training Episode";
        _agents[agentId]["Policy Hyperparameters"] = _trainingCurrentPolicy;
        _agents[agentId]["State Rescaling"]["Means"] = _stateRescalingMeans;
        _agents[agentId]["State Rescaling"]["Standard Deviations"] = _stateRescalingSigmas;

        KORALI_START(_agents[agentId]);
        _isAgentRunning[agentId] = true;
      }

    // Listening to _agents for incoming experiences
    KORALI_LISTEN(_agents);

    // Attending to running agents, checking if any experience has been received
    for (size_t agentId = 0; agentId < _concurrentEnvironments; agentId++)
      if (_isAgentRunning[agentId] == true)
        attendAgent(agentId);

    // Perform optimization steps on the critic/policy, if reached the minimum replay memory size
    if (_experienceCount >= _experienceReplayStartSize)
    {
      // If we performed enough policy updates, we rescale rewards again
      if (_rewardRescalingEnabled == true)
        if (_policyUpdateCount >= _rewardRescalingFrequency * _rewardRescalingCount)
          calculateRewardRescalingFactors();

      // If we accumulated enough experiences, we rescale the states (once)
      if (_stateRescalingEnabled == true)
        if (_policyUpdateCount == 0)
          rescaleStates();

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

        // Updating REFER learning rate and beta parameters
        _currentLearningRate = _learningRate / (1.0f + _experienceReplayOffPolicyAnnealingRate * (float)_policyUpdateCount);
        if (_experienceReplayOffPolicyRatio > _experienceReplayOffPolicyTarget)
          _experienceReplayOffPolicyREFERBeta = (1.0f - _currentLearningRate) * _experienceReplayOffPolicyREFERBeta;
        else
          _experienceReplayOffPolicyREFERBeta = (1.0f - _currentLearningRate) * _experienceReplayOffPolicyREFERBeta + _currentLearningRate;
      }

      // Getting new policy hyperparameters (for agents to generate actions)
      _trainingCurrentPolicy = getAgentPolicy();
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
  _trainingAverageReward = 0.0f;
  ssize_t startEpisodeId = _trainingRewardHistory.size() - _trainingAverageDepth;
  ssize_t endEpisodeId = _trainingRewardHistory.size() - 1;
  if (startEpisodeId < 0) startEpisodeId = 0;
  for (ssize_t e = startEpisodeId; e <= endEpisodeId; e++)
    _trainingAverageReward += _trainingRewardHistory[e];
  _trainingAverageReward /= (float)(endEpisodeId - startEpisodeId + 1);

  // Increasing session's generation count
  _sessionGeneration++;
}

void Agent::testingGeneration()
{
  // Allocating testing agents
  std::vector<Sample> testingAgents(_testingSampleIds.size());

  // Launching  agents
  for (size_t agentId = 0; agentId < _testingSampleIds.size(); agentId++)
  {
    testingAgents[agentId]["Sample Id"] = _testingSampleIds[agentId];
    testingAgents[agentId]["Module"] = "Problem";
    testingAgents[agentId]["Operation"] = "Run Testing Episode";
    testingAgents[agentId]["Policy Hyperparameters"] = _testingPolicy;
    testingAgents[agentId]["State Rescaling"]["Means"] = _stateRescalingMeans;
    testingAgents[agentId]["State Rescaling"]["Standard Deviations"] = _stateRescalingSigmas;

    KORALI_START(testingAgents[agentId]);
  }

  KORALI_WAITALL(testingAgents);

  for (size_t agentId = 0; agentId < _testingSampleIds.size(); agentId++)
    _testingReward[agentId] = testingAgents[agentId]["Testing Reward"].get<float>();
}

void Agent::rescaleStates()
{
  // Calculation of state moments
  std::vector<float> sumStates(_problem->_stateVectorSize, 0.0);
  std::vector<float> squaredSumStates(_problem->_stateVectorSize, 0.0);

  for (size_t i = 0; i < _stateVector.size(); ++i)
    for (size_t d = 0; d < _problem->_stateVectorSize; ++d)
    {
      sumStates[d] += _stateVector[i][d];
      squaredSumStates[d] += _stateVector[i][d] * _stateVector[i][d];
    }

  _k->_logger->logInfo("Detailed", " + Using State Normalization N(Mean, Sigma):\n");

  for (size_t d = 0; d < _problem->_stateVectorSize; ++d)
  {
    _stateRescalingMeans[d] = sumStates[d] / (float)_stateVector.size();
    if (std::isfinite(_stateRescalingMeans[d]) == false) _stateRescalingMeans[d] = 0.0f;

    _stateRescalingSigmas[d] = std::sqrt(squaredSumStates[d] / (float)_stateVector.size() - _stateRescalingMeans[d] * _stateRescalingMeans[d]);
    if (std::isfinite(_stateRescalingSigmas[d]) == false) _stateRescalingSigmas[d] = 1.0f;
    if (_stateRescalingSigmas[d] <= 1e-9) _stateRescalingSigmas[d] = 1.0f;

    _k->_logger->logInfo("Detailed", " + State [%zu]: N(%f, %f)\n", d, _stateRescalingMeans[d], _stateRescalingSigmas[d]);
  }

  // Actual rescaling of initial states
  for (size_t i = 0; i < _stateVector.size(); ++i)
    for (size_t d = 0; d < _problem->_stateVectorSize; ++d)
      _stateVector[i][d] = (_stateVector[i][d] - _stateRescalingMeans[d]) / _stateRescalingSigmas[d];
}

void Agent::calculateRewardRescalingFactors()
{
  float sumReward = 0.0;
  float sumSquareReward = 0.0;

  // Calculate mean and standard deviation of unscaled rewards.
  for (size_t i = 0; i < _rewardVector.size(); i++)
  {
    float reward = _rewardVector[i];
    sumReward += reward;
    sumSquareReward += reward * reward;
  }

  // Calculating reward scaling s,t. mean equals 0.0 and standard deviation 1.0.
  _rewardRescalingMean = sumReward / (float)_rewardVector.size();
  _rewardRescalingSigma = std::sqrt(sumSquareReward / (float)_rewardVector.size() - _rewardRescalingMean * _rewardRescalingMean + 1e-9);
  _rewardRescalingCount++;
}

void Agent::attendAgent(size_t agentId)
{
  auto beginTime = std::chrono::steady_clock::now(); // Profiling

  // Storage for the incoming message
  knlohmann::json message;

  // Retrieving the experience, if any has arrived for the current agent.
  if (_agents[agentId].retrievePendingMessage(message))
  {
    // Getting episode Id
    size_t episodeId = message["Sample Id"];

    // If agent requested new policy, send the new hyperparameters
    if (message["Action"] == "Request New Policy")
      KORALI_SEND_MSG_TO_SAMPLE(_agents[agentId], _trainingCurrentPolicy);

    // Process episode(s) incoming from the agent(s)
    if (message["Action"] == "Send Episodes")
    {
      // Process every episode received and its experiences (add them to replay memory)
      for (size_t i = 0; i < _problem->_agentsPerEnvironment; i++)
      {
       processEpisode(episodeId, message["Episodes"][i]);

       // Increasing total experience counters
       _experienceCount += message["Episodes"][i]["Experiences"].size();
       _sessionExperienceCount += message["Episodes"][i]["Experiences"].size();
      }

      // Waiting for the agent to come back with all the information
      KORALI_WAIT(_agents[agentId]);

      // Getting the training reward of the latest episodes
      for (size_t i = 0; i < message["Episodes"].size(); i++)
      {
       _trainingLastReward = _agents[agentId]["Training Rewards"][i].get<float>();

       // Keeping training statistics. Updating if exceeded best training policy so far.
       if (_trainingLastReward > _trainingBestReward)
       {
         _trainingBestReward = _trainingLastReward;
         _trainingBestEpisodeId = episodeId;
         _trainingBestPolicy = _agents[agentId]["Policy Hyperparameters"];
       }

       // Storing bookkeeping information
       _trainingRewardHistory.push_back(_trainingLastReward);
       _trainingExperienceHistory.push_back(message["Episodes"][i]["Experiences"].size());
      }

      // If the policy has exceeded the threshold during training, we gather its statistics
      if (_agents[agentId]["Tested Policy"] == true)
      {
        _testingCandidateCount++;

        _testingPreviousAverageReward = _testingAverageReward;
        _testingAverageReward = _agents[agentId]["Average Testing Reward"].get<float>();
        _testingStdevReward = _agents[agentId]["Stdev Testing Reward"].get<float>();
        _testingBestReward = _agents[agentId]["Best Testing Reward"].get<float>();
        _testingWorstReward = _agents[agentId]["Worst Testing Reward"].get<float>();

        // If the average testing reward is better than the previous best, replace it
        // and store hyperparameters as best so far.
        if (_testingAverageReward > _testingBestAverageReward)
        {
          _testingBestAverageReward = _testingAverageReward;
          _testingBestEpisodeId = episodeId;
          _testingBestPolicy = _agents[agentId]["Policy Hyperparameters"];
        }
      }

      // Obtaining profiling information
      _sessionAgentComputationTime += _agents[agentId]["Computation Time"].get<double>();
      _sessionAgentCommunicationTime += _agents[agentId]["Communication Time"].get<double>();
      _sessionAgentPolicyEvaluationTime += _agents[agentId]["Policy Evaluation Time"].get<double>();
      _generationAgentComputationTime += _agents[agentId]["Computation Time"].get<double>();
      _generationAgentCommunicationTime += _agents[agentId]["Communication Time"].get<double>();
      _generationAgentPolicyEvaluationTime += _agents[agentId]["Policy Evaluation Time"].get<double>();

      // Set agent as finished
      _isAgentRunning[agentId] = false;

      // Increasing session episode count
      _sessionEpisodeCount++;
    }
  }

  auto endTime = std::chrono::steady_clock::now();                                                                    // Profiling
  _sessionAgentAttendingTime += std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - beginTime).count();    // Profiling
  _generationAgentAttendingTime += std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - beginTime).count(); // Profiling
}

void Agent::processEpisode(size_t episodeId, knlohmann::json &episode)
{
  /*********************************************************************
  * Adding episode's experiences into the replay memory
  *********************************************************************/

  // Storage for the episode's cumulative reward
  float cumulativeReward = 0.0f;

  for (size_t expId = 0; expId < episode["Experiences"].size(); expId++)
  {
    // Getting state
    _stateVector.add(episode["Experiences"][expId]["State"].get<std::vector<float>>());

    // Getting action
    const auto action = episode["Experiences"][expId]["Action"].get<std::vector<float>>();
    _actionVector.add(action);

    // Getting reward
    float reward = episode["Experiences"][expId]["Reward"].get<float>();

    // If the action is outside the boundary, applying penalization factor
    if (_rewardOutboundPenalizationEnabled == true)
    {
      bool outOfBounds = false;
      for (size_t i = 0; i < _problem->_actionVectorSize; i++)
      {
        if (action[i] > _actionUpperBounds[i]) outOfBounds = true;
        if (action[i] < _actionLowerBounds[i]) outOfBounds = true;
      }

      if (outOfBounds == true)
      {
        reward = reward * _rewardOutboundPenalizationFactor;
        _rewardOutboundPenalizationCount++;
      }
    }

    _rewardVector.add(reward);

    // Keeping statistics
    cumulativeReward += reward;

    // Checking experience termination status and truncated state
    termination_t termination;
    std::vector<float> truncatedState;

    if (episode["Experiences"][expId]["Termination"] == "Non Terminal") termination = e_nonTerminal;
    if (episode["Experiences"][expId]["Termination"] == "Terminal") termination = e_terminal;
    if (episode["Experiences"][expId]["Termination"] == "Truncated")
    {
      termination = e_truncated;
      truncatedState = episode["Experiences"][expId]["Truncated State"].get<std::vector<float>>();
    }

    _terminationVector.add(termination);
    _truncatedStateVector.add(truncatedState);

    // Getting policy information and state value
    policy_t expPolicy;
    float stateValue;

    if (isDefined(episode["Experiences"][expId], "Policy", "State Value"))
    {
      expPolicy.stateValue = episode["Experiences"][expId]["Policy"]["State Value"].get<float>();
      stateValue = episode["Experiences"][expId]["Policy"]["State Value"].get<float>();
    }
    else
    {
      KORALI_LOG_ERROR("Policy has not produced state value for the current experience.\n");
    }

    if (isDefined(episode["Experiences"][expId], "Policy", "Distribution Parameters"))
      expPolicy.distributionParameters = episode["Experiences"][expId]["Policy"]["Distribution Parameters"].get<std::vector<float>>();

    if (isDefined(episode["Experiences"][expId], "Policy", "Action Index"))
      expPolicy.actionIndex = episode["Experiences"][expId]["Policy"]["Action Index"].get<size_t>();

    if (isDefined(episode["Experiences"][expId], "Policy", "Unbounded Action"))
      expPolicy.unboundedAction = episode["Experiences"][expId]["Policy"]["Unbounded Action"].get<std::vector<float>>();

    // Storing policy information
    _expPolicyVector.add(expPolicy);
    _curPolicyVector.add(expPolicy);
    _stateValueVector.add(stateValue);

    // Storing Episode information
    _episodeIdVector.add(episodeId);
    _episodePosVector.add(expId);

    // Adding placeholder for retrace value
    _retraceValueVector.add(0.0f);

    // If there's an outgoing experience and it's off policy, subtract the off policy counter
    if (_isOnPolicyVector.size() == _experienceReplayMaximumSize)
      if (_isOnPolicyVector[0] == false)
        _experienceReplayOffPolicyCount--;

    // Adding new experience's on policiness (by default is true when adding it to the ER)
    _isOnPolicyVector.add(true);

    // Updating experience's importance weight. Initially assumed to be 1.0 because its freshly produced
    _importanceWeightVector.add(1.0f);
    _truncatedImportanceWeightVector.add(1.0f);
  }

  /*********************************************************************
   * Computing initial retrace value for the newly added experiences
   *********************************************************************/

  // Storage for the retrace value
  float retV = 0.0f;

  // Getting position of the final experience of the episode in the replay memory
  ssize_t endId = (ssize_t)_stateVector.size() - 1;

  // Getting the starting ID of the initial experience of the episode in the replay memory
  ssize_t startId = endId - (ssize_t)episode.size() + 1;

  // If it was a truncated episode, add the value function for the terminal state to retV
  if (_terminationVector[endId] == e_truncated)
  {
    // Get state sequence, appending the truncated state to it and removing first time element
    auto expTruncatedStateSequence = getTruncatedStateSequence(endId);

    // Calculating the state value function of the truncated state
    auto truncatedPolicy = runPolicy({expTruncatedStateSequence})[0];
    float truncatedV = truncatedPolicy.stateValue;

    // Sanity checks for truncated state value
    if (std::isfinite(truncatedV) == false)
      KORALI_LOG_ERROR("Calculated state value for truncated state returned an invalid value: %f\n", truncatedV);

    // Adding truncated state value to the retrace value
    retV += _discountFactor * truncatedV;
  }

  // Now going backwards, setting the retrace value of every experience
  for (ssize_t expId = endId; expId >= startId; expId--)
  {
    // Calculating retrace value with the discount factor. Importance weight is 1.0f because the policy is current.
    retV = _discountFactor * retV + getScaledReward(_rewardVector[expId]);

    // Setting initial retrace value in the experience's cache
    _retraceValueVector[expId] = retV;
  }
}

std::vector<size_t> Agent::generateMiniBatch(size_t miniBatchSize)
{
  // Allocating storage for mini batch experiecne indexes
  std::vector<size_t> miniBatch(miniBatchSize);

  for (size_t i = 0; i < miniBatchSize; i++)
  {
    // Producing random (uniform) number for the selection of the experience
    float x = _uniformGenerator->getRandomNumber();

    // Selecting experience
    size_t expId = std::floor(x * (float)(_stateVector.size() - 1));

    // Setting experience
    miniBatch[i] = expId;
  }

  // Sorting minibatch -- this helps with locality and also
  // to quickly detect duplicates when updating metadata
  std::sort(miniBatch.begin(), miniBatch.end());

  // Returning generated minibatch
  return miniBatch;
}

void Agent::updateExperienceMetadata(const std::vector<size_t> &miniBatch, const std::vector<policy_t> &policyData)
{
  const size_t miniBatchSize = miniBatch.size();

  // Creating a selection of unique experiences from the mini batch
  // Important: this assumes the minibatch ids are sorted.
  std::vector<size_t> updateBatch;
  updateBatch.push_back(0);
  for (size_t i = 1; i < miniBatchSize; i++)
    if (miniBatch[i] != miniBatch[i - 1]) updateBatch.push_back(i);

#pragma omp parallel for
  for (size_t i = 0; i < updateBatch.size(); i++)
  {
    auto batchId = updateBatch[i];
    auto expId = miniBatch[batchId];

    // Get state, action, mean, Sigma for this experience
    const auto &expAction = _actionVector[expId];
    const auto &expPolicy = _expPolicyVector[expId];
    const auto &curPolicy = policyData[batchId];

    // Grabbing state value from the latenst policy
    auto stateValue = curPolicy.stateValue;

    // Sanity checks for state value
    if (std::isfinite(stateValue) == false)
      KORALI_LOG_ERROR("Calculated state value returned an invalid value: %f\n", stateValue);

    // Compute importance weight
    const float importanceWeight = calculateImportanceWeight(expAction, curPolicy, expPolicy);
    const float truncatedImportanceWeight = std::min(1.0f, importanceWeight);

    // Sanity checks for state value
    if (std::isfinite(importanceWeight) == false)
      KORALI_LOG_ERROR("Calculated value of importanceWeight returned an invalid value: %f\n", importanceWeight);

    // Checking if experience is still on policy
    bool isOnPolicy = (importanceWeight > (1.0f / _experienceReplayOffPolicyCurrentCutoff)) && (importanceWeight < _experienceReplayOffPolicyCurrentCutoff);

    // Updating off policy count if a change is detected
    if (_isOnPolicyVector[expId] == true && isOnPolicy == false)
       #pragma omp atomic
      _experienceReplayOffPolicyCount++;

    if (_isOnPolicyVector[expId] == false && isOnPolicy == true)
      #pragma omp atomic
      _experienceReplayOffPolicyCount--;

    // Store computed information for use in replay memory.
    _curPolicyVector[expId] = curPolicy;
    _stateValueVector[expId] = stateValue;
    _truncatedStateValueVector[expId] = 0.0f;
    _importanceWeightVector[expId] = importanceWeight;
    _isOnPolicyVector[expId] = isOnPolicy;
    _truncatedImportanceWeightVector[expId] = truncatedImportanceWeight;
  }

  // Storage to register whether the experience needs an updated truncated policy state value
  std::vector<size_t> needsTruncatedPolicyUpdate;
  for (size_t i = 0; i < updateBatch.size(); i++)
  {
   auto batchId = updateBatch[i];
   auto expId = miniBatch[batchId];
   if (_terminationVector[expId] == e_truncated)
    needsTruncatedPolicyUpdate.push_back(expId);
  }

  // If this is the truncated experience of an episode, then obtain truncated state value
  #pragma omp parallel for
  for (size_t i = 0; i < needsTruncatedPolicyUpdate.size(); i++)
  {
    auto expId = needsTruncatedPolicyUpdate[i];
    auto truncatedState = getTruncatedStateSequence(expId);
    auto truncatedPolicy = runPolicy({getTruncatedStateSequence(expId)})[0];
    _truncatedStateValueVector[expId] = truncatedPolicy.stateValue;
  }

  // Updating the off policy Ratio
  _experienceReplayOffPolicyRatio = (float)_experienceReplayOffPolicyCount / (float)_isOnPolicyVector.size();

  // Updating the off policy Cutoff
  _experienceReplayOffPolicyCurrentCutoff = _experienceReplayOffPolicyCutoffScale / (1.0f + _experienceReplayOffPolicyAnnealingRate * (float)_policyUpdateCount);

  // Now filtering experiences from the same episode
  std::vector<size_t> retraceMiniBatch;

  // Adding last experience from the sorted minibatch
  retraceMiniBatch.push_back(miniBatch[miniBatchSize - 1]);

  // Adding experiences so long as they do not repeat episodes
  for (ssize_t i = miniBatchSize - 2; i >= 0; i--)
  {
    size_t currExpId = miniBatch[i];
    size_t nextExpId = miniBatch[i + 1];
    size_t curEpisode = _episodeIdVector[currExpId];
    size_t nextEpisode = _episodeIdVector[nextExpId];
    if (curEpisode != nextEpisode) retraceMiniBatch.push_back(currExpId);
  }

// Calculating retrace value for the oldest experiences of unique episodes
#pragma omp parallel for schedule(guided, 1)
  for (size_t i = 0; i < retraceMiniBatch.size(); i++)
  {
    // Finding the earliest experience corresponding to the same episode as this experience
    ssize_t endId = retraceMiniBatch[i];
    ssize_t startId = endId - _episodePosVector[endId];

    // If the starting experience has already been discarded, take the earliest one that still remains
    if (startId < 0) startId = 0;

    // Storage for the retrace value
    float retV = 0.0f;

    // If it was a truncated episode, add the value function for the terminal state to retV
    if (_terminationVector[endId] == e_truncated)
      retV = _truncatedStateValueVector[endId];

    if (_terminationVector[endId] == e_nonTerminal)
      retV = _retraceValueVector[endId + 1];

    // Now iterating backwards to calculate the rest of vTbc
    for (ssize_t curId = endId; curId >= startId; curId--)
    {
      // Getting current reward, action, and state
      const float curReward = getScaledReward(_rewardVector[curId]);

      // Calculating state value function
      const float curV = _stateValueVector[curId];

      // Truncate importance weight
      const float truncatedImportanceWeight = _truncatedImportanceWeightVector[curId];

      // Calculating retrace value
      retV = curV + truncatedImportanceWeight * (curReward + _discountFactor * retV - curV);

      // Storing retrace value into the experience's cache
      _retraceValueVector[curId] = retV;
    }
  }
}

size_t Agent::getTimeSequenceStartExpId(size_t expId)
{
  size_t startId = expId;

  // Adding (tmax-1) time sequences to the given experience
  for (size_t t = 0; t < _timeSequenceLength - 1; t++)
  {
    // If we reached the start of the ER, this is the starting episode in the sequence
    if (startId == 0) break;

    // Now going back one experience
    startId--;

    // If we reached the end of the previous episode, then add one (this covers the case where the provided experience is also terminal) and break.
    if (_terminationVector[startId] != e_nonTerminal)
    {
      startId++;
      break;
    }
  }

  return startId;
}

void Agent::resetTimeSequence()
{
  _stateTimeSequence.clear();
}

std::vector<std::vector<std::vector<float>>> Agent::getMiniBatchStateSequence(const std::vector<size_t> &miniBatch, const bool includeAction)
{
  // Getting mini batch size
  const size_t miniBatchSize = miniBatch.size();

  // Allocating state sequence vector
  std::vector<std::vector<std::vector<float>>> stateSequence(miniBatchSize);

  // Calculating size of state vector
  const size_t stateSize = includeAction ? _problem->_stateVectorSize + _problem->_actionVectorSize : _problem->_stateVectorSize;

#pragma omp parallel for
  for (size_t b = 0; b < miniBatch.size(); b++)
  {
    // Getting current expId
    const size_t expId = miniBatch[b];

    // Getting starting expId
    const size_t startId = getTimeSequenceStartExpId(expId);

    // Calculating time sequence length
    const size_t T = expId - startId + 1;

    // Resizing state sequence vector to the correct time sequence length
    stateSequence[b].resize(T);

    // Now adding states (and actions, if required)
    for (size_t t = 0; t < T; t++)
    {
      size_t curId = startId + t;
      stateSequence[b][t].reserve(stateSize);
      stateSequence[b][t].insert(stateSequence[b][t].begin(), _stateVector[curId].begin(), _stateVector[curId].end());
      if (includeAction) stateSequence[b][t].insert(stateSequence[b][t].begin(), _actionVector[curId].begin(), _actionVector[curId].end());
    }
  }

  return stateSequence;
}

std::vector<std::vector<float>> Agent::getTruncatedStateSequence(size_t expId)
{
  // Getting starting expId
  size_t startId = getTimeSequenceStartExpId(expId);

  // Creating storage for the time sequence
  std::vector<std::vector<float>> timeSequence;

  // Now adding states, except for the initial one
  for (size_t e = startId + 1; e <= expId; e++)
    timeSequence.push_back(_stateVector[e]);

  // Lastly, adding truncated state
  timeSequence.push_back(_truncatedStateVector[expId]);

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
    for (size_t agentId = 0; agentId < _concurrentEnvironments; agentId++)
      if (_isAgentRunning[agentId] == true)
      {
        attendAgent(agentId);
        agentsRemain = true;
      }

    if (agentsRemain) KORALI_LISTEN(_agents);
  } while (agentsRemain == true);
}

void Agent::serializeExperienceReplay()
{
  _k->_logger->logInfo("Detailed", "Serializing Agent's Training State...\n");
  auto beginTime = std::chrono::steady_clock::now(); // Profiling

  // Creating JSON storage variable
  knlohmann::json stateJson;

  // Serializing agent's database into the JSON storage
  for (size_t i = 0; i < _stateVector.size(); i++)
  {
    stateJson["Experience Replay"][i]["Episode Id"] = _episodeIdVector[i];
    stateJson["Experience Replay"][i]["Episode Pos"] = _episodePosVector[i];
    stateJson["Experience Replay"][i]["State"] = _stateVector[i];
    stateJson["Experience Replay"][i]["Action"] = _actionVector[i];
    stateJson["Experience Replay"][i]["Reward"] = _rewardVector[i];
    stateJson["Experience Replay"][i]["State Value"] = _stateValueVector[i];
    stateJson["Experience Replay"][i]["Retrace Value"] = _retraceValueVector[i];
    stateJson["Experience Replay"][i]["Importance Weight"] = _importanceWeightVector[i];
    stateJson["Experience Replay"][i]["Truncated Importance Weight"] = _truncatedImportanceWeightVector[i];
    stateJson["Experience Replay"][i]["Is On Policy"] = _isOnPolicyVector[i];
    stateJson["Experience Replay"][i]["Truncated State"] = _truncatedStateVector[i];
    stateJson["Experience Replay"][i]["Truncated State Value"] = _truncatedStateValueVector[i];
    stateJson["Experience Replay"][i]["Termination"] = _terminationVector[i];

    stateJson["Experience Replay"][i]["Experience Policy"]["State Value"] = _expPolicyVector[i].stateValue;
    stateJson["Experience Replay"][i]["Experience Policy"]["Distribution Parameters"] = _expPolicyVector[i].distributionParameters;
    stateJson["Experience Replay"][i]["Experience Policy"]["Action Index"] = _expPolicyVector[i].actionIndex;
    stateJson["Experience Replay"][i]["Experience Policy"]["Unbounded Action"] = _expPolicyVector[i].unboundedAction;

    stateJson["Experience Replay"][i]["Current Policy"]["State Value"] = _curPolicyVector[i].stateValue;
    stateJson["Experience Replay"][i]["Current Policy"]["Distribution Parameters"] = _curPolicyVector[i].distributionParameters;
    stateJson["Experience Replay"][i]["Current Policy"]["Action Index"] = _curPolicyVector[i].actionIndex;
  }

  // Storing training/testing policies
  stateJson["Training"]["Current Policy"] = _trainingCurrentPolicy;
  stateJson["Training"]["Best Policy"] = _trainingBestPolicy;
  stateJson["Testing"]["Best Policy"] = _testingBestPolicy;

  // If results directory doesn't exist, create it
  if (!dirExists(_k->_fileOutputPath)) mkdir(_k->_fileOutputPath);

  // Resolving file path
  std::string statePath = _k->_fileOutputPath + "/state.json";

  // Storing database to file
  if (saveJsonToFile(statePath.c_str(), stateJson) != 0)
    KORALI_LOG_ERROR("Could not serialize training state into file %s\n", statePath.c_str());

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

  // Loading database from file
  _k->_logger->logInfo("Detailed", "Loading previous run training state from file %s...\n", statePath.c_str());
  if (loadJsonFromFile(stateJson, statePath.c_str()) == false)
    KORALI_LOG_ERROR("Trying to resume training or test policy but could not find or deserialize agent's state from from file %s...\n", statePath.c_str());

  // Clearing existing database
  _stateVector.clear();
  _actionVector.clear();
  _retraceValueVector.clear();
  _rewardVector.clear();
  _stateValueVector.clear();
  _importanceWeightVector.clear();
  _truncatedImportanceWeightVector.clear();
  _truncatedStateValueVector.clear();
  _truncatedStateVector.clear();
  _terminationVector.clear();
  _expPolicyVector.clear();
  _curPolicyVector.clear();
  _isOnPolicyVector.clear();
  _priorityVector.clear();
  _probabilityVector.clear();
  _episodePosVector.clear();
  _episodeIdVector.clear();

  // Deserializing database from JSON to the agent's state
  for (size_t i = 0; i < stateJson["Experience Replay"].size(); i++)
  {
    _episodeIdVector.add(stateJson["Experience Replay"][i]["Episode Id"].get<size_t>());
    _episodePosVector.add(stateJson["Experience Replay"][i]["Episode Pos"].get<size_t>());
    _stateVector.add(stateJson["Experience Replay"][i]["State"].get<std::vector<float>>());
    _actionVector.add(stateJson["Experience Replay"][i]["Action"].get<std::vector<float>>());
    _rewardVector.add(stateJson["Experience Replay"][i]["Reward"].get<float>());
    _stateValueVector.add(stateJson["Experience Replay"][i]["State Value"].get<float>());
    _retraceValueVector.add(stateJson["Experience Replay"][i]["Retrace Value"].get<float>());
    _importanceWeightVector.add(stateJson["Experience Replay"][i]["Importance Weight"].get<float>());
    _truncatedImportanceWeightVector.add(stateJson["Experience Replay"][i]["Truncated Importance Weight"].get<float>());
    _isOnPolicyVector.add(stateJson["Experience Replay"][i]["Is On Policy"].get<bool>());
    _truncatedStateVector.add(stateJson["Experience Replay"][i]["Truncated State"].get<std::vector<float>>());
    _truncatedStateValueVector.add(stateJson["Experience Replay"][i]["Truncated State Value"].get<float>());
    _terminationVector.add(stateJson["Experience Replay"][i]["Termination"].get<termination_t>());

    policy_t expPolicy;
    expPolicy.stateValue = stateJson["Experience Replay"][i]["Experience Policy"]["State Value"].get<float>();
    expPolicy.distributionParameters = stateJson["Experience Replay"][i]["Experience Policy"]["Distribution Parameters"].get<std::vector<float>>();
    expPolicy.unboundedAction = stateJson["Experience Replay"][i]["Experience Policy"]["Unbounded Action"].get<std::vector<float>>();
    expPolicy.actionIndex = stateJson["Experience Replay"][i]["Experience Policy"]["Action Index"].get<size_t>();
    _expPolicyVector.add(expPolicy);

    policy_t curPolicy;
    curPolicy.stateValue = stateJson["Experience Replay"][i]["Current Policy"]["State Value"].get<float>();
    curPolicy.distributionParameters = stateJson["Experience Replay"][i]["Current Policy"]["Distribution Parameters"].get<std::vector<float>>();
    curPolicy.actionIndex = stateJson["Experience Replay"][i]["Current Policy"]["Action Index"].get<size_t>();
    _curPolicyVector.add(curPolicy);
  }

  // Restoring training/testing policies
  _trainingCurrentPolicy = stateJson["Training"]["Current Policy"];
  _trainingBestPolicy = stateJson["Training"]["Best Policy"];
  _testingBestPolicy = stateJson["Testing"]["Best Policy"];

  // Setting current agent's training state
  setAgentPolicy(_trainingCurrentPolicy);

  // Resetting the optimizers that the algorithm might be using
  resetAgentOptimizers();

  auto endTime = std::chrono::steady_clock::now();                                                                         // Profiling
  double deserializationTime = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - beginTime).count() / 1.0e+9; // Profiling
  _k->_logger->logInfo("Detailed", "Took %fs to deserialize training state.\n", deserializationTime);
}

void Agent::printGenerationAfter()
{
  if (_mode == "Training")
  {
    _k->_logger->logInfo("Normal", "Replay Experience Statistics:\n");

    _k->_logger->logInfo("Normal", " + Experience Memory Size:      %lu/%lu\n", _stateVector.size(), _experienceReplayMaximumSize);

    if (_maxEpisodes > 0)
      _k->_logger->logInfo("Normal", " + Total Episodes Count:        %lu/%lu\n", _currentEpisode, _maxEpisodes);
    else
      _k->_logger->logInfo("Normal", " + Total Episodes Count:        %lu\n", _currentEpisode);

    if (_maxExperiences > 0)
      _k->_logger->logInfo("Normal", " + Total Experience Count:      %lu/%lu\n", _experienceCount, _maxExperiences);
    else
      _k->_logger->logInfo("Normal", " + Total Experience Count:      %lu\n", _experienceCount);

    if (_rewardOutboundPenalizationEnabled == true)
      _k->_logger->logInfo("Normal", " + Out of Bound Actions:        %lu (%.3f%%)\n", _rewardOutboundPenalizationCount, 100.0f * (float)_rewardOutboundPenalizationEnabled / (float)_experienceCount);

    _k->_logger->logInfo("Normal", "Off-Policy Statistics:\n");
    _k->_logger->logInfo("Normal", " + Count (Ratio/Target):        %lu/%lu (%.3f/%.3f)\n", _experienceReplayOffPolicyCount, _stateVector.size(), _experienceReplayOffPolicyRatio, _experienceReplayOffPolicyTarget);
    _k->_logger->logInfo("Normal", " + Importance Weight Cutoff:    [%.3f, %.3f]\n", 1.0f / _experienceReplayOffPolicyCurrentCutoff, _experienceReplayOffPolicyCurrentCutoff);
    _k->_logger->logInfo("Normal", " + REFER Beta Factor:           %f\n", _experienceReplayOffPolicyREFERBeta);

    _k->_logger->logInfo("Normal", "Training Statistics:\n");

    if (_maxPolicyUpdates > 0)
      _k->_logger->logInfo("Normal", " + Policy Update Count:         %lu/%lu\n", _policyUpdateCount, _maxPolicyUpdates);
    else
      _k->_logger->logInfo("Normal", " + Policy Update Count:         %lu\n", _policyUpdateCount);

    _k->_logger->logInfo("Normal", " + Latest Reward:               %f/%f\n", _trainingLastReward, _problem->_trainingRewardThreshold);
    _k->_logger->logInfo("Normal", " + %lu-Episode Average Reward:  %f\n", _trainingAverageDepth, _trainingAverageReward);
    _k->_logger->logInfo("Normal", " + Best Reward:                 %f (%lu)\n", _trainingBestReward, _trainingBestEpisodeId);

    if (isinf(_problem->_trainingRewardThreshold) == false)
    {
      _k->_logger->logInfo("Normal", "Testing Statistics:\n");

      _k->_logger->logInfo("Normal", " + Candidate Policies:          %lu\n", _testingCandidateCount);

      _k->_logger->logInfo("Normal", " + Latest Average (Stdev / Worst / Best) Reward: %f (%f / %f / %f)\n", _testingAverageReward, _testingStdevReward, _testingWorstReward, _testingBestReward);
    }

    if (_testingTargetAverageReward > -korali::Inf)
      _k->_logger->logInfo("Normal", " + Best Average Reward: %f/%f (%lu)\n", _testingBestAverageReward, _testingTargetAverageReward, _testingBestEpisodeId);
    else
      _k->_logger->logInfo("Normal", " + Best Average Reward: %f (%lu)\n", _testingBestAverageReward, _testingBestEpisodeId);

    printAgentInformation();
    _k->_logger->logInfo("Normal", " + Current Learning Rate:           %.3e\n", _currentLearningRate);

    if (_rewardRescalingEnabled)
      _k->_logger->logInfo("Normal", " + Reward Rescaling:            N(%.3e, %.3e)         \n", _rewardRescalingMean, _rewardRescalingSigma);

    if (_stateRescalingEnabled)
      _k->_logger->logInfo("Normal", " + Using State Rescaling\n");

    _k->_logger->logInfo("Detailed", "Profiling Information:                  [Generation] - [Session]\n");
    _k->_logger->logInfo("Detailed", " + Experience Serialization Time:       [%5.3fs] - [%3.3fs]\n", _generationSerializationTime / 1.0e+9, _sessionSerializationTime / 1.0e+9);
    _k->_logger->logInfo("Detailed", " + Agent Attending Time:                [%5.3fs] - [%3.3fs]\n", _generationAgentAttendingTime / 1.0e+9, _sessionAgentAttendingTime / 1.0e+9);
    _k->_logger->logInfo("Detailed", " + Avg Agent Computation Time:          [%5.3fs] - [%3.3fs]\n", _generationAgentComputationTime / 1.0e+9, _sessionAgentComputationTime / 1.0e+9);
    _k->_logger->logInfo("Detailed", " + Avg Agent Communication/Wait Time:   [%5.3fs] - [%3.3fs]\n", _generationAgentCommunicationTime / 1.0e+9, _sessionAgentCommunicationTime / 1.0e+9);
    _k->_logger->logInfo("Detailed", " + Avg Agent Policy Evaluation Time:    [%5.3fs] - [%3.3fs]\n", _generationAgentPolicyEvaluationTime / 1.0e+9, _sessionAgentPolicyEvaluationTime / 1.0e+9);
    _k->_logger->logInfo("Detailed", " + Policy Update Time:                  [%5.3fs] - [%3.3fs]\n", _generationPolicyUpdateTime / 1.0e+9, _sessionPolicyUpdateTime / 1.0e+9);
    _k->_logger->logInfo("Detailed", " + Running Time:                        [%5.3fs] - [%3.3fs]\n", _generationRunningTime / 1.0e+9, _sessionRunningTime / 1.0e+9);
    _k->_logger->logInfo("Detailed", " + [I/O] Result File Saving Time:        %5.3fs\n", _k->_resultSavingTime / 1.0e+9);
  }

  if (_mode == "Testing")
  {
    _k->_logger->logInfo("Normal", "Testing Results:\n");
    for (size_t agentId = 0; agentId < _testingSampleIds.size(); agentId++)
    {
      _k->_logger->logInfo("Normal", " + Sample %lu:\n", _testingSampleIds[agentId]);
      _k->_logger->logInfo("Normal", "   + (Average) Cumulative Reward            %f\n", _testingReward[agentId]);
    }
  }
}

void Agent::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

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
 try { _trainingRewardHistory = js["Training"]["Reward History"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Training']['Reward History']\n%s", e.what()); } 
   eraseValue(js, "Training", "Reward History");
 }

 if (isDefined(js, "Training", "Experience History"))
 {
 try { _trainingExperienceHistory = js["Training"]["Experience History"].get<std::vector<size_t>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Training']['Experience History']\n%s", e.what()); } 
   eraseValue(js, "Training", "Experience History");
 }

 if (isDefined(js, "Training", "Average Reward"))
 {
 try { _trainingAverageReward = js["Training"]["Average Reward"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Training']['Average Reward']\n%s", e.what()); } 
   eraseValue(js, "Training", "Average Reward");
 }

 if (isDefined(js, "Training", "Last Reward"))
 {
 try { _trainingLastReward = js["Training"]["Last Reward"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Training']['Last Reward']\n%s", e.what()); } 
   eraseValue(js, "Training", "Last Reward");
 }

 if (isDefined(js, "Training", "Best Reward"))
 {
 try { _trainingBestReward = js["Training"]["Best Reward"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Training']['Best Reward']\n%s", e.what()); } 
   eraseValue(js, "Training", "Best Reward");
 }

 if (isDefined(js, "Training", "Best Episode Id"))
 {
 try { _trainingBestEpisodeId = js["Training"]["Best Episode Id"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Training']['Best Episode Id']\n%s", e.what()); } 
   eraseValue(js, "Training", "Best Episode Id");
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

 if (isDefined(js, "Testing", "Stdev Reward"))
 {
 try { _testingStdevReward = js["Testing"]["Stdev Reward"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Testing']['Stdev Reward']\n%s", e.what()); } 
   eraseValue(js, "Testing", "Stdev Reward");
 }

 if (isDefined(js, "Testing", "Previous Average Reward"))
 {
 try { _testingPreviousAverageReward = js["Testing"]["Previous Average Reward"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Testing']['Previous Average Reward']\n%s", e.what()); } 
   eraseValue(js, "Testing", "Previous Average Reward");
 }

 if (isDefined(js, "Testing", "Best Average Reward"))
 {
 try { _testingBestAverageReward = js["Testing"]["Best Average Reward"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Testing']['Best Average Reward']\n%s", e.what()); } 
   eraseValue(js, "Testing", "Best Average Reward");
 }

 if (isDefined(js, "Experience Replay", "Off Policy", "Count"))
 {
 try { _experienceReplayOffPolicyCount = js["Experience Replay"]["Off Policy"]["Count"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Experience Replay']['Off Policy']['Count']\n%s", e.what()); } 
   eraseValue(js, "Experience Replay", "Off Policy", "Count");
 }

 if (isDefined(js, "Experience Replay", "Off Policy", "Ratio"))
 {
 try { _experienceReplayOffPolicyRatio = js["Experience Replay"]["Off Policy"]["Ratio"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Experience Replay']['Off Policy']['Ratio']\n%s", e.what()); } 
   eraseValue(js, "Experience Replay", "Off Policy", "Ratio");
 }

 if (isDefined(js, "Experience Replay", "Off Policy", "Current Cutoff"))
 {
 try { _experienceReplayOffPolicyCurrentCutoff = js["Experience Replay"]["Off Policy"]["Current Cutoff"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Experience Replay']['Off Policy']['Current Cutoff']\n%s", e.what()); } 
   eraseValue(js, "Experience Replay", "Off Policy", "Current Cutoff");
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

 if (isDefined(js, "Current Sample ID"))
 {
 try { _currentSampleID = js["Current Sample ID"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Current Sample ID']\n%s", e.what()); } 
   eraseValue(js, "Current Sample ID");
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

 if (isDefined(js, "Reward", "Rescaling", "Mean"))
 {
 try { _rewardRescalingMean = js["Reward"]["Rescaling"]["Mean"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Reward']['Rescaling']['Mean']\n%s", e.what()); } 
   eraseValue(js, "Reward", "Rescaling", "Mean");
 }

 if (isDefined(js, "Reward", "Rescaling", "Sigma"))
 {
 try { _rewardRescalingSigma = js["Reward"]["Rescaling"]["Sigma"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Reward']['Rescaling']['Sigma']\n%s", e.what()); } 
   eraseValue(js, "Reward", "Rescaling", "Sigma");
 }

 if (isDefined(js, "Reward", "Rescaling", "Count"))
 {
 try { _rewardRescalingCount = js["Reward"]["Rescaling"]["Count"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Reward']['Rescaling']['Count']\n%s", e.what()); } 
   eraseValue(js, "Reward", "Rescaling", "Count");
 }

 if (isDefined(js, "Reward", "Outbound Penalization", "Count"))
 {
 try { _rewardOutboundPenalizationCount = js["Reward"]["Outbound Penalization"]["Count"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Reward']['Outbound Penalization']['Count']\n%s", e.what()); } 
   eraseValue(js, "Reward", "Outbound Penalization", "Count");
 }

 if (isDefined(js, "State Rescaling", "Means"))
 {
 try { _stateRescalingMeans = js["State Rescaling"]["Means"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['State Rescaling']['Means']\n%s", e.what()); } 
   eraseValue(js, "State Rescaling", "Means");
 }

 if (isDefined(js, "State Rescaling", "Sigmas"))
 {
 try { _stateRescalingSigmas = js["State Rescaling"]["Sigmas"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['State Rescaling']['Sigmas']\n%s", e.what()); } 
   eraseValue(js, "State Rescaling", "Sigmas");
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

 if (isDefined(js, "Testing", "Policy"))
 {
 _testingPolicy = js["Testing"]["Policy"].get<knlohmann::json>();

   eraseValue(js, "Testing", "Policy");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Testing']['Policy'] required by agent.\n"); 

 if (isDefined(js, "Training", "Average Depth"))
 {
 try { _trainingAverageDepth = js["Training"]["Average Depth"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Training']['Average Depth']\n%s", e.what()); } 
   eraseValue(js, "Training", "Average Depth");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Training']['Average Depth'] required by agent.\n"); 

 if (isDefined(js, "Concurrent Environments"))
 {
 try { _concurrentEnvironments = js["Concurrent Environments"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Concurrent Environments']\n%s", e.what()); } 
   eraseValue(js, "Concurrent Environments");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Concurrent Environments'] required by agent.\n"); 

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

 if (isDefined(js, "Mini Batch", "Strategy"))
 {
 try { _miniBatchStrategy = js["Mini Batch"]["Strategy"].get<std::string>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Mini Batch']['Strategy']\n%s", e.what()); } 
{
 bool validOption = false; 
 if (_miniBatchStrategy == "Uniform") validOption = true; 
 if (validOption == false) KORALI_LOG_ERROR(" + Unrecognized value (%s) provided for mandatory setting: ['Mini Batch']['Strategy'] required by agent.\n", _miniBatchStrategy.c_str()); 
}
   eraseValue(js, "Mini Batch", "Strategy");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Mini Batch']['Strategy'] required by agent.\n"); 

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

 if (isDefined(js, "Reward", "Rescaling", "Frequency"))
 {
 try { _rewardRescalingFrequency = js["Reward"]["Rescaling"]["Frequency"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Reward']['Rescaling']['Frequency']\n%s", e.what()); } 
   eraseValue(js, "Reward", "Rescaling", "Frequency");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Reward']['Rescaling']['Frequency'] required by agent.\n"); 

 if (isDefined(js, "Reward", "Outbound Penalization", "Enabled"))
 {
 try { _rewardOutboundPenalizationEnabled = js["Reward"]["Outbound Penalization"]["Enabled"].get<int>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Reward']['Outbound Penalization']['Enabled']\n%s", e.what()); } 
   eraseValue(js, "Reward", "Outbound Penalization", "Enabled");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Reward']['Outbound Penalization']['Enabled'] required by agent.\n"); 

 if (isDefined(js, "Reward", "Outbound Penalization", "Factor"))
 {
 try { _rewardOutboundPenalizationFactor = js["Reward"]["Outbound Penalization"]["Factor"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Reward']['Outbound Penalization']['Factor']\n%s", e.what()); } 
   eraseValue(js, "Reward", "Outbound Penalization", "Factor");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Reward']['Outbound Penalization']['Factor'] required by agent.\n"); 

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

 if (isDefined(js, "Termination Criteria", "Testing", "Target Average Reward"))
 {
 try { _testingTargetAverageReward = js["Termination Criteria"]["Testing"]["Target Average Reward"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Termination Criteria']['Testing']['Target Average Reward']\n%s", e.what()); } 
   eraseValue(js, "Termination Criteria", "Testing", "Target Average Reward");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Termination Criteria']['Testing']['Target Average Reward'] required by agent.\n"); 

 if (isDefined(js, "Termination Criteria", "Testing", "Average Reward Increment"))
 {
 try { _testingAverageRewardIncrement = js["Termination Criteria"]["Testing"]["Average Reward Increment"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ agent ] \n + Key:    ['Termination Criteria']['Testing']['Average Reward Increment']\n%s", e.what()); } 
   eraseValue(js, "Termination Criteria", "Testing", "Average Reward Increment");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Termination Criteria']['Testing']['Average Reward Increment'] required by agent.\n"); 

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
   js["Testing"]["Policy"] = _testingPolicy;
   js["Training"]["Average Depth"] = _trainingAverageDepth;
   js["Concurrent Environments"] = _concurrentEnvironments;
   js["Episodes Per Generation"] = _episodesPerGeneration;
   js["Mini Batch"]["Size"] = _miniBatchSize;
   js["Mini Batch"]["Strategy"] = _miniBatchStrategy;
   js["Time Sequence Length"] = _timeSequenceLength;
   js["Learning Rate"] = _learningRate;
   js["L2 Regularization"]["Enabled"] = _l2RegularizationEnabled;
   js["L2 Regularization"]["Importance"] = _l2RegularizationImportance;
   js["Neural Network"]["Hidden Layers"] = _neuralNetworkHiddenLayers;
   js["Neural Network"]["Optimizer"] = _neuralNetworkOptimizer;
   js["Neural Network"]["Engine"] = _neuralNetworkEngine;
   js["Discount Factor"] = _discountFactor;
   js["Experience Replay"]["Serialize"] = _experienceReplaySerialize;
   js["Experience Replay"]["Start Size"] = _experienceReplayStartSize;
   js["Experience Replay"]["Maximum Size"] = _experienceReplayMaximumSize;
   js["Experience Replay"]["Off Policy"]["Cutoff Scale"] = _experienceReplayOffPolicyCutoffScale;
   js["Experience Replay"]["Off Policy"]["Target"] = _experienceReplayOffPolicyTarget;
   js["Experience Replay"]["Off Policy"]["Annealing Rate"] = _experienceReplayOffPolicyAnnealingRate;
   js["Experience Replay"]["Off Policy"]["REFER Beta"] = _experienceReplayOffPolicyREFERBeta;
   js["Experiences Between Policy Updates"] = _experiencesBetweenPolicyUpdates;
   js["State Rescaling"]["Enabled"] = _stateRescalingEnabled;
   js["Reward"]["Rescaling"]["Enabled"] = _rewardRescalingEnabled;
   js["Reward"]["Rescaling"]["Frequency"] = _rewardRescalingFrequency;
   js["Reward"]["Outbound Penalization"]["Enabled"] = _rewardOutboundPenalizationEnabled;
   js["Reward"]["Outbound Penalization"]["Factor"] = _rewardOutboundPenalizationFactor;
   js["Termination Criteria"]["Max Episodes"] = _maxEpisodes;
   js["Termination Criteria"]["Max Experiences"] = _maxExperiences;
   js["Termination Criteria"]["Testing"]["Target Average Reward"] = _testingTargetAverageReward;
   js["Termination Criteria"]["Testing"]["Average Reward Increment"] = _testingAverageRewardIncrement;
   js["Termination Criteria"]["Max Policy Updates"] = _maxPolicyUpdates;
   js["Action Lower Bounds"] = _actionLowerBounds;
   js["Action Upper Bounds"] = _actionUpperBounds;
   js["Current Episode"] = _currentEpisode;
   js["Training"]["Reward History"] = _trainingRewardHistory;
   js["Training"]["Experience History"] = _trainingExperienceHistory;
   js["Training"]["Average Reward"] = _trainingAverageReward;
   js["Training"]["Last Reward"] = _trainingLastReward;
   js["Training"]["Best Reward"] = _trainingBestReward;
   js["Training"]["Best Episode Id"] = _trainingBestEpisodeId;
   js["Testing"]["Reward"] = _testingReward;
   js["Testing"]["Best Reward"] = _testingBestReward;
   js["Testing"]["Worst Reward"] = _testingWorstReward;
   js["Testing"]["Best Episode Id"] = _testingBestEpisodeId;
   js["Testing"]["Candidate Count"] = _testingCandidateCount;
   js["Testing"]["Average Reward"] = _testingAverageReward;
   js["Testing"]["Stdev Reward"] = _testingStdevReward;
   js["Testing"]["Previous Average Reward"] = _testingPreviousAverageReward;
   js["Testing"]["Best Average Reward"] = _testingBestAverageReward;
   js["Experience Replay"]["Off Policy"]["Count"] = _experienceReplayOffPolicyCount;
   js["Experience Replay"]["Off Policy"]["Ratio"] = _experienceReplayOffPolicyRatio;
   js["Experience Replay"]["Off Policy"]["Current Cutoff"] = _experienceReplayOffPolicyCurrentCutoff;
   js["Current Learning Rate"] = _currentLearningRate;
   js["Policy Update Count"] = _policyUpdateCount;
   js["Current Sample ID"] = _currentSampleID;
 if(_uniformGenerator != NULL) _uniformGenerator->getConfiguration(js["Uniform Generator"]);
   js["Experience Count"] = _experienceCount;
   js["Reward"]["Rescaling"]["Mean"] = _rewardRescalingMean;
   js["Reward"]["Rescaling"]["Sigma"] = _rewardRescalingSigma;
   js["Reward"]["Rescaling"]["Count"] = _rewardRescalingCount;
   js["Reward"]["Outbound Penalization"]["Count"] = _rewardOutboundPenalizationCount;
   js["State Rescaling"]["Means"] = _stateRescalingMeans;
   js["State Rescaling"]["Sigmas"] = _stateRescalingSigmas;
 for (size_t i = 0; i <  _k->_variables.size(); i++) { 
 } 
 Solver::getConfiguration(js);
} 

void Agent::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Episodes Per Generation\": 1, \"Concurrent Environments\": 1, \"Discount Factor\": 0.995, \"Time Sequence Length\": 1, \"State Rescaling\": {\"Enabled\": false}, \"Reward\": {\"Rescaling\": {\"Enabled\": false, \"Frequency\": 1000}, \"Outbound Penalization\": {\"Enabled\": false, \"Factor\": 0.5}}, \"Mini Batch\": {\"Strategy\": \"Uniform\", \"Size\": 256}, \"L2 Regularization\": {\"Enabled\": false, \"Importance\": 0.0001}, \"Training\": {\"Average Depth\": 100}, \"Testing\": {\"Sample Ids\": [], \"Policy\": {}}, \"Termination Criteria\": {\"Max Episodes\": 0, \"Max Experiences\": 0, \"Max Policy Updates\": 0, \"Testing\": {\"Target Average Reward\": -Infinity, \"Average Reward Increment\": 0.0}}, \"Experience Replay\": {\"Serialize\": true, \"Off Policy\": {\"Cutoff Scale\": 4.0, \"Target\": 0.1, \"REFER Beta\": 0.3, \"Annealing Rate\": 0.0}}, \"Uniform Generator\": {\"Type\": \"Univariate/Uniform\", \"Minimum\": 0.0, \"Maximum\": 1.0}}";
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

 if ((_mode == "Training") && (_testingTargetAverageReward > -korali::Inf) && (_testingBestAverageReward >= _testingTargetAverageReward))
 {
  _terminationCriteria.push_back("agent['Testing']['Target Average Reward'] = " + std::to_string(_testingTargetAverageReward) + ".");
  hasFinished = true;
 }

 if ((_mode == "Training") && (_testingAverageRewardIncrement > 0.0) && (_testingPreviousAverageReward > -korali::Inf) && (_testingAverageReward + _testingStdevReward * _testingAverageRewardIncrement < _testingPreviousAverageReward))
 {
  _terminationCriteria.push_back("agent['Testing']['Average Reward Increment'] = " + std::to_string(_testingAverageRewardIncrement) + ".");
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



} //solver
} //korali

