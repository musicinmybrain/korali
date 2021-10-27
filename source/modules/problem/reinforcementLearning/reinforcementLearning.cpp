#include "engine.hpp"
#include "modules/problem/reinforcementLearning/reinforcementLearning.hpp"
#include "modules/solver/agent/agent.hpp"
#include "sample/sample.hpp"

namespace korali
{
namespace problem
{
;

/**
 * @brief Pointer to the current agent, it is immediately copied as to avoid concurrency problems
 */
Sample *__currentSample;

/**
 * @brief Identifier of the current environment function Id.
 */
size_t __envFunctionId;

/**
 * @brief Pointer to the agent (Korali solver module)
 */
solver::Agent *_agent;

/**
 * @brief Pointer to the engine's conduit
 */
Conduit *_conduit;

/**
 * @brief Stores the environment thread (coroutine).
 */
cothread_t _envThread;

/**
 * @brief Stores the current launch Id for the current sample
 */
size_t _launchId;

void ReinforcementLearning::initialize()
{
  // Processing state/action variable configuration
  _stateVectorIndexes.clear();
  _actionVectorIndexes.clear();
  for (size_t i = 0; i < _k->_variables.size(); i++)
  {
    if (_k->_variables[i]->_type == "State") _stateVectorIndexes.push_back(i);
    if (_k->_variables[i]->_type == "Action") _actionVectorIndexes.push_back(i);
  }

  _actionVectorSize = _actionVectorIndexes.size();
  _stateVectorSize = _stateVectorIndexes.size();

  if (_actionVectorSize == 0) KORALI_LOG_ERROR("No action variables have been defined.\n");
  if (_stateVectorSize == 0) KORALI_LOG_ERROR("No state variables have been defined.\n");

  // Validating observations
  _numberObservedTrajectories = _observationsStates.size();
  if (_numberObservedTrajectories == 0) KORALI_LOG_ERROR("No states have been recorded.\n");
  if (_observationsActions.size() == 0) KORALI_LOG_ERROR("No actions have been recorded.\n");
  if (_observationsFeatures.size() == 0) KORALI_LOG_ERROR("No features have been recorded.\n");
  if (_observationsStates[0].size() == 0) KORALI_LOG_ERROR("Observed states empty.\n");
  if (_observationsActions[0].size() == 0) KORALI_LOG_ERROR("Observed actions empty.\n");
  if (_observationsFeatures[0].size() == 0) KORALI_LOG_ERROR("Observed features empty.\n");

  _featureVectorSize = _observationsFeatures[0][0].size();
  if (_featureVectorSize == 0) KORALI_LOG_ERROR("No features have been defined.\n");

  if (_observationsActions.size() != _numberObservedTrajectories)
    KORALI_LOG_ERROR("Number of trajectories mismatch between observed states and observed actions.\n");

  if (_observationsFeatures.size() != _numberObservedTrajectories)
    KORALI_LOG_ERROR("Number of trajectories mismacht between observed states and observed features.\n");

  _totalObservedStateActionPairs = 0;
  for (size_t t = 0; t < _numberObservedTrajectories; ++t)
  {
    size_t trajectoryLength = _observationsStates[t].size();
    _totalObservedStateActionPairs += trajectoryLength;

    for (size_t i = 0; i < trajectoryLength; ++i)
    {
      if (_observationsStates[t][i].size() != _stateVectorSize)
        KORALI_LOG_ERROR("Dimension of observed state (trajectory %zu index %zu) does not agree with problem configuration.\n", t, i);
    }

    if (_observationsActions[t].size() != trajectoryLength)
      KORALI_LOG_ERROR("Trajectory (%zu) length of observed actions (%zu) does not agree with trajectory length of observed states (%zu)\n", t, _observationsActions[t].size(), trajectoryLength);

    for (size_t i = 0; i < trajectoryLength; ++i)
    {
      if (_observationsActions[t][i].size() != _actionVectorSize)
        KORALI_LOG_ERROR("Dimension of observed action (%zu) does not agree with problem configuration (trajectory %zu index %zu).\n", _observationsActions[t][i].size(), t, i);
    }

    if (_observationsFeatures[t].size() != trajectoryLength)
      KORALI_LOG_ERROR("Trajectory length of observed features (trajectory %zu) does not agree with observed states\n", t);

    for (size_t i = 0; i < trajectoryLength; ++i)
    {
      if (_observationsFeatures[t][i].size() != _featureVectorSize)
        KORALI_LOG_ERROR("Dimension (%zu) of observed features (trajectory %zu index %zu) does not agree with problem configuration.\n", _observationsFeatures[t][i].size(), t, i, _featureVectorSize);
    }
  }

  // Setting initial launch id (0)
  _launchId = 0;
}

/**
 * @brief Thread wrapper to run an environment
 */
void __environmentWrapper()
{
  Sample *agent = __currentSample;

  // Setting and increasing agent's launch Id
  (*agent)["Launch Id"] = _launchId++;
  agent->run(__envFunctionId);

  // If this is not the leader rank within the worker group, return immediately without checking termination state
  if (_conduit->isWorkerLeadRank() == false) return;

  if ((*agent)["Termination"] == "Non Terminal") KORALI_LOG_ERROR("Environment function terminated, but agent termination status (success or truncated) was not set.\n");

  bool terminationRecognized = false;
  if ((*agent)["Termination"] == "Terminal") terminationRecognized = true;
  if ((*agent)["Termination"] == "Truncated") terminationRecognized = true;

  if (terminationRecognized == false) KORALI_LOG_ERROR("Environment function terminated, but agent termination status (%s) is neither 'Terminal' nor 'Truncated'.\n", (*agent)["Termination"].get<std::string>().c_str());

  co_switch(agent->_workerThread);

  KORALI_LOG_ERROR("Resuming a finished agent\n");
}

void ReinforcementLearning::runTrainingEpisode(Sample &agent)
{
  printf("XX\n");
  // Profiling information - Computation and communication time taken by the agent
  _agentPolicyEvaluationTime = 0.0;
  _agentComputationTime = 0.0;
  _agentCommunicationTime = 0.0;

  // Initializing environment configuration
  initializeEnvironment(agent);

  // Counter for the total number of actions taken
  size_t actionCount = 0;

  // Setting mode to traing to add exploratory noise or random actions
  agent["Mode"] = "Training";

  // Reserving message storage for sending back the episodes
  knlohmann::json episodes;

  // Storage to keep track of cumulative reward
  std::vector<float> trainingRewards(_agentsPerEnvironment, 0.0);

  // Setting termination status of initial state (and the following ones) to non terminal.
  // The environment will change this at the last state, indicating whether the episodes was
  // "Success" or "Truncated".
  agent["Termination"] = "Non Terminal";

  // Setting standard value for environment Id
  agent["Environment Id"] = 0;

  // Getting first state
  runEnvironment(agent);

  // If this is not the leader rank within the worker group, return immediately
  if (_k->_engine->_conduit->isWorkerLeadRank() == false)
  {
    finalizeEnvironment();
    return;
  }

  printf("X\n");
  // Get environment iId value from agent
  auto environmentId = KORALI_GET(size_t, agent, "Environment Id");

  // Check whether the env id provided does not exceed the maximum specified
  if (environmentId >= _environmentCount) KORALI_LOG_ERROR("Environment Id provided (%lu) exceeds the maximum environment count defined (>= %lu).\n", environmentId, _environmentCount);

  // Store the current environment Id in the experience
  for (size_t i = 0; i < _agentsPerEnvironment; i++) episodes[i]["Environment Id"] = environmentId;

  // Saving experiences
  while (agent["Termination"] == "Non Terminal")
  {
    printf("Y\n");
    // Generating new action from the agent's policy
    getAction(agent);

    // Store the current state in the experience
    for (size_t i = 0; i < _agentsPerEnvironment; i++)
      episodes[i]["Experiences"][actionCount]["State"] = agent["State"][i];

    // Storing the current action
    for (size_t i = 0; i < _agentsPerEnvironment; i++)
      episodes[i]["Experiences"][actionCount]["Action"] = agent["Action"][i];

    printf("Z\n");
    // Storing the experience's policy
    for (size_t i = 0; i < _agentsPerEnvironment; i++)
      episodes[i]["Experiences"][actionCount]["Policy"] = agent["Policy"][i];

    printf("A\n");
    // Sanity checks for the features
    for (size_t i = 0; i < _agentsPerEnvironment; i++)
      for (size_t j = 0; j < _featureVectorSize; j++)
        if (std::isfinite(agent["Features"][i][j].get<float>()) == false)
          KORALI_LOG_ERROR("Feature variable %lu returned an invalid value: %f\n", agent["Features"][i][j].get<float>());

    printf("B\n");

    // Storing features of the reward function
    for (size_t i = 0; i < _agentsPerEnvironment; i++)
      episodes[i]["Experiences"][actionCount]["Features"] = agent["Features"][i];

    // Sanity check for reward
    for (size_t i = 0; i < _agentsPerEnvironment; i++)
    {
      episodes[i]["Experiences"][actionCount]["Reward"] = _agent->calculateReward(agent["Features"][i].get<std::vector<float>>());
      if (std::isfinite(episodes[i]["Experiences"][actionCount]["Reward"].get<float>()) == false)
        KORALI_LOG_ERROR("Environment reward returned an invalid value: %f\n", episodes[i]["Experiences"][actionCount]["Reward"].get<float>());
    }

    printf("C\n");
    // If single agent, put action into a single vector
    // In case of this being a single agent, support returning state as only vector
    if (_agentsPerEnvironment == 1) agent["Action"] = agent["Action"][0].get<std::vector<float>>();

    // Jumping back into the agent's environment
    runEnvironment(agent);

    // Storing termination status
    for (size_t i = 0; i < _agentsPerEnvironment; i++)
      episodes[i]["Experiences"][actionCount]["Termination"] = agent["Termination"];

    // If the episodes was truncated, then save the terminal state
    if (agent["Termination"] == "Truncated")
      for (size_t i = 0; i < _agentsPerEnvironment; i++)
        episodes[i]["Experiences"][actionCount]["Truncated State"] = agent["State"][i];

    // Adding to cumulative training rewards
    for (size_t i = 0; i < _agentsPerEnvironment; i++)
      trainingRewards[i] += _agent->calculateReward(agent["Features"][i].get<std::vector<float>>());

    // Increasing counter for generated actions
    actionCount++;

    printf("D\n");
    // Checking if we requested the given number of actions in between policy updates and it is not a terminal state
    if ((_actionsBetweenPolicyUpdates > 0) &&
        (agent["Termination"] == "Non Terminal") &&
        (actionCount % _actionsBetweenPolicyUpdates == 0)) requestNewPolicy(agent);
  }

  // Setting cumulative reward
  agent["Training Rewards"] = trainingRewards;

  // Sending last experience last (after testing)
  // This is important to prevent the engine for block-waiting for the return of the sample
  // while the testing runs are being performed.
  knlohmann::json message;
  message["Action"] = "Send Episodes";
  message["Sample Id"] = agent["Sample Id"];
  message["Episodes"] = episodes;
  KORALI_SEND_MSG_TO_ENGINE(message);

  printf("E\n");
  // Finalizing Environment
  finalizeEnvironment();

  // Adding profiling information to agent
  agent["Computation Time"] = _agentComputationTime;
  agent["Communication Time"] = _agentCommunicationTime;
  agent["Policy Evaluation Time"] = _agentPolicyEvaluationTime;
}

void ReinforcementLearning::runTestingEpisode(Sample &agent)
{
  std::vector<float> testingRewards(_agentsPerEnvironment, 0.0);

  // Initializing Environment
  initializeEnvironment(agent);

  // Setting mode to testing to prevent the addition of noise or random actions
  agent["Mode"] = "Testing";

  // Setting initial non terminal state
  agent["Termination"] = "Non Terminal";

  // Getting first state
  runEnvironment(agent);

  // If this is not the leader rank within the worker group, return immediately
  if (_k->_engine->_conduit->isWorkerLeadRank() == false)
  {
    finalizeEnvironment();
    return;
  }

  // Running environment using the last policy only
  while (agent["Termination"] == "Non Terminal")
  {
    getAction(agent);

    // If single agent, put action into a single vector
    // In case of this being a single agent, support returning state as only vector
    if (_agentsPerEnvironment == 1) agent["Action"] = agent["Action"][0].get<std::vector<float>>();

    runEnvironment(agent);

    for (size_t i = 0; i < _agentsPerEnvironment; i++)
      testingRewards[i] += agent["Reward"][i].get<float>();
  }

  // Calculating average reward between testing episodes
  float rewardSum = 0.0f;
  for (size_t i = 0; i < _agentsPerEnvironment; i++)
    rewardSum += testingRewards[i];

  // Storing the average cumulative reward of the testing episode
  agent["Testing Reward"] = rewardSum / _agentsPerEnvironment;

  // Finalizing Environment
  finalizeEnvironment();
}

void ReinforcementLearning::initializeEnvironment(Sample &agent)
{
  // Getting RL-compatible solver
  _agent = dynamic_cast<solver::Agent *>(_k->_solver);

  // Getting agent's conduit
  _conduit = _agent->_k->_engine->_conduit;

  // First, we update the initial policy's hyperparameters
  _agent->setAgentPolicy(agent["Policy Hyperparameters"]);

  // Then, we reset the state sequence for time-dependent learners
  _agent->resetTimeSequence();

  // Define state rescaling variables
  _stateRescalingMeans = agent["State Rescaling"]["Means"].get<std::vector<float>>();
  _stateRescalingSdevs = agent["State Rescaling"]["Standard Deviations"].get<std::vector<float>>();

  // Appending any user-defined settings
  agent["Custom Settings"] = _customSettings;

  // Creating agent coroutine
  __currentSample = &agent;
  __envFunctionId = _environmentFunction;
  agent._workerThread = co_active();

  // Creating coroutine
  _envThread = co_create(1 << 28, __environmentWrapper);

  // Initializing rewards
  if (_agentsPerEnvironment == 1) agent["Reward"] = 0.0f;
  if (_agentsPerEnvironment > 1) agent["Reward"] = std::vector<float>(_agentsPerEnvironment, 0.0f);
}

void ReinforcementLearning::finalizeEnvironment()
{
  // Freeing training co-routine memory
  co_delete(_envThread);
}

void ReinforcementLearning::requestNewPolicy(Sample &agent)
{
  auto t0 = std::chrono::steady_clock::now(); // Profiling

  // Reserving message storage for requesting new policy
  knlohmann::json message;

  // Sending request to engine
  message["Sample Id"] = agent["Sample Id"];
  message["Action"] = "Request New Policy";
  KORALI_SEND_MSG_TO_ENGINE(message);

  // If requested new policy, wait for incoming message containing new hyperparameters
  agent["Policy Hyperparameters"] = KORALI_RECV_MSG_FROM_ENGINE();
  _agent->setAgentPolicy(agent["Policy Hyperparameters"]);

  auto t1 = std::chrono::steady_clock::now();                                                       // Profiling
  _agentCommunicationTime += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(); // Profiling
}

void ReinforcementLearning::getAction(Sample &agent)
{
  // Generating new action from policy
  auto t0 = std::chrono::steady_clock::now(); // Profiling

  _agent->getAction(agent);

  auto t1 = std::chrono::steady_clock::now();                                                          // Profiling
  _agentPolicyEvaluationTime += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(); // Profiling
}

void ReinforcementLearning::runEnvironment(Sample &agent)
{
  // Switching back to the environment's thread
  auto beginTime = std::chrono::steady_clock::now(); // Profiling
  co_switch(_envThread);
  auto endTime = std::chrono::steady_clock::now();                                                            // Profiling
  _agentComputationTime += std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - beginTime).count(); // Profiling

  // If this is not the leader rank within the worker group, return immediately
  if (_conduit->isWorkerLeadRank() == false) return;

  // In case of this being a single agent, support returning state as only vector
  if (_agentsPerEnvironment == 1)
  {
    auto state = KORALI_GET(std::vector<float>, agent, "State");
    agent._js.getJson().erase("State");
    agent["State"][0] = state;
    
    auto features = KORALI_GET(std::vector<float>, agent, "Features");
    agent._js.getJson().erase("Features");
    agent["Features"][0] = features;
  }

  // Checking correct format of state
  if (agent["State"].is_array() == false) KORALI_LOG_ERROR("Agent state variable returned by the environment is not a vector.\n");
  if (agent["State"].size() != _agentsPerEnvironment) KORALI_LOG_ERROR("Agents state vector returned with the wrong size: %lu, expected: %lu.\n", agent["State"].size(), _agentsPerEnvironment);

  // Sanity checks for state
  for (size_t i = 0; i < _agentsPerEnvironment; i++)
  {
    if (agent["State"][i].is_array() == false) KORALI_LOG_ERROR("Agent state variable returned by the environment is not a vector.\n");
    if (agent["State"][i].size() != _stateVectorSize) KORALI_LOG_ERROR("Agents state vector %lu returned with the wrong size: %lu, expected: %lu.\n", i, agent["State"][i].size(), _stateVectorSize);

    for (size_t j = 0; j < _stateVectorSize; j++)
      if (std::isfinite(agent["State"][i][j].get<float>()) == false) KORALI_LOG_ERROR("Agent %lu state variable %lu returned an invalid value: %f\n", i, j, agent["State"][i][j].get<float>());
  }

  // Normalizing State
  for (size_t i = 0; i < _agentsPerEnvironment; i++)
  {
    auto state = agent["State"][i].get<std::vector<float>>();

    // Scale the state
    for (size_t d = 0; d < _stateVectorSize; ++d)
      state[d] = (state[d] - _stateRescalingMeans[d]) / _stateRescalingSdevs[d];

    // Re-storing state into agent
    agent["State"][i] = state;
  }

  // Parsing reward
  if (_agentsPerEnvironment == 1)
  {
    auto reward = KORALI_GET(float, agent, "Reward");
    agent._js.getJson().erase("Reward");
    agent["Reward"][0] = reward;
  }

  // Checking correct format of reward
  if (agent["Reward"].is_array() == false) KORALI_LOG_ERROR("Agent reward variable returned by the environment is not a vector.\n");
  if (agent["Reward"].size() != _agentsPerEnvironment) KORALI_LOG_ERROR("Agents reward vector returned with the wrong size: %lu, expected: %lu.\n", agent["Reward"].size(), _agentsPerEnvironment);

  // Sanity checks for reward
  for (size_t i = 0; i < _agentsPerEnvironment; i++)
    if (std::isfinite(agent["Reward"][i].get<float>()) == false) KORALI_LOG_ERROR("Agent %lu reward returned an invalid value: %f\n", i, agent["Reward"][i].get<float>());
}

void ReinforcementLearning::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Action Vector Size"))
 {
 try { _actionVectorSize = js["Action Vector Size"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ reinforcementLearning ] \n + Key:    ['Action Vector Size']\n%s", e.what()); } 
   eraseValue(js, "Action Vector Size");
 }

 if (isDefined(js, "State Vector Size"))
 {
 try { _stateVectorSize = js["State Vector Size"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ reinforcementLearning ] \n + Key:    ['State Vector Size']\n%s", e.what()); } 
   eraseValue(js, "State Vector Size");
 }

 if (isDefined(js, "Action Vector Indexes"))
 {
 try { _actionVectorIndexes = js["Action Vector Indexes"].get<std::vector<size_t>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ reinforcementLearning ] \n + Key:    ['Action Vector Indexes']\n%s", e.what()); } 
   eraseValue(js, "Action Vector Indexes");
 }

 if (isDefined(js, "State Vector Indexes"))
 {
 try { _stateVectorIndexes = js["State Vector Indexes"].get<std::vector<size_t>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ reinforcementLearning ] \n + Key:    ['State Vector Indexes']\n%s", e.what()); } 
   eraseValue(js, "State Vector Indexes");
 }

 if (isDefined(js, "Number Observed Trajectories"))
 {
 try { _numberObservedTrajectories = js["Number Observed Trajectories"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ reinforcementLearning ] \n + Key:    ['Number Observed Trajectories']\n%s", e.what()); } 
   eraseValue(js, "Number Observed Trajectories");
 }

 if (isDefined(js, "Feature Vector Size"))
 {
 try { _featureVectorSize = js["Feature Vector Size"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ reinforcementLearning ] \n + Key:    ['Feature Vector Size']\n%s", e.what()); } 
   eraseValue(js, "Feature Vector Size");
 }

 if (isDefined(js, "Total Observed State Action Pairs"))
 {
 try { _totalObservedStateActionPairs = js["Total Observed State Action Pairs"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ reinforcementLearning ] \n + Key:    ['Total Observed State Action Pairs']\n%s", e.what()); } 
   eraseValue(js, "Total Observed State Action Pairs");
 }

 if (isDefined(js, "Agents Per Environment"))
 {
 try { _agentsPerEnvironment = js["Agents Per Environment"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ reinforcementLearning ] \n + Key:    ['Agents Per Environment']\n%s", e.what()); } 
   eraseValue(js, "Agents Per Environment");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Agents Per Environment'] required by reinforcementLearning.\n"); 

 if (isDefined(js, "Environment Count"))
 {
 try { _environmentCount = js["Environment Count"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ reinforcementLearning ] \n + Key:    ['Environment Count']\n%s", e.what()); } 
   eraseValue(js, "Environment Count");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Environment Count'] required by reinforcementLearning.\n"); 

 if (isDefined(js, "Environment Function"))
 {
 try { _environmentFunction = js["Environment Function"].get<std::uint64_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ reinforcementLearning ] \n + Key:    ['Environment Function']\n%s", e.what()); } 
   eraseValue(js, "Environment Function");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Environment Function'] required by reinforcementLearning.\n"); 

 if (isDefined(js, "Actions Between Policy Updates"))
 {
 try { _actionsBetweenPolicyUpdates = js["Actions Between Policy Updates"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ reinforcementLearning ] \n + Key:    ['Actions Between Policy Updates']\n%s", e.what()); } 
   eraseValue(js, "Actions Between Policy Updates");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Actions Between Policy Updates'] required by reinforcementLearning.\n"); 

 if (isDefined(js, "Custom Settings"))
 {
 _customSettings = js["Custom Settings"].get<knlohmann::json>();

   eraseValue(js, "Custom Settings");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Custom Settings'] required by reinforcementLearning.\n"); 

 if (isDefined(js, "Observations", "States"))
 {
 try { _observationsStates = js["Observations"]["States"].get<std::vector<std::vector<std::vector<float>>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ reinforcementLearning ] \n + Key:    ['Observations']['States']\n%s", e.what()); } 
   eraseValue(js, "Observations", "States");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Observations']['States'] required by reinforcementLearning.\n"); 

 if (isDefined(js, "Observations", "Actions"))
 {
 try { _observationsActions = js["Observations"]["Actions"].get<std::vector<std::vector<std::vector<float>>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ reinforcementLearning ] \n + Key:    ['Observations']['Actions']\n%s", e.what()); } 
   eraseValue(js, "Observations", "Actions");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Observations']['Actions'] required by reinforcementLearning.\n"); 

 if (isDefined(js, "Observations", "Features"))
 {
 try { _observationsFeatures = js["Observations"]["Features"].get<std::vector<std::vector<std::vector<float>>>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ reinforcementLearning ] \n + Key:    ['Observations']['Features']\n%s", e.what()); } 
   eraseValue(js, "Observations", "Features");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Observations']['Features'] required by reinforcementLearning.\n"); 

 if (isDefined(_k->_js.getJson(), "Variables"))
 for (size_t i = 0; i < _k->_js["Variables"].size(); i++) { 
 if (isDefined(_k->_js["Variables"][i], "Type"))
 {
 try { _k->_variables[i]->_type = _k->_js["Variables"][i]["Type"].get<std::string>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ reinforcementLearning ] \n + Key:    ['Type']\n%s", e.what()); } 
{
 bool validOption = false; 
 if (_k->_variables[i]->_type == "State") validOption = true; 
 if (_k->_variables[i]->_type == "Action") validOption = true; 
 if (validOption == false) KORALI_LOG_ERROR(" + Unrecognized value (%s) provided for mandatory setting: ['Type'] required by reinforcementLearning.\n", _k->_variables[i]->_type.c_str()); 
}
   eraseValue(_k->_js["Variables"][i], "Type");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Type'] required by reinforcementLearning.\n"); 

 if (isDefined(_k->_js["Variables"][i], "Lower Bound"))
 {
 try { _k->_variables[i]->_lowerBound = _k->_js["Variables"][i]["Lower Bound"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ reinforcementLearning ] \n + Key:    ['Lower Bound']\n%s", e.what()); } 
   eraseValue(_k->_js["Variables"][i], "Lower Bound");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Lower Bound'] required by reinforcementLearning.\n"); 

 if (isDefined(_k->_js["Variables"][i], "Upper Bound"))
 {
 try { _k->_variables[i]->_upperBound = _k->_js["Variables"][i]["Upper Bound"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ reinforcementLearning ] \n + Key:    ['Upper Bound']\n%s", e.what()); } 
   eraseValue(_k->_js["Variables"][i], "Upper Bound");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Upper Bound'] required by reinforcementLearning.\n"); 

 } 
  bool detectedCompatibleSolver = false; 
  std::string solverName = toLower(_k->_js["Solver"]["Type"]); 
  std::string candidateSolverName; 
  solverName.erase(remove_if(solverName.begin(), solverName.end(), isspace), solverName.end()); 
   candidateSolverName = toLower("Agent"); 
   candidateSolverName.erase(remove_if(candidateSolverName.begin(), candidateSolverName.end(), isspace), candidateSolverName.end()); 
   if (solverName.rfind(candidateSolverName, 0) == 0) detectedCompatibleSolver = true;
  if (detectedCompatibleSolver == false) KORALI_LOG_ERROR(" + Specified solver (%s) is not compatible with problem of type: reinforcementLearning\n",  _k->_js["Solver"]["Type"].dump(1).c_str()); 

 Problem::setConfiguration(js);
 _type = "reinforcementLearning";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: reinforcementLearning: \n%s\n", js.dump(2).c_str());
} 

void ReinforcementLearning::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Agents Per Environment"] = _agentsPerEnvironment;
   js["Environment Count"] = _environmentCount;
   js["Environment Function"] = _environmentFunction;
   js["Actions Between Policy Updates"] = _actionsBetweenPolicyUpdates;
   js["Custom Settings"] = _customSettings;
   js["Observations"]["States"] = _observationsStates;
   js["Observations"]["Actions"] = _observationsActions;
   js["Observations"]["Features"] = _observationsFeatures;
   js["Action Vector Size"] = _actionVectorSize;
   js["State Vector Size"] = _stateVectorSize;
   js["Action Vector Indexes"] = _actionVectorIndexes;
   js["State Vector Indexes"] = _stateVectorIndexes;
   js["Number Observed Trajectories"] = _numberObservedTrajectories;
   js["Feature Vector Size"] = _featureVectorSize;
   js["Total Observed State Action Pairs"] = _totalObservedStateActionPairs;
 for (size_t i = 0; i <  _k->_variables.size(); i++) { 
   _k->_js["Variables"][i]["Type"] = _k->_variables[i]->_type;
   _k->_js["Variables"][i]["Lower Bound"] = _k->_variables[i]->_lowerBound;
   _k->_js["Variables"][i]["Upper Bound"] = _k->_variables[i]->_upperBound;
 } 
 Problem::getConfiguration(js);
} 

void ReinforcementLearning::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Agents Per Environment\": 1, \"Environment Count\": 1, \"Actions Between Policy Updates\": 0, \"Custom Settings\": {}}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Problem::applyModuleDefaults(js);
} 

void ReinforcementLearning::applyVariableDefaults() 
{

 std::string defaultString = "{\"Type\": \"State\", \"Lower Bound\": -Infinity, \"Upper Bound\": Infinity}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 if (isDefined(_k->_js.getJson(), "Variables"))
  for (size_t i = 0; i < _k->_js["Variables"].size(); i++) 
   mergeJson(_k->_js["Variables"][i], defaultJs); 
 Problem::applyVariableDefaults();
} 

bool ReinforcementLearning::runOperation(std::string operation, korali::Sample& sample)
{
 bool operationDetected = false;

 if (operation == "Run Training Episode")
 {
  runTrainingEpisode(sample);
  return true;
 }

 if (operation == "Run Testing Episode")
 {
  runTestingEpisode(sample);
  return true;
 }

 operationDetected = operationDetected || Problem::runOperation(operation, sample);
 if (operationDetected == false) KORALI_LOG_ERROR(" + Operation %s not recognized for problem ReinforcementLearning.\n", operation.c_str());
 return operationDetected;
}

;

} //problem
} //korali
;
