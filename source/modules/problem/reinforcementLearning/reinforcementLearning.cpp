#include "engine.hpp"
#include "modules/problem/reinforcementLearning/reinforcementLearning.hpp"
#include "modules/solver/agent/agent.hpp"
#include "sample/sample.hpp"

namespace korali
{
namespace problem
{


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

  if ((*agent)["Termination"] == "Non Terminal")
    KORALI_LOG_ERROR("Environment function terminated, but agent termination status (success or truncated) was not set.\n");

  bool terminationRecognized = false;
  if ((*agent)["Termination"] == "Terminal") terminationRecognized = true;
  if ((*agent)["Termination"] == "Truncated") terminationRecognized = true;

  if (terminationRecognized == false)
    KORALI_LOG_ERROR("Environment function terminated, but agent termination status (%s) is neither 'Terminal' nor 'Truncated'.\n", (*agent)["Termination"].get<std::string>().c_str());

  co_switch(agent->_workerThread);

  KORALI_LOG_ERROR("Resuming a finished agent\n");
}

void ReinforcementLearning::runTrainingEpisode(Sample &agent)
{
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

  // Reserving message storage for sending back the episode
  knlohmann::json episode;

  // Storage to keep track of cumulative reward
  float trainingReward = 0.0;

  // Setting termination status of initial state (and the following ones) to non terminal.
  // The environment will change this at the last state, indicating whether the episode was
  // "Success" or "Truncated".
  agent["Termination"] = "Non Terminal";

  // Getting first state
  runEnvironment(agent);

  // Saving experiences
  while (agent["Termination"] == "Non Terminal")
  {
    // Generating new action from the agent's policy
    getAction(agent);

    // Store the current state in the experience
    episode["Experiences"][actionCount]["State"] = agent["State"];

    // Storing the current action
    episode["Experiences"][actionCount]["Action"] = agent["Action"];

    // Storing the experience's policy
    episode["Experiences"][actionCount]["Policy"] = agent["Policy"];

    // Jumping back into the agent's environment
    runEnvironment(agent);

    // Storing experience's reward
    episode["Experiences"][actionCount]["Reward"] = agent["Reward"];

    // Storing termination status
    episode["Experiences"][actionCount]["Termination"] = agent["Termination"];

    // If the episode was truncated, then save the terminal state
    if (agent["Termination"] == "Truncated") episode["Experiences"][actionCount]["Truncated State"] = agent["State"];

    // Adding to cumulative training reward
    trainingReward += agent["Reward"].get<float>();

    // Increasing counter for generated actions
    actionCount++;

    // Checking if we requested the given number of actions in between policy updates and it is not a terminal state
    if ((_actionsBetweenPolicyUpdates > 0) &&
        (agent["Termination"] == "Non Terminal") &&
        (actionCount % _actionsBetweenPolicyUpdates == 0)) requestNewPolicy(agent);
  }

  // Setting cumulative reward
  agent["Training Reward"] = trainingReward;

  // Finalizing Environment
  finalizeEnvironment();

  // Setting tested policy flag to false, unless we do testing
  agent["Tested Policy"] = false;

  // If the training reward exceeds the threshold or meets the periodic conditions, then also run testing on it
  bool runTest = false;
  runTest |= trainingReward > _trainingRewardThreshold;
  runTest |= (_testingFrequency > 0) && (_k->_currentGeneration % _testingFrequency == 0);
  if (runTest)
  {
    float averageTestingReward = 0.0;
    float stdevTestingReward = 0.0;
    float bestTestingReward = -Inf;
    float worstTestingReward = +Inf;

    for (size_t i = 0; i < _policyTestingEpisodes; i++)
    {
      runTestingEpisode(agent);

      // Getting current testing reward
      float currentTestingReward = agent["Testing Reward"];

      // Adding current testing reward to the average and keeping statistics
      averageTestingReward += currentTestingReward;
      stdevTestingReward += currentTestingReward * currentTestingReward;
      if (currentTestingReward > bestTestingReward) bestTestingReward = currentTestingReward;
      if (currentTestingReward < worstTestingReward) worstTestingReward = currentTestingReward;
    }

    // Normalizing average
    averageTestingReward /= (float)_policyTestingEpisodes;
    stdevTestingReward = std::sqrt(stdevTestingReward / (float)_policyTestingEpisodes - averageTestingReward * averageTestingReward);

    // Storing testing information
    agent["Average Testing Reward"] = averageTestingReward;
    agent["Stdev Testing Reward"] = stdevTestingReward;
    agent["Best Testing Reward"] = bestTestingReward;
    agent["Worst Testing Reward"] = worstTestingReward;

    // Indicate that the agent has been tested
    agent["Tested Policy"] = true;
  }

  // Sending last experience last (after testing)
  // This is important to prevent the engine for block-waiting for the return of the sample
  // while the testing runs are being performed.
  knlohmann::json message;
  message["Action"] = "Send Episode";
  message["Sample Id"] = agent["Sample Id"];
  message["Experiences"] = episode["Experiences"];
  KORALI_SEND_MSG_TO_ENGINE(message);

  // Adding profiling information to agent
  agent["Computation Time"] = _agentComputationTime;
  agent["Communication Time"] = _agentCommunicationTime;
  agent["Policy Evaluation Time"] = _agentPolicyEvaluationTime;
}

void ReinforcementLearning::runTestingEpisode(Sample &agent)
{
  float testingReward = 0.0;

  // Initializing Environment
  initializeEnvironment(agent);

  // Setting mode to testing to prevent the addition of noise or random actions
  agent["Mode"] = "Testing";

  // Setting initial non terminal state
  agent["Termination"] = "Non Terminal";

  // Getting first state
  runEnvironment(agent);

  // Running environment using the last policy only
  while (agent["Termination"] == "Non Terminal")
  {
    getAction(agent);
    runEnvironment(agent);
    testingReward += agent["Reward"].get<float>();
  }

  // Storing the cumulative reward of the testing episode
  agent["Testing Reward"] = testingReward;

  // Finalizing Environment
  finalizeEnvironment();
}

void ReinforcementLearning::initializeEnvironment(Sample &agent)
{
  // Getting RL-compatible solver
  _agent = dynamic_cast<solver::Agent *>(_k->_solver);

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

  // Sanity checks for state
  for (size_t i = 0; i < _stateVectorSize; i++)
    if (std::isfinite(agent["State"][i].get<float>()) == false)
      KORALI_LOG_ERROR("Environment state variable %lu returned an invalid value: %f\n", i, agent["State"][i].get<float>());

  // Normalizing State
  auto state = agent["State"].get<std::vector<float>>();

  // Scale the state
  for (size_t d = 0; d < _stateVectorSize; ++d)
    state[d] = (state[d] - _stateRescalingMeans[d]) / _stateRescalingSdevs[d];

  // Re-storing state into agent
  agent["State"] = state;
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

 if (isDefined(js, "Testing Frequency"))
 {
 try { _testingFrequency = js["Testing Frequency"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ reinforcementLearning ] \n + Key:    ['Testing Frequency']\n%s", e.what()); } 
   eraseValue(js, "Testing Frequency");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Testing Frequency'] required by reinforcementLearning.\n"); 

 if (isDefined(js, "Training Reward Threshold"))
 {
 try { _trainingRewardThreshold = js["Training Reward Threshold"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ reinforcementLearning ] \n + Key:    ['Training Reward Threshold']\n%s", e.what()); } 
   eraseValue(js, "Training Reward Threshold");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Training Reward Threshold'] required by reinforcementLearning.\n"); 

 if (isDefined(js, "Policy Testing Episodes"))
 {
 try { _policyTestingEpisodes = js["Policy Testing Episodes"].get<size_t>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ reinforcementLearning ] \n + Key:    ['Policy Testing Episodes']\n%s", e.what()); } 
   eraseValue(js, "Policy Testing Episodes");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Policy Testing Episodes'] required by reinforcementLearning.\n"); 

 if (isDefined(js, "Custom Settings"))
 {
 _customSettings = js["Custom Settings"].get<knlohmann::json>();

   eraseValue(js, "Custom Settings");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Custom Settings'] required by reinforcementLearning.\n"); 

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
   js["Environment Function"] = _environmentFunction;
   js["Actions Between Policy Updates"] = _actionsBetweenPolicyUpdates;
   js["Testing Frequency"] = _testingFrequency;
   js["Training Reward Threshold"] = _trainingRewardThreshold;
   js["Policy Testing Episodes"] = _policyTestingEpisodes;
   js["Custom Settings"] = _customSettings;
   js["Action Vector Size"] = _actionVectorSize;
   js["State Vector Size"] = _stateVectorSize;
   js["Action Vector Indexes"] = _actionVectorIndexes;
   js["State Vector Indexes"] = _stateVectorIndexes;
 for (size_t i = 0; i <  _k->_variables.size(); i++) { 
   _k->_js["Variables"][i]["Type"] = _k->_variables[i]->_type;
   _k->_js["Variables"][i]["Lower Bound"] = _k->_variables[i]->_lowerBound;
   _k->_js["Variables"][i]["Upper Bound"] = _k->_variables[i]->_upperBound;
 } 
 Problem::getConfiguration(js);
} 

void ReinforcementLearning::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Testing Frequency\": 0, \"Policy Testing Episodes\": 5, \"Actions Between Policy Updates\": 0, \"Custom Settings\": {}}";
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



} //problem
} //korali

