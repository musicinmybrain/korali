#include "engine.hpp"
#include "modules/solver/agent/continuous/VRACER/VRACER.hpp"
#ifdef _OPENMP
  #include "omp.h"
#endif
#include "sample/sample.hpp"

#include <gsl/gsl_sf_psi.h>

namespace korali
{
namespace solver
{
namespace agent
{
namespace continuous
{
;

// Declare reduction clause for vectors
#pragma omp declare reduction(vec_float_plus        \
                              : std::vector <float> \
                              : std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus <float>())) initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

void VRACER::initializeAgent()
{
  // Initializing common discrete agent configuration
  Continuous::initializeAgent();

  _effectiveMinibatchSize = _miniBatchSize * _problem->_agentsPerEnvironment;

  if (_multiAgentRelationship == "Competition")
    _effectiveMinibatchSize = _miniBatchSize;

  // Minibatch statistics
  _miniBatchPolicyMean.resize(_problem->_actionVectorSize);
  _miniBatchPolicyStdDev.resize(_problem->_actionVectorSize);
}

std::vector<std::vector<float>> VRACER::trainPolicy(const std::vector<std::pair<size_t, size_t>> &miniBatch, const std::vector<float>& stateValues, const std::vector<std::vector<float>> &distributionParams)
{
  // Gathering state sequences for selected minibatch
  const auto stateSequenceBatch = getMiniBatchStateSequence(miniBatch);
  if (stateSequenceBatch.size() != distributionParams.size())
    KORALI_LOG_ERROR("Batch size mismatch between state sequence and distribution params");
  if (stateSequenceBatch.size() != stateValues.size())
    KORALI_LOG_ERROR("Batch size mismatch between state sequence and state value");

  // Forward NN
  std::vector<policy_t> policyInfo(miniBatch.size());

  // Assign external distribution params
  for (size_t b = 0; b < miniBatch.size(); ++b)
  {
    policyInfo[b].stateValue = stateValues[b];
    policyInfo[b].distributionParameters.assign(distributionParams[b].begin(), distributionParams[b].end());
  }

  // Using policy information to update experience's metadata
  updateExperienceMetadata(miniBatch, policyInfo);

  // Now calculating policy gradients
  const auto& policyGradient = calculatePolicyGradients(miniBatch, 0);

  return policyGradient;
}

std::vector<std::vector<float>> VRACER::calculatePolicyGradients(const std::vector<std::pair<size_t, size_t>> &miniBatch, const size_t policyIdx)
{
  // Resetting statistics
  std::fill(_miniBatchPolicyMean.begin(), _miniBatchPolicyMean.end(), 0.0);
  std::fill(_miniBatchPolicyStdDev.begin(), _miniBatchPolicyStdDev.end(), 0.0);

  const size_t miniBatchSize = miniBatch.size();

  const size_t numAgents = _problem->_agentsPerEnvironment;

  std::vector<std::vector<float>> gradientPolicyParams(miniBatchSize, std::vector<float>(1 + _policyParameterCount, 0.0f));

#pragma omp parallel for schedule(guided, numAgents) reduction(vec_float_plus \
                                                               : _miniBatchPolicyMean, _miniBatchPolicyStdDev)
  for (size_t b = 0; b < miniBatchSize; b++)
  {
    // Getting index of current experiment
    const size_t expId = miniBatch[b].first;
    const size_t agentId = miniBatch[b].second;

    // Get policy and action for this experience
    const auto &expPolicy = _expPolicyBuffer[expId][agentId];
    const auto &expAction = _actionBuffer[expId][agentId];

    // Gathering metadata
    const auto &stateValue = _stateValueBufferContiguous[expId * numAgents + agentId];
    const auto &curPolicy = _curPolicyBuffer[expId][agentId];
    const auto &expVtbc = _retraceValueBufferContiguous[expId * numAgents + agentId];

    // Gradient of Value Function V(s) (eq. (9); *-1 because the optimizer is maximizing)
    gradientPolicyParams[b][0] = _currentLearningRate*(expVtbc - stateValue);

    // Gradient has to be divided by Number of Agents in Cooperation models
    if (_multiAgentRelationship == "Cooperation")
      gradientPolicyParams[b][0] /= numAgents;

    // Compute policy gradient inside trust region
    if (_isOnPolicyBuffer[expId][agentId])
    {
      // Qret for terminal state is just reward
      float Qret = getScaledReward(_rewardBufferContiguous[expId * numAgents + agentId]);

      // If experience is non-terminal, add Vtbc
      if (_terminationBuffer[expId] == e_nonTerminal)
      {
        const float nextExpVtbc = _retraceValueBufferContiguous[(expId + 1) * numAgents + agentId];
        Qret += _discountFactor * nextExpVtbc;
      }

      // If experience is truncated, add truncated state value
      if (_terminationBuffer[expId] == e_truncated)
      {
        const float nextExpVtbc = _truncatedStateValueBuffer[expId][agentId];
        Qret += _discountFactor * nextExpVtbc;
      }

      // Compute Off-Policy Objective (eq. 5)
      const float lossOffPolicy = Qret - stateValue;

      // Get importance weight
      const auto importanceWeight = _importanceWeightBuffer[expId][agentId];

      // Compute Off-Policy Gradient
      auto polGrad = calculateImportanceWeightGradient(expAction, curPolicy, expPolicy, importanceWeight);

      // Multi-agent correlation implies additional factor
      if (_multiAgentCorrelation)
      {
        const float correlationFactor = _productImportanceWeightBuffer[expId] / _importanceWeightBuffer[expId][agentId];
        for (size_t i = 0; i < polGrad.size(); i++)
          polGrad[i] *= correlationFactor;
      }

      // Set Gradient of Loss wrt Params
      for (size_t i = 0; i < _policyParameterCount; i++)
        gradientPolicyParams[b][1+i] = _currentLearningRate * _experienceReplayOffPolicyREFERCurrentBeta[agentId] * lossOffPolicy * polGrad[i];
    }

    // Compute derivative of KL divergence
    const auto klGrad = calculateKLDivergenceGradient(expPolicy, curPolicy);

    // Compute factor for KL penalization
    const float klGradMultiplier = -_currentLearningRate * (1.0f - _experienceReplayOffPolicyREFERCurrentBeta[agentId]);

    // Add KL contribution
    for (size_t i = 0; i < _problem->_actionVectorSize; i++)
    {
      if (std::isfinite(klGrad[i]) == false)
      {
        KORALI_LOG_ERROR("KL gradient returned an invalid value (%f/%f) (exp policy/cur policy): %f\n", expPolicy.distributionParameters[i], curPolicy.distributionParameters[i], klGrad[i]);
      }

      if (std::isfinite(klGrad[i + _problem->_actionVectorSize]) == false)
      {
        KORALI_LOG_ERROR("KL gradient returned an invalid value (%f/%f) (exp policy/cur policy): %f\n", expPolicy.distributionParameters[i + _problem->_actionVectorSize], curPolicy.distributionParameters[i + _problem->_actionVectorSize], klGrad[i + _problem->_actionVectorSize]);
      }

      gradientPolicyParams[b][1+i] += klGradMultiplier * klGrad[i];
      gradientPolicyParams[b][1+i + _problem->_actionVectorSize] += klGradMultiplier * klGrad[i + _problem->_actionVectorSize];

      if (std::isfinite(gradientPolicyParams[b][1+i]) == false)
        KORALI_LOG_ERROR("Gradient loss returned an invalid value: %f\n", gradientPolicyParams[b][1+i]);

      if (std::isfinite(gradientPolicyParams[b][1+i + _problem->_actionVectorSize]) == false)
        KORALI_LOG_ERROR("Gradient loss returned an invalid value: %f\n", gradientPolicyParams[b][1+i + _problem->_actionVectorSize]);
    }

    // Compute statistics
    for (size_t i = 0; i < _problem->_actionVectorSize; i++)
    {
      _miniBatchPolicyMean[i] += curPolicy.distributionParameters[i];
      _miniBatchPolicyStdDev[i] += curPolicy.distributionParameters[_problem->_actionVectorSize + i];
    }
  }

  // Normalize statistics
  for (size_t i = 0; i < _problem->_actionVectorSize; i++)
  {
    _miniBatchPolicyMean[i] /= (float)miniBatchSize;
    _miniBatchPolicyStdDev[i] /= (float)miniBatchSize;
  }

  return gradientPolicyParams;
}


float VRACER::calculateStateValue(const std::vector<std::vector<float>> &stateSequence, size_t policyIdx)
{
KORALI_LOG_ERROR("calculateStateValue deprecated for external poliy mode");
  return -1.0;
}


void VRACER::runPolicy(const std::vector<std::vector<std::vector<float>>> &stateSequenceBatch, std::vector<policy_t> &policyInfo, size_t policyIdx)
{
KORALI_LOG_ERROR("runPolicy deprecated for external poliy mode");
  // Getting batch size
  size_t batchSize = stateSequenceBatch.size();

  // Preparing storage for results
  policyInfo.resize(batchSize);

  // Forward the neural network
  const auto evaluation = _criticPolicyLearner[policyIdx]->getEvaluation(stateSequenceBatch);

// Write results to policyInfo
#pragma omp parallel for
  for (size_t b = 0; b < batchSize; b++)
  {
    policyInfo[b].stateValue = evaluation[b][0];
  }
}

knlohmann::json VRACER::getPolicy()
{
KORALI_LOG_ERROR("getPolicy deprecated for external poliy mode");
  knlohmann::json hyperparameters;
  for (size_t p = 0; p < _problem->_policiesPerEnvironment; p++)
    hyperparameters["Policy Hyperparameters"][p] = _criticPolicyLearner[p]->getHyperparameters();
  return hyperparameters;
}

void VRACER::setPolicy(const knlohmann::json &hyperparameters)
{
KORALI_LOG_ERROR("setPolicy deprecated for external poliy mode");
  for (size_t p = 0; p < _problem->_policiesPerEnvironment; p++)
    _criticPolicyLearner[p]->_neuralNetwork->setHyperparameters(hyperparameters[p].get<std::vector<float>>());
}

void VRACER::printInformation()
{
  _k->_logger->logInfo("Detailed", " + [VRACER] Policy Learning Rate: %.3e\n", _currentLearningRate);
  _k->_logger->logInfo("Detailed", " + [VRACER] Policy Parameters (Mu & Sigma):\n");
  for (size_t i = 0; i < _problem->_actionVectorSize; i++)
    _k->_logger->logInfo("Detailed", " + [VRACER] Action %zu: (%.3e,%.3e)\n", i, _miniBatchPolicyMean[i], _miniBatchPolicyStdDev[i]);
}

void VRACER::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(_k->_js.getJson(), "Variables"))
 for (size_t i = 0; i < _k->_js["Variables"].size(); i++) { 
 if (isDefined(_k->_js["Variables"][i], "Initial Exploration Noise"))
 {
 try { _k->_variables[i]->_initialExplorationNoise = _k->_js["Variables"][i]["Initial Exploration Noise"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ VRACER ] \n + Key:    ['Initial Exploration Noise']\n%s", e.what()); } 
   eraseValue(_k->_js["Variables"][i], "Initial Exploration Noise");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Initial Exploration Noise'] required by VRACER.\n"); 

 } 
 Continuous::setConfiguration(js);
 _type = "agent/continuous/VRACER";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: VRACER: \n%s\n", js.dump(2).c_str());
} 

void VRACER::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
 for (size_t i = 0; i <  _k->_variables.size(); i++) { 
   _k->_js["Variables"][i]["Initial Exploration Noise"] = _k->_variables[i]->_initialExplorationNoise;
 } 
 Continuous::getConfiguration(js);
} 

void VRACER::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Continuous::applyModuleDefaults(js);
} 

void VRACER::applyVariableDefaults() 
{

 std::string defaultString = "{\"Initial Exploration Noise\": -1.0}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 if (isDefined(_k->_js.getJson(), "Variables"))
  for (size_t i = 0; i < _k->_js["Variables"].size(); i++) 
   mergeJson(_k->_js["Variables"][i], defaultJs); 
 Continuous::applyVariableDefaults();
} 

bool VRACER::checkTermination()
{
 bool hasFinished = false;

 hasFinished = hasFinished || Continuous::checkTermination();
 return hasFinished;
}

;

} //continuous
} //agent
} //solver
} //korali
;
