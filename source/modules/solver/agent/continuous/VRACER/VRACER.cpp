#include "engine.hpp"
#include "modules/solver/agent/continuous/VRACER/VRACER.hpp"
#include "omp.h"
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

  /*********************************************************************
   * Initializing Critic/Policy Neural Network Optimization Experiment
   *********************************************************************/
  _criticPolicyLearner.resize(_problem->_policiesPerEnvironment);
  _criticPolicyExperiment.resize(_problem->_policiesPerEnvironment);
  _criticPolicyProblem.resize(_problem->_policiesPerEnvironment);

  _effectiveMinibatchSize = _miniBatchSize * _problem->_agentsPerEnvironment;

  if( (_multiAgentRelationship == "Competition") || _problem->_ensembleLearning )
    _effectiveMinibatchSize = _miniBatchSize;

  for (size_t p = 0; p < _problem->_policiesPerEnvironment; p++)
  {
    _criticPolicyExperiment[p]["Problem"]["Type"] = "Supervised Learning";
    _criticPolicyExperiment[p]["Problem"]["Max Timesteps"] = _timeSequenceLength;
    _criticPolicyExperiment[p]["Problem"]["Training Batch Size"] = _effectiveMinibatchSize;
    _criticPolicyExperiment[p]["Problem"]["Testing Batch Size"] = 1;
    _criticPolicyExperiment[p]["Problem"]["Input"]["Size"] = _problem->_stateVectorSize;
    _criticPolicyExperiment[p]["Problem"]["Solution"]["Size"] = 1 + _policyParameterCount;

    _criticPolicyExperiment[p]["Solver"]["Type"] = "Learner/DeepSupervisor";
    _criticPolicyExperiment[p]["Solver"]["Mode"] = "Training";
    _criticPolicyExperiment[p]["Solver"]["L2 Regularization"]["Enabled"] = _l2RegularizationEnabled;
    _criticPolicyExperiment[p]["Solver"]["L2 Regularization"]["Importance"] = _l2RegularizationImportance;
    _criticPolicyExperiment[p]["Solver"]["Learning Rate"] = _currentLearningRate;
    _criticPolicyExperiment[p]["Solver"]["Loss Function"] = "Direct Gradient";
    _criticPolicyExperiment[p]["Solver"]["Neural Network"]["Optimizer"] = _neuralNetworkOptimizer;
    _criticPolicyExperiment[p]["Solver"]["Neural Network"]["Engine"] = _neuralNetworkEngine;
    _criticPolicyExperiment[p]["Solver"]["Neural Network"]["Hidden Layers"] = _neuralNetworkHiddenLayers;
    _criticPolicyExperiment[p]["Solver"]["Output Weights Scaling"] = 0.001;

    // No transformations for the state value output
    _criticPolicyExperiment[p]["Solver"]["Neural Network"]["Output Layer"]["Transformation Mask"][0] = "Identity";
    _criticPolicyExperiment[p]["Solver"]["Neural Network"]["Output Layer"]["Scale"][0] = 1.0f;
    _criticPolicyExperiment[p]["Solver"]["Neural Network"]["Output Layer"]["Shift"][0] = 0.0f;

    // Setting transformations for the selected policy distribution output
    for (size_t i = 0; i < _policyParameterCount; i++)
    {
      _criticPolicyExperiment[p]["Solver"]["Neural Network"]["Output Layer"]["Transformation Mask"][i + 1] = _policyParameterTransformationMasks[i];
      _criticPolicyExperiment[p]["Solver"]["Neural Network"]["Output Layer"]["Scale"][i + 1] = _policyParameterScaling[i];
      _criticPolicyExperiment[p]["Solver"]["Neural Network"]["Output Layer"]["Shift"][i + 1] = _policyParameterShifting[i];
    }

    // Running initialization to verify that the configuration is correct
    _criticPolicyExperiment[p].setEngine(_k->_engine);
    _criticPolicyExperiment[p].initialize();
    _criticPolicyProblem[p] = dynamic_cast<problem::SupervisedLearning *>(_criticPolicyExperiment[p]._problem);
    _criticPolicyLearner[p] = dynamic_cast<solver::learner::DeepSupervisor *>(_criticPolicyExperiment[p]._solver);

    // Preallocating space in the underlying supervised problem's input and solution data structures (for performance, we don't reinitialize it every time)
    _criticPolicyProblem[p]->_inputData.resize(_effectiveMinibatchSize);
    _criticPolicyProblem[p]->_solutionData.resize(_effectiveMinibatchSize);
  }

  // Minibatch statistics
  _miniBatchPolicyMean.resize(_problem->_actionVectorSize);
  _miniBatchPolicyStdDev.resize(_problem->_actionVectorSize);
}

void VRACER::trainPolicy()
{
  auto beginTime = std::chrono::steady_clock::now();
  // Obtaining Minibatch experience ids
  const auto miniBatch = generateMiniBatch();

  // Gathering state sequences for selected minibatch
  const auto stateSequenceBatch = getMiniBatchStateSequence(miniBatch);
  auto endTime = std::chrono::steady_clock::now(); 
  _sessionCreateMinibatchTime += std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - beginTime).count();
  _generationCreateMinibatchTime += std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - beginTime).count();

  // Prepare stateSequenceBatch and miniBatch
  auto _miniBatch          = miniBatch;
  auto _stateSequenceBatch = stateSequenceBatch;

  // Buffer for policy info to update experience metadata
  std::vector<policy_t> policyInfoUpdateMetadata(miniBatch.size());

  // Get number of policies
  const size_t numPolicies = _problem->_policiesPerEnvironment;

  // Run training generation for all policies
  for (size_t p = 0; p < numPolicies; p++)
  {
    // Disable experience sharing competing agents or Bayesian reinforcement learning
    if( (_multiAgentRelationship == "Competition") || _problem->_ensembleLearning )
    {
      std::vector<std::pair<size_t, size_t>> miniBatchBuffer(_miniBatchSize);
      std::vector<std::vector<std::vector<float>>> stateSequenceBuffer(_miniBatchSize);
      for( size_t i = 0; i<_miniBatchSize; i++ )
      {
        miniBatchBuffer[i]     = miniBatch[ i*numPolicies + p ];
        stateSequenceBuffer[i] = stateSequenceBatch[ i*numPolicies + p ];
      }
      _miniBatch          = miniBatchBuffer;
      _stateSequenceBatch = stateSequenceBuffer;
    }

    // Forward NN
    std::vector<policy_t> policyInfo;
    beginTime = std::chrono::steady_clock::now(); 
    runPolicy(_stateSequenceBatch, policyInfo, p);
    endTime = std::chrono::steady_clock::now(); 
    _sessionRunPolicyTime += std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - beginTime).count();
    _generationRunPolicyTime += std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - beginTime).count();

    // Using policy information to update experience's metadata
    beginTime = std::chrono::steady_clock::now();
    updateExperienceMetadata(_miniBatch, policyInfo);
    endTime = std::chrono::steady_clock::now();
    _sessionUpdateMetadataTime += std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - beginTime).count();
    _generationUpdateMetadataTime += std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - beginTime).count();

    // Now calculating policy gradients
    beginTime = std::chrono::steady_clock::now();
    calculatePolicyGradients(_miniBatch, p);
    endTime = std::chrono::steady_clock::now();
    _sessionPolicyGradientTime += std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - beginTime).count();
    _generationPolicyGradientTime += std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - beginTime).count();

    // Updating learning rate for critic/policy learner guided by REFER
    _criticPolicyLearner[p]->_learningRate = _currentLearningRate;

    // Now applying gradients to update policy NN
    beginTime = std::chrono::steady_clock::now();
    _criticPolicyLearner[p]->runGeneration();
    endTime = std::chrono::steady_clock::now();
    _sessionRunGenerationTime += std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - beginTime).count();
    _generationRunGenerationTime += std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - beginTime).count();

    // Store policyData for agent p for later update of metadata
    if ( (numPolicies > 1) && (_multiAgentRelationship != "Competition") && !_problem->_ensembleLearning )
      for (size_t b = 0; b < _miniBatchSize; b++)
        policyInfoUpdateMetadata[b * numPolicies + p] = policyInfo[b * numPolicies + p];
  }

  // Correct experience metadata
  if ( (numPolicies > 1) && (_multiAgentRelationship != "Competition") && !_problem->_ensembleLearning )
    updateExperienceMetadata(_miniBatch, policyInfoUpdateMetadata);
}

void VRACER::calculatePolicyGradients(const std::vector<std::pair<size_t, size_t>> &miniBatch, const size_t policyIdx)
{
  // Resetting statistics
  std::fill(_miniBatchPolicyMean.begin(), _miniBatchPolicyMean.end(), 0.0);
  std::fill(_miniBatchPolicyStdDev.begin(), _miniBatchPolicyStdDev.end(), 0.0);

  const size_t miniBatchSize = miniBatch.size();
  const size_t numAgents = _problem->_agentsPerEnvironment;

#pragma omp parallel for schedule(guided, numAgents) reduction(vec_float_plus \
                                                               : _miniBatchPolicyMean, _miniBatchPolicyStdDev)
  for (size_t b = 0; b < miniBatchSize; b++)
  {
    // Getting index of current experiment
    const size_t expId = miniBatch[b].first;
    const size_t agentId = miniBatch[b].second;

    // Get policy and action for this experience
    const auto &expPolicy = _expPolicyVector[expId][agentId];
    const auto &expAction = _actionVector[expId][agentId];

    // Gathering metadata
    const float stateValue = _stateValueVectorContiguous[expId + agentId];
    const float expVtbc = _retraceValueVectorContiguous[expId + agentId];
    const auto &curPolicy = _curPolicyVector[expId][agentId];

    // Storage for the update gradient
    std::vector<float> gradientLoss(1 + 2 * _problem->_actionVectorSize, 0.0f);

    // Gradient of Value Function V(s) (eq. (9); *-1 because the optimizer is maximizing)
    gradientLoss[0] = (expVtbc - stateValue);

    //Gradient has to be divided by Number of Agents in Cooperation models
    if (_multiAgentRelationship == "Cooperation")
      gradientLoss[0] /= numAgents;

    // Check value of gradient
    if (std::isfinite(gradientLoss[0]) == false)
      KORALI_LOG_ERROR("Value Gradient has an invalid value: %f\n", gradientLoss[0]);

    // Compute policy gradient inside trust region
    if (_isOnPolicyVector[expId][agentId])
    {
      // Qret for terminal state is just reward
      float Qret = getScaledReward(_rewardVectorContiguous[expId+agentId]);

      // If experience is non-terminal, add Vtbc
      if (_terminationVector[expId] == e_nonTerminal)
      {
        const float nextExpVtbc = _retraceValueVectorContiguous[expId + 1 + agentId];
        Qret += _discountFactor * nextExpVtbc;
      }

      // If experience is truncated, add truncated state value
      if (_terminationVector[expId] == e_truncated)
      {
        const float nextExpVtbc = _truncatedStateValueVector[expId][agentId];
        Qret += _discountFactor * nextExpVtbc;
      }

      // Compute Off-Policy Objective (eq. 5)
      const float lossOffPolicy = Qret - stateValue;

      // Compute Off-Policy Gradient
      auto polGrad = calculateImportanceWeightGradient(expAction, curPolicy, expPolicy);

      // Multi-agent correlation implies additional factor
      if (_multiAgentCorrelation)
      {
        const float correlationFactor = _productImportanceWeightVector[expId] / _importanceWeightVector[expId][agentId];
        for (size_t i = 0; i < polGrad.size(); i++)
          polGrad[i] *= correlationFactor;
      }

      // Set Gradient of Loss wrt Params
      for (size_t i = 0; i < 2 * _problem->_actionVectorSize; i++)
      {
        gradientLoss[1 + i] = _experienceReplayOffPolicyREFERCurrentBeta[agentId] * lossOffPolicy * polGrad[i];

        if (std::isfinite(gradientLoss[i + 1]) == false)
          KORALI_LOG_ERROR("Policy Gradient has an invalid value: %f\n", gradientLoss[i + 1]);
      }
    }

    // Compute derivative of KL divergence
    const auto klGrad = calculateKLDivergenceGradient(expPolicy, curPolicy);

    // Compute factor for KL penalization
    const float klGradMultiplier = -(1.0f - _experienceReplayOffPolicyREFERCurrentBeta[agentId]);

    // Add KL contribution
    for (size_t i = 0; i < _problem->_actionVectorSize; i++)
    {
      gradientLoss[1 + i] += klGradMultiplier * klGrad[i];
      gradientLoss[1 + i + _problem->_actionVectorSize] += klGradMultiplier * klGrad[i + _problem->_actionVectorSize];

      if (std::isfinite(gradientLoss[i + 1]) == false)
        KORALI_LOG_ERROR("Policy Gradient has an invalid value after KL correction: %f\n", gradientLoss[i + 1]);

      if (std::isfinite(gradientLoss[i + 1 + _problem->_actionVectorSize]) == false)
        KORALI_LOG_ERROR("Gradient has an invalid value after KL correction: %f\n", gradientLoss[i + 1 + _problem->_actionVectorSize]);
    }

    // Add noise to mimic Stochastic Gradient Langevin Dynamics (https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf) - PROBABLY NOT NEEDED IN RL (very noisy gradients)
    if( _langevinDynamics > 0.0 )
    for( size_t i = 0; i<2*_problem->_actionVectorSize+1; i++ )
      gradientLoss[i] += std::sqrt(_langevinDynamics * _currentLearningRate) * _normalGenerator->getRandomNumber();

    // Set Gradient of Loss as Solution
    _criticPolicyProblem[policyIdx]->_solutionData[b] = gradientLoss;

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
}

float VRACER::calculateStateValue(const std::vector<std::vector<float>> &stateSequence, size_t policyIdx)
{
  // Forward the neural network for this state to get the state value
  const auto evaluation = _criticPolicyLearner[policyIdx]->getEvaluation({stateSequence});
  return evaluation[0][0];
}

void VRACER::runPolicy(const std::vector<std::vector<std::vector<float>>> &stateSequenceBatch, std::vector<policy_t> &policyInfo, size_t policyIdx)
{
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
    policyInfo[b].distributionParameters.assign(evaluation[b].begin() + 1, evaluation[b].end());
  }
}

knlohmann::json VRACER::getAgentPolicy()
{
  knlohmann::json hyperparameters;
  for (size_t p = 0; p < _problem->_policiesPerEnvironment; p++)
    hyperparameters["Policy Hyperparameters"][p] = _criticPolicyLearner[p]->getHyperparameters();
  return hyperparameters;
}

void VRACER::setAgentPolicy(const knlohmann::json &hyperparameters)
{
  for (size_t p = 0; p < _problem->_policiesPerEnvironment; p++)
    _criticPolicyLearner[p]->setHyperparameters(hyperparameters[p].get<std::vector<float>>());
}

void VRACER::printAgentInformation()
{
  _k->_logger->logInfo("Normal", " + [VRACER] Policy Learning Rate: %.3e\n", _currentLearningRate);
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
