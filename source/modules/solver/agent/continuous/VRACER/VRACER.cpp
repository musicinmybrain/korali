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

  /*********************************************************************
   * Initializing Critic/Policy Neural Network Optimization Experiment
   *********************************************************************/
  _criticPolicyLearner.resize(_problem->_policiesPerEnvironment);
  _criticPolicyExperiment.resize(_problem->_policiesPerEnvironment);
  _criticPolicyProblem.resize(_problem->_policiesPerEnvironment);

  _effectiveMinibatchSize = _miniBatchSize * _problem->_agentsPerEnvironment;

  if ((_multiAgentRelationship == "Competition") || _problem->_ensembleLearning)
    _effectiveMinibatchSize = _miniBatchSize;

  if (_effectiveMinibatchSize == 1)
    KORALI_LOG_ERROR("Effective Minibatch Size (%ld) should be greater than one!", _effectiveMinibatchSize);

  // Parallel initialization of neural networks (first touch!)
  for (size_t p = 0; p < _problem->_policiesPerEnvironment; p++)
  {
    _criticPolicyExperiment[p]["Random Seed"] = _k->_randomSeed++;

    _criticPolicyExperiment[p]["Problem"]["Type"] = "Supervised Learning";
    _criticPolicyExperiment[p]["Problem"]["Max Timesteps"] = _timeSequenceLength;
    _criticPolicyExperiment[p]["Problem"]["Training Batch Size"] = _effectiveMinibatchSize;
    _criticPolicyExperiment[p]["Problem"]["Testing Batch Size"] = 1;
    _criticPolicyExperiment[p]["Problem"]["Input"]["Size"] = _problem->_stateVectorSize;
    _criticPolicyExperiment[p]["Problem"]["Solution"]["Size"] = 1 + _policyParameterCount;

    _criticPolicyExperiment[p]["Solver"]["Type"] = "DeepSupervisor";
    _criticPolicyExperiment[p]["Solver"]["Mode"] = "Training";
    _criticPolicyExperiment[p]["Solver"]["Number Of Policy Threads"] = 1;
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
    _criticPolicyLearner[p] = dynamic_cast<solver::DeepSupervisor *>(_criticPolicyExperiment[p]._solver);

    // Preallocating space in the underlying supervised problem's input and solution data structures (for performance, we don't reinitialize it every time)
    _criticPolicyProblem[p]->_inputData.resize(_effectiveMinibatchSize);
    _criticPolicyProblem[p]->_solutionData.resize(_effectiveMinibatchSize);
  }

  // Minibatch statistics
  _miniBatchPolicyMean.resize(_problem->_actionVectorSize);
  _miniBatchPolicyStdDev.resize(_problem->_actionVectorSize);
  _miniBatchCurrentPolicyMean.resize(_problem->_actionVectorSize);
  _miniBatchCurrentPolicyStdDev.resize(_problem->_actionVectorSize);
  _miniBatchPolicyGradientMean.resize(_problem->_actionVectorSize);
  _miniBatchPolicyGradientStdDev.resize(_problem->_actionVectorSize);
  _miniBatchKLGradientMean.resize(_problem->_actionVectorSize);
  _miniBatchKLGradientStdDev.resize(_problem->_actionVectorSize);
}

void VRACER::trainPolicy()
{
  // Obtaining Minibatch experience ids
  const auto miniBatch = generateMiniBatch();

  // Gathering state sequences for selected minibatch
  const auto stateSequenceBatch = getMiniBatchStateSequence(miniBatch);

  // Buffer for policy info to update experience metadata
  std::vector<policy_t> policyInfoUpdateMetadata(miniBatch.size());

  // Get number of policies
  size_t numPolicies = _problem->_policiesPerEnvironment;
  if (_problem->_ensembleLearning && (_currentEpisode < _burnIn))
    numPolicies = 1;

  // Create vector with policy indices
  std::vector<size_t> policyIndices;
  for( size_t p = 0; p<numPolicies; p++)
    policyIndices.push_back(p);

  if( _problem->_ensembleLearning )
  {
    // Shuffle vector with policy indices [https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle]
    for( size_t p = 0; p<numPolicies-1; p++)
    {
      // Producing random (uniform) number for the selection of the index
      const float x1 = _uniformGenerator->getRandomNumber();

      // Sample increment and compute j
      size_t increment = std::floor(x1 * (float)(numPolicies-p));
      increment = (p + increment == numPolicies) ? increment - 1 : increment;
      const size_t j = p + increment;

      // Shuffle value
      const size_t tmp = policyIndices[j];
      policyIndices[j] = policyIndices[p];
      policyIndices[p] = tmp;
    }
  }

  // Run training generation for all policies
  for (size_t p : policyIndices)
  {
    // For "Competition" and "Ensemble Learning", the minibatch needs to be modified, create private copy
    auto miniBatchCopy = miniBatch;
    auto stateSequenceBatchCopy = stateSequenceBatch;

    // Disable experience sharing for competing agents or Bayesian reinforcement learning
    if ((_multiAgentRelationship == "Competition") || _problem->_ensembleLearning)
    {
      std::vector<std::pair<size_t, size_t>> miniBatchModified(_miniBatchSize);
      std::vector<std::vector<std::vector<float>>> stateSequenceModified(_miniBatchSize);
      for (size_t i = 0; i < _miniBatchSize; i++)
      {
        miniBatchModified[i] = miniBatch[i * numPolicies + p];
        stateSequenceModified[i] = stateSequenceBatch[i * numPolicies + p];
      }
      miniBatchCopy = miniBatchModified;
      stateSequenceBatchCopy = stateSequenceModified;
    }

    // Container for parameters
    std::vector<policy_t> policyInfo;

    // For bayesian RL, compute predictive posterior distribution
    if ((_problem->_ensembleLearning || _bayesianLearning) &&
        (_currentEpisode >= _burnIn) &&
        _useGaussianApproximation)
      computePredictivePosteriorDistribution(stateSequenceBatchCopy, policyInfo, p);
    else // Forward Policy
      runPolicy(stateSequenceBatchCopy, policyInfo, p);

    // Using policy information to update experience's metadata
    updateExperienceMetadata(miniBatchCopy, policyInfo);

    // Now calculating policy gradients
    calculatePolicyGradients(miniBatchCopy, p);

    // Updating learning rate for critic/policy learner guided by REFER
    _criticPolicyLearner[p]->_learningRate = _currentLearningRate;

    // Now applying gradients to update policy NN
    _criticPolicyLearner[p]->runGeneration();

    // Add noise for Stochastic Gradient Langevin Dynamics (https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf)
    if (_langevinDynamics)
    {
      auto hyperparameters = _criticPolicyLearner[p]->getHyperparameters();

      // #pragma omp parallel for simd
      for (size_t n = 0; n < hyperparameters.size(); n++)
        hyperparameters[n] += std::sqrt(2 * _currentLearningRate) * _normalGenerator->getRandomNumber();
    }

    // Store policyData for agent p for later update of metadata
    if ((numPolicies > 1) && (_multiAgentRelationship != "Competition") && !_problem->_ensembleLearning)
      for (size_t b = 0; b < _miniBatchSize; b++)
        policyInfoUpdateMetadata[b * numPolicies + p] = policyInfo[b * numPolicies + p];
  }

  // Correct experience metadata
  if ((numPolicies > 1) && (_multiAgentRelationship != "Competition") && !_problem->_ensembleLearning)
    updateExperienceMetadata(miniBatch, policyInfoUpdateMetadata);
}

void VRACER::calculatePolicyGradients(const std::vector<std::pair<size_t, size_t>> &miniBatch, const size_t policyIdx)
{
  // Resetting statistics
  std::fill(_miniBatchPolicyMean.begin(), _miniBatchPolicyMean.end(), 0.0);
  std::fill(_miniBatchPolicyStdDev.begin(), _miniBatchPolicyStdDev.end(), 0.0);
  std::fill(_miniBatchCurrentPolicyMean.begin(), _miniBatchCurrentPolicyMean.end(), 0.0);
  std::fill(_miniBatchCurrentPolicyStdDev.begin(), _miniBatchCurrentPolicyStdDev.end(), 0.0);
  std::fill(_miniBatchPolicyGradientMean.begin(), _miniBatchPolicyGradientMean.end(), 0.0);
  std::fill(_miniBatchPolicyGradientStdDev.begin(), _miniBatchPolicyGradientStdDev.end(), 0.0);
  std::fill(_miniBatchKLGradientMean.begin(), _miniBatchKLGradientMean.end(), 0.0);
  std::fill(_miniBatchKLGradientStdDev.begin(), _miniBatchKLGradientStdDev.end(), 0.0);
  _valueLoss = 0.0;
  _policyLoss = 0.0;

  const size_t miniBatchSize = miniBatch.size();
  const size_t numAgents = _problem->_agentsPerEnvironment;

#pragma omp parallel for schedule(guided, numAgents) reduction(vec_float_plus                                  \
                                                               : _miniBatchPolicyMean, _miniBatchPolicyStdDev, _miniBatchPolicyGradientMean, _miniBatchPolicyGradientStdDev, _miniBatchKLGradientMean, _miniBatchKLGradientStdDev, _miniBatchCurrentPolicyMean, _miniBatchCurrentPolicyStdDev) \
  reduction(+                                                                                                  \
            : _valueLoss, _policyLoss)
  for (size_t b = 0; b < miniBatchSize; b++)
  {
    // Getting index of current experiment
    const size_t expId = miniBatch[b].first;
    const size_t agentId = miniBatch[b].second;

    // Get policy and action for this experience
    const auto &expPolicy = _expPolicyBuffer[expId][agentId];
    const auto &expAction = _actionBuffer[expId][agentId];

    // Gathering metadata
    const auto stateValue = _stateValueBufferContiguous[expId * numAgents + agentId];
    const auto &curPolicy = _curPolicyBuffer[expId][agentId];
    const auto expVtbc = _retraceValueBufferContiguous[expId * numAgents + agentId];

    // Storage for the update gradient
    std::vector<float> gradientLoss(1 + _policyParameterCount, 0.0f);

    // Gradient of Value Function V(s) (eq. (9); *-1 because the optimizer is maximizing)
    gradientLoss[0] = (expVtbc - stateValue);

    // Cumulate value loss
    _valueLoss += 0.5 * (expVtbc - stateValue) * (expVtbc - stateValue);

    // Gradient has to be divided by Number of Agents in Cooperation models
    if (_multiAgentRelationship == "Cooperation")
      gradientLoss[0] /= numAgents;

    // Gradient has to be divided by Number of Samples in Bayesian learning
    if (_problem->_ensembleLearning || _bayesianLearning)
      gradientLoss[0] /= _numberOfSamples;

    // Check value of gradient
    if (std::isfinite(gradientLoss[0]) == false)
      KORALI_LOG_ERROR("Value Gradient has an invalid value: %f\n", gradientLoss[0]);

    // Compute policy gradient inside trust region
    std::vector<float> polGrad(_policyParameterCount, 0.0);
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

      // Cumulate policy loss
      const float importanceWeight = _importanceWeightBuffer[expId][agentId];
      _policyLoss += importanceWeight * (Qret - stateValue);

      // Compute Off-Policy Gradient
      polGrad = calculateImportanceWeightGradient(expAction, curPolicy, expPolicy, importanceWeight);

      // Multi-agent correlation implies additional factor
      if (_multiAgentCorrelation)
      {
        const float correlationFactor = _productImportanceWeightBuffer[expId] / _importanceWeightBuffer[expId][agentId];
        for (size_t i = 0; i < polGrad.size(); i++)
          polGrad[i] *= correlationFactor;
      }

      // Modifications of Gradient for  Bayesian Learning
      if ((_problem->_ensembleLearning || _bayesianLearning) && (_currentEpisode >= _burnIn))
      {
        // Additional factors from Gaussian approximation
        if (_useGaussianApproximation)
        {
          const float invN = 1 / _numberOfSamples;
          for (size_t i = 0; i < _problem->_actionVectorSize; i++)
          {
            // Get distribution parameter from predictive posterior policy
            const float mean = curPolicy.distributionParameters[i];
            const float sigma = curPolicy.distributionParameters[_problem->_actionVectorSize + i];
            const float invSigma = 1 / sigma;

            // Get sigma for current hyperparameters
            const float curMean = curPolicy.currentDistributionParameters[i];
            const float curSigma = curPolicy.currentDistributionParameters[_problem->_actionVectorSize + i];

            // Scaling mean gradient by number of samples
            polGrad[i] *= invN;

            // Adding contribution from standard deviation
            polGrad[i] += invN * invSigma * (curMean - mean) * polGrad[i + _problem->_actionVectorSize];

            // Scaling standard deviation gradient by number of samples
            polGrad[i + _problem->_actionVectorSize] *= invN * invSigma * curSigma;
          }
        }
      }

      // Set Gradient of Loss wrt Params
      for (size_t i = 0; i < _policyParameterCount; i++)
      {
        gradientLoss[1 + i] = _experienceReplayOffPolicyREFERCurrentBeta[agentId] * lossOffPolicy * polGrad[i];

        if (std::isfinite(gradientLoss[i + 1]) == false)
          KORALI_LOG_ERROR("Policy Gradient i=%ld has an invalid value: %f\n", i, gradientLoss[i + 1]);
      }
    }

    // Compute derivative of KL divergence
    auto klGrad = calculateKLDivergenceGradient(expPolicy, curPolicy);

    // Compute factor for KL penalization
    const float klGradMultiplier = -(1.0f - _experienceReplayOffPolicyREFERCurrentBeta[agentId]);

    // Safety check
    if (std::isfinite(klGradMultiplier) == false)
      KORALI_LOG_ERROR("KL multiplier has an invalid value: %f\n", klGradMultiplier);

    // When not using Gaussian approximation, clipp KL correction
    for (size_t i = 0; i < _problem->_actionVectorSize; i++)
    {
      // Clip large values of the KL grad
      if ( std::abs(klGrad[i]) > 1e7)
        klGrad[i] = klGrad[i] < 0 ? -1e7 : 1e7;

      if ( std::abs(klGrad[i + _problem->_actionVectorSize]) > 1e7)
        klGrad[i + _problem->_actionVectorSize] = klGrad[i + _problem->_actionVectorSize] < 0 ? -1e7 : 1e7;

      // Modifications of Gradient for  Bayesian Learning
      if ((_problem->_ensembleLearning || _bayesianLearning) && (_currentEpisode >= _burnIn))
      {
        // Additional factors from Gaussian approximation
        if (_useGaussianApproximation)
        {
          const float invN = 1 / _numberOfSamples;
          for (size_t i = 0; i < _problem->_actionVectorSize; i++)
          {
            // Get distribution parameter from predictive posterior policy
            const float mean = curPolicy.distributionParameters[i];
            const float sigma = curPolicy.distributionParameters[_problem->_actionVectorSize + i];
            const float invSigma = 1 / sigma;

            // Get sigma for current hyperparameters
            const float curMean = curPolicy.currentDistributionParameters[i];
            const float curSigma = curPolicy.currentDistributionParameters[_problem->_actionVectorSize + i];

            // Scaling mean gradient by number of samples
            klGrad[i] *= invN;

            // Adding contribution from standard deviation
            klGrad[i] += invN * invSigma * (curMean - mean) * klGrad[i + _problem->_actionVectorSize];

            // Scaling standard deviation gradient by number of samples
            klGrad[i + _problem->_actionVectorSize] *= invN * invSigma * curSigma;
          }
        }
      }

      // Safety checks
      if (std::isfinite(klGrad[i]) == false)
        KORALI_LOG_ERROR("KL Gradient has an invalid value %f\n", klGrad[i]);

      if (std::isfinite(klGrad[i + _problem->_actionVectorSize]) == false)
        KORALI_LOG_ERROR("KL Gradient has an invalid value %f\n", klGrad[i + _problem->_actionVectorSize]);

      // Add KL contribution
      gradientLoss[1 + i] += klGradMultiplier * klGrad[i];
      gradientLoss[1 + i + _problem->_actionVectorSize] += klGradMultiplier * klGrad[i + _problem->_actionVectorSize];

      // Safety checks
      if (std::isfinite(gradientLoss[1 + i]) == false)
        KORALI_LOG_ERROR("Gradient has an invalid value %f\n", gradientLoss[1 + i]);

      if (std::isfinite(gradientLoss[1 + i + _problem->_actionVectorSize]) == false)
        KORALI_LOG_ERROR("Gradient has an invalid value %f\n", gradientLoss[1 + i + _problem->_actionVectorSize]);
    }

    // Set Gradient of Loss as Solution
    _criticPolicyProblem[policyIdx]->_solutionData[b] = gradientLoss;

    // Compute statistics
    for (size_t i = 0; i < _problem->_actionVectorSize; i++)
    {
      _miniBatchPolicyMean[i] += curPolicy.distributionParameters[i];
      _miniBatchPolicyStdDev[i] += curPolicy.distributionParameters[_problem->_actionVectorSize + i];
      _miniBatchCurrentPolicyMean[i] += curPolicy.currentDistributionParameters[i];
      _miniBatchCurrentPolicyStdDev[i] += curPolicy.currentDistributionParameters[_problem->_actionVectorSize + i];
      _miniBatchPolicyGradientMean[i] += polGrad[i];
      _miniBatchPolicyGradientStdDev[i] += polGrad[i + _problem->_actionVectorSize];
      _miniBatchKLGradientMean[i] += klGrad[i];
      _miniBatchKLGradientStdDev[i] += klGrad[i + _problem->_actionVectorSize];
    }
  }

  // Normalize statistics
  for (size_t i = 0; i < _problem->_actionVectorSize; i++)
  {
    _miniBatchPolicyMean[i] /= (float)miniBatchSize;
    _miniBatchPolicyStdDev[i] /= (float)miniBatchSize;
    _miniBatchCurrentPolicyMean[i] /= (float)miniBatchSize;
    _miniBatchCurrentPolicyStdDev[i] /= (float)miniBatchSize;
    _miniBatchPolicyGradientMean[i] /= (float)miniBatchSize;
    _miniBatchPolicyGradientStdDev[i] /= (float)miniBatchSize;
    _miniBatchKLGradientMean[i] /= (float)miniBatchSize;
    _miniBatchKLGradientStdDev[i] /= (float)miniBatchSize;
  }

  _valueLoss /= (float)miniBatchSize;
  _policyLoss /= (float)miniBatchSize;
}

float VRACER::calculateStateValue(const std::vector<std::vector<float>> &stateSequence, const size_t policyIdx)
{
  // Forward the neural network for this state to get the state value
  const auto evaluation = _criticPolicyLearner[policyIdx]->getEvaluation({stateSequence});
  return evaluation[0][0];
}

void VRACER::runPolicy(const std::vector<std::vector<std::vector<float>>> &stateSequenceBatch, std::vector<policy_t> &policyInfo, const size_t policyIdx)
{
  // Getting batch size
  const size_t batchSize = stateSequenceBatch.size();

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
    policyInfo[b].currentDistributionParameters.assign(evaluation[b].begin() + 1, evaluation[b].end());
  }
}

void VRACER::computePredictivePosteriorDistribution(const std::vector<std::vector<std::vector<float>>> &stateSequenceBatch, std::vector<policy_t> &curPolicy, const size_t policyIdx)
{
  // Get minibatch-size
  const size_t batchSize = stateSequenceBatch.size();

  // Intitialize curPolicy
  curPolicy.resize(batchSize);
#pragma omp parallel for
  for (size_t b = 0; b < batchSize; b++)
  {
    curPolicy[b].stateValue = 0.0f;
    curPolicy[b].distributionParameters.resize(_policyParameterCount, 0.0);
    curPolicy[b].currentDistributionParameters.resize(_policyParameterCount, 0.0);
  }

  // Create empty policy to forward samples
  std::vector<policy_t> policy;

  // Determine the number of samples
  size_t numSamples = _numberOfSamples;

  // Take care of situation where not enough hyperparameters are in buffer
  if (_numberOfStoredHyperparameters > 1)
    numSamples = std::max(numSamples, _hyperparameterBuffer.size());

  // For swag and dropout we have to include current hyperparameters at the end
  if (_swag || (_dropoutProbability > 0.0))
    numSamples--;

  // Compute moments for Gaussian approximation to predictive posterior distribution
  for (size_t p = 0; p < _problem->_policiesPerEnvironment; p++)
    for (size_t s = 0; s < numSamples; s++)
    {
      // Get sample
      std::vector<float> hyperparameters;
      if (_swag || (_dropoutProbability > 0.0))
        hyperparameters = samplePosterior(p);
      else
        hyperparameters = _hyperparameterBuffer[s][p];

      // Set parameters in neural network
      _criticPolicyLearner[p]->_neuralNetwork->setHyperparameters(hyperparameters);

      // Forward policy
      runPolicy(stateSequenceBatch, policy, p);

      // Update statistics of predictive posterior distribution
      // mean = 1/N sum{ mean_i }, var = 1/N sum{ mean_i^2 + var_i^2 } - mean
#pragma omp parallel for
      for (size_t b = 0; b < batchSize; b++)
      {
        // Accumulate State Value
        curPolicy[b].stateValue += policy[b].stateValue;
        for (size_t i = 0; i < _problem->_actionVectorSize; i++)
        {
          // Get mean, squared mean, standard deviation and variance
          const float mean = policy[b].distributionParameters[i];
          const float meanSquared = mean * mean;
          const float standardDeviation = policy[b].distributionParameters[_problem->_actionVectorSize + i];
          const float variance = standardDeviation * standardDeviation;

          // Accumulate Mean
          curPolicy[b].distributionParameters[i] += mean;

          // Accumulate Variance
          curPolicy[b].distributionParameters[_problem->_actionVectorSize + i] += (meanSquared + variance);
        }
      }
    }

  // Get current hyperparameters
  const auto &hyperparameters = _hyperparameterBuffer[_hyperparameterBuffer.size() - 1][policyIdx];

  // Set current hyperparameters
  _criticPolicyLearner[policyIdx]->setHyperparameters(hyperparameters);

  // Forward policy for current hyperparameters // TODO: avoid forwarding current policy twice
  runPolicy(stateSequenceBatch, policy, policyIdx);

  for (size_t b = 0; b < batchSize; b++)
    for (size_t i = 0; i < _problem->_actionVectorSize; i++)
    {
      // Get mean and standard deviation
      const float mean = policy[b].distributionParameters[i];
      float standardDeviation = policy[b].distributionParameters[_problem->_actionVectorSize + i];

      // Save mean and standard deviation for current hyperparameters
      curPolicy[b].currentDistributionParameters[i] = mean;
      curPolicy[b].currentDistributionParameters[_problem->_actionVectorSize + i] = standardDeviation;
    }

  // For consistency of SWAG and dropout, add contribution from the current policy
  if (_swag || (_dropoutProbability > 0.0))
  {
#pragma omp parallel for
    for (size_t b = 0; b < batchSize; b++)
    {
      // Accumulate State Value
      curPolicy[b].stateValue += policy[b].stateValue;
      for (size_t i = 0; i < _problem->_actionVectorSize; i++)
      {
        // Get mean, squared mean, standard deviation and variance
        const float mean = policy[b].distributionParameters[i];
        const float meanSquared = mean * mean;
        const float standardDeviation = policy[b].distributionParameters[_problem->_actionVectorSize + i];
        const float variance = standardDeviation * standardDeviation;

        // Accumulate Mean
        curPolicy[b].distributionParameters[i] += mean;

        // Accumulate Variance
        curPolicy[b].distributionParameters[_problem->_actionVectorSize + i] += (meanSquared + variance);
      }
    }
  }

  // Complete computation of predictive posterior distribution
  const float invTotNumSamples = 1.0 / (_numberOfSamples * _problem->_policiesPerEnvironment);
#pragma omp parallel for
  for (size_t b = 0; b < batchSize; b++)
  {
    // Finalize State Value
    curPolicy[b].stateValue *= invTotNumSamples;
    for (size_t i = 0; i < _problem->_actionVectorSize; i++)
    {
      // Finalize Mean
      const float mixtureMean = curPolicy[b].distributionParameters[i] * invTotNumSamples;
      curPolicy[b].distributionParameters[i] = mixtureMean;

      // Finalize Variance
      curPolicy[b].distributionParameters[_problem->_actionVectorSize + i] *= invTotNumSamples;
      curPolicy[b].distributionParameters[_problem->_actionVectorSize + i] -= mixtureMean * mixtureMean;
      curPolicy[b].distributionParameters[_problem->_actionVectorSize + i] = std::sqrt(curPolicy[b].distributionParameters[_problem->_actionVectorSize + i]);
    }
  }
}

knlohmann::json VRACER::getPolicy()
{
  knlohmann::json hyperparameters;
  for (size_t p = 0; p < _problem->_policiesPerEnvironment; p++)
    hyperparameters["Policy Hyperparameters"][p] = _criticPolicyLearner[p]->getHyperparameters();
  return hyperparameters;
}

void VRACER::setPolicy(const knlohmann::json &hyperparameters)
{
  for (size_t p = 0; p < _problem->_policiesPerEnvironment; p++)
    _criticPolicyLearner[p]->_neuralNetwork->setHyperparameters(hyperparameters[p].get<std::vector<float>>());
}

void VRACER::printInformation()
{
  _k->_logger->logInfo("Normal", " + [VRACER] Policy Learning Rate: %.3e\n", _currentLearningRate);
  if ( _useGaussianApproximation )
  {
    _k->_logger->logInfo("Detailed", " + [VRACER] Policy Parameters (Mu & Sigma - Gaussian) - (Mu & Sigma - Individual):\n");
    for (size_t i = 0; i < _problem->_actionVectorSize; i++)
      _k->_logger->logInfo("Detailed", " + [VRACER] Action %zu: (%.3e,%.3e) - (%.3e,%.3e)\n", i, _miniBatchPolicyMean[i], _miniBatchPolicyStdDev[i], _miniBatchCurrentPolicyMean[i], _miniBatchCurrentPolicyStdDev[i]);
  }
  else
  {
    _k->_logger->logInfo("Detailed", " + [VRACER] Policy Parameters (Mu & Sigma):\n");
    for (size_t i = 0; i < _problem->_actionVectorSize; i++)
      _k->_logger->logInfo("Detailed", " + [VRACER] Action %zu: (%.3e,%.3e)\n", i, _miniBatchPolicyMean[i], _miniBatchPolicyStdDev[i]);
  }
  _k->_logger->logInfo("Detailed", " + [VRACER] Policy Gradients (Mu & Sigma - IW) - (Mu & Sigma - KL):\n");
  for (size_t i = 0; i < _problem->_actionVectorSize; i++)
  {
    _k->_logger->logInfo("Detailed", " + [VRACER] Action %zu: (%.3e,%.3e) - (%.3e,%.3e)\n", i, _miniBatchPolicyGradientMean[i], _miniBatchPolicyGradientStdDev[i], _miniBatchKLGradientMean[i], _miniBatchKLGradientStdDev[i]);
  }
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
