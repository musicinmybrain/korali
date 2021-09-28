#include "engine.hpp"
#include "modules/solver/agent/discrete/discrete.hpp"
#include "sample/sample.hpp"

namespace korali
{
namespace solver
{
namespace agent
{
;

void Discrete::initializeAgent()
{
  // Getting discrete problem pointer
  _problem = dynamic_cast<problem::reinforcementLearning::Discrete *>(_k->_problem);
    
  _policyParameterCount = _problem->_possibleActions.size() + 1; // q values and inverseTemperature
}

void Discrete::getAction(korali::Sample &sample)
{
  // Get action for all the agents in the environment
  for (size_t i = 0; i < sample["State"].size(); i++)
  {
    // Getting current state
    auto state = sample["State"][i].get<std::vector<float>>();

    // Adding state to the state time sequence
    _stateTimeSequence.add(state);

    // Getting the probability of the actions given by the agent's policy
    auto policy = runPolicy({_stateTimeSequence.getVector()})[0];
    const auto &qValAndInvTemp = policy.distributionParameters;
    const auto &pActions = policy.actionProbabilities;

    // Storage for the action index to use
    size_t actionIdx = 0;

    /*****************************************************************************
  * During training, we follow the Epsilon-greedy strategy. Choose, given a
  * probability (pEpsilon), one from the following:
  *  - Uniformly random action among all possible actions
  *  - Sample action guided by the policy's probability distribution
  ****************************************************************************/

    if (sample["Mode"] == "Training")
    {
      // Producing random (uniform) number for the selection of the action
      const float x = _uniformGenerator->getRandomNumber();

      // Categorical action sampled from action probabilites (from ACER paper [Wang2017])
      float curSum = 0.0;
      for (actionIdx = 0; actionIdx < pActions.size() - 1; actionIdx++)
      {
        curSum += pActions[actionIdx];
        if (x < curSum) break;
      }

      // NOTE: In original DQN paper [Minh2015] we choose max
      // actionIdx = std::distance(pActions.begin(), std::max_element(pActions.begin(), pActions.end()));
    }

    /*****************************************************************************
  * During testing, we just select the action with the largest probability
  * given by the policy.
  ****************************************************************************/

    // Finding the best action index from the probabilities
    if (sample["Mode"] == "Testing")
      actionIdx = std::distance(pActions.begin(), std::max_element(pActions.begin(), pActions.end()));

    /*****************************************************************************
  * Storing the action itself
 ****************************************************************************/

    // Storing action itself, its idx, and probabilities
    sample["Policy"][i]["Distribution Parameters"] = qValAndInvTemp;
    sample["Policy"][i]["Action Probabilities"] = pActions;
    sample["Policy"][i]["Action Index"] = actionIdx;
    sample["Policy"][i]["State Value"] = policy.stateValue;
    sample["Action"][i] = _problem->_possibleActions[actionIdx];
  }
}

float Discrete::calculateImportanceWeight(const std::vector<float> &action, const policy_t &curPolicy, const policy_t &oldPolicy)
{
  const auto oldActionIdx = oldPolicy.actionIndex;
  const auto pCurPolicy = curPolicy.actionProbabilities[oldActionIdx];
  const auto pOldPolicy = oldPolicy.actionProbabilities[oldActionIdx];

  // Now calculating importance weight for the old s,a experience
  float constexpr epsilon = 0.00000001f;
  float importanceWeight = pCurPolicy / (pOldPolicy + epsilon);

  // Safety checks
  if (importanceWeight > 1024.0f) importanceWeight = 1024.0f;
  if (importanceWeight < -1024.0f) importanceWeight = -1024.0f;

  return importanceWeight;
}

std::vector<float> Discrete::calculateImportanceWeightGradient(const policy_t &curPolicy, const policy_t &oldPolicy)
{
  std::vector<float> grad(_problem->_possibleActions.size() + 1, 0.0);

  const float invTemperature = curPolicy.distributionParameters[_problem->_possibleActions.size()];
  const auto &curDistParams = curPolicy.distributionParameters;
  
  const size_t oldActionIdx = oldPolicy.actionIndex;
  const auto pCurPolicy = curPolicy.actionProbabilities[oldActionIdx];
  const auto pOldPolicy = oldPolicy.actionProbabilities[oldActionIdx];

  // Now calculating importance weight for the old s,a experience
  float constexpr epsilon = 0.00000001f;
  float importanceWeight = pCurPolicy / (pOldPolicy + epsilon);

  // Safety checks
  if (importanceWeight > 1024.0f) importanceWeight = 1024.0f;
  if (importanceWeight < -1024.0f) importanceWeight = -1024.0f;

  float qpSum = 0.;
  // calculate gradient of importance weight wrt. pvals
  for (size_t i = 0; i < _problem->_possibleActions.size(); i++)
  {
    if (i == oldActionIdx)
      grad[i] = importanceWeight * (1. - curPolicy.actionProbabilities[i]) * invTemperature;
    else
      grad[i] = -importanceWeight * curPolicy.actionProbabilities[i] * invTemperature;

    qpSum += curDistParams[i] * curPolicy.actionProbabilities[i];
  }

  // calculate gradient of importance weight wrt. inverse temperature
  grad[_problem->_possibleActions.size()] = importanceWeight * (curDistParams[oldActionIdx] - qpSum);


  return grad;
}

std::vector<float> Discrete::calculateKLDivergenceGradient(const policy_t &oldPolicy, const policy_t &curPolicy)
{
  const float invTemperature = curPolicy.distributionParameters[_problem->_possibleActions.size()];
  const auto &curDistParams = curPolicy.distributionParameters;
  
  std::vector<float> klGrad(_problem->_possibleActions.size() + 1, 0.0);
  
  // Gradient wrt NN output i (qvalue i)
  for (size_t i = 0; i < _problem->_possibleActions.size(); ++i)
  {
    // Iterate over all pvalues
    for (size_t j = 0; j < _problem->_possibleActions.size(); ++j)
    {
      if (i == j)
        klGrad[i] -= invTemperature * oldPolicy.actionProbabilities[j] * (1.0 - curPolicy.actionProbabilities[i]);
      else
        klGrad[i] += invTemperature * oldPolicy.actionProbabilities[j] * curPolicy.actionProbabilities[i];
    }
  }
 
  float qpSum = 0.;
  for (size_t j = 0; j < _problem->_possibleActions.size(); ++j)
    qpSum += curDistParams[j] * curPolicy.actionProbabilities[j];

  // Gradient wrt inverse temperature parameter
  for (size_t j = 0; j < _problem->_possibleActions.size(); ++j)
    klGrad[_problem->_possibleActions.size()] -= oldPolicy.actionProbabilities[j] * (curDistParams[j] - qpSum); 

  return klGrad;
}

void Discrete::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(_k->_js.getJson(), "Variables"))
 for (size_t i = 0; i < _k->_js["Variables"].size(); i++) { 
 } 
 Agent::setConfiguration(js);
 _type = "agent/discrete";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: discrete: \n%s\n", js.dump(2).c_str());
} 

void Discrete::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
 for (size_t i = 0; i <  _k->_variables.size(); i++) { 
 } 
 Agent::getConfiguration(js);
} 

void Discrete::applyModuleDefaults(knlohmann::json& js) 
{

 Agent::applyModuleDefaults(js);
} 

void Discrete::applyVariableDefaults() 
{

 Agent::applyVariableDefaults();
} 

bool Discrete::checkTermination()
{
 bool hasFinished = false;

 hasFinished = hasFinished || Agent::checkTermination();
 return hasFinished;
}

;

} //agent
} //solver
} //korali
;
