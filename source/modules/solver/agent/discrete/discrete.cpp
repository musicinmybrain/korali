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

  _problem->_actionCount = _problem->_possibleActions.size();
  _policyParameterCount = _problem->_actionCount + 1; // q values and inverseTemperature
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
  std::vector<float> grad(_problem->_actionCount + 1, 0.0);

  const float invTemperature = curPolicy.distributionParameters[_problem->_actionCount];
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
  for (size_t i = 0; i < _problem->_actionCount; i++)
  {
    if (i == oldActionIdx)
      grad[i] = importanceWeight * (1. - curPolicy.actionProbabilities[i]) * invTemperature;
    else
      grad[i] = -importanceWeight * curPolicy.actionProbabilities[i] * invTemperature;

    qpSum += curDistParams[i] * curPolicy.actionProbabilities[i];
  }

  // calculate gradient of importance weight wrt. inverse temperature
  grad[_problem->_actionCount] = importanceWeight * (curDistParams[oldActionIdx] - qpSum);

  return grad;
}

std::vector<float> Discrete::calculateKLDivergenceGradient(const policy_t &oldPolicy, const policy_t &curPolicy)
{
  const float invTemperature = curPolicy.distributionParameters[_problem->_actionCount];
  const auto &curDistParams = curPolicy.distributionParameters;

  std::vector<float> klGrad(_problem->_actionCount + 1, 0.0);

  // Gradient wrt NN output i (qvalue i)
  for (size_t i = 0; i < _problem->_actionCount; ++i)
  {
    // Iterate over all pvalues
    for (size_t j = 0; j < _problem->_actionCount; ++j)
    {
      if (i == j)
        klGrad[i] -= invTemperature * oldPolicy.actionProbabilities[j] * (1.0 - curPolicy.actionProbabilities[i]);
      else
        klGrad[i] += invTemperature * oldPolicy.actionProbabilities[j] * curPolicy.actionProbabilities[i];
    }
  }

  float qpSum = 0.;
  for (size_t j = 0; j < _problem->_actionCount; ++j)
    qpSum += curDistParams[j] * curPolicy.actionProbabilities[j];

  // Gradient wrt inverse temperature parameter
  for (size_t j = 0; j < _problem->_actionCount; ++j)
    klGrad[_problem->_actionCount] -= oldPolicy.actionProbabilities[j] * (curDistParams[j] - qpSum);

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
