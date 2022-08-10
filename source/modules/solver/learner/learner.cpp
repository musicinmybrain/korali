#include "modules/solver/learner/learner.hpp"

namespace korali
{
namespace solver
{
;

std::vector<std::vector<float>> &
Learner::getEvaluation(const std::vector<std::vector<std::vector<float>>> &input)
{
  KORALI_LOG_ERROR("This solver does not provide an evaluate operation.\n");
}

void Learner::setConfiguration(knlohmann::json& js) 
{
  if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(_k->_js.getJson(), "Variables"))
 for (size_t i = 0; i < _k->_js["Variables"].size(); i++) { 
 } 
 Solver::setConfiguration(js);
 _type = "learner";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: learner: \n%s\n", js.dump(2).c_str());
} 

void Learner::getConfiguration(knlohmann::json& js) 
{

  js["Type"] = _type;
 Solver::getConfiguration(js);
} 

void Learner::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Solver::applyModuleDefaults(js);
} 

void Learner::applyVariableDefaults() 
{

 Solver::applyVariableDefaults();
} 

bool Learner::checkTermination()
{
  bool hasFinished = false;

  hasFinished = hasFinished || Solver::checkTermination();
  return hasFinished;
}

;

} //solver
} //korali
;
