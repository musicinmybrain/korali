#include "modules/problem/learning/unsupervisedLearning/unsupervisedLearning.hpp"

namespace korali
{
namespace problem
{
namespace learning
{
;

void UnsupervisedLearning::initialize()
{
  Learning::initialize();
  // Checking batch size
}

void UnsupervisedLearning::verifyData()
{
  Learning::verifyData();
}

void UnsupervisedLearning::setConfiguration(knlohmann::json& js) 
{
  if (isDefined(js, "Results"))  eraseValue(js, "Results");

  bool detectedCompatibleSolver = false; 
  std::string solverName = toLower(_k->_js["Solver"]["Type"]); 
  std::string candidateSolverName; 
  solverName.erase(remove_if(solverName.begin(), solverName.end(), isspace), solverName.end()); 
   candidateSolverName = toLower("Learner/DeepSupervisor"); 
   candidateSolverName.erase(remove_if(candidateSolverName.begin(), candidateSolverName.end(), isspace), candidateSolverName.end()); 
   if (solverName.rfind(candidateSolverName, 0) == 0) detectedCompatibleSolver = true;
   candidateSolverName = toLower("Learner/Gaussian Process"); 
   candidateSolverName.erase(remove_if(candidateSolverName.begin(), candidateSolverName.end(), isspace), candidateSolverName.end()); 
   if (solverName.rfind(candidateSolverName, 0) == 0) detectedCompatibleSolver = true;
  if (detectedCompatibleSolver == false) KORALI_LOG_ERROR(" + Specified solver (%s) is not compatible with problem of type: unsupervisedLearning\n",  _k->_js["Solver"]["Type"].dump(1).c_str()); 

 Learning::setConfiguration(js);
 _type = "learning/unsupervisedLearning";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: unsupervisedLearning: \n%s\n", js.dump(2).c_str());
} 

void UnsupervisedLearning::getConfiguration(knlohmann::json& js) 
{

  js["Type"] = _type;
 Learning::getConfiguration(js);
} 

void UnsupervisedLearning::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Learning::applyModuleDefaults(js);
} 

void UnsupervisedLearning::applyVariableDefaults() 
{

 Learning::applyVariableDefaults();
} 

;

} //learning
} //problem
} //korali
;
