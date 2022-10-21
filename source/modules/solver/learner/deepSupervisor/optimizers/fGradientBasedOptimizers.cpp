#include "modules/solver/learner/deepSupervisor/optimizers/fGradientBasedOptimizers.hpp"

namespace korali
{
namespace solver
{
namespace learner
{
namespace optimizer
{
;

void FastGradientBasedOptimizer::initialize()
{
  _currentValue.resize(_nVars, 0.0f);
};

void FastGradientBasedOptimizer::preProcessResult(std::vector<float> &gradient)
{
  if (gradient.size() != _nVars)
  {
    fprintf(stderr, "Size of sample's gradient evaluations vector (%lu) is different from the number of problem variables defined (%lu).\n", gradient.size(), _nVars);
    throw std::runtime_error("Bad Inputs for Optimizer.");
  }
  for (const float v : gradient)
  {
    if (!std::isfinite(v))
      KORALI_LOG_ERROR("\nOptimizer recieved non-finite gradient");
  }
};

void FastGradientBasedOptimizer::postProcessResult(std::vector<float> &parameters)
{
  for (const float v : parameters)
    if (!std::isfinite(v))
      KORALI_LOG_ERROR("\nOptimizer calculated non-finite hyperparameters");
  _modelEvaluationCount++;
};

void FastGradientBasedOptimizer::setConfiguration(knlohmann::json& js) 
{
  if (isDefined(js, "Results"))  eraseValue(js, "Results");

  if (isDefined(js, "Model Evaluation Count"))
  {
    try
    {
      _modelEvaluationCount = js["Model Evaluation Count"].get<size_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ optimizers ] \n + Key:    ['Model Evaluation Count']\n%s", e.what());
    }
    eraseValue(js, "Model Evaluation Count");
  }
  if (isDefined(js, "N Vars"))
  {
    try
    {
      _nVars = js["N Vars"].get<size_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ optimizers ] \n + Key:    ['N Vars']\n%s", e.what());
    }
    eraseValue(js, "N Vars");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['N Vars'] required by optimizers.\n"); 

  if (isDefined(js, "Eta"))
  {
    try
    {
      _eta = js["Eta"].get<float>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ optimizers ] \n + Key:    ['Eta']\n%s", e.what());
    }
    eraseValue(js, "Eta");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Eta'] required by optimizers.\n"); 

 Module::setConfiguration(js);
 _type = "learner/deepSupervisor/optimizers";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: optimizers: \n%s\n", js.dump(2).c_str());
} 

void FastGradientBasedOptimizer::getConfiguration(knlohmann::json& js) 
{

  js["Type"] = _type;
   js["N Vars"] = _nVars;
   js["Eta"] = _eta;
   js["Model Evaluation Count"] = _modelEvaluationCount;
 Module::getConfiguration(js);
} 

void FastGradientBasedOptimizer::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"N Vars\": 0, \"Model Evaluation Count\": 0}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Module::applyModuleDefaults(js);
} 

void FastGradientBasedOptimizer::applyVariableDefaults() 
{

 Module::applyVariableDefaults();
} 

;

} //optimizer
} //learner
} //solver
} //korali
;
