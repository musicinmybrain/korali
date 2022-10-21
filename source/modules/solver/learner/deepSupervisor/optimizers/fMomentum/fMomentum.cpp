#include "modules/solver/learner/deepSupervisor/optimizers/fMomentum/fMomentum.hpp"
#include <vector>

namespace korali
{
namespace solver
{
namespace learner
{
namespace optimizer
{
;

void fMomentum::initialize()
{
  FastGradientBasedOptimizer::initialize();
  _gsmoothed.resize(_nVars);
  reset();
}

void fMomentum::reset()
{
  _modelEvaluationCount = 0;
#pragma omp parallel for simd
  for (size_t i = 0; i < _nVars; i++)
  {
    _currentValue[i] = 0.0f;
    _gsmoothed[i] = 0.0f;
  }
}

void fMomentum::processResult(std::vector<float> &gradient)
{
  FastGradientBasedOptimizer::preProcessResult(gradient);

  float not_smoothing = 1.0f - _smoothing;
#pragma omp parallel for simd
  for (size_t i = 0; i < _nVars; i++)
  {
    _gsmoothed[i] = not_smoothing * gradient[i] + _smoothing * _gsmoothed[i];
    _currentValue[i] += _eta * _gsmoothed[i];
    // std::fma(_eta, _gsmoothed[i], _currentValue[i]);
  }

  FastGradientBasedOptimizer::postProcessResult(_currentValue);
}

void fMomentum::setConfiguration(knlohmann::json& js) 
{
  if (isDefined(js, "Results"))  eraseValue(js, "Results");

  if (isDefined(js, "G smoothed"))
  {
    try
    {
      _gsmoothed = js["G smoothed"].get<std::vector<float>>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ fMomentum ] \n + Key:    ['G smoothed']\n%s", e.what());
    }
    eraseValue(js, "G smoothed");
  }
  if (isDefined(js, "Smoothing"))
  {
    try
    {
      _smoothing = js["Smoothing"].get<float>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ fMomentum ] \n + Key:    ['Smoothing']\n%s", e.what());
    }
    eraseValue(js, "Smoothing");
  }
 FastGradientBasedOptimizer::setConfiguration(js);
 _type = "learner/deepSupervisor/optimizers/fMomentum";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: fMomentum: \n%s\n", js.dump(2).c_str());
} 

void fMomentum::getConfiguration(knlohmann::json& js) 
{

  js["Type"] = _type;
   js["G smoothed"] = _gsmoothed;
   js["Smoothing"] = _smoothing;
 FastGradientBasedOptimizer::getConfiguration(js);
} 

void fMomentum::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Eta\": 0.001, \"Smoothing\": 0.001}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 FastGradientBasedOptimizer::applyModuleDefaults(js);
} 

void fMomentum::applyVariableDefaults() 
{

 FastGradientBasedOptimizer::applyVariableDefaults();
} 

;

} //optimizer
} //learner
} //solver
} //korali
;
