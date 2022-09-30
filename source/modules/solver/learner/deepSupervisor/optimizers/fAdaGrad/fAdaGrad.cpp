#include "modules/solver/learner/deepSupervisor/optimizers/fAdaGrad/fAdaGrad.hpp"
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

void fAdaGrad::initialize() {
  FastGradientBasedOptimizer::initialize();
  _gdiag.resize(_nVars, 0.0f);
  reset();
}

void fAdaGrad::reset()
{
  _modelEvaluationCount = 0;
  #pragma omp parallel for simd
  for (size_t i = 0; i < _nVars; i++){
    _currentValue[i] = 0.0f;
    _gdiag[i] = 0.0f;
  }
}

void fAdaGrad::processResult(std::vector<float> &gradient)
{
  FastGradientBasedOptimizer::preProcessResult(gradient);

  #pragma omp parallel for simd
  for (size_t i = 0; i < _nVars; i++)
  {
    _gdiag[i] = _gdiag[i] + (gradient[i] * gradient[i]);
    _currentValue[i] += (_eta / std::sqrt(_gdiag[i] + _epsilon)) * gradient[i];
  }

  FastGradientBasedOptimizer::postProcessResult(_currentValue);
}

void fAdaGrad::setConfiguration(knlohmann::json& js) 
{
  if (isDefined(js, "Results"))  eraseValue(js, "Results");

  if (isDefined(js, "Gdiag"))
  {
    try
    {
      _gdiag = js["Gdiag"].get<std::vector<float>>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ fAdaGrad ] \n + Key:    ['Gdiag']\n%s", e.what());
    }
    eraseValue(js, "Gdiag");
  }
  if (isDefined(js, "Epsilon"))
  {
    try
    {
      _epsilon = js["Epsilon"].get<float>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ fAdaGrad ] \n + Key:    ['Epsilon']\n%s", e.what());
    }
    eraseValue(js, "Epsilon");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Epsilon'] required by fAdaGrad.\n"); 

 FastGradientBasedOptimizer::setConfiguration(js);
 _type = "learner/deepSupervisor/optimizers/fAdaGrad";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: fAdaGrad: \n%s\n", js.dump(2).c_str());
} 

void fAdaGrad::getConfiguration(knlohmann::json& js) 
{

  js["Type"] = _type;
   js["Epsilon"] = _epsilon;
   js["Gdiag"] = _gdiag;
 FastGradientBasedOptimizer::getConfiguration(js);
} 

void fAdaGrad::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Epsilon\": 1e-08, \"Eta\": 0.001}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 FastGradientBasedOptimizer::applyModuleDefaults(js);
} 

void fAdaGrad::applyVariableDefaults() 
{

 FastGradientBasedOptimizer::applyVariableDefaults();
} 

;

} //optimizer
} //learner
} //solver
} //korali
;
