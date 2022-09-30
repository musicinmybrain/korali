#include "modules/solver/learner/deepSupervisor/optimizers/fRMSProp/fRMSProp.hpp"
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

void fRMSProp::initialize() {
  FastGradientBasedOptimizer::initialize();
  _r.resize(_nVars);
  _v.resize(_nVars);
  reset();
}

void fRMSProp::reset()
{
  _modelEvaluationCount = 0;
  #pragma omp parallel for simd
  for (size_t i = 0; i < _nVars; i++){
    _currentValue[i] = 0.0f;
    _r[i] = 0.0f;
    _v[i] = 0.0f;
  }
}

void fRMSProp::processResult(std::vector<float> &gradient)
{
  FastGradientBasedOptimizer::preProcessResult(gradient);

  float lambda = _eta; // * std::sqrt((float)_modelEvaluationCount + 1.0f);
  #pragma omp parallel for simd
  for (size_t i = 0; i < _nVars; i++)
  {
    _r[i] = (1.0f-_smoothing) * (gradient[i]*gradient[i]) + _smoothing*_r[i];
    _v[i] = (_eta / (std::sqrt(_r[i]) + _epsilon)) * -gradient[i];
    _currentValue[i] = _currentValue[i] - _v[i];
  }

  FastGradientBasedOptimizer::postProcessResult(_currentValue);
}

void fRMSProp::setConfiguration(knlohmann::json& js) 
{
  if (isDefined(js, "Results"))  eraseValue(js, "Results");

  if (isDefined(js, "r"))
  {
    try
    {
      _r = js["r"].get<std::vector<float>>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ fRMSProp ] \n + Key:    ['r']\n%s", e.what());
    }
    eraseValue(js, "r");
  }
  if (isDefined(js, "v"))
  {
    try
    {
      _v = js["v"].get<std::vector<float>>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ fRMSProp ] \n + Key:    ['v']\n%s", e.what());
    }
    eraseValue(js, "v");
  }
  if (isDefined(js, "Smoothing"))
  {
    try
    {
      _smoothing = js["Smoothing"].get<float>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ fRMSProp ] \n + Key:    ['Smoothing']\n%s", e.what());
    }
    eraseValue(js, "Smoothing");
  }
  if (isDefined(js, "Epsilon"))
  {
    try
    {
      _epsilon = js["Epsilon"].get<float>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ fRMSProp ] \n + Key:    ['Epsilon']\n%s", e.what());
    }
    eraseValue(js, "Epsilon");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Epsilon'] required by fRMSProp.\n"); 

 FastGradientBasedOptimizer::setConfiguration(js);
 _type = "learner/deepSupervisor/optimizers/fRMSProp";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: fRMSProp: \n%s\n", js.dump(2).c_str());
} 

void fRMSProp::getConfiguration(knlohmann::json& js) 
{

  js["Type"] = _type;
   js["Epsilon"] = _epsilon;
   js["r"] = _r;
   js["v"] = _v;
   js["Smoothing"] = _smoothing;
 FastGradientBasedOptimizer::getConfiguration(js);
} 

void fRMSProp::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Epsilon\": 1e-08, \"Eta\": 0.001, \"Smoothing\": 0.999}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 FastGradientBasedOptimizer::applyModuleDefaults(js);
} 

void fRMSProp::applyVariableDefaults() 
{

 FastGradientBasedOptimizer::applyVariableDefaults();
} 

;

} //optimizer
} //learner
} //solver
} //korali
;
