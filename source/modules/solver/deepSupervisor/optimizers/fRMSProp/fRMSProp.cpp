#include "modules/solver/deepSupervisor/optimizers/fRMSProp/fRMSProp.hpp"

namespace korali
{
;

void fRMSProp::initialize()
{
  fGradientBasedOptimizer::initialize();
  _r.resize(_nVars, 0.0f);
  _v.resize(_nVars, 0.0f);
  reset();
}

void fRMSProp::reset()
{
#pragma omp parallel for simd
  for (size_t i = 0; i < _nVars; i++)
  {
    _currentValue[i] = 0.0f;
    _r[i] = 0.0f;
    _v[i] = 0.0f;
  }
}

void fRMSProp::processResult(std::vector<float> &gradient)
{
  fGradientBasedOptimizer::preProcessResult(gradient);

#pragma omp parallel for simd
  for (size_t i = 0; i < _nVars; i++)
  {
    _r[i] = (1.0f - _decay) * (gradient[i] * gradient[i]) + _decay * _r[i];
    _v[i] = (_eta / (std::sqrt(_r[i]) + _epsilon)) * -gradient[i];
    _currentValue[i] = _currentValue[i] - _v[i];
  }

  fGradientBasedOptimizer::postProcessResult(_currentValue);
}

void fRMSProp::printInternals()
{
  printf("_currentValue[i], _r[i], _v[i]:\n");
  for (size_t i = 0; i < 10; i++)
    printf("%f %f %f\n", _currentValue[i], _r[i], _v[i]);
  fflush(stdout);
}

void fRMSProp::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "r"))
 {
 try { _r = js["r"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ fRMSProp ] \n + Key:    ['r']\n%s", e.what()); } 
   eraseValue(js, "r");
 }

 if (isDefined(js, "v"))
 {
 try { _v = js["v"].get<std::vector<float>>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ fRMSProp ] \n + Key:    ['v']\n%s", e.what()); } 
   eraseValue(js, "v");
 }

 if (isDefined(js, "Decay"))
 {
 try { _decay = js["Decay"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ fRMSProp ] \n + Key:    ['Decay']\n%s", e.what()); } 
   eraseValue(js, "Decay");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Decay'] required by fRMSProp.\n"); 

 fGradientBasedOptimizer::setConfiguration(js);
 _type = "deepSupervisor/optimizers/fRMSProp";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: fRMSProp: \n%s\n", js.dump(2).c_str());
} 

void fRMSProp::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Decay"] = _decay;
   js["r"] = _r;
   js["v"] = _v;
 fGradientBasedOptimizer::getConfiguration(js);
} 

void fRMSProp::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Decay\": 0.999}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 fGradientBasedOptimizer::applyModuleDefaults(js);
} 

void fRMSProp::applyVariableDefaults() 
{

 fGradientBasedOptimizer::applyVariableDefaults();
} 

;

} //korali
;
