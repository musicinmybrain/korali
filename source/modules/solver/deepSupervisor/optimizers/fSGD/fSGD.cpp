#include "modules/solver/deepSupervisor/optimizers/fSGD/fSGD.hpp"

namespace korali
{
;

void fSGD::initialize()
{
  fGradientBasedOptimizer::initialize();
}

void fSGD::reset()
{
  
}

void fSGD::processResult(std::vector<float> &gradient)
{
  fGradientBasedOptimizer::preProcessResult(gradient);

  // Compute gradient norm and apply clipping
  float l2norm = 0;
#pragma omp parallel for simd reduction(+ : l2norm)
  for (size_t i = 0; i < _nVars; i++)
    l2norm += gradient[i] * gradient[i];
  l2norm = std::sqrt(l2norm);
  if (l2norm > _clippingThreshold)
  {
#pragma omp parallel for simd
    for (size_t i = 0; i < _nVars; i++)
      gradient[i] = _clippingThreshold * gradient[i] / l2norm;
  }

#pragma omp parallel for simd
  for (size_t i = 0; i < _nVars; i++)
    _currentValue[i] += _eta * gradient[i];

  fGradientBasedOptimizer::postProcessResult(_currentValue);
}

void fSGD::printInternals()
{
  printf("_currentValue[i]:\n");
  for (size_t i = 0; i < 10; i++)
    printf("%f\n", _currentValue[i]);
  fflush(stdout);
}

void fSGD::setConfiguration(knlohmann::json& js) 
{
 if (isDefined(js, "Results"))  eraseValue(js, "Results");

 if (isDefined(js, "Clipping Threshold"))
 {
 try { _clippingThreshold = js["Clipping Threshold"].get<float>();
} catch (const std::exception& e)
 { KORALI_LOG_ERROR(" + Object: [ fSGD ] \n + Key:    ['Clipping Threshold']\n%s", e.what()); } 
   eraseValue(js, "Clipping Threshold");
 }
  else   KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Clipping Threshold'] required by fSGD.\n"); 

 fGradientBasedOptimizer::setConfiguration(js);
 _type = "deepSupervisor/optimizers/fSGD";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: fSGD: \n%s\n", js.dump(2).c_str());
} 

void fSGD::getConfiguration(knlohmann::json& js) 
{

 js["Type"] = _type;
   js["Clipping Threshold"] = _clippingThreshold;
 fGradientBasedOptimizer::getConfiguration(js);
} 

void fSGD::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 fGradientBasedOptimizer::applyModuleDefaults(js);
} 

void fSGD::applyVariableDefaults() 
{

 fGradientBasedOptimizer::applyVariableDefaults();
} 

;

} //korali
;
