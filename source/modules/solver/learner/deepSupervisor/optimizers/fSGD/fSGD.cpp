#include "modules/solver/learner/deepSupervisor/optimizers/fSGD/fSGD.hpp"
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

void fSGD::initialize() {
  FastGradientBasedOptimizer::initialize();
  reset();
}

void fSGD::reset()
{
  _modelEvaluationCount = 0;
  #pragma omp parallel for simd
  for (size_t i = 0; i < _nVars; i++)
    _currentValue[i] = 0.0f;
}

void fSGD::processResult(std::vector<float> &gradient)
{
  FastGradientBasedOptimizer::preProcessResult(gradient);

  // clip gradients
  clipGradients(gradient);

  #pragma omp parallel for simd
  for (size_t i = 0; i < _nVars; i++)
  {
    _currentValue[i] += _eta * gradient[i];
  }

  FastGradientBasedOptimizer::postProcessResult(_currentValue);
}

void fSGD::clipGradients(std::vector<float> &gradient){
  if(_gradientClipping == "value-clipping"){
    float l2norm = 0;
    #pragma omp parallel for simd reduction(+:l2norm)
    for(size_t i = 0; i < _nVars; i++)
      l2norm += gradient[i]*gradient[i];
    l2norm = sqrt(l2norm);
    if (l2norm > _clippingThreshold){
      #pragma omp parallel for simd
      for(size_t i = 0; i < _nVars; i++)
        gradient[i] = _clippingThreshold*gradient[i]/l2norm;
    }
  };
}

void fSGD::setConfiguration(knlohmann::json& js) 
{
  if (isDefined(js, "Results"))  eraseValue(js, "Results");

 FastGradientBasedOptimizer::setConfiguration(js);
 _type = "learner/deepSupervisor/optimizers/fSGD";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: fSGD: \n%s\n", js.dump(2).c_str());
} 

void fSGD::getConfiguration(knlohmann::json& js) 
{

  js["Type"] = _type;
 FastGradientBasedOptimizer::getConfiguration(js);
} 

void fSGD::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Eta\": 0.0001}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 FastGradientBasedOptimizer::applyModuleDefaults(js);
} 

void fSGD::applyVariableDefaults() 
{

 FastGradientBasedOptimizer::applyVariableDefaults();
} 

;

} //optimizer
} //learner
} //solver
} //korali
;
