#include "modules/solver/learner/deepSupervisor/optimizers/fAdam/fAdam.hpp"
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

void fAdam::initialize()
{
  FastGradientBasedOptimizer::initialize();
  _firstMoment.resize(_nVars, 0.0f);
  _secondMoment.resize(_nVars, 0.0f);
  _weightDecay.resize(_nVars, 0.0f);
  reset();
}

void fAdam::reset()
{
  _modelEvaluationCount = 0;
  _beta1Pow = 1.0f;
  _beta2Pow = 1.0f;
#pragma omp parallel for simd
  for (size_t i = 0; i < _nVars; i++)
  {
    _currentValue[i] = 0.0f;
    _firstMoment[i] = 0.0f;
    _secondMoment[i] = 0.0f;
    _weightDecay[i] = 0.0f;
  }
}

void fAdam::processResult(std::vector<float> &gradient)
{
  FastGradientBasedOptimizer::preProcessResult(gradient);

  // Calculate powers of beta1 & beta2
  _beta1Pow *= _beta1;
  _beta2Pow *= _beta2;
  const float firstCentralMomentFactor = 1.0f / (1.0f - _beta1Pow);
  const float secondCentralMomentFactor = 1.0f / (1.0f - _beta2Pow);
  const float notBeta1 = 1.0f - _beta1;
  const float notBeta2 = 1.0f - _beta2;

// update first and second moment estimators and parameters
#pragma omp parallel for simd
  for (size_t i = 0; i < _nVars; i++)
  {
    _firstMoment[i] = _beta1 * _firstMoment[i] - notBeta1 * gradient[i];
    _secondMoment[i] = _beta2 * _secondMoment[i] + notBeta2 * gradient[i] * gradient[i];
    _currentValue[i] -= _eta / (std::sqrt(_secondMoment[i] * secondCentralMomentFactor) + _epsilon) * _firstMoment[i] * firstCentralMomentFactor;
    if (_addWeightDecay)
      _currentValue[i] -= _eta * _weightDecay[i];
  }

  FastGradientBasedOptimizer::postProcessResult(_currentValue);
}

void fAdam::setConfiguration(knlohmann::json& js) 
{
  if (isDefined(js, "Results"))  eraseValue(js, "Results");

  if (isDefined(js, "Beta1 Pow"))
  {
    try
    {
      _beta1Pow = js["Beta1 Pow"].get<float>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ fAdam ] \n + Key:    ['Beta1 Pow']\n%s", e.what());
    }
    eraseValue(js, "Beta1 Pow");
  }
  if (isDefined(js, "Beta2 Pow"))
  {
    try
    {
      _beta2Pow = js["Beta2 Pow"].get<float>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ fAdam ] \n + Key:    ['Beta2 Pow']\n%s", e.what());
    }
    eraseValue(js, "Beta2 Pow");
  }
  if (isDefined(js, "First Moment"))
  {
    try
    {
      _firstMoment = js["First Moment"].get<std::vector<double>>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ fAdam ] \n + Key:    ['First Moment']\n%s", e.what());
    }
    eraseValue(js, "First Moment");
  }
  if (isDefined(js, "Second Moment"))
  {
    try
    {
      _secondMoment = js["Second Moment"].get<std::vector<double>>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ fAdam ] \n + Key:    ['Second Moment']\n%s", e.what());
    }
    eraseValue(js, "Second Moment");
  }
  if (isDefined(js, "Epsilon"))
  {
    try
    {
      _epsilon = js["Epsilon"].get<float>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ fAdam ] \n + Key:    ['Epsilon']\n%s", e.what());
    }
    eraseValue(js, "Epsilon");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Epsilon'] required by fAdam.\n"); 

 FastGradientBasedOptimizer::setConfiguration(js);
 _type = "learner/deepSupervisor/optimizers/fAdam";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: fAdam: \n%s\n", js.dump(2).c_str());
} 

void fAdam::getConfiguration(knlohmann::json& js) 
{

  js["Type"] = _type;
   js["Epsilon"] = _epsilon;
   js["Beta1 Pow"] = _beta1Pow;
   js["Beta2 Pow"] = _beta2Pow;
   js["First Moment"] = _firstMoment;
   js["Second Moment"] = _secondMoment;
 FastGradientBasedOptimizer::getConfiguration(js);
} 

void fAdam::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Epsilon\": 1e-08, \"Eta\": 0.001}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 FastGradientBasedOptimizer::applyModuleDefaults(js);
} 

void fAdam::applyVariableDefaults() 
{

 FastGradientBasedOptimizer::applyVariableDefaults();
} 

;

} //optimizer
} //learner
} //solver
} //korali
;
