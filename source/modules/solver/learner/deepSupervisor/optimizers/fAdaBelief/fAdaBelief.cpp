#include "modules/solver/learner/deepSupervisor/optimizers/fAdaBelief/fAdaBelief.hpp"
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

void fAdaBelief::initialize() {
  _secondCentralMoment.resize(_nVars, 0.0f);
  fAdam::initialize();
  reset();
}

void fAdaBelief::reset()
{
  _modelEvaluationCount = 0;
  _beta1Pow = 1.0f;
  _beta2Pow = 1.0f;
  #pragma omp parallel for simd
  for (size_t i = 0; i < _nVars; i++){
    _currentValue[i] = 0.0f;
    _firstMoment[i] = 0.0f;
    _secondMoment[i] = 0.0f;
    _secondCentralMoment[i] = 0.0f;;
  }
}

void fAdaBelief::processResult(std::vector<float> &gradient)
{
  fAdam::preProcessResult(gradient);

  float biasCorrectedFirstMoment;
  float secondMomentGradientDiff;
  float biasCorrectedSecondCentralMoment;

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
    biasCorrectedFirstMoment = _firstMoment[i] * firstCentralMomentFactor;
    secondMomentGradientDiff = gradient[i] + _firstMoment[i];
    _secondCentralMoment[i] = _beta2 * _secondCentralMoment[i] + notBeta2 * secondMomentGradientDiff * secondMomentGradientDiff;

    biasCorrectedSecondCentralMoment = _secondCentralMoment[i] * secondCentralMomentFactor;
    _currentValue[i] -= _eta / (std::sqrt(biasCorrectedSecondCentralMoment) + _epsilon) * biasCorrectedFirstMoment;
  }

  fAdam::postProcessResult(_currentValue);
}

void fAdaBelief::setConfiguration(knlohmann::json& js) 
{
  if (isDefined(js, "Results"))  eraseValue(js, "Results");

  if (isDefined(js, "Second Central Moment"))
  {
    try
    {
      _secondCentralMoment = js["Second Central Moment"].get<std::vector<double>>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ fAdaBelief ] \n + Key:    ['Second Central Moment']\n%s", e.what());
    }
    eraseValue(js, "Second Central Moment");
  }
 fAdam::setConfiguration(js);
 _type = "learner/deepSupervisor/optimizers/fAdaBelief";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: fAdaBelief: \n%s\n", js.dump(2).c_str());
} 

void fAdaBelief::getConfiguration(knlohmann::json& js) 
{

  js["Type"] = _type;
   js["Second Central Moment"] = _secondCentralMoment;
 fAdam::getConfiguration(js);
} 

void fAdaBelief::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 fAdam::applyModuleDefaults(js);
} 

void fAdaBelief::applyVariableDefaults() 
{

 fAdam::applyVariableDefaults();
} 

;

} //optimizer
} //learner
} //solver
} //korali
;
