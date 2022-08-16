#include "modules/problem/integration/integration.hpp"
#include "sample/sample.hpp"

namespace korali
{
namespace problem
{
;

void Integration::initialize()
{
  if (_k->_variables.size() == 0) KORALI_LOG_ERROR("Integration problems require at least one variable.\n");

  for (size_t i = 0; i < _k->_variables.size(); i++)
  {
    if (_k->_variables[i]->_numberOfGridpoints <= 0) KORALI_LOG_ERROR("'Number Of Gridpoints' for variable %s must be a strictly positive integer", _k->_variables[i]->_name.c_str());

    if (_k->_variables[i]->_upperBound <= _k->_variables[i]->_lowerBound) KORALI_LOG_ERROR("'Upper Bound' is not strictly bigger then 'Lower Bound' for variable %s.\n", _k->_variables[i]->_name.c_str());
    double intervalSize = _k->_variables[i]->_upperBound - _k->_variables[i]->_lowerBound;
    double deltaX = intervalSize / (_k->_variables[i]->_numberOfGridpoints - 1);

    if (_integrationMethod == "Monte Carlo")
    {
      bool foundDistribution = false;
      // Validate the _samplingDistribution names
      for (size_t j = 0; j < _k->_distributions.size(); j++)
        if (_k->_variables[i]->_samplingDistribution == _k->_distributions[j]->_name)
        {
          foundDistribution = true;
          _k->_variables[i]->_distributionIndex = j;
        }

      if (foundDistribution == false)
        KORALI_LOG_ERROR("Did not find distribution %s, specified by variable %s\n", _k->_variables[i]->_samplingDistribution.c_str(), _k->_variables[i]->_name.c_str());

      _k->_variables[i]->_quadratureWeights.resize(1);
      _k->_variables[i]->_quadratureWeights[0] = intervalSize / _k->_variables[i]->_numberOfGridpoints;
    }
    else if (_integrationMethod == "Custom")
    {
      if (_k->_variables[i]->_samplePoints.size() != _k->_variables[i]->_quadratureWeights.size())
        KORALI_LOG_ERROR("Number of 'Sample Points' is not equal to number of 'Quadrature Points' provided for variable %s\n", _k->_variables[i]->_name.c_str());
    }
    else
    {
      _k->_variables[i]->_samplePoints.resize(_k->_variables[i]->_numberOfGridpoints);
      for (size_t j = 0; j < _k->_variables[i]->_numberOfGridpoints; j++)
        _k->_variables[i]->_samplePoints[j] = _k->_variables[i]->_lowerBound + j * deltaX;

      _k->_variables[i]->_quadratureWeights.resize(_k->_variables[i]->_numberOfGridpoints);
      if (_integrationMethod == "Rectangle")
      {
        for (size_t j = 0; j < _k->_variables[i]->_numberOfGridpoints; j++)
          _k->_variables[i]->_quadratureWeights[j] = deltaX;
      }
      else if (_integrationMethod == "Trapezoidal")
      {
        for (size_t j = 0; j < _k->_variables[i]->_numberOfGridpoints; j++)
          if (j > 0 && j < _k->_variables[i]->_numberOfGridpoints - 1)
            _k->_variables[i]->_quadratureWeights[j] = deltaX;
          else
            _k->_variables[i]->_quadratureWeights[j] = deltaX / 2.;
      }
      else if (_integrationMethod == "Simpson")
      {
        for (size_t j = 0; j < _k->_variables[i]->_numberOfGridpoints; j++)
          if (j > 0 && j < _k->_variables[i]->_numberOfGridpoints - 1)
          {
            if (j % 2 == 0)
              _k->_variables[i]->_quadratureWeights[j] = 2. * deltaX / 3.;
            else
              _k->_variables[i]->_quadratureWeights[j] = 4. * deltaX / 3.;
          }
          else
            _k->_variables[i]->_quadratureWeights[j] = deltaX / 3.;
      }
    }
  }
}

void Integration::execute(Sample &sample)
{
  // Evaluating Sample
  sample.run(_integrand);

  auto evaluation = KORALI_GET(double, sample, "Evaluation");

  if (std::isnan(evaluation)) KORALI_LOG_ERROR("The function evaluation returned NaN.\n");
}

void Integration::setConfiguration(knlohmann::json& js) 
{
  if (isDefined(js, "Results"))  eraseValue(js, "Results");

  if (isDefined(js, "Integrand"))
  {
    try
    {
      _integrand = js["Integrand"].get<std::uint64_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ integration ] \n + Key:    ['Integrand']\n%s", e.what());
    }
    eraseValue(js, "Integrand");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Integrand'] required by integration.\n"); 

  if (isDefined(js, "Integration Method"))
  {
    try
    {
      _integrationMethod = js["Integration Method"].get<std::string>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ integration ] \n + Key:    ['Integration Method']\n%s", e.what());
    }
      {
        bool validOption = false; 
        if (_integrationMethod == "Rectangle") validOption = true; 
        if (_integrationMethod == "Trapezoidal") validOption = true; 
        if (_integrationMethod == "Simpson") validOption = true; 
        if (_integrationMethod == "Monte Carlo") validOption = true; 
        if (_integrationMethod == "Custom") validOption = true; 
        if (validOption == false) KORALI_LOG_ERROR("Unrecognized value (%s) provided for mandatory setting: ['Integration Method'] required by integration.\n Valid Options are:\n  - Rectangle\n  - Trapezoidal\n  - Simpson\n  - Monte Carlo\n  - Custom\n",_integrationMethod.c_str()); 
      }
    eraseValue(js, "Integration Method");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Integration Method'] required by integration.\n"); 

 if (isDefined(_k->_js.getJson(), "Variables"))
 for (size_t i = 0; i < _k->_js["Variables"].size(); i++) { 
  if (isDefined(_k->_js["Variables"][i], "Lower Bound"))
  {
    try
    {
      _k->_variables[i]->_lowerBound = _k->_js["Variables"][i]["Lower Bound"].get<double>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ integration ] \n + Key:    ['Lower Bound']\n%s", e.what());
    }
    eraseValue(_k->_js["Variables"][i], "Lower Bound");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Lower Bound'] required by integration.\n"); 

  if (isDefined(_k->_js["Variables"][i], "Upper Bound"))
  {
    try
    {
      _k->_variables[i]->_upperBound = _k->_js["Variables"][i]["Upper Bound"].get<double>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ integration ] \n + Key:    ['Upper Bound']\n%s", e.what());
    }
    eraseValue(_k->_js["Variables"][i], "Upper Bound");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Upper Bound'] required by integration.\n"); 

  if (isDefined(_k->_js["Variables"][i], "Number Of Gridpoints"))
  {
    try
    {
      _k->_variables[i]->_numberOfGridpoints = _k->_js["Variables"][i]["Number Of Gridpoints"].get<size_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ integration ] \n + Key:    ['Number Of Gridpoints']\n%s", e.what());
    }
    eraseValue(_k->_js["Variables"][i], "Number Of Gridpoints");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Number Of Gridpoints'] required by integration.\n"); 

  if (isDefined(_k->_js["Variables"][i], "Sampling Distribution"))
  {
    try
    {
      _k->_variables[i]->_samplingDistribution = _k->_js["Variables"][i]["Sampling Distribution"].get<std::string>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ integration ] \n + Key:    ['Sampling Distribution']\n%s", e.what());
    }
    eraseValue(_k->_js["Variables"][i], "Sampling Distribution");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Sampling Distribution'] required by integration.\n"); 

  if (isDefined(_k->_js["Variables"][i], "Distribution Index"))
  {
    try
    {
      _k->_variables[i]->_distributionIndex = _k->_js["Variables"][i]["Distribution Index"].get<size_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ integration ] \n + Key:    ['Distribution Index']\n%s", e.what());
    }
    eraseValue(_k->_js["Variables"][i], "Distribution Index");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Distribution Index'] required by integration.\n"); 

  if (isDefined(_k->_js["Variables"][i], "Sample Points"))
  {
    try
    {
      _k->_variables[i]->_samplePoints = _k->_js["Variables"][i]["Sample Points"].get<std::vector<double>>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ integration ] \n + Key:    ['Sample Points']\n%s", e.what());
    }
    eraseValue(_k->_js["Variables"][i], "Sample Points");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Sample Points'] required by integration.\n"); 

  if (isDefined(_k->_js["Variables"][i], "Quadrature Weights"))
  {
    try
    {
      _k->_variables[i]->_quadratureWeights = _k->_js["Variables"][i]["Quadrature Weights"].get<std::vector<double>>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ integration ] \n + Key:    ['Quadrature Weights']\n%s", e.what());
    }
    eraseValue(_k->_js["Variables"][i], "Quadrature Weights");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Quadrature Weights'] required by integration.\n"); 

 } 
  bool detectedCompatibleSolver = false; 
  std::string solverName = toLower(_k->_js["Solver"]["Type"]); 
  std::string candidateSolverName; 
  solverName.erase(remove_if(solverName.begin(), solverName.end(), isspace), solverName.end()); 
   candidateSolverName = toLower("Integrator"); 
   candidateSolverName.erase(remove_if(candidateSolverName.begin(), candidateSolverName.end(), isspace), candidateSolverName.end()); 
   if (solverName.rfind(candidateSolverName, 0) == 0) detectedCompatibleSolver = true;
  if (detectedCompatibleSolver == false) KORALI_LOG_ERROR(" + Specified solver (%s) is not compatible with problem of type: integration\n",  _k->_js["Solver"]["Type"].dump(1).c_str()); 

 Problem::setConfiguration(js);
 _type = "integration";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: integration: \n%s\n", js.dump(2).c_str());
} 

void Integration::getConfiguration(knlohmann::json& js) 
{

  js["Type"] = _type;
   js["Integrand"] = _integrand;
   js["Integration Method"] = _integrationMethod;
  for (size_t i = 0; i <  _k->_variables.size(); i++) { 
   _k->_js["Variables"][i]["Lower Bound"] = _k->_variables[i]->_lowerBound;
   _k->_js["Variables"][i]["Upper Bound"] = _k->_variables[i]->_upperBound;
   _k->_js["Variables"][i]["Number Of Gridpoints"] = _k->_variables[i]->_numberOfGridpoints;
   _k->_js["Variables"][i]["Sampling Distribution"] = _k->_variables[i]->_samplingDistribution;
   _k->_js["Variables"][i]["Distribution Index"] = _k->_variables[i]->_distributionIndex;
   _k->_js["Variables"][i]["Sample Points"] = _k->_variables[i]->_samplePoints;
   _k->_js["Variables"][i]["Quadrature Weights"] = _k->_variables[i]->_quadratureWeights;
 } 
 Problem::getConfiguration(js);
} 

void Integration::applyModuleDefaults(knlohmann::json& js) 
{

 Problem::applyModuleDefaults(js);
} 

void Integration::applyVariableDefaults() 
{

 std::string defaultString = "{\"Sampling Distribution\": \" \", \"Distribution Index\": -1, \"Sample Points\": [], \"Quadrature Weights\": []}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 if (isDefined(_k->_js.getJson(), "Variables"))
  for (size_t i = 0; i < _k->_js["Variables"].size(); i++) 
   mergeJson(_k->_js["Variables"][i], defaultJs); 
 Problem::applyVariableDefaults();
} 

bool Integration::runOperation(std::string operation, korali::Sample& sample)
{
 bool operationDetected = false;

 if (operation == "Execute")
 {
  execute(sample);
  return true;
 }

 operationDetected = operationDetected || Problem::runOperation(operation, sample);
 if (operationDetected == false) KORALI_LOG_ERROR(" + Operation %s not recognized for problem Integration.\n", operation.c_str());
 return operationDetected;
}

;

} //problem
} //korali
;
