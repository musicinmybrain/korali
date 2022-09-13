#include "modules/problem/learning/learning.hpp"

namespace korali
{
namespace problem
{
;

void Learning::initialize()
{
  // Checking batch size
  if (_maxTimesteps == 0) KORALI_LOG_ERROR("Need at least one timestep: %lu.\n", _maxTimesteps);
  if (_inputSize == 0) KORALI_LOG_ERROR("Empty input vector size provided.\n");
}

void Learning::verifyData()
{
  // Checking for empty input and solution data
  if (_inputData.size() == 0) KORALI_LOG_ERROR("Empty input dataset provided.\n");
  // Checking that all timestep entries have the correct size
  for (size_t b = 0; b < _inputData.size(); b++)
  {
    if (_inputData[b].size() > _maxTimesteps)
      KORALI_LOG_ERROR("More timesteps (%lu) provided in batch %lu than max specified in the configuration (%lu).\n", _inputData[b].size(), b, _maxTimesteps);

    // Checking that all batch entries have the correct size
    for (size_t t = 0; t < _inputData[b].size(); t++)
      if (_inputData[b][t].size() != _inputSize)
        KORALI_LOG_ERROR("InputData[%zu][%zu].size() = %lu, is inconsistent with specified input size e['Problem']['Input']['Size'] = %lu.\n", b, t, _inputData[b][t].size(), _inputSize);
  }
}

void Learning::setConfiguration(knlohmann::json& js) 
{
  if (isDefined(js, "Results"))  eraseValue(js, "Results");

  if (isDefined(js, "Max Timesteps"))
  {
    try
    {
      _maxTimesteps = js["Max Timesteps"].get<size_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ learning ] \n + Key:    ['Max Timesteps']\n%s", e.what());
    }
    eraseValue(js, "Max Timesteps");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Max Timesteps'] required by learning.\n"); 

  if (isDefined(js, "Testing Batch Sizes"))
  {
    try
    {
      _testingBatchSizes = js["Testing Batch Sizes"].get<std::vector<size_t>>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ learning ] \n + Key:    ['Testing Batch Sizes']\n%s", e.what());
    }
    eraseValue(js, "Testing Batch Sizes");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Testing Batch Sizes'] required by learning.\n"); 

  if (isDefined(js, "Testing Batch Size"))
  {
    try
    {
      _testingBatchSize = js["Testing Batch Size"].get<size_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ learning ] \n + Key:    ['Testing Batch Size']\n%s", e.what());
    }
    eraseValue(js, "Testing Batch Size");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Testing Batch Size'] required by learning.\n"); 

  if (isDefined(js, "Input", "Data"))
  {
    try
    {
      _inputData = js["Input"]["Data"].get<std::vector<std::vector<std::vector<float>>>>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ learning ] \n + Key:    ['Input']['Data']\n%s", e.what());
    }
    eraseValue(js, "Input", "Data");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Input']['Data'] required by learning.\n"); 

  if (isDefined(js, "Input", "Size"))
  {
    try
    {
      _inputSize = js["Input"]["Size"].get<size_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ learning ] \n + Key:    ['Input']['Size']\n%s", e.what());
    }
    eraseValue(js, "Input", "Size");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Input']['Size'] required by learning.\n"); 

  bool detectedCompatibleSolver = false; 
  std::string solverName = toLower(_k->_js["Solver"]["Type"]); 
  std::string candidateSolverName; 
  solverName.erase(remove_if(solverName.begin(), solverName.end(), isspace), solverName.end()); 
   candidateSolverName = toLower("Learner/DeepSupervisor"); 
   candidateSolverName.erase(remove_if(candidateSolverName.begin(), candidateSolverName.end(), isspace), candidateSolverName.end()); 
   if (solverName.rfind(candidateSolverName, 0) == 0) detectedCompatibleSolver = true;
   candidateSolverName = toLower("Learner/Gaussian Process"); 
   candidateSolverName.erase(remove_if(candidateSolverName.begin(), candidateSolverName.end(), isspace), candidateSolverName.end()); 
   if (solverName.rfind(candidateSolverName, 0) == 0) detectedCompatibleSolver = true;
  if (detectedCompatibleSolver == false) KORALI_LOG_ERROR(" + Specified solver (%s) is not compatible with problem of type: learning\n",  _k->_js["Solver"]["Type"].dump(1).c_str()); 

 Problem::setConfiguration(js);
 _type = "learning";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: learning: \n%s\n", js.dump(2).c_str());
} 

void Learning::getConfiguration(knlohmann::json& js) 
{

  js["Type"] = _type;
   js["Max Timesteps"] = _maxTimesteps;
   js["Testing Batch Sizes"] = _testingBatchSizes;
   js["Testing Batch Size"] = _testingBatchSize;
   js["Input"]["Data"] = _inputData;
   js["Input"]["Size"] = _inputSize;
 Problem::getConfiguration(js);
} 

void Learning::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Max Timesteps\": 1, \"Input\": {\"Data\": []}, \"Testing Batch Sizes\": [], \"Testing Batch Size\": 0}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Problem::applyModuleDefaults(js);
} 

void Learning::applyVariableDefaults() 
{

 Problem::applyVariableDefaults();
} 

;

} //problem
} //korali
;
