#include "modules/problem/learning/supervisedLearning/supervisedLearning.hpp"

namespace korali
{
namespace problem
{
namespace learning
{
;

void SupervisedLearning::initialize()
{
  Learning::initialize();
  if (_trainingBatchSize == 0) KORALI_LOG_ERROR("Empty input batch provided.\n");
  if (_solutionSize == 0) KORALI_LOG_ERROR("Empty solution vector size provided.\n");
}

void SupervisedLearning::verifyData()
{
  // Checking for empty input and solution data
  Learning::verifyData();
  if (_solutionData.size() == 0) KORALI_LOG_ERROR("Empty solution dataset provided.\n");
  // Checking that all timestep entries have the correct size
  // Checking batch size for solution data
  if (_inputData.size() != _solutionData.size())
    KORALI_LOG_ERROR("The provided number of training targets (%lu) is different than the number of training samples (%lu).\n", _solutionData.size(), _inputData.size());

  // Checking that all solution batch entries have the correct size
  for (size_t b = 0; b < _solutionData.size(); b++)
    if (_solutionData[b].size() != _solutionSize)
      KORALI_LOG_ERROR("Solution vector size of batch %lu is inconsistent. Size: %lu - Expected: %lu.\n", b, _solutionData[b].size(), _solutionSize);
}

void SupervisedLearning::setConfiguration(knlohmann::json& js) 
{
  if (isDefined(js, "Results"))  eraseValue(js, "Results");

  if (isDefined(js, "Training Batch Size"))
  {
    try
    {
      _trainingBatchSize = js["Training Batch Size"].get<size_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ supervisedLearning ] \n + Key:    ['Training Batch Size']\n%s", e.what());
    }
    eraseValue(js, "Training Batch Size");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Training Batch Size'] required by supervisedLearning.\n"); 

  if (isDefined(js, "Validation Batch Size"))
  {
    try
    {
      _validationBatchSize = js["Validation Batch Size"].get<ssize_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ supervisedLearning ] \n + Key:    ['Validation Batch Size']\n%s", e.what());
    }
    eraseValue(js, "Validation Batch Size");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Validation Batch Size'] required by supervisedLearning.\n"); 

  if (isDefined(js, "Data", "Validation", "Input"))
  {
    try
    {
      _dataValidationInput = js["Data"]["Validation"]["Input"].get<std::vector<std::vector<std::vector<float>>>>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ supervisedLearning ] \n + Key:    ['Data']['Validation']['Input']\n%s", e.what());
    }
    eraseValue(js, "Data", "Validation", "Input");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Data']['Validation']['Input'] required by supervisedLearning.\n"); 

  if (isDefined(js, "Data", "Validation", "Solution"))
  {
    try
    {
      _dataValidationSolution = js["Data"]["Validation"]["Solution"].get<std::vector<std::vector<float>>>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ supervisedLearning ] \n + Key:    ['Data']['Validation']['Solution']\n%s", e.what());
    }
    eraseValue(js, "Data", "Validation", "Solution");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Data']['Validation']['Solution'] required by supervisedLearning.\n"); 

  if (isDefined(js, "Solution", "Data"))
  {
    try
    {
      _solutionData = js["Solution"]["Data"].get<std::vector<std::vector<float>>>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ supervisedLearning ] \n + Key:    ['Solution']['Data']\n%s", e.what());
    }
    eraseValue(js, "Solution", "Data");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Solution']['Data'] required by supervisedLearning.\n"); 

  if (isDefined(js, "Solution", "Size"))
  {
    try
    {
      _solutionSize = js["Solution"]["Size"].get<size_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ supervisedLearning ] \n + Key:    ['Solution']['Size']\n%s", e.what());
    }
    eraseValue(js, "Solution", "Size");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Solution']['Size'] required by supervisedLearning.\n"); 

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
  if (detectedCompatibleSolver == false) KORALI_LOG_ERROR(" + Specified solver (%s) is not compatible with problem of type: supervisedLearning\n",  _k->_js["Solver"]["Type"].dump(1).c_str()); 

 Learning::setConfiguration(js);
 _type = "learning/supervisedLearning";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: supervisedLearning: \n%s\n", js.dump(2).c_str());
} 

void SupervisedLearning::getConfiguration(knlohmann::json& js) 
{

  js["Type"] = _type;
   js["Training Batch Size"] = _trainingBatchSize;
   js["Validation Batch Size"] = _validationBatchSize;
   js["Data"]["Validation"]["Input"] = _dataValidationInput;
   js["Data"]["Validation"]["Solution"] = _dataValidationSolution;
   js["Solution"]["Data"] = _solutionData;
   js["Solution"]["Size"] = _solutionSize;
 Learning::getConfiguration(js);
} 

void SupervisedLearning::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Max Timesteps\": 1, \"Input\": {\"Data\": []}, \"Solution\": {\"Data\": []}, \"Data\": {\"Validation\": {\"Input\": [], \"Solution\": []}}, \"Training Batch Size\": 0, \"Testing Batch Sizes\": [], \"Testing Batch Size\": 0, \"Validation Batch Size\": -1}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Learning::applyModuleDefaults(js);
} 

void SupervisedLearning::applyVariableDefaults() 
{

 Learning::applyVariableDefaults();
} 

;

} //learning
} //problem
} //korali
;
