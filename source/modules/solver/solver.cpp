#include "modules/solver/solver.hpp"

namespace korali
{
;

void Solver::printGenerationAfter()
{
  _k->_logger->logInfo("Detailed", "Experiment: %lu - Generation Time: %.3fs\n", _k->_experimentId, _k->_genTime);
};

void Solver::printGenerationBefore()
{
  _k->_logger->logInfo("minimal", "--------------------------------------------------------------------\n");
  _k->_logger->logInfo("minimal", "current generation: #%zu: \n", _k->_currentGeneration);
};

void Solver::printRunAfter()
{
  _k->_logger->logInfo("Minimal", "--------------------------------------------------------------------\n");
  _k->_logger->logInfo("Minimal", "%s finished correctly.\n", this->getType().c_str());
  for (auto &crit : _terminationCriteria)
  {
    _k->_logger->logInfo("Normal", "Termination Criterion Met: %s\n", crit.c_str());
  }
  _k->_logger->logInfo("Normal", "Final Generation: %lu\n", _k->_currentGeneration);
  _k->_logger->logInfo("Normal", "Elapsed Time: %.3fs\n", _k->_genTime);
};

void Solver::setInitialConfiguration(){};

void Solver::setConfiguration(knlohmann::json& js) 
{
  if (isDefined(js, "Results"))  eraseValue(js, "Results");

  if (isDefined(js, "Variable Count"))
  {
    try
    {
      _variableCount = js["Variable Count"].get<size_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ solver ] \n + Key:    ['Variable Count']\n%s", e.what());
    }
    eraseValue(js, "Variable Count");
  }
  if (isDefined(js, "Model Evaluation Count"))
  {
    try
    {
      _modelEvaluationCount = js["Model Evaluation Count"].get<size_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ solver ] \n + Key:    ['Model Evaluation Count']\n%s", e.what());
    }
    eraseValue(js, "Model Evaluation Count");
  }
  if (isDefined(js, "Termination Criteria", "Max Model Evaluations"))
  {
    try
    {
      _maxModelEvaluations = js["Termination Criteria"]["Max Model Evaluations"].get<size_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ solver ] \n + Key:    ['Termination Criteria']['Max Model Evaluations']\n%s", e.what());
    }
    eraseValue(js, "Termination Criteria", "Max Model Evaluations");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Termination Criteria']['Max Model Evaluations'] required by solver.\n"); 

  if (isDefined(js, "Termination Criteria", "Max Generations"))
  {
    try
    {
      _maxGenerations = js["Termination Criteria"]["Max Generations"].get<size_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ solver ] \n + Key:    ['Termination Criteria']['Max Generations']\n%s", e.what());
    }
    eraseValue(js, "Termination Criteria", "Max Generations");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Termination Criteria']['Max Generations'] required by solver.\n"); 

 Module::setConfiguration(js);
 _type = ".";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: solver: \n%s\n", js.dump(2).c_str());
} 

void Solver::getConfiguration(knlohmann::json& js) 
{

  js["Type"] = _type;
   js["Termination Criteria"]["Max Model Evaluations"] = _maxModelEvaluations;
   js["Termination Criteria"]["Max Generations"] = _maxGenerations;
   js["Variable Count"] = _variableCount;
   js["Model Evaluation Count"] = _modelEvaluationCount;
 Module::getConfiguration(js);
} 

void Solver::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Termination Criteria\": {\"Max Model Evaluations\": 1000000000, \"Max Generations\": 10000000000}, \"Variable Count\": 0, \"Model Evaluation Count\": 0}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Module::applyModuleDefaults(js);
} 

void Solver::applyVariableDefaults() 
{

 Module::applyVariableDefaults();
} 

bool Solver::checkTermination()
{
  bool hasFinished = false;

  if (_maxModelEvaluations <= _modelEvaluationCount)
  {
    _terminationCriteria.push_back("solver['Max Model Evaluations'] = " + std::to_string(_maxModelEvaluations) + ".");
    hasFinished = true;
  }

  if (_k->_currentGeneration > _maxGenerations)
  {
    _terminationCriteria.push_back("solver['Max Generations'] = " + std::to_string(_maxGenerations) + ".");
    hasFinished = true;
  }

  hasFinished = hasFinished || Module::checkTermination();
  return hasFinished;
}

;

} //korali
;
