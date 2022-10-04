/** \namespace optimizer
* @brief Namespace declaration for modules of type: optimizer.
*/

/** \file
* @brief Header file for module: FastGradientBasedOptimizer.
*/

/** \dir solver/learner/deepSupervisor/optimizers
* @brief Contains code, documentation, and scripts for module: FastGradientBasedOptimizer.
*/

#pragma once

#include <cstddef>
#include <vector>
#include <cstdlib>
#include <stdexcept>
#include <cmath>
#include "auxiliar/logger.hpp"
#include "modules/module.hpp"

namespace korali
{
namespace solver
{
namespace learner
{
namespace optimizer
{
;

/**
* @brief Class declaration for module: FastGradientBasedOptimizer.
*/
class FastGradientBasedOptimizer : public Module
{
  public: 
  /**
  * @brief Size of variable space size(x) of f(x)
  */
   size_t _nVars;
  /**
  * @brief Step size/learning rate for current iterration.
  */
   float _eta;
  /**
  * @brief [Internal Use] Counter for the current iterration
  */
   size_t _modelEvaluationCount;
  
 
  /**
  * @brief Obtains the entire current state and configuration of the module.
  * @param js JSON object onto which to save the serialized state of the module.
  */
  void getConfiguration(knlohmann::json& js) override;
  /**
  * @brief Sets the entire state and configuration of the module, given a JSON object.
  * @param js JSON object from which to deserialize the state of the module.
  */
  void setConfiguration(knlohmann::json& js) override;
  /**
  * @brief Applies the module's default configuration upon its creation.
  * @param js JSON object containing user configuration. The defaults will not override any currently defined settings.
  */
  void applyModuleDefaults(knlohmann::json& js) override;
  /**
  * @brief Applies the module's default variable configuration to each variable in the Experiment upon creation.
  */
  void applyVariableDefaults() override;
  

    // VARIABLES =================================================
    /**
    * @brief [Internal Use] Current value of parameters.
    */
    std::vector<float> _currentValue;
    /**
    * @brief Wether to add weight decay.
    */
    bool _addWeightDecay{false};
    /**
    * @brief Weight Decay To add
    */
    std::vector<float> _weightDecay;
    // FUNCTIONS =================================================
    FastGradientBasedOptimizer() = default;
    ~FastGradientBasedOptimizer() = default;
    /**
    * @brief Takes a sample evaluation and its gradient and calculates the next set of parameters
    * @param gradient The gradient of the objective function at the current set of parameters
    */
    virtual void processResult(std::vector<float> &gradient) = 0;
    /**
    * @brief size checks
    * @param gradient input gradients to be checked
    */
    virtual void preProcessResult(std::vector<float> &gradient);
    /**
    * @brief checks for infinity values and increments modelEvaluationCount
    * @param parameters calculated by our optimizer
    */
    virtual void postProcessResult(std::vector<float> &parameters);
    /**
    * @brief Restores the optimizer to the initial state
    */
    virtual void reset() = 0;
    /**
    * @brief Wether the Optimizer Implements Weight Decay set in initialize.
    */
    virtual bool _implementsWeightDecay(){ return 0; };
    // Overriden FUNCTIONS =======================================
    virtual void initialize() override;
};

} //optimizer
} //learner
} //solver
} //korali
;

