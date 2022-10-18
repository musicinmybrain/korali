/** \namespace optimizer
* @brief Namespace declaration for modules of type: optimizer.
*/

/** \file
* @brief Header file for module: fRMSProp.
*/

/** \dir solver/learner/deepSupervisor/optimizers/fRMSProp
* @brief Contains code, documentation, and scripts for module: fRMSProp.
*/

#pragma once

#include "modules/solver/learner/deepSupervisor/optimizers/fGradientBasedOptimizers.hpp"
#include <string>
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

/**
* @brief Class declaration for module: fRMSProp.
*/
class fRMSProp : public FastGradientBasedOptimizer
{
  public: 
  /**
  * @brief Term to guard agains numerical instability.
  */
   float _epsilon;
  /**
  * @brief [Internal Use] Square gradient contribution.
  */
   std::vector<float> _r;
  /**
  * @brief [Internal Use] Update rule.
  */
   std::vector<float> _v;
  /**
  * @brief [Internal Use] Smoothing term between next squared grad and previous v contribution.
  */
   float _smoothing;
  
 
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
  // FUNCTIONS =================================================
  // OVERRIDEN FUNCTIONS =======================================
  virtual void initialize() override;
  virtual void processResult(std::vector<float> &gradient) override;
  virtual void reset() override;
};

} //optimizer
} //learner
} //solver
} //korali
;
