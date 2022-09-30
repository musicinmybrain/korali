/** \namespace optimizer
* @brief Namespace declaration for modules of type: optimizer.
*/

/** \file
* @brief Header file for module: fAdaBelief.
*/

/** \dir solver/learner/deepSupervisor/optimizers/fAdaBelief
* @brief Contains code, documentation, and scripts for module: fAdaBelief.
*/

#pragma once

#include <string>
#include <vector>
#include "modules/solver/learner/deepSupervisor/optimizers/fAdam/fAdam.hpp"

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
* @brief Class declaration for module: fAdaBelief.
*/
class fAdaBelief : public fAdam
{
  public: 
  /**
  * @brief [Internal Use] Second central moment.
  */
   std::vector<double> _secondCentralMoment;
  
 
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
