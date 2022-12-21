/** \namespace korali
* @brief Namespace declaration for modules of type: korali.
*/

/** \file
* @brief Header file for module: fRMSProp.
*/

/** \dir solver/deepSupervisor/optimizers/fRMSProp
* @brief Contains code, documentation, and scripts for module: fRMSProp.
*/

#pragma once

#include "modules/solver/deepSupervisor/optimizers/fGradientBasedOptimizer.hpp"

namespace korali
{
;

/**
* @brief Class declaration for module: fRMSProp.
*/
class fRMSProp : public fGradientBasedOptimizer
{
  public: 
  /**
  * @brief Decay Rate.
  */
   float _decay;
  /**
  * @brief [Internal Use] Second moment of Gradient.
  */
   std::vector<float> _r;
  /**
  * @brief [Internal Use] Scaled Gradient.
  */
   std::vector<float> _v;
  
 
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
  

  virtual void initialize() override;
  virtual void processResult(std::vector<float> &gradient) override;
  virtual void reset() override;
  virtual void printInternals() override;
};

} //korali
;
