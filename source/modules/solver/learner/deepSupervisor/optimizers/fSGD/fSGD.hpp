/** \namespace optimizer
* @brief Namespace declaration for modules of type: optimizer.
*/

/** \file
* @brief Header file for module: fSGD.
*/

/** \dir solver/learner/deepSupervisor/optimizers/fSGD
* @brief Contains code, documentation, and scripts for module: fSGD.
*/

#pragma once

#include <string>
#include <vector>
#include "modules/solver/learner/deepSupervisor/optimizers/fGradientBasedOptimizers.hpp"

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
* @brief Class declaration for module: fSGD.
*/
class fSGD : public FastGradientBasedOptimizer
{
  public: 
  
 
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
  * @brief Maximum gradient size before clipping
  */
  float _clippingThreshold{10000};
  // FUNCTIONS =================================================
  /**
  * @brief wether to perform graddient clipping and what kind of.
  * @details
  * - value-clipping: clip each gradient value individually
  */
  std::string _gradientClipping{"value-clipping"};
  /**
  * @brief Takes the gradients and clips them if they become too large
  * @param gradient The gradient of the objective function at the current set of parameters
  */
  void clipGradients(std::vector<float> &gradient);
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
