/** \namespace optimizer
* @brief Namespace declaration for modules of type: optimizer.
*/

/** \file
* @brief Header file for module: fAdam.
*/

/** \dir solver/learner/deepSupervisor/optimizers/fAdam
* @brief Contains code, documentation, and scripts for module: fAdam.
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
* @brief Class declaration for module: fAdam.
*/
class fAdam : public FastGradientBasedOptimizer
{
  public: 
  /**
  * @brief Term to guard agains numerical instability.
  */
   float _epsilon;
  /**
  * @brief [Internal Use] First running powers of beta_1^t.
  */
   float _beta1Pow;
  /**
  * @brief [Internal Use] Second running powers of beta_2^t.
  */
   float _beta2Pow;
  /**
  * @brief [Internal Use] First moment of Gradient.
  */
   std::vector<double> _firstMoment;
  /**
  * @brief [Internal Use] Second moment of Gradient.
  */
   std::vector<double> _secondMoment;
  
 
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
  * @brief Beta for momentum update
  */
  float _beta1{0.9f};
  /**
  * @brief Beta for gradient update
  */
  float _beta2{0.999f};
  /**
  * @brief Weight Decay To add
  */
  // FUNCTIONS =================================================
  // OVERRIDEN FUNCTIONS =======================================
  virtual void initialize() override;
  virtual void processResult(std::vector<float> &gradient) override;
  virtual void reset() override;
  virtual bool _implementsWeightDecay() override { return 1; };
};

} //optimizer
} //learner
} //solver
} //korali
;
