/** \namespace learning
* @brief Namespace declaration for modules of type: learning.
*/

/** \file
* @brief Header file for module: UnsupervisedLearning.
*/

/** \dir problem/learning/unsupervisedLearning
* @brief Contains code, documentation, and scripts for module: UnsupervisedLearning.
*/

#pragma once

#include "modules/problem/learning/learning.hpp"

namespace korali
{
namespace problem
{
namespace learning
{
;

/**
* @brief Class declaration for module: UnsupervisedLearning.
*/
class UnsupervisedLearning : public Learning
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
  

  void initialize() override;

  /**
   * @brief Checks whether the input data has the correct shape
   */
  void verifyData();
};

} //learning
} //problem
} //korali
;
