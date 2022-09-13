/** \namespace learning
* @brief Namespace declaration for modules of type: learning.
*/

/** \file
* @brief Header file for module: SupervisedLearning.
*/

/** \dir problem/learning/supervisedLearning
* @brief Contains code, documentation, and scripts for module: SupervisedLearning.
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
* @brief Class declaration for module: SupervisedLearning.
*/
class SupervisedLearning : public Learning
{
  public: 
  /**
  * @brief Stores the batch size of the training dataset.
  */
   size_t _trainingBatchSize;
  /**
  * @brief Stores the batch size of the validation dataset; -1 indicates _validationBatchSize == validation input size.
  */
   ssize_t _validationBatchSize;
  /**
  * @brief Provides the validation set with layout NV*T*IC, where NV is the sample size, T is the sequence length and IC is the vector size of the input.
  */
   std::vector<std::vector<std::vector<float>>> _dataValidationInput;
  /**
  * @brief Provides the solution for one-step ahead prediction with layout NV*OC, where N is the batch size and OC is the vector size of the output.
  */
   std::vector<std::vector<float>> _dataValidationSolution;
  /**
  * @brief Provides the solution for one-step ahead prediction with layout N*OC, where N is the batch size and OC is the vector size of the output.
  */
   std::vector<std::vector<float>> _solutionData;
  /**
  * @brief Indicates the vector size of the output (OC).
  */
   size_t _solutionSize;
  
 
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
