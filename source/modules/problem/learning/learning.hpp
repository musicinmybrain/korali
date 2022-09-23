/** \namespace problem
* @brief Namespace declaration for modules of type: problem.
*/

/** \file
* @brief Header file for module: Learning.
*/

/** \dir problem/learning
* @brief Contains code, documentation, and scripts for module: Learning.
*/

#pragma once

#include "modules/problem/problem.hpp"

namespace korali
{
namespace problem
{
;

/**
* @brief Class declaration for module: Learning.
*/
class Learning : public Problem
{
  public: 
  /**
  * @brief Stores the length of the sequence for recurrent neural networks.
  */
   size_t _maxTimesteps;
  /**
  * @brief Stores the allowed Testing Batch Sizes for the testing dataset.
  */
   std::vector<size_t> _testingBatchSizes;
  /**
  * @brief Stores the batch size of the testing dataset.
  */
   size_t _testingBatchSize;
  /**
  * @brief Provides the input data with layout N*T*IC, where N is the sample size, T is the sequence length and IC is the vector size of the input.
  */
   std::vector<std::vector<std::vector<float>>> _inputData;
  /**
  * @brief Indicates the vector size of the input (IC).
  */
   size_t _inputSize;
  
 
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

  /**
   * @brief Checks whether the input data has the correct shape
   */
  void verifyData();
};

} //problem
} //korali
;
