/** \namespace layer
* @brief Namespace declaration for modules of type: layer.
*/

/** \file
* @brief Header file for module: Resampling.
*/

/** \dir neuralNetwork/layer/resampling
* @brief Contains code, documentation, and scripts for module: Resampling.
*/

#pragma once

#include "modules/neuralNetwork/layer/layer.hpp"

namespace korali
{
namespace neuralNetwork
{
namespace layer
{
;

/**
* @brief Class declaration for module: Resampling.
*/
class Resampling : public Layer
{
  public: 
  /**
  * @brief Indicates the resampling method to apply.
  */
   std::string _resamplingType;
  /**
  * @brief Height of the incoming 2D image.
  */
   ssize_t _imageHeight;
  /**
  * @brief Width of the incoming 2D image.
  */
   ssize_t _imageWidth;
  /**
  * @brief Height of the resampled 2D image.
  */
   ssize_t _outputHeight;
  /**
  * @brief Width of the resampled 2D image.
  */
   ssize_t _outputWidth;
  
 
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
  

  /********************************************************
   * Engine specific members
   *******************************************************/
  /**
   * @brief Pre-calculated value for Mini-Batch Size
   */
  ssize_t N;
  /**
   * @brief Pre-calculated value for Input Channels
   */
  ssize_t IC;
  /**
   * @brief Pre-calculated value for Input Image Height
   */
  ssize_t IH;
  /**
   * @brief Pre-calculated value for Input Image Width
   */
  ssize_t IW;
  /**
   * @brief Pre-calculated value for Output Channels
   */
  ssize_t OC;
  /**
   * @brief Pre-calculated value for Output Image Height
   */
  ssize_t OH;
  /**
   * @brief Pre-calculated value for Output Image Width
   */
  ssize_t OW;

#if defined(_KORALI_USE_ONEDNN) && defined(_KORALI_USE_CUDNN)
  /**
   * @brief Defines the resampling method to use either of type oneDNN or cuDNN.
   */
    std::variant<dnnl::algorithm, cudnnResampleMode_t> _algorithm_t;
#endif

#ifdef _KORALI_USE_ONEDNN
  #ifndef _KORALI_USE_CUDNN
      /**
      * @brief Defines the resampling method to be used for oneDNN.
      */
      std::variant<dnnl::algorithm> _algorithm_t;
  #endif
  /**
   * @brief Memory descriptor for the 2D mapping of the scalar input channels
   */
  dnnl::memory::desc _srcMemDesc;
  /**
   * @brief Memory descriptor for the 2D mapping of the scalar output channels
   */
  dnnl::memory::desc _dstMemDesc;
  /**
   * @brief oneDNN primitive descriptor to run the forward resampling operation
   */
  dnnl::resampling_forward::primitive_desc _forwardResamplingPrimitiveDesc;
  /**
   * @brief oneDNN primitive to run the forward resampling operation
   */
  dnnl::primitive _forwardResamplingPrimitive;
  /**
   * @brief oneDNN Arguments for the backward propagation of the gradient wrt Data
   */
  std::unordered_map<int, dnnl::memory> _backwardDataArgs;
  /**
   * @brief oneDNN primitive for the backward propagation of the gradient wrt Data
   */
  dnnl::primitive _backwardDataPrimitive;
#endif

#ifdef _KORALI_USE_CUDNN
  #ifndef _KORALI_USE_ONEDNN
      /**
      * @brief Defines the resampling method to be used with cuDNN.
      */
      std::variant<cudnnResampleMode_t> _algorithm_t;
  #endif
  /**
   * @brief cuDNN Descriptor for the resampling function
   */
  cudnnBackendDescriptor_t _cudnnResamplingDesc;
  // cudnnActivationDescriptor_t _activationDesc;
    // cudnnBackendCreateDescriptor(CUDNN_BACKEND_RESAMPLE_DESCRIPTOR, &desc);

#endif

  void initialize() override;
  void createForwardPipeline() override;
  void createBackwardPipeline() override;
  void forwardData(const size_t t) override;
  void backwardData(const size_t t) override;
};

} //layer
} //neuralNetwork
} //korali
;
