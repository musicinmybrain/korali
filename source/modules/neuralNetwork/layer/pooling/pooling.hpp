/** \namespace layer
* @brief Namespace declaration for modules of type: layer.
*/

/** \file
* @brief Header file for module: Pooling.
*/

/** \dir neuralNetwork/layer/pooling
* @brief Contains code, documentation, and scripts for module: Pooling.
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
* @brief Class declaration for module: Pooling.
*/
class Pooling : public Layer
{
  public: 
  /**
  * @brief Indicates the pooling function to apply.
  */
   std::string _function;
  /**
  * @brief Height of the incoming 2D image.
  */
   ssize_t _imageHeight;
  /**
  * @brief Width of the incoming 2D image.
  */
   ssize_t _imageWidth;
  /**
  * @brief Height/Width of the 2D filter.
  */
   ssize_t _kernelSize;
  /**
  * @brief Height of the 2D filter.
  */
   ssize_t _kernelHeight;
  /**
  * @brief Width of the 2D filter.
  */
   ssize_t _kernelWidth;
  /**
  * @brief Strides for the image on the vertical dimension.
  */
   ssize_t _verticalStride;
  /**
  * @brief Strides for the image on the horizontal dimension.
  */
   ssize_t _horizontalStride;
  /**
  * @brief Strides for the image on the vertical/horizontal dimension.
  */
   ssize_t _strideSize;
  /**
  * @brief Paddings for the image left side.
  */
   ssize_t _paddingLeft;
  /**
  * @brief Paddings for the image right side.
  */
   ssize_t _paddingRight;
  /**
  * @brief Paddings for the image top side.
  */
   ssize_t _paddingTop;
  /**
  * @brief Paddings for the image Bottom side.
  */
   ssize_t _paddingBottom;
  /**
  * @brief Paddings for the image Top and Bottom side.
  */
   ssize_t _paddingVertical;
  /**
  * @brief Paddings for the image left and right side.
  */
   ssize_t _paddingHorizontal;
  /**
  * @brief Paddings for the image Bottom side.
  */
   ssize_t _paddingSize;
  /**
  * @brief If specified 'Ouput Channels is calculated automatically. For -1: Filters = IC
  */
   ssize_t _filters;
  
 
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

  /**
   * @brief Pre-calculated value for Kernel Image Height
   */
  ssize_t KH;

  /**
   * @brief Pre-calculated value for Kernel Image Width
   */
  ssize_t KW;

  /**
   * @brief Pre-calculated values for padding left
   */
  ssize_t PL;

  /**
   * @brief Pre-calculated values for padding right
   */
  ssize_t PR;

  /**
   * @brief Pre-calculated values for padding top
   */
  ssize_t PT;

  /**
   * @brief Pre-calculated values for padding bottom
   */
  ssize_t PB;

  /**
   * @brief Pre-calculated values for horizontal stride
   */
  ssize_t SH;

  /**
   * @brief Pre-calculated values for vertical stride
   */
  ssize_t SV;

#if defined(_KORALI_USE_ONEDNN) && defined(_KORALI_USE_CUDNN)
  /**
   * @brief Defines the pooling method to be useed for either of type oneDNN or cuDNN.
   */
    std::variant<dnnl::algorithm, cudnnPoolingMode_t> _algorithm_t;
#endif

#ifdef _KORALI_USE_ONEDNN
  #ifndef _KORALI_USE_CUDNN
      /**
      * @brief Defines the pooling method to be used for oneDNN.
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
   * @brief oneDNN Memory object descriptor to contain the pooling workspace
   */
  std::vector<dnnl::memory> _workspaceMem;

  /**
   * @brief oneDNN primitive attributes that describe the forward pooling primitive
   */
  dnnl::pooling_forward::primitive_desc _forwardPoolingPrimitiveDesc;

  /**
   * @brief oneDNN primitive to run the inner product + bias addition operation
   */
  dnnl::primitive _forwardPoolingPrimitive;

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
      * @brief Defines the pooling method to be used with cuDNN.
      */
      std::variant<cudnnPoolingMode_t> _algorithm_t;
  #endif
  /**
   * @brief cuDNN Descriptor for the input data
   */
  cudnnTensorDescriptor_t _inputDescriptor;
  /**
   * @brief cuDNN Descriptor for the output data
   */
  cudnnTensorDescriptor_t _outputDescriptor;
  /**
   * @brief cuDNN pooling descriptor.
   */
  cudnnTensorDescriptor_t  _poolingDescriptor;
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
