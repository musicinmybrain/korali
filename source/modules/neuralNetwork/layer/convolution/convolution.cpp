#include "modules/neuralNetwork/layer/convolution/convolution.hpp"
#include "modules/neuralNetwork/neuralNetwork.hpp"
#include "modules/neuralNetwork/layer/pooling/pooling.hpp"
#include "modules/neuralNetwork/layer/deconvolution/deconvolution.hpp"
#include "modules/neuralNetwork/layer/resampling/resampling.hpp"

#ifdef _KORALI_USE_CUDNN
  #include "auxiliar/cudaUtils.hpp"
#endif

#ifdef _KORALI_USE_ONEDNN
  #include "auxiliar/dnnUtils.hpp"
using namespace dnnl;
#endif

#include <Eigen/Dense>
using namespace Eigen;

namespace korali
{
namespace neuralNetwork
{
namespace layer
{
;

void Convolution::initialize()
{
  // Checking Layer size
  if (_filters == -1 && _outputChannels == 0) KORALI_LOG_ERROR("[%s Layer %lu] No output channesl or filter number specified for layer (%lu).\n", _type.c_str(), _index-1);
  // Checking position
  if (_index == 0) KORALI_LOG_ERROR("[%s Layer %lu] layers cannot be the starting layer of the NN\n", _type.c_str(), _index-1);
  if (_index == _nn->_layers.size() - 1) KORALI_LOG_ERROR("[%s Layer %lu] layers cannot be the last layer of the NN\n", _type.c_str(), _index-1);

  // Precalculating values for the convolution operation
  N = _batchSize;
  IH = _imageHeight;
  IW = _imageWidth;
  // Kernel ===============================================================================
  KH = KW = _kernelSize;
  if(_kernelWidth != -1)
    KW = _kernelWidth;
  if( _kernelHeight != -1)
    KH = _kernelHeight;
  // Strides ==============================================================================
  SV = SH = _strideSize;
  if( _verticalStride != -1)
    SV = _verticalStride;
  if( _horizontalStride != -1)
    SH = _horizontalStride;
  // Paddings =============================================================================
  PT = PL = PB = PR = _paddingSize;
  if( _paddingVertical != -1)
    PT = PB = _paddingVertical;
  if( _paddingHorizontal != -1)
    PL = PR = _paddingVertical;
  if( _paddingTop != -1)
    PT = _paddingTop;
  if( _paddingBottom != -1)
    PB = _paddingBottom;
  if( _paddingLeft != -1)
    PL = _paddingLeft;
  if( _paddingRight != -1)
    PR = _paddingRight;

  if (IW <= 0) KORALI_LOG_ERROR("[%s layer %zu] Image width not given.\n", _type.c_str(), _index-1);
  if (IH <= 0) KORALI_LOG_ERROR("[%s layer %zu] Image height not given.\n", _type.c_str(), _index-1);
  // ======================================================================================
  // ======================================================================================
  // Check for non zeros
  if (KH <= 0) KORALI_LOG_ERROR("Kernel height must be larger than zero for convolutional layer.\n");
  if (KW <= 0) KORALI_LOG_ERROR("Kernel width must be larger than zero for convolutional layer.\n");
  if (SV <= 0) KORALI_LOG_ERROR("Vertical stride must be larger than zero for convolutional layer.\n");
  if (SH <= 0) KORALI_LOG_ERROR("Horizontal stride must be larger than zero for convolutional layer.\n");

  // Several sanity checks
  if (KH > IH + PR + PL) KORALI_LOG_ERROR("[%s layer %zu] Kernel height cannot be larger than input image height plus padding.\n", _type.c_str(), _index-1);
  if (KW > IW + PT + PB) KORALI_LOG_ERROR("[%s layer %zu] Kernel width cannot be larger than input image width plus padding.\n", _type.c_str(), _index-1);

  // Check whether the output channels of the previous layer is divided by the height and width
  if (_prevLayer->_outputChannels % (IH * IW) > 0) KORALI_LOG_ERROR("Previous layer %zu to convolutional layer %zu contains a number of channels (%lu) not divisible by the convolutional 2D HxW setup (%lux%lu).\n", _index-2, _index-1, _prevLayer->_outputChannels, IH, IW);
  IC = _prevLayer->_outputChannels / (IH * IW);

  OH = (IH - KH + PT + PB) / SV + 1;
  OW = (IW - KW + PR + PL) / SH + 1;
  if( ((IH - KH + PT + PB) % SV) != 0)
    _k->_logger->logInfo("Detailed", "[Convolutional layer %zu] OH = (IH - KH + PT + PB) / SV + 1 = (%lu - %lu - %lu + %lu) / %lu +1 = %lu using floor.\n",
                         _index-1, IH, KH, PT, PB, SV, OH);
  if( ((IW - KW + PR + PL) % SH) != 0)
    _k->_logger->logInfo("Detailed", "[Convolutional layer %zu] OW = (IW - KW + PR + PL) / SH = (%lu - %lu - %lu + %lu) / %lu = %lu using floor.\n",
                         _index-1, IW, KW, PR, PL, SH, OW);

  if(_outputChannels == 0)
    _outputChannels = _filters*OH*OW;
  // Check whether the output channels of the previous layer is divided by the height and width
  if (_outputChannels % (OH * OW) > 0)
    KORALI_LOG_ERROR("[Convolutional layer %zu] Number of output channels (%lu) not divisible by the output image size (%lux%lu) given kernel (%lux%lu) size and padding/stride configuration.\n", _index-1, _outputChannels, OH, OW, KH, KW);
  OC = _outputChannels / (OH * OW);

#ifdef _KORALI_USE_CUDNN
    if (_nn->_engine == "CuDNN")
    {
      if (PT != PB)
        KORALI_LOG_ERROR("[Convolutional layer %zu] does not allow an symmetric top %zu and bottom %zu padding.\n", _index-1, PT, PB);
      if (PT != PB)
        KORALI_LOG_ERROR("[Convolutional layer %zu] does not allow an symmetric left %zu and right %zu padding.\n", _index-1, PL, PR);
    }
#endif
}

std::vector<float> Convolution::generateInitialHyperparameters()
{
  std::vector<float> hyperparameters;
  // If this is not the initial layer, calculate hyperparameters for weight and bias operation
  if (_prevLayer != nullptr)
  {
    // Setting value for this layer's xavier constant
    float xavierConstant = std::sqrt(6.0f) / std::sqrt(_outputChannels + _prevLayer->_outputChannels);

    // Adding layer's weights hyperparameter values
    for (auto i = 0; i < _weightsCount; i++)
      hyperparameters.push_back(_weightScaling * xavierConstant * _nn->_uniformGenerator->getRandomNumber());

    // Adding layer's bias hyperparameter values
    for (auto i = 0; i < _biasCount; i++)
      hyperparameters.push_back(0.0f);
  }

  return hyperparameters;
}

void Convolution::createHyperparameterMemory()
{
  // Setting hyperparameter count
  _weightsCount = OC * IC * KH * KW;
  _biasCount = OC;

  _hyperparameterCount = _weightsCount + _biasCount;

  std::exception_ptr eptr;
  try{
#ifdef _KORALI_USE_ONEDNN
    if (_nn->_engine == "OneDNN")
    {
      memory::dims weightDims = {OC, IC, KH, KW};
      auto weightMemDesc = memory::desc(weightDims, memory::data_type::f32, memory::format_tag::oihw);
      _weightsMem = memory(weightMemDesc, _nn->_dnnlEngine);

      auto biasMemDesc = memory::desc({OC}, memory::data_type::f32, memory::format_tag::a);
      _biasMem = memory(biasMemDesc, _nn->_dnnlEngine);
    }
#endif

#ifdef _KORALI_USE_CUDNN
    if (_nn->_engine == "CuDNN")
    {
      // Kernel/Filter Memory
      cudnnErrCheck(cudnnCreateFilterDescriptor(&_weightsFilterDesc));
      cudnnErrCheck(cudnnSetFilter4dDescriptor(_weightsFilterDesc,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                               /*format=*/CUDNN_TENSOR_NCHW,
                                               /*out_channels=*/OC,
                                               /*in_channels=*/IC,
                                               /*kernel_height=*/KH,
                                               /*kernel_width=*/KW));
      // _weightsFilter.resize(_nn->_timestepCount);
      // for (size_t i = 0; i < _nn->_timestepCount; i++)
      cudaErrCheck(cudaMalloc((void **)&_weightsFilter, _weightsCount * sizeof(float)));
      // Create Bias Tensors
      cudnnErrCheck(cudnnCreateTensorDescriptor(&_biasTensorDesc));
      cudnnErrCheck(cudnnSetTensor4dDescriptor(_biasTensorDesc,
                                               CUDNN_TENSOR_NCHW,
                                               CUDNN_DATA_FLOAT,
                                               1,
                                               OC,
                                               1,
                                               1));
      cudaErrCheck(cudaMalloc((void **)&_biasTensor, _biasCount * sizeof(float)));
    }
#endif
  } catch (...) {
    eptr = std::current_exception();
  }
  try{
    Layer::exceptionHandler(eptr);
  } catch(...){
    eptr = std::current_exception();
  }
  exceptionHandler(eptr);

}

void Convolution::copyHyperparameterPointers(Layer *dstLayer)
{
  Convolution *dstPtr = dynamic_cast<Convolution *>(dstLayer);
  dstPtr->_hyperparameterCount = _hyperparameterCount;
  dstPtr->_weightsCount = _weightsCount;
  dstPtr->_biasCount = _biasCount;
#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    dstPtr->_weightsMem = _weightsMem;
    dstPtr->_biasMem = _biasMem;
  }
#endif

#ifdef _KORALI_USE_CUDNN
  if (_nn->_engine == "CuDNN")
  {
    // copy kernel weights and bias
    dstPtr->_weightsFilterDesc = _weightsFilterDesc;
    dstPtr->_weightsFilter = _weightsFilter;
    dstPtr->_biasTensorDesc = _biasTensorDesc;
    dstPtr->_biasTensor = _biasTensor;
  }
#endif

}

void Convolution::createForwardPipeline()
{
  Layer::createForwardPipeline();

  if (_nn->_engine == "Korali") KORALI_LOG_ERROR("Convolutional Layers still not supported in Korali's NN backend. Use OneDNN.\n");

  std::exception_ptr eptr;
  try{
#ifdef _KORALI_USE_ONEDNN
    if (_nn->_engine == "OneDNN")
    {
      // Creating memory descriptor mappings for input memory
      _srcMemDesc = memory::desc({N, IC, IH, IW}, memory::data_type::f32, memory::format_tag::nchw);
      _dstMemDesc = memory::desc({N, OC, OH, OW}, memory::data_type::f32, memory::format_tag::nchw);

      // Creating padding dims
      memory::dims ST = {SV, SH};  // Horizontal Vertical
      memory::dims PTL = {PT, PL}; // Top Left
      memory::dims PBR = {PB, PR}; // Bottom Right

      // We create the convolution operation
      auto convolution_d = convolution_forward::desc(_propKind, algorithm::convolution_auto, _srcMemDesc, _weightsMem.get_desc(), _biasMem.get_desc(), _dstMemDesc, ST, PTL, PBR);

      // Create inner product primitive descriptor.
      dnnl::primitive_attr convolutionPrimitiveAttributes;
      _forwardConvolutionPrimitiveDesc = convolution_forward::primitive_desc(convolution_d, convolutionPrimitiveAttributes, _nn->_dnnlEngine);

      // Create the weights+bias primitive.
      _forwardConvolutionPrimitive = convolution_forward(_forwardConvolutionPrimitiveDesc);
    }
#endif

#ifdef _KORALI_USE_CUDNN
    // Calling base layer function
    /*
    ** - cuDNN: 1. sets _forwardMode
    **          2. set OC = _outputChannels (here _outputChannels / (OH * OW) )
    **          3. creates _outputTensorDesc of size N, OC, 1, 1
    **          4. creates _outputTensor of size N x OC of size float
    */
    if (_nn->_engine == "CuDNN")
    {
      // Input Tensor
      cudnnErrCheck(cudnnCreateTensorDescriptor(&_inputDescriptor));
      cudnnErrCheck(cudnnSetTensor4dDescriptor(
                      /*Inp. Tensor Descr.=*/ _inputDescriptor,
                      /*format=*/CUDNN_TENSOR_NCHW,
                      /*dataType=*/CUDNN_DATA_FLOAT,
                      /*batch_size=*/N,
                      /*channels=*/IC,
                      /*image_height=*/IH,
                      /*image_width=*/IW));
      // Output Tensor
      cudnnErrCheck(cudnnCreateTensorDescriptor(&_outputDescriptor));
      cudnnErrCheck(cudnnSetTensor4dDescriptor(
                      /*Output. Tensor Descr.=*/ _outputDescriptor,
                      /*format=*/CUDNN_TENSOR_NCHW,
                      /*dataType=*/CUDNN_DATA_FLOAT,
                      /*batch_size=*/N,
                      /*channels=*/OC,
                      /*image_height=*/OH,
                      /*image_width=*/OW));

      // Convolution Descriptor describes the type of convolution we want to perform
      cudnnErrCheck(cudnnCreateConvolutionDescriptor(&_convolutionDescriptor));
      cudnnErrCheck(cudnnSetConvolution2dDescriptor(_convolutionDescriptor,
                                                    /*pad_height=PB=*/PT,
                                                    /*pad_width=PL=*/PR,
                                                    /*vertical_stride=*/SV,
                                                    /*horizontal_stride=*/SH,
                                                    /*dilation_height=*/1,
                                                    /*dilation_width=*/1,
                                                    /*mode=*/CUDNN_CONVOLUTION,
                                                    /*computeType=*/CUDNN_DATA_FLOAT));
      // TODO: change this to cuDNN 8 API
      // cudnnErrCheck(cudnnGetConvolutionForwardAlgorithm_v7(_nn->_cuDNNHandle,
      //                                                      _inputDescriptor,
      //                                                      _weightsFilterDesc,
      //                                                      _convolutionDescriptor,
      //                                                      _outputDescriptor,
      //                                                      CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
      //                                                      /*memoryLimitInBytes=*/0,
      //                                                      &_convolutionAlgorithm));
      _convolutionFwdAlgorithm = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
      cudnnErrCheck(cudnnGetConvolutionForwardWorkspaceSize(_nn->_cuDNNHandle,
                                                            _inputDescriptor,
                                                            _weightsFilterDesc,
                                                            _convolutionDescriptor,
                                                            _outputDescriptor,
                                                            /*_convolutionAlgorith=*/_convolutionFwdAlgorithm,
                                                            &_convolutionWorkspaceSize));
      // Create workspace memory in createBackwardpipeline
    }
#endif
  } catch (...) {
    eptr = std::current_exception();
  }
  exceptionHandler(eptr);
}

void Convolution::createBackwardPipeline()
{
  //  Initializing memory objects and primitives for BACKWARD propagation

  // Calling base layer function
  Layer::createBackwardPipeline();

#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    // Creating memory descriptor mappings for input memory
    _srcMemDesc = memory::desc({N, IC, IH, IW}, memory::data_type::f32, memory::format_tag::nchw);
    _dstMemDesc = memory::desc({N, OC, OH, OW}, memory::data_type::f32, memory::format_tag::nchw);

    // Creating padding dims
    memory::dims ST = {SV, SH};  // Horizontal Vertical
    memory::dims PTL = {PT, PL}; // Top Left
    memory::dims PBR = {PB, PR}; // Bottom Right

    // Setting strides and padding configuration
    _weightsGradientMem = memory(_weightsMem.get_desc(), _nn->_dnnlEngine);
    _biasGradientMem = memory(_biasMem.get_desc(), _nn->_dnnlEngine);

    auto backwardDataDesc = convolution_backward_data::desc(
      algorithm::convolution_auto,
      _srcMemDesc,
      _weightsMem.get_desc(),
      _dstMemDesc,
      ST,
      PTL,
      PBR);

    // Create the primitive.
    auto backwardDataPrimitiveDesc = convolution_backward_data::primitive_desc(backwardDataDesc, _nn->_dnnlEngine, _forwardConvolutionPrimitiveDesc);
    _backwardDataPrimitive = convolution_backward_data(backwardDataPrimitiveDesc);

    auto backwardWeightsDesc = convolution_backward_weights::desc(
      algorithm::convolution_auto,
      _srcMemDesc,
      _weightsMem.get_desc(),
      _biasMem.get_desc(),
      _dstMemDesc,
      ST,
      PTL,
      PBR);

    // Create the primitive.
    auto backwardWeightsPrimitiveDesc = convolution_backward_weights::primitive_desc(backwardWeightsDesc, _nn->_dnnlEngine, _forwardConvolutionPrimitiveDesc);
    _backwardWeightsPrimitive = convolution_backward_weights(backwardWeightsPrimitiveDesc);
  }
#endif

#ifdef _KORALI_USE_CUDNN
  // TODO
  if (_nn->_engine == "CuDNN")
  {
    cudaErrCheck(cudaMalloc((void **)&_weightsGradientFilter, _weightsCount * sizeof(float)));
    cudaErrCheck(cudaMalloc((void **)&_biasGradientTensor, _biasCount * sizeof(float)));
    auto backwardWsSize = getBackwardWsSize();
    _convolutionWorkspaceSize = std::max(_convolutionWorkspaceSize, backwardWsSize);
    cudaErrCheck(cudaMalloc((void **)&_convolutionWorkspace, _convolutionWorkspaceSize * sizeof(float)));
    _convolutionBwdDataAlgorithm = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
    _convolutionBwdFilterAlgorithm = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
  //   v8: cudnnFindConvolutionBackwardFilterAlgorithm
  // TODO Either v7 API
  //   cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithm_v7(
  //     cudnnHandle_t                          handle,
  //     const cudnnTensorDescriptor_t          xDesc,
  //     const cudnnTensorDescriptor_t          dyDesc,
  //     const cudnnConvolutionDescriptor_t     convDesc,
  //     const cudnnFilterDescriptor_t          dwDesc,
  //     const int                              requestedAlgoCount,
  //     int                                   *returnedAlgoCount,
  //     cudnnConvolutionBwdFilterAlgoPerf_t   *perfResults)
  //     }

  // v8: cudnnFindConvolutionBackwardDataAlgorithm()
  // cudnnStatus_t cudnnGetConvolutionBackwardDataAlgorithm_v7(
  //   cudnnHandle_t                          handle,
  //   const cudnnFilterDescriptor_t          wDesc,
  //   const cudnnTensorDescriptor_t          dyDesc,
  //   const cudnnConvolutionDescriptor_t     convDesc,
  //   const cudnnTensorDescriptor_t          dxDesc,
  //   const int                              requestedAlgoCount,
  //   int                                   *returnedAlgoCount,
  //   cudnnConvolutionBwdDataAlgoPerf_t     *perfResults)
#ifdef DEBUG
      _k->_logger->logInfo("Detailed", "[%s layer %zu] Allocating %f MB for cuDNN convolution workspace.\n", _type.c_str(), _index-1, _convolutionWorkspaceSize/(1024.0*1024.0));
#endif
  }
#endif
}

void Convolution::forwardData(const size_t t)
{
#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    // Arguments to the inner product operation
    std::unordered_map<int, dnnl::memory> forwardConvolutionArgs;
    forwardConvolutionArgs[DNNL_ARG_SRC] = _prevLayer->_outputMem[t];
    forwardConvolutionArgs[DNNL_ARG_WEIGHTS] = _weightsMem;
    forwardConvolutionArgs[DNNL_ARG_BIAS] = _biasMem;
    forwardConvolutionArgs[DNNL_ARG_DST] = _outputMem[t];

    _forwardConvolutionPrimitive.execute(_nn->_dnnlStream, forwardConvolutionArgs);
  }
#endif

#ifdef _KORALI_USE_CUDNN
  if (_nn->_engine == "CuDNN")
  {
    float alpha1 = 1.0f;
    float alpha2 = 0.0f;
    cudnnErrCheck(cudnnConvolutionForward(_nn->_cuDNNHandle,
                                          /*alpha=*/&alpha1,
                                          /*xDesc/inputDesc=*/_inputDescriptor,
                                          /*x/input=*/_prevLayer->_outputTensor[t],
                                          _weightsFilterDesc,
                                          _weightsFilter,
                                          _convolutionDescriptor,
                                          _convolutionFwdAlgorithm,
                                          _convolutionWorkspace,
                                          _convolutionWorkspaceSize,
                                          /*beta=*/&alpha2,
                                          /*yDesc/outputDesc=*/_outputDescriptor,
                                          /*y/output=*/_outputTensor[t]));
    float alpha = 1.0f;
    float beta = 1.0f;
    cudnnAddTensor(_nn->_cuDNNHandle, &alpha, _biasTensorDesc, _biasTensor, &beta, _outputDescriptor, _outputTensor[t]);
    // cudnnConvolutionBiasActivationForward()
  }
#endif
}

void Convolution::backwardData(const size_t t)
{
  if (_nn->_mode == "Inference")
    KORALI_LOG_ERROR("Requesting Layer backward data propagation but NN was configured for inference only.\n");

#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    _backwardDataArgs[DNNL_ARG_DIFF_DST] = _outputGradientMem[t];             // Input
    _backwardDataArgs[DNNL_ARG_WEIGHTS] = _weightsMem;                        // Input
    _backwardDataArgs[DNNL_ARG_DIFF_SRC] = _prevLayer->_outputGradientMem[t]; // Output

    _backwardDataPrimitive.execute(_nn->_dnnlStream, _backwardDataArgs);
  }
#endif

#ifdef _KORALI_USE_CUDNN
  if (_nn->_engine == "CuDNN")
  {
    float alpha = 1.0f;
    float beta = 0.0f;
    cudnnErrCheck(cudnnConvolutionBackwardData(
      _nn->_cuDNNHandle,
      &alpha,
      _weightsFilterDesc,
      _weightsFilter,
      /*dyDesc=*/_outputDescriptor,
      /*dy=*/_outputGradientTensor[t],
      _convolutionDescriptor,
      // TODO: change algorithm type
      _convolutionBwdDataAlgorithm,
      _convolutionWorkspace,
      _convolutionWorkspaceSize,
      &beta,
      /*dxDesc=*/_inputDescriptor,
      // =====================================
      /*dx=*/_prevLayer->_outputGradientTensor[t]));
  }
#endif
}

// 1.

void Convolution::backwardHyperparameters(size_t t)
{
  if (_nn->_mode == "Inference")
    KORALI_LOG_ERROR("Requesting Layer hyperparameter gradient propagation but NN was configured for inference only.\n");

#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    // Arguments for the backward propagation of the gradient wrt Weights and Biases
    std::unordered_map<int, dnnl::memory> backwardWeightsArgs;
    backwardWeightsArgs[DNNL_ARG_SRC] = _prevLayer->_outputMem[t];    // Input
    backwardWeightsArgs[DNNL_ARG_DIFF_DST] = _outputGradientMem[t];   // Input
    backwardWeightsArgs[DNNL_ARG_DIFF_WEIGHTS] = _weightsGradientMem; // Output
    backwardWeightsArgs[DNNL_ARG_DIFF_BIAS] = _biasGradientMem;       // Output

    _backwardWeightsPrimitive.execute(_nn->_dnnlStream, backwardWeightsArgs);
  }
#endif

#ifdef _KORALI_USE_CUDNN
  if (_nn->_engine == "CuDNN")
  {
    float alpha = 1.0f;
    float beta = 0.0f;
    cudnnErrCheck(cudnnConvolutionBackwardBias(
      _nn->_cuDNNHandle,
      &alpha,
      _outputDescriptor,
      _outputGradientTensor[t],
      &beta,
      _biasTensorDesc,
      _biasGradientTensor));

    cudnnErrCheck(cudnnConvolutionBackwardFilter(
                    _nn->_cuDNNHandle,
                    &alpha,
                    /*xDesc=*/_inputDescriptor,
                    /*x=*/_prevLayer->_outputTensor[t],
                    /*dyDesc=*/_outputDescriptor,
                    /*dy=*/_outputGradientTensor[t],
                    _convolutionDescriptor,
                    _convolutionBwdFilterAlgorithm,
                    _convolutionWorkspace,
                    _convolutionWorkspaceSize,
                    &beta,
                    _weightsFilterDesc,
                    _weightsGradientFilter));


  }
#endif

}

void Convolution::setHyperparameters(const float *hyperparameters)
{
#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    write_to_dnnl_memory(&hyperparameters[0], _weightsMem);
    write_to_dnnl_memory(&hyperparameters[_weightsCount], _biasMem);
  }
#endif

#ifdef _KORALI_USE_CUDNN
  if (_nn->_engine == "CuDNN"){
    cudaErrCheck(cudaMemcpy(_weightsFilter, &hyperparameters[0], _weightsCount * sizeof(float), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(_biasTensor, &hyperparameters[_weightsCount], _biasCount * sizeof(float), cudaMemcpyHostToDevice));
  }
#endif
}

void Convolution::getHyperparameters(float *hyperparameters)
{
#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    read_from_dnnl_memory(&hyperparameters[0], _weightsMem);
    read_from_dnnl_memory(&hyperparameters[_weightsCount], _biasMem);
  }
#endif

#ifdef _KORALI_USE_CUDNN
  if (_nn->_engine == "CuDNN"){
    cudaErrCheck(cudaMemcpy(&hyperparameters[0], _weightsFilter, _weightsCount * sizeof(float), cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(&hyperparameters[_weightsCount], _biasTensor, _biasCount * sizeof(float), cudaMemcpyDeviceToHost));
  }
#endif
}

void Convolution::getHyperparameterGradients(float *gradient)
{
#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    read_from_dnnl_memory(&gradient[0], _weightsGradientMem);
    read_from_dnnl_memory(&gradient[_weightsCount], _biasGradientMem);
  }
#endif

#ifdef _KORALI_USE_CUDNN
  if (_nn->_engine == "CuDNN")
  {
    cudaErrCheck(cudaMemcpy(&gradient[0], _weightsGradientFilter, _weightsCount * sizeof(float), cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(&gradient[_weightsCount], _biasGradientTensor, _biasCount * sizeof(float), cudaMemcpyDeviceToHost));
  }
#endif
}

#ifdef _KORALI_USE_CUDNN
size_t Convolution::getBackwardWsSize() {
        size_t sizeFilterAlg = 0;
        size_t sizeDataAlg = 0;
        // if (!_algorithmBackwardFilter.empty())
        // TODO: algorithm search cuDNN v8
        cudnnErrCheck(cudnnGetConvolutionBackwardFilterWorkspaceSize(_nn->_cuDNNHandle,
                                                                     _inputDescriptor,
                                                                     _outputDescriptor,
                                                                     _convolutionDescriptor,
                                                                     _weightsFilterDesc,
                                                                     _convolutionBwdFilterAlgorithm,
                                                                     &sizeFilterAlg));
        // if (!_algorithmBackwardData.empty())
        // TODO: algorithm search cuDNN v8
        cudnnErrCheck(cudnnGetConvolutionBackwardDataWorkspaceSize(_nn->_cuDNNHandle,
                                                                   _weightsFilterDesc,
                                                                   _outputDescriptor,
                                                                   _convolutionDescriptor,
                                                                   _inputDescriptor,
                                                                   _convolutionBwdDataAlgorithm,
                                                                   &sizeDataAlg));
        return std::max(sizeFilterAlg, sizeDataAlg);
}
#endif


void Convolution::setConfiguration(knlohmann::json& js) 
{
  if (isDefined(js, "Results"))  eraseValue(js, "Results");

  if (isDefined(js, "Image Height"))
  {
    try
    {
      _imageHeight = js["Image Height"].get<ssize_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ convolution ] \n + Key:    ['Image Height']\n%s", e.what());
    }
    eraseValue(js, "Image Height");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Image Height'] required by convolution.\n"); 

  if (isDefined(js, "Image Width"))
  {
    try
    {
      _imageWidth = js["Image Width"].get<ssize_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ convolution ] \n + Key:    ['Image Width']\n%s", e.what());
    }
    eraseValue(js, "Image Width");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Image Width'] required by convolution.\n"); 

  if (isDefined(js, "Kernel Size"))
  {
    try
    {
      _kernelSize = js["Kernel Size"].get<ssize_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ convolution ] \n + Key:    ['Kernel Size']\n%s", e.what());
    }
    eraseValue(js, "Kernel Size");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Kernel Size'] required by convolution.\n"); 

  if (isDefined(js, "Kernel Height"))
  {
    try
    {
      _kernelHeight = js["Kernel Height"].get<ssize_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ convolution ] \n + Key:    ['Kernel Height']\n%s", e.what());
    }
    eraseValue(js, "Kernel Height");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Kernel Height'] required by convolution.\n"); 

  if (isDefined(js, "Kernel Width"))
  {
    try
    {
      _kernelWidth = js["Kernel Width"].get<ssize_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ convolution ] \n + Key:    ['Kernel Width']\n%s", e.what());
    }
    eraseValue(js, "Kernel Width");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Kernel Width'] required by convolution.\n"); 

  if (isDefined(js, "Vertical Stride"))
  {
    try
    {
      _verticalStride = js["Vertical Stride"].get<ssize_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ convolution ] \n + Key:    ['Vertical Stride']\n%s", e.what());
    }
    eraseValue(js, "Vertical Stride");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Vertical Stride'] required by convolution.\n"); 

  if (isDefined(js, "Horizontal Stride"))
  {
    try
    {
      _horizontalStride = js["Horizontal Stride"].get<ssize_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ convolution ] \n + Key:    ['Horizontal Stride']\n%s", e.what());
    }
    eraseValue(js, "Horizontal Stride");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Horizontal Stride'] required by convolution.\n"); 

  if (isDefined(js, "Stride Size"))
  {
    try
    {
      _strideSize = js["Stride Size"].get<ssize_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ convolution ] \n + Key:    ['Stride Size']\n%s", e.what());
    }
    eraseValue(js, "Stride Size");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Stride Size'] required by convolution.\n"); 

  if (isDefined(js, "Padding Left"))
  {
    try
    {
      _paddingLeft = js["Padding Left"].get<ssize_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ convolution ] \n + Key:    ['Padding Left']\n%s", e.what());
    }
    eraseValue(js, "Padding Left");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Padding Left'] required by convolution.\n"); 

  if (isDefined(js, "Padding Right"))
  {
    try
    {
      _paddingRight = js["Padding Right"].get<ssize_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ convolution ] \n + Key:    ['Padding Right']\n%s", e.what());
    }
    eraseValue(js, "Padding Right");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Padding Right'] required by convolution.\n"); 

  if (isDefined(js, "Padding Top"))
  {
    try
    {
      _paddingTop = js["Padding Top"].get<ssize_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ convolution ] \n + Key:    ['Padding Top']\n%s", e.what());
    }
    eraseValue(js, "Padding Top");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Padding Top'] required by convolution.\n"); 

  if (isDefined(js, "Padding Bottom"))
  {
    try
    {
      _paddingBottom = js["Padding Bottom"].get<ssize_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ convolution ] \n + Key:    ['Padding Bottom']\n%s", e.what());
    }
    eraseValue(js, "Padding Bottom");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Padding Bottom'] required by convolution.\n"); 

  if (isDefined(js, "Padding Vertical"))
  {
    try
    {
      _paddingVertical = js["Padding Vertical"].get<ssize_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ convolution ] \n + Key:    ['Padding Vertical']\n%s", e.what());
    }
    eraseValue(js, "Padding Vertical");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Padding Vertical'] required by convolution.\n"); 

  if (isDefined(js, "Padding Horizontal"))
  {
    try
    {
      _paddingHorizontal = js["Padding Horizontal"].get<ssize_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ convolution ] \n + Key:    ['Padding Horizontal']\n%s", e.what());
    }
    eraseValue(js, "Padding Horizontal");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Padding Horizontal'] required by convolution.\n"); 

  if (isDefined(js, "Padding Size"))
  {
    try
    {
      _paddingSize = js["Padding Size"].get<ssize_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ convolution ] \n + Key:    ['Padding Size']\n%s", e.what());
    }
    eraseValue(js, "Padding Size");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Padding Size'] required by convolution.\n"); 

  if (isDefined(js, "Filters"))
  {
    try
    {
      _filters = js["Filters"].get<ssize_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ convolution ] \n + Key:    ['Filters']\n%s", e.what());
    }
    eraseValue(js, "Filters");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Filters'] required by convolution.\n"); 

  if (isDefined(js, "Algorithm", "Forward"))
  {
    try
    {
      _algorithmForward = js["Algorithm"]["Forward"].get<std::string>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ convolution ] \n + Key:    ['Algorithm']['Forward']\n%s", e.what());
    }
    eraseValue(js, "Algorithm", "Forward");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Algorithm']['Forward'] required by convolution.\n"); 

  if (isDefined(js, "Algorithm", "Backward", "Data"))
  {
    try
    {
      _algorithmBackwardData = js["Algorithm"]["Backward"]["Data"].get<std::string>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ convolution ] \n + Key:    ['Algorithm']['Backward']['Data']\n%s", e.what());
    }
    eraseValue(js, "Algorithm", "Backward", "Data");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Algorithm']['Backward']['Data'] required by convolution.\n"); 

  if (isDefined(js, "Algorithm", "Backward", "Filter"))
  {
    try
    {
      _algorithmBackwardFilter = js["Algorithm"]["Backward"]["Filter"].get<std::string>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ convolution ] \n + Key:    ['Algorithm']['Backward']['Filter']\n%s", e.what());
    }
    eraseValue(js, "Algorithm", "Backward", "Filter");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Algorithm']['Backward']['Filter'] required by convolution.\n"); 

 Layer::setConfiguration(js);
 _type = "layer/convolution";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: convolution: \n%s\n", js.dump(2).c_str());
} 

void Convolution::getConfiguration(knlohmann::json& js) 
{

  js["Type"] = _type;
   js["Image Height"] = _imageHeight;
   js["Image Width"] = _imageWidth;
   js["Kernel Size"] = _kernelSize;
   js["Kernel Height"] = _kernelHeight;
   js["Kernel Width"] = _kernelWidth;
   js["Vertical Stride"] = _verticalStride;
   js["Horizontal Stride"] = _horizontalStride;
   js["Stride Size"] = _strideSize;
   js["Padding Left"] = _paddingLeft;
   js["Padding Right"] = _paddingRight;
   js["Padding Top"] = _paddingTop;
   js["Padding Bottom"] = _paddingBottom;
   js["Padding Vertical"] = _paddingVertical;
   js["Padding Horizontal"] = _paddingHorizontal;
   js["Padding Size"] = _paddingSize;
   js["Filters"] = _filters;
   js["Algorithm"]["Forward"] = _algorithmForward;
   js["Algorithm"]["Backward"]["Data"] = _algorithmBackwardData;
   js["Algorithm"]["Backward"]["Filter"] = _algorithmBackwardFilter;
 Layer::getConfiguration(js);
} 

void Convolution::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Image Height\": -1, \"Image Width\": -1, \"Kernel Size\": -1, \"Kernel Width\": -1, \"Kernel Height\": -1, \"Padding Top\": -1, \"Padding Bottom\": -1, \"Padding Left\": -1, \"Padding Right\": -1, \"Padding Vertical\": -1, \"Padding Horizontal\": -1, \"Padding Size\": 0, \"Vertical Stride\": -1, \"Horizontal Stride\": -1, \"Stride Size\": 1, \"Filters\": -1, \"Algorithm\": {\"Forward\": \"\", \"Backward\": {\"Data\": \"\", \"Filter\": \"\"}}}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Layer::applyModuleDefaults(js);
} 

void Convolution::applyVariableDefaults() 
{

 Layer::applyVariableDefaults();
} 

;

} //layer
} //neuralNetwork
} //korali
;
