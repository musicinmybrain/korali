#include "modules/neuralNetwork/layer/deconvolution/deconvolution.hpp"
#include "modules/neuralNetwork/neuralNetwork.hpp"

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

void Deconvolution::initialize()
{
  // Checking Layer size
  if (_filters == -1 && _outputChannels == 0) KORALI_LOG_ERROR("[%s Layer %lu] No output channesl or filter number specified for layer (%lu).\n", _type.c_str(), _index-1);

  // Checking position
  if (_index == 0) KORALI_LOG_ERROR("[%s Layer %lu] layers cannot be the starting layer of the NN\n", _type.c_str(), _index-1);
  if (_index == _nn->_layers.size() - 1) KORALI_LOG_ERROR("[%s Layer %lu] layers cannot be the last layer of the NN\n", _type.c_str(), _index-1);

  // Precalculating values for the deconvolution operation
  N = _batchSize;
  OH = _imageHeight;
  OW = _imageWidth;
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
  // Check for non zeros
  if (OH <= 0) KORALI_LOG_ERROR("[%s layer %zu] Image height must be larger than zero for deconvolutional layer.\n", _type.c_str(), _index-1);
  if (OW <= 0) KORALI_LOG_ERROR("[%s layer %zu] Image width must be larger than zero for deconvolutional layer.\n", _type.c_str(), _index-1);
  if (KH <= 0) KORALI_LOG_ERROR("[%s layer %zu] Kernel height must be larger than zero for deconvolutional layer.\n", _type.c_str(), _index-1);
  if (KW <= 0) KORALI_LOG_ERROR("[%s layer %zu] Kernel width must be larger than zero for deconvolutional layer.\n", _type.c_str(), _index-1);
  if (SV <= 0) KORALI_LOG_ERROR("[%s layer %zu] Vertical stride must be larger than zero for deconvolutional layer.\n", _type.c_str(), _index-1);
  if (SH <= 0) KORALI_LOG_ERROR("[%s layer %zu] Horizontal stride must be larger than zero for deconvolutional layer.\n", _type.c_str(), _index-1);

  // Several sanity checks
  if (KH > OH + PR + PL) KORALI_LOG_ERROR("[%s layer %zu] Kernel height cannot be larger than output image height plus padding.\n", _type.c_str(), _index-1);
  if (KW > OW + PT + PB) KORALI_LOG_ERROR("[%s layer %zu] Kernel width cannot be larger than output image width plus padding.\n", _type.c_str(), _index-1);
  if(_outputChannels == 0)
    _outputChannels = _filters*OH*OW;

  // Check whether the output channels of the previous layer is divided by the height and width
  if (_outputChannels % (OH * OW) > 0) KORALI_LOG_ERROR("[Deconvolutional layer %zu] Number of channels (%lu) not divisible by the 2D HxW setup (%lux%lu).\n", _index-1, _outputChannels, OH, OW);
  OC = _outputChannels / (OH * OW);
  // Deriving input height and width
  // TODO Check if this is simply same padding convolution
  IH = ((OH - KH + PT + PB) / SV) + 1;
  IW = ((OW - KW + PR + PL) / SH) + 1;
  // Check whether the output channels of the previous layer is divided by the height and width i.e. if previous layer is a linear layer
  if (_prevLayer->_outputChannels % (IH * IW) > 0) KORALI_LOG_ERROR("[%s layer %zu] Previous layer contains a number of output channels (%lu) not divisible by the image size (%lux%lu) \
                                                                      given kernel (%lux%lu) size and padding/stride configuration.\n", _index-1, _type.c_str(), _prevLayer->_outputChannels, IH, IW, KH, KW);
  IC = _prevLayer->_outputChannels / (IH * IW);
}

std::vector<float> Deconvolution::generateInitialHyperparameters()
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

void Deconvolution::createHyperparameterMemory()
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
                                               // TODO probably need to turn OC and IC arround here
                                               /*out_channels=*/IC,
                                               /*in_channels=*/OC,
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

void Deconvolution::copyHyperparameterPointers(Layer *dstLayer)
{
  Deconvolution *dstPtr = dynamic_cast<Deconvolution *>(dstLayer);
  dstPtr->_hyperparameterCount = _hyperparameterCount;

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

void Deconvolution::createForwardPipeline()
{
  // Calling base layer function
  Layer::createForwardPipeline();

  if (_nn->_engine == "Korali") KORALI_LOG_ERROR("Deconvolutional Layers still not supported in Korali's NN backend. Use OneDNN.\n");

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

    // We create the deconvolution operation
    auto deconvolution_d = deconvolution_forward::desc(_propKind, algorithm::deconvolution_direct, _srcMemDesc, _weightsMem.get_desc(), _biasMem.get_desc(), _dstMemDesc, ST, PTL, PBR);

    // Create inner product primitive descriptor.
    dnnl::primitive_attr deconvolutionPrimitiveAttributes;
    _forwardDeconvolutionPrimitiveDesc = deconvolution_forward::primitive_desc(deconvolution_d, deconvolutionPrimitiveAttributes, _nn->_dnnlEngine);

    // Create the weights+bias primitive.
    _forwardDeconvolutionPrimitive = deconvolution_forward(_forwardDeconvolutionPrimitiveDesc);
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
                      /*channels=*/IC, // 1
                      /*image_height=*/IH, // 12
                      /*image_width=*/IW)); // 12
      // Output Tensor
      cudnnErrCheck(cudnnCreateTensorDescriptor(&_outputDescriptor));
      cudnnErrCheck(cudnnSetTensor4dDescriptor(
                      /*Output. Tensor Descr.=*/ _outputDescriptor,
                      /*format=*/CUDNN_TENSOR_NCHW,
                      /*dataType=*/CUDNN_DATA_FLOAT,
                      /*batch_size=*/N,
                      /*channels=*/OC, // 1
                      /*image_height=*/OH, // 28
                      /*image_width=*/OW)); //28

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
      // NOTE: this must be like in forward to get right ws -> need to turn output and input arround again
        cudnnErrCheck(cudnnGetConvolutionBackwardDataWorkspaceSize(_nn->_cuDNNHandle,
                                                                      /*wDesc=*/_weightsFilterDesc,
                                       /*dyDesc/_outputGradientTensor[t] desc=*/_inputDescriptor,
                                                                   /*convDesc=*/_convolutionDescriptor,
                           /*dxDesc/_prevLayer->_outputGradientTensor[t] desc=*/_outputDescriptor,
                                                                       /*algo=*/CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
                                                              /* *sizeInBytes=*/&_convolutionWorkspaceSize));

#ifdef DEBUG
      // TODO remove at some point
      _k->_logger->logInfo("Detailed", "[%s layer %zu] Allocating %f MB for cuDNN convolution workspace.\n", _type.c_str(), _index-1, _convolutionWorkspaceSize/(1024.0*1024.0));
#endif
      // Create workspace memory in createBackwardpipeline
    }
#endif
  } catch (...) {
    eptr = std::current_exception();
  }
  exceptionHandler(eptr);
}

void Deconvolution::createBackwardPipeline()
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

    auto backwardDataDesc = deconvolution_backward_data::desc(
      algorithm::deconvolution_direct,
      _srcMemDesc,
      _weightsMem.get_desc(),
      _dstMemDesc,
      ST,
      PTL,
      PBR);

    // Create the primitive.
    auto backwardDataPrimitiveDesc = deconvolution_backward_data::primitive_desc(backwardDataDesc, _nn->_dnnlEngine, _forwardDeconvolutionPrimitiveDesc);
    _backwardDataPrimitive = deconvolution_backward_data(backwardDataPrimitiveDesc);

    auto backwardWeightsDesc = deconvolution_backward_weights::desc(
      algorithm::deconvolution_direct,
      _srcMemDesc,
      _weightsMem.get_desc(),
      _biasMem.get_desc(),
      _dstMemDesc,
      ST,
      PTL,
      PBR);

    // Create the primitive.
    auto backwardWeightsPrimitiveDesc = deconvolution_backward_weights::primitive_desc(backwardWeightsDesc, _nn->_dnnlEngine, _forwardDeconvolutionPrimitiveDesc);
    _backwardWeightsPrimitive = deconvolution_backward_weights(backwardWeightsPrimitiveDesc);
  }
#endif

#ifdef _KORALI_USE_CUDNN
  if (_nn->_engine == "CuDNN")
  {
    cudaErrCheck(cudaMalloc((void **)&_weightsGradientFilter, _weightsCount * sizeof(float)));
    cudaErrCheck(cudaMalloc((void **)&_biasGradientTensor, _biasCount * sizeof(float)));
    auto backwardWsSize = getBackwardWsSize();
    _convolutionWorkspaceSize = std::max(_convolutionWorkspaceSize, backwardWsSize);
    cudaErrCheck(cudaMalloc((void **)&_convolutionWorkspace, _convolutionWorkspaceSize * sizeof(float)));

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

#endif
  }
}

void Deconvolution::forwardData(const size_t t)
{
#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    // Arguments to the inner product operation
    std::unordered_map<int, dnnl::memory> forwardDeconvolutionArgs;
    forwardDeconvolutionArgs[DNNL_ARG_SRC] = _prevLayer->_outputMem[t];
    forwardDeconvolutionArgs[DNNL_ARG_WEIGHTS] = _weightsMem;
    forwardDeconvolutionArgs[DNNL_ARG_BIAS] = _biasMem;
    forwardDeconvolutionArgs[DNNL_ARG_DST] = _outputMem[t];

    _forwardDeconvolutionPrimitive.execute(_nn->_dnnlStream, forwardDeconvolutionArgs);
  }
#endif

#ifdef _KORALI_USE_CUDNN
  if (_nn->_engine == "CuDNN")
  {
    float alpha = 1.0f;
    float beta = 0.0f;
    cudnnErrCheck(cudnnConvolutionBackwardData(_nn->_cuDNNHandle,
                                               &alpha,
                                               _weightsFilterDesc,
                                               _weightsFilter,
                                               /*dyDesc/_outputDescriptor desc=*/_inputDescriptor,
                                               /*dy/_outputGradientTensor[t]=*/_prevLayer->_outputTensor[t],
                                               /*convDesc*/_convolutionDescriptor,
                                               CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
                                               _convolutionWorkspace,
                                               _convolutionWorkspaceSize,
                                               &beta,
                                               /*dxDesc/_inputDescriptor desc=*/_outputDescriptor,
                                               /*dx/_prevLayer->_outputGradientTensor[t]/result=*/_outputTensor[t]));
  }
#endif
}

void Deconvolution::backwardData(const size_t t)
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
    float alpha1 = 1.0f;
    float alpha2 = 0.0f;
    cudnnErrCheck(cudnnConvolutionForward(_nn->_cuDNNHandle,
                                          /*alpha=*/&alpha1,
                                          /*xDesc/inputDesc=*/_outputDescriptor,
                                          /*x/_prevLayer->_outputTensor[t]=*/_outputGradientTensor[t],
                                          _weightsFilterDesc,
                                          _weightsFilter,
                                          _convolutionDescriptor,
                                          _convolutionAlgorithm,
                                          _convolutionWorkspace,
                                          _convolutionWorkspaceSize,
                                          /*beta=*/&alpha2,
                                          /*yDesc/outputDesc=*/_inputDescriptor,
                                          /*y/_outputTensor[t])/result=*/_prevLayer->_outputGradientTensor[t]));
    float alpha = 1.0f;
    float beta = 1.0f;
    // cudnnAddTensor(_nn->_cuDNNHandle, &alpha, _biasTensorDesc, _biasTensor, &beta, _outputDescriptor, _outputTensor[t]);
    // cudnnConvolutionBiasActivationForward()
  }
#endif
}

void Deconvolution::backwardHyperparameters(size_t t)
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
    // cudnnErrCheck(cudnnConvolutionBackwardBias(
    //   _nn->_cuDNNHandle,
    //   &alpha,
    //   _outputDescriptor,
    //   _outputGradientTensor[t],
    //   &beta,
    //   _biasTensorDesc,
    //   _biasGradientTensor));

    // cudnnErrCheck(cudnnConvolutionBackwardFilter(
    //                 _nn->_cuDNNHandle,
    //                 &alpha,
    //                 /*xDesc=*/_inputDescriptor,
    //                 /*x=*/_prevLayer->_outputTensor[t],
    //                 /*dyDesc=*/_outputDescriptor,
    //                 /*y=*/_outputGradientTensor[t],
    //                 _convolutionDescriptor,
    //                 CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1,
    //                 _convolutionWorkspace,
    //                 _convolutionWorkspaceSize,
    //                 &beta,
    //                 _weightsFilterDesc,
    //                 _weightsGradientFilter));


  }
#endif
}

void Deconvolution::setHyperparameters(const float *hyperparameters)
{
#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    write_to_dnnl_memory(&hyperparameters[0], _weightsMem);
    write_to_dnnl_memory(&hyperparameters[OC * IC * KH * KW], _biasMem);
  }
#endif
}

void Deconvolution::getHyperparameters(float *hyperparameters)
{
#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    read_from_dnnl_memory(&hyperparameters[0], _weightsMem);
    read_from_dnnl_memory(&hyperparameters[OC * IC * KH * KW], _biasMem);
  }
#endif
}

void Deconvolution::getHyperparameterGradients(float *gradient)
{
#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    read_from_dnnl_memory(&gradient[0], _weightsGradientMem);
    read_from_dnnl_memory(&gradient[OC * IC * KH * KW], _biasGradientMem);
  }
#endif
}

#ifdef _KORALI_USE_CUDNN
size_t Deconvolution::getBackwardWsSize() {
        size_t sizeFilterAlg = 0;
        size_t sizeDataAlg = 0;
        // if (!_algorithmBackwardFilter.empty())
        // TODO: algorithm search cuDNN v8
      cudnnFilterDescriptor_t tmpWeightsFilterDesc;
      cudnnErrCheck(cudnnCreateFilterDescriptor(&tmpWeightsFilterDesc));
      cudnnErrCheck(cudnnSetFilter4dDescriptor(tmpWeightsFilterDesc,
                                              /*dataType=*/CUDNN_DATA_FLOAT,
                                               /*format=*/CUDNN_TENSOR_NCHW,
                                               /*out_channels=*/IC,
                                               /*in_channels=*/OC,
                                               /*kernel_height=*/KH,
                                               /*kernel_width=*/KW));
        cudnnErrCheck(cudnnGetConvolutionBackwardFilterWorkspaceSize(_nn->_cuDNNHandle,
                                                                     _outputDescriptor,
                                                                     _inputDescriptor,
                                                                     _convolutionDescriptor,
                                                                     tmpWeightsFilterDesc,
                                                                     CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1,
                                                                     &sizeFilterAlg));
        // if (!_algorithmBackwardData.empty())
        // TODO: algorithm search cuDNN v8
       // Backward Data Part
       // Input descriptor is now output descriptor as this is backward
        cudnnErrCheck(cudnnGetConvolutionForwardWorkspaceSize(_nn->_cuDNNHandle,
                                                              /*xDesc=*/_outputDescriptor,
                                                              _weightsFilterDesc,
                                                              _convolutionDescriptor,
                                                              /*yDesc*/_inputDescriptor,
                                                              /*_convolutionAlgorith=*/CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
                                                              &_convolutionWorkspaceSize));
        return std::max(sizeFilterAlg, sizeDataAlg);
}
#endif

void Deconvolution::setConfiguration(knlohmann::json& js) 
{
  if (isDefined(js, "Results"))  eraseValue(js, "Results");

  if (isDefined(js, "Image Height"))
  {
    try
    {
      _imageHeight = js["Image Height"].get<ssize_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ deconvolution ] \n + Key:    ['Image Height']\n%s", e.what());
    }
    eraseValue(js, "Image Height");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Image Height'] required by deconvolution.\n"); 

  if (isDefined(js, "Image Width"))
  {
    try
    {
      _imageWidth = js["Image Width"].get<ssize_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ deconvolution ] \n + Key:    ['Image Width']\n%s", e.what());
    }
    eraseValue(js, "Image Width");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Image Width'] required by deconvolution.\n"); 

  if (isDefined(js, "Kernel Size"))
  {
    try
    {
      _kernelSize = js["Kernel Size"].get<ssize_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ deconvolution ] \n + Key:    ['Kernel Size']\n%s", e.what());
    }
    eraseValue(js, "Kernel Size");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Kernel Size'] required by deconvolution.\n"); 

  if (isDefined(js, "Kernel Height"))
  {
    try
    {
      _kernelHeight = js["Kernel Height"].get<ssize_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ deconvolution ] \n + Key:    ['Kernel Height']\n%s", e.what());
    }
    eraseValue(js, "Kernel Height");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Kernel Height'] required by deconvolution.\n"); 

  if (isDefined(js, "Kernel Width"))
  {
    try
    {
      _kernelWidth = js["Kernel Width"].get<ssize_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ deconvolution ] \n + Key:    ['Kernel Width']\n%s", e.what());
    }
    eraseValue(js, "Kernel Width");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Kernel Width'] required by deconvolution.\n"); 

  if (isDefined(js, "Vertical Stride"))
  {
    try
    {
      _verticalStride = js["Vertical Stride"].get<ssize_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ deconvolution ] \n + Key:    ['Vertical Stride']\n%s", e.what());
    }
    eraseValue(js, "Vertical Stride");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Vertical Stride'] required by deconvolution.\n"); 

  if (isDefined(js, "Horizontal Stride"))
  {
    try
    {
      _horizontalStride = js["Horizontal Stride"].get<ssize_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ deconvolution ] \n + Key:    ['Horizontal Stride']\n%s", e.what());
    }
    eraseValue(js, "Horizontal Stride");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Horizontal Stride'] required by deconvolution.\n"); 

  if (isDefined(js, "Stride Size"))
  {
    try
    {
      _strideSize = js["Stride Size"].get<ssize_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ deconvolution ] \n + Key:    ['Stride Size']\n%s", e.what());
    }
    eraseValue(js, "Stride Size");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Stride Size'] required by deconvolution.\n"); 

  if (isDefined(js, "Padding Left"))
  {
    try
    {
      _paddingLeft = js["Padding Left"].get<ssize_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ deconvolution ] \n + Key:    ['Padding Left']\n%s", e.what());
    }
    eraseValue(js, "Padding Left");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Padding Left'] required by deconvolution.\n"); 

  if (isDefined(js, "Padding Right"))
  {
    try
    {
      _paddingRight = js["Padding Right"].get<ssize_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ deconvolution ] \n + Key:    ['Padding Right']\n%s", e.what());
    }
    eraseValue(js, "Padding Right");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Padding Right'] required by deconvolution.\n"); 

  if (isDefined(js, "Padding Top"))
  {
    try
    {
      _paddingTop = js["Padding Top"].get<ssize_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ deconvolution ] \n + Key:    ['Padding Top']\n%s", e.what());
    }
    eraseValue(js, "Padding Top");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Padding Top'] required by deconvolution.\n"); 

  if (isDefined(js, "Padding Bottom"))
  {
    try
    {
      _paddingBottom = js["Padding Bottom"].get<ssize_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ deconvolution ] \n + Key:    ['Padding Bottom']\n%s", e.what());
    }
    eraseValue(js, "Padding Bottom");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Padding Bottom'] required by deconvolution.\n"); 

  if (isDefined(js, "Padding Vertical"))
  {
    try
    {
      _paddingVertical = js["Padding Vertical"].get<ssize_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ deconvolution ] \n + Key:    ['Padding Vertical']\n%s", e.what());
    }
    eraseValue(js, "Padding Vertical");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Padding Vertical'] required by deconvolution.\n"); 

  if (isDefined(js, "Padding Horizontal"))
  {
    try
    {
      _paddingHorizontal = js["Padding Horizontal"].get<ssize_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ deconvolution ] \n + Key:    ['Padding Horizontal']\n%s", e.what());
    }
    eraseValue(js, "Padding Horizontal");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Padding Horizontal'] required by deconvolution.\n"); 

  if (isDefined(js, "Padding Size"))
  {
    try
    {
      _paddingSize = js["Padding Size"].get<ssize_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ deconvolution ] \n + Key:    ['Padding Size']\n%s", e.what());
    }
    eraseValue(js, "Padding Size");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Padding Size'] required by deconvolution.\n"); 

  if (isDefined(js, "Filters"))
  {
    try
    {
      _filters = js["Filters"].get<ssize_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ deconvolution ] \n + Key:    ['Filters']\n%s", e.what());
    }
    eraseValue(js, "Filters");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Filters'] required by deconvolution.\n"); 

 Layer::setConfiguration(js);
 _type = "layer/deconvolution";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: deconvolution: \n%s\n", js.dump(2).c_str());
} 

void Deconvolution::getConfiguration(knlohmann::json& js) 
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
 Layer::getConfiguration(js);
} 

void Deconvolution::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Image Height\": -1, \"Image Width\": -1, \"Kernel Size\": -1, \"Kernel Width\": -1, \"Kernel Height\": -1, \"Padding Top\": -1, \"Padding Bottom\": -1, \"Padding Left\": -1, \"Padding Right\": -1, \"Padding Vertical\": -1, \"Padding Horizontal\": -1, \"Padding Size\": 0, \"Vertical Stride\": -1, \"Horizontal Stride\": -1, \"Stride Size\": 1, \"Filters\": -1}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Layer::applyModuleDefaults(js);
} 

void Deconvolution::applyVariableDefaults() 
{

 Layer::applyVariableDefaults();
} 

;

} //layer
} //neuralNetwork
} //korali
;
