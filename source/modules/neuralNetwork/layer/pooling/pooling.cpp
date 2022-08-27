#include "modules/neuralNetwork/layer/pooling/pooling.hpp"
#include "modules/neuralNetwork/layer/convolution/convolution.hpp"
#include "modules/neuralNetwork/layer/deconvolution/deconvolution.hpp"
#include "modules/neuralNetwork/layer/resampling/resampling.hpp"
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

void Pooling::initialize()
{
  // Checking Layer size
  if (_filters == -1 && _outputChannels == 0){
    _k->_logger->logInfo("Detailed", "[%s Layer %lu] No output channels specified, assuming OC = IC.\n", _type.c_str(), _index-1);
  }

  // Checking position
  if (_index == 0) KORALI_LOG_ERROR("Pooling layers cannot be the starting layer of the NN\n");
  if (_index == _nn->_layers.size() - 1) KORALI_LOG_ERROR("Pooling layers cannot be the last layer of the NN\n");

  // Precalculating values for the pooling operation
  N = _batchSize;
  IH = _imageHeight;
  IW = _imageWidth;
  KH = _kernelHeight;
  KW = _kernelWidth;

  // Strides ==============================================================================
  SV = SH = _stride;
  if( _verticalStride != -1)
    SV = _verticalStride;
  if( _horizontalStride != -1)
    SH = _horizontalStride;
  // Paddings =============================================================================
  PT = PL = PB = PR = _padding;
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

  // Image Height and Image Width (if not given deduction from prev layer) =================
  // TODO: make this better
  if (IW <= 0 || IH <= 0){
    Layer *_prevHiddenLayer{};
    if(_prevLayer->_type == "layer/activation")
      _prevHiddenLayer = _prevLayer->_prevLayer;
    else
      _prevHiddenLayer = _prevLayer;
    if (auto _prevCastedLayer = dynamic_cast<const Convolution*>(_prevHiddenLayer)){
      if (IH <= 0) {
        if (IH <= 0) IH = _prevCastedLayer->OH;
        _k->_logger->logInfo("Detailed", "[%s layer %zu] Deducing image height %lu from previous layer.\n", _type.c_str(), _index-1, IH);
      }
      if (IW <= 0){
        if (IW <= 0) IW = _prevCastedLayer->OW;
        _k->_logger->logInfo("Detailed", "[%s layer %zu] Deducing image width %lu from previous layer.\n", _type.c_str(), _index-1, IW);
      }
    } else if (auto _prevCastedLayer = dynamic_cast<const Deconvolution*>(_prevHiddenLayer)){
      if (IH <= 0) {
        if (IH <= 0) IH = _prevCastedLayer->OH;
        _k->_logger->logInfo("Detailed", "[%s layer %zu] Deducing image height %lu from previous layer.\n", _type.c_str(), _index-1, IH);
      }
      if (IW <= 0){
        if (IW <= 0) IW = _prevCastedLayer->OW;
        _k->_logger->logInfo("Detailed", "[%s layer %zu] Deducing image width %lu from previous layer.\n", _type.c_str(), _index-1, IW);
      }
    } else if (auto _prevCastedLayer = dynamic_cast<const Pooling*>(_prevHiddenLayer)){
      if (IH <= 0) {
        if (IH <= 0) IH = _prevCastedLayer->OH;
        _k->_logger->logInfo("Detailed", "[%s layer %zu] Deducing image height %lu from previous layer.\n", _type.c_str(), _index-1, IH);
      }
      if (IW <= 0){
        if (IW <= 0) IW = _prevCastedLayer->OW;
        _k->_logger->logInfo("Detailed", "[%s layer %zu] Deducing image width %lu from previous layer.\n", _type.c_str(), _index-1, IW);
      }
    } else if (auto _prevCastedLayer = dynamic_cast<const Resampling*>(_prevHiddenLayer)){
      if (IH <= 0) {
        if (IH <= 0) IH = _prevCastedLayer->OH;
        _k->_logger->logInfo("Detailed", "[%s layer %zu] Deducing image height %lu from previous layer.\n", _type.c_str(), _index-1, IH);
      }
      if (IW <= 0){
        if (IW <= 0) IW = _prevCastedLayer->OW;
        _k->_logger->logInfo("Detailed", "[%s layer %zu] Deducing image width %lu from previous layer.\n", _type.c_str(), _index-1, IW);
      }
    }
  }
  if (IW <= 0) KORALI_LOG_ERROR("[%s layer %zu] Image width cannot be deduced from previous layer and must be larger than zero.\n", _type.c_str(), _index-1);
  if (IH <= 0) KORALI_LOG_ERROR("[%s layer %zu] Image height cannot be deduced from previous layer and must be larger than zero.\n", _type.c_str(), _index-1);
  // ======================================================================================
  if (KH <= 0) KORALI_LOG_ERROR("Kernel height must be larger than zero for pooling layer.\n");
  if (KW <= 0) KORALI_LOG_ERROR("Kernel width must be larger than zero for pooling layer.\n");
  if (SV <= 0) KORALI_LOG_ERROR("Vertical stride must be larger than zero for pooling layer.\n");
  if (SH <= 0) KORALI_LOG_ERROR("Horizontal stride must be larger than zero for pooling layer.\n");

  // Several sanity checks
  if (KH > IH + PR + PL) KORALI_LOG_ERROR("[%s layer %zu] Kernel height cannot be larger than input image height plus padding.\n", _type.c_str(), _index-1);
  if (KW > IW + PT + PB) KORALI_LOG_ERROR("[%s layer %zu] Kernel width cannot be larger than input image width plus padding.\n", _type.c_str(), _index-1);

  // Check whether the output channels of the previous layer is divided by the height and width
  if (_prevLayer->_outputChannels % (IH * IW) > 0) KORALI_LOG_ERROR("[Pooling layer %zu] Previous layer contains a number of channels (%lu) not divisible by the pooling 2D HxW setup (%lux%lu).\n", _index-1, _prevLayer->_outputChannels, IH, IW);
  IC = _prevLayer->_outputChannels / (IH * IW);

  // Deriving output height and width
  OH = (IH - (KH - (PR + PL)) / SH) + 1;
  OW = (IW - (KW - (PT + PB)) / SV) + 1;
  if( ((IH - KH + PT + PB) % SV) != 0)
    _k->_logger->logInfo("Normal", "[Convolutional layer %zu] (IH - KH + PT + PB) / SV = %lu using floor.\n", _index-1, OH-1);
  if( ((IW - KW + PR + PL) % SH) != 0)
    _k->_logger->logInfo("Normal", "[Convolutional layer %zu] (IW - KW + PR + PL) / SH = %lu using floor.\n", _index-1, OW-1);
  if(_outputChannels == 0)
    if(_filters == -1){
      _filters = IC;
      _outputChannels = _filters*OH*OW;
    }
  // Check whether the output channels of the previous layer is divided by the height and width
  if (_outputChannels % (OH * OW) > 0) KORALI_LOG_ERROR("[Pooling layer %zu] Number of output channels (%lu) not divisible by the output image size (%lux%lu) given kernel (%lux%lu) size and padding/stride configuration.\n", _index-1, _outputChannels, OH, OW, KH, KW);
  OC = _outputChannels / (OH * OW);
}

void Pooling::createForwardPipeline()
{
  // Calling base layer function
  Layer::createForwardPipeline();

  if (_nn->_engine == "Korali") KORALI_LOG_ERROR("Pooling Layers still not supported in Korali's NN backend. Use OneDNN.\n");
  if (_nn->_engine == "CuDNN") KORALI_LOG_ERROR("Pooling Layers still not supported in CuDNNbackend. Use OneDNN.\n");

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

    // Creating work memory
    memory::dims kernelDims = {KH, KW};

    // Determining algorithm
    dnnl::algorithm algorithmType;
    if (_function == "Max") algorithmType = dnnl::algorithm::pooling_max;
    if (_function == "Inclusive Average" || _function == "Average") algorithmType = dnnl::algorithm::pooling_avg_include_padding;
    if (_function == "Exclusive Average") algorithmType = dnnl::algorithm::pooling_avg_exclude_padding;

    // We create the pooling operation
    auto pooling_d = pooling_forward::desc(_propKind, algorithmType, _srcMemDesc, _dstMemDesc, ST, kernelDims, PTL, PBR);

    // Create inner product primitive descriptor.
    dnnl::primitive_attr poolingPrimitiveAttributes;
    _forwardPoolingPrimitiveDesc = pooling_forward::primitive_desc(pooling_d, poolingPrimitiveAttributes, _nn->_dnnlEngine);

    // Create pooling workspace memory
    _workspaceMem.resize(_nn->_timestepCount);
    for (size_t t = 0; t < _nn->_timestepCount; t++)
      _workspaceMem[t] = memory(_forwardPoolingPrimitiveDesc.workspace_desc(), _nn->_dnnlEngine);

    // Create the weights+bias primitive.
    _forwardPoolingPrimitive = pooling_forward(_forwardPoolingPrimitiveDesc);
  }
#endif
}

void Pooling::createBackwardPipeline()
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

    // Creating work memory
    memory::dims kernelDims = {KH, KW};

    // Determining algorithm
    dnnl::algorithm algorithmType;
    if (_function == "Max") algorithmType = dnnl::algorithm::pooling_max;
    if (_function == "Inclusive Average" || _function == "Average") algorithmType = dnnl::algorithm::pooling_avg_include_padding;
    if (_function == "Exclusive Average") algorithmType = dnnl::algorithm::pooling_avg_exclude_padding;

    auto backwardDataDesc = pooling_backward::desc(
      algorithmType,
      _srcMemDesc,
      _dstMemDesc,
      ST,
      kernelDims,
      PTL,
      PBR);

    // Create the primitive.
    auto backwardDataPrimitiveDesc = pooling_backward::primitive_desc(backwardDataDesc, _nn->_dnnlEngine, _forwardPoolingPrimitiveDesc);
    _backwardDataPrimitive = pooling_backward(backwardDataPrimitiveDesc);
  }
#endif
}

void Pooling::forwardData(const size_t t)
{
#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    // Arguments to the inner product operation
    std::unordered_map<int, dnnl::memory> forwardPoolingArgs;
    forwardPoolingArgs[DNNL_ARG_SRC] = _prevLayer->_outputMem[t];
    forwardPoolingArgs[DNNL_ARG_DST] = _outputMem[t];
    forwardPoolingArgs[DNNL_ARG_WORKSPACE] = _workspaceMem[t];
    _forwardPoolingPrimitive.execute(_nn->_dnnlStream, forwardPoolingArgs);
  }
#endif
}

void Pooling::backwardData(const size_t t)
{
  if (_nn->_mode == "Inference")
    KORALI_LOG_ERROR("Requesting Layer backward data propagation but NN was configured for inference only.\n");

#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    _backwardDataArgs[DNNL_ARG_DIFF_DST] = _outputGradientMem[t];             // Input
    _backwardDataArgs[DNNL_ARG_DIFF_SRC] = _prevLayer->_outputGradientMem[t]; // Output
    _backwardDataArgs[DNNL_ARG_WORKSPACE] = _workspaceMem[t];
    _backwardDataPrimitive.execute(_nn->_dnnlStream, _backwardDataArgs);
  }
#endif
}

void Pooling::setConfiguration(knlohmann::json& js) 
{
  if (isDefined(js, "Results"))  eraseValue(js, "Results");

  if (isDefined(js, "Function"))
  {
    try
    {
      _function = js["Function"].get<std::string>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ pooling ] \n + Key:    ['Function']\n%s", e.what());
    }
      {
        bool validOption = false; 
        if (_function == "Max") validOption = true; 
        if (_function == "Inclusive Average") validOption = true; 
        if (_function == "Average") validOption = true; 
        if (_function == "Exclusive Average") validOption = true; 
        if (validOption == false) KORALI_LOG_ERROR("Unrecognized value (%s) provided for mandatory setting: ['Function'] required by pooling.\n Valid Options are:\n  - Max\n  - Inclusive Average\n  - Average\n  - Exclusive Average\n",_function.c_str()); 
      }
    eraseValue(js, "Function");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Function'] required by pooling.\n"); 

  if (isDefined(js, "Image Height"))
  {
    try
    {
      _imageHeight = js["Image Height"].get<ssize_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ pooling ] \n + Key:    ['Image Height']\n%s", e.what());
    }
    eraseValue(js, "Image Height");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Image Height'] required by pooling.\n"); 

  if (isDefined(js, "Image Width"))
  {
    try
    {
      _imageWidth = js["Image Width"].get<ssize_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ pooling ] \n + Key:    ['Image Width']\n%s", e.what());
    }
    eraseValue(js, "Image Width");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Image Width'] required by pooling.\n"); 

  if (isDefined(js, "Kernel Height"))
  {
    try
    {
      _kernelHeight = js["Kernel Height"].get<ssize_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ pooling ] \n + Key:    ['Kernel Height']\n%s", e.what());
    }
    eraseValue(js, "Kernel Height");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Kernel Height'] required by pooling.\n"); 

  if (isDefined(js, "Kernel Width"))
  {
    try
    {
      _kernelWidth = js["Kernel Width"].get<ssize_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ pooling ] \n + Key:    ['Kernel Width']\n%s", e.what());
    }
    eraseValue(js, "Kernel Width");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Kernel Width'] required by pooling.\n"); 

  if (isDefined(js, "Vertical Stride"))
  {
    try
    {
      _verticalStride = js["Vertical Stride"].get<ssize_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ pooling ] \n + Key:    ['Vertical Stride']\n%s", e.what());
    }
    eraseValue(js, "Vertical Stride");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Vertical Stride'] required by pooling.\n"); 

  if (isDefined(js, "Horizontal Stride"))
  {
    try
    {
      _horizontalStride = js["Horizontal Stride"].get<ssize_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ pooling ] \n + Key:    ['Horizontal Stride']\n%s", e.what());
    }
    eraseValue(js, "Horizontal Stride");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Horizontal Stride'] required by pooling.\n"); 

  if (isDefined(js, "Stride"))
  {
    try
    {
      _stride = js["Stride"].get<ssize_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ pooling ] \n + Key:    ['Stride']\n%s", e.what());
    }
    eraseValue(js, "Stride");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Stride'] required by pooling.\n"); 

  if (isDefined(js, "Padding Left"))
  {
    try
    {
      _paddingLeft = js["Padding Left"].get<ssize_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ pooling ] \n + Key:    ['Padding Left']\n%s", e.what());
    }
    eraseValue(js, "Padding Left");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Padding Left'] required by pooling.\n"); 

  if (isDefined(js, "Padding Right"))
  {
    try
    {
      _paddingRight = js["Padding Right"].get<ssize_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ pooling ] \n + Key:    ['Padding Right']\n%s", e.what());
    }
    eraseValue(js, "Padding Right");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Padding Right'] required by pooling.\n"); 

  if (isDefined(js, "Padding Top"))
  {
    try
    {
      _paddingTop = js["Padding Top"].get<ssize_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ pooling ] \n + Key:    ['Padding Top']\n%s", e.what());
    }
    eraseValue(js, "Padding Top");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Padding Top'] required by pooling.\n"); 

  if (isDefined(js, "Padding Bottom"))
  {
    try
    {
      _paddingBottom = js["Padding Bottom"].get<ssize_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ pooling ] \n + Key:    ['Padding Bottom']\n%s", e.what());
    }
    eraseValue(js, "Padding Bottom");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Padding Bottom'] required by pooling.\n"); 

  if (isDefined(js, "Padding Vertical"))
  {
    try
    {
      _paddingVertical = js["Padding Vertical"].get<ssize_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ pooling ] \n + Key:    ['Padding Vertical']\n%s", e.what());
    }
    eraseValue(js, "Padding Vertical");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Padding Vertical'] required by pooling.\n"); 

  if (isDefined(js, "Padding Horizontal"))
  {
    try
    {
      _paddingHorizontal = js["Padding Horizontal"].get<ssize_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ pooling ] \n + Key:    ['Padding Horizontal']\n%s", e.what());
    }
    eraseValue(js, "Padding Horizontal");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Padding Horizontal'] required by pooling.\n"); 

  if (isDefined(js, "Padding"))
  {
    try
    {
      _padding = js["Padding"].get<ssize_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ pooling ] \n + Key:    ['Padding']\n%s", e.what());
    }
    eraseValue(js, "Padding");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Padding'] required by pooling.\n"); 

  if (isDefined(js, "Filters"))
  {
    try
    {
      _filters = js["Filters"].get<ssize_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ pooling ] \n + Key:    ['Filters']\n%s", e.what());
    }
    eraseValue(js, "Filters");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Filters'] required by pooling.\n"); 

 Layer::setConfiguration(js);
 _type = "layer/pooling";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: pooling: \n%s\n", js.dump(2).c_str());
} 

void Pooling::getConfiguration(knlohmann::json& js) 
{

  js["Type"] = _type;
   js["Function"] = _function;
   js["Image Height"] = _imageHeight;
   js["Image Width"] = _imageWidth;
   js["Kernel Height"] = _kernelHeight;
   js["Kernel Width"] = _kernelWidth;
   js["Vertical Stride"] = _verticalStride;
   js["Horizontal Stride"] = _horizontalStride;
   js["Stride"] = _stride;
   js["Padding Left"] = _paddingLeft;
   js["Padding Right"] = _paddingRight;
   js["Padding Top"] = _paddingTop;
   js["Padding Bottom"] = _paddingBottom;
   js["Padding Vertical"] = _paddingVertical;
   js["Padding Horizontal"] = _paddingHorizontal;
   js["Padding"] = _padding;
   js["Filters"] = _filters;
 Layer::getConfiguration(js);
} 

void Pooling::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Image Height\": -1, \"Image Width\": -1, \"Padding Top\": -1, \"Padding Bottom\": -1, \"Padding Left\": -1, \"Padding Right\": -1, \"Padding Vertical\": -1, \"Padding Horizontal\": -1, \"Padding\": 0, \"Vertical Stride\": -1, \"Horizontal Stride\": -1, \"Stride\": 1, \"Filters\": -1}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Layer::applyModuleDefaults(js);
} 

void Pooling::applyVariableDefaults() 
{

 Layer::applyVariableDefaults();
} 

;

} //layer
} //neuralNetwork
} //korali
;
