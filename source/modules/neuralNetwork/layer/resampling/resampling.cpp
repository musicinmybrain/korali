#include "modules/neuralNetwork/layer/resampling/resampling.hpp"
#include "modules/neuralNetwork/neuralNetwork.hpp"

#ifdef _KORALI_USE_CUDNN
  #include "auxiliar/cudaUtils.hpp"
#endif

#ifdef _KORALI_USE_ONEDNN
  #include "auxiliar/dnnUtils.hpp"
using namespace dnnl;
#endif

namespace korali
{
namespace neuralNetwork
{
namespace layer
{
;

void Resampling::initialize()
{
  if (_nn->_engine == "Korali" || _nn->_engine == "CuDNN")
  {
    KORALI_LOG_ERROR("Resampling layer not yet implemented for engine %s .\n", _nn->_engine);
  }
  // Checking Layer size
  if (_outputChannels == 0) KORALI_LOG_ERROR("Node count for layer (%lu) should be larger than zero.\n", _index);

  // Checking position
  if (_index == 0) KORALI_LOG_ERROR("Feed Forward layers cannot be the starting layer of the NN\n");
  if (_index == _nn->_layers.size() - 1) KORALI_LOG_ERROR("Feed Forward layers cannot be the last layer of the NN\n");
  // Precalculating values for the resampling operation
  N = _batchSize;
  IH = _imageHeight;
  IW = _imageWidth;
  OH = _outputHeight;
  OW = _outputWidth;
  IC = _prevLayer->_outputChannels / (IH * IW);
  OC = _outputChannels / (OH * OW);
  // Determining algorithm
#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    if (_resamplingType == "Linear") _algorithm_t = dnnl::algorithm::resampling_linear;
    else if (_resamplingType == "Nearest") _algorithm_t = dnnl::algorithm::resampling_nearest;
    else KORALI_LOG_ERROR("[Layer %zu] resmpling method \"%s\" is not a valid option [\"Linear\", \"Nearest\"].\n", _index-1, _resamplingType.c_str());
  }
#endif

#ifdef _KORALI_USE_CUDNN
  if (_nn->_engine == "CuDNN")
  {
    if (_resamplingType == "Linear") _algorithm_t = CUDNN_RESAMPLE_BILINEAR;
    else if (_resamplingType == "Nearest") _algorithm_t = CUDNN_RESAMPLE_NEAREST;
    else KORALI_LOG_ERROR("[Layer %zu] resmpling method \"%s\" is not a valid option [\"Linear\", \"Nearest\"].\n", _index-1, _resamplingType.c_str());
  }
#endif
}

void Resampling::createForwardPipeline()
{

#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    // Setting propagation kind
    _propKind = _nn->_mode == "Training" ? prop_kind::forward_training : prop_kind::forward_inference;
    // Creating layer's data memory storage
    // const memory::dims layerDims = {N, OC*OH*OW};
    const memory::dims layerDims = {N, OC*OH*OW};
    auto dataMemDesc = memory::desc(layerDims, memory::data_type::f32, memory::format_tag::nc);
    // Creating activation layer memory
    _outputMem.resize(_nn->_timestepCount);
    for (size_t t = 0; t < _nn->_timestepCount; t++)
      _outputMem[t] = memory(dataMemDesc, _nn->_dnnlEngine);

    // Creating memory descriptor mappings for input memory
    _srcMemDesc = memory::desc({N, IC, IH, IW}, memory::data_type::f32, memory::format_tag::nchw);
    _dstMemDesc = memory::desc({N, OC, OH, OW}, memory::data_type::f32, memory::format_tag::nchw);
    // Create resampling operation descriptor (No difference between training and inference).
    auto resampling_d = resampling_forward::desc(prop_kind::forward_training, std::get<dnnl::algorithm>(_algorithm_t), _srcMemDesc, _dstMemDesc);
    // Create primitive descriptor.
    _forwardResamplingPrimitiveDesc = resampling_forward::primitive_desc(resampling_d, _nn->_dnnlEngine);
    // Create the primitive.
    _forwardResamplingPrimitive = resampling_forward(_forwardResamplingPrimitiveDesc);
  }
#endif

#ifdef _KORALI_USE_CUDNN
  if (_nn->_engine == "CuDNN")
  {
    // TODO
    cudnnBackendCreateDescriptor(CUDNN_BACKEND_RESAMPLE_DESCRIPTOR, &_cudnnResamplingDesc);
    // cudnnBackendSetAttribute(_cudnnResamplingDesc,
    //                          CUDNN_ATTR_RESAMPLE_MODE,
    //                          CUDNN_TYPE_RESAMPLE_MODE, 1, &resample_mode);

    // std::get<dnnl::algorithm>(_algorithm_t)
  }
#endif
}

void Resampling::createBackwardPipeline()
{
  /*********************************************************************************
   *  Initializing memory objects and primitives for BACKWARD propagation
   *********************************************************************************/
  // Calling base layer function
  Layer::createBackwardPipeline();
  if (_nn->_engine == "Korali")
  {
    // TODO
  }

// Creating backward propagation primitives
#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    // Creating memory descriptor mappings for input memory
    _srcMemDesc = memory::desc({N, IC, IH, IW}, memory::data_type::f32, memory::format_tag::nchw);
    _dstMemDesc = memory::desc({N, OC, OH, OW}, memory::data_type::f32, memory::format_tag::nchw);

    auto backwardDataDesc = resampling_backward::desc(std::get<dnnl::algorithm>(_algorithm_t), _srcMemDesc, _dstMemDesc);
    // Create the primitivel descriptor.
    auto backwardDataPrimitiveDesc = resampling_backward::primitive_desc(backwardDataDesc, _nn->_dnnlEngine, _forwardResamplingPrimitiveDesc);
    // Create the primitive.
    _backwardDataPrimitive = resampling_backward(backwardDataPrimitiveDesc);
  }
#endif

#ifdef _KORALI_USE_CUDNN
  if (_nn->_engine == "CuDNN")
  {
    // TODO
    // cudnnBackendCreateDescriptor(CUDNN_BACKEND_RESAMPLE_DESCRIPTOR, _cudnnResamplingDesc);
  }
#endif
}

void Resampling::forwardData(const size_t t)
{
  if (_nn->_engine == "Korali")
  {
    // TODO
  }
#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    std::unordered_map<int, memory> resampling_args;
    resampling_args[DNNL_ARG_SRC] = _prevLayer->_outputMem[t];
    resampling_args[DNNL_ARG_DST] = _outputMem[t];
    _forwardResamplingPrimitive.execute(_nn->_dnnlStream, resampling_args);
  }
#endif

#ifdef _KORALI_USE_CUDNN
  // TODO
#endif
}

void Resampling::backwardData(const size_t t)
{
  if (_nn->_mode == "Inference")
    KORALI_LOG_ERROR("Requesting Layer backward data propagation but NN was configured for inference only.\n");

  if (_nn->_engine == "Korali")
  {
    // TODO
  }

#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    _backwardDataArgs[DNNL_ARG_DIFF_DST] = _outputGradientMem[t];             // Input
    _backwardDataArgs[DNNL_ARG_DIFF_SRC] = _prevLayer->_outputGradientMem[t]; // Output
    _backwardDataPrimitive.execute(_nn->_dnnlStream, _backwardDataArgs);
  }
#endif

#ifdef _KORALI_USE_CUDNN
  if (_nn->_engine == "CuDNN")
  {
    // TODO
  }
#endif
}



void Resampling::setConfiguration(knlohmann::json& js) 
{
  if (isDefined(js, "Results"))  eraseValue(js, "Results");

  if (isDefined(js, "Resampling Type"))
  {
    try
    {
      _resamplingType = js["Resampling Type"].get<std::string>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ resampling ] \n + Key:    ['Resampling Type']\n%s", e.what());
    }
      {
        bool validOption = false; 
        if (_resamplingType == "Nearest") validOption = true; 
        if (_resamplingType == "Linear") validOption = true; 
        if (validOption == false) KORALI_LOG_ERROR("Unrecognized value (%s) provided for mandatory setting: ['Resampling Type'] required by resampling.\n Valid Options are:\n  - Nearest\n  - Linear\n",_resamplingType.c_str()); 
      }
    eraseValue(js, "Resampling Type");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Resampling Type'] required by resampling.\n"); 

  if (isDefined(js, "Image Height"))
  {
    try
    {
      _imageHeight = js["Image Height"].get<ssize_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ resampling ] \n + Key:    ['Image Height']\n%s", e.what());
    }
    eraseValue(js, "Image Height");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Image Height'] required by resampling.\n"); 

  if (isDefined(js, "Image Width"))
  {
    try
    {
      _imageWidth = js["Image Width"].get<ssize_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ resampling ] \n + Key:    ['Image Width']\n%s", e.what());
    }
    eraseValue(js, "Image Width");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Image Width'] required by resampling.\n"); 

  if (isDefined(js, "Output Height"))
  {
    try
    {
      _outputHeight = js["Output Height"].get<ssize_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ resampling ] \n + Key:    ['Output Height']\n%s", e.what());
    }
    eraseValue(js, "Output Height");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Output Height'] required by resampling.\n"); 

  if (isDefined(js, "Output Width"))
  {
    try
    {
      _outputWidth = js["Output Width"].get<ssize_t>();
    } catch (const std::exception& e) {
      KORALI_LOG_ERROR(" + Object: [ resampling ] \n + Key:    ['Output Width']\n%s", e.what());
    }
    eraseValue(js, "Output Width");
  }  else  KORALI_LOG_ERROR(" + No value provided for mandatory setting: ['Output Width'] required by resampling.\n"); 

 Layer::setConfiguration(js);
 _type = "layer/resampling";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: resampling: \n%s\n", js.dump(2).c_str());
} 

void Resampling::getConfiguration(knlohmann::json& js) 
{

  js["Type"] = _type;
   js["Resampling Type"] = _resamplingType;
   js["Image Height"] = _imageHeight;
   js["Image Width"] = _imageWidth;
   js["Output Height"] = _outputHeight;
   js["Output Width"] = _outputWidth;
 Layer::getConfiguration(js);
} 

void Resampling::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{\"Resampling Type\": \"Linear\"}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Layer::applyModuleDefaults(js);
} 

void Resampling::applyVariableDefaults() 
{

 Layer::applyVariableDefaults();
} 

;

} //layer
} //neuralNetwork
} //korali
;
