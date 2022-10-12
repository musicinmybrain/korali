#include "modules/neuralNetwork/layer/recurrent/lstm/lstm.hpp"
#include "modules/neuralNetwork/neuralNetwork.hpp"
#include<csignal>

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
namespace recurrent
{
;

void LSTM::initialize()
{
  Recurrent::initialize();

  // Setting number of recurrent gates to 4 (LSTM)
  _gateCount = 4;

#ifdef _KORALI_USE_CUDNN
  if (_nn->_engine == "CuDNN") _rnnMode = CUDNN_LSTM;
#endif
}

void LSTM::createForwardPipeline()
{
  // Calling base layer function
  Layer::createForwardPipeline();

  // Checking Layer sizes
  if (_outputChannels == 0) KORALI_LOG_ERROR("Node count for layer (%lu) should be larger than zero.\n", _index);

#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    // Checking Layer sizes
    const memory::dim T = 1;                            // time steps
    const memory::dim G = _gateCount;                   // Gates
    const memory::dim N = _batchSize;                   // Batch size
    const memory::dim IC = _prevLayer->_outputChannels; // channels
    const memory::dim OC = _outputChannels;             // channels
    const memory::dim L = _depth;                       // layers
    const memory::dim D = 1;                            // directions

    // Creating descriptor for layer memory
    const memory::dims layerInputDims = {T, N, IC};
    auto layerInputMemDesc = memory::desc(layerInputDims, memory::data_type::f32, memory::format_tag::tnc);

    // Creating descriptor for layer memory
    const memory::dims layerOutputDims = {T, N, OC};
    auto layerOutputMemDesc = memory::desc(layerOutputDims, memory::data_type::f32, memory::format_tag::tnc);

    // Creating descriptor for the hidden state memory
    const memory::dims stateLayerDims = {L, D, N, OC};
    auto stateMemDesc = memory::desc(stateLayerDims, memory::data_type::f32, memory::format_tag::ldnc);

    // Creating descriptor for the weights memory
    memory::dims weightsInputDims = {L, D, IC, G, OC};
    auto weightInputMemDesc = memory::desc(weightsInputDims, memory::data_type::f32, memory::format_tag::any);

    memory::dims weightsRecurrentDims = {L, D, OC, G, OC};
    auto weightRecurrentMemDesc = memory::desc(weightsRecurrentDims, memory::data_type::f32, memory::format_tag::any);

    // Creating memory for the hidden state
    _hiddenStateMem.resize(_nn->_timestepCount);
    for (size_t i = 0; i < _nn->_timestepCount; i++) _hiddenStateMem[i] = memory(stateMemDesc, _nn->_dnnlEngine);

    // Creating memory for the cell state
    _cellStateMem.resize(_nn->_timestepCount);
    for (size_t i = 0; i < _nn->_timestepCount; i++) _cellStateMem[i] = memory(stateMemDesc, _nn->_dnnlEngine);

    // Crating null hidden state mems for initial timestep
    _nullStateInputMem = memory(stateMemDesc, _nn->_dnnlEngine);
    _nullStateOutputMem = memory(stateMemDesc, _nn->_dnnlEngine);

    // Setting them to zero
    std::vector<float> nullState(L * D * N * OC, 0.0f);
    write_to_dnnl_memory(nullState.data(), _nullStateInputMem);
    write_to_dnnl_memory(nullState.data(), _nullStateOutputMem);

    // Creating descriptor for the LSTM operation
    auto forwardLSTMDesc = lstm_forward::desc(
      _propKind,                                // aprop_kind
      rnn_direction::unidirectional_left2right, // direction
      layerInputMemDesc,                        // src_layer_desc
      stateMemDesc,                             // src_iter_desc
      stateMemDesc,                             // src_iter_c_desc
      weightInputMemDesc,                       // weights_layer_desc
      weightRecurrentMemDesc,                   // weights_iter_desc
      _biasMem.get_desc(),                      // bias_desc
      layerOutputMemDesc,                       // dst_layer_desc
      stateMemDesc,                             // dst_iter_desc
      stateMemDesc                              // dst_iter_c_desc
    );

    // Create LSTM primitive descriptor.
    dnnl::primitive_attr lstmPrimitiveAttributes;
    _forwardLSTMPrimitiveDesc = lstm_forward::primitive_desc(forwardLSTMDesc, lstmPrimitiveAttributes, _nn->_dnnlEngine);

    // Create the primitive.
    _forwardLSTMPrimitive = lstm_forward(_forwardLSTMPrimitiveDesc);

    // Now allocating workspace
    _workspaceMem.resize(_nn->_timestepCount);
    for (size_t i = 0; i < _nn->_timestepCount; i++)
      _workspaceMem[i] = memory(_forwardLSTMPrimitiveDesc.workspace_desc(), _nn->_dnnlEngine);
  }
#endif

#ifdef _KORALI_USE_CUDNN
  if (_nn->_engine == "CuDNN")
  {
    // Obtaining batch size
    const size_t L = _depth;
    const size_t N = _batchSize;
    const size_t IC = _prevLayer->_outputChannels;
    const size_t OC = _outputChannels;

    int dimA[3];
    int strideA[3];

    dimA[0] = L;  // Hidden Layer count
    dimA[1] = N;  // Minibatch size
    dimA[2] = OC; // Hidden Size

    strideA[0] = dimA[2] * dimA[1];
    strideA[1] = dimA[2];
    strideA[2] = 1;

    // Allocating hidden state descriptor
    cudnnErrCheck(cudnnCreateTensorDescriptor(&_hTensorDesc));
    cudnnErrCheck(cudnnSetTensorNdDescriptor(_hTensorDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));

    // Allocating hidden state tensors
    _hStateTensor.resize(_nn->_timestepCount);
    for (size_t i = 0; i < _nn->_timestepCount; i++) cudaErrCheck(cudaMalloc((void **)&_hStateTensor[i], L * N * OC * sizeof(float)));

    // Allocating cell state descriptor
    cudnnErrCheck(cudnnCreateTensorDescriptor(&_cTensorDesc));
    cudnnErrCheck(cudnnSetTensorNdDescriptor(_cTensorDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));

    // Allocating cell state tensors
    _cStateTensor.resize(_nn->_timestepCount);
    for (size_t i = 0; i < _nn->_timestepCount; i++) cudaErrCheck(cudaMalloc((void **)&_cStateTensor[i], L * N * OC * sizeof(float)));

    // Creating RNN data descriptors for input and output
    cudnnErrCheck(cudnnCreateRNNDataDescriptor(&_inputRNNDataDesc));
    cudnnErrCheck(cudnnCreateRNNDataDescriptor(&_outputRNNDataDesc));

    // Setting and copying sequence length array to device
    std::vector<int> seqLengthArray(N, 1);
    cudaErrCheck(cudaMalloc((void **)&_devSequenceLengths, N * sizeof(int)));
    cudaErrCheck(cudaMemcpy(_devSequenceLengths, seqLengthArray.data(), N * sizeof(int), cudaMemcpyHostToDevice));

    // Setting intput/output RNN data descriptors
    cudnnErrCheck(cudnnSetRNNDataDescriptor(
      _inputRNNDataDesc,
      CUDNN_DATA_FLOAT,
      CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED,
      1, // Max Sequence Length
      N,
      IC,
      seqLengthArray.data(),
      NULL));

    cudnnErrCheck(cudnnSetRNNDataDescriptor(
      _outputRNNDataDesc,
      CUDNN_DATA_FLOAT,
      CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED,
      1, // Max Sequence Length
      N,
      OC,
      seqLengthArray.data(),
      NULL));

    // Now allocating workspace
    cudnnErrCheck(cudnnGetRNNTempSpaceSizes(
      _nn->_cuDNNHandle,
      _rnnDesc,
      _forwardMode,
      _inputRNNDataDesc,
      &_workSpaceSize,
      &_reserveSpaceSize));

    _workSpaceTensor.resize(_nn->_timestepCount);
    for (size_t t = 0; t < _nn->_timestepCount; t++) cudaErrCheck(cudaMalloc((void **)&_workSpaceTensor[t], _workSpaceSize));

    _reserveSpaceTensor.resize(_nn->_timestepCount);
    for (size_t t = 0; t < _nn->_timestepCount; t++) cudaErrCheck(cudaMalloc((void **)&_reserveSpaceTensor[t], _reserveSpaceSize));
  }
#endif
}

void LSTM::createBackwardPipeline()
{
  // Calling base layer function
  Recurrent::createBackwardPipeline();

  // Checking Layer sizes
  if (_outputChannels == 0) KORALI_LOG_ERROR("Node count for layer (%lu) should be larger than zero.\n", _index);

  std::exception_ptr eptr;
try{
#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    // Checking Layer sizes
    const memory::dim T = 1;                            // time steps
    const memory::dim G = _gateCount;                   // layers
    const memory::dim N = _batchSize;                   // Batch size
    const memory::dim IC = _prevLayer->_outputChannels; // channels
    const memory::dim OC = _outputChannels;             // channels
    const memory::dim L = _depth;                       // layers
    const memory::dim D = 1;                            // directions

    // Creating descriptor for layer memory
    const memory::dims layerInputDims = {T, N, IC};
    auto layerInputMemDesc = memory::desc(layerInputDims, memory::data_type::f32, memory::format_tag::tnc);

    // Creating descriptor for layer memory
    const memory::dims layerOutputDims = {T, N, OC};
    auto layerOutputMemDesc = memory::desc(layerOutputDims, memory::data_type::f32, memory::format_tag::tnc);

    // Creating descriptor for the hidden state memory
    const memory::dims stateLayerDims = {L, D, N, OC};
    auto stateMemDesc = memory::desc(stateLayerDims, memory::data_type::f32, memory::format_tag::ldnc);

    // Creating descriptor for the weights memory
    memory::dims weightsInputDims = {L, D, IC, G, OC};
    auto weightInputMemDesc = memory::desc(weightsInputDims, memory::data_type::f32, memory::format_tag::any);

    memory::dims weightsRecurrentDims = {L, D, OC, G, OC};
    auto weightRecurrentMemDesc = memory::desc(weightsRecurrentDims, memory::data_type::f32, memory::format_tag::any);

    // Creating memory for the hidden state
    _hiddenStateGradientMem.resize(_nn->_timestepCount);
    for (size_t i = 0; i < _nn->_timestepCount; i++) _hiddenStateGradientMem[i] = memory(_hiddenStateMem[i].get_desc(), _nn->_dnnlEngine);

    // Creating memory for the hidden state
    _cellStateGradientMem.resize(_nn->_timestepCount);
    for (size_t i = 0; i < _nn->_timestepCount; i++) _cellStateGradientMem[i] = memory(_cellStateMem[i].get_desc(), _nn->_dnnlEngine);


    // Creating descriptor for the LSTM operation
    auto backwardLSTMDesc = lstm_backward::desc(
      prop_kind::backward,           // aprop_kind
      rnn_direction::unidirectional, // direction
      layerInputMemDesc,             // src_layer_desc
      stateMemDesc,                  // src_iter_desc
      stateMemDesc,                  // src_iter_c_desc
      weightInputMemDesc,            // weights_layer_desc
      weightRecurrentMemDesc,        // weights_iter_desc
      _biasMem.get_desc(),           // bias_desc
      layerOutputMemDesc,            // dst_layer_desc
      stateMemDesc,                  // dst_iter_desc
      stateMemDesc,                  // dst_iter_c_desc
      layerInputMemDesc,             // diff_src_layer_desc
      stateMemDesc,                  // diff_src_iter_desc
      stateMemDesc,                  // diff_src_iter_c_desc
      weightInputMemDesc,            // diff_weights_layer_desc
      weightRecurrentMemDesc,        // diff_weights_iter_desc
      _biasGradientMem.get_desc(),   // diff_bias_desc
      layerOutputMemDesc,            // diff_dst_layer_desc
      stateMemDesc,                  // diff_dst_iter_desc
      stateMemDesc                   // diff_dst_iter_c_desc
    );

    // Create LSTM primitive descriptor.
    _backwardLSTMPrimitiveDesc = lstm_backward::primitive_desc(backwardLSTMDesc, _nn->_dnnlEngine, _forwardLSTMPrimitiveDesc);

    // Create the primitive.
    _backwardLSTMPrimitive = lstm_backward(_backwardLSTMPrimitiveDesc);
  }
#endif

#ifdef _KORALI_USE_CUDNN
  if (_nn->_engine == "CuDNN")
  {
    // Obtaining batch size
    size_t L = _depth;
    size_t N = _batchSize;
    size_t C = _outputChannels;

    // Allocating hidden state tensors
    _hGradientTensor.resize(_nn->_timestepCount);
    for (size_t i = 0; i < _nn->_timestepCount; i++) cudaErrCheck(cudaMalloc((void **)&_hGradientTensor[i], L * N * C * sizeof(float)));

    // Allocating hidden state tensors
    _cGradientTensor.resize(_nn->_timestepCount);
    for (size_t i = 0; i < _nn->_timestepCount; i++) cudaErrCheck(cudaMalloc((void **)&_cGradientTensor[i], L * N * C * sizeof(float)));
  }
#endif
  } catch (...) {
    eptr = std::current_exception();
  }
  exceptionHandler(eptr);
}

void LSTM::forwardData(const size_t t)
{
#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    // Configuring forward arguments
    std::unordered_map<int, memory> forwardLSTMArgs;
    forwardLSTMArgs.insert({DNNL_ARG_SRC_LAYER, _prevLayer->_outputMem[t]});
    forwardLSTMArgs.insert({DNNL_ARG_WEIGHTS_LAYER, _weightsLayerMem});
    forwardLSTMArgs.insert({DNNL_ARG_WEIGHTS_ITER, _weightsRecurrentMem});
    forwardLSTMArgs.insert({DNNL_ARG_BIAS, _biasMem});
    forwardLSTMArgs.insert({DNNL_ARG_DST_LAYER, _outputMem[t]});
    forwardLSTMArgs.insert({DNNL_ARG_SRC_ITER, t == 0 ? _nullStateInputMem : _hiddenStateMem[t - 1]}); // Input
    forwardLSTMArgs.insert({DNNL_ARG_SRC_ITER_C, t == 0 ? _nullStateInputMem : _cellStateMem[t - 1]}); // Input
    forwardLSTMArgs.insert({DNNL_ARG_DST_ITER, _hiddenStateMem[t]});                                   // Output
    forwardLSTMArgs.insert({DNNL_ARG_DST_ITER_C, _cellStateMem[t]});                                   // Output
    forwardLSTMArgs.insert({DNNL_ARG_WORKSPACE, _workspaceMem[t]});

    // Primitive execution
    _forwardLSTMPrimitive.execute(_nn->_dnnlStream, forwardLSTMArgs);
  }
#endif

#ifdef _KORALI_USE_CUDNN
  if (_nn->_engine == "CuDNN")
  {
    const size_t N = _batchSize;

    // Creating array of sequence lengths necessary for CuDNN
    std::vector<int> seqLengthArray(N, 1);

    cudnnErrCheck(cudnnRNNForward(
      _nn->_cuDNNHandle, // handle
      _rnnDesc,          // rnnDesc
      _forwardMode,
      _devSequenceLengths,                  // devSeqLengths
      _inputRNNDataDesc,                    // xDesc
      _prevLayer->_outputTensor[t],         // x
      _outputRNNDataDesc,                   // yDesc
      _outputTensor[t],                     // y
      _hTensorDesc,                         // hDesc
      t == 0 ? NULL : _hStateTensor[t - 1], // hx
      _hStateTensor[t],                     // hy
      _cTensorDesc,                         // cDesc
      t == 0 ? NULL : _cStateTensor[t - 1], // cx
      _cStateTensor[t],                     // cy
      _weightsSize,
      _weightsTensor,
      _workSpaceSize,
      _workSpaceTensor[t],
      _reserveSpaceSize,
      _reserveSpaceTensor[t]));
  }
#endif
#ifdef DEBUG
  auto outVec = getOutput();
  // Check for non-finite values
  for(auto& batch : outVec){
    if(std::any_of(batch.begin(), batch.end(), [](const float v) { return !std::isfinite(v);}))
      KORALI_LOG_ERROR("[Layer %zu/Type %s/Time %zu] Non-finite value inside forward output values.", _index, _type.c_str(), t);
  }
#endif
}

void LSTM::backwardData(const size_t t)
{
  if (_nn->_mode == "Inference")
    KORALI_LOG_ERROR("Requesting Layer backward data propagation but NN was configured for inference only.\n");

#ifdef _KORALI_USE_ONEDNN
  if (_nn->_engine == "OneDNN")
  {
    // Resetting current weight gradients to zero
    std::vector<float> nullWeightLayerGradients(_weightsInputCount, 0.0f);
    std::vector<float> nullWeightRecurrentGradients(_weightsRecurrentCount, 0.0f);
    std::vector<float> nullBiasGradients(_biasCount, 0.0f);
    write_to_dnnl_memory(nullWeightLayerGradients.data(), _weightsLayerGradientMem);
    write_to_dnnl_memory(nullWeightRecurrentGradients.data(), _weightsRecurrentGradientMem);
    write_to_dnnl_memory(nullBiasGradients.data(), _biasGradientMem);
    // =============================================
    // REORDER weights to RNN cell suggested format:
    // =============================================
    // SRC and DST Layer Weights
    auto weights_layer_bwd_memory = _weightsLayerMem;
    if (_backwardLSTMPrimitiveDesc.weights_layer_desc() != _forwardLSTMPrimitiveDesc.weights_layer_desc()) {
        weights_layer_bwd_memory = memory(_backwardLSTMPrimitiveDesc.weights_layer_desc(), _nn->_dnnlEngine);
        reorder(_weightsLayerMem, weights_layer_bwd_memory).execute(_nn->_dnnlStream, _weightsLayerMem, weights_layer_bwd_memory);
    }
    // Iter Weights for src and dst
    auto weights_iter_bwd_memory = _weightsRecurrentMem;
    if (_backwardLSTMPrimitiveDesc.weights_iter_desc() != _forwardLSTMPrimitiveDesc.weights_iter_desc()) {
        weights_iter_bwd_memory = memory(_backwardLSTMPrimitiveDesc.weights_iter_desc(), _nn->_dnnlEngine);
        reorder(_weightsRecurrentMem, weights_iter_bwd_memory).execute(_nn->_dnnlStream, _weightsRecurrentMem, weights_iter_bwd_memory);
    }
    // Diffs
    auto reorder_diff_weights_layer = false;
    auto diff_weights_layer_memory = _weightsLayerGradientMem;
    if (_backwardLSTMPrimitiveDesc.diff_weights_layer_desc() != _weightsLayerGradientMem.get_desc()) {
        diff_weights_layer_memory = dnnl::memory(_backwardLSTMPrimitiveDesc.diff_weights_layer_desc(), _nn->_dnnlEngine);
        reorder(_weightsLayerGradientMem, diff_weights_layer_memory).execute(_nn->_dnnlStream, _weightsLayerGradientMem, diff_weights_layer_memory);
        reorder_diff_weights_layer = true;
    }
    auto reorder_diff_weights_iter = false;
    auto diff_weights_iter_memory = _weightsRecurrentGradientMem;
    if (_backwardLSTMPrimitiveDesc.diff_weights_iter_desc() != _weightsRecurrentGradientMem.get_desc()) {
        diff_weights_iter_memory = dnnl::memory(_backwardLSTMPrimitiveDesc.diff_weights_iter_desc(), _nn->_dnnlEngine);
        reorder(_weightsRecurrentGradientMem, diff_weights_iter_memory).execute(_nn->_dnnlStream, _weightsRecurrentGradientMem, diff_weights_iter_memory);
        reorder_diff_weights_iter = true;
    }

    // Configuring backward arguments
    std::unordered_map<int, memory> backwardArgs;
    backwardArgs.insert({DNNL_ARG_SRC_LAYER, _prevLayer->_outputMem[t]});
    backwardArgs.insert({DNNL_ARG_SRC_ITER, t == 0 ? _nullStateInputMem : _hiddenStateMem[t - 1]});
    backwardArgs.insert({DNNL_ARG_SRC_ITER_C, t == 0 ? _nullStateInputMem : _cellStateMem[t - 1]});
    backwardArgs.insert({DNNL_ARG_BIAS, _biasMem});
    backwardArgs.insert({DNNL_ARG_DST_LAYER, _outputMem[t]});
    backwardArgs.insert({DNNL_ARG_DST_ITER, t == _nn->_timestepCount - 1 ? _nullStateInputMem : _hiddenStateMem[t]});
    backwardArgs.insert({DNNL_ARG_DST_ITER_C, t == _nn->_timestepCount - 1 ? _nullStateInputMem : _cellStateMem[t]});
    // DONE: need differently reordered memory here
    backwardArgs.insert({DNNL_ARG_WEIGHTS_LAYER, weights_layer_bwd_memory});
    backwardArgs.insert({DNNL_ARG_WEIGHTS_ITER, weights_iter_bwd_memory});
    // DONE: need differently reordered memory here
    backwardArgs.insert({DNNL_ARG_DIFF_WEIGHTS_LAYER, diff_weights_layer_memory});
    backwardArgs.insert({DNNL_ARG_DIFF_WEIGHTS_ITER, diff_weights_iter_memory});

    backwardArgs.insert({DNNL_ARG_DIFF_BIAS, _biasGradientMem});
    backwardArgs.insert({DNNL_ARG_DIFF_SRC_LAYER, _prevLayer->_outputGradientMem[t]});
    backwardArgs.insert({DNNL_ARG_DIFF_SRC_ITER, t == 0 ? _nullStateOutputMem : _hiddenStateGradientMem[t - 1]});
    backwardArgs.insert({DNNL_ARG_DIFF_SRC_ITER_C, t == 0 ? _nullStateOutputMem : _cellStateGradientMem[t - 1]});
    backwardArgs.insert({DNNL_ARG_DIFF_DST_LAYER, _outputGradientMem[t]});
    backwardArgs.insert({DNNL_ARG_DIFF_DST_ITER, t == _nn->_timestepCount - 1 ? _nullStateInputMem : _hiddenStateGradientMem[t]});
    backwardArgs.insert({DNNL_ARG_DIFF_DST_ITER_C, t == _nn->_timestepCount - 1 ? _nullStateInputMem : _cellStateGradientMem[t]});
    backwardArgs.insert({DNNL_ARG_WORKSPACE, _workspaceMem[t]});
    _backwardLSTMPrimitive.execute(_nn->_dnnlStream, backwardArgs);

    if (reorder_diff_weights_layer) {
        reorder(diff_weights_layer_memory, _weightsLayerGradientMem).execute(_nn->_dnnlStream, diff_weights_layer_memory, _weightsLayerGradientMem);
        // _weightsLayerGradientMem = diff_weights_layer_memory;
    }
    if (reorder_diff_weights_iter) {
        reorder(diff_weights_iter_memory, _weightsRecurrentGradientMem).execute(_nn->_dnnlStream, diff_weights_iter_memory, _weightsRecurrentGradientMem);
        // _weightsRecurrentGradientMem = diff_weights_iter_memory;
    }
  }
#endif

#ifdef _KORALI_USE_CUDNN
  if (_nn->_engine == "CuDNN")
  {
    cudnnErrCheck(cudnnRNNBackwardData_v8(
      _nn->_cuDNNHandle,                                         // handle
      _rnnDesc,                                                  // rnnDesc
      _devSequenceLengths,                                       // devSeqLengths
      _outputRNNDataDesc,                                        // yDesc
      _outputTensor[t],                                          // y
      _outputGradientTensor[t],                                  // dy
      _inputRNNDataDesc,                                         // xDesc
      _prevLayer->_outputGradientTensor[t],                      // dx
      _hTensorDesc,                                              // hDesc
      t == 0 ? NULL : _hStateTensor[t - 1],                      // hx
      t == _nn->_timestepCount - 1 ? NULL : _hGradientTensor[t], // dhy
      t == 0 ? NULL : _hGradientTensor[t - 1],                   // dhx
      _cTensorDesc,                                              // cDesc
      t == 0 ? NULL : _cStateTensor[t - 1],                      // cx
      t == _nn->_timestepCount - 1 ? NULL : _cGradientTensor[t], // chy
      t == 0 ? NULL : _cGradientTensor[t - 1],                   // chx
      _weightsSize,
      _weightsTensor,
      _workSpaceSize,
      _workSpaceTensor[t],
      _reserveSpaceSize,
      _reserveSpaceTensor[t]));
  }
#endif
}

void LSTM::setConfiguration(knlohmann::json& js) 
{
  if (isDefined(js, "Results"))  eraseValue(js, "Results");

 Recurrent::setConfiguration(js);
 _type = "layer/recurrent/lstm";
 if(isDefined(js, "Type")) eraseValue(js, "Type");
 if(isEmpty(js) == false) KORALI_LOG_ERROR(" + Unrecognized settings for Korali module: lstm: \n%s\n", js.dump(2).c_str());
} 

void LSTM::getConfiguration(knlohmann::json& js) 
{

  js["Type"] = _type;
 Recurrent::getConfiguration(js);
} 

void LSTM::applyModuleDefaults(knlohmann::json& js) 
{

 std::string defaultString = "{}";
 knlohmann::json defaultJs = knlohmann::json::parse(defaultString);
 mergeJson(js, defaultJs); 
 Recurrent::applyModuleDefaults(js);
} 

void LSTM::applyVariableDefaults() 
{

 Recurrent::applyVariableDefaults();
} 

;

} //recurrent
} //layer
} //neuralNetwork
} //korali
;
