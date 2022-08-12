import math

def configure_autencoder(e, img_width, img_height, channels, latentDim):
    """Configure one hidden layer autoencoder.

    :param e: korali experiment
    :param latentDim: encoding dimension
    :param img_height: input/output image height
    :param img_width: input/output image height
    """
    input_size = output_size = img_width*img_height*channels
    paddingCNN = [1, 1, 0]
    paddingDeCNN = [0, 1, 1]
    output_padding_DeCnn = [0, 1, 1]
    strideCNN = [2, 2, 2]
    kernelSizeCNN = [3, 3, 3]
    ocCNN = [8, 16, 32]
    ocDeCNN = [16, 8, 1]
    get_output_size = lambda inp, k, s, p : math.floor((inp-k+2*p)/s)+1
    get_output_size_deconv = lambda inp, k, s, p, out_padding : s*(inp-1)+k-2*p+out_padding
    CnnLayers = 3
    ocLin = [128, latentDim]
    # ===================== Input Layer
    e["Problem"]["Input"]["Size"] = input_size
    #  ==========================================================================================
    #                       Encoder
    #  ==========================================================================================
    lidx = 0
    #  CNN ======================================================================================
    # w x h x oc: 28x28x1 -> 14x14x8 -> 7x7x16 -> 3x3x32
    for i in range(0, CnnLayers):
        ## Conv =============================================================================
        e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Type"] = "Layer/Convolution"
        e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Image Height"]      = img_width
        e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Image Width"]       = img_height
        e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Padding Left"]      = paddingCNN[i]
        e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Padding Right"]     = paddingCNN[i]
        e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Padding Top"]       = paddingCNN[i]
        e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Padding Bottom"]    = paddingCNN[i]
        e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Kernel Height"]     = kernelSizeCNN[i]
        e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Kernel Width"]      = kernelSizeCNN[i]
        e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Vertical Stride"]   = strideCNN[i]
        e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Horizontal Stride"] = strideCNN[i]
        img_width = get_output_size(img_width, kernelSizeCNN[i], strideCNN[i], paddingCNN[i])
        img_height = get_output_size(img_height, kernelSizeCNN[i], strideCNN[i], paddingCNN[i])
        e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Output Channels"]   = img_height*img_width*ocCNN[i]
        ## Activation =============================================================================
        lidx += 1
        e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Type"] = "Layer/Activation"
        e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Function"] = "Elementwise/ReLU"
        lidx += 1
    ## Linear Layer =============================================================================
    ## 3x3x32 -> 128
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Type"] = "Layer/Linear"
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Output Channels"] = ocLin[0]
    ## Activation =============================================================================
    lidx += 1
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Type"] = "Layer/Activation"
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Function"] = "Elementwise/ReLU"
    ## Linear Layer =============================================================================
    ## 128 -> latentDim
    lidx += 1
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Type"] = "Layer/Linear"
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Output Channels"] = ocLin[1]
    #  ==========================================================================================
    #                       Decoder
    #  ==========================================================================================
    ## Linear Layer =============================================================================
    ## latentDim -> 128
    lidx += 1
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Type"] = "Layer/Linear"
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Output Channels"] = ocLin[0]
    ## Activation =============================================================================
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Type"] = "Layer/Activation"
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Function"] = "Elementwise/ReLU"
    ## Linear Layer =============================================================================
    ## 128 -> 3x3x32
    lidx += 1
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Type"] = "Layer/Linear"
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Output Channels"] = img_height*img_width*ocCNN[-1]
    ## Activation =============================================================================
    lidx += 1
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Type"] = "Layer/Activation"
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Function"] = "Elementwise/ReLU"
    #  De-CNN ======================================================================================
    # w x h x oc: 3x3x32 -> 7x7x16 -> 14x14x8 -> 28x28x1
    for i in range(0, CnnLayers):
        lidx += 1
        ## Conv =============================================================================
        e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Type"] = "Layer/Deconvolution"
        img_width = get_output_size_deconv(img_width, kernelSizeCNN[i], strideCNN[i], paddingDeCNN[i], output_padding_DeCnn[i])
        img_height = get_output_size_deconv(img_height, kernelSizeCNN[i], strideCNN[i], paddingDeCNN[i], output_padding_DeCnn[i])
        e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Image Height"]      = img_width
        e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Image Width"]       = img_height
        e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Kernel Height"]     = kernelSizeCNN[i]
        e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Kernel Width"]      = kernelSizeCNN[i]
        e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Vertical Stride"]   = strideCNN[i]
        e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Horizontal Stride"] = strideCNN[i]
        e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Padding Left"]      = paddingDeCNN[i]
        e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Padding Right"]     = paddingDeCNN[i]
        e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Padding Top"]       = paddingDeCNN[i]
        e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Padding Bottom"]    = paddingDeCNN[i]
        e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Output Channels"]   = img_height*img_width*ocDeCNN[i]
        ## Activation =============================================================================
        lidx += 1
        e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Type"] = "Layer/Activation"
        if i in [0, 1]:
            e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Function"] = "Elementwise/ReLU"
        else:
            e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Function"] = "Elementwise/Logistic"
