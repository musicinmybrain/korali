import math
def LeNet1(e, img_width, img_height, channels):
    """Configure one hidden layer autoencoder.

    :param e: korali experiment
    :param img_height: input/output image height
    :param img_width: input/output image height
    """
    assert img_width == img_height == 28
    input_size = img_width*img_height
    e["Problem"]["Input"]["Size"] = input_size
    ## Convolutional Layer ======================================================================
    ## 28*28 -> 4 x 24*24
    lidx = 0
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Type"] = "Layer/Convolution"
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Image Height"]      = img_width
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Image Width"]       = img_height
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Kernel Size"]       = 5
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Filters"]           = 6
    ## Activation =============================================================================
    lidx += 1
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Type"] = "Layer/Activation"
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Function"] = "Elementwise/Tanh"
    ## Average Pooling =========================================================================
    ## 4 x 24*24 -> 4 x 12*12
    lidx += 1
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Type"] = "Layer/Pooling"
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Function"] = "Average"
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Image Height"]      = 24
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Image Width"]       = 24
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Kernel Size"]       = 2
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Stride Size"]       = 2
    ## Convolutional Layer ======================================================================
    ## 4 x 12*12 -> 12 x 8*8
    lidx += 1
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Type"] = "Layer/Convolution"
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Image Height"]      = 12
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Image Width"]       = 12
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Kernel Size"]       = 5
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Filters"]           = 3
    # ## Activation =============================================================================
    lidx += 1
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Type"] = "Layer/Activation"
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Function"] = "Elementwise/Tanh"
    # ## Average Pooling =========================================================================
    # ## 12 x 8*8 -> 12 x 4*4
    lidx += 1
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Type"] = "Layer/Pooling"
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Function"] = "Average"
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Image Height"]      = 8
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Image Width"]       = 8
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Kernel Size"]       = 2
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Stride Size"]       = 2
    # Convolutiona =============================================================================
    e["Solver"]["Neural Network"]["Output Activation"] = "Softmax"

def LeNet5(e, img_width, img_height, channels):
    """Configure one hidden layer autoencoder.

    :param e: korali experiment
    :param img_height: input/output image height
    :param img_width: input/output image height
    """
    assert img_width == img_height == 32
    input_size = img_width*img_height
    e["Problem"]["Input"]["Size"] = input_size
    ## Convolutional Layer ======================================================================
    ## 32*32 -> 6 x 28*28
    lidx = 0
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Type"] = "Layer/Convolution"
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Image Height"]      = img_width
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Image Width"]       = img_height
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Kernel Size"]       = 5
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Padding Size"]      = 2
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Filters"]           = 6
    ## Activation =============================================================================
    lidx += 1
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Type"] = "Layer/Activation"
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Function"] = "Elementwise/Sigmoid"
    ## Average Pooling =========================================================================
    ## 6 x 28*28 -> 6 x 14*14
    lidx += 1
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Type"] = "Layer/Pooling"
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Function"] = "Average"
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Image Height"]      = 28
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Image Width"]       = 28
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Kernel Size"]       = 2
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Stride Size"]       = 2
    ## Convolutional Layer ======================================================================
    ## 6 x 14*14 -> 16 x 10*10
    lidx += 1
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Type"] = "Layer/Convolution"
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Image Height"]      = 14
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Image Width"]       = 14
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Kernel Size"]       = 5
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Filters"]           = 16
    # ## Activation =============================================================================
    lidx += 1
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Type"] = "Layer/Activation"
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Function"] = "Elementwise/Sigmoid"
    # Average Pooling =========================================================================
    # 16 x 10*10 -> 16 x 5*5
    lidx += 1
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Type"] = "Layer/Pooling"
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Function"] = "Average"
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Image Height"]      = 10
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Image Width"]       = 10
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Kernel Size"]       = 2
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Stride Size"]       = 2
    # Linear Layer =============================================================================
    lidx += 1
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Type"] = "Layer/Linear"
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Output Channels"] = 120
    # ## Activation =============================================================================
    lidx += 1
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Type"] = "Layer/Activation"
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Function"] = "Elementwise/Sigmoid"
    # Linear Layer =============================================================================
    lidx += 1
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Type"] = "Layer/Linear"
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Output Channels"] = 84
    # ## Activation =============================================================================
    lidx += 1
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Type"] = "Layer/Activation"
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Function"] = "Elementwise/Sigmoid"
    # Convolutiona =============================================================================
    # Orignially no output activation
    # e["Solver"]["Neural Network"]["Output Activation"] = "Softmax"
