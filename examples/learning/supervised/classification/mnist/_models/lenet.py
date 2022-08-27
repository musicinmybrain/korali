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
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Kernel Height"]     = 5
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Kernel Width"]      = 5
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Filters"]           = 4
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
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Kernel Height"]     = 2
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Kernel Width"]      = 2
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Stride"]            = 2
    ## Convolutional Layer ======================================================================
    ## 4 x 12*12 -> 12 x 8*8
    lidx += 1
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Type"] = "Layer/Convolution"
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Image Height"]      = 12
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Image Width"]       = 12
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Kernel Height"]     = 5
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Kernel Width"]      = 5
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Filters"]           = 3
    # ## Activation =============================================================================
    lidx += 1
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Type"] = "Layer/Activation"
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Function"] = "Elementwise/Tanh"
    # ## Average Pooling =========================================================================
    # ## 12 x 8*8 -> 12 x 4*4
    lidx += 1
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Type"] = "Layer/Pooling"
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Image Height"]      = 8
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Image Width"]       = 8
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Function"] = "Average"
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Kernel Height"]     = 2
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Kernel Width"]      = 2
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Stride"]            = 2
    # Convolutiona =============================================================================
    e["Solver"]["Neural Network"]["Output Activation"] = "Softmax"
