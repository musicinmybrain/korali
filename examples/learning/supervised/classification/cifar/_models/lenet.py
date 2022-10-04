import math

def LeNet5(e, img_width, img_height, label_size, channels=3):
    """Configures LeNet5
    But with ReLU instead of Tanh activaton and Softmax output instead of nothing.

    :param e: korali experiment
    :param img_height: input/output image height
    :param img_width: input/output image height
    """
    assert img_width == img_height == 32
    input_size = img_width*img_height
    ## Convolutional Layer ======================================================================
    ## 32*32 -> 6 x 28*28
    lidx = 0
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Type"] = "Layer/Convolution"
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Image Height"]      = img_width
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Image Width"]       = img_height
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Kernel Size"]       = 5
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Filters"]           = 6
    ## Activation =============================================================================
    lidx += 1
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Type"] = "Layer/Activation"
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Function"] = "Elementwise/ReLU"
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
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Function"] = "Elementwise/ReLU"
    # Average Pooling =========================================================================
    # 16 x 10*10 -> 16 x 5*5
    lidx += 1
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Type"] = "Layer/Pooling"
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Function"] = "Average"
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Image Height"]      = 10
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Image Width"]       = 10
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Kernel Size"]       = 2
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Stride Size"]       = 2
    ## Convolutional Layer ======================================================================
    ## 16 x 5*5 -> 120
    lidx += 1
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Type"] = "Layer/Convolution"
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Image Height"]      = 5
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Image Width"]       = 5
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Kernel Size"]       = 5
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Filters"]           = 120
    # ## Activation =============================================================================
    lidx += 1
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Type"] = "Layer/Activation"
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Function"] = "Elementwise/ReLU"
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
    e["Solver"]["Neural Network"]["Output Activation"] = "Softmax"
