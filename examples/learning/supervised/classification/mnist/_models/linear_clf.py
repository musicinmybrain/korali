import math


def LogisticRegression(e, img_width, img_height, channels, latentDim):
    """Configure one hidden layer autoencoder.

    :param e: korali experiment
    :param latentDim: encoding dimension
    :param img_height: input/output image height
    :param img_width: input/output image height
    """
    input_size = img_width*img_height*channels
    e["Problem"]["Input"]["Size"] = input_size
    ## Linear Layer =============================================================================
    ## Is automatically added by giving the output size of 10
    e["Solver"]["Neural Network"]["Output Activation"] = "Softmax"


def LinearClf(e, img_width, img_height, channels, latentDim):
    """Configure one hidden layer autoencoder.

    :param e: korali experiment
    :param latentDim: encoding dimension
    :param img_height: input/output image height
    :param img_width: input/output image height
    """
    input_size = img_width*img_height*channels
    e["Problem"]["Input"]["Size"] = input_size
    ## Linear Layer =============================================================================
    lidx = 0
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Type"] = "Layer/Linear"
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Output Channels"] = 300
    ## Activation =============================================================================
    lidx += 1
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Type"] = "Layer/Activation"
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Function"] = "Elementwise/Tanh"
    ## Linear Layer =============================================================================
    lidx += 1
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Type"] = "Layer/Linear"
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Output Channels"] = 100
    ## Activation =============================================================================
    lidx += 1
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Type"] = "Layer/Activation"
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Function"] = "Elementwise/Tanh"
    ## Linear Layer =============================================================================
    e["Solver"]["Neural Network"]["Output Activation"] = "Softmax"
