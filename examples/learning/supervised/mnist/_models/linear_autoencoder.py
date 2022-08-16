import math

def configure_autencoder(e, img_width, img_height, channels, latentDim):
    """Configure one hidden layer autoencoder.

    :param e: korali experiment
    :param latentDim: encoding dimension
    :param img_height: input/output image height
    :param img_width: input/output image height
    """
    input_size = output_size = img_width*img_height*channels
    e["Problem"]["Input"]["Size"] = input_size
    #  ==========================================================================================
    #                       Encoder
    #  ==========================================================================================
    # w x h x oc: 28x28x1 -> latentDim
    ## Linear Layer =============================================================================
    lidx = 0
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Type"] = "Layer/Linear"
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Output Channels"] = latentDim
    #  ==========================================================================================
    #                       Decoder
    #  ==========================================================================================
    ## latentDim -> 28x28x1
    ## Linear Layer =============================================================================
    e["Solver"]["Neural Network"]["Output Activation"] = "Elementwise/Logistic"
