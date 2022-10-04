import math

def LogisticRegression(e, img_width, img_height, label_size, channels=1):
    """Configure one hidden layer autoencoder.

    :param e: korali experiment
    :param img_height: input/output image height
    :param img_width: input/output image height
    """
    input_size = img_width*img_height*channels
    e["Problem"]["Input"]["Size"] = input_size
    ## Linear Layer =============================================================================
    ## Is automatically added by giving the output size of 10
    e["Solver"]["Neural Network"]["Output Activation"] = "Softmax"
