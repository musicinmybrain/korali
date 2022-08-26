import math

def set_complex_model(e, input_dims):
    """Configure one hidden layer autoencoder.

    :param e: korali experiment
    :param input_dims: encoding dimension
    """
    # ===================== Input Layer
    e["Problem"]["Input"]["Size"] = input_dims
    e["Problem"]["Solution"]["Size"] = input_dims
    # ===================== Linear Layer
    e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Linear"
    e["Solver"]["Neural Network"]["Hidden Layers"][1]["Output Channels"] = 160
    ## Activation ========================
    e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Activation"
    e["Solver"]["Neural Network"]["Hidden Layers"][2]["Function"] = "Elementwise/ReLU"
    # ===================== Linear Layer
    e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Linear"
    e["Solver"]["Neural Network"]["Hidden Layers"][3]["Output Channels"] = 480
    ## Activation ========================
    e["Solver"]["Neural Network"]["Hidden Layers"][4]["Type"] = "Layer/Activation"
    e["Solver"]["Neural Network"]["Hidden Layers"][4]["Function"] = "Elementwise/ReLU"
    # ===================== Linear Layer
    e["Solver"]["Neural Network"]["Hidden Layers"][5]["Type"] = "Layer/Linear"
    e["Solver"]["Neural Network"]["Hidden Layers"][5]["Output Channels"] = 256


def set_simple_model(e, input_dims):
    # ===================== Input Layer
    e["Problem"]["Input"]["Size"] = input_dims
    e["Problem"]["Solution"]["Size"] = input_dims
    # ===================== Linear Layer
    e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear"
    e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 32
    ## Activation ========================
    e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation"
    e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/Tanh"
    # ===================== Linear Layer
    e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear"
    e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = 32
    ## Activation ========================
    e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation"
    e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/Tanh"
