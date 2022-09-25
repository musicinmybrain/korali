def set_complex_model(e, input_dims):
    """Configure one hidden layer autoencoder.

    :param e: korali experiment
    :param input_dims: encoding dimension
    """
    # ===================== Input Layer
    e["Problem"]["Input"]["Size"] = input_dims
    e["Problem"]["Solution"]["Size"] = input_dims
    # ===================== Linear Layer
    e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear"
    e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 200
    ## Activation ========================
    e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation"
    e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/ReLU"
    # ===================== Linear Layer
    e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear"
    e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = 100
    ## Activation ========================
    e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation"
    e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/ReLU"
