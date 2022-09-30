def set_simple_model(e, input_dims):
    # ===================== Input Layer
    e["Problem"]["Input"]["Size"] = input_dims
    e["Problem"]["Solution"]["Size"] = input_dims
    # ===================== Linear Layer
    lidx = 0
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Type"] = "Layer/Linear"
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Output Channels"] = 32
    ## Activation ========================
    lidx += 1
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Type"] = "Layer/Activation"
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Function"] = "Elementwise/Tanh"
    # ===================== Linear Layer
    lidx += 1
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Type"] = "Layer/Linear"
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Output Channels"] = 32
    ## Activation ========================
    lidx += 1
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Type"] = "Layer/Activation"
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Function"] = "Elementwise/Tanh"
