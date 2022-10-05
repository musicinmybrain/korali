def single_layer_ffnn(e, dim, layers, smallest_layer_size):
    """Configure one hidden layer autoencoder.

    :param e: korali experiment
    :param dims: input/output/layers dimensions
    """
    IC=OC=HS=dim
    lidx = 0
    # ===================== Linear Layer
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Type"] = "Layer/Linear"
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Output Channels"] = HS
    # Activation ========================
    lidx+=1
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Type"] = "Layer/Activation"
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Function"] = "Elementwise/ReLU"
    # Activation ========================
    # Note: Input/Output in [0,1] s.t. we can do this
    e["Solver"]["Neural Network"]["Output Activation"] = "Elementwise/ReLU"

def multi_layer_ffnn(e, dim, layers, smallest_layer_size):
    """Configure one hidden layer autoencoder.

    :param e: korali experiment
    :param dims: input/output/layers dimensions
    """
    lidx = 0
    for i in range(layers):
        # ===================== Linear Layer
        e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Type"] = "Layer/Linear"
        e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Output Channels"] = smallest_layer_size
        lidx+=1
        # Activation ========================
        e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Type"] = "Layer/Activation"
        e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Function"] = "Elementwise/ReLU"
        lidx+=1
    # Activation ========================
    # Note: Input/Output in [0,1] s.t. we can do this
    e["Solver"]["Neural Network"]["Output Activation"] = "Elementwise/ReLU"
