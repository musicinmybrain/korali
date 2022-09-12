import math

def simple_one_layer_cnn(e, img_width, img_height, label_size, channels=1):
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
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Filters"]           = 4
    e["Solver"]["Neural Network"]["Output Activation"] = "Softmax"


def test_cnn_only(e, img_width, img_height, label_size, channels=1):
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
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Filters"]           = 4
    e["Solver"]["Neural Network"]["Hidden Layers"][lidx]["Is Layer Trainable"] = True

    e["Solver"]["Neural Network"]["Output Layer"]["Type"] = "Layer/Linear"
    e["Solver"]["Neural Network"]["Output Layer"]["Is Layer Trainable"] = False
    e["Solver"]["Neural Network"]["Output Layer"]["Output Channels"] = label_size
    e["Solver"]["Neural Network"]["Output Activation"] = "Softmax"
