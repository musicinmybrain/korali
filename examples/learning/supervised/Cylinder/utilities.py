import pickle

def min_max_scalar(arr):
    """Scales data to [0, 1] range

    :param arr:
    :returns:

    """
    return (arr - arr.min()) / (arr.max() - arr.min())

def print_header(text="", width=80, sep="="):
    """Prints header with seperator

    :param text:
    :param width:
    :param sep:
    :returns:
    """
    if len(text) == 0:
        print(sep*width)
    else:
        txt_legnth = len(text)+2
        fill_width = int((width-txt_legnth)/2)
        print(sep*fill_width+" "+text+" "+sep*fill_width)

def print_args(d, header_text = "Running with args", width=30, header_width=80, sep="="):
    """print args from args parser formated nicely

    :param d:
    :param heder_text:
    :param width:
    :param header_width:
    :param sep:
    :returns:
    """
    print_header(header_text)
    for key, value in d.items():
        # print('\t' * indent + str(key))
        # if isinstance(value, dict):
        #    pretty(value, indent+1)
        # else:
            # print('\t' * (indent+1) + str(value))
        out_string = '\t{:<{width}} {:<}'.format(key, value, width=width)
        print(out_string)
    print_header()

def get_output_dim(I, P1, P2, K, S):
    img = (I+P1+P2-K)
    if img % 2 == 1:
        raise ValueError(
            "(I+P1+P2-K) has to be divisible by K ({:}+{:}+{:}-{:})/{:}"
            .format(I, P1, P2, K, S))
    return int((I+P1+P2-K)/S+1)

def getSamePadding(stride, image_size, filter_size):
    """TODO describe function

    :param stride:
    :param image_size:
    :param filter_size:
    :returns:

    """
    # Input image (W_i,W_i)
    # Output image (W_o,W_o) with W_o = (W_i - F + 2P)/S + 1
    # W_i == W_o -> P = ((S-1)W + F - S)/2
    S = stride
    W = image_size  # width or height
    F = filter_size
    half_pad = int((S - 1) * W - S + F)
    if half_pad % 2 == 1:
        raise ValueError(
            "(S-1) * W  - S + F has to be divisible by two ({:}-1)*{:} - {:} + {:} = {:}"
            .format(S, W, S, F, half_pad))
    else:
        pad = int(half_pad / 2)
    if (pad > image_size / 2):
        raise ValueError(
            "Very large padding P={:}, compared to input width {:}. Reduce the strides."
            .format(pad, image_size))
    return pad

def make_test_data(input_path, output_path, samples):
    """Helper function to create test file

    :param input_path:
    :param output_path:
    :param samples: number of samples to keep
    :returns:

    """
    with open(input_path, "rb") as file:
        data = pickle.load(file)
        assert samples < len(data["trajectories"])
        trajectories = data["trajectories"][:samples]
        del data
    with open(output_path, "wb") as file:
        pickle.dump(trajectories, file)
