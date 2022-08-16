import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def make_parser():
    """Create the argument parser."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--epochs',
        help='Maximum Number of epochs to run',
        default=30,
        type=int,
        required=False)
    parser.add_argument(
        '--validationSplit',
        help='Batch size to use for validation data',
        type=float,
        default=0.2,
        required=False)
    parser.add_argument(
        '--trainingBS',
        help='Batch size to use for training data',
        default=256,
        type=int,
        required=False)
    parser.add_argument(
        '--testingBS',
        help='Batch size to use for test data',
        default=256,
        type=int,
        required=False)
    parser.add_argument(
        '--optimizer',
        help='Optimizer to use for NN parameter updates',
        default='Adam',
        required=False)
    parser.add_argument(
        '-lr',
        '--learningRate',
        help='Learning rate for the selected optimizer',
        default=0.001,
        type=float,
        required=False)
    parser.add_argument(
        '--regularizerCoefficient',
        help='Reguilrizer Coefficient',
        default=0,
        type=float,
        required=False)
    parser.add_argument(
        "--latentDim",
        help="Latent dimension of the encoder",
        default=4,
        required=False,
        type=int
    )
    parser.add_argument(
        "-p",
        "--plot",
        help="Indicates if to plot the losses.",
        required=False,
        action="store_true"
    )
    parser.add_argument(
        "-s",
        "--save",
        help="Indicates if to save the results to _results.",
        required=False,
        action="store_true"
    )
    parser.add_argument(
        "--seed",
        help="Indicates if to save the results to _results.",
        required=False,
        type=int,
        default=42
    )
    parser.add_argument(
        "--shuffle",
        help="Indicates whether to shuffel the training, validation and test data.",
        required=False,
        type=bool,
        default=True
    )
    parser.add_argument(
        "--model",
        help="Indicates which model to use.",
        required=False,
        type=str,
        choices=["linear", "lenet"],
        default="linear"
    )
    parser.add_argument(
        "--yscale",
        help="yscale to plot",
        default="log",
        required=False,
        choices=["linear", "log"]
    )
    return parser
