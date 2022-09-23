import argparse
import seaborn as sns
import korali
import torch
import numpy as np
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
        default=0.01,
        required=False)
    parser.add_argument(
        '--trainingBS',
        help='Batch size to use for training data',
        default=64,
        type=int,
        required=False)
    parser.add_argument(
        '--testingBS',
        help='Batch size to use for test data',
        default=64,
        type=int,
        required=False)
    parser.add_argument(
        '--optimizer',
        help='Optimizer to use for NN parameter updates',
        default='Adam',
        required=False)
    parser.add_argument(
        '--regularizerCoefficient',
        help='Reguilrizer Coefficient',
        default=0,
        type=float,
        required=False)
    parser.add_argument(
        "--plot",
        "-p",
        help="Indicates if to plot the losses.",
        required=False,
        action="store_true"
    )
    parser.add_argument(
        "--save",
        "-s",
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
        choices=["Logistic Regression", "300-100", "lenet1", "cnn1", "test"],
        default="Logistic Regression"
    )
    parser.add_argument(
        "--yscale",
        help="yscale to plot",
        default="log",
        required=False,
        choices=["linear", "log"]
    )
    parser.add_argument(
        "--verbosity",
        help="How much output to print.",
        default="Normal",
        required=False,
        choices=["Silent", "Normal", "Detailed"]
    )
    parser.add_argument(
        "--test",
        help="Use a reduce data set for faster training/evaluation.",
        required=False,
        action="store_true"
    )
    return parser


class pcolors:
    """Helper function to print colored output.

    Example: print(bcolors.WARNING + "Warning" + bcolors.ENDC)
    """
    def __init__(self, palette = sns.color_palette("deep")):
        self.palette = palette
        self.train = palette[1]
        self.val = palette[3]
        self.lr = palette[5]
        self.test = palette[4]
        self.fun = palette[0]
    # plt.rcParams['text.usetex'] = True

def train_epoch(e, k, args, data, save_last_model = False):
    # Set train mode for both the encoder and the decoder
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    trainingImages, trainingLabels = data
    train_loss = []
    stepsPerEpoch = int(len(trainingImages) / args.trainingBS)
    e["Solver"]["Mode"] = "Training"
    for step in range(stepsPerEpoch):
        # Creating minibatch
        image_batch = trainingImages[step * args.trainingBS : (step+1) * args.trainingBS] # N x C
        label_batch = trainingLabels[step * args.trainingBS : (step+1) * args.trainingBS]
        # Save only is save_last_model == True and last generation of epoch
        e["File Output"]["Enabled"] = True if save_last_model and step == stepsPerEpoch-1 else False
        # Train models =======================================================
        e["Problem"]["Input"]["Data"] = image_batch
        e["Problem"]["Solution"]["Data"] = label_batch
        k.run(e)
        loss = e["Results"]["Training Loss"]
        train_loss.append(loss)
        # print(f'Step {step+1} / {stepsPerEpoch}\t Training Loss {loss}')
    return np.mean(train_loss).item()

def train_epoch_direct_grad(e, k, args, data, save_last_model = False):
    # TODO: training with direct gradient is not training atm somehow.
    trainingImages, trainingLabels = data
    loss_fu = torch.nn.CrossEntropyLoss()
    def loss_fu(y_pred, y_true):
        y_true = torch.tensor(y_true)
        y_pred_logits = torch.tensor(y_pred)
        _, y_true_label = y_true.max(dim=1)
        loss = torch.nn.CrossEntropyLoss()(y_pred_logits, y_true_label).item()
        return  loss
    def dloss_fu(y_pred_logits, y_true, useHardLabel=False):
        y_pred_logits = torch.tensor(y_pred_logits)
        y_true = torch.tensor(y_true)
        yhat = torch.nn.Softmax(dim=1)(y_pred_logits)
        return (yhat-y_true).tolist()
    e["Solver"]["Loss Function"] = "DG"
    train_loss = []
    stepsPerEpoch = int(len(trainingImages) / args.trainingBS)
    for step in range(stepsPerEpoch):
        # Creating minibatch
        image_batch = trainingImages[step * args.trainingBS : (step+1) * args.trainingBS] # N x C
        label_batch = trainingLabels[step * args.trainingBS : (step+1) * args.trainingBS]
        # Passing minibatch to Korali
        if save_last_model and step == stepsPerEpoch-1:
            e["File Output"]["Enabled"] = True
        else:
            e["File Output"]["Enabled"] = False
        # Predict new samples ================================================
        e["Solver"]["Mode"] = "Predict"
        e["Problem"]["Input"]["Data"] = image_batch
        k.run(e)
        y_pred_logits = e["Solver"]["Evaluation"]
        # Calculate the loss and gradient of the loss =======================
        loss = loss_fu(y_pred_logits, label_batch)
        dloss = dloss_fu(y_pred_logits, label_batch)
        # Train models on direct grads =======================================
        e["Solver"]["Mode"] = "Training"
        e["Problem"]["Input"]["Data"] = image_batch
        e["Problem"]["Solution"]["Data"] = dloss
        k.run(e)
        train_loss.append(loss)
        # print(f'Step {step+1} / {stepsPerEpoch}\t Training Loss {loss}')
    return np.mean(train_loss).item()

def test_epoch(e, k, args, data):
    validationImages, validationLabels = data
    e["File Output"]["Enabled"] = False
    e["Problem"]["Testing Batch Size"] = args.testingBS
    e["Solver"]["Mode"] = "Testing"
    val_loss = []
    stepsPerEpoch = int(len(validationImages) / args.testingBS)
    for step in range(stepsPerEpoch):
        # Creating minibatch
        image_batch = validationImages[step * args.testingBS : (step+1) * args.testingBS] # N x T x C
        label_batch = validationLabels[step * args.trainingBS : (step+1) * args.trainingBS]
        # Passing minibatch to Korali
        e["Problem"]["Input"]["Data"] = image_batch
        e["Problem"]["Solution"]["Data"] = label_batch
        # Evaluate loss
        k.run(e)
        # print(f'Step {step+1} / {stepsPerEpoch}\t Testing Loss {loss}')
        loss = e["Results"]["Testing Loss"]
        val_loss.append(loss)
    return np.mean(val_loss).item()
