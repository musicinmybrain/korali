#!/usr/bin/env python
# coding: utf-8
import sys
import random
import os
import matplotlib.pyplot as plt # plotting library
import numpy as np # this module is useful to work with numerical arrays
import json
from korali.auxiliar.printing import *
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,random_split
from torch import _pin_memory, nn
import torch.optim as optim
sys.path.append(os.path.abspath('./_models'))
sys.path.append(os.path.abspath('..'))
# from cnn import Encoder, Decoder
from linear_autoencoder import Encoder, Decoder
from utilities import make_parser
import time
import argparse
data_dir = '_data/torch/'
#  Arguments =================================================================
parser = make_parser()
# If interactive IPython =====================================================
tmp = sys.argv
if len(sys.argv) != 0:
    if sys.argv[0] in ["/usr/bin/ipython", "/users/pollakg/.local/bin/ipython"]:
        sys.argv = ['']
        IPYTHON = True
# ============================================================================
args = parser.parse_args()
sys.argv = tmp
print_header('Pytorch', color=bcolors.HEADER, width=140)
#  Decide on the device where to train on ====================================
# Check if the GPU is available
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")
args.device = "CPU"
print_args(vars(args), sep=' ', header_width=140)
torch.manual_seed(args.seed)
#  Parameters ================================================================
#  Download Data =============================================================
train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True)
test_dataset  = torchvision.datasets.MNIST(data_dir, train=False, download=True)
# Transforms [0, 255] to [0, 1]
train_transform = transforms.Compose([transforms.ToTensor()])
test_transform = transforms.Compose([transforms.ToTensor()])
train_dataset.transform = train_transform
test_dataset.transform = test_transform
nb_training_samples = len(train_dataset)
#  Train and Validation Split ================================================
#train (55,000 images), val split (5,000 images)
train_data, val_data = random_split(train_dataset, [int(nb_training_samples-nb_training_samples*args.validationSplit), int(nb_training_samples*args.validationSplit)])
# The dataloaders handle shuffling, batching, etc...
loader_args = {
    "num_workers": 1,
    "pin_memory": False,
    "shuffle": args.shuffle
}
train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.trainingBS, **loader_args)
valid_loader = torch.utils.data.DataLoader(val_data, batch_size=args.trainingBS, **loader_args)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.testingBS, **loader_args)
#  Set the Models ============================================================
encoder = Encoder(encoded_space_dim=args.latentDim)
decoder = Decoder(encoded_space_dim=args.latentDim)
#  Set the optimzier =========================================================
loss_fn = torch.nn.MSELoss()
params_to_optimize = [
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
]
optim = torch.optim.Adam(params_to_optimize, lr=args.learningRate, weight_decay=args.regularizerCoefficient)
# Move both the encoder and the decoder to the selected device
encoder.to(device)
decoder.to(device)
#  Define the training and testing loop ======================================
def train_epoch(encoder, decoder, device, dataloader, loss_fn, optimizer):
    # Set train mode for both the encoder and the decoder
    encoder.train()
    decoder.train()
    train_loss = []
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for image_batch, _ in dataloader: # with "_" we just ignore the labels (the second element of the dataloader tuple)
        # Move tensor to the proper device
        image_batch = image_batch.to(device)
        # Encode data
        encoded_data = encoder(image_batch)
        # Decode data
        decoded_data = decoder(encoded_data)
        # Evaluate loss
        loss = loss_fn(decoded_data, image_batch)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.detach().cpu().numpy())
    return np.mean(train_loss).item()

### Testing function
def test_epoch(encoder, decoder, device, dataloader, loss_fn):
    # Set evaluation mode for encoder and decoder
    encoder.eval()
    decoder.eval()
    with torch.no_grad(): # No need to track the gradients
        # Define the lists to store the outputs for each batch
        conc_out = []
        conc_label = []
        for image_batch, _ in dataloader:
            # Move tensor to the proper device
            image_batch = image_batch.to(device)
            # Encode data
            encoded_data = encoder(image_batch)
            # Decode data
            decoded_data = decoder(encoded_data)
            # Append the network output and the original image to the lists
            conc_out.append(decoded_data.cpu())
            conc_label.append(image_batch.cpu())
        # Create a single tensor with all the values in the lists
        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label) 
        # Evaluate global loss
        val_loss = loss_fn(conc_out, conc_label)
    return val_loss.item()
# Training Loop ==============================================================
# Initial Error
history={'train_loss':[],'val_loss':[], 'time_per_epoch': []}
loss = test_epoch(encoder,decoder,device,test_loader,loss_fn)
print(f'EPOCH {0}/{args.epochs:<8} \t test loss {loss:.3f}')
for epoch in range(args.epochs):
   tp_start = time.time()
   train_loss = train_epoch(encoder,decoder,device,train_loader,loss_fn,optim)
   tp = time.time()-tp_start
   val_loss = test_epoch(encoder,decoder,device,valid_loader,loss_fn)
   print(f'EPOCH {epoch + 1}/{args.epochs} {tp:.3f}s \t train loss {train_loss:.3f} \t val loss {val_loss:.3f}')
   history['train_loss'].append(train_loss)
   history['val_loss'].append(val_loss)
   history['time_per_epoch'].append(tp)

if args.save:
    with open('_results/latest.json', 'w') as file:
        file.write(json.dumps(history))
if args.plot:
    plt.figure(figsize=(10,8))
    plt.semilogy(history['train_loss'], label='Train')
    plt.semilogy(history['val_loss'], label='Valid')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.legend()
#  Plot Images ===============================================================
    SAMPLES_TO_DISPLAY = 6
    fig, axes = plt.subplots(nrows=SAMPLES_TO_DISPLAY, ncols=2)
    for ax in axes:
        img = random.choice(test_dataset)[0].unsqueeze(0).to(device)
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            rec_img  = decoder(encoder(img))
        ax[0].imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
        ax[1].imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')
    plt.show()
print_header(width=140)
