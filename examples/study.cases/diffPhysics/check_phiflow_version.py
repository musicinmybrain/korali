# Use this script to check your PhiFlow version

# Important note:
#   The versions in this project require the latest versions (tested on version 2.1.4)
#   Other examples might require the version 1.5.1

# The code in this project is based on the Physics-based Deep Learning Book (v0.2)
# https://www.physicsbaseddeeplearning.org/intro.html

# Use the latest PhiFlow version to run this code:
# !pip install --upgrade --quiet phiflow

# --- Checking and printing version of PhiFlow ---

print("Checking your PhiFlow version...")

from phi.flow import *
from phi import __version__
print("You're using PhiFlow version: {}".format(phi.__version__))
if phi.__version__ == "2.1.4":
    print("The same version has been used to create and test the code\n")
else:
    print("Note: The code has been tested on another version: version 2.1.4\n")

# --- Checking and printing version of TensorFlow ---

print("Checking your TensorFlow version...")

from phi.tf.flow import *
print("You're using TensorFlow version: {}".format(tf.__version__))
if tf.__version__ == "2.9.1":
    print("The same version has been used to create and test the code\n")
else:
    print("Note: The code has been tested on another version: version 2.9.1\n")

# --- Checking and printing version of h5py used in DL ---

print("Checking your h5py version...")

import h5py
print("You're using h5py version: {}".format(h5py.__version__))
if h5py.__version__ == "2.6.0":
    print("The same version has been used to create and test the code\n")
else:
    print("Note: The code has been tested on another version: version 2.9.1\n")



# These lines will output the version too and perform other checks
#import phi
#phi.verify()
