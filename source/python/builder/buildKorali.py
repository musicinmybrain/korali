#!/usr/bin/env python3
import os

koraliDir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + '/../..') 

from buildSolvers import *
from buildProblems import *
from buildVariables import *
from buildConduits import *
from buildDistributions import *
from buildCxx import *
from buildPlotter import *
from buildSetup import *

# Creating cxx commands.
buildCxx(koraliDir)
 
# Creating plotter main.
buildPlotter(koraliDir) 
 
# Processing Solvers
buildSolvers(koraliDir)

# Processing Problems
buildProblems(koraliDir)

# Processing Conduits
buildConduits(koraliDir)

# Processing Distributions
buildDistributions(koraliDir)

# Processing Variables
buildVariables(koraliDir) 

# Creating setup.py
buildSetup(koraliDir) 

