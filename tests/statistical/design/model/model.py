#!/usr/bin/env python3
import os
import sys
import numpy as np

def pconstant(sample):
  x = sample["Parameters"][0]
  sample["Evaluation"] = 1.

def plinear(sample):
  x = sample["Parameters"][0]
  sample["Evaluation"] = x

def pquadratic(sample):
  x = sample["Parameters"][0]
  sample["Evaluation"] = x**2

def pcubic(sample):
  x = sample["Parameters"][0]
  sample["Evaluation"] = x**3


