#! /usr/bin/env python3
from subprocess import call
import os

r = call(["python3", "run-ffn.py", "--optimizer",  "AdaBelief", "--learningRate", "0.005", "--maxGenerations", "30"])
if r!=0:
  exit(r)

r = call(["python3", "run-ffn.py", "--optimizer",  "AdaGrad", "--learningRate", "0.005", "--maxGenerations", "30"])
if r!=0:
  exit(r)

r = call(["python3", "run-ffn.py", "--optimizer",  "Adam", "--learningRate", "0.005", "--maxGenerations", "30"])
if r!=0:
  exit(r)

r = call(["python3", "run-ffn.py", "--optimizer",  "MadGrad", "--learningRate", "0.005", "--maxGenerations", "30"])
if r!=0:
  exit(r)

r = call(["python3", "run-ffn.py", "--optimizer",  "Adam", "--learningRate", "0.005", "--maxGenerations", "30"])
if r!=0:
  exit(r)

r = call(["python3", "run-ffn.py", "--optimizer",  "Adam", "--engine", "Korali", "--learningRate", "0.005", "--maxGenerations", "30"])
if r!=0:
  exit(r)

r = call(["python3", "plot-loss.py"])
if r!=0:
  exit(r)

exit(0)