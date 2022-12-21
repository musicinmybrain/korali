#! /usr/bin/env python3
from subprocess import call
import os

r = call(["python3", "run-vracer.py", "--distribution", "Normal", "--maxExperiences", "8000"])
if r!=0:
  exit(r)

r = call(["python3", "run-vracer.py", "--distribution", "Squashed Normal", "--maxExperiences", "8000"])
if r!=0:
  exit(r)

r = call(["python3", "run-vracer.py", "--distribution", "Clipped Normal", "--maxExperiences", "8000"])
if r!=0:
  exit(r)

r = call(["python3", "run-vracer.py", "--distribution", "Truncated Normal", "--maxExperiences", "8000"])
if r!=0:
  exit(r)

exit(0)
