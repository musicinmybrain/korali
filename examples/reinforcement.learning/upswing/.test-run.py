#! /usr/bin/env python3
from subprocess import call
import os

r = call(["python3", "run-vracer-single.py", "--test"])
if r!=0:
  exit(r)

r = call(["python3", "run-vracer-double.py", "--test"])
if r!=0:
  exit(r)

exit(0)
