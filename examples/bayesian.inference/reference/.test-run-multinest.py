#! /usr/bin/env python3
from subprocess import call
from korali.plotter.__main__ import main

r = call(["python3", "run-multinest.py"])
if r!=0:
  exit(r)

r = main( path='_korali_result_multinest', check=False, test=True, output="", plotAll=False)
if r!=0: exit(r)

exit(0)