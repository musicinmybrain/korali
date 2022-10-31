#! /usr/bin/env python3
from subprocess import call

r = call(["python3", "run-sir-ssa.py"])
if r!=0:
  exit(r)
  
r = call(["python3", "run-sir-tauLeaping.py"])
if r!=0:
  exit(r)

r = call(["python3", "run-brusselator-ssa.py"])
if r!=0:
  exit(r)

r = call(["python3", "run-laczlacy-tauLeaping.py"])
if r!=0:
  exit(r)

exit(0)
