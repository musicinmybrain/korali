#! /usr/bin/env python3
from subprocess import call

r = call(["python3", "-m", "korali.cxx"])
if r==0:
  exit(1)

r = call(["python3", "-m", "korali.cxx", "--cflags"])
if r!=0:
  exit(r)

r = call(["python3", "-m", "korali.cxx", "--libs"])
if r!=0:
  exit(r)

r = call(["python3", "-m", "korali.cxx", "--help"])
if r!=0:
  exit(r)

exit(0)
