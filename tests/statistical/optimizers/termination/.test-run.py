#! /usr/bin/env python3
from subprocess import call

cmaes_criteria = [
  "Max Generations",
  "Max Generations",
  "Max Infeasible Resamplings",
  "Min Value Difference Threshold",
  "Min Standard Deviation",
  "Max Standard Deviation",
  "Max Condition Covariance Matrix"
  ]

cmaes_values = [
  1,     # Max Generations
  3,     # Max Generations
  1,     # Max Infeasible Resamplings
  0.1,   # Min Value Difference Threshold
  0.1,   # Min Standard Deviation
  0.9,   # Max Standard Deviation
  1.0,   # Max Condition Covariance
  ]

for c, v in zip(cmaes_criteria, cmaes_values):
  cmd = ["python3", "cmaes_termination.py", "--criterion", f'{c}', "--value", f"{v}"]
  r = call( cmd )
  if r!=0:
    exit(r)


dea_criteria = [
  "Max Generations",
  "Max Generations",
  "Max Infeasible Resamplings",
  "Min Value Difference Threshold"
  ]

dea_values = [
  1,     # Max Generations
  3,     # Max Generations
  1,     # Max Infeasible Resamplings
  0.1    # Min Value Difference Threshold
  ]

for c, v in zip(dea_criteria, dea_values):
  cmd = ["python3", "dea_termination.py", "--criterion", f'{c}', "--value", f"{v}"]
  r = call( cmd )
  if r!=0:
    exit(r)

exit(0)
