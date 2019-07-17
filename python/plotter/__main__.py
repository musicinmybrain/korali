#! /usr/bin/env python3
import os
import sys
import json
import argparse
import matplotlib
from korali.plotter.cmaes import plot_cmaes
from korali.plotter.ccmaes import plot_ccmaes
from korali.plotter.tmcmc import plot_tmcmc
from korali.plotter.mcmc import plot_mcmc
from korali.plotter.dea import plot_dea

def main(check, path, live, test, evolution):

 if (check == True):
  print("[Korali] Plotter correctly installed.")
  exit(0)
 
 if (test == True):
     matplotlib.use('Agg')

 firstResult = path + '/s00000.json'
 if ( not os.path.isfile(firstResult) ):
  print("[Korali] Error: Did not find any results in the {0} folder...".format(path))
  exit(-1)

 with open(firstResult) as f:
  data  = json.load(f)
 
 solver = data['Solver']
 if ( 'TMCMC' == solver ):
  print("[Korali] Running TMCMC Plotter...")
  plot_tmcmc(path, live, test)
  exit(0)
 
 if ( 'MCMC' == solver ):
  print("[Korali] Running MCMC Plotter...")
  plot_mcmc(path, live, test)
  exit(0)

 if ( 'CMAES' == solver):
  print("[Korali] Running CMAES Plotter...")
  plot_cmaes(path, live, test, evolution)
  exit(0)
  
 if ( 'CCMAES' == solver):
  print("[Korali] Running CcMAES Plotter...")
  plot_ccmaes(path, live, test, evolution)
  exit(0)

 if ( 'DEA' == solver ):
  print("[Korali] Running DEA Plotter...")
  plot_dea(path, live, test)
  exit(0)

 print("[Korali] Error: Did not recognize method for plotting...")
 exit(-1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='korali.plotter', description='Process korali results in _korali_result (default) folder.')
    parser.add_argument('--check', help='verifies that korali.plotter is available', action='store_true', required = False)
    parser.add_argument('--dir', help='directory of result files', default='_korali_result', required = False)
    parser.add_argument('--live', help='run live plotting', action='store_true', required = False)
    parser.add_argument('--test', help='run without graphics', action='store_true', required = False)
    parser.add_argument('--evolution', help='plot CMA-ES evolution (only in 2D)', action='store_true', required = False)
    args = parser.parse_args()
    
    main(args.check, args.dir, args.live, args.test, args.evolution)
