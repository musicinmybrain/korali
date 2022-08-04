#! /usr/bin/env python3
import os
import sys
import signal
import json
import argparse
import matplotlib
import importlib

curdir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

# Check if name has .png ending
def validateOutput(output):
  if not (output.endswith(".png") or output.endswith(".eps") or output.endswith(".svg") or "None"):
    print("[Korali] Error: Outputfile '{0}' must end with '.eps', '.png' or '.svg' suffix.".format(output))
    sys.exit(-1)

def get_generation_dicts(gen_result_files, configRunId):
  """Stores the json objects of the generations files files inside a dict
     and deletes the first generation that is saved before the first run().
  :param gen_result_files: file paths to the generation files
  :param configRunId: experiment id
  :returns: dicts of the generation files.

  """
  genList = {}
  for file in gen_result_files:
    with open(file) as f:
      genJs = json.load(f)
      solverRunId = genJs['Run ID']
      if (configRunId == solverRunId):
        curGen = genJs['Current Generation']
        genList[curGen] = genJs

  del genList[0]
  return genList



def main(exec_path, test, output, plotAll=False, others = None):

  if test or output and output != "None":
    matplotlib.use('Agg')

  if output:
    validateOutput(output)

  # This import has to be after matplotlib.use('Agg').
  import matplotlib.pyplot as plt

  signal.signal(signal.SIGINT, lambda x, y: exit(0))

  configFile = exec_path + '/gen00000000.json'
  if (not os.path.isfile(configFile)):
    print("[Korali] Error: Did not find any results in the {0} folder...".format(exec_path))
    exit(-1)

  with open(configFile) as f:
    config = json.load(f)
  configRunId = config['Run ID']

  resultFiles = [
      os.path.join(exec_path, f) for f in os.listdir(exec_path)
      if os.path.isfile(os.path.join(exec_path, f)) and f.startswith('gen')
  ]
  resultFiles = sorted(resultFiles)

  genList = get_generation_dicts(resultFiles, configRunId)

  solverName = config['Solver']['Type'].lower()
  solverDir = ""
  moduleName = ""

  if ("cmaes" in solverName):
   solverDir = curdir + '/HMC'
   moduleName = '.HMC'
     
  if ("cmaes" in solverName):
   solverDir = curdir + '/CMAES'
   moduleName = '.CMAES'

  if ("dea" in solverName):
   solverDir = curdir + '/DEA'
   moduleName = '.DEA'

  if ("lmcmaes" in solverName):
   solverDir = curdir + '/LMCMAES'
   moduleName = '.LMCMAES'

  if ("mocmaes" in solverName):
   solverDir = curdir + '/MOCMAES'
   moduleName = '.MOCMAES'

  if ("mcmc" in solverName):
   solverDir = curdir + '/MCMC'
   moduleName = '.MCMC'

  if ("nested" in solverName):
   solverDir = curdir + '/Nested'
   moduleName = '.Nested'

  if ("tmcmc" in solverName):
   solverDir = curdir + '/TMCMC'
   moduleName = '.TMCMC'

  if ("deepsupervisor" in solverName):
   solverDir = curdir + '/DEEPSUPERVISOR'
   moduleName = '.deepSupervisor'

  if (solverDir == ""):
   print("[Korali] Solver '{0}' does not provide support for plotting.".format(solverName))
   exit(0)

  sys.path.append(solverDir)
  solverLib = importlib.import_module(moduleName, package="plot")
  if ("deepsupervisor" in solverName):
    solverLib.plot(genList, config, others)
  else:
    solverLib.plot(genList, plotAll=plotAll)

  if not output:
    plt.show()
  elif output == "None":
    pass
  else:
      if output.endswith('.eps'):
        plt.savefig(output, format='eps')
      elif output.endswith('.svg'):
        plt.savefig(output, format='svg')
      else:
        plt.savefig(output, format='png')
      exit(0)

  return 0

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      prog='korali.plot',
      description='Plot the results of a Korali execution.')
  parser.add_argument(
      '--dir',
      help='directory of result files',
      default='_korali_result',
      required=False)
  parser.add_argument(
      '--test',
      help='run without graphics (for testing purpose)',
      action='store_true',
      required=False)
  parser.add_argument(
      '--output', help='save figure to file', type=str, default="")
  parser.add_argument(
      '--all', help='plot all generations', action='store_true', required=False)
  args, unknown = parser.parse_known_args()

  main(args.dir, args.test, args.output, args.all, unknown)
