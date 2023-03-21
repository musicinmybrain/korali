#include "_model/cylinderEnvironment.hpp"
#include "korali.hpp"

int main(int argc, char *argv[])
{
  // Initialize MPI
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  if (provided != MPI_THREAD_FUNNELED)
  {
    printf("Error initializing MPI\n");
    exit(-1);
  }

  // retreiving number of ranks
  const int nRanks  = atoi(argv[argc-1]);

  // Storing parameters for environment
  _argc = argc;
  _argv = argv;

  // Getting number of workers
  int N = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &N);
  N = N - 1; // Minus one for Korali's engine
  N = (int)(N / nRanks); // Divided by the ranks per worker

  // Setting results path
  std::string trainingResultsPath = "_optimizationResults/";

  // Creating Experiment
  auto e = korali::Experiment();

  // Configuring Experiment
  e["Problem"]["Type"] = "Optimization";
  e["Problem"]["Objective Function"] = &runEnvironment;
  e["Problem"]["Custom Settings"]["Dump Frequency"] = 0.0;
  e["Problem"]["Custom Settings"]["Dump Path"] = trainingResultsPath;
  e["Problem"]["Custom Settings"]["Mode"] = "Training";

  // Configuring CMA-ES parameters
  e["Solver"]["Type"] = "Optimizer/CMAES";
  e["Solver"]["Population Size"] = 32;
  e["Solver"]["Mu Value"] = 16;
  e["Solver"]["Termination Criteria"]["Min Value Difference Threshold"] = 1e-16;
  e["Solver"]["Termination Criteria"]["Max Generations"] = 500;

  // Setting up the variables
  const size_t numParams = 16;
  const double q = 1.0; //action bounds
  for (size_t idx = 0 ; idx < numParams; idx ++)
  {
    e["Variables"][idx]["Name"] = "Nozzle" + std::to_string(idx);
    e["Variables"][idx]["Type"] = "Action";
    e["Variables"][idx]["Lower Bound"] = -q;
    e["Variables"][idx]["Upper Bound"] = +q;
  }

  ////// Setting Korali output configuration
  e["Console Output"]["Verbosity"] = "Normal";
  e["File Output"]["Enabled"] = true;
  e["File Output"]["Frequency"] = 1;
  e["File Output"]["Path"] = trainingResultsPath;

  ////// Running Experiment
  auto k = korali::Engine();

  // Configuring conduit / communicator
  k["Conduit"]["Type"] = "Distributed";
  k["Conduit"]["Ranks Per Worker"] = nRanks;
  korali::setKoraliMPIComm(MPI_COMM_WORLD);

  // ..and run
  k.run(e);
}
