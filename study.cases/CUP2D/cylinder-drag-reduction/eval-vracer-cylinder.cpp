// Select which environment to use
#include "_model/cylinderEnvironment.hpp"
#include "korali.hpp"

int main(int argc, char *argv[])
{
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  if (provided != MPI_THREAD_FUNNELED)
  {
    printf("Error initializing MPI\n");
    exit(-1);
  }
  _argc = argc;
  _argv = argv;

  int nRanks  = atoi(argv[argc-1]);

  int N = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &N);
  N = N - 1; // Minus one for Korali's engine
  N = (int)(N / nRanks); // Divided by the ranks per worker

  std::string trainingResultsPath = "_trainingResults/";
  std::string testingResultsPath  = "_testingResults-" +std::to_string(modelDIM)+"D/";

  auto e = korali::Experiment();

  // Check if there is log files to continue training
  auto found = e.loadState(trainingResultsPath+"/latest");
  if (found == true) printf("[Korali] Evaluation results found...\n");
  else { fprintf(stderr, "[Korali] Error: cannot find previous results\n"); exit(0); } 

  auto k = korali::Engine();

  e["Problem"]["Environment Function"] = &runEnvironment;
  e["File Output"]["Path"] = trainingResultsPath;
  e["Solver"]["Mode"] = "Testing";
  k["Conduit"]["Type"] = "Distributed";
  k["Conduit"]["Ranks Per Worker"] = nRanks;
  korali::setKoraliMPIComm(MPI_COMM_WORLD);
  e["Problem"]["Custom Settings"]["Dump Frequency"] = 0.1;
  e["Problem"]["Custom Settings"]["Dump Path"] = testingResultsPath;

  // Use random seed to communicate the task that we are to evaluate
  for (int i = 0; i < N; i++) e["Solver"]["Testing"]["Sample Ids"][i] = atoi(argv[argc-7]) + i*10;

  k.run(e);
}
