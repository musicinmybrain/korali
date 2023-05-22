#include "_model/cylinderEnvironment.hpp"
#include "korali.hpp"

#define ranks_per_node 128
#define simulations_per_node 1

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

  int rank;
  int size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm KoraliComm;
  #if modelDIM == 3
  const int nRanks = size-1;
  MPI_Comm_split(MPI_COMM_WORLD, 0, rank, &KoraliComm);
  const int N = 1;
  #elif modelDIM == 2
  const int nRanks = ranks_per_node/simulations_per_node;
  MPI_Comm_split(MPI_COMM_WORLD, rank < nRanks-1 ? 1:0, rank, &KoraliComm);
  const int N = (size-nRanks)/nRanks;
  if (rank >= nRanks-1)
  #endif
  {
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
    korali::setKoraliMPIComm(KoraliComm);
    e["Problem"]["Custom Settings"]["Dump Frequency"] = 0.1;
    e["Problem"]["Custom Settings"]["Dump Path"] = testingResultsPath;

    for (int i = 0; i < N; i++) e["Solver"]["Testing"]["Sample Ids"][i] = i;
    k.run(e);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
}
