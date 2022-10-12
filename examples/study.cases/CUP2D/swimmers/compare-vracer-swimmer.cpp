// Select which environment to use
#include "_model/swimmerEnvironment.hpp"
#include "korali.hpp"

// #define TESTING

int main(int argc, char *argv[])
{
  /// Initialize MPI
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  if (provided != MPI_THREAD_FUNNELED)
  {
    printf("Error initializing MPI\n");
    exit(-1);
  }

  // retreiving path to trained policy, number of task, agents, and ranks
  std::string resultsPath = argv[argc-7];
  int nAgents = atoi(argv[argc-3]);
  int nRanks  = atoi(argv[argc-1]);

  // Storing parameters for environment
  _argc = argc;
  _argv = argv;

  // Getting number of workers
  int N = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &N);
  N = N - 1; // Minus one for Korali's engine
  N = (int)(N / nRanks); // Divided by the ranks per worker

  // Setting results path
  std::string trainingResultsPath = "_trainingResults/";
  std::string testingResultsPath = "_testingResults/";

  // Create new Korali experiment
  auto e = korali::Experiment();
  e["Problem"]["Type"] = "Reinforcement Learning / Continuous";
  e["Problem"]["Environment Function"] = &runEnvironment;
  e["Problem"]["Agents Per Environment"] = nAgents;
  // e["Problem"]["Policies Per Environment"] = nAgents;

  // Dump setting for environment
  #if defined(TESTING)
  e["Problem"]["Custom Settings"]["Dump Frequency"] = 0.1;
  e["Problem"]["Custom Settings"]["Dump Path"] = testingResultsPath;

  // Random seeds to evaluate task
  for (int i = 0; i < N; i++) e["Solver"]["Testing"]["Sample Ids"][i] = 1+i;
  #else
  e["Problem"]["Custom Settings"]["Dump Frequency"] = 0;
  e["Problem"]["Custom Settings"]["Dump Path"] = trainingResultsPath;
  #endif

  // Setting up the state variables
  size_t numStates = 10;
  #if defined(STEFANS_SENSORS_STATE)
  numStates = 16;
  #endif

  size_t curVariable = 0;
  for (; curVariable < numStates; curVariable++)
  {
    e["Variables"][curVariable]["Name"] = std::string("State") + std::to_string(curVariable);
    e["Variables"][curVariable]["Type"] = "State";
  }

  e["Variables"][curVariable]["Name"] = "Curvature";
  e["Variables"][curVariable]["Type"] = "Action";
  e["Variables"][curVariable]["Lower Bound"] = -1.0;
  e["Variables"][curVariable]["Upper Bound"] = +1.0;
  e["Variables"][curVariable]["Initial Exploration Noise"] = 0.50;

  curVariable++;
  e["Variables"][curVariable]["Name"] = "Swimming Period";
  e["Variables"][curVariable]["Type"] = "Action";
  e["Variables"][curVariable]["Lower Bound"] = -0.25;
  e["Variables"][curVariable]["Upper Bound"] = +0.25;
  e["Variables"][curVariable]["Initial Exploration Noise"] = 0.50;

  /// Defining Agent Configuration
  e["Solver"]["Type"] = "Agent / Continuous / VRACER";
  // e["Solver"]["Mode"] = "Testing";
  e["Solver"]["Mode"] = "Training";
  e["Solver"]["Episodes Per Generation"] = 1;
  e["Solver"]["Concurrent Environments"] = N;
  e["Solver"]["Experiences Between Policy Updates"] = 1;
  e["Solver"]["Learning Rate"] = 1e-4;
  e["Solver"]["Discount Factor"] = 0.95;
  e["Solver"]["Mini Batch"]["Size"] =  128;

  /// Defining the configuration of replay memory
  e["Solver"]["Experience Replay"]["Start Size"] = 1024;
  e["Solver"]["Experience Replay"]["Maximum Size"] = 65536;
  e["Solver"]["Experience Replay"]["Off Policy"]["Annealing Rate"] = 5.0e-8;
  e["Solver"]["Experience Replay"]["Off Policy"]["Cutoff Scale"] = 5.0;
  e["Solver"]["Experience Replay"]["Off Policy"]["REFER Beta"] = 0.3;
  e["Solver"]["Experience Replay"]["Off Policy"]["Target"] = 0.1;

  //// Defining Policy distribution and scaling parameters
  e["Solver"]["Policy"]["Distribution"] = "Clipped Normal";
  // e["Solver"]["State Rescaling"]["Enabled"] = true;
  e["Solver"]["Reward"]["Rescaling"]["Enabled"] = true;

  //// Defining Neural Network
  e["Solver"]["Neural Network"]["Engine"] = "OneDNN";
  e["Solver"]["Neural Network"]["Optimizer"] = "Adam";

  e["Solver"]["L2 Regularization"]["Enabled"] = true;
  e["Solver"]["L2 Regularization"]["Importance"] = 1.0;

  e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear";
  e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 128;

  e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation";
  e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/Tanh";

  e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear";
  e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = 128;

  e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation";
  e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/Tanh";

  ////// Defining Termination Criteria
  e["Solver"]["Termination Criteria"]["Max Experiences"] = 5e5;

  ////// Setting Korali output configuration
  e["Console Output"]["Verbosity"] = "Normal";
  e["File Output"]["Enabled"] = true;
  e["File Output"]["Frequency"] = 1;
  e["File Output"]["Use Multiple Files"] = false;
  e["File Output"]["Path"] = trainingResultsPath;

  // Korali experiments for previous results
  auto eOld = korali::Experiment();

  // Loading existing results and transplant training policy
  auto found = eOld.loadState(resultsPath + std::string("/latest"));
  
  if( found )
  {
    printf("[Korali] Continuing execution with policy learned in previous run...\n");
    e["Solver"]["Training"]["Current Policies"]["Policy Hyperparameters"] = eOld["Solver"]["Training"]["Current Policies"]["Policy Hyperparameters"];
  }
  else
  {
    printf("[Korali] Did not find the policy learned in previous run, training from scratch...\n");
  }

  // Creating Korali engine
  auto k = korali::Engine();

  // Configuring profiler output
  k["Profiling"]["Detail"] = "Full";
  k["Profiling"]["Path"] = trainingResultsPath + std::string("/profiling.json");
  k["Profiling"]["Frequency"] = 60;

  // Configuring conduit / communicator
  k["Conduit"]["Type"] = "Distributed";
  k["Conduit"]["Ranks Per Worker"] = nRanks;
  korali::setKoraliMPIComm(MPI_COMM_WORLD);

  // ..and run
  k.run(e);
}
