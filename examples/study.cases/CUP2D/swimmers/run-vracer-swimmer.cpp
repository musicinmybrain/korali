// Select which environment to use
#include "_model/swimmerEnvironment.hpp"
#include "korali.hpp"

int main(int argc, char *argv[])
{
  // Gathering actual arguments from MPI
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  if (provided != MPI_THREAD_FUNNELED)
  {
    printf("Error initializing MPI\n");
    exit(-1);
  }

  // Storing command-line arguments
  _argc = argc;
  _argv = argv;

  // Getting number of workers
  int N = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &N);
  N = N - 1; // Minus one for Korali's engine

  // Setting results path
  std::string trainingResultsPath = "_trainingResults/";
  std::string testingResultsPath = "_testingResults/";

  // Creating Experiment
  auto e = korali::Experiment();
  e["Problem"]["Type"] = "Reinforcement Learning / Continuous";

  // Check if existing results are there and continuing them
  auto found = e.loadState(trainingResultsPath + std::string("/latest"));
  if (found == true){
    printf("[Korali] Continuing execution from previous run...\n");
    e["Solver"]["Termination Criteria"]["Max Generations"] = e["Current Generation"].get<int>() + 10000;
  }

  // Configuring Experiment
  e["Problem"]["Environment Function"] = &runEnvironment;
  e["Problem"]["Training Reward Threshold"] = 100.0;
  e["Problem"]["Policy Testing Episodes"] = 5;
  // e["Problem"]["Actions Between Policy Updates"] = 1;

  // Setting results path an dumping frequency in CUP
  e["Problem"]["Custom Settings"]["Dump Frequency"] = 0.0;
  e["Problem"]["Custom Settings"]["Dump Path"] = trainingResultsPath;

  // Setting up the state variables
  size_t curVariable = 0;
  for (; curVariable < 16; curVariable++)
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
  e["Solver"]["Policy"]["Distribution"] = "Squashed Normal";
  e["Solver"]["State Rescaling"]["Enabled"] = true;
  e["Solver"]["Reward"]["Rescaling"]["Enabled"] = true;
  e["Solver"]["Reward"]["Rescaling"]["Frequency"] = 1000;

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
  e["Solver"]["Termination Criteria"]["Max Experiences"] = 1e6;

  ////// Setting Korali output configuration
  e["Console Output"]["Verbosity"] = "Detailed";
  e["File Output"]["Enabled"] = true;
  e["File Output"]["Frequency"] = 1;
  e["File Output"]["Path"] = trainingResultsPath;

  ////// Running Experiment
  auto k = korali::Engine();

  // Configuring profiler output
  k["Profiling"]["Detail"] = "Full";
  k["Profiling"]["Path"] = trainingResultsPath + std::string("/profiling.json");
  k["Profiling"]["Frequency"] = 60;

  k["Conduit"]["Type"] = "Distributed";
  korali::setKoraliMPIComm(MPI_COMM_WORLD);

  k.run(e);
}
