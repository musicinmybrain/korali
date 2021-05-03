// Select which environment to use
#include "_models/windmillEnvironment/windmillEnvironment.hpp"
#include "korali.hpp"

std::string _resultsPath;

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

  // Storing parameters
  _argc = argc;
  _argv = argv;

  // Getting number of workers
  int N = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &N);
  N = N - 1; // Minus one for Korali's engine

  // Initialize CUP2D
  _environment = new Simulation(_argc, _argv);
  _environment->init();

  // Set results path
  std::string trainingResultsPath = "_results_windmill_training/r_5_100/";
  std::string testingResultsPath = "_results_windmill_testing/r_5_100/";
  
  // Creating Korali experiment
  auto e = korali::Experiment();

  // Check if there is log files to continue training
  auto found = e.loadState(trainingResultsPath + std::string("/latest"));
  if (found == true) printf("[Korali] Continuing execution from previous run...\n");

  // Configuring problem
  e["Problem"]["Type"] = "Reinforcement Learning / Continuous";
  e["Problem"]["Environment Function"] = &runEnvironment;
  e["Problem"]["Training Reward Threshold"] = 8.0;
  e["Problem"]["Policy Testing Episodes"] = 5;

  // Adding custom setting to run the environment without dumping the state files during training
  e["Problem"]["Custom Settings"]["Dump Frequency"] = 0.0;
  e["Problem"]["Custom Settings"]["Dump Path"] = trainingResultsPath;

  const size_t numStates = 8;
  for (int curVariable = 0; curVariable < numStates; curVariable++)
  {
    if(curVariable%2==0){
      e["Variables"][curVariable]["Name"] = std::string("Angle ") + std::to_string(curVariable/2 + 1);
    } else{
      e["Variables"][curVariable]["Name"] = std::string("Omega ") + std::to_string(curVariable/2 + 1);
    }
    
    e["Variables"][curVariable]["Type"] = "State";
  }

  double max_torque = 1e-5;
  for(size_t j=numStates; j < numStates + 4; ++j){
    e["Variables"][j]["Name"] = "Torque " + std::to_string(j-numStates+1);
    e["Variables"][j]["Type"] = "Action";
    e["Variables"][j]["Lower Bound"] = -max_torque;
    e["Variables"][j]["Upper Bound"] = +max_torque;
    e["Variables"][j]["Initial Exploration Noise"] = 0.5;
  }

  /// Defining Agent Configuration
  e["Solver"]["Type"] = "Agent / Continuous / VRACER";
  e["Solver"]["Mode"] = "Training";
  e["Solver"]["Episodes Per Generation"] = 1;
  e["Solver"]["Agent Count"] = N;
  e["Solver"]["Experiences Between Policy Updates"] = 1;
  e["Solver"]["Learning Rate"] = 1e-4;
  e["Solver"]["Discount Factor"] = 0.95;
  e["Solver"]["Mini Batch"]["Size"] =  128;
  e["Solver"]["Policy"]["Distribution"] = "Normal";

  /// Defining the configuration of replay memory
  e["Solver"]["Experience Replay"]["Start Size"] = 1024;
  e["Solver"]["Experience Replay"]["Maximum Size"] = 65536;
  e["Solver"]["Experience Replay"]["Off Policy"]["Annealing Rate"] = 5.0e-8;
  e["Solver"]["Experience Replay"]["Off Policy"]["Cutoff Scale"] = 4.0;
  e["Solver"]["Experience Replay"]["Off Policy"]["REFER Beta"] = 0.3;
  e["Solver"]["Experience Replay"]["Off Policy"]["Target"] = 0.1;


  //// Defining Policy distribution and scaling parameters
  e["Solver"]["Policy"]["Distribution"] = "Squashed Normal";
  e["Solver"]["State Rescaling"]["Enabled"] = true;
  e["Solver"]["Reward"]["Rescaling"]["Enabled"] = false; // this was true
  e["Solver"]["Reward"]["Rescaling"]["Frequency"] = 1000;

  /// Configuring the neural network and its hidden layers
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

  e["Solver"]["Termination Criteria"]["Max Experiences"] = 1e7;

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
  k["Conduit"]["Communicator"] = MPI_COMM_WORLD;
  
  k.run(e);

  ////// Now testing policy, dumping trajectory results

  printf("[Korali] Done with training. Now running learned policy to dump the trajectory.\n");

  // Adding custom setting to run the environment dumping the state files during testing
  e["Problem"]["Custom Settings"]["Dump Frequency"] = 0.1;
  e["Problem"]["Custom Settings"]["Dump Path"] = testingResultsPath;

  e["File Output"]["Path"] = testingResultsPath;
  k["Profiling"]["Path"] = testingResultsPath + std::string("/profiling.json");
  e["Solver"]["Testing"]["Policy"] = e["Solver"]["Best Training Hyperparamters"];
  e["Solver"]["Mode"] = "Testing";
  for (int i = 0; i < N; i++) e["Solver"]["Testing"]["Sample Ids"][i] = i;

  k.run(e);
}
