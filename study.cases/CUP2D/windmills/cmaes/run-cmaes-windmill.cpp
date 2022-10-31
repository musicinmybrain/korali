// Select which environment to use
#include "../_model/windmillEnvironment.hpp"
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

    // retrieving task reward
    int task_reward = atoi(argv[argc-1]);

    // retrieving task a1
    double task_a1 = atof(argv[argc-3]);

    // retrieving task a2
    double task_a2 = atof(argv[argc-5]);

    // retrieving task f1
    double task_f1 = atof(argv[argc-7]);

    // retrieving task f2
    double task_f2 = atof(argv[argc-9]);

    // retrieving mu value
    int task_mu = atoi(argv[argc-11]);

    // retrieving population size
    int task_population = atoi(argv[argc-13]);

    // Storing parameters
    _argc = argc;
    _argv = argv;

    // Set results path
    std::string ResultsPath = "_results/";

    // Creating Korali experiment
    auto e = korali::Experiment();

    // Check if there is log files to continue training
    auto found = e.loadState(ResultsPath + std::string("/latest"));
    if (found == true) printf("[Korali] Continuing execution from previous run...\n");

    // Configuring problem
    e["Random Seed"] = 0xC0FEE;
    e["Problem"]["Type"] = "Optimization";
    e["Problem"]["Objective Function"] = &runEnvironmentCMAES;

    // Configuring CMA-ES parameters
    e["Solver"]["Type"] = "Optimizer/CMAES";
    e["Solver"]["Population Size"] = task_population; // number of different simulations
    e["Solver"]["Mu Value"] = task_mu; // number of selected search points, for updating mean, etc
    e["Solver"]["Termination Criteria"]["Min Value Difference Threshold"] = 1e-16;
    e["Solver"]["Termination Criteria"]["Max Generations"] = 150;

    // Setting up the variables
    e["Variables"][0]["Name"] = "a1"; // max angular velocity of first fan
    e["Variables"][0]["Initial Value"] = task_a1;
    e["Variables"][0]["Initial Standard Deviation"] = 0.5;

    e["Variables"][1]["Name"] = "a2"; // max angular velocity of second fan
    e["Variables"][1]["Initial Value"] = task_a2;
    e["Variables"][1]["Initial Standard Deviation"] = 0.5;

    e["Variables"][2]["Name"] = "k1"; // angular frequency of first fan
    e["Variables"][2]["Initial Value"] = task_f1;
    e["Variables"][2]["Initial Standard Deviation"] = 0.5;
    
    e["Variables"][3]["Name"] = "k2"; // angular frequency of second fan
    e["Variables"][3]["Initial Value"] = task_f2;
    e["Variables"][3]["Initial Standard Deviation"] = 0.5;

    ////// Setting Korali output configuration
    e["Console Output"]["Verbosity"] = "Detailed";
    e["File Output"]["Enabled"] = true;
    e["File Output"]["Frequency"] = 1;
    e["File Output"]["Path"] = ResultsPath;

    ////// Running Experiment
    auto k = korali::Engine();

    k["Conduit"]["Type"] = "Distributed";
    k["Conduit"]["Ranks Per Worker"] = 1;   // we give only one rank per worker bc GPU is used for fluid simulation.
    korali::setKoraliMPIComm(MPI_COMM_WORLD);
    k.run(e);
}