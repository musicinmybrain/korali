#include "swimmerEnvironment.hpp"

void runEnvironment(korali::Sample &s)
{
  // Get MPI subcommunicator, rank and size
  MPI_Comm comm = *(MPI_Comm*) korali::getWorkerMPIComm();
  int rank, size;
  MPI_Comm_rank(comm,&rank);
  MPI_Comm_size(comm,&size);
  int rankGlobal;
  MPI_Comm_rank(MPI_COMM_WORLD,&rankGlobal);

  // Setting seed
  const size_t sampleId = s["Sample Id"];
  _randomGenerator.seed(sampleId);

  // Create results directory and redirect all output to log file
  char resDir[64];
  FILE * logFile = nullptr;
  sprintf(resDir, "%s/sample%03u", s["Custom Settings"]["Dump Path"].get<std::string>().c_str(), rankGlobal/size);
  if( rank == 0 )
  {
    if( not std::filesystem::exists(resDir) )
    if( not std::filesystem::create_directories(resDir) )
    {
      fprintf(stderr, "[Korali] Error creating results directory for environment: %s.\n", resDir);
      exit(-1);
    };
    char logFilePath[128];
    sprintf(logFilePath, "%s/log.txt", resDir);
    logFile = freopen(logFilePath, "a", stdout);
    if (logFile == NULL)
    {
      printf("[Korali] Error creating log file: %s.\n", logFilePath);
      exit(-1);
    }
  }

  // Switch to results directory
  MPI_Barrier(comm); // Make sure logfile is created before switching path
  auto curPath = std::filesystem::current_path();
  std::filesystem::current_path(resDir);

  // Initialize environment and obtain agents
  Simulation *_environment = initializeEnvironment(s);
  auto & shapes = _environment->getShapes();
  std::vector<StefanFish *> agents(AGENTS);
  for(int i = 0; i<AGENTS; i++ )
    agents[i] = dynamic_cast<StefanFish *>(shapes[i].get());

  // Setting initial state [Careful, state function needs to be called by all ranks!]
  {
    std::vector<std::vector<double>> states(AGENTS);
    for(int i = 0; i<AGENTS; i++ ) states[i]  = getState(agents[i],_environment->sim,i);
    if (AGENTS > 1)  s["State"] = states;
    else             s["State"] = states[0];
  }

  // Variables for time and step conditions
  double t        = 0; // Current time
  size_t curStep  = 0; // Current Step
  double dtAct;        // Time until next action
  double tNextAct = 0; // Time of next action     
  const size_t maxSteps = 50; // Max steps before truncation

  std::vector<std::vector<double>> actions(AGENTS, std::vector<double>(ACTIONS));

  // Main environment loop
  bool done = false;
  while ( curStep < maxSteps && done == false )
  {
    if( rank == 0 ) // Get new actions, then broadcast and apply them
    {
      s.update();
      for(int i = 0; i<AGENTS; i++ )
        actions[i] = (AGENTS > 1) ? s["Action"][i].get<std::vector<double>>() : s["Action"].get<std::vector<double>>();
    }
    for(int i = 0; i<AGENTS; i++ )
    {
      MPI_Bcast( actions[i].data(), ACTIONS, MPI_DOUBLE, 0, comm );
      takeAction(agents[i],_environment->sim,i,actions[i],t);
    }

    if (rank == 0) //Write a file with the actions for every agent
    {
      for( int i = 0; i<AGENTS; i++ )
      {
        ofstream myfile;
        myfile.open ("actions"+std::to_string(i)+".txt",ios::app);
        myfile << t << " ";
          for (int j = 0; j < ACTIONS ; j++)
              myfile << actions[i][j] << " ";
          myfile << std::endl;
        myfile.close();
      }
    }

    // Run the simulation until next action is required
    dtAct = 0.;
    for(int i = 0; i<AGENTS; i++ )
    if( dtAct < agents[i]->getLearnTPeriod() * 0.5 )
      dtAct = agents[i]->getLearnTPeriod() * 0.5;
    tNextAct += dtAct;
    while ( t < tNextAct && done == false )
    {
      const double dt = std::min(_environment->calcMaxTimestep(), dtAct);
      t += dt;
      _environment->advance(dt);

      done = _environment->sim.bCollision; // if collision -> terminate

      // Check termination because leaving margins
      for(int i = 0; i<AGENTS; i++ )
        done = ( done || isTerminal(agents[i],_environment->sim,i) );
    }

    // Get and store state and reward 
    // [Careful, state function needs to be called by all ranks!] 
    {
      std::vector<std::vector<double>> states(AGENTS);
      std::vector<double> rewards(AGENTS);
      for(int i = 0; i<AGENTS; i++ )
      {
        states[i]  = getState (agents[i],_environment->sim,i);
        rewards[i] = getReward(agents[i],_environment->sim,i);
      }
      if (AGENTS > 1) { s["State"] = states   ; s["Reward"] = rewards   ;}
      else             { s["State"] = states[0]; s["Reward"] = rewards[0];}
    }

    // Print information
    if ( rank == 0 )
    {
      printf("[Korali] -------------------------------------------------------\n");
      printf("[Korali] Sample %lu - Step: %lu/%lu\n", sampleId, curStep, maxSteps);
      for(int i = 0; i<AGENTS; i++ )
      {
        auto state  = (AGENTS > 1) ? s["State"][i].get<std::vector<float>>():s["State"].get<std::vector<float>>();
        auto action = (AGENTS > 1) ? s["Action"][i].get<std::vector<float>>():s["Action"].get<std::vector<float>>();
        auto reward = (AGENTS > 1) ? s["Reward"][i].get<float>():s["Reward"].get<float>();
        printf("[Korali] AGENT %d/%d\n", i, AGENTS);
        printf("[Korali] State: [ %.3f", state[0]);
        for (size_t j = 1; j < state.size(); j++) printf(", %.3f", state[j]);
        printf("]\n");
        printf("[Korali] Action: [ %.3f, %.3f ]\n", action[0], action[1]);
        printf("[Korali] Reward: %.3f\n", reward);
        printf("[Korali] Terminal?: %d\n", done);
        printf("[Korali] -------------------------------------------------------\n");
      }
    }
    fflush(stdout);
    curStep++;// Advance to next step
  }

  // Setting termination status
  s["Termination"] = done ? "Terminal" : "Truncated";

  logger.flush();// Flush CUP logger

  delete _environment;// delete simulation class

  if( rank == 0 ) // Closing log file
    fclose(logFile);

  std::filesystem::current_path(curPath); // Switching back to experiment directory
}

int main(int argc, char *argv[])
{
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  if (provided != MPI_THREAD_FUNNELED)
  {
    printf("Error initializing MPI\n");
    exit(-1);
  }
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);


  // Getting number of workers
  const int nRanks = atoi(argv[argc-1]);
  int N = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &N);
  N = N - 1; // Minus one for Korali's engine
  N = (int)(N / nRanks); // Divided by the ranks per worker

  #ifdef EVALUATION
    // Setting results path
    std::string trainingResultsPath = "_trainingResults-"+std::to_string(modelDIM)+"D/";
    std::string testingResultsPath  = "_testingResults-" +std::to_string(modelDIM)+"D/";
  
    // Creating Korali experiment and engine
    auto e = korali::Experiment();
    auto k = korali::Engine();
  
    // Find log files from run that will be evaluated
    auto found = e.loadState(trainingResultsPath+"/latest");
    if (found == true) printf("[Korali] Evaluation results found...\n");
    else { fprintf(stderr, "[Korali] Error: cannot find previous results\n"); exit(0); } 
  
    // Configure Korali
    e["Problem"]["Environment Function"] = &runEnvironment;
    e["File Output"]["Path"] = trainingResultsPath;
    e["Solver"]["Mode"] = "Testing";
  
    // Configuring conduit / communicator
    k["Conduit"]["Type"] = "Distributed";
    k["Conduit"]["Ranks Per Worker"] = nRanks;
    korali::setKoraliMPIComm(MPI_COMM_WORLD);
  
    // Dump setting for environment
    e["Problem"]["Custom Settings"]["Dump Frequency"] = 0.1;
    e["Problem"]["Custom Settings"]["Dump Path"] = testingResultsPath;
  
    // random seeds for environment
    for (int i = 0; i < N; i++) e["Solver"]["Testing"]["Sample Ids"][i] = i;
  
    k.run(e);
  #else
    // Setting results path
    std::string trainingResultsPath = "_trainingResults-"+std::to_string(modelDIM)+"D/";
  
    // Creating Korali experiment
    auto e = korali::Experiment();
    e["Problem"]["Type"] = "Reinforcement Learning / Continuous";
  
    #if modelDIM == 2 
  
      //2D simply resumes execution if previous 2D results are available
      auto found = e.loadState(trainingResultsPath + std::string("/latest"));
      if (found)
      {
        printf("[Korali] Continuing execution from previous run...\n");
        // Hack to enable execution after Testing.
        e["Solver"]["Termination Criteria"]["Max Generations"] = std::numeric_limits<int>::max();
      }
  
    #elif modelDIM == 3
  
      //If 3D results are available we use them.
      //If not, we check for 2D results. 
      //If they are available, we use the 2D Policy Hyperparameters as initial guess for 3D policy.
  
      auto eOld2D = korali::Experiment();
      auto eOld3D = korali::Experiment();
      auto found2D = eOld2D.loadState("_trainingResults-2D/" + std::string("/latest"));
      auto found3D = eOld3D.loadState("_trainingResults-3D/" + std::string("/latest"));
  
      if (found3D)
      {
        auto found = e.loadState(trainingResultsPath + std::string("/latest"));
        if (rank == 0 && found)
          printf("[Korali] Continuing execution from previous run...\n");
  
        // Hack to enable execution after Testing.
        e["Solver"]["Termination Criteria"]["Max Generations"] = std::numeric_limits<int>::max();
  
        //uncomment to simply initialize 3D run with previous policy hyperparameters
        //e["Solver"]["Training"]["Current Policies"]["Policy Hyperparameters"] = 
        //                eOld3D["Solver"]["Training"]["Current Policies"]["Policy Hyperparameters"];
      }
      else if (found2D)
      {
        if (rank == 0)
          std::cout << "[Korali] 3D policy initialized with 2D policy." << std::endl;
  
        e["Solver"]["Training"]["Current Policies"]["Policy Hyperparameters"] = 
                     eOld2D["Solver"]["Training"]["Current Policies"]["Policy Hyperparameters"];
  
        // Hack to enable execution after Testing.
        e["Solver"]["Termination Criteria"]["Max Generations"] = std::numeric_limits<int>::max();
      }
  
    #endif
  
    e["Problem"]["Environment Function"] = &runEnvironment;
    //Results path and dumping frequency in CUP
    e["Problem"]["Custom Settings"]["Dump Frequency"] = 0.0;
    e["Problem"]["Custom Settings"]["Dump Path"] = trainingResultsPath;
    //Agent Configuration
    e["Solver"]["Type"] = "Agent / Continuous / VRACER";
    e["Solver"]["Mode"] = "Training";
    e["Solver"]["Episodes Per Generation"] = 1;
    e["Solver"]["Concurrent Environments"] = N;
  
    setupRL(e);//define state, action and neural network
   
    //Policy distribution and scaling parameters
    e["Solver"]["Policy"]["Distribution"] = "Clipped Normal";
    e["Solver"]["State Rescaling"]["Enabled"] = true;
    e["Solver"]["Reward"]["Rescaling"]["Enabled"] = true;
    //Termination Criteria
    e["Solver"]["Termination Criteria"]["Max Experiences"] = 1e9;
    //Korali output configuration
    e["Console Output"]["Verbosity"] = "Normal";
    e["File Output"]["Enabled"] = true;
    e["File Output"]["Frequency"] = 1;
    e["File Output"]["Use Multiple Files"] = false;
    e["File Output"]["Path"] = trainingResultsPath;
    //Running Experiment
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
  #endif
}