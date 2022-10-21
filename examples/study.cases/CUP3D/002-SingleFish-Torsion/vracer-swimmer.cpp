#include "swimmerEnvironment.hpp"

void runEnvironment(korali::Sample &s)
{
  // 1) Get MPI subcommunicator, rank and size
  int rank, size, rankGlobal;
  MPI_Comm comm = *(MPI_Comm*) korali::getWorkerMPIComm();
  MPI_Comm_rank(comm          ,&rank      );
  MPI_Comm_size(comm          ,&size      );
  MPI_Comm_rank(MPI_COMM_WORLD,&rankGlobal);

  // 2) Set random seed
  const size_t sampleId = s["Sample Id"];
  _randomGenerator.seed(sampleId);

  // 3) Create results directory and redirect all output to log file
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
    }
    char logFilePath[128];
    sprintf(logFilePath, "%s/log.txt", resDir);
    logFile = freopen(logFilePath, "a", stdout);
    if (logFile == NULL)
    {
      printf("[Korali] Error creating log file: %s.\n", logFilePath);
      exit(-1);
    }
  }

  // 4) Switch to results directory
  MPI_Barrier(comm); // Make sure logfile is created before switching path
  auto curPath = std::filesystem::current_path();
  std::filesystem::current_path(resDir);

  // 5) Initialize environment and obtain agents 
  Simulation *_environment = initializeEnvironment(s);
  auto & shapes = _environment->getShapes();
  std::vector<StefanFish *> agents(AGENTS);
  for(int i = 0; i<AGENTS; i++ )
    agents[i] = dynamic_cast<StefanFish *>(shapes[i].get());

  // 6) Set initial state (state function needs to be called by all ranks!)
  std::vector<std::vector<double>> actions(AGENTS, std::vector<double>(ACTIONS));
  std::vector<std::vector<double>> states (AGENTS, std::vector<double>(STATES ));
  std::vector<double> rewards(AGENTS);
  for(int i = 0; i<AGENTS; i++ ) states[i] = getState(agents,_environment->sim,i);
  if (AGENTS > 1)  s["State"] = states;
  else             s["State"] = states[0];

  // 7) Main environment loop
  double t        = 0; // Current time
  size_t curStep  = 0; // Current Step
  double tNextAct = 0; // Time of next action     
  bool done = false;
  while ( done == false )
  {
    // i) Get new actions, broadcast and apply them (also save them to file)
    if( rank == 0 )
    {
      s.update();
      for(int i = 0; i<AGENTS; i++ )
      {
        actions[i] = (AGENTS > 1) ? s["Action"][i].get<std::vector<double>>() : s["Action"].get<std::vector<double>>();
        ofstream myfile;
        myfile.open ("actions"+std::to_string(i)+".txt",ios::app);
        myfile << t << " ";
          for (int j = 0; j < ACTIONS ; j++)
              myfile << actions[i][j] << " ";
          myfile << std::endl;
        myfile.close();
      }
    }
    for(int i = 0; i<AGENTS; i++ )
    {
      MPI_Bcast( actions[i].data(), ACTIONS, MPI_DOUBLE, 0, comm );
      takeAction(agents[i],_environment->sim,i,actions[i],t);
    }

    // ii) Run the simulation until next action is required
    const double dtAct = AGENTS > 1 ? 0.5 : agents[0]->getLearnTPeriod() * 0.5;
    tNextAct += dtAct;
    while ( t < tNextAct && done == false )
    {
      const double dt = std::min(_environment->calcMaxTimestep(), dtAct);
      t += dt;
      _environment->advance(dt);

      // Check termination (leaving margins, collision, max steps etc.)
      for(int i = 0; i<AGENTS; i++ )
        done = ( done || isTerminal(agents[i],_environment->sim,i,curStep) );
    }

    // iii) Get and store state and reward (state function needs to be called by all ranks!)
    for(int i = 0; i<AGENTS; i++ )
    {
      states [i] = getState (agents,_environment->sim,i);
      rewards[i] = getReward(agents,_environment->sim,i);
    }
    if (AGENTS > 1) { s["State"] = states   ; s["Reward"] = rewards   ;}
    else            { s["State"] = states[0]; s["Reward"] = rewards[0];}

    // iv) Print information
    if ( rank == 0 )
    {
      printf("[Korali] -------------------------------------------------------\n");
      printf("[Korali] Sample %lu - Step: %lu\n", sampleId, curStep);
      for(int i = 0; i<AGENTS; i++ )
      {
        const auto state  = (AGENTS > 1) ? s["State" ][i].get<std::vector<float>>():s["State" ].get<std::vector<float>>();
        const auto action = (AGENTS > 1) ? s["Action"][i].get<std::vector<float>>():s["Action"].get<std::vector<float>>();
        const auto reward = (AGENTS > 1) ? s["Reward"][i].get            <float> ():s["Reward"].get            <float> ();
        printf("[Korali] AGENT %d/%d\n", i, AGENTS);
        printf("[Korali] State: [ %.3f", state[0]);
        for (size_t j = 1; j < state. size(); j++) printf(", %.3f", state[j]);
        printf("]\n");
        printf("[Korali] Action: [ %.3f", action[0]);
        for (size_t j = 1; j < action.size(); j++) printf(", %.3f", action[j]);
        printf("]\n");
        printf("[Korali] Reward: %.3f\n", reward);
        printf("[Korali] Terminal?: %d\n", done);
        printf("[Korali] -------------------------------------------------------\n");
      }
    }
    fflush(stdout);

    curStep++;// Advance to next step
  }

  s["Termination"] = "Terminal";// Setting termination status

  logger.flush();// Flush CUP logger

  delete _environment;// delete simulation class

  if( rank == 0 ) fclose(logFile); // Closing log file

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

#if 1
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
#else
  // Loading existing results and transplant best training policy
  auto eOld = korali::Experiment();
  auto found = eOld.loadState(trainingResultsPath + "/latest");
  if( found )
  {
    printf("[Korali] Continuing execution with policy learned in previous run...\n");
    e["Solver"]["Training"]["Current Policies"]["Policy Hyperparameters"] = eOld["Solver"]["Training"]["Current Policies"]["Policy Hyperparameters"];
  }
  else
  {
    printf("[Korali] Did not find the policy learned in previous run, training from scratch...\n");
  }
  e["Problem"]["Environment Function"] = &runEnvironment;
  //Results path and dumping frequency in CUP
  e["Problem"]["Custom Settings"]["Dump Frequency"] = 0.1;
  e["Problem"]["Custom Settings"]["Dump Path"] = trainingResultsPath;
  //Agent Configuration
  e["Solver"]["Type"] = "Agent / Continuous / VRACER";
  e["Solver"]["Mode"] = "Testing";
  e["Solver"]["Episodes Per Generation"] = 1;
  e["Solver"]["Concurrent Environments"] = N;
  for (int i = 0; i < N; i++) e["Solver"]["Testing"]["Sample Ids"][i] = 1+i;
#endif

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
