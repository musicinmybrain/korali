//  Korali environment for CubismUP-3D
//  Copyright (c) 2022 CSE-Lab, ETH Zurich, Switzerland.

#include "swimmerEnvironment3D.hpp"


int _argc;
char **_argv;
std::mt19937 _randomGenerator;

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

  // Creating results directory and redirecting all output to log file
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

  // Switching to results directory
  MPI_Barrier(comm); // Make sure logfile is created before switching path
  auto curPath = std::filesystem::current_path();
  std::filesystem::current_path(resDir);

  // Get task and number of agents from command line argument
  const int task = atoi(_argv[_argc-5]);
  int nAgents    = 0;

  if (task == 0)
  {
    nAgents = 1;
  }
  else
  {
      std::cerr << "Task given: " << task << " is not supported." << std::endl;
      MPI_Abort(comm,1);
  }

  Simulation *_environment = initializeEnvironment(s,task);
  // Obtain agents
  std::vector<std::shared_ptr<Obstacle>> shapes = _environment->getObstacleVector();
  std::vector<cubismup3d::StefanFish *> agents(nAgents);
  for(int i = 0; i<nAgents; i++ )
    agents[i] = dynamic_cast<cubismup3d::StefanFish *>(shapes[i].get());

  // Setting initial state [Careful, state function needs to be called by all ranks!]
  {
    std::vector<std::vector<double>> states(nAgents);
    for(int i = 0; i<nAgents; i++ ) states[i]  = agents[i]->state();
    if (nAgents > 1) s["State"] = states;
    else             s["State"] = states[0];
  }

  // Variables for time and step conditions
  double t        = 0; // Current time
  size_t curStep  = 0; // Current Step
  double dtAct;        // Time until next action
  double tNextAct = 0; // Time of next action     
  const size_t maxSteps = 50; // Max steps before truncation


  std::vector<std::vector<double>> actions(nAgents, std::vector<double>(NACTIONS));

  // Main environment loop
  bool done = false;
  while ( curStep < maxSteps && done == false )
  {
    if( rank == 0 ) // Get new actions, then broadcast and apply them
    {
      s.update();
      for(int i = 0; i<nAgents; i++ )
        actions[i] = (nAgents > 1) ? s["Action"][i].get<std::vector<double>>() : s["Action"].get<std::vector<double>>();
    }
    for(int i = 0; i<nAgents; i++ )
    {
      MPI_Bcast( actions[i].data(), NACTIONS, MPI_DOUBLE, 0, comm );
      agents[i]->act(t, actions[i]);
    }

    if (rank == 0) //Write a file with the actions for every agent
    {
      for( int i = 0; i<nAgents; i++ )
      {
        ofstream myfile;
        myfile.open ("actions"+std::to_string(i)+".txt",ios::app);
        myfile << t << " ";
	      for (int j = 0; j < NACTIONS ; j++)
		      myfile << actions[i][j] << " ";
	      myfile << std::endl;
        myfile.close();
      }
    }
    // Run the simulation until next action is required
    dtAct = 0.;
    for(int i = 0; i<nAgents; i++ )
    if( dtAct < agents[i]->getLearnTPeriod() * 0.5 )
      dtAct = agents[i]->getLearnTPeriod() * 0.5;
    tNextAct += dtAct;
    while ( t < tNextAct && done == false )
    {
      const double dt = std::min(_environment->calcMaxTimestep(), dtAct);
      t += dt;
      _environment->timestep(dt); // Advance simulation

      done = _environment->sim.bCollision; // if collision -> terminate

      // Check termination because leaving margins
      for(int i = 0; i<nAgents; i++ )
        done = ( done || isTerminal(agents[i]) );
    }

    // Get and store state and reward 
    // [Careful, state function needs to be called by all ranks!] 
    {
      std::vector<std::vector<double>> states(nAgents);
      std::vector<double> rewards(nAgents);
      for(int i = 0; i<nAgents; i++ )
      {
        states[i]  = agents[i]->state();
        rewards[i] = getReward(agents[i]);
      }
      if (nAgents > 1) { s["State"] = states   ; s["Reward"] = rewards   ;}
      else             { s["State"] = states[0]; s["Reward"] = rewards[0];}
    }

    // Printing Information:
    if ( rank == 0 )
    {
      printf("[Korali] -------------------------------------------------------\n");
      printf("[Korali] Sample %lu - Step: %lu/%lu\n", sampleId, curStep, maxSteps);
      for(int i = 0; i<nAgents; i++ )
      {
          auto state  = (nAgents > 1) ? s["State"][i].get<std::vector<float>>():s["State"].get<std::vector<float>>();
          auto action = (nAgents > 1) ? s["Action"][i].get<std::vector<float>>():s["Action"].get<std::vector<float>>();
          auto reward = (nAgents > 1) ? s["Reward"][i].get<float>():s["Reward"].get<float>();
          printf("[Korali] AGENT %d/%d\n", i, nAgents);
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

void setInitialConditions(cubismup3d::StefanFish *agent, size_t agentId, const bool isTraining, int rank, MPI_Comm comm)
{
  //double initialAngle = 0.0;
  std::array<double,3> initialPosition = {agent->origC[0],agent->origC[1],agent->origC[2]};
  if (rank == 0)
  {
    if (isTraining)// with noise
    {
      //std::uniform_real_distribution<double> disA(-5. / 180. * M_PI, 5. / 180. * M_PI);
      std::uniform_real_distribution<double> disX(-0.1, 0.1);
      std::uniform_real_distribution<double> disY(-0.1, 0.1);
      //std::uniform_real_distribution<double> disZ(-0.05, 0.05);
      //initialAngle = disA(_randomGenerator);
      initialPosition[0] = initialPosition[0] + disX(_randomGenerator);
      initialPosition[1] = initialPosition[1] + disY(_randomGenerator);
      //initialPosition[2] = initialPosition[2] + disZ(_randomGenerator);
    }
    printf("[Korali] Initial Condition Agent %ld:\n", agentId);
    printf("[Korali] x: %f\n", initialPosition[0]);
    printf("[Korali] y: %f\n", initialPosition[1]);
    printf("[Korali] z: %f\n", initialPosition[2]);
  }
  MPI_Bcast( &initialPosition[0], 3, MPI_DOUBLE, 0, comm );
  agent->absPos      [0] = initialPosition[0];
  agent->absPos      [1] = initialPosition[1];
  agent->absPos      [2] = initialPosition[2];
  agent->centerOfMass[0] = initialPosition[0];
  agent->centerOfMass[1] = initialPosition[1];
  agent->centerOfMass[2] = initialPosition[2];
  agent->position    [0] = initialPosition[0];
  agent->position    [1] = initialPosition[1];
  agent->position    [2] = initialPosition[2];
  agent->quaternion  [0] = 1.0;
  agent->quaternion  [1] = 0.0;
  agent->quaternion  [2] = 0.0;
  agent->quaternion  [3] = 0.0;
}

bool isTerminal(cubismup3d::StefanFish *agent)
{
  const double xMin = 0.1;
  const double xMax = 1.9;
  const double yMin = 0.1;
  const double yMax = 1.9;
  const double zMin = 0.1;
  const double zMax = 1.9;
  const double X = agent->absPos[0];
  const double Y = agent->absPos[1];
  const double Z = agent->absPos[2];
  if (X < xMin) return true;
  if (X > xMax) return true;
  if (Y < yMin) return true;
  if (Y > yMax) return true;
  if (Z < zMin) return true;
  if (Z > zMax) return true;
  const double Xt = 0.5;
  const double Yt = 0.5;
  const double Zt = agent->absPos[2];
  const double d  = std::pow((X -Xt)*(X -Xt) + (Y -Yt)*(Y -Yt) + (Z -Zt)*(Z -Zt),0.5);
  if (d < 1e-2) return true;
  return false;
}

double getReward(cubismup3d::StefanFish *agent)
{
  const double X = agent->absPos[0];
  const double Y = agent->absPos[1];
  const double Z = agent->absPos[2];
  const double Xt = 0.5;
  const double Yt = 0.5;
  const double Zt = agent->absPos[2];
  const double d  = std::pow((X -Xt)*(X -Xt) + (Y -Yt)*(Y -Yt) + (Z -Zt)*(Z -Zt),0.5);
  std::cout << "Current position = (" << X << "," << Y << ")" << std::endl;
  std::cout << "Distance from target = " << d << std::endl;
  if (d < 1e-2) return 20.0;
  return -d;
}
Simulation * initializeEnvironment(korali::Sample &s, const int task)
{
  MPI_Comm comm = *(MPI_Comm*) korali::getWorkerMPIComm();
  int rank;
  MPI_Comm_rank(comm,&rank);
  int nAgents    = 0;

  if (task == 0)
  {
    nAgents = 1;
  }
  else
  {
      std::cerr << "Task given: " << task << " is not supported." << std::endl;
      MPI_Abort(comm,1);
  }

  // Create simulation environment and set dump frequency
  ArgumentParser parser(_argc, _argv);
  Simulation *_environment = new Simulation(comm, parser);
  _environment->sim.saveTime = s["Custom Settings"]["Dump Frequency"].get<double>();

  // Obtain agents and set initial conditions
  std::vector<std::shared_ptr<Obstacle>> shapes = _environment->getObstacleVector();
  std::vector<cubismup3d::StefanFish *> agents(nAgents);
  for(int i = 0; i<nAgents; i++ )
  {
     agents[i] = dynamic_cast<cubismup3d::StefanFish *>(shapes[i].get());
     setInitialConditions(agents[i], i, s["Mode"] == "Training", rank, comm);
  }
  _environment->initialGridRefinement();//setting ICs means moving the fish from their default position, so the grid needs to be adapted.
  return _environment;
}
