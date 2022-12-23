//  Korali environment for CubismUP-2D
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.

#include "swimmerEnvironment.hpp"
#include "configs.hpp"
#include <iostream>
#include <fstream>
#include <filesystem>

int _argc;
char **_argv;
std::mt19937 _randomGenerator;
size_t NUMACTIONS = 2;

// Environment Function
void runEnvironment(korali::Sample &s)
{
  // Get MPI communicator
  MPI_Comm comm = *(MPI_Comm*) korali::getWorkerMPIComm();

  // Get rank and size of subcommunicator
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  // Get rank in world
  int rankGlobal;
  MPI_Comm_rank(MPI_COMM_WORLD, &rankGlobal);

  // Setting seed
  const size_t sampleId = s["Sample Id"];
  _randomGenerator.seed(sampleId);

  // Creating results directory
  char resDir[64];
  if( s["Mode"] == "Training" )
    sprintf(resDir, "%s/sample%03u", s["Custom Settings"]["Dump Path"].get<std::string>().c_str(), rankGlobal/size);
  else
    sprintf(resDir, "%s/sample%04lu", s["Custom Settings"]["Dump Path"].get<std::string>().c_str(), sampleId);
  if( rank == 0 )
  if( not std::filesystem::exists(resDir) )
  if( not std::filesystem::create_directories(resDir) )
  {
    fprintf(stderr, "[Korali] Error creating results directory for environment: %s.\n", resDir);
    exit(-1);
  };

  // Redirecting all output to the log file
  FILE * logFile = nullptr;
  if( rank == 0 ) {
    char logFilePath[128];
    sprintf(logFilePath, "%s/log.txt", resDir);
    logFile = freopen(logFilePath, "w", stdout);
    if (logFile == NULL)
    {
      printf("[Korali] Error creating log file: %s.\n", logFilePath);
      exit(-1);
    }
  }

  // Make sure folder / logfile is created before switching path
  MPI_Barrier(comm);

  // Switching to results directory
  auto curPath = std::filesystem::current_path();
  std::filesystem::current_path(resDir);

  // Get number of agents from command line argument
  const auto nAgents = atoi(_argv[_argc-3]);

  // Argument string to inititialize Simulation
  std::string argumentString = "CUP-RL " + OPTIONS + " -shapes ";

  // Add Obstacles

  /* Add Agent(s) */
  std::string AGENTANGLE = " planarAngle=";
  std::string AGENTPOSX  = " xpos=";
  std::string AGENTPOSY  = " ypos=";
  std::string AGENTPOSZ  = " zpos=0.25";

  std::string AGENT = " \n\
  StefanFish heightProfile=danio widthProfile=stefan bFixToPlanar=1 L=0.2 T=1";

  // Declare initial data vector
  double initialData[3];

  // Set initial angle
  initialData[0] = 0.0;

  // To over all agents
  for( int a = 0; a < nAgents; a++ )
  {
    // Get initial position
    std::vector<double> initialPosition = initialPositions[a];
    
    // Set value in initial data vector
    initialData[1] = initialPosition[0];
    initialData[2] = initialPosition[1];

    // During training, add noise to inital position of agent
    // if ( (s["Mode"] == "Training") || (sampleId == 0) )
    {
      // only rank 0 samples initial data
      if( rank == 0 )
      {
        std::uniform_real_distribution<double> disA(-5. / 180. * M_PI, 5. / 180. * M_PI);
        std::uniform_real_distribution<double> disX(-0.05, 0.05);
        std::uniform_real_distribution<double> disY(-0.05, 0.05);
        initialData[0] = initialData[0] + disA(_randomGenerator);
        initialData[1] = initialData[1] + disX(_randomGenerator);
        initialData[2] = initialData[2] + disY(_randomGenerator);
      }
      // Broadcast initial data to all ranks
      MPI_Bcast(initialData, 3, MPI_DOUBLE, 0, comm);
    }

    // Append agent to argument string
    argumentString = argumentString + AGENTANGLE + std::to_string(initialData[0]) + AGENTPOSX + std::to_string(initialData[1]) + AGENTPOSY + std::to_string(initialData[2]) + AGENTPOSZ;
  }

  // printf("%s\n",argumentString.c_str());
  // fflush(0);

  std::stringstream ss(argumentString);
  std::string item;
  std::vector<std::string> arguments;
  while ( std::getline(ss, item, ' ') )
    arguments.push_back(item);

  // Create argc / argv to pass to CUP
  std::vector<char*> argv;
  for (const auto& arg : arguments)
    argv.push_back((char*)arg.data());
  argv.push_back(nullptr);

  // Creating and initializing simulation environment
  ArgumentParser parser(argv.size()-1, argv.data());
  Simulation *_environment = new Simulation(comm, parser);

  // Obtaining agents
  auto & shapes = _environment->getShapes();
  std::vector<StefanFish *> agents(nAgents);
  for( int i = 0; i<nAgents; i++ )
    agents[i] = dynamic_cast<StefanFish *>(shapes[i].get());

  // Establishing environment's dump frequency
  _environment->sim.dumpTime = s["Custom Settings"]["Dump Frequency"].get<double>();

  // Setting initial state
  std::vector<std::vector<double>> states(nAgents);
  for( int i = 0; i<nAgents; i++ )
  {
    std::vector<double> initialPosition = initialPositions[i];
    std::vector<double> state = getState(agents[i], initialPosition, _environment->sim);

    // assign state/reward to container
    states[i]  = state;
  }
  s["State"] = states;

  // Variables for time and step conditions
  double t = 0;        // Current time
  size_t curStep = 0;  // current Step
  double dtAct;        // Time until next action
  double tNextAct = 0; // Time of next action

  // Setting maximum number of steps before truncation
  const size_t maxSteps = 200;

  // Container for actions
  std::vector<std::vector<double>> actions(nAgents, std::vector<double>(NUMACTIONS));

  // Starting main environment loop
  bool done = false;
  std::vector<char> bFailed(nAgents, false);
  while ( curStep < maxSteps && done == false )
  {
    // Only rank 0 communicates with Korali
    if( rank == 0 ) {
      // Getting new action(s)
      s.update();
      auto actionsJson = s["Action"];

      // Setting action for each agent
      for( int i = 0; i<nAgents; i++ )
      {
        std::vector<double> action;
        if( nAgents > 1 )
          action = actionsJson[i].get<std::vector<double>>();
        else
          action = actionsJson.get<std::vector<double>>();
        actions[i] = action;
      }
    }

    // Broadcast and apply action(s) [Careful, hardcoded the number of action(s)!]
    for( int i = 0; i<nAgents; i++ )
    {
      if( actions[i].size() != NUMACTIONS )
      {
        std::cout << "Korali returned the wrong number of actions " << actions[i].size() << "\n";
        fflush(0);
        abort();
      }
      MPI_Bcast( actions[i].data(), NUMACTIONS, MPI_DOUBLE, 0, comm );

      auto * const cFish = dynamic_cast<CurvatureDefinedFishData*>( agents[i]->myFish );
      cFish->action_curvature(_environment->sim.time,t, actions[i][0]);
      cFish->action_period   (_environment->sim.time,t, actions[i][1]);
    }

    if (rank == 0) //Write a file with the actions for every agent
    {
      for( int i = 0; i<nAgents; i++ )
      {
          ofstream myfile;
          myfile.open ("actions"+std::to_string(i)+".txt",ios::app);
          myfile << t << " ";
          for( size_t a = 0; a<actions[i].size(); a++ )
            myfile << actions[i][a] << " ";
          myfile << std::endl;
          myfile.close();
      }
    }

    // Run the simulation until next action is required
    dtAct = 0.5;
    tNextAct += dtAct;
    while ( t < tNextAct && done == false )
    {
      // Compute timestep
      const double dt = std::min(_environment->calcMaxTimestep(), dtAct);
      t += dt;

      // Advance simulation
      _environment->advance(dt);

      // Check if there was a collision
      if( _environment->sim.bCollision )
      {
        // Determine which agents collided
        for (size_t i = 0; i < _environment->sim.bCollisionID.size(); i++)
        {
          size_t indx = _environment->sim.bCollisionID[i];
          bFailed[indx] = true;
        }

        // Terminate simulation
        done = true;
      }

      // Check termination because leaving margins
      for( int i = 0; i<nAgents; i++ )
      {
        bFailed[i] = isTerminal(agents[i]);
        done = done || bFailed[i];
      }
    }

    // Get and store state and reward [Carful, state function needs to be called by all ranks!] 
    std::vector<std::vector<double>> states(nAgents);
    std::vector<double> rewards(nAgents);
    for( int i = 0; i<nAgents; i++ )
    {
      std::vector<double> initialPosition = initialPositions[i];
      std::vector<double> state = getState(agents[i],initialPosition,_environment->sim);

      // assign state/reward to container
      states[i]  = state;

      // Get reward
      rewards[i] = bFailed[i] ? -10.0 : agents[i]->EffPDefBnd;
    }
    s["State"]  = states;
    s["Reward"] = rewards;

    // Print information
    if ( rank == 0 )
    {
      printf("[Korali] -------------------------------------------------------\n");
      printf("[Korali] Sample %lu - Step: %lu/%lu\n", sampleId, curStep, maxSteps);
      for(int i = 0; i<nAgents; i++ )
      {
        const auto state  = (nAgents > 1) ? s["State" ][i].get<std::vector<float>>():s["State" ].get<std::vector<float>>();
        const auto action = (nAgents > 1) ? s["Action"][i].get<std::vector<float>>():s["Action"].get<std::vector<float>>();
        const auto reward = (nAgents > 1) ? s["Reward"][i].get            <float> ():s["Reward"].get            <float> ();
        printf("[Korali] AGENT %d/%d\n", i, nAgents);
        printf("[Korali] State: [ %.3f", state[0]);
        for (size_t j = 1; j < state.size(); j++) printf(", %.3f", state[j]);
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

    // Advancing to next step
    curStep++;
  }

  // Setting termination status
  s["Termination"] = done ? "Terminal" : "Truncated";

  // Flush CUP logger
  logger.flush();

  // Closing log file
  if( rank == 0 )
    fclose(logFile);

  // delete simulation class
  delete _environment;

  // Switching back to experiment directory
  std::filesystem::current_path(curPath);
}

bool isTerminal(StefanFish *agent)
{
  // Get swimmer position
  const double X = agent->position[0];
  const double Y = agent->position[1];

  // Get margins
  const double xMin = 0.1;
  const double xMax = 1.9;

  const double yMin = 0.1;
  const double yMax = 0.9;

  // Check if Swimmer is out of Bounds
  bool terminal = false;
  if (X < xMin) terminal = true;
  if (X > xMax) terminal = true;
  if (Y < yMin) terminal = true;
  if (Y > yMax) terminal = true;

  return terminal;
}

std::vector<double> getState(StefanFish *agent, const std::vector<double>& origin, const SimulationData & sim)
{
  auto * const cFish = dynamic_cast<CurvatureDefinedFishData*>( agent->myFish );
  const double length = agent->length;
  const double Tperiod = cFish->Tperiod;
  std::vector<double> S(7,0);
  S[0] = ( agent->position[0] - origin[0] )/ length;
  S[1] = ( agent->position[1] - origin[1] )/ length;
  S[2] = 2 * std::atan2(agent->quaternion[3], agent->quaternion[0]);
  S[3] = agent->getPhase( sim.time );
  S[4] = agent->transVel[0] * Tperiod / length;
  S[5] = agent->transVel[1] * Tperiod / length;
  S[6] = agent->angVel[2] * Tperiod;
  S[7] = cFish->lastTact;
  S[8] = cFish->lastCurv;
  S[9] = cFish->oldrCurv;

  return S;
}
