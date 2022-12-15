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

  // Get task and number of agents from command line argument
  const auto nAgents = atoi(_argv[_argc-3]);
  auto       task    = atoi(_argv[_argc-5]);

  // Process given task variable
  if(task == -1 )
  {
    if( s["Mode"] == "Training" )
    {
      // Sample task
      std::uniform_int_distribution<> disT(0, 1);
      task = disT(_randomGenerator);
      // For multitask learning, Korali has to know the task
      s["Environment Id"] = task;
    }
    else
    {
      // task = sampleId / 10;
      task = 0;
      s["Environment Id"] = task;
    }
  }

  // Argument string to inititialize Simulation
  std::string argumentString = "CUP-RL " + ( task == 5 ? OPTIONS_periodic : s["Mode"] == "Training"  ? OPTIONS : OPTIONS_testing ) + " -shapes ";

  /* Add Obstacle for cases 0,1,2,3; cases >3 do not have an obstacle!
    CASE 0: Halfdisk, maximize efficiency
    CASE 1: Hydrofoil, maximize efficiency
    CASE 2: Stefanfish, minimize lateral displacement
    CASE 3: Stefanfish, maximize efficiency
    CASE 4: None, minimize displacement while maximizing efficiency
    CASE 5: None, periodic domain 
    CASE 6: None, minimize displacement
    CASE 7: None, maximize efficiency
  */
  switch(task) {
    case 0 : // DCYLINDER
    {
      // Only rank 0 samples the radius
      double radius;
      if( s["Mode"] == "Training" )
      {
        if( rank == 0 )
        {
          std::uniform_real_distribution<double> radiusDist(0.03,0.07);
          radius = radiusDist(_randomGenerator);
        }
      }
      else
      {
        radius = 0.03 + (sampleId%10)*(0.07-0.03)/9.0;
      }

      // Broadcast radius to the other ranks
      MPI_Bcast(&radius, 1, MPI_DOUBLE, 0, comm);

      // Set argument string
      std::string OBJECT = "halfDisk angle=10 xpos=0.6 bForced=1 bFixed=1 xvel=0.15 tAccel=5 radius=";
      argumentString =  argumentString + OBJECT + std::to_string(radius);
      break;
    }
    case 1 : // HYDROFOIL
    {
      // Only rank 0 samples the frequency
      double frequency;
      if( s["Mode"] == "Training" )
      {
        if( rank == 0 )
        {
          std::uniform_real_distribution<double> frequencyDist(0.28,1.37);
          frequency = frequencyDist(_randomGenerator);
        }
      }
      else
      {
        frequency = 0.28 + (sampleId%10)*(1.37-0.28)/9.0;
      }

      // Broadcast frequency to the other ranks
      MPI_Bcast(&frequency, 1, MPI_DOUBLE, 0, comm);

      // Set argument string
      std::string OBJECT = "NACA L=0.12 xpos=0.6 angle=0 fixedCenterDist=0.299412 bFixed=1 xvel=0.15 Apitch=13.15 tAccel=5 Fpitch=";
      argumentString = argumentString + OBJECT + std::to_string(frequency);
      break;
    }
    case 2 : // Y-DISPLACEMENT
    case 3 : // EFFICIENCY
    {
      // Only rank 0 samples the length
      double length = 0.2;
      // if( s["Mode"] == "Training" )
      // {
      //   if( rank == 0 )
      //   {
      //     std::uniform_real_distribution<double> lengthDist(0.10,0.20);
      //     length = lengthDist(_randomGenerator);
      //   }
      // }
      // else
      // {
      //   length = 0.15 + (sampleId%10)*(0.25-0.15)/9.0;
      // }
      
      // // Broadcast length to the other ranks
      // MPI_Bcast(&length, 1, MPI_DOUBLE, 0, comm);

      // Set argument string
      // std::string OBJECT = "stefanfish T=1 xpos=0.6 bFixed=1 pid=1 L=";
      std::string OBJECT = "stefanfish T=1 xpos=0.6 bFixed=1 L=";
      argumentString = argumentString + OBJECT + std::to_string(length);
      break;
    }
  }

  /* Add Agent(s) */
  std::string AGENTPOSX  = " xpos=";
  std::string AGENTPOSY  = " ypos=";
  std::string AGENTANGLE = " angle=";

  // SINGLE AGENT
  std::string AGENT = " \n\
  stefanfish L=0.2 T=1";

  // AGENT IN PERIODIC DOMAIN
  std::string AGENT_periodic = " \n\
  stefanfish L=0.2 T=1 bFixed=1 ";

  // Declare initial data vector
  double initialData[3];

  // Set initial angle
  initialData[0] = 0.0;
  for( int a = 0; a < nAgents; a++ )
  {
    // Set initial position
    std::vector<double> initialPosition{ 0.9, 0.5 };
    if( (nAgents > 1) && (task != 8) )
    {
      initialPosition = initialPositions[a];
    }
    initialData[1] = initialPosition[0];
    initialData[2] = initialPosition[1];

    // During training, add noise to inital position of agent
    if ( s["Mode"] == "Training" )
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
    argumentString = argumentString + ( task>3 ? AGENT_periodic : AGENT ) + AGENTANGLE + std::to_string(initialData[0]) + AGENTPOSX + std::to_string(initialData[1]) + AGENTPOSY + std::to_string(initialData[2]);
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
  Simulation *_environment = new Simulation(argv.size() - 1, argv.data(), comm);
  _environment->init();

  // Obtaining agents
  std::vector<std::shared_ptr<Shape>> shapes = _environment->getShapes();
  std::vector<StefanFish *> agents(nAgents);
  if( task > 3 )
  {
    // All fish are agents
    for( int i = 0; i<nAgents; i++ )
      agents[i] = dynamic_cast<StefanFish *>(shapes[i].get());
  }
  else
  {
    // the first shape is the obstacle
    for( int i = 1; i<nAgents+1; i++ )
      agents[i-1] = dynamic_cast<StefanFish *>(shapes[i].get());
  }

  // Establishing environment's dump frequency
  _environment->sim.dumpTime = s["Custom Settings"]["Dump Frequency"].get<double>();

  // Setting initial state
  if( nAgents > 1 ) {
    std::vector<std::vector<double>> states(nAgents);
    for( int i = 0; i<nAgents; i++ )
    {
      std::vector<double> initialPosition = initialPositions[i];
      std::vector<double> state = getState(agents[i],initialPosition,_environment->sim,nAgents,i,task);

      // assign state/reward to container
      states[i]  = state;
    }
    s["State"] = states;
  }
  else
  {
    std::vector<double> initialPosition{ 0.9, 0.5 };
    s["State"] = getState(agents[0],initialPosition,_environment->sim,nAgents,0,task);
  }

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
      agents[i]->act(t, actions[i]);
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
    if( nAgents > 1 )
      dtAct = 0.5;
    else
      dtAct = agents[0]->getLearnTPeriod() * 0.5;
    tNextAct += dtAct;
    while ( t < tNextAct && done == false )
    {
      // Compute timestep
      const double dt = std::min(_environment->calcMaxTimestep(), dtAct);
      t += dt;

      // Advance simulation
      _environment->advance(dt);

      // Check if there was a collision -> termination.
      if( _environment->sim.bCollision )
      {
        if( nAgents > 1 )
        {
          for (size_t i = 0; i < _environment->sim.bCollisionID.size(); i++)
          {
            size_t indx = task > 3 ? _environment->sim.bCollisionID[i] : _environment->sim.bCollisionID[i]-1;
            bFailed[indx] = true;
          }
        }
        else
          bFailed[0] = true;
        done = true;
      }

      // Check termination because leaving margins
      for( int i = 0; i<nAgents; i++ )
      {
        bFailed[i] = isTerminal(agents[i], nAgents, task);
        done = done || bFailed[i];
      }
    }

    // Get and store state and reward [Carful, state function needs to be called by all ranks!] 
    if( nAgents > 1 ) {
      std::vector<std::vector<double>> states(nAgents);
      std::vector<double> rewards(nAgents);
      for( int i = 0; i<nAgents; i++ )
      {
        std::vector<double> initialPosition = initialPositions[i];
        std::vector<double> state = getState(agents[i],initialPosition,_environment->sim,nAgents,i,task);

        // assign state/reward to container
        states[i]  = state;
        if( (task == 2) || (task == 6) )
        {
          double xDisplacement = agents[i]->center[0]-initialPosition[0];
          double yDisplacement = agents[i]->center[1]-initialPosition[1];
          rewards[i] = bFailed[i] ? -10.0 : -std::sqrt(xDisplacement*xDisplacement + yDisplacement*yDisplacement);
        }
        else if( task == 4 )
        {
          double xDisplacement = agents[i]->center[0]-initialPosition[0];
          double yDisplacement = agents[i]->center[1]-initialPosition[1];
          rewards[i] = bFailed[i] ? -10.0 : agents[i]->EffPDefBnd-std::sqrt(xDisplacement*xDisplacement + yDisplacement*yDisplacement);
        }
        else
          rewards[i] = bFailed[i] ? -10.0 : agents[i]->EffPDefBnd;
      }
      s["State"]  = states;
      s["Reward"] = rewards;
    }
    else {
      std::vector<double> initialPosition{ 0.9, 0.5 };
      s["State"] = getState(agents[0],initialPosition,_environment->sim,nAgents,0,task);
      s["Reward"] = done ? -10.0 : agents[0]->EffPDefBnd;
    }

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

bool isTerminal(StefanFish *agent, int nAgents, int task)
{
  double xMin, xMax, yMin, yMax;
  if( nAgents == 1 ){
    // xMin = 0.6;
    // xMax = 1.4;
    xMin = 0.1;
    xMax = 1.9;

    // yMin = 0.2;
    // yMax = 0.8;
    yMin = 0.1;
    yMax = 0.9;
  }
  else if( nAgents == 2 ){
    xMin = 0.1;
    xMax = 1.9;
    yMin = 0.1;
    yMax = 0.9;
    if( task == 5 )
    {
      xMin = 0.10;
      xMax = 0.50;
      yMin = 0.10;
      yMax = 0.30;
    }
  }
  else if( nAgents == 3 ){
    // FOR SCHOOL OF FISH
    // small domain
    // xMin = 0.4;
    // xMax = 1.4;
    // yMin = 0.2;
    // yMax = 0.8;

    // large domain
    // xMin = 0.4;
    // xMax = 1.4;
    // yMin = 0.7;
    // yMax = 1.3;

    // FOR COLUMN OF FISH
    xMin = 0.1;
    xMax = 1.9;
    yMin = 0.1;
    yMax = 0.9;
  }
  else if( nAgents == 4 ){
    // small domain
    xMin = 0.1;
    xMax = 1.9;
    yMin = 0.1;
    yMax = 0.9;
    // large domain
    // xMin = 0.1;
    // xMax = 3.9;
    // yMin = 0.1;
    // yMax = 1.9;
    if( task == 5 )
    {
      xMin = 0.10;
      xMax = 1.10;
      yMin = 0.10;
      yMax = 0.30;
    }
  }
  else if( nAgents == 9 ){
    xMin = 0.4;
    xMax = 2.0;
    yMin = 0.6;
    yMax = 1.4;
  }
  else if( nAgents == 16 )
  {
    xMin = 0.4;
    xMax = 2.6;
    yMin = 0.5;
    yMax = 1.5;
  }
  else if( nAgents == 20 )
  {
    xMin = 0.1;
    xMax = 3.9;
    yMin = 0.1;
    yMax = 1.9;
  }
  else if( nAgents == 25 )
  {
    xMin = 0.4;
    xMax = 3.2;
    yMin = 0.4;
    yMax = 1.6;
  }
  else if( nAgents == 100 )
  {
    xMin = 0.1;
    xMax = 3.9;
    yMin = 0.1;
    yMax = 3.9;
  }
  else{
    fprintf(stderr, "Number of Agents unknown, isTerminal not implemented...\n");
    exit(-1);
  }

  const double X = agent->center[0];
  const double Y = agent->center[1];

  // Check if Swimmer is out of Bounds
  bool terminal = false;
  if (X < xMin) terminal = true;
  if (X > xMax) terminal = true;
  if (Y < yMin) terminal = true;
  if (Y > yMax) terminal = true;

  return terminal;
}


std::vector<double> getState(StefanFish *agent, const std::vector<double>& origin, const SimulationData & sim, const int nAgents, const int agentID, const int task)
{
  #ifdef STEFANS_SENSORS_STATE
    return agent->state(origin);
  #else
    const CurvatureFish* const cFish = dynamic_cast<CurvatureFish*>( agent->myFish );
    const double length = agent->length;
    const double Tperiod = agent->Tperiod;
    std::vector<double> S(10,0);
    S[0] = ( agent->center[0] - origin[0] )/ length;
    S[1] = ( agent->center[1] - origin[1] )/ length;
    S[2] = agent->getOrientation();
    S[3] = agent->getPhase( sim.time );
    S[4] = agent->getU() * Tperiod / length;
    S[5] = agent->getV() * Tperiod / length;
    S[6] = agent->getW() * Tperiod;
    S[7] = cFish->lastTact;
    S[8] = cFish->lastCurv;
    S[9] = cFish->oldrCurv;
  #endif
  // OLD STATE FOR REFERENCE
  #if 0
  S.resize(16);

  // Get velInfo
  const std::vector<cubism::BlockInfo>& velInfo = sim.vel->getBlocksInfo();

  // Get fish skin
  const auto &DU = cFish->upperSkin, &DL = cFish->lowerSkin;

  //// Sensor Signal on Front of Fish ////
  ////////////////////////////////////////

  // get front-point
  const std::array<Real,2> pFront = {DU.xSurf[0], DU.ySurf[0]};

  // first point of the two skins is the same
  // normal should be almost the same: take the mean
  const std::array<Real,2> normalFront = { (DU.normXSurf[0] + DL.normXSurf[0]) / 2,
                                           (DU.normYSurf[0] + DL.normYSurf[0]) / 2 };

  // compute shear stress
  std::array<Real,2> tipShear = agent->getShear( pFront, normalFront, velInfo );

  //// Sensor Signal on Side of Fish ////
  ///////////////////////////////////////

  std::array<Real,2> lowShear={}, topShear={};

  // get index for sensors on the side of head
  int iHeadSide = 0;
  for(int i=0; i<cFish->Nm-1; ++i)
     if( cFish->rS[i] <= 0.04*length && cFish->rS[i+1] > 0.04*length )
       iHeadSide = i;
  assert(iHeadSide>0);

  for(int a = 0; a<2; ++a)
  {
    // distinguish upper and lower skin
    const auto& D = a==0 ? cFish->upperSkin : cFish->lowerSkin;

    // get point
    const std::array<Real,2> pSide = { D.midX[iHeadSide], D.midY[iHeadSide] };

    // get normal to surface
    const std::array<Real,2> normSide = { D.normXSurf[iHeadSide], D.normYSurf[iHeadSide] };

    // compute shear stress
    std::array<Real,2> sideShear = agent->getShear( pSide, normSide, velInfo );

    // now figure out how to rotate it along the fish skin for consistency:
    const Real dX = D.xSurf[iHeadSide+1] - D.xSurf[iHeadSide];
    const Real dY = D.ySurf[iHeadSide+1] - D.ySurf[iHeadSide];
    const Real proj = dX * normSide[0] - dY * normSide[1];
    const Real tangX = proj>0?  normSide[0] : -normSide[0]; // s.t. tang points from head
    const Real tangY = proj>0? -normSide[1] :  normSide[1]; // to tail, normal outward
     (a==0? topShear[0] : lowShear[0]) = sideShear[0] * normSide[0] + sideShear[1] * normSide[1];
     (a==0? topShear[1] : lowShear[1]) = sideShear[0] * tangX + sideShear[1] * tangY;
  }

  // put non-dimensional results into state into state
  S[10] = tipShear[0] * Tperiod / length;
  S[11] = tipShear[1] * Tperiod / length;
  S[12] = lowShear[0] * Tperiod / length;
  S[13] = lowShear[1] * Tperiod / length;
  S[14] = topShear[0] * Tperiod / length;
  S[15] = topShear[1] * Tperiod / length;
  // printf("shear tip:[%f %f] lower side:[%f %f] upper side:[%f %f]\n", S[10],S[11], S[12],S[13], S[14],S[15]);
  // fflush(0);

   #endif //OLD_STATE

  return S;
}
