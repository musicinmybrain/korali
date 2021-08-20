//  Korali environment for CubismUP-2D
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.

#include "swimmerEnvironment.hpp"
#include <chrono>
#include <filesystem>

int _argc;
char **_argv;
std::mt19937 _randomGenerator;

// Swimmer following an obstacle
void runEnvironment(korali::Sample &s)
{
  // Setting seed
  size_t sampleId = s["Sample Id"];
  _randomGenerator.seed(sampleId);

  // Get rank to create/switch to results folder
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Creating results directory
  char resDir[64];
  sprintf(resDir, "%s/sample%03u", s["Custom Settings"]["Dump Path"].get<std::string>().c_str(), rank);
  if( not std::filesystem::exists(resDir) )
  if( not std::filesystem::create_directories(resDir) )
  {
    fprintf(stderr, "[Korali] Error creating results directory for environment: %s.\n", resDir);
    exit(-1);
  };

  // Redirecting all output to the log file
  char logFilePath[128];
  sprintf(logFilePath, "%s/log.txt", resDir);
  auto logFile = freopen(logFilePath, "a", stdout);
  if (logFile == NULL)
  {
    printf("[Korali] Error creating log file: %s.\n", logFilePath);
    exit(-1);
  }

  // Switching to results directory
  auto curPath = std::filesystem::current_path();
  std::filesystem::current_path(resDir);

  // Creating simulation environment
  Simulation *_environment = new Simulation(_argc, _argv);
  _environment->init();

  // Obtaining agents
  std::vector<Shape*> shapes = _environment->getShapes();
  size_t nAgents = shapes.size();
  if( nAgents == 2 )
    nAgents -= 1;
  std::vector<StefanFish *> agents(nAgents);
  for( size_t i = 0; i<nAgents; i++ )
    agents[i] = dynamic_cast<StefanFish *>(shapes[i]);

  // Establishing environment's dump frequency
  _environment->sim.dumpTime = s["Custom Settings"]["Dump Frequency"].get<double>();

  // Resetting environment and setting initial conditions
  for( size_t i = 0; i<nAgents; i++ )
    setInitialConditions(agents[i], i, s["Mode"] == "Training");
  // After moving the agent, the obstacles have to be restarted
  _environment->startObstacles();

  // Setting initial state
  if( nAgents > 1 )
  {
    std::vector<std::vector<double>> states(nAgents);
    size_t rowEndId = 0;
    size_t rowId = 0;
    size_t colId = 0;
    size_t nNextRow = 2;
    size_t nCurrRow = 1;
    bool increment = true;
    for( size_t i = 0; i<nAgents; i++ )
    {
      std::vector<double> state = agents[i]->state();
      // add column identifier ~ num fish in front
      state.push_back( colId );
      // number of fish to left
      state.push_back( rowId );
      // number of fish to right
      state.push_back( ( nCurrRow-1 ) - rowId );
      // increment counter
      rowId++;
      if( i == rowEndId )
      {
        rowId = 0;
        colId++;
        nCurrRow = nNextRow;
        rowEndId += nNextRow;
        if( nNextRow == (size_t)std::sqrt(nAgents) )
          increment = false;
        if( increment )
          nNextRow++;
        else
          nNextRow--;
      }
      // assign state/reward to container
      states[i]  = state;
    }
    s["State"] = states;
  }
  else
    s["State"] = agents[0]->state();

  // Variables for time and step conditions
  double t = 0;        // Current time
  size_t curStep = 0;  // current Step
  double dtAct;        // Time until next action
  double tNextAct = 0; // Time of next action     

  // Setting maximum number of steps before truncation
  size_t maxSteps = 200;

  // File to write actions
  std::stringstream filename;
  filename<<"actions.txt";
  ofstream myfile(filename.str().c_str());

  // Starting main environment loop
  bool done = false;
  while ( curStep < maxSteps && done == false )
  {
    // Getting new actions
    s.update();

    // Reading new action(s)
    auto actions = s["Action"];

    // Setting action for each agent
    for( size_t i = 0; i<nAgents; i++ )
    {
      std::vector<double> action;
      if( nAgents > 1 )
        action = actions[i].get<std::vector<double>>();
      else
        action = actions.get<std::vector<double>>();

      // Write action to file
      if (myfile.is_open())
      {
        myfile << i << " " << action[0] << " " << action[1] << std::endl;
      }
      else{
        fprintf(stderr, "Unable to open %s file...\n", filename.str().c_str());
        exit(-1);
      }

      // Apply action
      agents[i]->act(t, action);
    }

    // Run the simulation until next action is required
    dtAct = 0.;
    for( size_t i = 0; i<nAgents; i++ )
    if( dtAct < agents[i]->getLearnTPeriod() * 0.5 )
      dtAct = agents[i]->getLearnTPeriod() * 0.5;
    tNextAct += dtAct;
    while ( t < tNextAct && done == false )
    {
      // Compute timestep
      const double dt = std::min(_environment->calcMaxTimestep(), dtAct);
      t += dt;

      // Advance simulation
      _environment->advance(dt);

      // Check if there was a collision -> termination.
      done = _environment->sim.bCollision;

      // Check termination because leaving margins
      for( size_t i = 0; i<nAgents; i++ )
        done = ( done || isTerminal(agents[i], nAgents) );
    }

    // Get and store state and action
    if( nAgents > 1 )
    {
      std::vector<std::vector<double>> states(nAgents);
      std::vector<double> rewards(nAgents);
      size_t rowEndId = 0;
      size_t rowId = 0;
      size_t colId = 0;
      size_t nNextRow = 2;
      size_t nCurrRow = 1;
      bool increment = true;
      for( size_t i = 0; i<nAgents; i++ )
      {
        std::vector<double> state = agents[i]->state();
        // add column identifier
        state.push_back( colId );
        // number of fish to left
        state.push_back( rowId );
        // number of fish to right
        state.push_back( ( nCurrRow-1 ) - rowId );
        // increment counter
        rowId++;
        if( i == rowEndId )
        {
          rowId = 0;
          colId++;
          nCurrRow = nNextRow;
          rowEndId += nNextRow;
          if( nNextRow == (size_t)std::sqrt(nAgents) )
            increment = false;
          if( increment )
            nNextRow++;
          else
            nNextRow--;
        }
        // assign state/reward to container
        states[i]  = state;
        rewards[i] = done ? -10.0 : agents[i]->EffPDefBnd;
      }
      s["State"]  = states;
      s["Reward"] = rewards;
    }
    else{
      s["State"]  = agents[0]->state();
      s["Reward"] = done ? -10.0 : agents[0]->EffPDefBnd;
    }

    // Printing Information:
    printf("[Korali] -------------------------------------------------------\n");
    printf("[Korali] Sample %lu - Step: %lu/%lu\n", sampleId, curStep, maxSteps);
    if( nAgents > 1 )
    {
      for( size_t i = 0; i<nAgents; i++ )
      {
        auto state  = s["State"][i].get<std::vector<float>>();
        auto action = s["Action"][i].get<std::vector<float>>();
        auto reward = s["Reward"][i].get<float>();
        printf("[Korali] AGENT %ld/%ld\n", i, nAgents);
        printf("[Korali] State: [ %.3f", state[0]);
        for (size_t j = 1; j < state.size(); j++) printf(", %.3f", state[j]);
        printf("]\n");
        printf("[Korali] Action: [ %.3f, %.3f ]\n", action[0], action[1]);
        printf("[Korali] Reward: %.3f\n", reward);
        printf("[Korali] Terminal?: %d\n", done);
        printf("[Korali] -------------------------------------------------------\n");
      }
    }
    else{
      auto state  = s["State"].get<std::vector<float>>();
      auto action = s["Action"].get<std::vector<float>>();
      auto reward = s["Reward"].get<float>();
      printf("[Korali] State: [ %.3f", state[0]);
      for (size_t j = 1; j < state.size(); j++) printf(", %.3f", state[j]);
      printf("]\n");
      printf("[Korali] Action: [ %.3f, %.3f ]\n", action[0], action[1]);
      printf("[Korali] Reward: %.3f\n", reward);
      printf("[Korali] Terminal?: %d\n", done);
      printf("[Korali] -------------------------------------------------------\n");
    }
    fflush(stdout);

    // Advancing to next step
    curStep++;
  }

  // Close file to write actions
  myfile.close();

  // Flush CUP logger
  logger.flush();

  // delete simulation class
  delete _environment;

  // Setting finalization status
  if (done == true)
    s["Termination"] = "Terminal";
  else
    s["Termination"] = "Truncated";

  // Switching back to experiment directory
  std::filesystem::current_path(curPath);

  // Closing log file
  fclose(logFile);
}

void setInitialConditions(StefanFish *agent, size_t agentId, const bool isTraining)
{
  // Initial fixed conditions
  double initialAngle = 0.0;
  std::vector<double> initialPosition{ agent->origC[0], agent->origC[1] };

  // with noise
  if (isTraining)
  {
    std::uniform_real_distribution<double> disA(-5. / 180. * M_PI, 5. / 180. * M_PI);
    std::uniform_real_distribution<double> disX(-0.025, 0.025);
    std::uniform_real_distribution<double> disY(-0.05, 0.05);

    initialAngle = disA(_randomGenerator);
    initialPosition[0] = initialPosition[0] + disX(_randomGenerator);
    initialPosition[1] = initialPosition[1] + disY(_randomGenerator);
  }

  printf("[Korali] Initial Condition Agent %ld:\n", agentId);
  printf("[Korali] angle: %f\n", initialAngle);
  printf("[Korali] x: %f\n", initialPosition[0]);
  printf("[Korali] y: %f\n", initialPosition[1]);

  // Write initial condition to file
  std::stringstream filename;
  filename<<"initialCondition.txt";
  ofstream myfile(filename.str().c_str(), std::ofstream::app);
  if (myfile.is_open())
  {
    myfile << agentId << " " << initialAngle << " " << initialPosition[0] << " " << initialPosition[1] << std::endl;
    myfile.close();
  }
  else{
    fprintf(stderr, "Unable to open %s file...\n", filename.str().c_str());
    exit(-1);
  }

  // Setting initial position and orientation for the fish
  agent->setCenterOfMass(initialPosition.data());
  agent->setOrientation(initialAngle);
}

bool isTerminal(StefanFish *agent, size_t nAgents)
{
  double xMin, xMax, yMin, yMax;
  if( nAgents == 1 ){
    xMin = 0.8;
    xMax = 1.4;
    yMin = 0.8;
    yMax = 1.2;
  }
  else if( nAgents == 4 ){
    xMin = 0.4;
    xMax = 1.4;
    yMin = 0.7;
    yMax = 1.3;
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
  else if( nAgents == 25 )
  {
    xMin = 0.4;
    xMax = 3.2;
    yMin = 0.4;
    yMax = 1.6;
  }
  else{
    fprintf(stderr, "Number of Agents unknown, can not finish isTerminal...\n");
    exit(-1);
  }

  const double X = agent->center[0];
  const double Y = agent->center[1];

  bool terminal = false;
  if (X < xMin) terminal = true;
  if (X > xMax) terminal = true;
  if (Y < yMin) terminal = true;
  if (Y > yMax) terminal = true;

  return terminal;
}
