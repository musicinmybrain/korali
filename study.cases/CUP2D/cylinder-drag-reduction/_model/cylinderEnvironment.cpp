//  Korali environment for CubismUP-2D
//  Copyright (c) 2023 CSE-Lab, ETH Zurich, Switzerland.

#include "cylinderEnvironment.hpp"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <exception>

//For Re=400
//std::string OPTIONS_testing = "-restart 1 -poissonSolver iterative -bMeanConstraint 2 -CFL 0.5 -bpdx 8 -bpdy 4 -levelMax 6 -levelStart 4 -Rtol 1.0 -Ctol 0.1 -extent 4. -tdump 0 -tend 0. -muteAll 0 -verbose 0 -poissonTol 0. -poissonTolRel 1e-4 -bAdaptChiGradient 1";
//std::string OPTIONS         = "-restart 1 -poissonSolver iterative -bMeanConstraint 2 -CFL 0.5 -bpdx 8 -bpdy 4 -levelMax 5 -levelStart 4 -Rtol 1.0 -Ctol 0.1 -extent 4. -tdump 0 -tend 0. -muteAll 0 -verbose 0 -poissonTol 0. -poissonTolRel 1e-4 -bAdaptChiGradient 1";

//For Re=4000
std::string OPTIONS_testing = "-restart 1 -bMeanConstraint 2 -bpdx 8 -bpdy 4 -levelMax 6 -levelStart 4 -Rtol 1.0 -Ctol 0.1 -extent 2 -CFL 0.50 -tdump 0.1 -tend 0 -muteAll 0 -verbose 0 -poissonTol 1e-7 -poissonTolRel 1e-4 -bAdaptChiGradient 1 -poissonSolver iterative";
std::string OPTIONS         = "-restart 1 -bMeanConstraint 2 -bpdx 8 -bpdy 4 -levelMax 5 -levelStart 4 -Rtol 1.0 -Ctol 0.1 -extent 2 -CFL 0.50 -tdump 0.1 -tend 0 -muteAll 0 -verbose 0 -poissonTol 1e-7 -poissonTolRel 1e-4 -bAdaptChiGradient 1 -poissonSolver iterative";

int _argc;
char **_argv;
std::mt19937 _randomGenerator;

namespace fs = std::filesystem;

void runEnvironment(korali::Sample &s)
{
  MPI_Comm comm = *(MPI_Comm*) korali::getWorkerMPIComm();
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  // Set seed
  const size_t sampleId = s["Sample Id"];
  _randomGenerator.seed(sampleId);

  // Create results directory
  char resDir[64];
  {
    int rankGlobal;
    MPI_Comm_rank(MPI_COMM_WORLD, &rankGlobal);

    if( s["Mode"] == "Training" ) sprintf(resDir, "%s/sample%03u" , s["Custom Settings"]["Dump Path"].get<std::string>().c_str(), rankGlobal/size);
    else                          sprintf(resDir, "%s/sample%04lu", s["Custom Settings"]["Dump Path"].get<std::string>().c_str(), sampleId       );

    if( rank == 0 && fs::exists(resDir) == false )
      if( not fs::create_directories(resDir) )
      {
        fprintf(stderr, "[Korali] Error creating results directory for environment: %s.\n", resDir);
        exit(-1);
      };
  }

  // Redirect all output to the log file
  FILE * logFile = nullptr;
  if( rank == 0 )
  {
    char logFilePath[128];
    sprintf(logFilePath, "%s/log.txt", resDir);
    logFile = freopen(logFilePath, "w", stdout);
    if (logFile == NULL)
    {
      printf("[Korali] Error creating log file: %s.\n", logFilePath);
      exit(-1);
    }
  }

  //sample IC
  //Re= 400, 4000, 25000
  const double nu_values [3] = {0.0001,0.00001,0.0000016};
  int index_ic = 1;
  MPI_Bcast(&index_ic, 1, MPI_INT, 0, comm );
  const double nu_ic = nu_values[index_ic];

  if ( rank == 0 )
  {
    try
    {
	if ( s["Mode"] == "Training" )
                fs::copy("../IC"+std::to_string(index_ic)+"/", resDir, fs::copy_options::overwrite_existing | fs::copy_options::recursive);
	else
                fs::copy("../ICtesting"+std::to_string(index_ic)+"/", resDir, fs::copy_options::overwrite_existing | fs::copy_options::recursive);
                //fs::copy("../customIC25000/", resDir, fs::copy_options::overwrite_existing | fs::copy_options::recursive);
    }
    catch (std::exception& e)
    {
        std::cout << e.what();
    }
  }

  // Switch to results directory
  MPI_Barrier(comm);
  auto curPath = fs::current_path();
  fs::current_path(resDir);

  const int nAgents = 1;
  const double reg = std::stod((_argv[_argc-3]));

  // Argument string to inititialize Simulation
  std::string argumentString = "CUP-RL " + (s["Mode"] == "Training" ? OPTIONS : OPTIONS_testing);
  argumentString += " -nu " + std::to_string(nu_ic);
  argumentString += " -shapes cylinderNozzle xpos=0.5 bForced=1 bFixed=1 xvel=0.2 Nactuators="+std::to_string(NUMACTIONS)+" actuator_theta=8 radius=0.1 dumpSurf=1 regularizer=" + std::to_string(reg);

  // Create argc / argv to pass to CUP
  std::stringstream ss(argumentString);
  std::string item;
  std::vector<std::string> arguments;
  while ( std::getline(ss, item, ' ') )
    arguments.push_back(item);

  std::vector<char*> argv;
  for (const auto& arg : arguments)
    argv.push_back((char*)arg.data());
  argv.push_back(nullptr);

  // Create simulation environment
  Simulation *_environment = new Simulation(argv.size() - 1, argv.data(), comm);
  _environment->init();

  // Get environment's dump frequency
  _environment->sim.dumpTime = s["Custom Settings"]["Dump Frequency"].get<double>();

  // Set initial state
  _environment->advance(_environment->calcMaxTimestep()); //perform one timestep so that returned state does not have NaNs
  CylinderNozzle * cylinder = dynamic_cast<CylinderNozzle *> (_environment->getShapes()[0].get());
  if( nAgents > 1 )
  {
    std::vector<std::vector<double>> states(nAgents);
    for( int i = 0; i<nAgents; i++ )
    {
      states[i] = cylinder->state(i);
    }
    s["State"] = states;
  }
  else
  {
    s["State"] = cylinder->state(0);
  }

  // Variables for time and step conditions
  double t = _environment->sim.time; // Current time
  size_t curStep = 0;                // current Step
  double dtAct;                      // Time until next action
  double tNextAct = t;               // Time of next action

  // Setting maximum number of steps before truncation
  const size_t maxSteps = ( s["Mode"] == "Training" ) ? 500 : 5000; //simulate 500*0.1 = 50T

  // Container for actions
  std::vector<std::vector<double>> actions(nAgents, std::vector<double>(NUMACTIONS));

  // Starting main environment loop
  bool done = false;
  std::vector<char> bFailed(nAgents, false);
  while ( curStep < maxSteps && done == false )
  {
      
    // Only rank 0 communicates with Korali
    if( rank == 0 )
    {
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
      cylinder->act(actions[i],i);
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
    //dtAct = 1.0;
    dtAct = 0.10;
    tNextAct += dtAct;
    tNextAct = std::max(5.0,tNextAct);
    while ( t < tNextAct && done == false )
    {
      const double dt = std::min(_environment->calcMaxTimestep(), dtAct);
      t += dt;
      _environment->advance(dt);
      done = done || dt < 1e-6;
    }

    // Get and store state and reward [Carful, state function needs to be called by all ranks!] 
    if( nAgents > 1 ) {
      std::vector<std::vector<double>> states(nAgents);
      std::vector<double> rewards(nAgents);
      for( int i = 0; i<nAgents; i++ )
      {
        std::vector<double> state = cylinder->state(i);
        states[i]  = state;
        rewards[i] = done ? -100.0 : cylinder->reward(i);
      }
      s["State"]  = states;
      s["Reward"] = rewards;
    }
    else {
      s["State"] = cylinder->state(0);
      s["Reward"] = done ? -100.0 : cylinder->reward(0);
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
  if( rank == 0 ) fclose(logFile);

  // delete simulation class
  delete _environment;

  // Switching back to experiment directory
  fs::current_path(curPath);
}
