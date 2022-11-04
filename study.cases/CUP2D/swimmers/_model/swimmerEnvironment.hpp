//  Korali environment for CubismUP-2D
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.

#include "korali.hpp"
#include <algorithm>
#include <random>
#include "Obstacles/StefanFish.h"
#include "Simulation.h"
#include "Utils/BufferedLogger.h"

// #define STEFANS_SENSORS_STATE

// command line arguments are read in Korali application
extern int _argc;
extern char **_argv;

void runEnvironment(korali::Sample &s);

bool isTerminal(StefanFish *agent, 
                const int nAgents, 
                const int task);

std::vector<double> getState(StefanFish *agent, 
                             const std::vector<double>& origin,
                             const SimulationData & sim,
                             const int nAgents,
                             const int agentID,
                             const int task);
