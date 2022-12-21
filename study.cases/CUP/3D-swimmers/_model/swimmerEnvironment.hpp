//  Korali environment for CubismUP-3D
//  Copyright (c) 2022 CSE-Lab, ETH Zurich, Switzerland.

#include "korali.hpp"
#include <algorithm>
#include <random>
#include <filesystem>
#include <iostream>
#include <fstream>
#include "Obstacles/StefanFish.h"
#include "Simulation.h"
#include "Utils/BufferedLogger.h"
#include <Cubism/ArgumentParser.h>

using namespace cubismup3d;

// command line arguments are read in Korali application
extern int _argc;
extern char **_argv;

void runEnvironment(korali::Sample &s);

bool isTerminal(StefanFish *agent);

std::vector<double> getState(StefanFish *agent, const std::vector<double>& origin, const SimulationData & sim);