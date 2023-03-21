//  Korali environment for CubismUP-2D
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.

#include "korali.hpp"
#include <algorithm>
#include <random>
#include "Obstacles/CylinderNozzle.h"
#include "Simulation.h"
#include "Utils/BufferedLogger.h"

// command line arguments are read in Korali application
extern int _argc;
extern char **_argv;

const size_t NUMACTIONS = 8;

void runEnvironment(korali::Sample &s);
