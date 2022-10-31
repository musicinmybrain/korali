//  Korali environment for CubismUP_2D For Fish Following Experiment
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.

#include "korali.hpp"
#include <fstream>
#include <random>
#include <type_traits>

void runEnvironment(korali::Sample &s);
void initializeEnvironment(const std::string confFileName);

extern std::string _resultDir;
extern bool _isTraining;
extern std::mt19937 _randomGenerator;

#include "msode/core/factory.h"
#include "msode/core/file_parser.h"
#include "msode/core/log.h"
#include "msode/rl/environment.h"
#include "msode/rl/factory.h"

using namespace msode;
using namespace msode::rl;
using namespace msode::factory;

// Global variables for the simulation (ideal if this would be a class instead)
extern std::unique_ptr<msode::rl::MSodeEnvironment> _environment;

using Status = typename std::remove_pointer<decltype(_environment.get())>::type::Status;
