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

#if modelDIM == 2
    std::string OPTIONS = "-bpdx 4 -bpdy 4 -levelMax 7 -levelStart 4 -Rtol 10000.0 -Ctol 100.0 -extent 2 -CFL 0.7 -poissonTol 1e-5 -poissonTolRel 0.0 -bMeanConstraint 1 -bAdaptChiGradient 0 -tdump 0 -nu 0.00004 -tend 0 -muteAll 0 -verbose 1";
    std::vector<std::vector<double>> initialPositions{{{1.0, 1.0,1.0}}};
#else
    std::string OPTIONS = " -bpdx 4 -bpdy 4 -bpdz 4 -extentx 2.0 -levelMax 7 -levelStart 4 -Rtol 10000.00 -Ctol 100.00 -fsave 0 -tdump 0 -tend 0 -CFL 0.7 -lambda 1e6 -nu 0.00001 -poissonTol 1e-6 -poissonTolRel 1e-3 -bMeanConstraint 1";
    std::vector<std::vector<double>> initialPositions{{{1.0, 1.0, 1.0}}};
    using namespace cubismup3d;
#endif

#define ACTIONS 3
#define STATES 7
#define AGENTS 1

std::mt19937 _randomGenerator;

void runEnvironment(korali::Sample &s);

namespace
{

Simulation * initializeEnvironment(korali::Sample &s)
{
    MPI_Comm comm = *(MPI_Comm*) korali::getWorkerMPIComm();
    int rank;
    MPI_Comm_rank(comm,&rank);

    // Argument string to inititialize Simulation
    #if modelDIM == 2
        std::string argumentString = "CUP-RL " + OPTIONS + " -shapes ";
        std::string AGENT = " \n\
        stefanfish L=0.2 T=1";
    #else
        std::string argumentString = "CUP-RL " + OPTIONS + " -factory-content ";
        std::string AGENT = " \n\
        StefanFish L=0.2 T=1";
    #endif

    // Set initial position for all agents
    for( int a = 0; a < AGENTS; a++ )
    {
        std::vector<double> initialPosition = initialPositions[a];
        double initialData[3];
        initialData[0] = initialPosition[0];
        initialData[1] = initialPosition[1];
        initialData[2] = initialPosition[2];
        if ( s["Mode"] == "Training" ) // During training, add noise to inital configuration of agent
        {
            if (rank == 0) // only rank 0 samples initial data and broadcasts it
            {
                std::uniform_real_distribution<double> disX(-0.01, 0.01);
                std::uniform_real_distribution<double> disY(-0.01, 0.01);
                initialData[0] = initialData[0] + disX(_randomGenerator);
                initialData[1] = initialData[1] + disY(_randomGenerator);
            }
            MPI_Bcast(initialData, 3, MPI_DOUBLE, 0, comm);
        }
        // Append agent to argument string
        #if modelDIM == 2
            argumentString = argumentString + AGENT + " xpos=" + std::to_string(initialData[0]) + " ypos=" + std::to_string(initialData[1]);
        #else
            argumentString = argumentString + AGENT + " xpos=" + std::to_string(initialData[0]) 
                                                    + " ypos=" + std::to_string(initialData[1])
                                                    + " zpos=" + std::to_string(initialData[2])
                                                    + " heightProfile=danio" + " widthProfile=stefan" + " bFixToPlanar=1";
        #endif
    }

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

    #if modelDIM == 2
        Simulation *_environment = new Simulation(argv.size() - 1, argv.data(), comm);
        _environment->init();
    #else
        ArgumentParser parser(argv.size()-1, argv.data());
        Simulation *_environment = new Simulation(comm, parser);
    #endif
    // Establishing environment's dump frequency
    _environment->sim.dumpTime = s["Custom Settings"]["Dump Frequency"].get<double>();

    return _environment;
}

bool isTerminal(StefanFish *agent, const SimulationData & sim, const int agentID)
{
    const double xMin = 0.1;
    const double xMax = 1.9;
    const double yMin = 0.1;
    const double yMax = 1.9;
    const double Xt   = 0.5;
    const double Yt   = 0.5;

    #if modelDIM == 2
        const double X = agent->center[0];
        const double Y = agent->center[1];
    #endif
    #if modelDIM == 3
        const double X = agent->absPos[0];
        const double Y = agent->absPos[1];
    #endif

    const double d = sqrt( (X-Xt)*(X-Xt)+(Y-Yt)*(Y-Yt) );
    if (d < 0.05) return true;
    bool terminal = false;
    if (X < xMin) terminal = true;
    if (X > xMax) terminal = true;
    if (Y < yMin) terminal = true;
    if (Y > yMax) terminal = true;
    return terminal;
}

double getReward(StefanFish *agent, const SimulationData & sim, const int agentID)
{
    #if modelDIM == 2
        const double X = agent->center[0];
        const double Y = agent->center[1];
    #endif

    #if modelDIM == 3
        const double X = agent->absPos[0];
        const double Y = agent->absPos[1];
    #endif

    const double Xt = 0.50;
    const double Yt = 0.50;
    const double d  = std::pow((X -Xt)*(X -Xt) + (Y -Yt)*(Y -Yt),0.5);
    if (d < 0.05) return 20.0;
    return -d;
}

std::vector<double> getState(StefanFish *agent, const SimulationData & sim, const int agentID)
{
    std::vector<double> S(7);

    #if modelDIM == 3
        const double q[4] = {agent->quaternion[0],agent->quaternion[1],agent->quaternion[2],agent->quaternion[3]};
        const double norm = sqrt(q[1]*q[1]+q[2]*q[2]+q[3]*q[3])+1e-21;
        double ax   = q[1]/norm;
        double ay   = q[2]/norm;
        double az   = q[3]/norm;
        double th   = 2.0*atan2(norm,q[0]);
        if (norm < 1e-20)
        {
            ax = 0;
            ay = 0;
            az = 1;
            th = 0;
        }
        S[0] = agent->absPos[0];
        S[1] = agent->absPos[1];
        S[2] = agent->absPos[2];
        S[3] = ax;
        S[4] = ay;
        S[5] = az;
        S[6] = th;
    #endif

    #if modelDIM == 2
        S[0] = agent->center[0];
        S[1] = agent->center[1];
        S[2] = 0.0; //Z = 0
        S[3] = 0.0; //axis x-component = 0
        S[4] = 0.0; //axis y-component = 0
        S[5] = 1.0; //axis z-component = 1
        S[6] = agent->getOrientation();
    #endif

    return S;
}

void takeAction(StefanFish *agent, const SimulationData & sim, const int agentID, const std::vector<double> & action, const double l_tnext)
{
    #if modelDIM == 3
        //three actions defined for compatibility with 3D motion, only two used for planar motion
        auto * const cFish = dynamic_cast<CurvatureDefinedFishData*>( agent->myFish );
        cFish->action_curvature(sim.time,l_tnext,action[0]);
        cFish->action_period   (sim.time,l_tnext,action[1]);
    #endif

    #if modelDIM == 2
        agent->act(l_tnext,action);
    #endif
}

#ifndef EVALUATION
void setupRL(korali::Experiment & e)
{
    e["Problem"]["Agents Per Environment"] = AGENTS;

    size_t curVariable = 0;
    for (; curVariable < STATES; curVariable++)
    {
        e["Variables"][curVariable]["Name"] = std::string("State") + std::to_string(curVariable);
        e["Variables"][curVariable]["Type"] = "State";
    }

    e["Variables"][curVariable]["Name"] = "Curvature";
    e["Variables"][curVariable]["Type"] = "Action";
    e["Variables"][curVariable]["Lower Bound"] = -1.0;
    e["Variables"][curVariable]["Upper Bound"] = +1.0;
    e["Variables"][curVariable]["Initial Exploration Noise"] = 0.50;

    curVariable++;
    e["Variables"][curVariable]["Name"] = "Swimming Period";
    e["Variables"][curVariable]["Type"] = "Action";
    e["Variables"][curVariable]["Lower Bound"] = -0.25;
    e["Variables"][curVariable]["Upper Bound"] = +0.25;
    e["Variables"][curVariable]["Initial Exploration Noise"] = 0.50;

    curVariable++;
    e["Variables"][curVariable]["Name"] = "Pitching Motion";
    e["Variables"][curVariable]["Type"] = "Action";
    e["Variables"][curVariable]["Lower Bound"] = -20*0.2;
    e["Variables"][curVariable]["Upper Bound"] = +20*0.2;
    e["Variables"][curVariable]["Initial Exploration Noise"] = 5*0.2;

    e["Solver"]["Experiences Between Policy Updates"] = 1;
    e["Solver"]["Learning Rate"] = 1e-4;
    e["Solver"]["Discount Factor"] = 0.95;
    e["Solver"]["Mini Batch"]["Size"] =  128;
 
    /// Defining the configuration of replay memory
    e["Solver"]["Experience Replay"]["Start Size"] = 1024;
    e["Solver"]["Experience Replay"]["Maximum Size"] = 65536;
    e["Solver"]["Experience Replay"]["Off Policy"]["Annealing Rate"] = 5.0e-8;
    e["Solver"]["Experience Replay"]["Off Policy"]["Cutoff Scale"] = 5.0;
    e["Solver"]["Experience Replay"]["Off Policy"]["REFER Beta"] = 0.3;
    e["Solver"]["Experience Replay"]["Off Policy"]["Target"] = 0.1;

    //// Defining Neural Network
    e["Solver"]["Neural Network"]["Engine"] = "OneDNN";
    e["Solver"]["Neural Network"]["Optimizer"] = "fAdam";
 
    e["Solver"]["L2 Regularization"]["Enabled"] = true;
    e["Solver"]["L2 Regularization"]["Importance"] = 1.0;
 
    e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear";
    e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 64;
 
    e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation";
    e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/Tanh";
}
#endif
}
