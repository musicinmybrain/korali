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
    std::string OPTIONS = " -bpdx 8 -bpdy 8 -extent 8.0 -levelMax 7 -levelStart 4 -Rtol 10000.0 -Ctol 100.0 -CFL 0.7 -poissonTol 1e-5 -poissonTolRel 1e-2 -nu 0.00001 -bMeanConstraint 0 -bAdaptChiGradient 0 -poissonSolver cuda_iterative";
    std::vector<std::vector<double>> actual_initialPositions;
#else
    std::string OPTIONS = " -bpdx 2 -bpdy 2 -bpdz 2 -extentx 2.0 -levelMax 6 -levelStart 5 -Rtol 10000.0 -Ctol 100.0 -CFL 0.7 -poissonTol 1e-5 -poissonTolRel 1e-2 -nu 0.00001 -bMeanConstraint 0 -bAdaptChiGradient 0 ";
    std::vector<std::vector<double>> initialPositions{{{1.8,1.0,1.0},{0.8,1.0,1.0}}};
    std::vector<std::vector<double>> actual_initialPositions;
    using namespace cubismup3d;
#endif

#define ACTIONS 2
#define STATES 5
#define AGENTS 2

std::mt19937 _randomGenerator;

void runEnvironment(korali::Sample &s);

namespace
{

Simulation * initializeEnvironment(korali::Sample &s)
{
    MPI_Comm comm = *(MPI_Comm*) korali::getWorkerMPIComm();
    int rank;
    MPI_Comm_rank(comm,&rank);

    std::string argumentString = "CUP-RL " + OPTIONS + " -shapes ";
    std::string PREDATOR = " \n\
    stefanfish L=0.225 T=1";
    std::string AGENT = " \n\
    stefanfish L=0.200 T=1";

    argumentString += PREDATOR + " xpos=7.0 ypos=4.0";
    #if modelDIM == 3
        argumentString += " zpos=4.0 heightProfile=danio widthProfile=stefan";
    #endif

    // Set initial position for all prey
    for( int a = 1; a < AGENTS; a++ )
    {
        std::vector<double> initialData(3);
        initialData[0] = 2.0;
        initialData[1] = 4.0;
        initialData[2] = 0.0;

        if (rank == 0) // only rank 0 samples initial data and broadcasts it
        {
            std::uniform_real_distribution<double> disX(-1.00, +1.00);
            std::uniform_real_distribution<double> disY(-2.00, +2.00);
            std::uniform_real_distribution<double> disA(-25. / 180. * M_PI, 25. / 180. * M_PI);
            initialData[0] += disX(_randomGenerator);
            initialData[1] += disY(_randomGenerator);
            initialData[2]  = disA(_randomGenerator);
        }
        MPI_Bcast(initialData.data(), 3, MPI_DOUBLE, 0, comm);
        actual_initialPositions.push_back(initialData);
        argumentString += AGENT + " xpos="  + std::to_string(initialData[0])
                                + " ypos="  + std::to_string(initialData[1])
                                + " angle=" + std::to_string(initialData[2]);
        #if modelDIM == 3
            argumentString += " zpos=" + std::to_string(initialData[2]) + " heightProfile=danio widthProfile=stefan";
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
        _environment->sim.dumpTime = (s["Mode"] == "Training") ? 0.0 : 0.1;    
    #else
        ArgumentParser parser(argv.size()-1, argv.data());
        Simulation *_environment = new Simulation(comm, parser);
        _environment->sim.saveTime = (s["Mode"] == "Training") ? 0.0 : 0.1;
    #endif

    return _environment;
}

bool isTerminal(StefanFish *agent, const SimulationData & sim, const int agentID, const size_t step)
{
    const size_t maxSteps = 1000; // Max steps before truncation

    if (step >= maxSteps) return true;

    if (sim.bCollision) return true;

    const double xMin = 0.1;
    const double xMax = 7.9;
    const double yMin = 0.1;
    const double yMax = 7.9;
    const double zMin = 0.1;
    const double zMax = 7.9;
    #if modelDIM == 2
        const double X = agent->center[0];
        const double Y = agent->center[1];
        const double Z = 1.0;
    #endif
    #if modelDIM == 3
        const double X = agent->absPos[0];
        const double Y = agent->absPos[1];
        const double Z = agent->absPos[2];
    #endif
    if (X < xMin) return true;
    if (X > xMax) return true;
    if (Y < yMin) return true;
    if (Y > yMax) return true;
    if (Z < zMin) return true;
    if (Z > zMax) return true;
    return false;
}

double getReward(std::vector<StefanFish *> & agents, const SimulationData & sim, const int agentID)
{
    #if modelDIM == 2
        const double Xpredator = agents[0]->center[0];
        const double Ypredator = agents[0]->center[1];
        const double Zpredator = 1.0;
        const double Xprey     = agents[1]->center[0];
        const double Yprey     = agents[1]->center[1];
        const double Zprey     = 1.0;
    #endif

    #if modelDIM == 3
        const double Xpredator = agents[0]->absPos[0];
        const double Ypredator = agents[0]->absPos[1];
        const double Zpredator = agents[0]->absPos[2];
        const double Xprey     = agents[1]->absPos[0];
        const double Yprey     = agents[1]->absPos[1];
        const double Zprey     = agents[1]->absPos[2];
    #endif

    const double d     = std::pow((Xpredator-Xprey)*(Xpredator-Xprey)+
                                  (Ypredator-Yprey)*(Ypredator-Yprey)+
                                  (Zpredator-Zprey)*(Zpredator-Zprey),0.5);

    const double xMin = 0.1;
    const double xMax = 7.9;
    const double yMin = 0.1;
    const double yMax = 7.9;
    const double zMin = 0.1;
    const double zMax = 7.9;
    #if modelDIM == 2
        const double X = agents[agentID]->center[0];
        const double Y = agents[agentID]->center[1];
        const double Z = 1.0;
    #endif
    #if modelDIM == 3
        const double X = agents[agentID]->absPos[0];
        const double Y = agents[agentID]->absPos[1];
        const double Z = agents[agentID]->absPos[2];
    #endif
    if (X < xMin) return -10;
    if (X > xMax) return -10;
    if (Y < yMin) return -10;
    if (Y > yMax) return -10;
    if (Z < zMin) return -10;
    if (Z > zMax) return -10;

    if (agentID == 0) //predator
    {
        if (sim.bCollision) return 1000;
        return -d;
    }
    else
    {
        if (sim.bCollision) return -1000;
        return d;
    }
}

std::vector<double> getState(std::vector<StefanFish *> & agents, const SimulationData & sim, const int agentID)
{
    std::vector<double> S(STATES);

    #if modelDIM == 3
        MPI_Abort(MPI_COMM_WORLD,666);
        //const double q[4] = {agents[agentID]->quaternion[0],agents[agentID]->quaternion[1],
        //                     agents[agentID]->quaternion[2],agents[agentID]->quaternion[3]};
        //const double norm = sqrt(q[1]*q[1]+q[2]*q[2]+q[3]*q[3])+1e-21;
        //double ax   = q[1]/norm;
        //double ay   = q[2]/norm;
        //double az   = q[3]/norm;
        //double th   = 2.0*atan2(norm,q[0]);
        //if (norm < 1e-20)
        //{
        //    ax = 0;
        //    ay = 0;
        //    az = 1;
        //    th = 0;
        //}
        //S[0] = agents[agentID]->absPos[0];
        //S[1] = agents[agentID]->absPos[1];
        //S[2] = agents[agentID]->absPos[2];
        //S[3] = ax;
        //S[4] = ay;
        //S[5] = az;
        //S[6] = th;
        //S[7] = agents[1 - agentID]->absPos[0];
        //S[8] = agents[1 - agentID]->absPos[1];
        //S[9] = agents[1 - agentID]->absPos[2];
    #endif

    #if modelDIM == 2
        S[0] = agents[agentID]->center[0];
        S[1] = agents[agentID]->center[1];
        S[2] = agents[agentID]->getOrientation();
        S[3] = agents[1 - agentID]->center[0];
        S[4] = agents[1 - agentID]->center[1];
    #endif

    return S;
}

void takeAction(StefanFish *agent, const SimulationData & sim, const int agentID, const std::vector<double> & action, const double l_tnext)
{
    #if modelDIM == 3
        auto * const cFish = dynamic_cast<CurvatureDefinedFishData*>( agent->myFish );
        cFish->action_curvature(sim.time,l_tnext, action[0]);
        cFish->action_period   (sim.time,l_tnext, action[1]);
    #else
        agent->act(l_tnext,action);
    #endif
}

#ifndef EVALUATION
void setupRL(korali::Experiment & e)
{
    e["Problem"]["Agents Per Environment"] = AGENTS;
    e["Problem"]["Policies Per Environment"] = AGENTS;
    e["Solver"]["Multi Agent Relationship"] = "Competition";

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

    e["Solver"]["Experiences Between Policy Updates"] = 1;
    e["Solver"]["Learning Rate"] = 1e-4;
    e["Solver"]["Discount Factor"] = 0.95;
    e["Solver"]["Mini Batch"]["Size"] =  128;
 
    // Configuration of replay memory
    e["Solver"]["Experience Replay"]["Start Size"] = 1024;
    e["Solver"]["Experience Replay"]["Maximum Size"] = 65536;
    e["Solver"]["Experience Replay"]["Off Policy"]["Annealing Rate"] = 5.0e-8;
    e["Solver"]["Experience Replay"]["Off Policy"]["Cutoff Scale"] = 5.0;
    e["Solver"]["Experience Replay"]["Off Policy"]["REFER Beta"] = 0.3;
    e["Solver"]["Experience Replay"]["Off Policy"]["Target"] = 0.1;

    // Neural Network
    e["Solver"]["Neural Network"]["Engine"] = "OneDNN";
    e["Solver"]["Neural Network"]["Optimizer"] = "Adam";
 
    e["Solver"]["L2 Regularization"]["Enabled"] = true;
    e["Solver"]["L2 Regularization"]["Importance"] = 1.0;
 
    e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear";
    e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 64;
 
    e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation";
    e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/Tanh";

    e["Problem"]["Testing Frequency"] = 512;
    //e["Problem"]["Policy Testing Episodes"] = 4;
}
#endif
}
