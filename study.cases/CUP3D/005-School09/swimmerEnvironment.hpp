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

#define STATES 23
#define AGENTS 8
#define ACTIONS 5

std::vector<std::vector<double>> actual_initialPositions;
#if modelDIM == 3
  using namespace cubismup3d;
#endif

std::mt19937 _randomGenerator;

void runEnvironment(korali::Sample &s);

namespace
{

Simulation * initializeEnvironment(korali::Sample &s)
{
    MPI_Comm comm = *(MPI_Comm*) korali::getWorkerMPIComm();
    int rank;
    MPI_Comm_rank(comm,&rank);

    std::vector<std::vector<double>> initialPositions{{
          {0.6, 1.0, 0.8},
          {1.0, 0.8, 0.8},
          {1.0, 1.2, 0.8},
          {1.4, 1.0, 0.8},
          {0.6, 1.0, 1.2},
          {1.0, 0.8, 1.2},
          {1.0, 1.2, 1.2},
          {1.4, 1.0, 1.2}}};

    // Argument string to inititialize Simulation
    #if modelDIM == 2
        std::string OPTIONS = "-bpdx 2 -bpdy 2 -extent 2.0 -CFL 0.7 ";
	OPTIONS += "-levelMax 7 -levelStart 5 -Rtol 5.0 -Ctol 0.1 -nu 0.00001 ";
        OPTIONS += "-poissonTol 1e-5 -poissonTolRel 1e-2 -bMeanConstraint 1 -poissonSolver cuda_iterative";
        std::string AGENT = " \n\
        stefanfish L=0.2 T=1 bFixed=1 ";
    #else
	//training
        std::string OPTIONS = "-bpdx 2 -bpdy 2 -bpdz 2 -extentx 2.0 -CFL 0.7 ";
	OPTIONS += "-levelMax 6 -levelStart 4 -Rtol 5.0 -Ctol 0.1 -nu 0.00001 ";
        OPTIONS += "-poissonTol 1e-5 -poissonTolRel 1e-2 -bMeanConstraint 1";
	//gpu-testing
        //std::string OPTIONS = "-bpdx 2 -bpdy 2 -bpdz 2 -extentx 2.0 -CFL 0.7 ";
        //OPTIONS += "-levelMax 8 -levelStart 4 -Rtol 5.0 -Ctol 0.1 -nu 0.00001 ";
        //OPTIONS += "-poissonTol 1e-6 -poissonTolRel 1e-3 -bMeanConstraint 1 ";
        //OPTIONS += "-poissonSolver cuda_iterative";
        std::string AGENT = " \n\
        StefanFish L=0.2 T=1 bFixFrameOfRef=1 ";
    #endif
    std::string argumentString = "CUP-RL " + OPTIONS + " -shapes ";

    // Set initial position for all prey
    for( int a = 0; a < AGENTS; a++ )
    {
        std::vector<double> initialData = initialPositions[a];
        if ( s["Mode"] == "Training" ) // During training, add noise to initial positions
        {
            if (rank == 0) // only rank 0 samples initial data and broadcasts it
            {
                std::uniform_real_distribution<double> disX(-0.05, 0.05);
                std::uniform_real_distribution<double> disY(-0.05, 0.05);
                std::uniform_real_distribution<double> disZ(-0.05, 0.05);
                initialData[0] += disX(_randomGenerator);
                initialData[1] += disY(_randomGenerator);
                initialData[2] += disZ(_randomGenerator);
            }
            MPI_Bcast(initialData.data(), 3, MPI_DOUBLE, 0, comm);
        }
        actual_initialPositions.push_back(initialData);
        // Append agent to argument string
        argumentString += AGENT + " xpos=" + std::to_string(initialData[0]) + " ypos=" + std::to_string(initialData[1]);
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
    #else
        ArgumentParser parser(argv.size()-1, argv.data());
        Simulation *_environment = new Simulation(comm, parser);
    #endif
    // Establishing environment's dump frequency
    _environment->sim.dumpTime = s["Custom Settings"]["Dump Frequency"].get<double>();

    return _environment;
}

bool isTerminal(StefanFish *agent, const SimulationData & sim, const int agentID, const size_t step)
{
    const size_t maxSteps = 200; // Max steps before truncation (=100 periods)

    if (step >= maxSteps || sim.bCollision) return true;

    #if modelDIM == 2
      const double length = agent->length;
      const double margin = length;
      const double X = agent->center[0];
      const double Y = agent->center[1];
    #else
      const auto & myFish = agent->myFish;
      auto * const cFish = dynamic_cast<CurvatureDefinedFishData*>( myFish );
      const double length = cFish->length;
      const double margin = length;
      const double X = agent->position[0];
      const double Y = agent->position[1];
      const double Z = agent->position[2];
      const double zMin = margin;
      const double zMax = sim.extents[2]-margin;
      if (Z < zMin) return true;
      if (Z > zMax) return true;
    #endif
    const double xMin = margin;
    const double yMin = margin;
    const double xMax = sim.extents[0]-margin;
    const double yMax = sim.extents[1]-margin;
    if (X < xMin) return true;
    if (X > xMax) return true;
    if (Y < yMin) return true;
    if (Y > yMax) return true;
    return false;
}

double getReward(std::vector<StefanFish *> & agents, const SimulationData & sim, const int agentID)
{
    // I) Check if this fish collided and return penalty only if it did.
    if (sim.bCollision) 
        for (size_t i = 0; i < sim.bCollisionID.size(); i++)
            if (sim.bCollisionID[i] == agentID) return -10.0;

    // II) Return penalty if fish exited the domain.
    #if modelDIM == 2
      const double length = agents[agentID]->length;
      const double X = agents[agentID]->center[0];
      const double Y = agents[agentID]->center[1];
      const double margin = length;
    #else
      const auto & myFish = agents[agentID]->myFish;
      auto * const cFish = dynamic_cast<CurvatureDefinedFishData*>( myFish );
      const double length = cFish->length;
      const double margin = length;
      const double X = agents[agentID]->position[0];
      const double Y = agents[agentID]->position[1];
      const double Z = agents[agentID]->position[2];
      const double zMin = margin;
      const double zMax = sim.extents[2]-margin;
      if (Z < zMin) return -10;
      if (Z > zMax) return -10;
    #endif
    const double xMin = margin;
    const double yMin = margin;
    const double xMax = sim.extents[0]-margin;
    const double yMax = sim.extents[1]-margin;
    if (X < xMin) return -10;
    if (X > xMax) return -10;
    if (Y < yMin) return -10;
    if (Y > yMax) return -10;

    // III) Reward swimming efficiency otherwise.
    return agents[agentID]->EffPDefBnd;
}

std::vector<double> getState(std::vector<StefanFish *> & agents, const SimulationData & sim, const int agentID)
{
    const StefanFish * agent = agents[agentID];
    std::vector<double> S(14);
    #if modelDIM == 3
	const auto & myFish = agent->myFish;
        auto * const cFish = dynamic_cast<CurvatureDefinedFishData*>( myFish );
        assert( cFish != nullptr);
        const double length  = cFish->length;
        const double Tperiod = cFish->Tperiod;
        S[0 ] = agent->position[0];
        S[1 ] = agent->position[1];
        S[2 ] = agent->position[2];
        S[3 ] = agent->quaternion[0];
        S[4 ] = agent->quaternion[1];
        S[5 ] = agent->quaternion[2];
        S[6 ] = agent->quaternion[3];
        S[7 ] = agent->getPhase(sim.time);
        S[8 ] = agent->transVel[0] * Tperiod / length;
        S[9 ] = agent->transVel[1] * Tperiod / length;
        S[10] = agent->transVel[2] * Tperiod / length;
        S[11] = agent->angVel[0] * Tperiod;
        S[12] = agent->angVel[1] * Tperiod;
        S[13] = agent->angVel[2] * Tperiod;
    #else
    #endif

    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> z;
    std::vector<double> d;
    for (int i = 0 ; i < (int)agents.size(); i++)
    {
        if (i == agentID) continue;
        #if modelDIM == 2
            const double xfish = agents[i]->center[0]-agents[agentID]->center[0];
            const double yfish = agents[i]->center[1]-agents[agentID]->center[1];
            const double zfish = 0.0;
        #else
            const double xfish = agents[i]->position[0]-agents[agentID]->position[0];
            const double yfish = agents[i]->position[1]-agents[agentID]->position[1];
            const double zfish = agents[i]->position[2]-agents[agentID]->position[2];
        #endif
        x.push_back(xfish);
        y.push_back(yfish);
        z.push_back(zfish);
        d.push_back(std::sqrt(xfish*xfish+yfish*yfish+zfish*zfish));
    }
    std::vector<int> indices(d.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](int A, int B) -> bool {return d[A] < d[B];});
    
    for (int i = 0 ; i < 3; i++)//distance from three closest fish saved
    {
        S.push_back(x[indices[i]]);
        S.push_back(y[indices[i]]);
        S.push_back(z[indices[i]]);
    }
    if (S.size() != STATES)
    {
       std::cerr << "wrong state size " << S.size() << "->" << STATES << std::endl;
       abort();
    }
    return S;
}

void takeAction(StefanFish *agent, const SimulationData & sim, const int agentID, const std::vector<double> & action, const double l_tnext)
{
    #if modelDIM == 3
        auto * const cFish = dynamic_cast<CurvatureDefinedFishData*>( agent->myFish );
        cFish->action_curvature(sim.time,l_tnext, action[0]);
        cFish->action_period   (sim.time,l_tnext, action[1]);
        cFish->action_torsion  (sim.time,l_tnext,&action[2]);
    #else
        agent->act(l_tnext,action);
    #endif
}

#ifndef EVALUATION
void setupRL(korali::Experiment & e)
{
    e["Problem"]["Agents Per Environment"] = AGENTS;
    e["Problem"]["Policies Per Environment"] = 1;

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
    e["Variables"][curVariable]["Name"] = "Torsion point 0";
    e["Variables"][curVariable]["Type"] = "Action";
    e["Variables"][curVariable]["Lower Bound"] = -0.5;
    e["Variables"][curVariable]["Upper Bound"] = +0.5;
    e["Variables"][curVariable]["Initial Exploration Noise"] = 0.2;
    curVariable++;
    e["Variables"][curVariable]["Name"] = "Torsion point 1";
    e["Variables"][curVariable]["Type"] = "Action";
    e["Variables"][curVariable]["Lower Bound"] = -0.5;
    e["Variables"][curVariable]["Upper Bound"] = +0.5;
    e["Variables"][curVariable]["Initial Exploration Noise"] = 0.2;
    curVariable++;
    e["Variables"][curVariable]["Name"] = "Torsion point 2";
    e["Variables"][curVariable]["Type"] = "Action";
    e["Variables"][curVariable]["Lower Bound"] = -0.5;
    e["Variables"][curVariable]["Upper Bound"] = +0.5;
    e["Variables"][curVariable]["Initial Exploration Noise"] = 0.2;

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
 
    e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear";
    e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = 64;
    e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation";
    e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/Tanh";

    e["Solver"]["Neural Network"]["Hidden Layers"][4]["Type"] = "Layer/Linear";
    e["Solver"]["Neural Network"]["Hidden Layers"][4]["Output Channels"] = 64;
    e["Solver"]["Neural Network"]["Hidden Layers"][5]["Type"] = "Layer/Activation";
    e["Solver"]["Neural Network"]["Hidden Layers"][5]["Function"] = "Elementwise/Tanh";
}
#endif
}
