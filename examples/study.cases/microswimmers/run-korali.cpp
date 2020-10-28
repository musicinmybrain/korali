#include "_environment/environment.hpp"
#include "korali.hpp"

int main(int argc, char *argv[])
{
  initializeEnvironment("_deps/msode/launch_scripts/rl/config/helix_2d_eu_const.json");

  auto [lowerBounds, upperBounds] = _environment->getActionBounds();

  auto e = korali::Experiment();

  e["Problem"]["Type"] = "Reinforcement Learning / Continuous";
  e["Problem"]["Environment Function"] = &runEnvironment;
  e["Problem"]["Action Repeat"] = 1;
  e["Problem"]["Actions Between Policy Updates"] = 1;

  e["Variables"][0]["Name"] = "Swimmer 1 - Pos X";
  e["Variables"][0]["Type"] = "State";
  e["Variables"][1]["Name"] = "Swimmer 1 - Pos Y";
  e["Variables"][1]["Type"] = "State";
  e["Variables"][2]["Name"] = "Swimmer 1 - Pos Z";
  e["Variables"][2]["Type"] = "State";
  e["Variables"][3]["Name"] = "Swimmer 1 - Quaternion X";
  e["Variables"][3]["Type"] = "State";
  e["Variables"][4]["Name"] = "Swimmer 1 - Quaternion Y";
  e["Variables"][4]["Type"] = "State";
  e["Variables"][5]["Name"] = "Swimmer 1 - Quaternion Z";
  e["Variables"][5]["Type"] = "State";
  e["Variables"][6]["Name"] = "Swimmer 1 - Quaternion W";
  e["Variables"][6]["Type"] = "State";

  e["Variables"][7]["Name"] = "Swimmer 2 - Pos X";
  e["Variables"][7]["Type"] = "State";
  e["Variables"][8]["Name"] = "Swimmer 2 - Pos Y";
  e["Variables"][8]["Type"] = "State";
  e["Variables"][9]["Name"] = "Swimmer 2 - Pos Z";
  e["Variables"][9]["Type"] = "State";
  e["Variables"][10]["Name"] = "Swimmer 2 - Quaternion X";
  e["Variables"][10]["Type"] = "State";
  e["Variables"][11]["Name"] = "Swimmer 2 - Quaternion Y";
  e["Variables"][11]["Type"] = "State";
  e["Variables"][12]["Name"] = "Swimmer 2 - Quaternion Z";
  e["Variables"][12]["Type"] = "State";
  e["Variables"][13]["Name"] = "Swimmer 2 - Quaternion W";
  e["Variables"][13]["Type"] = "State";

  e["Variables"][14]["Name"] = "Frequency (w)";
  e["Variables"][14]["Type"] = "Action";
  e["Variables"][14]["Lower Bound"] = lowerBounds[0];
  e["Variables"][14]["Upper Bound"] = upperBounds[0];
  e["Variables"][14]["Exploration Sigma"] = (upperBounds[0] - lowerBounds[0]) * 0.05;

  e["Variables"][15]["Name"] = "Rotation X";
  e["Variables"][15]["Type"] = "Action";
  e["Variables"][15]["Lower Bound"] = lowerBounds[1];
  e["Variables"][15]["Upper Bound"] = upperBounds[1];
  e["Variables"][15]["Exploration Sigma"] = (upperBounds[1] - lowerBounds[1]) * 0.01;

  e["Variables"][16]["Name"] = "Rotation Y";
  e["Variables"][16]["Type"] = "Action";
  e["Variables"][16]["Lower Bound"] = lowerBounds[2];
  e["Variables"][16]["Upper Bound"] = upperBounds[2];
  e["Variables"][16]["Exploration Sigma"] = (upperBounds[2] - lowerBounds[2]) * 0.01;

  e["Variables"][17]["Name"] = "Rotation Z";
  e["Variables"][17]["Type"] = "Action";
  e["Variables"][17]["Lower Bound"] = lowerBounds[3];
  e["Variables"][17]["Upper Bound"] = upperBounds[3];
  e["Variables"][17]["Exploration Sigma"] = (upperBounds[3] - lowerBounds[3]) * 0.01;

  //// Defining Agent Configuration

  e["Solver"]["Type"] = "Agent / Continuous / GFPT";
  e["Solver"]["Experiences Between Agent Trainings"] = 1;
  e["Solver"]["Experiences Between Target Network Updates"] = 1;
  e["Solver"]["Optimization Steps Per Update"] = 1;

  e["Solver"]["Random Action Probability"]["Initial Value"] = 0.5;
  e["Solver"]["Random Action Probability"]["Target Value"] = 0.01;
  e["Solver"]["Random Action Probability"]["Decrease Rate"] = 0.05;

  e["Solver"]["Experience Replay"]["Start Size"] = 1000;
  e["Solver"]["Experience Replay"]["Maximum Size"] = 100000;

  //// Defining Critic Configuration

  e["Solver"]["Critic"]["Learning Rate"] = 0.001;
  e["Solver"]["Critic"]["Discount Factor"] = 0.99;
  e["Solver"]["Critic"]["Mini Batch Size"] = 64;

  e["Solver"]["Critic"]["Neural Network"]["Layers"][0]["Type"] = "Layer/Dense";
  e["Solver"]["Critic"]["Neural Network"]["Layers"][0]["Activation Function"]["Type"] = "Elementwise/Linear";
  e["Solver"]["Critic"]["Neural Network"]["Layers"][0]["Batch Normalization"]["Enabled"] = true;

  e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Type"] = "Layer/Dense";
  e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Node Count"] = 128;
  e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Activation Function"]["Type"] = "Elementwise/Tanh";
  e["Solver"]["Critic"]["Neural Network"]["Layers"][1]["Batch Normalization"]["Enabled"] = true;

  e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Type"] = "Layer/Dense";
  e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Node Count"] = 128;
  e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Activation Function"]["Type"] = "Elementwise/Tanh";
  e["Solver"]["Critic"]["Neural Network"]["Layers"][2]["Batch Normalization"]["Enabled"] = true;

  e["Solver"]["Critic"]["Neural Network"]["Layers"][3]["Type"] = "Layer/Dense";
  e["Solver"]["Critic"]["Neural Network"]["Layers"][3]["Activation Function"]["Type"] = "Elementwise/Linear";
  e["Solver"]["Critic"]["Neural Network"]["Layers"][3]["Batch Normalization"]["Enabled"] = true;

  //// Defining Policy Configuration

  e["Solver"]["Policy"]["Learning Rate"] = 0.001;
  e["Solver"]["Policy"]["Mini Batch Size"] = 32;

  e["Solver"]["Policy"]["Neural Network"]["Layers"][0]["Type"] = "Layer/Dense";
  e["Solver"]["Policy"]["Neural Network"]["Layers"][0]["Activation Function"]["Type"] = "Elementwise/Linear";

  e["Solver"]["Policy"]["Neural Network"]["Layers"][1]["Type"] = "Layer/Dense";
  e["Solver"]["Policy"]["Neural Network"]["Layers"][1]["Node Count"] = 128;
  e["Solver"]["Policy"]["Neural Network"]["Layers"][1]["Activation Function"]["Type"] = "Elementwise/Tanh";

  e["Solver"]["Policy"]["Neural Network"]["Layers"][2]["Type"] = "Layer/Dense";
  e["Solver"]["Policy"]["Neural Network"]["Layers"][2]["Node Count"] = 128;
  e["Solver"]["Policy"]["Neural Network"]["Layers"][2]["Activation Function"]["Type"] = "Elementwise/Tanh";

  e["Solver"]["Policy"]["Neural Network"]["Layers"][3]["Type"] = "Layer/Dense";
  e["Solver"]["Policy"]["Neural Network"]["Layers"][3]["Activation Function"]["Type"] = "Elementwise/Tanh";

  e["Solver"]["Policy"]["Neural Network"]["Output Scaling"] = {
    (upperBounds[0] - lowerBounds[0]) * 0.5,
    (upperBounds[1] - lowerBounds[1]) * 0.5,
    (upperBounds[2] - lowerBounds[2]) * 0.5,
    (upperBounds[3] - lowerBounds[3]) * 0.5};

  e["Solver"]["Policy"]["Neural Network"]["Output Shift"] = {
    (upperBounds[0] + lowerBounds[0]) * 0.5,
    (upperBounds[1] + lowerBounds[1]) * 0.5,
    (upperBounds[2] + lowerBounds[2]) * 0.5,
    (upperBounds[3] + lowerBounds[3]) * 0.5};

  ////// Defining Termination Criteria

  e["Solver"]["Termination Criteria"]["Max Generations"] = 6;
  e["Solver"]["Training Reward Threshold"] = 1.0;
  e["Solver"]["Policy Testing Episodes"] = 20;
  e["Solver"]["Termination Criteria"]["Target Average Testing Reward"] = 1.3;

  ////// Setting file output configuration

  e["File Output"]["Enabled"] = false;

  ////// Running Experiment

  auto k = korali::Engine();
  k.run(e);
}
