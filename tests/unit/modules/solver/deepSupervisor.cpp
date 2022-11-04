#include "gtest/gtest.h"
#include "korali.hpp"
#include "modules/solver/deepSupervisor/deepSupervisor.hpp"
#include "modules/problem/supervisedLearning/supervisedLearning.hpp"

namespace
{
 using namespace korali;
 using namespace korali::solver;
 using namespace korali::problem;

 //////////////// Deep Supervisor CLASS ////////////////////////

  TEST(learners, deepSupervisor)
  {
   // Creating base experiment
   Experiment e;
   auto& experimentJs = e._js.getJson();
   experimentJs["Variables"][0]["Name"] = "X";
   Variable v;
   e._variables.push_back(&v);

   // Creating optimizer configuration Json
   knlohmann::json supervisorJs;

   // Using a neural network solver (deep learning) for inference

   supervisorJs["Type"] = "DeepSupervisor";
   supervisorJs["Mode"] = "Training";
   supervisorJs["Loss Function"] = "Mean Squared Error";
   supervisorJs["Learning Rate"] = 0.0001;

   // Defining the shape of the neural network

   supervisorJs["Neural Network"]["Engine"] = "Korali";
   supervisorJs["Neural Network"]["Optimizer"] = "fAdam";

   supervisorJs["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear";
   supervisorJs["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 32;

   supervisorJs["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation";
   supervisorJs["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/Tanh";

   supervisorJs["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear";
   supervisorJs["Neural Network"]["Hidden Layers"][2]["Output Channels"] = 32;

   supervisorJs["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation";
   supervisorJs["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/Tanh";

   // Creating module
   DeepSupervisor* supervisor;
   ASSERT_NO_THROW(supervisor = dynamic_cast<DeepSupervisor *>(Module::getModule(supervisorJs, &e)));

   // Defaults should be applied without a problem
   ASSERT_NO_THROW(supervisor->applyModuleDefaults(supervisorJs));

   // Covering variable functions (no effect)
   ASSERT_NO_THROW(supervisor->applyVariableDefaults());

   // Running base routines
   // ASSERT_ANY_THROW(supervisor->getEvaluation({{{0.0f}}}));

   // Backup the correct base configuration
   auto baseOptJs = supervisorJs;
   auto baseExpJs = experimentJs;

   // Setting up optimizer correctly
   supervisorJs = baseOptJs;
   experimentJs = baseExpJs;
   ASSERT_NO_THROW(supervisor->setConfiguration(supervisorJs));

   // Testing optional parameters
   supervisorJs = baseOptJs;
   experimentJs = baseExpJs;
   supervisorJs["Current Loss"] = "Not a Number";
   ASSERT_ANY_THROW(supervisor->setConfiguration(supervisorJs));

   supervisorJs = baseOptJs;
   experimentJs = baseExpJs;
   supervisorJs["Current Loss"] = 1.0;
   ASSERT_NO_THROW(supervisor->setConfiguration(supervisorJs));

   supervisorJs = baseOptJs;
   experimentJs = baseExpJs;
   supervisorJs["Normalization Means"] = "Not a Number";
   ASSERT_ANY_THROW(supervisor->setConfiguration(supervisorJs));

   supervisorJs = baseOptJs;
   experimentJs = baseExpJs;
   supervisorJs["Normalization Means"] = std::vector<float>({1.0f});
   ASSERT_NO_THROW(supervisor->setConfiguration(supervisorJs));

   supervisorJs = baseOptJs;
   experimentJs = baseExpJs;
   supervisorJs["Normalization Variances"] = "Not a Number";
   ASSERT_ANY_THROW(supervisor->setConfiguration(supervisorJs));

   supervisorJs = baseOptJs;
   experimentJs = baseExpJs;
   supervisorJs["Normalization Variances"] = std::vector<float>({1.0f});
   ASSERT_NO_THROW(supervisor->setConfiguration(supervisorJs));

   // Testing mandatory parameters

   supervisorJs = baseOptJs;
   experimentJs = baseExpJs;
   supervisorJs["Neural Network"].erase("Hidden Layers");
   ASSERT_ANY_THROW(supervisor->setConfiguration(supervisorJs));

   supervisorJs = baseOptJs;
   experimentJs = baseExpJs;
   supervisorJs["Neural Network"]["Hidden Layers"] = knlohmann::json();
   ASSERT_NO_THROW(supervisor->setConfiguration(supervisorJs));

   supervisorJs = baseOptJs;
   experimentJs = baseExpJs;
   supervisorJs["Neural Network"].erase("Output Activation");
   ASSERT_ANY_THROW(supervisor->setConfiguration(supervisorJs));

   supervisorJs = baseOptJs;
   experimentJs = baseExpJs;
   supervisorJs["Neural Network"]["Output Activation"] = knlohmann::json();
   ASSERT_NO_THROW(supervisor->setConfiguration(supervisorJs));

   supervisorJs = baseOptJs;
   experimentJs = baseExpJs;
   supervisorJs["Neural Network"].erase("Output Layer");
   ASSERT_ANY_THROW(supervisor->setConfiguration(supervisorJs));

   supervisorJs = baseOptJs;
   experimentJs = baseExpJs;
   supervisorJs["Neural Network"]["Output Layer"] = knlohmann::json();
   ASSERT_NO_THROW(supervisor->setConfiguration(supervisorJs));

   supervisorJs = baseOptJs;
   experimentJs = baseExpJs;
   supervisorJs["Neural Network"].erase("Engine");
   ASSERT_ANY_THROW(supervisor->setConfiguration(supervisorJs));

   supervisorJs = baseOptJs;
   experimentJs = baseExpJs;
   supervisorJs["Neural Network"]["Engine"] = 1.0;
   ASSERT_ANY_THROW(supervisor->setConfiguration(supervisorJs));

   supervisorJs = baseOptJs;
   experimentJs = baseExpJs;
   supervisorJs["Neural Network"]["Engine"] = "Engine";
   ASSERT_NO_THROW(supervisor->setConfiguration(supervisorJs));

   supervisorJs = baseOptJs;
   experimentJs = baseExpJs;
   supervisorJs["Neural Network"].erase("Optimizer");
   ASSERT_ANY_THROW(supervisor->setConfiguration(supervisorJs));

   supervisorJs = baseOptJs;
   experimentJs = baseExpJs;
   supervisorJs["Neural Network"]["Optimizer"] = 1.0;
   ASSERT_ANY_THROW(supervisor->setConfiguration(supervisorJs));

   supervisorJs = baseOptJs;
   experimentJs = baseExpJs;
   supervisorJs["Neural Network"]["Optimizer"] = "fAdam";
   ASSERT_NO_THROW(supervisor->setConfiguration(supervisorJs));

   supervisorJs = baseOptJs;
   experimentJs = baseExpJs;
   supervisorJs.erase("Loss Function");
   ASSERT_ANY_THROW(supervisor->setConfiguration(supervisorJs));

   supervisorJs = baseOptJs;
   experimentJs = baseExpJs;
   supervisorJs["Loss Function"] = 1.0;
   ASSERT_ANY_THROW(supervisor->setConfiguration(supervisorJs));

   supervisorJs = baseOptJs;
   experimentJs = baseExpJs;
   supervisorJs["Loss Function"] = "Direct Gradient";
   ASSERT_NO_THROW(supervisor->setConfiguration(supervisorJs));

   supervisorJs = baseOptJs;
   experimentJs = baseExpJs;
   supervisorJs.erase("Learning Rate");
   ASSERT_ANY_THROW(supervisor->setConfiguration(supervisorJs));

   supervisorJs = baseOptJs;
   experimentJs = baseExpJs;
   supervisorJs["Learning Rate"] = "Not a Number";
   ASSERT_ANY_THROW(supervisor->setConfiguration(supervisorJs));

   supervisorJs = baseOptJs;
   experimentJs = baseExpJs;
   supervisorJs["Learning Rate"] = 0.001;
   ASSERT_NO_THROW(supervisor->setConfiguration(supervisorJs));

   supervisorJs = baseOptJs;
   experimentJs = baseExpJs;
   supervisorJs["L2 Regularization"].erase("Enabled");
   ASSERT_ANY_THROW(supervisor->setConfiguration(supervisorJs));

   supervisorJs = baseOptJs;
   experimentJs = baseExpJs;
   supervisorJs["L2 Regularization"]["Enabled"] = "Not a Number";
   ASSERT_ANY_THROW(supervisor->setConfiguration(supervisorJs));

   supervisorJs = baseOptJs;
   experimentJs = baseExpJs;
   supervisorJs["L2 Regularization"]["Enabled"] = true;
   ASSERT_NO_THROW(supervisor->setConfiguration(supervisorJs));

   supervisorJs = baseOptJs;
   experimentJs = baseExpJs;
   supervisorJs["L2 Regularization"].erase("Importance");
   ASSERT_ANY_THROW(supervisor->setConfiguration(supervisorJs));

   supervisorJs = baseOptJs;
   experimentJs = baseExpJs;
   supervisorJs["L2 Regularization"]["Importance"] = "Not a Number";
   ASSERT_ANY_THROW(supervisor->setConfiguration(supervisorJs));

   supervisorJs = baseOptJs;
   experimentJs = baseExpJs;
   supervisorJs["L2 Regularization"]["Importance"] = 1.0;
   ASSERT_NO_THROW(supervisor->setConfiguration(supervisorJs));

   supervisorJs = baseOptJs;
   experimentJs = baseExpJs;
   supervisorJs.erase("Output Weights Scaling");
   ASSERT_ANY_THROW(supervisor->setConfiguration(supervisorJs));

   supervisorJs = baseOptJs;
   experimentJs = baseExpJs;
   supervisorJs["Output Weights Scaling"] = "Not a Number";
   ASSERT_ANY_THROW(supervisor->setConfiguration(supervisorJs));

   supervisorJs = baseOptJs;
   experimentJs = baseExpJs;
   supervisorJs["Output Weights Scaling"] = 1.0;
   ASSERT_NO_THROW(supervisor->setConfiguration(supervisorJs));

   supervisorJs = baseOptJs;
   experimentJs = baseExpJs;
   supervisorJs["Termination Criteria"].erase("Target Loss");
   ASSERT_ANY_THROW(supervisor->setConfiguration(supervisorJs));

   supervisorJs = baseOptJs;
   experimentJs = baseExpJs;
   supervisorJs["Termination Criteria"]["Target Loss"] = "Not a Number";
   ASSERT_ANY_THROW(supervisor->setConfiguration(supervisorJs));

   supervisorJs = baseOptJs;
   experimentJs = baseExpJs;
   supervisorJs["Termination Criteria"]["Target Loss"] = 1.0;
   ASSERT_NO_THROW(supervisor->setConfiguration(supervisorJs));
  }

} // namespace
