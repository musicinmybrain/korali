#include "gtest/gtest.h"
#include "korali.hpp"
#include "modules/solver/integrator/integrator.hpp"
#include "modules/problem/integration/integration.hpp"
#include "modules/solver/integrator/montecarlo/MonteCarlo.hpp"
#include "modules/solver/integrator/quadrature/Quadrature.hpp"

namespace
{
 using namespace korali;
 using namespace korali::solver::integrator;
 using namespace korali::problem;

 //////////////// Integrator CLASS ////////////////////////

  TEST(samplers, executorClass)
  {
   // Creating base experiment
   Experiment e;
   auto& experimentJs = e._js.getJson();
   
   // Creating initial variable
   Variable v;
   e._variables.push_back(&v);
   e["Variables"][0]["Name"] = "Var 1";
   e["Variables"][0]["Lower Bound"] = 0.0;
   e["Variables"][0]["Upper Bound"] = 1.0;
   e["Variables"][0]["Number Of Gridpoints"] = 100;

   // Creating optimizer configuration Json
   knlohmann::json integratorJs;
   integratorJs["Type"] = "Integrator/Quadrature";
   integratorJs["Method"] = "Simpson";

   // Creating module
   korali::solver::Integrator* itr;
   ASSERT_NO_THROW(itr = dynamic_cast<Quadrature*>(Module::getModule(integratorJs, &e)));

   // Defaults should be applied without a problem
   ASSERT_NO_THROW(itr->applyModuleDefaults(integratorJs));

   // Covering variable functions (no effect)
   ASSERT_NO_THROW(itr->applyVariableDefaults());

   // Backup the correct base configuration
   auto baseOptJs = integratorJs;
   auto baseExpJs = experimentJs;

   // Setting up optimizer correctly
   ASSERT_NO_THROW(itr->setConfiguration(integratorJs));

   // Testing optional parameters
   integratorJs = baseOptJs;
   experimentJs = baseExpJs;
   integratorJs.erase("Executions Per Generation");
   ASSERT_ANY_THROW(itr->setConfiguration(integratorJs));

   integratorJs = baseOptJs;
   experimentJs = baseExpJs;
   integratorJs["Executions Per Generation"] = "Not a Number";
   ASSERT_ANY_THROW(itr->setConfiguration(integratorJs));

   integratorJs = baseOptJs;
   experimentJs = baseExpJs;
   integratorJs["Executions Per Generation"] = 1;
   ASSERT_NO_THROW(itr->setConfiguration(integratorJs));
  }


} // namespace
