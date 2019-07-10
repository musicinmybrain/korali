#include "korali.h"
#include "model/posteriorModel.h"

int main(int argc, char* argv[])
{
 auto k = Korali::Engine();

 std::vector<double> x, y; // Reference Data
 getReferenceData(x, y);

 k.setModel([x](Korali::ModelData& d) { posteriorModel(d.getVariables(), d.getResults(), x); });

 k["Problem"] = "Bayesian";
 k["Bayesian"]["Likelihood"]["Type"] = "Reference";
 k["Bayesian"]["Likelihood"]["Model"] = "Additive Gaussian";
 k["Bayesian"]["Likelihood"]["Reference Data"] = y;


 k["Variables"][0]["Name"] = "a";
 k["Variables"][0]["Bayesian"]["Type"] = "Computational";
 k["Variables"][0]["Bayesian"]["Prior Distribution"]["Type"] = "Uniform";
 k["Variables"][0]["Bayesian"]["Prior Distribution"]["Minimum"] = -5.0;
 k["Variables"][0]["Bayesian"]["Prior Distribution"]["Maximum"] = +5.0;

 k["Variables"][1]["Name"] = "b";
 k["Variables"][1]["Bayesian"]["Type"] = "Computational";
 k["Variables"][1]["Bayesian"]["Prior Distribution"]["Type"] = "Uniform";
 k["Variables"][1]["Bayesian"]["Prior Distribution"]["Minimum"] = -5.0;
 k["Variables"][1]["Bayesian"]["Prior Distribution"]["Maximum"] = +5.0;

 k["Variables"][2]["Name"] = "Sigma";
 k["Variables"][2]["Bayesian"]["Type"] = "Statistical";
 k["Variables"][2]["Bayesian"]["Prior Distribution"]["Type"] = "Uniform";
 k["Variables"][2]["Bayesian"]["Prior Distribution"]["Minimum"] = 0.0;
 k["Variables"][2]["Bayesian"]["Prior Distribution"]["Maximum"] = +5.0;

 k["Solver"] = "CMAES";

 k["Variables"][0]["CMAES"]["Lower Bound"] = -5.0;
 k["Variables"][0]["CMAES"]["Upper Bound"] = +5.0;
 k["Variables"][1]["CMAES"]["Lower Bound"] = -5.0;
 k["Variables"][1]["CMAES"]["Upper Bound"] = +5.0;
 k["Variables"][2]["CMAES"]["Lower Bound"] = 0.0;
 k["Variables"][2]["CMAES"]["Upper Bound"] = +5.0;

 k["CMAES"]["Objective"] = "Maximize";
 k["CMAES"]["Sample Count"] = 12;

 k["CMAES"]["Termination Criteria"]["Max Generations"]["Value"] = 100;

 k["Console Output Frequency"] = 5;
 k["File Output Frequency"] = 5;

 k["Result Directory"] = "_a3_bayesian_inference_cmaes";

 k.run();
}
