#ifndef _KORALI_SOLVER_MCMC_H_
#define _KORALI_SOLVER_MCMC_H_

#include "solvers/base.h"
#include <gsl/gsl_rng.h>
#include <gsl/gsl_vector.h>

namespace Korali { namespace Solver {

class MCMC : public Base
{
 public:

 // Constructor / Destructor
 MCMC(nlohmann::json& js, std::string name);
 ~MCMC();

 // MCMC Configuration
 unsigned int _s; /* Population Size */
 //bool _useLocalCov; /* Using local covariance instead of sample cov */
 size_t _burnin; /* burn in generations */
 bool _isTermCondMaxFunEvals;
 size_t _termCondMaxFunEvals;
 bool _isTermCondMaxGenerations;
 size_t _termCondMaxGenerations;
 char _terminationReason[500];

 // MCMC Runtime Variables
 gsl_rng *gslGen;
 double* clPoint; /*  Leader parameter values */
 double clLogLikelihood; /* Leader fitness value */
 double* ccPoint; /*  Candidate parameter values */
 double ccLogLikelihood; /* Candidate fitness value */
 double ccLogPrior; /* Candidate prior value */
 double acceptanceRateProposals;
 size_t countgens;
 size_t chainLength;
 size_t databaseEntries;
 double* databasePoints; /* Variable values of samples in DB */
 double* databaseFitness; /* Fitness of samples in DB */
 size_t  countevals; /* Number of function evaluations */

 // MCMC Status variables
 double* _covarianceMatrix; /* Covariance of leader fitness values */
 //double **local_cov; /* Local covariances of leaders */

  // Korali Methods
 void run() override;
 void processSample(size_t c, double fitness) override;

  // Internal MCMC Methods
 void initializeSamples();
 void updateDatabase(double* point, double fitness);
 void generateCandidate();
 //void computeChainCovariances(double** chain_cov, size_t newchains) const;
 void updateSolver();
 bool isFeasibleCandidate();
 bool checkTermination();

 // Serialization Methods
 nlohmann::json getConfiguration() override;
 void setConfiguration(nlohmann::json& js) override;
 void setState(nlohmann::json& js) override;

 // Print Methods
 void printGeneration() const;
 void printFinal() const;
};

} } // namespace Korali::Solver

#endif // _KORALI_SOLVER_MCMC_H_
