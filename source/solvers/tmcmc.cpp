#include "korali.h"
#include <numeric>
#include <limits>
#include <chrono>

#include <gsl/gsl_sort_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_multimin.h>

typedef struct fparam_s {
    const double *fj;
    size_t        fn;
    double        pj;
    double        cov;
} fparam_t;

Korali::Solver::TMCMC::TMCMC()
{
 range = gsl_rng_alloc (gsl_rng_default);
 gsl_rng_set(range, _k->_seed++);
}

Korali::Solver::TMCMC::~TMCMC()
{
 gsl_rng_free(range);
}

void Korali::Solver::TMCMC::initialize()
{
 // Allocating TMCMC memory
 _covarianceMatrix.resize(_k->N*_k->N);
 _meanTheta.resize(_k->N);
 _chainCandidatesParameters.resize(_k->N*_populationSize);
 _chainCandidatesLogLikelihoods.resize(_populationSize);
 _chainLeadersParameters.resize(_k->N*_populationSize);
 _chainLeadersLogLikelihoods.resize(_populationSize);
 _chainPendingFitness.resize(_populationSize);
 _currentChainStep.resize(_populationSize);
 _chainLengths.resize(_populationSize);

 if(_useLocalCovariance)
 {
  _localCovarianceMatrices.resize(_populationSize);
  for (size_t pos = 0; pos < _populationSize; ++pos)
  {
   _localCovarianceMatrices[pos].resize(_k->N*_k->N);
   for (size_t i = 0; i < _k->N; i++)
   for (size_t j = 0; j < _k->N; j++) _localCovarianceMatrices[pos][i*_k->N+j] = 0.0;
   for (size_t i = 0; i < _k->N; i++) _localCovarianceMatrices[pos][i*_k->N+i] = 1.0;
  }
 }

 // Initializing Runtime Variables
 for (size_t c = 0; c < _populationSize; c++) _currentChainStep[c] = 0;
 for (size_t c = 0; c < _populationSize; c++) _chainLengths[c] = 1;
 for (size_t c = 0; c < _populationSize; c++) _chainPendingFitness[c] = false;

 // Init Generation
 _annealingExponent       = 0.0;
 _logEvidence             = 0.0;
 _coefficientOfVariation  = 0.0;
 _finishedChainsCount     = 0;
 _databaseEntryCount      = 0;
 _proposalsAcceptanceRate = 1.0;
 _selectionAcceptanceRate = 1.0;
 _acceptedSamplesCount    = _populationSize;
 _chainCount              = _populationSize;
 for (size_t c = 0; c < _chainCount; c++) _currentChainStep[c]    = 0;
 for (size_t c = 0; c < _chainCount; c++) _chainPendingFitness[c] = false;

 _burnIn = std::vector<size_t>(_termCondMaxGenerations, _burnInDefault);
 if(_burnInSteps.size() > _burnIn.size())
 { printf("[Korali] Error: Number of defined Burn In Steps (%zu) larger than Max Generations (%zu)\n", _burnInSteps.size(), _termCondMaxGenerations); exit(-1); }
 std::copy(_burnInSteps.begin(), _burnInSteps.end(), _burnIn.begin());

 initializeSamples();
}

void Korali::Solver::TMCMC::runGeneration()
{
 if (_k->currentGeneration > 1) resampleGeneration();

 while (_finishedChainsCount < _chainCount)
 {
  for (size_t c = 0; c < _chainCount; c++) if (_currentChainStep[c] < _chainLengths[c]) if (_chainPendingFitness[c] == false)
  {
  _chainPendingFitness[c] = true;
  generateCandidate(c);
  evaluateSample(c);
  }
  _k->_conduit->checkProgress();
 }
}

void Korali::Solver::TMCMC::processSample(size_t c, double fitness)
{
 double ccLogPrior = _k->_problem->evaluateLogPrior(&_chainCandidatesParameters[c*_k->N]);
 double clLogPrior = _k->_problem->evaluateLogPrior(&_chainLeadersParameters[c*_k->N]);

 _chainCandidatesLogLikelihoods[c] = fitness;
 double L = exp((_chainCandidatesLogLikelihoods[c]-_chainLeadersLogLikelihoods[c])*_annealingExponent + (ccLogPrior-clLogPrior));

 if ( L >= 1.0 || L > gsl_ran_flat(range, 0.0, 1.0) ) {
   for (size_t i = 0; i < _k->N; i++) _chainLeadersParameters[c*_k->N + i] = _chainCandidatesParameters[c*_k->N + i];
   _chainLeadersLogLikelihoods[c] = _chainCandidatesLogLikelihoods[c];
   if (_currentChainStep[c] == _chainLengths[c]-1) _acceptedSamplesCount++; // XXX: is that correct? (DW)
 }

 _currentChainStep[c]++;
 if (_currentChainStep[c] > _burnIn[_k->currentGeneration] ) updateDatabase(&_chainLeadersParameters[c*_k->N], _chainLeadersLogLikelihoods[c]);
 _chainPendingFitness[c] = false;
 if (_currentChainStep[c] == _chainLengths[c]) _finishedChainsCount++;
}

void Korali::Solver::TMCMC::evaluateSample(size_t c)
{
  std::vector<double> _logTransformedSample(_k->N);

  for(size_t d = 0; d<_k->N; ++d) 
    if (_k->_variables[d]->_isLogSpace == true)
        _logTransformedSample[d] = std::exp(_chainCandidatesParameters[c*_k->N+d]);
    else 
        _logTransformedSample[d] = _chainCandidatesParameters[c*_k->N+d];

  _k->_conduit->evaluateSample(_logTransformedSample.data(), c);
}

void Korali::Solver::TMCMC::updateDatabase(double* point, double fitness)
{
 for (size_t i = 0; i < _k->N; i++) _sampleParametersDatabase.push_back(point[i]);
 _sampleFitnessDatabase.push_back(fitness);
 _databaseEntryCount++;
}

void Korali::Solver::TMCMC::generateCandidate(size_t c)
{
 double* covariance = _useLocalCovariance ? &_localCovarianceMatrices[c][0] : &_covarianceMatrix[0];
 gsl_vector_view out_view    = gsl_vector_view_array(&_chainCandidatesParameters[c*_k->N], _k->N);
 gsl_matrix_view sigma_view  = gsl_matrix_view_array(covariance, _k->N,_k->N);
 gsl_vector_view mean_view   = gsl_vector_view_array(&_chainLeadersParameters[c*_k->N], _k->N);
 gsl_ran_multivariate_gaussian(range, &mean_view.vector, &sigma_view.matrix, &out_view.vector);
}

void Korali::Solver::TMCMC::initializeSamples()
{
  for (size_t c = 0; c < _populationSize; c++) {
     for (size_t d = 0; d < _k->N; d++) {
       _chainLeadersParameters[c*_k->N + d] = _chainCandidatesParameters[c*_k->N + d] = _k->_variables[d]->getRandomNumber();
       _chainLeadersLogLikelihoods[c] += log( _k->_variables[d]->getDensity(_chainLeadersParameters[c*_k->N + d]) );
     }
     updateDatabase(&_chainLeadersParameters[c*_k->N], _chainLeadersLogLikelihoods[c]);
     _finishedChainsCount++;
  }
}

void Korali::Solver::TMCMC::resampleGeneration()
{
 std::vector<double> flcp(_databaseEntryCount);
 std::vector<double> weight(_databaseEntryCount);
 std::vector<double> q(_databaseEntryCount);
 std::vector<unsigned int> nn(_databaseEntryCount);
 std::vector<size_t> sel(_databaseEntryCount);

 double fmin = 0, xmin = 0;
 minSearch(&_sampleFitnessDatabase[0], _databaseEntryCount, _annealingExponent, _targetCVar, xmin, fmin);

 double _prevAnnealingExponent = _annealingExponent;

 if (xmin > _prevAnnealingExponent + _maxRhoUpdate)
 {
  if ( _k->_verbosity >= KORALI_DETAILED ) printf("[Korali] Warning: Annealing Step larger than Max Rho Update, updating Annealing Exponent by %f (Max Rho Update). \n", _maxRhoUpdate);
  _annealingExponent      = _prevAnnealingExponent + _maxRhoUpdate;
  _coefficientOfVariation = sqrt(tmcmc_objlogp(_annealingExponent, &_sampleFitnessDatabase[0], _databaseEntryCount, _prevAnnealingExponent, _targetCVar)) + _targetCVar;
 }
 else if (xmin > _prevAnnealingExponent)
 {
  _annealingExponent      = xmin;
  _coefficientOfVariation = sqrt(fmin) + _targetCVar;
 }
 else
 {
  if ( _k->_verbosity >= KORALI_DETAILED ) printf("[Korali] Warning: Annealing Step smaller than Min Rho Update, updating Annealing Exponent by %f (Min Rho Update). \n", _minRhoUpdate);
  _annealingExponent      = _prevAnnealingExponent + _minRhoUpdate;
  _coefficientOfVariation = sqrt(tmcmc_objlogp(_annealingExponent, &_sampleFitnessDatabase[0], _databaseEntryCount, _prevAnnealingExponent, _targetCVar)) + _targetCVar;
 }

 /* Compute weights and normalize*/

 for (size_t i = 0; i < _databaseEntryCount; i++) flcp[i] = _sampleFitnessDatabase[i]*(_annealingExponent-_prevAnnealingExponent);
 const double fjmax = gsl_stats_max(flcp.data(), 1, _databaseEntryCount);
 for (size_t i = 0; i < _databaseEntryCount; i++) weight[i] = exp( flcp[i] - fjmax );

 double sum_weight = std::accumulate(weight.begin(), weight.end(), 0.0);
 _logEvidence  += log(sum_weight) + fjmax - log(_databaseEntryCount);

 for (size_t i = 0; i < _databaseEntryCount; i++) q[i] = weight[i]/sum_weight;

 gsl_ran_multinomial(range, _databaseEntryCount, _populationSize, q.data(), nn.data());
 size_t zeroCount = 0;
 for (size_t i = 0; i < _databaseEntryCount; i++) { sel[i] = nn[i]; if ( nn[i] == 0 ) zeroCount++; }

 size_t uniqueSelections   = _databaseEntryCount - zeroCount;
 _proposalsAcceptanceRate  = (1.0*_acceptedSamplesCount)/_populationSize;
 _selectionAcceptanceRate = (1.0*uniqueSelections)/_populationSize;

 for (size_t i = 0; i < _k->N; i++)
 {
  _meanTheta[i] = 0;
  for (size_t j = 0; j < _databaseEntryCount; j++) _meanTheta[i] += _sampleParametersDatabase[j*_k->N + i]*q[j];
 }

 for (size_t i = 0; i < _k->N; i++) for (size_t j = i; j < _k->N; ++j)
 {
  double s = 0.0;
  for (size_t k = 0; k < _databaseEntryCount; ++k) s += q[k]*(_sampleParametersDatabase[k*_k->N+i]-_meanTheta[i])*(_sampleParametersDatabase[k*_k->N+j]-_meanTheta[j]);
  _covarianceMatrix[i*_k->N + j] = _covarianceMatrix[j*_k->N + i] = s*_covarianceScaling;
 }

 gsl_matrix_view sigma = gsl_matrix_view_array(&_covarianceMatrix[0], _k->N,_k->N);
 gsl_linalg_cholesky_decomp( &sigma.matrix );

 size_t ldi = 0;
 for (size_t i = 0; i < _databaseEntryCount; i++) {
   if (sel[i] != 0) {
     for (size_t j = 0; j < _k->N ; j++) _chainLeadersParameters[ldi*_k->N + j] = _sampleParametersDatabase[i*_k->N + j];
     _chainLeadersLogLikelihoods[ldi] = _sampleFitnessDatabase[i];
     _chainLengths[ldi] = sel[i] + _burnIn[_k->currentGeneration];
     ldi++;
   }
 }
 
 if (_useLocalCovariance) computeChainCovariances(_localCovarianceMatrices, uniqueSelections);

 _sampleFitnessDatabase.clear();
 _sampleParametersDatabase.clear();

 _databaseEntryCount = 0;
 _acceptedSamplesCount   = 0;
 _finishedChainsCount   = 0;
 _chainCount         = uniqueSelections;
 
 for (size_t c = 0; c < _chainCount; c++) _currentChainStep[c] = 0;
 for (size_t c = 0; c < _chainCount; c++) _chainPendingFitness[c] = false;
}

void Korali::Solver::TMCMC::computeChainCovariances(std::vector< std::vector<double> >& chain_cov, size_t newchains)
{
 //printf("Precomputing chain covariances for the current generation...\n");

 // allocate space
 std::vector<size_t> nn_ind(newchains);
 std::vector<size_t> nn_count(newchains);
 std::vector<double> diam(_k->N);
 std::vector<double> chain_mean(_k->N);
 gsl_matrix* work = gsl_matrix_alloc(_k->N, _k->N);

 // find diameters
 for (size_t d = 0; d < _k->N; ++d) {
  double d_min = +1e6;
  double d_max = -1e6;
  for (size_t pos = 0; pos < _populationSize; ++pos) {
   double s = _sampleParametersDatabase[pos*_k->N+d];
   if (d_min > s) d_min = s;
   if (d_max < s) d_max = s;
  }
  diam[d] = d_max-d_min;
  //printf("Diameter %ld: %.6lf\n", d, diam[d]);
 }

 size_t idx, pos;
 int status = 0;
 double ds = 0.05;
 for (double scale = 0.1; scale <= 1.0; scale += ds) {
  // find neighbors in a rectangle - O(_populationSize^2)
  for (pos = 0; pos < newchains; ++pos) {
   nn_count[pos] = 0;
   double* curr = &_chainLeadersParameters[pos*_k->N];
   for (size_t i = 0; i < _populationSize; i++) {
    double* s = &_sampleParametersDatabase[i*_k->N];
    bool isInRectangle = true;
     for (size_t d = 0; d < _k->N; d++)  if (fabs(curr[d]-s[d]) > scale*diam[d]) isInRectangle = false;
     if (isInRectangle) {
      nn_ind[pos*_populationSize+nn_count[pos]] = i;
      nn_count[pos]++;
     }
   }
  }

  // compute the covariances
  for (pos = 0; pos < newchains; ++pos) {
   for (size_t d = 0; d < _k->N; ++d) {
    chain_mean[d] = 0;
    for (size_t k = 0; k < nn_count[pos]; ++k) {
     idx = nn_ind[pos*_populationSize+k];
     chain_mean[d] += _sampleParametersDatabase[idx*_k->N+d];
    }
    chain_mean[d] /= nn_count[pos];
   }

   for (size_t i = 0; i < _k->N; i++)
    for (size_t j = 0; j < _k->N; ++j) {
     double s = 0;
     for (size_t k = 0; k < nn_count[pos]; k++) {
      idx = nn_ind[pos*_populationSize+k];
      s  += (_sampleParametersDatabase[idx*_k->N+i]-chain_mean[i]) *
         (_sampleParametersDatabase[idx*_k->N+j]-chain_mean[j]);
     }
     chain_cov[pos][i*_k->N+j] = chain_cov[pos][j*_k->N+i] = s/nn_count[pos];
    }

   // check if the matrix is positive definite
   for (size_t i = 0; i < _k->N; i++)
    for (size_t j = 0; j < _k->N; ++j) {
     double s = chain_cov[pos][i*_k->N+j];
     gsl_matrix_set(work, i, j, s);
    }
   gsl_set_error_handler_off();
   status = gsl_linalg_cholesky_decomp(work);
   if (status == GSL_SUCCESS) break;
  }
 }

 for (pos = 0; pos < newchains; ++pos) {
   gsl_matrix_view sigma  = gsl_matrix_view_array(&chain_cov[pos][0], _k->N,_k->N);
   gsl_linalg_cholesky_decomp( &sigma.matrix );
 }

 if (status != GSL_SUCCESS) {
  fprintf(stderr, "[Korali] TMCMC Error: GSL failed to create Chain Covariance Matrix.\n");
 }

 // deallocate space
 gsl_matrix_free(work);
}

double Korali::Solver::TMCMC::tmcmc_objlogp(double x, const double *fj, size_t fn, double pj, double zero)
{
 std::vector<double> weight(fn);
 std::vector<double> q(fn);
 const double fjmax = gsl_stats_max(fj, 1, fn);

 for(size_t i = 0; i <fn; i++) weight[i] = exp((fj[i]-fjmax)*(x-pj));
 double sum_weight = std::accumulate(weight.begin(), weight.end(), 0.0);
 for(size_t i = 0; i < fn; i++) q[i] = weight[i]/sum_weight;

 double mean_q = gsl_stats_mean(q.data(), 1, fn);
 double std_q  = gsl_stats_sd_m(q.data(), 1, fn, mean_q);
 double cov2   = (std_q/mean_q-zero); 
 cov2 *= cov2;

 return cov2;
}

double Korali::Solver::TMCMC::objLog(const gsl_vector *v, void *param)
{
 double x = gsl_vector_get(v, 0);
 fparam_t *fp = (fparam_t *) param;
 return Korali::Solver::TMCMC::tmcmc_objlogp(x, fp->fj, fp->fn, fp->pj, fp->cov);
}

void Korali::Solver::TMCMC::minSearch(double const *fj, size_t fn, double pj, double objCov, double& xmin, double& fmin)
{
 // Minimizer Options
 size_t MaxIter     = 100;    /* Max number of search iterations */
 double Tol         = 1e-12;  /* Tolerance for root finding */
 double Step        = 1e-8;   /* Search stepsize */

 const gsl_multimin_fminimizer_type *T;
 gsl_multimin_fminimizer *s = NULL;
 gsl_vector *ss, *x;
 gsl_multimin_function minex_func;

 size_t iter     = 0;
 int status;
 double size;

 fparam_t fp;
 fp.fj = fj;
 fp.fn = fn;
 fp.pj = pj;
 fp.cov = objCov;

 x = gsl_vector_alloc (1);
 gsl_vector_set (x, 0, pj);

 ss = gsl_vector_alloc (1);
 gsl_vector_set_all (ss, Step);

 minex_func.n      = 1;
 minex_func.f      = objLog;
 minex_func.params = &fp;

 // SELECT ONE MINIMIZER STRATEGY
 T = gsl_multimin_fminimizer_nmsimplex;
 /* T = gsl_multimin_fminimizer_nmsimplex2; */
 /* T = gsl_multimin_fminimizer_nmsimplex2rand; (warning: not reliable)  */
 s = gsl_multimin_fminimizer_alloc (T, 1);
 gsl_multimin_fminimizer_set (s, &minex_func, x, ss);

 fmin = 0;
 xmin = 0.0;

 do {
   iter++;
   status = gsl_multimin_fminimizer_iterate(s);
   size   = gsl_multimin_fminimizer_size (s);
   status = gsl_multimin_test_size (size, Tol);
 } while (status == GSL_CONTINUE && iter < MaxIter);

 if (_k->_verbosity >= KORALI_DETAILED)
 {
   if(status == GSL_SUCCESS && s->fval >  Tol) printf("[Korali] Min Search converged but did not find minimum. \n");
   if(status != GSL_SUCCESS && s->fval <= Tol) printf("[Korali] Min Search did not converge but minimum found\n");
   if(status != GSL_SUCCESS && s->fval >  Tol) printf("[Korali] Min Search did not converge and did not find minimum\n");
   if(iter >= MaxIter) printf("[Korali] Min Search MaxIter (%zu) reached\n", MaxIter);
 }

 if (s->fval <= Tol) {
   fmin = s->fval;
   xmin = gsl_vector_get(s->x, 0);
 }

 if (xmin >= 1.0) {
   fmin = tmcmc_objlogp(1.0, fj, fn, pj, objCov);
   xmin = 1.0;
 }

 gsl_vector_free(x);
 gsl_vector_free(ss);
 gsl_multimin_fminimizer_free (s);
}


bool Korali::Solver::TMCMC::isFeasibleCandidate(size_t c)
{
 double clLogPrior = _k->_problem->evaluateLogPrior(&_chainLeadersParameters[c*_k->N]);
 if (clLogPrior > -INFINITY) return true;
 return false;
}

bool Korali::Solver::TMCMC::checkTermination()
{
 bool isFinished = false;

 if(_annealingExponent >= 1.0)
 {
  isFinished = true;
  if(_k->_verbosity >= KORALI_MINIMAL) printf("[Korali] Annealing completed (1.0).\n");
 }

 if( _termCondMaxGenerationsEnabled && (_k->currentGeneration >= _termCondMaxGenerations) )
 {
  isFinished = true;
  if(_k->_verbosity >= KORALI_MINIMAL) printf("[Korali] Maximum number of Generations reached (%lu).\n", _termCondMaxGenerations);
 }

 return isFinished;

}

void Korali::Solver::TMCMC::finalize()
{

}

void Korali::Solver::TMCMC::printGeneration()
{
 if (_k->_verbosity >= KORALI_MINIMAL)
 {
  printf("--------------------------------------------------------------------\n");
  printf("[Korali] Generation %ld - Annealing Exponent:  %.3e.\n", _k->currentGeneration, _annealingExponent);
 }

 if (_k->_verbosity >= KORALI_NORMAL)
 {
  printf("[Korali] Acceptance Rate (proposals / selections): (%.2f%% / %.2f%%)\n", 100*_proposalsAcceptanceRate, 100*_selectionAcceptanceRate);
  printf("[Korali] Coefficient of Variation: %.2f%%\n", 100.0*_coefficientOfVariation);
 }

 if (_k->_verbosity >= KORALI_DETAILED)
 {
  printf("[Korali] Sample Mean:\n");
  for (size_t i = 0; i < _k->N; i++) printf(" %s = %+6.3e\n", _k->_variables[i]->_name.c_str(), _meanTheta[i]);
  printf("[Korali] Sample Covariance:\n");
  for (size_t i = 0; i < _k->N; i++)
  {
   printf("   | ");
   for (size_t j = 0; j < _k->N; j++)
    if(j <= i)  printf("%+6.3e  ",_covarianceMatrix[i*_k->N+j]);
    else        printf("     -      ");
   printf(" |\n");
  }
 }
}
