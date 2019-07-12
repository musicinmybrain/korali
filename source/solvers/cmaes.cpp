#include "korali.h"

#include <stdio.h>
#include <unistd.h>
#include <chrono>
#include <numeric>   // std::iota
#include <algorithm> // std::sort

using namespace Korali::Solver;

CMAES::CMAES()
{
 // Initializing Gaussian Generator
 auto jsGaussian = nlohmann::json();
 jsGaussian["Type"] = "Gaussian";
 jsGaussian["Mean"] = 0.0;
 jsGaussian["Sigma"] = 1.0;
 jsGaussian["Seed"] = _k->_seed++;
 _gaussianGenerator = new Variable();
 _gaussianGenerator->setDistribution(jsGaussian);
}

CMAES::~CMAES()
{
  delete _gaussianGenerator;
}


void CMAES::initialize()
{
 size_t s_max  = std::max(_sampleCount,  _viabilitySampleCount);
 size_t mu_max = std::max(_muValue, _viabilityMu);

 _chiN = sqrt((double) _k->N) * (1. - 1./(4.*_k->N) + 1./(21.*_k->N*_k->N));

 _constraintsDefined = (_k->_fconstraints.size() > 0);
 if(_constraintsDefined) { _isViabilityRegime = true; printf("bla"); }
 else                    _isViabilityRegime = false;


 if(_isViabilityRegime) {
     _currentSampleCount  = _viabilitySampleCount;
     _currentSampleMu = _viabilityMu;
 } else {
     _currentSampleCount  = _sampleCount;
     _currentSampleMu = _muValue;
 }

 // Allocating Memory
 _samplePopulation.resize(_k->N*s_max);

 _evolutionPath.resize(_k->N);
 _conjugateEvolutionPath.resize(_k->N);
 _BDZtmp.resize(_k->N);
 _y.resize(_k->N);
 _mean.resize(_k->N);
 _previousMean.resize(_k->N);
 _bestEverSample.resize(_k->N);
 _axisD.resize(_k->N);
 _axisDtmp.resize(_k->N);
 _currentBestSample.resize(_k->N);

 _sortingIndex.resize(s_max);

 _isInitializedSample.resize(s_max);
 _fitnessVector.resize(s_max);

 _C.resize(_k->N*_k->N);
 _Ctmp.resize(_k->N*_k->N);
 _B.resize(_k->N*_k->N);
 _Btmp.resize(_k->N*_k->N);

 _Z.resize(s_max*_k->N);
 _BDZ.resize(s_max*_k->N);

 _transformedSamples.resize(s_max*_k->N);
 

 if (_objective == "Maximize")     _evaluationSign = 1.0;
 else if(_objective == "Minimize") _evaluationSign = -1.0;
 else { fprintf(stderr,"[Korali] Warning: Objective must be either be initialized to \'Maximize\' or \'Minimize\' (is %s).\n", _objective.c_str()); }         

 // Initailizing Mu
 _muWeights.resize(mu_max);

 
 // CCMA-ES variables
 if (_constraintsDefined)
 {
  if( (_globalSuccessLearningRate <= 0.0) || (_globalSuccessLearningRate > 1.0) ) 
        { fprintf( stderr, "[Korali] CMA-ES Error: Invalid Global Success Learning Rate (%f), must be greater than 0.0 and less than 1.0\n",  _globalSuccessLearningRate ); exit(-1); }
  if( (_targetSuccessRate <= 0.0) || (_targetSuccessRate > 1.0) )
        { fprintf( stderr, "[Korali] CMA-ES Error: Invalid Target Success Rate (%f), must be greater than 0.0 and less than 1.0\n",  _targetSuccessRate ); exit(-1); }
  if(_covMatrixAdaptionStrength <= 0.0) 
        { fprintf( stderr, "[Korali] CMA-ES Error: Invalid Adaption Size (%f), must be greater than 0.0\n", _covMatrixAdaptionStrength ); exit(-1); }

  _globalSuccessRate = 0.5;
  _bestValidSample = -1;
  _constraintEvaluationCount = 0;
  _adaptationCount = 0;
  _maxViolationCount = 0;
  _sampleViolationCounts.resize(s_max);
  _viabilityBoundaries.resize(_k->_fconstraints.size());

  _viabilityImprovement.resize(s_max);
  _viabilityIndicator.resize(_k->_fconstraints.size());
  _constraintEvaluations.resize(_k->_fconstraints.size());

  for (size_t c = 0; c < _k->_fconstraints.size(); c++) _viabilityIndicator[c].resize(s_max);
  for (size_t c = 0; c < _k->_fconstraints.size(); c++) _constraintEvaluations[c].resize(s_max);

  _constraintNormal.resize(_k->_fconstraints.size());
  for (size_t i = 0; i < _k->_fconstraints.size(); i++) _constraintNormal[i].resize(_k->N);

  _bestEverConstraintEvaluation.resize(_k->_fconstraints.size());

  _normalVectorLearningRate = 1.0/(2.0+_k->N);
  _beta = _covMatrixAdaptionStrength/(_k->N+2.);

 }

 // Setting algorithm internal variables
 if (_constraintsDefined) { initMuWeights(_viabilityMu); }
 else initMuWeights(_muValue);

 initCovariance();

 // Setting eigensystem evaluation Frequency
 //if( _covarianceEigenEvalFreq < 0.0) _covarianceEigenEvalFreq = 1.0/_covarianceMatrixLearningRate/((double)_k->N)/10.0;

 _isEigenSystemUpdate = true;

 _infeasibleSampleCount = 0;
 _resampleCount       = 0;
 
 _bestEverValue = -std::numeric_limits<double>::max();
 _conjugateEvolutionPathL2Norm = 0.0;

 /* set _mean */
 for (size_t i = 0; i < _k->N; ++i)
 {
   if(_variableSettings[i].initialMean < _variableSettings[i].lowerBound || _variableSettings[i].initialMean > _variableSettings[i].upperBound) if(_k->_verbosity >= KORALI_MINIMAL)
    fprintf(stderr,"[Korali] Warning: Initial Value (%.4f) of variable \'%s\' is out of bounds (%.4f-%.4f).\n",
            _variableSettings[i].initialMean,
            _k->_variables[i]->_name.c_str(),
            _variableSettings[i].lowerBound,
            _variableSettings[i].upperBound);
   _mean[i] = _previousMean[i] = _variableSettings[i].initialMean;
 }

}


void CMAES::runGeneration()
{
 if ( _constraintsDefined ) checkMeanAndSetRegime();
 prepareGeneration();
 if ( _constraintsDefined ){ updateConstraints(); handleConstraints(); }
 evaluateSamples();
 updateDistribution();
}


void CMAES::evaluateSamples()
{
 for (size_t i = 0; i < _currentSampleCount; i++) for(size_t d = 0; d < _k->N; ++d)
  if(_k->_variables[d]->_isLogSpace == true)
   _transformedSamples[i*_k->N+d] = std::exp(_samplePopulation[i*_k->N+d]);
  else
   _transformedSamples[i*_k->N+d] = _samplePopulation[i*_k->N+d];

  while (_finishedSampleCount < _currentSampleCount)
  {
    for (size_t i = 0; i < _currentSampleCount; i++) if (_isInitializedSample[i] == false)
    {
      _isInitializedSample[i] = true;
      _k->_conduit->evaluateSample(&_transformedSamples[0], i);
    }
    _k->_conduit->checkProgress();
  }
}

void CMAES::initMuWeights(size_t numsamplesmu)
{
 // Initializing Mu Weights
 if      (_muType == "Linear")       for (size_t i = 0; i < numsamplesmu; i++) _muWeights[i] = numsamplesmu - i;
 else if (_muType == "Equal")        for (size_t i = 0; i < numsamplesmu; i++) _muWeights[i] = 1.;
 else if (_muType == "Logarithmic")  for (size_t i = 0; i < numsamplesmu; i++) _muWeights[i] = log(std::max( (double)numsamplesmu, 0.5*_currentSampleCount)+0.5)-log(i+1.);
 else  { fprintf( stderr, "[Korali] CMA-ES Error: Invalid setting of Mu Type (%s) (Linear, Equal, or Logarithmic accepted).",  _muType.c_str()); exit(-1); }

 // Normalize weights vector and set mueff
 double s1 = 0.0;
 double s2 = 0.0;

 for (size_t i = 0; i < numsamplesmu; i++)
 {
  s1 += _muWeights[i];
  s2 += _muWeights[i]*_muWeights[i];
 }
 _effectiveMu = s1*s1/s2;

 for (size_t i = 0; i < numsamplesmu; i++) _muWeights[i] /= s1;
 
 // Setting Cumulative Covariancea
 if( (_initialCumulativeCovariance <= 0) || (_initialCumulativeCovariance > 1) ) _cumulativeCovariance = (4.0 + _effectiveMu/(1.0*_k->N)) / (_k->N+4.0 + 2.0*_effectiveMu/(1.0*_k->N));
 else _cumulativeCovariance = _initialCumulativeCovariance;

 // Setting Sigma Cumulation Factor
 _sigmaCumulationFactor = _initialSigmaCumulationFactor;
 if (_sigmaCumulationFactor <= 0 || _sigmaCumulationFactor >= 1) 
    if (_constraintsDefined) _sigmaCumulationFactor = sqrt(_effectiveMu) / ( sqrt(_effectiveMu) + sqrt(_k->N) );
    else                     _sigmaCumulationFactor = (_effectiveMu + 2.0) / (_k->N + _effectiveMu + 3.0);

 // Setting Damping Factor
 _dampFactor = _initialDampFactor;
 if (_dampFactor <= 0.0)
     _dampFactor = (1.0 + 2*std::max(0.0, sqrt((_effectiveMu-1.0)/(_k->N+1.0)) - 1)) + _sigmaCumulationFactor;

}


void CMAES::initCovariance()
{

 for(size_t d = 0; d < _k->N; ++d)
 {
    if ( _variableSettings[d].initialStdDev<0 )
    if ( std::isfinite(_variableSettings[d].lowerBound) && std::isfinite(_variableSettings[d].upperBound ) )
            _variableSettings[d].initialStdDev = 0.3*(_variableSettings[d].upperBound-_variableSettings[d].lowerBound);
    else 
        { printf("[Korali] Warning: Lower/Upper Bound not defined, and Initial Standard Dev not defined for variable '%s'\n", _k->_variables[d]->_name.c_str()); exit(-1); }
 }
 
 // Setting Sigma
 _trace = 0.0;
 for (size_t i = 0; i < _k->N; ++i) _trace += _variableSettings[i].initialStdDev*_variableSettings[i].initialStdDev;
 _sigma = sqrt(_trace/_k->N);

 // Setting B, C and _axisD
 for (size_t i = 0; i < _k->N; ++i)
 {
  _B[i*_k->N+i] = 1.0;
  _C[i*_k->N+i] = _axisD[i] = _variableSettings[i].initialStdDev * sqrt(_k->N / _trace);
  _C[i*_k->N+i] *= _C[i*_k->N+i];
 }

 _minCovarianceEigenvalue = *std::min_element(std::begin(_axisD), std::end(_axisD));
 _maxCovarianceEigenvalue = *std::max_element(std::begin(_axisD), std::end(_axisD));
 
 _minCovarianceEigenvalue = _minCovarianceEigenvalue * _minCovarianceEigenvalue;
 _maxCovarianceEigenvalue = _maxCovarianceEigenvalue * _maxCovarianceEigenvalue;

 _maxDiagCElement=_C[0]; for(size_t i=1;i<_k->N;++i) if(_maxDiagCElement<_C[i*_k->N+i]) _maxDiagCElement=_C[i*_k->N+i];
 _minDiagCElement=_C[0]; for(size_t i=1;i<_k->N;++i) if(_minDiagCElement>_C[i*_k->N+i]) _minDiagCElement=_C[i*_k->N+i];
}


void CMAES::processSample(size_t sampleId, double fitness)
{
 double logPrior = _k->_problem->evaluateLogPrior(&_samplePopulation[sampleId*_k->N]);
 fitness = _evaluationSign * (logPrior+fitness);
 if(std::isfinite(fitness) == false)
 {
   fitness = _evaluationSign * std::numeric_limits<double>::max();
   printf("[Korali] Warning: sample %zu returned non finite fitness (set to %e)!\n", sampleId, fitness);
 }

 _fitnessVector[sampleId] = fitness;
 _finishedSampleCount++;
}


void CMAES::checkMeanAndSetRegime()
{
  if (_isViabilityRegime == false) return; /* mean already inside valid domain, no udpates */

  for (size_t c = 0; c < _k->_fconstraints.size(); c++){
    _constraintEvaluationCount++;
    std::vector<double> sample(_mean);
    if ( _k->_fconstraints[c](sample) > 0.) return; /* mean violates constraint, do nothing */
  }

  /* mean inside domain, switch regime and update internal variables */
  _isViabilityRegime = false;

  for (size_t c = 0; c < _k->_fconstraints.size(); c++) { _viabilityBoundaries[c] = 0; }
  _currentSampleCount = _sampleCount;
  _currentSampleMu = _muValue;

  _bestEverValue = -std::numeric_limits<double>::max();
  initMuWeights(_currentSampleMu);
  initCovariance();
}


void CMAES::updateConstraints() //TODO: maybe parallelize constraint evaluations (DW)
{

 for(size_t i = 0; i < _currentSampleCount; i++) _sampleViolationCounts[i] = 0;
 _maxViolationCount = 0;

 for(size_t c = 0; c < _k->_fconstraints.size(); c++)
 {
  double maxviolation = 0.0;
  for(size_t i = 0; i < _currentSampleCount; ++i)
  {
    _constraintEvaluationCount++;
    std::vector<double> sample(&_samplePopulation[i*_k->N], &_samplePopulation[(i+1)*_k->N]);

    _constraintEvaluations[c][i] = _k->_fconstraints[c]( sample );

    if ( _constraintEvaluations[c][i] > maxviolation ) maxviolation = _constraintEvaluations[c][i];
    if ( _k->currentGeneration == 0 && _isViabilityRegime ) _viabilityBoundaries[c] = maxviolation;

    if ( _constraintEvaluations[c][i] > _viabilityBoundaries[c] + 1e-12 ) _sampleViolationCounts[i]++;
    if ( _sampleViolationCounts[i] > _maxViolationCount ) _maxViolationCount = _sampleViolationCounts[i];

  }
 }

}


void CMAES::reEvaluateConstraints() //TODO: maybe we can parallelize constraint evaluations (DW)
{
  _maxViolationCount = 0;
  for(size_t i = 0; i < _currentSampleCount; ++i) if(_sampleViolationCounts[i] > 0)
  {
    _sampleViolationCounts[i] = 0;
    for(size_t c = 0; c < _k->_fconstraints.size(); c++)
    {
      _constraintEvaluationCount++;
      std::vector<double> sample(&_samplePopulation[i*_k->N], &_samplePopulation[(i+1)*_k->N]);

      _constraintEvaluations[c][i] = _k->_fconstraints[c]( sample );

      if( _constraintEvaluations[c][i] > _viabilityBoundaries[c] + 1e-12 ) { _viabilityIndicator[c][i] = true; _sampleViolationCounts[i]++; }
      else _viabilityIndicator[c][i] = false;

    }
    if (_sampleViolationCounts[i] > _maxViolationCount) _maxViolationCount = _sampleViolationCounts[i];
  }
}


void CMAES::updateViabilityBoundaries()
{
 for(size_t c = 0; c < _k->_fconstraints.size(); c++)
 {
  double maxviolation = 0.0;
  for(size_t i = 0; i < _currentSampleMu /* _currentSampleCount alternative */ ; ++i) if (_constraintEvaluations[c][_sortingIndex[i]] > maxviolation)
    maxviolation = _constraintEvaluations[c][_sortingIndex[i]];

  _viabilityBoundaries[c] = std::max( 0.0, std::min(_viabilityBoundaries[c], 0.5*(maxviolation + _viabilityBoundaries[c])) );
 }
}


bool CMAES::isFeasible(size_t sampleIdx) const
{
 for (size_t d = 0; d < _k->N; ++d)
  if (_samplePopulation[ sampleIdx*_k->N+d ] < _variableSettings[d].lowerBound || _samplePopulation[ sampleIdx*_k->N+d ] > _variableSettings[d].upperBound) return false;
 return true;
}


void CMAES::prepareGeneration()
{

 /* calculate eigensystem */
 for (size_t d = 0; d < _k->N; ++d) _Ctmp.assign(std::begin(_C), std::end(_C));
 updateEigensystem(_Ctmp);

 for (size_t i = 0; i < _currentSampleCount; ++i)
 {
     size_t initial_infeasible = _infeasibleSampleCount;
     sampleSingle(i);
     while( isFeasible( i ) == false )
     {
       _infeasibleSampleCount++;
       sampleSingle(i);

       if ( _termCondMaxInfeasibleResamplingsEnabled )
       if ( (_infeasibleSampleCount - initial_infeasible) > _termCondMaxInfeasibleResamplings )
       {
        if(_k->_verbosity >= KORALI_MINIMAL) printf("[Korali] Warning: Exiting resampling loop (sample %zu), max resamplings (%zu) reached.\n", i, _termCondMaxInfeasibleResamplings);
        exit(-1);
       }
     }
 }

 _finishedSampleCount = 0;
 for (size_t i = 0; i < _currentSampleCount; i++) _isInitializedSample[i] = false;
}


void CMAES::sampleSingle(size_t sampleIdx)
{

  /* generate scaled random vector (D * z) */
  for (size_t d = 0; d < _k->N; ++d)
  {
   _Z[sampleIdx*_k->N+d] = _gaussianGenerator->getRandomNumber();
   if (_isDiag) {
     _BDZ[sampleIdx*_k->N+d] = _axisD[d] * _Z[sampleIdx*_k->N+d];
     _samplePopulation[sampleIdx * _k->N + d] = _mean[d] + _sigma * _BDZ[sampleIdx*_k->N+d];
   }
   else _BDZtmp[d] = _axisD[d] * _Z[sampleIdx*_k->N+d];
  }

  if (!_isDiag)
   for (size_t d = 0; d < _k->N; ++d) {
    _BDZ[sampleIdx*_k->N+d] = 0.0;
    for (size_t e = 0; e < _k->N; ++e) _BDZ[sampleIdx*_k->N+d] += _B[d*_k->N+e] * _BDZtmp[e];
    _samplePopulation[sampleIdx * _k->N + d] = _mean[d] + _sigma * _BDZ[sampleIdx*_k->N+d];
  }
}


void CMAES::updateDistribution()
{

 /* Generate _sortingIndex */
 sort_index(_fitnessVector, _sortingIndex, _currentSampleCount);

 if( (_constraintsDefined == false) ||  _isViabilityRegime )
  _bestValidSample = 0;
 else
 {
  _bestValidSample = -1;
  for (size_t i = 0; i < _currentSampleCount; i++) if(_sampleViolationCounts[_sortingIndex[i]] == 0) _bestValidSample = _sortingIndex[i];
  if(_k->_verbosity >= KORALI_DETAILED && _bestValidSample == -1)
    { printf("[Korali] Warning: all samples violate constraints, no updates taking place.\n"); return; }
 }

 /* update function value history */
 _previousBestValue = _currentBestValue;

 /* update current best */
 _currentBestValue = _fitnessVector[_bestValidSample];
 for (size_t d = 0; d < _k->N; ++d) _currentBestSample[d] = _samplePopulation[_bestValidSample*_k->N + d];

 /* update xbestever */
 if ( _currentBestValue > _bestEverValue )
 {
  _previousBestEverValue = _bestEverValue;
  _bestEverValue = _currentBestValue;
  for (size_t d = 0; d < _k->N; ++d) _bestEverSample[d]   = _currentBestSample[d];
  for (size_t c = 0; c < _k->_fconstraints.size(); c++) _bestEverConstraintEvaluation[c] = _constraintEvaluations[c][_bestValidSample];
 }

 /* set weights */
 for (size_t d = 0; d < _k->N; ++d) {
   _previousMean[d] = _mean[d];
   _mean[d] = 0.;
   for (size_t i = 0; i < _currentSampleMu; ++i)
     _mean[d] += _muWeights[i] * _samplePopulation[_sortingIndex[i]*_k->N + d];

   _y[d] = (_mean[d] - _previousMean[d])/_sigma;
 }

 /* calculate z := D^(-1) * B^(T) * _y into _BDZtmp */
 for (size_t d = 0; d < _k->N; ++d) {
  double sum = 0.0;
  if (_isDiag) sum = _y[d];
  else for (size_t e = 0; e < _k->N; ++e) sum += _B[e*_k->N+d] * _y[e]; /* B^(T) * _y ( iterating B[e][d] = B^(T) ) */

  _BDZtmp[d] = sum / _axisD[d]; /* D^(-1) * B^(T) * _y */
 }

 _conjugateEvolutionPathL2Norm = 0.0;

 /* cumulation for _sigma (ps) using B*z */
 for (size_t d = 0; d < _k->N; ++d) {
    double sum = 0.0;
    if (_isDiag) sum = _BDZtmp[d];
    else for (size_t e = 0; e < _k->N; ++e) sum += _B[d*_k->N+e] * _BDZtmp[e];

    _conjugateEvolutionPath[d] = (1. - _sigmaCumulationFactor) * _conjugateEvolutionPath[d] + sqrt(_sigmaCumulationFactor * (2. - _sigmaCumulationFactor) * _effectiveMu) * sum;

    /* calculate norm(ps)^2 */
    _conjugateEvolutionPathL2Norm += _conjugateEvolutionPath[d] * _conjugateEvolutionPath[d];
 }

 int hsig = sqrt(_conjugateEvolutionPathL2Norm) / sqrt(1. - pow(1.-_sigmaCumulationFactor, 2.0*(1.0+_k->currentGeneration))) / _chiN  < 1.4 + 2./(_k->N+1);

 /* cumulation for covariance matrix (pc) using B*D*z~_k->N(0,C) */
 for (size_t d = 0; d < _k->N; ++d)
   _evolutionPath[d] = (1. - _cumulativeCovariance) * _evolutionPath[d] + hsig * sqrt( _cumulativeCovariance * (2. - _cumulativeCovariance) * _effectiveMu ) * _y[d];

 /* update of C  */
 adaptC(hsig);

 double sigma_upper = sqrt(_trace/_k->N);
 // update _sigma & viability boundaries
 if( _constraintsDefined && (_isViabilityRegime == false) )
 {
   updateViabilityBoundaries();

   if ( _previousBestEverValue == _bestEverValue ) _globalSuccessRate = (1-_globalSuccessLearningRate)*_globalSuccessRate;
   else _globalSuccessRate = (1-_globalSuccessLearningRate)*_globalSuccessRate + _globalSuccessLearningRate;
   _sigma *= exp(1.0/_dampFactor*(_globalSuccessRate-(_targetSuccessRate/(1.0-_targetSuccessRate))*(1-_globalSuccessRate)));
 }
 else
 {
   // _sigma *= exp(((sqrt(_rgpsL2Norm)/_chiN)-1.)*_sigmaCumulationFactor/_dampFactor) (orig, alternative)
   _sigma *= exp(std::min(1.0, ((sqrt(_conjugateEvolutionPathL2Norm)/_chiN)-1.)*_sigmaCumulationFactor/_dampFactor));
 }

 if(_k->_verbosity >= KORALI_DETAILED && _sigma > sigma_upper)
     printf("[Korali] Warning: Sigma exceeding inital value of _sigma (%f > %f), increase Initial Standard Deviation of variables.\n", _sigma, sigma_upper);
 
 /* upper bound for _sigma */
 if( _isSigmaBounded && (_sigma > sigma_upper) ) _sigma = sigma_upper;

 /* numerical error management */

 //treat maximal standard deviations
 //TODO

 //treat minimal standard deviations
 for (size_t d = 0; d < _k->N; ++d) if (_sigma * sqrt(_C[d*_k->N+d]) < _variableSettings[d].minStdDevChange)
 {
   _sigma = (_variableSettings[d].minStdDevChange)/sqrt(_C[d*_k->N+d]) * exp(0.05+_sigmaCumulationFactor/_dampFactor);
   if (_k->_verbosity >= KORALI_DETAILED) fprintf(stderr, "[Korali] Warning: Sigma increased due to minimal standard deviation.\n");
 }

 //too low coordinate axis deviations
 //TODO

 //treat numerical precision provblems
 //TODO

 /* Test if function values are identical, escape flat fitness */
 if (_currentBestValue == _fitnessVector[_sortingIndex[(int)_currentSampleMu]]) {
   _sigma *= exp(0.2+_sigmaCumulationFactor/_dampFactor);
   if (_k->_verbosity >= KORALI_DETAILED) {
     fprintf(stderr, "[Korali] Warning: Sigma increased due to equal function values.\n");
   }
 }

}

void CMAES::adaptC(int hsig)
{

 //if (_covarianceMatrixLearningRate != 0.0)
 if (true)
 {
  /* definitions for speeding up inner-most loop */
  //double ccov1  = std::min(_covarianceMatrixLearningRate * (1./_muCovariance) * (_isDiag ? (_k->N+1.5) / 3. : 1.), 1.); (orig, alternative)
  //double ccovmu = std::min(_covarianceMatrixLearningRate * (1-1./_muCovariance) * (_isDiag ? (_k->N+1.5) / 3. : 1.), 1.-ccov1); (orig, alternative)
  double ccov1  = 2.0 / (std::pow(_k->N+1.3,2)+_effectiveMu);
  double ccovmu = std::min(1.0-ccov1,  2.0 * (_effectiveMu-2+1/_effectiveMu) / (std::pow(_k->N+2.0,2)+_effectiveMu));
  double sigmasquare = _sigma * _sigma;

  _isEigenSystemUpdate = false;

  /* update covariance matrix */
  for (size_t d = 0; d < _k->N; ++d)
   for (size_t e = _isDiag ? d : 0; e <= d; ++e) {
     _C[d*_k->N+e] = (1 - ccov1 - ccovmu) * _C[d*_k->N+e] + ccov1 * (_evolutionPath[d] * _evolutionPath[e] + (1-hsig)*ccov1*_cumulativeCovariance*(2.-_cumulativeCovariance) * _C[d*_k->N+e]);
     for (size_t k = 0; k < _currentSampleMu; ++k)
         _C[d*_k->N+e] += ccovmu * _muWeights[k] * (_samplePopulation[_sortingIndex[k]*_k->N + d] - _previousMean[d]) * (_samplePopulation[_sortingIndex[k]*_k->N + e] - _previousMean[e]) / sigmasquare;
     if (e < d) _C[e*_k->N+d] = _C[d*_k->N+e];
   }

  /* update maximal and minimal diagonal value */
  _maxDiagCElement = _minDiagCElement = _C[0];
  for (size_t d = 1; d < _k->N; ++d) {
  if (_maxDiagCElement < _C[d*_k->N+d]) _maxDiagCElement = _C[d*_k->N+d];
  else if (_minDiagCElement > _C[d*_k->N+d])  _minDiagCElement = _C[d*_k->N+d];
  }
 } /* if ccov... */
}

void CMAES::handleConstraints()
{
 size_t initial_resampled = _resampleCount;
 size_t initial_corrections = _adaptationCount;

 while( _maxViolationCount > 0 )
 {
  for (size_t i = 0; i < _k->N; i++) _Ctmp.assign(std::begin(_C), std::end(_C));

  for(size_t i = 0; i < _currentSampleCount; ++i) if (_sampleViolationCounts[i] > 0)
  {
    //update constraint normal
    for( size_t c = 0; c < _k->_fconstraints.size(); c++ ) if ( _viabilityIndicator[c][i] == true )
    {
        _adaptationCount++;

        double v2 = 0;
        for( size_t d = 0; d < _k->N; ++d)
        {
            _constraintNormal[c][d] = (1.0-_normalVectorLearningRate)*_constraintNormal[c][d]+_normalVectorLearningRate*_BDZ[i*_k->N+d];
            v2 += _constraintNormal[c][d]*_constraintNormal[c][d];
        }
        for( size_t d = 0; d < _k->N; ++d)
          for( size_t e = 0; e < _k->N; ++e)
            _Ctmp[d*_k->N+e] = _Ctmp[d*_k->N+e] - ((_beta * _beta * _constraintNormal[c][d]*_constraintNormal[c][e])/(v2*_sampleViolationCounts[i]*_sampleViolationCounts[i]));

        _isEigenSystemUpdate = false;
    }
   }

  updateEigensystem(_Ctmp);

  /* in original some stopping criterion (TOLX) */
  // TODO

  //resample invalid points
  for(size_t i = 0; i < _currentSampleCount; ++i) if(_sampleViolationCounts[i] > 0)
  {
    do
    {
     _resampleCount++;
     sampleSingle(i);

     if(_termCondMaxInfeasibleResamplingsEnabled)
     if(_resampleCount-initial_resampled > _termCondMaxInfeasibleResamplings)
     {
        if(_k->_verbosity >= KORALI_DETAILED) printf("[Korali] Warning: Exiting resampling loop, max resamplings (%zu) reached.\n", _termCondMaxInfeasibleResamplings);
        reEvaluateConstraints();

        return;
     }

    }
    while( isFeasible(i) == false );
  }

  reEvaluateConstraints();

  if(_adaptationCount - initial_corrections > _maxCovMatrixCorrections)
  {
    if(_k->_verbosity >= KORALI_DETAILED) printf("[Korali] Warning: Exiting adaption loop, max adaptions (%zu) reached.\n", _maxCovMatrixCorrections);
    return;
  }

 }//while _maxViolationCount > 0

}

bool CMAES::checkTermination()
{
 
 bool isFinished = false;
 if ( _termCondMinFitnessEnabled && (_isViabilityRegime == false) && (_k->currentGeneration > 1) && (_bestEverValue >= _termCondMinFitness) )
 {
  isFinished = true;
  if(_k->_verbosity >= KORALI_MINIMAL)  printf("[Korali] Min fitness value (%+6.3e) > (%+6.3e).\n",  _bestEverValue, _termCondMinFitness);
 }
 
 if ( _termCondMaxFitnessEnabled && (_isViabilityRegime == false) && (_k->currentGeneration > 1) && (_bestEverValue >= _termCondMaxFitness) )
 {
  isFinished = true;
  if(_k->_verbosity >= KORALI_MINIMAL) printf("[Korali] Max fitness value (%+6.3e) > (%+6.3e)\n",  _bestEverValue, _termCondMaxFitness);
 }

 double range = fabs(_currentBestValue - _previousBestValue);
 if ( _termCondMinFitnessDiffThresholdEnabled && (_k->currentGeneration > 1) && (range <= _termCondMinFitnessDiffThreshold) )
 {
  isFinished = true;
  if(_k->_verbosity >= KORALI_MINIMAL) printf("[Korali] Function value differences (%+6.3e) < (%+6.3e)\n",  range, _termCondMinFitnessDiffThreshold);
 }

 size_t idx;
 
 if ( _termCondMinStandardDeviationEnabled )
 {
  size_t cTemp = 0;
  for(idx = 0; idx <_k->N; ++idx )
   cTemp += (_sigma * sqrt(_C[idx*_k->N+idx]) < _termCondMinStandardDeviation * _variableSettings[idx].initialStdDev) ? 1 : 0;
  
  if (cTemp == _k->N) {
   isFinished = true;
   if(_k->_verbosity >= KORALI_MINIMAL) printf("[Korali] Object variable changes < %+6.3e\n", _termCondMinStandardDeviation * _variableSettings[idx].initialStdDev);
  }

  for(idx = 0; idx <_k->N; ++idx )
   if ( _termCondMaxStandardDeviationEnabled && (_sigma * sqrt(_C[idx*_k->N+idx]) > _termCondMaxStandardDeviation * _variableSettings[idx].initialStdDev) )
   {
    isFinished = true;
    if(_k->_verbosity >= KORALI_MINIMAL) printf("[Korali] Standard deviation increased by more than %7.2e, larger initial standard deviation recommended \n", _termCondMaxStandardDeviation * _variableSettings[idx].initialStdDev);
    break;
   }
 }

 if ( _termCondMaxCovMatrixConditionEnabled && (_maxCovarianceEigenvalue >= _minCovarianceEigenvalue * _termCondMaxCovMatrixCondition) )
 {
   isFinished = true;
   if(_k->_verbosity >= KORALI_MINIMAL) printf("[Korali] Maximal condition number %7.2e reached. _maxCovarianceEigenvalue=%7.2e, minEig=%7.2e, _maxDiagCElement=%7.2e, _minDiagCElement=%7.2e\n",
                                _termCondMaxCovMatrixCondition, _maxCovarianceEigenvalue, _minCovarianceEigenvalue, _maxDiagCElement, _minDiagCElement);
 }

 double fac;
 size_t iAchse = 0;
 size_t iKoo = 0;
 /* Component of _mean is not changed anymore */
 if( _termCondMinStandardDeviationStepFactorEnabled)
 if (!_isDiag )
 {
    for (iAchse = 0; iAchse < _k->N; ++iAchse)
    {
    fac = _termCondMinStandardDeviationStepFactor * _sigma * _axisD[iAchse];
    for (iKoo = 0; iKoo < _k->N; ++iKoo){
      if (_mean[iKoo] != _mean[iKoo] + fac * _B[iKoo*_k->N+iAchse])
      break;
    }
    if (iKoo == _k->N)
    {
      isFinished = true;
      if(_k->_verbosity >= KORALI_MINIMAL) printf("[Korali] Standard deviation %f*%7.2e in principal axis %ld without effect.\n", _termCondMinStandardDeviationStepFactor, _sigma*_axisD[iAchse], iAchse);
      break;
    }
  }
 }

 /* Component of _mean is not changed anymore */
 if( _termCondMinStandardDeviationStepFactorEnabled )
 for (iKoo = 0; iKoo < _k->N; ++iKoo)
 {
  if (_mean[iKoo] == _mean[iKoo] + _termCondMinStandardDeviationStepFactor*_sigma*sqrt(_C[iKoo*_k->N+iKoo]) )
  {
   isFinished = true;
   if(_k->_verbosity >= KORALI_MINIMAL) printf("[Korali] Standard deviation %f*%7.2e in coordinate %ld without effect.\n", _termCondMinStandardDeviationStepFactor, _sigma*sqrt(_C[iKoo*_k->N+iKoo]), iKoo);
   break;
  }

 }

 if( _termCondMaxGenerationsEnabled && (_k->currentGeneration >= _termCondMaxGenerations) )
 {
  isFinished = true;
  if(_k->_verbosity >= KORALI_MINIMAL) printf("[Korali] Maximum number of Generations reached (%lu).\n", _termCondMaxGenerations);
 }

 return isFinished;
}


void CMAES::updateEigensystem(std::vector<double>& M, int flgforce)
{
 if(flgforce == 0 && _isEigenSystemUpdate) return;
 /* if(_k->currentGeneration % _covarianceEigenEvalFreq == 0) return; */

 eigen(_k->N, M, _axisDtmp, _Btmp);
 
 /* find largest and smallest eigenvalue, they are supposed to be sorted anyway */
 double minEWtmp = *std::min_element(std::begin(_axisDtmp), std::end(_axisDtmp));
 double maxEWtmp = *std::max_element(std::begin(_axisDtmp), std::end(_axisDtmp));

 if (minEWtmp <= 0.0) 
 { if(_k->_verbosity >= KORALI_NORMAL) printf("[Korali] Warning: Min Eigenvalue smaller or equal 0.0 (%+6.3e) after Eigen decomp (no update possible).\n", minEWtmp ); return; }

 for (size_t d = 0; d < _k->N; ++d) 
 {
     _axisDtmp[d] = sqrt(_axisDtmp[d]); 
     if (std::isfinite(_axisDtmp[d]) == false)
     {
       if(_k->_verbosity >= KORALI_NORMAL) printf("[Korali] Warning: Could not calculate root of Eigenvalue (%+6.3e) after Eigen decomp (no update possible).\n", _axisDtmp[d] );
       return; 
     }
    for (size_t e = 0; e < _k->N; ++e) if (std::isfinite(_B[d*_k->N+e]) == false)
    {
       if(_k->_verbosity >= KORALI_NORMAL) printf("[Korali] Warning: Non finite value detected in B (no update possible).\n");
       return;
    }
 }
 
 /* write back */
 for (size_t d = 0; d < _k->N; ++d) _axisD[d] = _axisDtmp[d];
 _B.assign(std::begin(_Btmp), std::end(_Btmp));

 _minCovarianceEigenvalue = minEWtmp;
 _maxCovarianceEigenvalue = maxEWtmp;

 _isEigenSystemUpdate = true;
}


/************************************************************************/
/*                    Additional Methods                                */
/************************************************************************/


void CMAES::eigen(size_t size, std::vector<double>& M,  std::vector<double>& diag, std::vector<double>& Q) const
{
 std::vector<double> data(size * size);

 for (size_t i = 0; i <  size; i++)
 for (size_t j = 0; j <= i; j++)
 {
  data[i*size + j] = M[i*_k->N+j];
  data[j*size + i] = M[i*_k->N+j];
 }

 // GSL Workspace
 gsl_vector* gsl_eval = gsl_vector_alloc(_k->N);
 gsl_matrix* gsl_evec = gsl_matrix_alloc(_k->N, _k->N);
 gsl_eigen_symmv_workspace* gsl_work =  gsl_eigen_symmv_alloc(_k->N);

 gsl_matrix_view m = gsl_matrix_view_array (data.data(), size, size);

 gsl_eigen_symmv (&m.matrix, gsl_eval, gsl_evec, gsl_work);
 gsl_eigen_symmv_sort (gsl_eval, gsl_evec, GSL_EIGEN_SORT_ABS_ASC);

 for (size_t i = 0; i < size; i++)
 {
  gsl_vector_view gsl_evec_i = gsl_matrix_column (gsl_evec, i);
  for (size_t j = 0; j < size; j++) Q[j*_k->N+i] = gsl_vector_get (&gsl_evec_i.vector, j);
 }

 for (size_t i = 0; i < size; i++) diag[i] = gsl_vector_get (gsl_eval, i);

 gsl_vector_free(gsl_eval);
 gsl_matrix_free(gsl_evec);
 gsl_eigen_symmv_free(gsl_work);
}


void CMAES::sort_index(const std::vector<double>& vec, std::vector<size_t>& _sortingIndex, size_t n) const
{
  // initialize original _sortingIndex locations
  std::iota(std::begin(_sortingIndex), std::begin(_sortingIndex)+n, (size_t) 0);

  // sort indexes based on comparing values in vec
  std::sort(std::begin(_sortingIndex), std::begin(_sortingIndex)+n, [vec](size_t i1, size_t i2) { return vec[i1] > vec[i2]; } );

}


void CMAES::printGeneration()
{
 if (_k->_verbosity >= KORALI_NORMAL) printf("[Korali] CMA-ES Generation %zu\n", _k->currentGeneration);

 if ( _constraintsDefined && _isViabilityRegime && _k->_verbosity >= KORALI_NORMAL)
 {
   printf("\n[Korali] Searching start (MeanX violates constraints) .. \n");
   printf("[Korali] Viability Bounds:\n");
   for (size_t c = 0; c < _k->_fconstraints.size(); c++)  printf("         %s = (%+6.3e)\n", _k->_variables[c]->_name.c_str(), _viabilityBoundaries[c]);
   printf("\n");
 }

 if (_k->_verbosity >= KORALI_NORMAL)
 {
  printf("[Korali] Sigma:                        %+6.3e\n", _sigma);
  printf("[Korali] Current Function Value: Max = %+6.3e - Best = %+6.3e\n", _currentBestValue, _bestEverValue);
  printf("[Korali] Diagonal Covariance:    Min = %+6.3e -  Max = %+6.3e\n", _minDiagCElement, _maxDiagCElement);
  printf("[Korali] Covariance Eigenvalues: Min = %+6.3e -  Max = %+6.3e\n", _minCovarianceEigenvalue, _maxCovarianceEigenvalue);
 }

 if (_k->_verbosity >= KORALI_DETAILED)
 {
  printf("[Korali] Variable = (MeanX, BestX):\n");
  for (size_t d = 0; d < _k->N; d++)  printf("         %s = (%+6.3e, %+6.3e)\n", _k->_variables[d]->_name.c_str(), _mean[d], _bestEverSample[d]);

  printf("[Korali] Constraint Evaluation at Current Function Value:\n");
  if ( _constraintsDefined )
  {
   if ( _bestValidSample >= 0 )
      for (size_t c = 0; c < _k->_fconstraints.size(); c++) printf("         ( %+6.3e )\n", _constraintEvaluations[c][_bestValidSample]);
   else
      for (size_t c = 0; c < _k->_fconstraints.size(); c++) printf("         ( %+6.3e )\n", _constraintEvaluations[c][0]);
  }

  printf("[Korali] Covariance Matrix:\n");
  for (size_t d = 0; d < _k->N; d++)
  {
   for (size_t e = 0; e <= d; e++) printf("   %+6.3e  ",_C[d*_k->N+e]);
   printf("\n");
  }

  printf("[Korali] Number of Infeasible Samples: %zu\n", _infeasibleSampleCount);
  if ( _constraintsDefined ) { printf("[Korali] Number of Constraint Evaluations: %zu\n", _constraintEvaluationCount); printf("[Korali] Number of Matrix Corrections: %zu\n", _adaptationCount ); }
 }

 if (_k->_verbosity >= KORALI_NORMAL)
   printf("--------------------------------------------------------------------\n");
}

void CMAES::finalize()
{
 if (_k->_verbosity >= KORALI_MINIMAL)
 {
    printf("[Korali] CMA-ES Finished\n");
    printf("[Korali] Optimum (%s) found: %e\n", _objective.c_str(), _bestEverValue);
    printf("[Korali] Optimum (%s) found at:\n", _objective.c_str());
    for (size_t d = 0; d < _k->N; ++d) printf("         %s = %+6.3e\n", _k->_variables[d]->_name.c_str(), _bestEverSample[d]);
    if ( _constraintsDefined ) { printf("[Korali] Constraint Evaluation at Optimum:\n"); for (size_t c = 0; c < _k->_fconstraints.size(); c++) printf("         ( %+6.3e )\n", _bestEverConstraintEvaluation[c]); }
    printf("[Korali] Number of Infeasible Samples: %zu\n", _infeasibleSampleCount);
    printf("--------------------------------------------------------------------\n");
 }
}
