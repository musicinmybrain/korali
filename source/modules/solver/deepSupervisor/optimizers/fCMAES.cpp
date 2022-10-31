#include "fCMAES.hpp"
#include <algorithm> // std::sort
#include <numeric>   // std::iota
#include <stdio.h>
#include <stdexcept>

namespace korali
{
fCMAES::fCMAES(size_t nVars, size_t populationSize, size_t muSize)
{
  _currentGeneration = 1;

  // CMAES Parameters
  _populationSize = populationSize;
  _muValue = 0;
  _initialSigmaCumulationFactor = -1.0;
  _initialDampFactor = -1.0;
  _isSigmaBounded = false;
  _initialCumulativeCovariance = -1.0;
  _isDiagonal = false;
  _muType = "Linear";

  // Variable Parameters
  _nVars = nVars;
  _initialMeans.resize(_nVars);
  _initialStandardDeviations.resize(_nVars);
  _lowerBounds.resize(_nVars);
  _upperBounds.resize(_nVars);
  _minMeanUpdates.resize(_nVars);

  for (size_t i = 0; i < _nVars; i++)
  {
    _minMeanUpdates[i] = -std::numeric_limits<float>::infinity();
    _lowerBounds[i] = -std::numeric_limits<float>::infinity();
    _upperBounds[i] = +std::numeric_limits<float>::infinity();
    _initialMeans[i] = std::numeric_limits<float>::signaling_NaN();
    _initialStandardDeviations[i] = std::numeric_limits<float>::signaling_NaN();
  }

  // Termination Criteria
  _maxGenerations = 10000000;
  _maxInfeasibleResamplings = 10000000;
  _maxConditionCovarianceMatrix = std::numeric_limits<float>::infinity();
  _minValue = -std::numeric_limits<float>::infinity();
  _maxValue = +std::numeric_limits<float>::infinity();
  _minValueDifferenceThreshold = -std::numeric_limits<float>::infinity();
  _targetMaxStandardDeviation = -std::numeric_limits<float>::infinity();

  // Random Number Generators
  _normalGenerator = std::normal_distribution<float>(0.0, 1.0);

  // Statistics
  _bestEverValue = -std::numeric_limits<float>::infinity();
  _currentMinStandardDeviation = -std::numeric_limits<float>::infinity();
  _currentMaxStandardDeviation = +std::numeric_limits<float>::infinity();
  _minimumCovarianceEigenvalue = -std::numeric_limits<float>::infinity();
  _maximumCovarianceEigenvalue = +std::numeric_limits<float>::infinity();

  if (_populationSize == 0) _populationSize = ceil(4.0 + floor(3 * log((float)_nVars)));
  if (_muValue == 0) _muValue = _populationSize / 2;

  _chiSquareNumber = sqrtf((float)_nVars) * (1. - 1. / (4. * _nVars) + 1. / (21. * _nVars * _nVars));

  // Allocating Memory
  _samplePopulation.resize(_populationSize);
  for (size_t i = 0; i < _populationSize; i++) _samplePopulation[i].resize(_nVars);

  _evolutionPath.resize(_nVars);
  _conjugateEvolutionPath.resize(_nVars);
  _auxiliarBDZMatrix.resize(_nVars);
  _meanUpdate.resize(_nVars);
  _currentMean.resize(_nVars);
  _previousMean.resize(_nVars);
  _bestEverVariables.resize(_nVars);
  _axisLengths.resize(_nVars);
  _auxiliarAxisLengths.resize(_nVars);
  _currentBestVariables.resize(_nVars);

  _sortingIndex.resize(_populationSize);
  _valueVector.resize(_populationSize);

  _covarianceMatrix.resize(_nVars * _nVars);
  _auxiliarCovarianceMatrix.resize(_nVars * _nVars);
  _covarianceEigenvectorMatrix.resize(_nVars * _nVars);
  _auxiliarCovarianceEigenvectorMatrix.resize(_nVars * _nVars);
  _bDZMatrix.resize(_populationSize * _nVars);

  _muWeights.resize(_muValue);

  // GSL Workspace
  _gsl_eval = gsl_vector_alloc(_nVars);
  _gsl_evec = gsl_matrix_alloc(_nVars, _nVars);
  _gsl_work = gsl_eigen_symmv_alloc(_nVars);
}

void fCMAES::reset()
{
  _currentGeneration = 1;

  // Establishing optimization goal
  _bestEverValue = -std::numeric_limits<float>::infinity();
  _previousBestEverValue = -std::numeric_limits<float>::infinity();
  _previousBestValue = -std::numeric_limits<float>::infinity();
  _currentBestValue = -std::numeric_limits<float>::infinity();

  _globalSuccessRate = -1.0;
  _covarianceMatrixAdaptionFactor = -1.0;
  _covarianceMatrixAdaptationCount = 0;

  initMuWeights(_muValue);
  initCovariance();
  _infeasibleSampleCount = 0;
  _conjugateEvolutionPathL2Norm = 0.0;

  for (size_t i = 0; i < _nVars; i++) _currentMean[i] = _previousMean[i] = _initialMeans[i];

  _currentMinStandardDeviation = +std::numeric_limits<float>::infinity();
  _currentMaxStandardDeviation = -std::numeric_limits<float>::infinity();
}

void fCMAES::initMuWeights(size_t numsamplesmu)
{
  // Initializing Mu Weights
  bool muTypeRecognized = false;

  if (_muType == "Linear") { for (size_t i = 0; i < numsamplesmu; i++) _muWeights[i] = numsamplesmu - i; muTypeRecognized = true; }
  if (_muType == "Equal") { for (size_t i = 0; i < numsamplesmu; i++) _muWeights[i] = 1.; muTypeRecognized = true; }
  if (_muType == "Logarithmic") { for (size_t i = 0; i < numsamplesmu; i++) _muWeights[i] = log(std::max((float)numsamplesmu, 0.5f * _populationSize) + 0.5f) - log(i + 1.0f); muTypeRecognized = true; }
  if (muTypeRecognized == false) throw std::runtime_error("Invalid setting of Mu Type (Linear, Equal, or Logarithmic accepted).");

  // Normalize weights vector and set mueff
  float s1 = 0.0;
  float s2 = 0.0;

  for (size_t i = 0; i < numsamplesmu; i++)
  {
    s1 += _muWeights[i];
    s2 += _muWeights[i] * _muWeights[i];
  }
  _effectiveMu = s1 * s1 / s2;

  for (size_t i = 0; i < numsamplesmu; i++) _muWeights[i] /= s1;

  // Setting Cumulative Covariancea
  if ((_initialCumulativeCovariance <= 0) || (_initialCumulativeCovariance > 1))
    _cumulativeCovariance = (4.0 + _effectiveMu / (1.0 * _nVars)) / (_nVars + 4.0 + 2.0 * _effectiveMu / (1.0 * _nVars));
  else
    _cumulativeCovariance = _initialCumulativeCovariance;

  // Setting Sigma Cumulation Factor
  _sigmaCumulationFactor = _initialSigmaCumulationFactor;
  if (_sigmaCumulationFactor <= 0 || _sigmaCumulationFactor >= 1)
    _sigmaCumulationFactor = (_effectiveMu + 2.0) / (_nVars + _effectiveMu + 3.0);

  // Setting Damping Factor
  _dampFactor = _initialDampFactor;
  if (_dampFactor <= 0.0)
    _dampFactor = (1.0 + 2 * std::max(0.0f, sqrtf((_effectiveMu - 1.0) / (_nVars + 1.0)) - 1)) + _sigmaCumulationFactor;
}

void fCMAES::initCovariance()
{
  // Setting Sigma
  _trace = 0.0;
  for (size_t i = 0; i < _nVars; ++i) _trace += _initialStandardDeviations[i] * _initialStandardDeviations[i];
  _sigma = sqrtf(_trace / _nVars);

  // Setting B, C and _axisD
  for (size_t i = 0; i < _nVars; ++i)
  {
    _covarianceEigenvectorMatrix[i * _nVars + i] = 1.0;
    _covarianceMatrix[i * _nVars + i] = _axisLengths[i] = _initialStandardDeviations[i] * sqrtf(_nVars / _trace);
    _covarianceMatrix[i * _nVars + i] *= _covarianceMatrix[i * _nVars + i];
  }

  _minimumCovarianceEigenvalue = *std::min_element(std::begin(_axisLengths), std::end(_axisLengths));
  _maximumCovarianceEigenvalue = *std::max_element(std::begin(_axisLengths), std::end(_axisLengths));

  _minimumCovarianceEigenvalue = _minimumCovarianceEigenvalue * _minimumCovarianceEigenvalue;
  _maximumCovarianceEigenvalue = _maximumCovarianceEigenvalue * _maximumCovarianceEigenvalue;

  _maximumDiagonalCovarianceMatrixElement = _covarianceMatrix[0];

  for (size_t i = 1; i < _nVars; ++i)
    if (_maximumDiagonalCovarianceMatrixElement < _covarianceMatrix[i * _nVars + i]) _maximumDiagonalCovarianceMatrixElement = _covarianceMatrix[i * _nVars + i];

  _minimumDiagonalCovarianceMatrixElement = _covarianceMatrix[0];

  for (size_t i = 1; i < _nVars; ++i)
    if (_minimumDiagonalCovarianceMatrixElement > _covarianceMatrix[i * _nVars + i]) _minimumDiagonalCovarianceMatrixElement = _covarianceMatrix[i * _nVars + i];
}

bool fCMAES::isSampleFeasible(const std::vector<float> &sample)
{
  for (size_t i = 0; i < sample.size(); i++)
  {
    if (std::isfinite(sample[i]) == false) return false;
    if (sample[i] < _lowerBounds[i]) return false;
    if (sample[i] > _upperBounds[i]) return false;
  }
  return true;
}

void fCMAES::prepareGeneration()
{
  for (size_t d = 0; d < _nVars; ++d) _auxiliarCovarianceMatrix = _covarianceMatrix;
  updateEigensystem(_auxiliarCovarianceMatrix);
  bool isFeasible;
  for (size_t i = 0; i < _populationSize; ++i)
    do
    {
      sampleSingle(i);

      isFeasible = isSampleFeasible(_samplePopulation[i]);

      if (isFeasible == false) _infeasibleSampleCount++;

    } while (isFeasible == false);
}

void fCMAES::sampleSingle(size_t sampleIdx)
{
  for (size_t d = 0; d < _nVars; ++d)
  {
    float randomNumber = _normalGenerator(_randomGenerator);

    if (_isDiagonal)
    {
      _bDZMatrix[sampleIdx * _nVars + d] = _axisLengths[d] * randomNumber;
      _samplePopulation[sampleIdx][d] = _currentMean[d] + _sigma * _bDZMatrix[sampleIdx * _nVars + d];
    }
    else
      _auxiliarBDZMatrix[d] = _axisLengths[d] * randomNumber;
  }

  if (!_isDiagonal)
    for (size_t d = 0; d < _nVars; ++d)
    {
      _bDZMatrix[sampleIdx * _nVars + d] = 0.0;
      for (size_t e = 0; e < _nVars; ++e) _bDZMatrix[sampleIdx * _nVars + d] += _covarianceEigenvectorMatrix[d * _nVars + e] * _auxiliarBDZMatrix[e];

      //auto variance = _sigma * _bDZMatrix[sampleIdx * _nVars + d];
      //      if (variance > abs(_upperBounds[d] - _lowerBounds[d])*10.0) printf("Excessive Variance\n");
      _samplePopulation[sampleIdx][d] = _currentMean[d] + _sigma * _bDZMatrix[sampleIdx * _nVars + d];
    }
}

void fCMAES::updateDistribution(const std::vector<float> &evaluations)
{
  _valueVector = evaluations;

  /* Generate _sortingIndex */
  sort_index(_valueVector, _sortingIndex, _populationSize);

  size_t bestSampleIdx = _sortingIndex[0];

  /* update function value history */
  _previousBestValue = _currentBestValue;

  /* update current best */
  _currentBestValue = _valueVector[bestSampleIdx];

  for (size_t d = 0; d < _nVars; ++d) _currentBestVariables[d] = _samplePopulation[bestSampleIdx][d];

  /* update xbestever */
  if (_currentBestValue > _bestEverValue || _currentGeneration == 1)
  {
    _previousBestEverValue = _bestEverValue;
    _bestEverValue = _currentBestValue;

    for (size_t d = 0; d < _nVars; ++d)
      _bestEverVariables[d] = _currentBestVariables[d];
  }

  /* set weights */
  for (size_t d = 0; d < _nVars; ++d)
  {
    _previousMean[d] = _currentMean[d];
    _currentMean[d] = 0.;
    for (size_t i = 0; i < _muValue; ++i)
      _currentMean[d] += _muWeights[i] * _samplePopulation[_sortingIndex[i]][d];

    _meanUpdate[d] = (_currentMean[d] - _previousMean[d]) / _sigma;
  }

  /* calculate z := D^(-1) * B^(T) * _meanUpdate into _auxiliarBDZMatrix */
  for (size_t d = 0; d < _nVars; ++d)
  {
    float sum = 0.0;
    if (_isDiagonal)
      sum = _meanUpdate[d];
    else
      for (size_t e = 0; e < _nVars; ++e) sum += _covarianceEigenvectorMatrix[e * _nVars + d] * _meanUpdate[e]; /* B^(T) * _meanUpdate ( iterating B[e][d] = B^(T) ) */

    _auxiliarBDZMatrix[d] = sum / _axisLengths[d]; /* D^(-1) * B^(T) * _meanUpdate */
  }

  _conjugateEvolutionPathL2Norm = 0.0;

  /* cumulation for _sigma (ps) using B*z */
  for (size_t d = 0; d < _nVars; ++d)
  {
    float sum = 0.0;
    if (_isDiagonal)
      sum = _auxiliarBDZMatrix[d];
    else
      for (size_t e = 0; e < _nVars; ++e) sum += _covarianceEigenvectorMatrix[d * _nVars + e] * _auxiliarBDZMatrix[e];

    _conjugateEvolutionPath[d] = (1.0f - _sigmaCumulationFactor) * _conjugateEvolutionPath[d] + sqrtf(_sigmaCumulationFactor * (2.0f - _sigmaCumulationFactor) * _effectiveMu) * sum;

    /* calculate norm(ps)^2 */
    _conjugateEvolutionPathL2Norm += std::pow(_conjugateEvolutionPath[d], 2.0);
  }
  _conjugateEvolutionPathL2Norm = sqrtf(_conjugateEvolutionPathL2Norm);

  int hsig = (1.4f + 2.0f / ((float)_nVars + 1)) > (_conjugateEvolutionPathL2Norm / sqrtf(1.0f - pow(1.0f - _sigmaCumulationFactor, 2.0f * (1.0f + _currentGeneration))) / _chiSquareNumber);

  /* cumulation for covariance matrix (pc) using B*D*z~_nVars(0,C) */
  for (size_t d = 0; d < _nVars; ++d)
    _evolutionPath[d] = (1.0f - _cumulativeCovariance) * _evolutionPath[d] + hsig * sqrtf(_cumulativeCovariance * (2.0f - _cumulativeCovariance) * _effectiveMu) * _meanUpdate[d];

  /* update covariance matrix  */
  adaptC(hsig);

  /* update sigma */
  updateSigma();

  _currentMinStandardDeviation = std::numeric_limits<float>::infinity();
  _currentMaxStandardDeviation = -std::numeric_limits<float>::infinity();

  // Calculating current Minimum and Maximum STD Devs
  for (size_t i = 0; i < _nVars; ++i)
  {
    _currentMinStandardDeviation = std::min(_currentMinStandardDeviation, _sigma * sqrtf(_covarianceMatrix[i * _nVars + i]));
    _currentMaxStandardDeviation = std::max(_currentMaxStandardDeviation, _sigma * sqrtf(_covarianceMatrix[i * _nVars + i]));
  }
}

void fCMAES::adaptC(int hsig)
{
  float ccov1 = 2.0f / (std::pow(_nVars + 1.3f, 2.0f) + _effectiveMu);
  float ccovmu = std::min(1.0f - ccov1, 2.0f * (_effectiveMu - 2.0f + 1.0f / _effectiveMu) / (std::pow(_nVars + 2.0f, 2.0f) + _effectiveMu));
  float sigmasquare = _sigma * _sigma;

  /* update covariance matrix */
  for (size_t d = 0; d < _nVars; ++d)
    for (size_t e = _isDiagonal ? d : 0; e <= d; ++e)
    {
      _covarianceMatrix[d * _nVars + e] = (1 - ccov1 - ccovmu) * _covarianceMatrix[d * _nVars + e] + ccov1 * (_evolutionPath[d] * _evolutionPath[e] + (1.0f - hsig) * _cumulativeCovariance * (2.0f - _cumulativeCovariance) * _covarianceMatrix[d * _nVars + e]);

      for (size_t k = 0; k < _muValue; ++k)
        _covarianceMatrix[d * _nVars + e] += ccovmu * _muWeights[k] * (_samplePopulation[_sortingIndex[k]][d] - _previousMean[d]) * (_samplePopulation[_sortingIndex[k]][e] - _previousMean[e]) / sigmasquare;

      if (e < d) _covarianceMatrix[e * _nVars + d] = _covarianceMatrix[d * _nVars + e];
    }

  /* update maximal and minimal diagonal value */
  _maximumDiagonalCovarianceMatrixElement = _minimumDiagonalCovarianceMatrixElement = _covarianceMatrix[0];
  for (size_t d = 1; d < _nVars; ++d)
  {
    if (_maximumDiagonalCovarianceMatrixElement < _covarianceMatrix[d * _nVars + d])
      _maximumDiagonalCovarianceMatrixElement = _covarianceMatrix[d * _nVars + d];
    else if (_minimumDiagonalCovarianceMatrixElement > _covarianceMatrix[d * _nVars + d])
      _minimumDiagonalCovarianceMatrixElement = _covarianceMatrix[d * _nVars + d];
  }
}

void fCMAES::updateSigma()
{
  _sigma *= exp(_sigmaCumulationFactor / _dampFactor * (_conjugateEvolutionPathL2Norm / _chiSquareNumber - 1.));

  /* escape flat evaluation */
  if (_currentBestValue == _valueVector[_sortingIndex[(int)_muValue]])
  {
    _sigma *= exp(0.2 + _sigmaCumulationFactor / _dampFactor);
    //fprintf(stderr, "Sigma increased due to equal function values.\n");
  }

  /* upper bound check for _sigma */
  float _upperBound = sqrtf(_trace / _nVars);

  if (_sigma > _upperBound)
  {
    //fprintf(stderr, "[fCMAES] Sigma exceeding inital value of _sigma (%f > %f), increase Initial Standard Deviation of variables.\n", _sigma, _upperBound);
    if (_isSigmaBounded)
    {
      _sigma = _upperBound;
      //fprintf(stderr, "[fCMAES] Sigma set to upper bound (%f) due to solver configuration 'Is Sigma Bounded' = 'true'.\n", _sigma);
    }
  }
}

void fCMAES::updateEigensystem(std::vector<float> &M)
{
  eigen(_nVars, M, _auxiliarAxisLengths, _auxiliarCovarianceEigenvectorMatrix);

  /* find largest and smallest eigenvalue, they are supposed to be sorted anyway */
  float minCovEVal = *std::min_element(std::begin(_auxiliarAxisLengths), std::end(_auxiliarAxisLengths));

  for (size_t d = 0; d < _nVars; ++d)
    _auxiliarAxisLengths[d] = sqrtf(_auxiliarAxisLengths[d]);

  /* write back */
  _minimumCovarianceEigenvalue = minCovEVal;
  _maximumCovarianceEigenvalue = *std::max_element(std::begin(_auxiliarAxisLengths), std::end(_auxiliarAxisLengths));
  for (size_t d = 0; d < _nVars; ++d) _axisLengths[d] = _auxiliarAxisLengths[d];
  _covarianceEigenvectorMatrix.assign(std::begin(_auxiliarCovarianceEigenvectorMatrix), std::end(_auxiliarCovarianceEigenvectorMatrix));
}

/************************************************************************/
/*                    Additional Methods                                */
/************************************************************************/

void fCMAES::eigen(size_t size, std::vector<float> &M, std::vector<float> &diag, std::vector<float> &Q) const
{
  std::vector<double> data(size * size);

  for (size_t i = 0; i < size; i++)
    for (size_t j = 0; j <= i; j++)
    {
      data[i * size + j] = M[i * _nVars + j];
      data[j * size + i] = M[i * _nVars + j];
    }

  gsl_matrix_view m = gsl_matrix_view_array(data.data(), size, size);

  gsl_eigen_symmv(&m.matrix, _gsl_eval, _gsl_evec, _gsl_work);
  gsl_eigen_symmv_sort(_gsl_eval, _gsl_evec, GSL_EIGEN_SORT_ABS_ASC);

  for (size_t i = 0; i < size; i++)
  {
    gsl_vector_view gsl_evec_i = gsl_matrix_column(_gsl_evec, i);
    for (size_t j = 0; j < size; j++) Q[j * _nVars + i] = gsl_vector_get(&gsl_evec_i.vector, j);
  }

  for (size_t i = 0; i < size; i++) diag[i] = gsl_vector_get(_gsl_eval, i);
}

void fCMAES::sort_index(const std::vector<float> &vec, std::vector<size_t> &sortingIndex, size_t N) const
{
  // initialize original sortingIndex locations
  std::iota(std::begin(sortingIndex), std::begin(sortingIndex) + N, (size_t)0);

  // sort indexes based on comparing values in vec
  std::sort(std::begin(sortingIndex), std::begin(sortingIndex) + N, [vec](size_t i1, size_t i2) { return vec[i1] > vec[i2]; });
}

void fCMAES::printInfo()
{
  fprintf(stderr, "[fCMAES] Sigma:                        %+6.3e\n", _sigma);
  fprintf(stderr, "[fCMAES] Current Function Value: Max = %+6.3e - Best = %+6.3e\n", _currentBestValue, _bestEverValue);
  fprintf(stderr, "[fCMAES] Diagonal Covariance:    Min = %+6.3e -  Max = %+6.3e\n", _minimumDiagonalCovarianceMatrixElement, _maximumDiagonalCovarianceMatrixElement);
  fprintf(stderr, "[fCMAES] Covariance Eigenvalues: Min = %+6.3e -  Max = %+6.3e\n", _minimumCovarianceEigenvalue, _maximumCovarianceEigenvalue);

  fprintf(stderr, "[fCMAES] Variable = (MeanX, BestX):\n");
  for (size_t d = 0; d < _nVars; d++) fprintf(stderr, "        Var %lu = (%+6.3e, %+6.3e)\n", d, _currentMean[d], _bestEverVariables[d]);

  fprintf(stderr, "[fCMAES] Covariance Matrix:\n");
  for (size_t d = 0; d < _nVars; d++)
  {
    for (size_t e = 0; e <= d; e++) fprintf(stderr, "   %+6.3e  ", _covarianceMatrix[d * _nVars + e]);
    fprintf(stderr, "[fCMAES] \n");
  }

  fprintf(stderr, "[fCMAES] Number of Infeasible Samples: %zu\n", _infeasibleSampleCount);
}

void fCMAES::setSeed(size_t seed)
{
  _randomGenerator.seed(seed);
}

fCMAES::~fCMAES()
{
  gsl_vector_free(_gsl_eval);
  gsl_matrix_free(_gsl_evec);
  gsl_eigen_symmv_free(_gsl_work);
}

bool fCMAES::checkTermination()
{
  if (_currentGeneration > 1 && ((_maxInfeasibleResamplings > 0) && (_infeasibleSampleCount >= _maxInfeasibleResamplings))) return true;
  if (_currentGeneration > 1 && (_maximumCovarianceEigenvalue >= _maxConditionCovarianceMatrix * _minimumCovarianceEigenvalue)) return true;
  if (_currentGeneration > 1 && (-_bestEverValue < _minValue)) return true;
  if (_currentGeneration > 1 && (+_bestEverValue > _maxValue)) return true;
  if (_currentGeneration > 1 && (fabs(_currentBestValue - _previousBestValue) < _minValueDifferenceThreshold)) return true;
  if (_currentGeneration > 1 && (_currentMaxStandardDeviation <= _targetMaxStandardDeviation)) return true;

  if (_currentGeneration > 1)
  {
    bool areMeansAbove = false;
    for (size_t d = 0; d < _nVars; d++)
      if (std::fabs(_meanUpdate[d]) > _minMeanUpdates[d]) areMeansAbove = true;
    if (areMeansAbove == false) return true;
  }

  if (_currentGeneration >= _maxGenerations) return true;

  return false;
}

} // namespace korali
