/**************************************************************
 * A single-precision fast version of CMAES for Learning
 **************************************************************/

#ifndef _KORALI_FAST_CMAES_HPP_
#define _KORALI_FAST_CMAES_HPP_

#include <cstddef>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf.h>
#include <random>
#include <vector>

namespace korali
{
/**
* @brief Class declaration for module: fCMAES.
*/
class fCMAES
{
  public:
  /**
 * @brief Default constructor for the optimizer
 * @param nVars Variable-space dimensionality
 * @param populationSize How many samples per generation to use
 * @param muSize How many sample selections for covariance matrix adaptation
 */
  fCMAES(size_t nVars, size_t populationSize = 0, size_t muSize = 0);

  /**
  * @brief Default destructor for the optimizer
  */
  ~fCMAES();

  /**
  * @brief Number of problem variables
  */
  size_t _nVars;

  /**
   * @brief Counter for the current generation
   */
  size_t _currentGeneration;

  /**
   * @brief Indicates the initial gaussian means for all variables
   */
  std::vector<float> _initialMeans;

  /**
   * @brief Indicates the initial gaussian standard deviations for all variables
   */
  std::vector<float> _initialStandardDeviations;

  /**
   * @brief Indicates the lower bounds for all variables
   */
  std::vector<float> _lowerBounds;

  /**
   * @brief Indicates the upper bounds for all variables
   */
  std::vector<float> _upperBounds;

  /**
   * @brief Random number generator
   */
  std::default_random_engine _randomGenerator;

  /**
   * @brief Gaussian number generator
   */
  std::normal_distribution<float> _normalGenerator;

  /**
* @brief Specifies the number of samples to evaluate per generation (default: $4+3*log(N)$, where $N$ is the number of variables).
*/
  size_t _populationSize;
  /**
* @brief Number of best samples used to update the covariance matrix and the mean (by default it is half the Sample Count).
*/
  size_t _muValue;
  /**
* @brief Weights given to the Mu best values to update the covariance matrix and the mean.
*/
  std::string _muType;
  /**
* @brief Controls the learning rate of the conjugate evolution path (by default this variable is internally calibrated).
*/
  float _initialSigmaCumulationFactor;
  /**
* @brief Controls the updates of the covariance matrix scaling factor (by default this variable is internally calibrated).
*/
  float _initialDampFactor;
  /**
* @brief Sets an upper bound for the covariance matrix scaling factor. The upper bound is given by the average of the initial standard deviation of the variables.
*/
  int _isSigmaBounded;
  /**
* @brief Controls the learning rate of the evolution path for the covariance update (must be in (0,1], by default this variable is internally calibrated).
*/
  float _initialCumulativeCovariance;
  /**
* @brief Covariance matrix updates will be optimized for diagonal matrices.
*/
  int _isDiagonal;
  /**
* @brief [Internal Use] Objective function values.
*/
  std::vector<float> _valueVector;
  /**
* @brief [Internal Use] Objective function values from previous generation.
*/
  std::vector<float> _previousValueVector;
  /**
* @brief [Internal Use] Weights for each of the Mu samples.
*/
  std::vector<float> _muWeights;
  /**
* @brief [Internal Use] Variance effective selection mass.
*/
  float _effectiveMu;
  /**
* @brief [Internal Use] Increment for sigma, calculated from muEffective and dimension.
*/
  float _sigmaCumulationFactor;
  /**
* @brief [Internal Use] Dampening parameter controls step size adaption.
*/
  float _dampFactor;
  /**
* @brief [Internal Use] Controls the step size adaption.
*/
  float _cumulativeCovariance;
  /**
* @brief [Internal Use] Expectation of $||N(0,I)||^2$.
*/
  float _chiSquareNumber;
  /**
* @brief [Internal Use] Establishes how frequently the eigenvalues are updated.
*/
  size_t _covarianceEigenvalueEvaluationFrequency;
  /**
* @brief [Internal Use] Determines the step size.
*/
  float _sigma;

  /**
* @brief [Internal Use] This flag remembers if it was not possible to produce an update and terminates
*/
  bool _noUpdatePossible;

  /**
* @brief [Internal Use] The trace of the initial covariance matrix.
*/
  float _trace;
  /**
* @brief [Internal Use] Sample coordinate information.
*/
  std::vector<std::vector<float>> _samplePopulation;
  /**
* @brief [Internal Use] Counter of evaluated samples to terminate evaluation.
*/
  size_t _finishedSampleCount;
  /**
* @brief [Internal Use] Best model evaluation from current generation.
*/
  float _currentBestValue;
  /**
* @brief [Internal Use] Best variables of current generation.
*/
  std::vector<float> _currentBestVariables;
  /**
* @brief [Internal Use] Best ever found variables.
*/
  std::vector<float> _bestEverVariables;
  /**
* @brief [Internal Use] Best model evaluation from previous generation.
*/
  float _previousBestValue;
  /**
* @brief [Internal Use] Index of the best sample in current generation.
*/
  size_t _bestSampleIndex;
  /**
* @brief [Internal Use] Best ever model evaluation.
*/
  float _bestEverValue;
  /**
* @brief [Internal Use] Best ever model evaluation as of previous generation.
*/
  float _previousBestEverValue;
  /**
* @brief [Internal Use] Sorted indeces of samples according to their model evaluation.
*/
  std::vector<size_t> _sortingIndex;
  /**
* @brief [Internal Use] (Unscaled) covariance Matrix of proposal distribution.
*/
  std::vector<float> _covarianceMatrix;
  /**
* @brief [Internal Use] Temporary Storage for Covariance Matrix.
*/
  std::vector<float> _auxiliarCovarianceMatrix;
  /**
* @brief [Internal Use] Matrix with eigenvectors in columns.
*/
  std::vector<float> _covarianceEigenvectorMatrix;
  /**
* @brief [Internal Use] Temporary Storage for Matrix with eigenvectors in columns.
*/
  std::vector<float> _auxiliarCovarianceEigenvectorMatrix;
  /**
* @brief [Internal Use] Axis lengths (sqrt(Evals))
*/
  std::vector<float> _axisLengths;
  /**
* @brief [Internal Use] Temporary storage for Axis lengths.
*/
  std::vector<float> _auxiliarAxisLengths;
  /**
* @brief [Internal Use] Temporary storage.
*/
  std::vector<float> _bDZMatrix;
  /**
* @brief [Internal Use] Temporary storage.
*/
  std::vector<float> _auxiliarBDZMatrix;
  /**
* @brief [Internal Use] Current mean of proposal distribution.
*/
  std::vector<float> _currentMean;
  /**
* @brief [Internal Use] Previous mean of proposal distribution.
*/
  std::vector<float> _previousMean;
  /**
* @brief [Internal Use] Update differential from previous to current mean.
*/
  std::vector<float> _meanUpdate;
  /**
* @brief [Internal Use] Evolution path for Covariance Matrix update.
*/
  std::vector<float> _evolutionPath;
  /**
* @brief [Internal Use] Conjugate evolution path for sigma update.
*/
  std::vector<float> _conjugateEvolutionPath;
  /**
* @brief [Internal Use] L2 Norm of the conjugate evolution path.
*/
  float _conjugateEvolutionPathL2Norm;
  /**
* @brief [Internal Use] Keeps count of the number of infeasible samples.
*/
  size_t _infeasibleSampleCount;
  /**
* @brief [Internal Use] Maximum diagonal element of the Covariance Matrix.
*/
  float _maximumDiagonalCovarianceMatrixElement;
  /**
* @brief [Internal Use] Minimum diagonal element of the Covariance Matrix.
*/
  float _minimumDiagonalCovarianceMatrixElement;
  /**
* @brief [Internal Use] Maximum Covariance Matrix Eigenvalue.
*/
  float _maximumCovarianceEigenvalue;
  /**
* @brief [Internal Use] Minimum Covariance Matrix Eigenvalue.
*/
  float _minimumCovarianceEigenvalue;
  /**
* @brief [Internal Use] Flag determining if the covariance eigensystem is up to date.
*/
  int _isEigensystemUpdated;
  /**
* @brief [Internal Use] This is the beta factor that indicates how fast the covariance matrix is adapted.
*/
  float _covarianceMatrixAdaptionFactor;
  /**
* @brief [Internal Use] Estimated Global Success Rate, required for calibration of covariance matrix scaling factor updates.
*/
  float _globalSuccessRate;
  /**
* @brief [Internal Use] Number of Covariance Matrix Adaptations.
*/
  size_t _covarianceMatrixAdaptationCount;
  /**
* @brief [Internal Use] Current minimum standard deviation of any variable.
*/
  float _currentMinStandardDeviation;
  /**
* @brief [Internal Use] Current maximum standard deviation of any variable.
*/
  float _currentMaxStandardDeviation;
  /**
* @brief [Termination Criteria] Maximum number of resamplings per candidate per generation if sample is outside of Lower and Upper Bound.
*/
  size_t _maxInfeasibleResamplings;
  /**
* @brief [Termination Criteria] Specifies the maximum condition of the covariance matrix.
*/
  float _maxConditionCovarianceMatrix;
  /**
* @brief [Termination Criteria] Specifies the minimum target fitness to stop minimization.
*/
  float _minValue;
  /**
* @brief [Termination Criteria] Specifies the maximum target fitness to stop maximization.
*/
  float _maxValue;
  /**
* @brief [Termination Criteria] Specifies the minimum fitness differential between two consecutive generations before stopping execution.
*/
  float _minValueDifferenceThreshold;
  /**
* @brief [Termination Criteria] Finishes execution when the maximum standard deviation falls below this threshold
*/
  float _targetMaxStandardDeviation;

  /**
* @brief [Termination Criteria] Specifies the mininum update to the variable means before triggering termination.
*/
  std::vector<float> _minMeanUpdates;

  /**
  * @brief Stores how many generations to run for
  */
  size_t _maxGenerations;

  /**
  * @brief Internal GSL storage
  */
  gsl_vector *_gsl_eval;

  /**
  * @brief Internal GSL storage
  */
  gsl_matrix *_gsl_evec;

  /**
  * @brief Internal GSL storage
  */
  gsl_eigen_symmv_workspace *_gsl_work;

  /**
   * @brief Defines the random number generator seed
   * @param seed Random seed
   */
  void setSeed(size_t seed);

  /**
* @brief Determines whether the module can trigger termination of an experiment run.
* @return True, if it should trigger termination; false, otherwise.
*/
  bool checkTermination();

  /**
  * @brief Prepares generation for the next set of evaluations
 */
  void prepareGeneration();

  /**
  * @brief Evaluates a single sample
  * @param sampleIdx Index of the sample to evaluate
 */
  void sampleSingle(size_t sampleIdx);

  /**
   * @brief Adapts the covariance matrix.
   * @param hsig Sign
  */
  void adaptC(int hsig);

  /**
    * @brief Updates scaling factor of covariance matrix.
 */
  void updateSigma(); /* update Sigma */

  /**
 * @brief Updates mean and covariance of Gaussian proposal distribution.
 * @param evaluations Model evaluations for all proposed samples
 */
  void updateDistribution(const std::vector<float> &evaluations);

  /**
    * @brief Updates the system of eigenvalues and eigenvectors
    * @param M Input matrix
 */
  void updateEigensystem(std::vector<float> &M);

  /**
   * @brief Function for eigenvalue decomposition.
   * @param N Matrix size
   * @param C Input matrix
   * @param diag Means
   * @param Q Output Matrix
 */
  void eigen(size_t N, std::vector<float> &C, std::vector<float> &diag, std::vector<float> &Q) const;

  /**
  * @brief Descending sort of vector elements, stores ordering in _sortingIndex.
  * @param _sortingIndex Ordering of elements in vector
  * @param vec Vector to sort
  * @param N Number of current samples.
 */
  void sort_index(const std::vector<float> &vec, std::vector<size_t> &_sortingIndex, size_t N) const;

  /**
  * @brief Initializes the weights of the mu vector
  * @param numsamples Length of mu vector
 */
  void initMuWeights(size_t numsamples); /* init _muWeights and dependencies */

  /**
  * @brief Initialize Covariance Matrix and Cholesky Decomposition
 */
  void initCovariance(); /* init sigma, C and B */

  /**
 * @brief Console output after generation.
 */
  void printInfo();

  /**
  * @brief Restores the optimizer to the initial state
  */
  void reset();

  /**
  * @brief Checks whether a proposed sample is feasible (all variables within acceptable range)
  * @param sample Sample to check
  * @return Whether the sample is feasible
  */
  bool isSampleFeasible(const std::vector<float> &sample);
};

} // namespace korali

#endif // _KORALI_FAST_CMAES_HPP_
