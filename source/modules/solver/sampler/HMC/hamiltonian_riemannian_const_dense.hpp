#ifndef HAMILTONIAN_RIEMANNIAN_CONST_DENSE_H
#define HAMILTONIAN_RIEMANNIAN_CONST_DENSE_H

#include "hamiltonian_riemannian_base.hpp"
#include "modules/distribution/multivariate/normal/normal.hpp"
#include <gsl/gsl_blas.h>
#include <gsl/gsl_eigen.h>

namespace korali
{
namespace solver
{
namespace sampler
{
/**
* \class HamiltonianRiemannianConstDense
* @brief Used for dense Riemannian metric.
*/
class HamiltonianRiemannianConstDense : public HamiltonianRiemannian
{
  public:
  /**
  * @brief Constructor with State Space Dim.
  * @param stateSpaceDim Dimension of State Space.
  */
  HamiltonianRiemannianConstDense(const size_t stateSpaceDim, const std::vector<double>& metric, korali::Experiment *k) : HamiltonianRiemannian{stateSpaceDim, k}
  {
    _inverseRegularizationParam = 1.0;

    // Initialize multivariate normal distribution
    _multivariateGenerator->_meanVector = std::vector<double>(_stateSpaceDim, 0.0);
    _multivariateGenerator->_sigma = std::vector<double>(_stateSpaceDim * _stateSpaceDim, 0.0);

    // Cholesky Decomposition
    for (size_t d = 0; d < _stateSpaceDim; ++d)
      _multivariateGenerator->_sigma[d * _stateSpaceDim + d] = sqrt(metric[d * _stateSpaceDim + d]);

    _multivariateGenerator->updateDistribution();

    // Memory allocation
    Q = gsl_matrix_alloc(stateSpaceDim, stateSpaceDim);
    lambda = gsl_vector_alloc(stateSpaceDim);
    w = gsl_eigen_symmv_alloc(stateSpaceDim);
    lambdaSoftAbs = gsl_matrix_alloc(stateSpaceDim, stateSpaceDim);
    inverseLambdaSoftAbs = gsl_matrix_alloc(stateSpaceDim, stateSpaceDim);
    tmpMatOne = gsl_matrix_alloc(stateSpaceDim, stateSpaceDim);
    tmpMatTwo = gsl_matrix_alloc(stateSpaceDim, stateSpaceDim);
    tmpMatThree = gsl_matrix_alloc(stateSpaceDim, stateSpaceDim);
    tmpMatFour = gsl_matrix_alloc(stateSpaceDim, stateSpaceDim);
  }

  /**
  * @brief Constructor with State Space Dim.
  * @param stateSpaceDim Dimension of State Space.
  * @param multivariateGenerator Multivariate generator needed for momentum sampling.
  */
  HamiltonianRiemannianConstDense(const size_t stateSpaceDim, const std::vector<double>& metric, korali::distribution::multivariate::Normal *multivariateGenerator, korali::Experiment *k) : HamiltonianRiemannian{stateSpaceDim, k}
  {

    _multivariateGenerator = multivariateGenerator;
    _inverseRegularizationParam = 1.0;

    // Memory allocation
    Q = gsl_matrix_alloc(stateSpaceDim, stateSpaceDim);
    lambda = gsl_vector_alloc(stateSpaceDim);
    w = gsl_eigen_symmv_alloc(stateSpaceDim);
    lambdaSoftAbs = gsl_matrix_alloc(stateSpaceDim, stateSpaceDim);
    inverseLambdaSoftAbs = gsl_matrix_alloc(stateSpaceDim, stateSpaceDim);
    tmpMatOne = gsl_matrix_alloc(stateSpaceDim, stateSpaceDim);
    tmpMatTwo = gsl_matrix_alloc(stateSpaceDim, stateSpaceDim);
    tmpMatThree = gsl_matrix_alloc(stateSpaceDim, stateSpaceDim);
    tmpMatFour = gsl_matrix_alloc(stateSpaceDim, stateSpaceDim);
  }

  /**
  * @brief Constructor with State Space Dim.
  * @param stateSpaceDim Dimension of State Space.
  * @param multivariateGenerator Generator needed for momentum sampling.
  * @param inverseRegularizationParam Inverse regularization parameter of SoftAbs metric that controls hardness of approximation: For large values inverseMetric is closer to analytical formula (and therefore closer to degeneracy in certain cases). 
  */
  HamiltonianRiemannianConstDense(const size_t stateSpaceDim, const std::vector<double> &metric, korali::distribution::multivariate::Normal *multivariateGenerator, const double inverseRegularizationParam, korali::Experiment *k) : HamiltonianRiemannian{stateSpaceDim, k}
  {

    _multivariateGenerator = multivariateGenerator;
    _inverseRegularizationParam = inverseRegularizationParam;

    // Memory allocation
    Q = gsl_matrix_alloc(stateSpaceDim, stateSpaceDim);
    lambda = gsl_vector_alloc(stateSpaceDim);
    w = gsl_eigen_symmv_alloc(stateSpaceDim);
    lambdaSoftAbs = gsl_matrix_alloc(stateSpaceDim, stateSpaceDim);
    inverseLambdaSoftAbs = gsl_matrix_alloc(stateSpaceDim, stateSpaceDim);
    tmpMatOne = gsl_matrix_alloc(stateSpaceDim, stateSpaceDim);
    tmpMatTwo = gsl_matrix_alloc(stateSpaceDim, stateSpaceDim);
    tmpMatThree = gsl_matrix_alloc(stateSpaceDim, stateSpaceDim);
    tmpMatFour = gsl_matrix_alloc(stateSpaceDim, stateSpaceDim);
  }

  /**
  * @brief Destructor of derived class.
  */
  ~HamiltonianRiemannianConstDense()
  {
  }

  /**
  * @brief Total energy function used for Hamiltonian Dynamics.
  * @param p Current momentum.
  * @return Total energy.
  */
  double H(const std::vector<double> &momentum, const std::vector<double>& inverseMetric) override
  {
    return this->K(momentum, inverseMetric) + this->U();
  }

  /**
  * @brief Purely virtual kinetic energy function K(q, p) = 0.5 * p.T * inverseMetric(q) * p + 0.5 * logDetMetric(q) used for Hamiltonian Dynamics.
  * @param p Current momentum.
  * @return Kinetic energy.
  */
  double K(const std::vector<double> &momentum, const std::vector<double>& inverseMetric) override
  {
    double result = this->tau(momentum, inverseMetric) + 0.5 * _logDetMetric;

    return result;
  }

  /**
  * @brief Purely virtual gradient of kintetic energy function dK(q, p) = inverseMetric(q) * p + 0.5 * dlogDetMetric_dq(q) used for Hamiltonian Dynamics.
  * @param p Current momentum.
  * @param inverseMetric Current inverse metric.
  * @return Gradient of Kinetic energy with current momentum.
  */
  std::vector<double> dK(const std::vector<double> &momentum, const std::vector<double> &inverseMetric) override
  {
    std::vector<double> gradient(_stateSpaceDim, 0.0);
    double tmpScalar = 0.0;

    for (size_t i = 0; i < _stateSpaceDim; ++i)
    {
      tmpScalar = 0.0;
      for (size_t j = 0; j < _stateSpaceDim; ++j)
      {
        tmpScalar += inverseMetric[i * _stateSpaceDim + j] * momentum[j];
      }
      gradient[i] = tmpScalar;
    }

    return gradient;
  }

  /**
  * @brief Purely virtual function tau(q, p) = 0.5 * p^T * inverseMetric(q) * p (no logDetMetric term)
  * @param momentum Current momentum.
  * @return Gradient of Kinetic energy with current momentum.
  */
  double tau(const std::vector<double> &momentum, const std::vector<double> &inverseMetric) override
  {
    double tmpScalar = 0.0;

    // this->updateHamiltonian(q);

    for (size_t i = 0; i < _stateSpaceDim; ++i)
    {
      tmpScalar += momentum[i] * inverseMetric[i] * momentum[i];
    }

    return 0.5 * tmpScalar;
  }

  /**
  * @brief Purely virtual gradient of dtau_dq(q, p) = 0.5 * p^T * dinverseMetric_dq(q) * p used for Hamiltonian Dynamics.
  * @param p Current momentum.
  * @return Gradient of Kinetic energy with current momentum.
  */
  std::vector<double> dtau_dq(const std::vector<double> &momentum, const std::vector<double>& inverseMetric) override
  {
    std::vector<double> result(_stateSpaceDim, 0.0);

    return result;
  }

  /**
  * @brief Purely virtual gradient of dtau_dp(q, p) = inverseMetric(q) * p used for Hamiltonian Dynamics.
  * @param p Current momentum.
  * @param inverseMetric Current inverse metric.
  * @return Gradient of Kinetic energy with current momentum.
  */
  std::vector<double> dtau_dp(const std::vector<double> &momentum, const std::vector<double> &inverseMetric) override
  {
    std::vector<double> result = this->dK(momentum, inverseMetric);

    return result;
  }

  /**
  * @brief Purely virtual gradient of phi(q) = 0.5 * logDetMetric(q) + U(q) used for Hamiltonian Dynamics.
  * @return Gradient of Kinetic energy with current momentum.
  */
  double phi() override
  {
    return this->U() + 0.5 * _logDetMetric;
  }

  /**
  * @brief Purely virtual gradient of dphi_dq(q) = 0.5 * dlogDetMetric_dq(q) + dU(q) used for Hamiltonian Dynamics.
  * @return Gradient of Kinetic energy with current momentum.
  */
  std::vector<double> dphi_dq() override
  {
    std::vector<double> result = this->dU();

    return result;
  }

  /**
  * @brief Updates current position of hamiltonian.
  * @param q Current position.
  * @param inverseMetric Current inverse metric.
  * @param _k Experiment object.
  */
  void updateHamiltonian(const std::vector<double> &q, std::vector<double>& metric, std::vector<double>& inverseMetric) override
  {
    auto sample = korali::Sample();
    sample["Sample Id"] = _modelEvaluationCount;
    sample["Module"] = "Problem";
    sample["Operation"] = "Evaluate";
    sample["Parameters"] = q;

    KORALI_START(sample);
    KORALI_WAIT(sample);
    _modelEvaluationCount++;
    _currentEvaluation = KORALI_GET(double, sample, "logP(x)");

    if (samplingProblemPtr != nullptr)
    {
      samplingProblemPtr->evaluateGradient(sample);
      samplingProblemPtr->evaluateHessian(sample);
    }
    else
    {
      bayesianProblemPtr->evaluateGradient(sample);
      bayesianProblemPtr->evaluateHessian(sample);
    }

    _currentGradient = sample["grad(logP(x))"].get<std::vector<double>>();
    _currentHessian = sample["H(logP(x))"].get<std::vector<double>>();
  }

  /**
  * @brief Generates sample of momentum.
  * @param metric Current metric.
  * @return Sample of momentum from normal distribution with covariance matrix metric. Only variance taken into account with diagonal metric.
  */
  std::vector<double> sampleMomentum(const std::vector<double>& metric) const override
  {
    std::vector<double> result(_stateSpaceDim, 0.0);
    _multivariateGenerator->getRandomVector(&result[0], _stateSpaceDim);
    return result;
  }

  /**
  * @brief Calculates inner product induces by inverse metric.
  * @param pLeft Left argument (momentum).
  * @param pRight Right argument (momentum).
  * @param inverseMetric Current inverse metric.
  * @return pLeft.transpose * inverseMetric * pRight.
  */
  double innerProduct(const std::vector<double> &pLeft, const std::vector<double> &pRight, const std::vector<double> &inverseMetric) const
  {
    double result = 0.0;

    for (size_t i = 0; i < _stateSpaceDim; ++i)
    {
      for (size_t j = 0; j < _stateSpaceDim; ++j)
      {
        result += pLeft[i] * inverseMetric[i * _stateSpaceDim + j] * pRight[j];
      }
    }

    return result;
  }

  /**
  * @brief Updates Metric and Inverse Metric according to SoftAbs.
  * @param q Current position.
  * @param metric Current metric.
  * @param inverseMetric Current inverse metric.
  * @param _k Experiment object.
  * @return Returns error code of Cholesky decomposition of GSL.
  */
  int updateMetricMatricesRiemannian(const std::vector<double> &q, std::vector<double>& metric,  std::vector<double>& inverseMetric) override
  {
    auto hessian = _currentHessian;
    gsl_matrix_view Xv = gsl_matrix_view_array(hessian.data(), _stateSpaceDim, _stateSpaceDim);
    gsl_matrix *X = &Xv.matrix;

    gsl_eigen_symmv(X, lambda, Q, w);

    gsl_matrix_set_all(lambdaSoftAbs, 0.0);

    gsl_matrix_set_all(inverseLambdaSoftAbs, 0.0);

    _logDetMetric = 0.0;
    for (size_t i = 0; i < _stateSpaceDim; ++i)
    {
      double lambdaSoftAbs_i = __softAbsFunc(gsl_vector_get(lambda, i), _inverseRegularizationParam);
      gsl_matrix_set(lambdaSoftAbs, i, i, lambdaSoftAbs_i);
      gsl_matrix_set(inverseLambdaSoftAbs, i, i, 1.0 / lambdaSoftAbs_i);
      _logDetMetric += std::log(lambdaSoftAbs_i);
    }

    gsl_matrix_set_all(tmpMatOne, 0.0);
    gsl_matrix_set_all(tmpMatTwo, 0.0);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, Q, lambdaSoftAbs, 0.0, tmpMatOne); // Q * \lambda_{SoftAbs}
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, tmpMatOne, Q, 0.0, tmpMatTwo);       // Q * \lambda_{SoftAbs} * Q^T

    gsl_matrix_set_all(tmpMatThree, 0.0);
    gsl_matrix_set_all(tmpMatFour, 0.0);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, Q, inverseLambdaSoftAbs, 0.0, tmpMatThree); // Q * (\lambda_{SoftAbs})^{-1}
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, tmpMatThree, Q, 0.0, tmpMatFour);             // Q * (\lambda_{SoftAbs})^{-1} * Q^T

    for (size_t i = 0; i < _stateSpaceDim; ++i)
    {
      for (size_t j = 0; j < _stateSpaceDim; ++j)
      {
        metric[i + j * _stateSpaceDim] = gsl_matrix_get(tmpMatTwo, i, j);
        inverseMetric[i + j * _stateSpaceDim] = gsl_matrix_get(tmpMatFour, i, j);
      }
    }

    _multivariateGenerator->_sigma = metric;

    // Cholesky Decomp
    gsl_matrix_view sigma = gsl_matrix_view_array(&_multivariateGenerator->_sigma[0], _stateSpaceDim, _stateSpaceDim);

    int err = gsl_linalg_cholesky_decomp(&sigma.matrix);
    if (err != GSL_EDOM)
    {
      _multivariateGenerator->updateDistribution();
    }

    return err;
  }

  /**
  * @brief Inverse regularization parameter of SoftAbs metric that controls hardness of approximation: For large values inverseMetric is closer to analytical formula (and therefore closer to degeneracy in certain cases). 
  */
  double _inverseRegularizationParam;

  private:
  /**
  * @brief Multi dimensional normal generator needed for sampling of momentum from dense metric.
  */
  korali::distribution::multivariate::Normal *_multivariateGenerator;

  gsl_matrix *Q;
  gsl_vector *lambda;
  gsl_eigen_symmv_workspace *w;
  gsl_matrix *lambdaSoftAbs;
  gsl_matrix *inverseLambdaSoftAbs;

  gsl_matrix *tmpMatOne;
  gsl_matrix *tmpMatTwo;
  gsl_matrix *tmpMatThree;
  gsl_matrix *tmpMatFour;
};

} // namespace sampler
} // namespace solver
} // namespace korali

#endif
