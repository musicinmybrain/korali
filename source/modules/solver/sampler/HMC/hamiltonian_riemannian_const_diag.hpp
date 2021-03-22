#ifndef HAMILTONIAN_RIEMANNIAN_CONST_DIAG_H
#define HAMILTONIAN_RIEMANNIAN_CONST_DIAG_H

#include "hamiltonian_riemannian_base.hpp"
#include "modules/distribution/univariate/normal/normal.hpp"

namespace korali
{
namespace solver
{
namespace sampler
{
/**
* \class HamiltonianRiemannianConstDiag
* @brief Used for diagonal Riemannian metric.
*/
class HamiltonianRiemannianConstDiag : public HamiltonianRiemannian
{
  public:
  /**
  * @brief Constructor with State Space Dim.
  * @param stateSpaceDim Dimension of State Space.
  */
  HamiltonianRiemannianConstDiag(const size_t stateSpaceDim, korali::Experiment *k) : HamiltonianRiemannian{stateSpaceDim, k}
  {
    _inverseRegularizationParam = 1.0;
  }

  /**
  * @brief Constructor with State Space Dim.
  * @param stateSpaceDim Dimension of State Space.
  * @param normalGenerator Generator needed for momentum sampling.
  */
  HamiltonianRiemannianConstDiag(const size_t stateSpaceDim, korali::distribution::univariate::Normal *normalGenerator, korali::Experiment *k) : HamiltonianRiemannian{stateSpaceDim, k}
  {
    _normalGenerator = normalGenerator;
    _inverseRegularizationParam = 1.0;
  }

  /**
  * @brief Constructor with State Space Dim.
  * @param stateSpaceDim Dimension of State Space.
  * @param normalGenerator Generator needed for momentum sampling.
  * @param inverseRegularizationParam Inverse regularization parameter of SoftAbs metric that controls hardness of approximation: For large values inverseMetric is closer to analytical formula (and therefore closer to degeneracy in certain cases). 
  */
  HamiltonianRiemannianConstDiag(const size_t stateSpaceDim, korali::distribution::univariate::Normal *normalGenerator, const double inverseRegularizationParam, korali::Experiment *k) : HamiltonianRiemannian{stateSpaceDim, k}
  {
    _normalGenerator = normalGenerator;
    _inverseRegularizationParam = inverseRegularizationParam;
  }

  /**
  * @brief Constructor with State Space Dim.
  * @param stateSpaceDim Dimension of State Space.
  * @param normalGenerator Generator needed for momentum sampling.
  * @param inverseRegularizationParam Inverse regularization parameter of SoftAbs metric that controls hardness of approximation: For large values inverseMetric is closer to analytical formula (and therefore closer to degeneracy in certain cases). 
  */
  HamiltonianRiemannianConstDiag(const size_t stateSpaceDim, korali::distribution::univariate::Normal *normalGenerator, const double inverseRegularizationParam, korali::Experiment *k) : HamiltonianRiemannianConstDiag{stateSpaceDim, normalGenerator, inverseRegularizationParam, k}
  {
  }

  /**
  * @brief Destructor of derived class.
  */
  ~HamiltonianRiemannianConstDiag()
  {
  }

  /**
  * @brief Total energy function used for Hamiltonian Dynamics.
  * @param p Current momentum.
  * @return Total energy.
  */
  double H(const std::vector<double> &p) override
  {
    return this->K(p) + this->U();
  }

  /**
  * @brief Purely virtual kinetic energy function K(q, p) = 0.5 * p.T * inverseMetric(q) * p + 0.5 * logDetMetric(q) used for Hamiltonian Dynamics.
  * @param p Current momentum.
  * @return Kinetic energy.
  */
  double K(const std::vector<double> &p) override
  {
    double result = this->tau(p, inverseMetric) + 0.5 * _logDetMetric;

    return result;
  }

  /**
  * @brief Purely virtual gradient of kintetic energy function dK(q, p) = inverseMetric(q) * p + 0.5 * dlogDetMetric_dq(q) used for Hamiltonian Dynamics.
  * @param p Current momentum.
  * @return Gradient of Kinetic energy with current momentum.
  */
  std::vector<double> dK(const std::vector<double> &p, const std::vector<double> &inverseMetric) override
  {
    std::vector<double> gradient(_stateSpaceDim, 0.0);
    for (size_t i = 0; i < _stateSpaceDim; ++i)
    {
      gradient[i] = inverseMetric[i] * p[i];
    }

    return gradient;
  }

  /**
  * @brief Purely virtual function tau(q, p) = 0.5 * p^T * inverseMetric(q) * p (no logDetMetric term)
  * @param p Current momentum.
  * @return Gradient of Kinetic energy with current momentum.
  */
  double tau(const std::vector<double> &p, const std::vector<double>& inverseMetric) override
  {
    double tmpScalar = 0.0;

    // this->updateHamiltonian(q);

    for (size_t i = 0; i < _stateSpaceDim; ++i)
    {
      tmpScalar += p[i] * inverseMetric[i] * p[i];
    }

    return 0.5 * tmpScalar;
  }

  /**
  * @brief Purely virtual gradient of dtau_dq(q, p) = 0.5 * p^T * dinverseMetric_dq(q) * p used for Hamiltonian Dynamics.
  * @param p Current momentum.
  * @return Gradient of Kinetic energy with current momentum.
  */
  std::vector<double> dtau_dq(const std::vector<double> &p, const std::vector<double> inverseMetric) override
  {
    std::vector<double> result(_stateSpaceDim, 0.0);

    return result;
  }

  /**
  * @brief Purely virtual gradient of dtau_dp(q, p) = inverseMetric(q) * p used for Hamiltonian Dynamics.
  * @param p Current momentum.
  * @return Gradient of Kinetic energy with current momentum.
  */
  std::vector<double> dtau_dp(const std::vector<double> &p, const std::vector<double> &inverseMetric) override
  {
    std::vector<double> result = this->dK(p, inverseMetric);

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
    std::vector<double> result(_stateSpaceDim);

    for (size_t i = 0; i < _stateSpaceDim; ++i)
    {
      result[i] = std::sqrt(metric[i]) * _normalGenerator->getRandomNumber();
    }

    return result;
  }

  /**
  * @brief Calculates inner product induces by inverse metric.
  * @param pLeft Left argument (momentum).
  * @param pRight Right argument (momentum).
  * @return pLeft.transpose * inverseMetric * pRight.
  */
  double innerProduct(const std::vector<double> &pLeft, const std::vector<double> &pRight, const std::vector<double> &inverseMetric) const
  {
    double result = 0.0;

    for (size_t i = 0; i < _stateSpaceDim; ++i)
    {
      result += pLeft[i] * inverseMetric[i] * pRight[i];
    }

    return result;
  }

  /**
  * @brief Updates Metric and Inverse Metric according to SoftAbs.
  * @param q Current position.
  * @param metric Current metric.
  * @param inverseMetric Current inverse metric.
  * @param _k Experiment object.
  * @return Returns error code to indicate if update was unsuccessful. 
  */
  int updateMetricMatricesRiemannian(const std::vector<double> &q, std::vector<double>& metric, std::vector<double>& inverseMetric) override
  {
    auto hessian = _currentHessian;

    // constant for condition number of metric
    double detMetric = 1.0;

    for (size_t i = 0; i < _stateSpaceDim; ++i)
    {
      metric[i] = this->__softAbsFunc(hessian[i + i * _stateSpaceDim], _inverseRegularizationParam);
      inverseMetric[i] = 1.0 / metric[i];
      detMetric *= metric[i];
    }
    _logDetMetric = std::log(detMetric);

    return 0;
  }

  /**
  * @brief Inverse regularization parameter of SoftAbs metric that controls hardness of approximation: For large values inverseMetric is closer to analytical formula (and therefore closer to degeneracy in certain cases). 
  */
  double _inverseRegularizationParam;

  private:
  /**
  * @brief One dimensional normal generator needed for sampling of momentum from diagonal metric.
  */
  korali::distribution::univariate::Normal *_normalGenerator;
};

} // namespace sampler
} // namespace solver
} // namespace korali

#endif
