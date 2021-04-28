#ifndef LEAPFROG_IMPLICIT_H
#define LEAPFROG_IMPLICIT_H

#include "engine.hpp"
#include "hamiltonian_base.hpp"
#include "hamiltonian_riemannian_base.hpp"
#include "leapfrog_base.hpp"

#include <limits>

namespace korali
{
namespace solver
{
namespace sampler
{
/**
* \class LeapfrogImplicit
* @brief Used for time propagation according to hamiltonian dynamics via implicit Leapfrog integration.
*/
class LeapfrogImplicit : public Leapfrog
{
  private:
  /**
  * @brief Maximum fixed point iterations during each step.
  */
  size_t _maxNumFixedPointIter;

  public:
  /**
  * @brief Constructor for implicit leapfrog stepper.
  * @param maxNumFixedPointIter Maximum fixed point iterations.
  */
  LeapfrogImplicit(size_t maxNumFixedPointIter, std::shared_ptr<Hamiltonian> hamiltonian) : Leapfrog(hamiltonian), _maxNumFixedPointIter(maxNumFixedPointIter){};
  /**
  * @brief Implicit Leapfrog stepping scheme used for evolving Hamiltonian Dynamics.
  * @param q Position which is evolved.
  * @param p Momentum which is evolved.
  * @param stepSize Step Size used for Leap Frog Scheme.
  */
  void step(std::vector<double> &q, std::vector<double> &p, std::vector<double>& metric, std::vector<double>& inverseMetric, const double stepSize) override
  {
    size_t dim = p.size();
    double delta = 1e-6 * stepSize;

    // half step of momentum
    _hamiltonian->updateHamiltonian(q, metric, inverseMetric);
    std::vector<double> dphi_dq = _hamiltonian->dphi_dq();
    for (size_t i = 0; i < dim; ++i)
    {
      p[i] = p[i] - stepSize / 2.0 * dphi_dq[i];
    }

    std::vector<double> rho = p;
    std::vector<double> pPrime(dim);
    double deltaP;

    size_t numIter = 0;
    do
    {
      deltaP = 0.0;
      _hamiltonian->updateHamiltonian(q, metric, inverseMetric);
      std::vector<double> dtau_dq = _hamiltonian->dtau_dq(p, inverseMetric);
      for (size_t i = 0; i < dim; ++i)
      {
        pPrime[i] = rho[i] - stepSize / 2.0 * dtau_dq[i];
      }

      // find max delta
      for (size_t i = 0; i < dim; ++i)
      {
        if (std::abs(p[i] - pPrime[i]) > deltaP)
        {
          deltaP = std::abs(p[i] - pPrime[i]);
        }
      }

      p = pPrime;
      ++numIter;

    } while (deltaP > delta && numIter < _maxNumFixedPointIter);

    std::vector<double> qPrime(dim);
    std::vector<double> sigma = q;
    double deltaQ;

    numIter = 0;
    do
    {
      deltaQ = 0.0;
      _hamiltonian->updateHamiltonian(sigma, metric, inverseMetric);
      std::vector<double> dtau_dp_sigma = _hamiltonian->dtau_dp(p, inverseMetric);
      _hamiltonian->updateHamiltonian(q, metric, inverseMetric);
      std::vector<double> dtau_dp_q = _hamiltonian->dtau_dp(p, inverseMetric);
      for (size_t i = 0; i < dim; ++i)
      {
        qPrime[i] = sigma[i] + stepSize / 2.0 * dtau_dp_sigma[i] + stepSize / 2.0 * dtau_dp_q[i];
      }

      // find max delta
      for (size_t i = 0; i < dim; ++i)
      {
        if (std::abs(q[i] - qPrime[i]) > deltaQ)
        {
          deltaQ = std::abs(q[i] - qPrime[i]);
        }
      }

      q = qPrime;
      ++numIter;
    } while (deltaQ > delta && numIter < _maxNumFixedPointIter);

    _hamiltonian->updateHamiltonian(q, metric, inverseMetric);
    std::vector<double> dtau_dq = _hamiltonian->dtau_dq(p, inverseMetric);
    for (size_t i = 0; i < dim; ++i)
    {
      p[i] = p[i] - stepSize / 2.0 * dtau_dq[i];
    }

    dphi_dq = _hamiltonian->dphi_dq();
    for (size_t i = 0; i < dim; ++i)
    {
      p[i] = p[i] - stepSize / 2.0 * dphi_dq[i];
    }
  }
};

} // namespace sampler
} // namespace solver
} // namespace korali

#endif
