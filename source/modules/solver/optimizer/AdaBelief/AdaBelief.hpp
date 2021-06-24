/** \namespace optimizer
* @brief Namespace declaration for modules of type: optimizer.
*/

/** \file
* @brief Header file for module: AdaBelief.
*/

/** \dir solver/optimizer/AdaBelief
* @brief Contains code, documentation, and scripts for module: AdaBelief.
*/


#ifndef _KORALI_SOLVER_OPTIMIZER_ADABELIEF_
#define _KORALI_SOLVER_OPTIMIZER_ADABELIEF_
;

#include "modules/solver/optimizer/optimizer.hpp"

namespace korali
{
namespace solver
{
namespace optimizer
{
;

/**
* @brief Class declaration for module: AdaBelief.
*/
class AdaBelief : public Optimizer
{
  public: 
  /**
  * @brief Beta for momentum update
  */
   double _beta1;
  /**
  * @brief Beta for gradient update
  */
   double _beta2;
  /**
  * @brief Learning Rate
  */
   double _eta;
  /**
  * @brief Term for numerical stability.
  */
   double _epsilon;
  /**
  * @brief [Internal Use] Current value of parameters.
  */
   std::vector<double> _currentVariable;
  /**
  * @brief [Internal Use] Gradient of Function with respect to Parameters.
  */
   std::vector<double> _gradient;
  /**
  * @brief [Internal Use] Gradient of function with respect to Best Ever Variables.
  */
   std::vector<double> _bestEverGradient;
  /**
  * @brief [Internal Use] Norm of gradient of function with respect to Parameters.
  */
   double _gradientNorm;
  /**
  * @brief [Internal Use] Estimate of first moment of Gradient.
  */
   std::vector<double> _firstMoment;
  /**
  * @brief [Internal Use] Bias corrected estimate of first moment of Gradient.
  */
   std::vector<double> _biasCorrectedFirstMoment;
  /**
  * @brief [Internal Use] Old estimate of second moment of Gradient.
  */
   std::vector<double> _secondCentralMoment;
  /**
  * @brief [Internal Use] Bias corrected estimate of second moment of Gradient.
  */
   std::vector<double> _biasCorrectedSecondCentralMoment;
  /**
  * @brief [Termination Criteria] Specifies the minimal norm for the gradient of function with respect to Parameters.
  */
   double _minGradientNorm;
  /**
  * @brief [Termination Criteria] Specifies the minimal norm for the gradient of function with respect to Parameters.
  */
   double _maxGradientNorm;
  
 
  /**
  * @brief Determines whether the module can trigger termination of an experiment run.
  * @return True, if it should trigger termination; false, otherwise.
  */
  bool checkTermination() override;
  /**
  * @brief Obtains the entire current state and configuration of the module.
  * @param js JSON object onto which to save the serialized state of the module.
  */
  void getConfiguration(knlohmann::json& js) override;
  /**
  * @brief Sets the entire state and configuration of the module, given a JSON object.
  * @param js JSON object from which to deserialize the state of the module.
  */
  void setConfiguration(knlohmann::json& js) override;
  /**
  * @brief Applies the module's default configuration upon its creation.
  * @param js JSON object containing user configuration. The defaults will not override any currently defined settings.
  */
  void applyModuleDefaults(knlohmann::json& js) override;
  /**
  * @brief Applies the module's default variable configuration to each variable in the Experiment upon creation.
  */
  void applyVariableDefaults() override;
  

  /**
   * @brief Takes a sample evaluation and its gradient and calculates the next set of parameters
   * @param evaluation The value of the objective function at the current set of parameters
   * @param gradient The gradient of the objective function at the current set of parameters
   */
  void processResult(double evaluation, std::vector<double> &gradient);

  void finalize() override;
  void setInitialConfiguration() override;
  void runGeneration() override;
  void printGenerationBefore() override;
  void printGenerationAfter() override;
};

} //optimizer
} //solver
} //korali
;

#endif // _KORALI_SOLVER_OPTIMIZER_ADABELIEF_
;