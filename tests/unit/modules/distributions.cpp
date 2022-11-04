#include "gtest/gtest.h"
#include "korali.hpp"
#include "modules/distribution/univariate/beta/beta.hpp"
#include "modules/distribution/univariate/cauchy/cauchy.hpp"
#include "modules/distribution/univariate/exponential/exponential.hpp"
#include "modules/distribution/univariate/gamma/gamma.hpp"
#include "modules/distribution/univariate/geometric/geometric.hpp"
#include "modules/distribution/univariate/igamma/igamma.hpp"
#include "modules/distribution/univariate/laplace/laplace.hpp"
#include "modules/distribution/univariate/logNormal/logNormal.hpp"
#include "modules/distribution/univariate/normal/normal.hpp"
#include "modules/distribution/univariate/truncatedNormal/truncatedNormal.hpp"
#include "modules/distribution/univariate/uniform/uniform.hpp"
#include "modules/distribution/univariate/uniformratio/uniformratio.hpp"
#include "modules/distribution/univariate/poisson/poisson.hpp"
#include "modules/distribution/univariate/weibull/weibull.hpp"
#include "modules/distribution/multivariate/normal/normal.hpp"

#define PDENSITY_ERROR_TOLERANCE 0.0000001

namespace
{
 using namespace korali;

 TEST(Distrtibutions, BaseDistribution)
 {
   knlohmann::json distributionJs;
   Experiment e;
   distribution::univariate::Beta* d;

   // Creating distribution with an incorrect name
   distributionJs["Type"] = "Distribution/Univariate/Beta";
   distributionJs["Name"] = "Name";
   distributionJs["Range"] = "Range";
   distributionJs["Random Seed"] = 0;
   distributionJs["Alpha"] = 0.5;
   distributionJs["Beta"] = 0.5;
   ASSERT_ANY_THROW(d = dynamic_cast<korali::distribution::univariate::Beta *>(Module::getModule(distributionJs, &e)));

   // Creating distribution correctly now
   distributionJs["Type"] = "Univariate/Beta";
   ASSERT_NO_THROW(d = dynamic_cast<korali::distribution::univariate::Beta *>(Module::getModule(distributionJs, &e)));

   auto baseJs = distributionJs;

   // Triggering configuration errors
   distributionJs.clear();
   distributionJs["Range"] = "Range";
   distributionJs["Random Seed"] = 0;
   distributionJs["Alpha"] = 0.5;
   distributionJs["Beta"] = 0.5;
   ASSERT_ANY_THROW(d->setConfiguration(distributionJs)); // Missing name
   distributionJs["Name"] = 0;
   ASSERT_ANY_THROW(d->setConfiguration(distributionJs)); // Bad name format

   distributionJs.clear();
   distributionJs["Name"] = "Name";
   distributionJs["Random Seed"] = 0;
   distributionJs["Alpha"] = 0.5;
   distributionJs["Beta"] = 0.5;
   ASSERT_ANY_THROW(d->setConfiguration(distributionJs)); // Missing range
   distributionJs["Range"] = 0;
   ASSERT_ANY_THROW(d->setConfiguration(distributionJs)); // Bad range format

   distributionJs.clear();
   distributionJs["Name"] = "Name";
   distributionJs["Range"] = "Range";
   distributionJs["Alpha"] = 0.5;
   distributionJs["Beta"] = 0.5;
   ASSERT_ANY_THROW(d->setConfiguration(distributionJs)); // Missing seed
   distributionJs["Random Seed"] = "Seed";
   ASSERT_ANY_THROW(d->setConfiguration(distributionJs)); // Bad seed format

   distributionJs = baseJs;
   distributionJs.erase("Random Seed");
   ASSERT_ANY_THROW(d->setConfiguration(distributionJs));

   distributionJs = baseJs;
   distributionJs["Random Seed"] = "Not a Number";
   ASSERT_ANY_THROW(d->setConfiguration(distributionJs));

   distributionJs = baseJs;
   distributionJs["Random Seed"] = 1;
   ASSERT_NO_THROW(d->setConfiguration(distributionJs));

   distributionJs = baseJs;
   distributionJs.erase("Range");
   ASSERT_ANY_THROW(d->setConfiguration(distributionJs));

   distributionJs = baseJs;
   distributionJs["Range"] = 1.0;
   ASSERT_ANY_THROW(d->setConfiguration(distributionJs));

   distributionJs = baseJs;
   distributionJs["Range"] = "Not a Number";
   ASSERT_NO_THROW(d->setConfiguration(distributionJs));
 }

 TEST(Distrtibutions, BetaDistribution)
 {
  knlohmann::json distributionJs;
  Experiment e;
  distribution::univariate::Beta* d;

  // Creating distribution with an incorrect name
  distributionJs["Type"] = "Distribution/Univariate/Beta";
  ASSERT_ANY_THROW(d = dynamic_cast<korali::distribution::univariate::Beta *>(Module::getModule(distributionJs, &e)));

  // Creating distribution correctly now
  distributionJs["Type"] = "Univariate/Beta";
  ASSERT_NO_THROW(d = dynamic_cast<korali::distribution::univariate::Beta *>(Module::getModule(distributionJs, &e)));

  // Getting module defaults
  distributionJs["Name"] = "Test";
  ASSERT_NO_THROW(d->applyModuleDefaults(distributionJs));
  ASSERT_NO_THROW(d->applyVariableDefaults());
  auto baseJs = distributionJs;

  // Testing conditional variables
  distributionJs = baseJs;
  distributionJs["Alpha"] = "Conditional 1";
  distributionJs["Beta"] = "Conditional 2";
  ASSERT_NO_THROW(d->setConfiguration(distributionJs));

  // Testing get configuration method
  d->getConfiguration(distributionJs);
  ASSERT_EQ(distributionJs["Type"].get<std::string>(), "univariate/beta");
  ASSERT_EQ(distributionJs["Alpha"].get<std::string>(), "Conditional 1");
  ASSERT_EQ(distributionJs["Beta"].get<std::string>(), "Conditional 2");

  // Testing correct shape
  distributionJs["Alpha"] = 0.5;
  distributionJs["Beta"] = 0.5;
  ASSERT_NO_THROW(d->setConfiguration(distributionJs));
  ASSERT_NO_THROW(d->updateDistribution());

  // Testing get configuration method
  d->getConfiguration(distributionJs);
  ASSERT_EQ(distributionJs["Type"].get<std::string>(), "univariate/beta");
  ASSERT_EQ(distributionJs["Alpha"].get<double>(), 0.5);
  ASSERT_EQ(distributionJs["Beta"].get<double>(), 0.5);

  // Testing get property pointer
  ASSERT_TRUE(d->getPropertyPointer("Alpha") != NULL);
  ASSERT_TRUE(d->getPropertyPointer("Beta") != NULL);
  ASSERT_ANY_THROW(d->getPropertyPointer("Undefined")); // Distribution not recognized

  // Testing incorrect shape (alpha)
  d->_alpha = -0.5;
  d->_beta = 0.5;
  ASSERT_ANY_THROW(d->updateDistribution());

  // Testing correct shape (alpha)
  d->_alpha = 0.5;
  d->_beta = 0.5;
  ASSERT_NO_THROW(d->updateDistribution());

  // Distributions generated with https://keisan.casio.com/exec/system/1180573226

  // Testing expected density
  EXPECT_NEAR(d->getDensity( 0.01 ), 3.199134726, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.02 ), 2.273642044, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.03 ), 1.865965599, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.04 ), 1.624368336, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.05 ), 1.460505923, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.06 ), 1.340326411, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.07 ), 1.247554804, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.08 ), 1.173305807, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.09 ), 1.112264757, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.10 ), 1.061032954, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.11 ), 1.017322808, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.12 ), 0.979530962, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.13 ), 0.946496091, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.14 ), 0.917353841, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.15 ), 0.891445988, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.16 ), 0.868261398, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.17 ), 0.847396411, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.18 ), 0.828527540, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.19 ), 0.811392179, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.20 ), 0.795774716, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.21 ), 0.781496315, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.22 ), 0.768407306, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.23 ), 0.756381411, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.24 ), 0.745311308, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.25 ), 0.735105194, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.26 ), 0.725684076, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.27 ), 0.716979627, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.28 ), 0.708932462, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.29 ), 0.701490759, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.30 ), 0.694609118, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.31 ), 0.688247647, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.32 ), 0.682371189, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.33 ), 0.676948696, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.34 ), 0.671952696, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.35 ), 0.667358854, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.36 ), 0.663145596, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.37 ), 0.659293798, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.38 ), 0.655786518, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.39 ), 0.652608771, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.40 ), 0.649747334, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.41 ), 0.647190589, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.42 ), 0.644928375, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.43 ), 0.642951882, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.44 ), 0.641253540, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.45 ), 0.639826945, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.46 ), 0.638666787, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.47 ), 0.637768791, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.48 ), 0.637129680, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.49 ), 0.636747135, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.50 ), 0.636619772, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.51 ), 0.636747135, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.52 ), 0.637129680, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.53 ), 0.637768791, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.54 ), 0.638666787, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.55 ), 0.639826945, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.56 ), 0.641253540, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.57 ), 0.642951882, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.58 ), 0.644928375, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.59 ), 0.647190589, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.60 ), 0.649747334, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.61 ), 0.652608771, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.62 ), 0.655786518, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.63 ), 0.659293798, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.64 ), 0.663145596, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.65 ), 0.667358854, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.66 ), 0.671952696, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.67 ), 0.676948696, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.68 ), 0.682371189, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.69 ), 0.688247647, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.70 ), 0.694609118, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.71 ), 0.701490759, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.72 ), 0.708932462, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.73 ), 0.716979627, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.74 ), 0.725684076, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.75 ), 0.735105194, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.76 ), 0.745311308, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.77 ), 0.756381411, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.78 ), 0.768407306, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.79 ), 0.781496315, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.80 ), 0.795774716, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.81 ), 0.811392179, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.82 ), 0.828527540, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.83 ), 0.847396411, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.84 ), 0.868261398, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.85 ), 0.891445988, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.86 ), 0.917353841, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.87 ), 0.946496091, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.88 ), 0.979530962, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.89 ), 1.017322808, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.90 ), 1.061032954, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.91 ), 1.112264757, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.92 ), 1.173305807, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.93 ), 1.247554804, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.94 ), 1.340326411, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.95 ), 1.460505923, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.96 ), 1.624368336, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.97 ), 1.865965599, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.98 ), 2.273642044, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.99 ), 3.199134726, PDENSITY_ERROR_TOLERANCE);

  // Testing log density function
  EXPECT_NEAR(d->getLogDensity( 0.01 ), 1.162880375, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.02 ), 0.821382970, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.03 ), 0.623778667, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.04 ), 0.485119024, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.05 ), 0.378782898, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.06 ), 0.292913175, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.07 ), 0.221185479, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.08 ), 0.159825241, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.09 ), 0.106398258, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.10 ), 0.059242919, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.11 ), 0.017174479, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.12 ), -0.020681432, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.13 ), -0.054988438, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.14 ), -0.086262013, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.15 ), -0.114910429, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.16 ), -0.141262460, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.17 ), -0.165586676, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.18 ), -0.188105202, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.19 ), -0.209003767, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.20 ), -0.228439154, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.21 ), -0.246544845, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.22 ), -0.263435340, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.23 ), -0.279209519, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.24 ), -0.293953285, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.25 ), -0.307741669, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.26 ), -0.320640515, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.27 ), -0.332707853, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.28 ), -0.343995015, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.29 ), -0.354547553, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.30 ), -0.364406012, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.31 ), -0.373606555, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.32 ), -0.382181504, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.33 ), -0.390159790, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.34 ), -0.397567333, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.35 ), -0.404427366, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.36 ), -0.410760711, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.37 ), -0.416586019, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.38 ), -0.421919972, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.39 ), -0.426777455, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.40 ), -0.431171708, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.41 ), -0.435114455, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.42 ), -0.438616014, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.43 ), -0.441685392, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.44 ), -0.444330362, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.45 ), -0.446557537, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.46 ), -0.448372421, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.47 ), -0.449779458, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.48 ), -0.450782065, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.49 ), -0.451382665, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.50 ), -0.451582705, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.51 ), -0.451382665, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.52 ), -0.450782065, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.53 ), -0.449779458, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.54 ), -0.448372421, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.55 ), -0.446557537, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.56 ), -0.444330362, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.57 ), -0.441685392, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.58 ), -0.438616014, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.59 ), -0.435114455, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.60 ), -0.431171708, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.61 ), -0.426777455, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.62 ), -0.421919972, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.63 ), -0.416586019, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.64 ), -0.410760711, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.65 ), -0.404427366, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.66 ), -0.397567333, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.67 ), -0.390159790, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.68 ), -0.382181504, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.69 ), -0.373606555, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.70 ), -0.364406012, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.71 ), -0.354547553, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.72 ), -0.343995015, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.73 ), -0.332707853, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.74 ), -0.320640515, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.75 ), -0.307741669, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.76 ), -0.293953285, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.77 ), -0.279209519, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.78 ), -0.263435340, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.79 ), -0.246544845, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.80 ), -0.228439154, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.81 ), -0.209003767, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.82 ), -0.188105202, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.83 ), -0.165586676, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.84 ), -0.141262460, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.85 ), -0.114910429, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.86 ), -0.086262013, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.87 ), -0.054988438, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.88 ), -0.020681432, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.89 ), 0.017174479, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.90 ), 0.059242919, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.91 ), 0.106398258, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.92 ), 0.159825241, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.93 ), 0.221185479, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.94 ), 0.292913175, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.95 ), 0.378782898, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.96 ), 0.485119024, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.97 ), 0.623778667, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.98 ), 0.821382970, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.99 ), 1.162880375, PDENSITY_ERROR_TOLERANCE);

  // Testing extreme cases for log density
  EXPECT_EQ(d->getLogDensity( -0.001 ), -INFINITY);
  EXPECT_EQ(d->getLogDensity( 1.001 ), -INFINITY);

 // Checking random numbers are within the expected range
  for (size_t i = 0; i < 100; i++)
  {
   double y = d->getRandomNumber();
   EXPECT_TRUE((y >= 0.0) && (y <= 1.0));
  }

  // Testing extreme for log density gradient and hessian
  EXPECT_EQ(d->getLogDensityGradient( -0.001 ), 0.0);
  EXPECT_EQ(d->getLogDensityGradient( 1.001 ), 0.0);
  EXPECT_EQ(d->getLogDensityHessian( -0.001 ), 0.0);
  EXPECT_EQ(d->getLogDensityHessian( 1.001 ), 0.0);

  // Normal case for log density gradient and hessian
  ASSERT_NO_THROW(d->getLogDensityGradient( 0.5 ));
  ASSERT_NO_THROW(d->getLogDensityHessian( 0.5 ));
 }

 TEST(Distrtibutions, CauchyDistribution)
 {
  knlohmann::json distributionJs;
  Experiment e;
  distribution::univariate::Cauchy* d;

  // Creating distribution with an incorrect name
  distributionJs["Type"] = "Distribution/Univariate/Cauchy";
  ASSERT_ANY_THROW(d = dynamic_cast<korali::distribution::univariate::Cauchy *>(Module::getModule(distributionJs, &e)));

  // Creating distribution correctly now
  distributionJs["Type"] = "Univariate/Cauchy";
  ASSERT_NO_THROW(d = dynamic_cast<korali::distribution::univariate::Cauchy *>(Module::getModule(distributionJs, &e)));

  //////////////////////////////////////////////////

  // Getting module defaults
  distributionJs["Name"] = "Test";
  ASSERT_NO_THROW(d->applyModuleDefaults(distributionJs));
  ASSERT_NO_THROW(d->applyVariableDefaults());
  auto baseJs = distributionJs;

  // Testing conditional variables
  distributionJs = baseJs;
  distributionJs["Scale"] = "Conditional 1";
  ASSERT_NO_THROW(d->setConfiguration(distributionJs));

  // Testing get configuration method
  d->getConfiguration(distributionJs);
  ASSERT_EQ(distributionJs["Type"].get<std::string>(), "univariate/cauchy");
  ASSERT_EQ(distributionJs["Scale"].get<std::string>(), "Conditional 1");

  // Testing correct shape
  distributionJs["Scale"] = 0.5;
  ASSERT_NO_THROW(d->setConfiguration(distributionJs));
  ASSERT_NO_THROW(d->updateDistribution());

  // Testing get configuration method
  d->getConfiguration(distributionJs);
  ASSERT_EQ(distributionJs["Type"].get<std::string>(), "univariate/cauchy");
  ASSERT_EQ(distributionJs["Scale"].get<double>(), 0.5);

  // Testing get property pointer
  ASSERT_TRUE(d->getPropertyPointer("Scale") != NULL);
  ASSERT_ANY_THROW(d->getPropertyPointer("Undefined")); // Distribution not recognized

  // Testing incorrect shape (alpha)
  d->_scale = -0.5;
  ASSERT_ANY_THROW(d->updateDistribution());

  // Testing correct shape (alpha)
  d->_scale = 0.5;
  ASSERT_NO_THROW(d->updateDistribution());

  /////////////////////////////////////////////////

  // Distributions generated with https://keisan.casio.com/exec/system/1180573226

  // Testing expected density
  EXPECT_NEAR(d->getDensity( -5.00 ), 0.006303166, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -4.90 ), 0.006560385, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -4.80 ), 0.006833617, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -4.70 ), 0.007124214, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -4.60 ), 0.007433673, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -4.50 ), 0.007763656, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -4.40 ), 0.008116009, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -4.30 ), 0.008492793, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -4.20 ), 0.008896308, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -4.10 ), 0.009329129, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -4.00 ), 0.009794150, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -3.90 ), 0.010294628, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -3.80 ), 0.010834237, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -3.70 ), 0.011417141, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -3.60 ), 0.012048065, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -3.50 ), 0.012732395, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -3.40 ), 0.013476286, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -3.30 ), 0.014286799, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -3.20 ), 0.015172063, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -3.10 ), 0.016141475, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -3.00 ), 0.017205940, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -2.90 ), 0.018378169, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -2.80 ), 0.019673046, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -2.70 ), 0.021108083, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -2.60 ), 0.022703986, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -2.50 ), 0.024485376, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -2.40 ), 0.026481688, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -2.30 ), 0.028728329, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -2.20 ), 0.031268162, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -2.10 ), 0.034153421, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -2.00 ), 0.037448222, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -1.90 ), 0.041231851, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -1.80 ), 0.045603136, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -1.70 ), 0.050686288, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -1.60 ), 0.056638770, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -1.50 ), 0.063661977, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -1.40 ), 0.072015811, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -1.30 ), 0.082038630, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -1.20 ), 0.094174523, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -1.10 ), 0.109010235, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -1.00 ), 0.127323955, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -0.90 ), 0.150146173, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -0.80 ), 0.178825779, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -0.70 ), 0.215074247, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -0.60 ), 0.260909743, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -0.50 ), 0.318309886, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -0.40 ), 0.388182788, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -0.30 ), 0.468102774, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -0.20 ), 0.548810149, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -0.10 ), 0.612134397, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.00 ), 0.636619772, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.10 ), 0.612134397, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.20 ), 0.548810149, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.30 ), 0.468102774, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.40 ), 0.388182788, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.50 ), 0.318309886, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.60 ), 0.260909743, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.70 ), 0.215074247, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.80 ), 0.178825779, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.90 ), 0.150146173, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.00 ), 0.127323955, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.10 ), 0.109010235, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.20 ), 0.094174523, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.30 ), 0.082038630, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.40 ), 0.072015811, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.50 ), 0.063661977, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.60 ), 0.056638770, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.70 ), 0.050686288, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.80 ), 0.045603136, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.90 ), 0.041231851, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.00 ), 0.037448222, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.10 ), 0.034153421, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.20 ), 0.031268162, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.30 ), 0.028728329, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.40 ), 0.026481688, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.50 ), 0.024485376, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.60 ), 0.022703986, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.70 ), 0.021108083, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.80 ), 0.019673046, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.90 ), 0.018378169, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.00 ), 0.017205940, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.10 ), 0.016141475, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.20 ), 0.015172063, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.30 ), 0.014286799, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.40 ), 0.013476286, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.50 ), 0.012732395, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.60 ), 0.012048065, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.70 ), 0.011417141, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.80 ), 0.010834237, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.90 ), 0.010294628, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.00 ), 0.009794150, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.10 ), 0.009329129, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.20 ), 0.008896308, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.30 ), 0.008492793, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.40 ), 0.008116009, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.50 ), 0.007763656, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.60 ), 0.007433673, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.70 ), 0.007124214, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.80 ), 0.006833617, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.90 ), 0.006560385, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 5.00 ), 0.006303166, PDENSITY_ERROR_TOLERANCE);

  // Testing log density function
  EXPECT_NEAR(d->getLogDensity( -5.00 ), -5.066703222, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -4.90 ), -5.026705970, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -4.80 ), -4.985901150, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -4.70 ), -4.944255860, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -4.60 ), -4.901735169, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -4.50 ), -4.858301952, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -4.40 ), -4.813916707, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -4.30 ), -4.768537343, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -4.20 ), -4.722118964, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -4.10 ), -4.674613608, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -4.00 ), -4.625969975, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -3.90 ), -4.576133109, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -3.80 ), -4.525044056, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -3.70 ), -4.472639472, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -3.60 ), -4.418851185, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -3.50 ), -4.363605711, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -3.40 ), -4.306823697, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -3.30 ), -4.248419301, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -3.20 ), -4.188299489, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -3.10 ), -4.126363235, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -3.00 ), -4.062500618, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -2.90 ), -3.996591789, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -2.80 ), -3.928505797, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -2.70 ), -3.858099248, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -2.60 ), -3.785214767, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -2.50 ), -3.709679243, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -2.40 ), -3.631301815, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -2.30 ), -3.549871567, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -2.20 ), -3.465154897, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -2.10 ), -3.376892515, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -2.00 ), -3.284796049, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -1.90 ), -3.188544250, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -1.80 ), -3.087778803, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -1.70 ), -2.982099866, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -1.60 ), -2.871061550, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -1.50 ), -2.754167798, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -1.40 ), -2.630869582, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -1.30 ), -2.500565039, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -1.20 ), -2.362605595, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -1.10 ), -2.216313502, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -1.00 ), -2.061020618, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -0.90 ), -1.896145975, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -0.80 ), -1.721343250, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -0.70 ), -1.536771974, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -0.60 ), -1.343580744, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -0.50 ), -1.144729886, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -0.40 ), -0.946278947, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -0.30 ), -0.759067405, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -0.20 ), -0.600002710, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -0.10 ), -0.490803418, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.00 ), -0.451582705, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.10 ), -0.490803418, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.20 ), -0.600002710, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.30 ), -0.759067405, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.40 ), -0.946278947, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.50 ), -1.144729886, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.60 ), -1.343580744, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.70 ), -1.536771974, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.80 ), -1.721343250, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.90 ), -1.896145975, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 1.00 ), -2.061020618, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 1.10 ), -2.216313502, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 1.20 ), -2.362605595, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 1.30 ), -2.500565039, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 1.40 ), -2.630869582, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 1.50 ), -2.754167798, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 1.60 ), -2.871061550, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 1.70 ), -2.982099866, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 1.80 ), -3.087778803, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 1.90 ), -3.188544250, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.00 ), -3.284796049, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.10 ), -3.376892515, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.20 ), -3.465154897, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.30 ), -3.549871567, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.40 ), -3.631301815, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.50 ), -3.709679243, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.60 ), -3.785214767, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.70 ), -3.858099248, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.80 ), -3.928505797, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.90 ), -3.996591789, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.00 ), -4.062500618, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.10 ), -4.126363235, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.20 ), -4.188299489, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.30 ), -4.248419301, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.40 ), -4.306823697, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.50 ), -4.363605711, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.60 ), -4.418851185, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.70 ), -4.472639472, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.80 ), -4.525044056, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.90 ), -4.576133109, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.00 ), -4.625969975, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.10 ), -4.674613608, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.20 ), -4.722118964, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.30 ), -4.768537343, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.40 ), -4.813916707, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.50 ), -4.858301952, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.60 ), -4.901735169, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.70 ), -4.944255860, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.80 ), -4.985901150, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.90 ), -5.026705970, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 5.00 ), -5.066703222, PDENSITY_ERROR_TOLERANCE);

  // Normal case for get random number, ge log density gradient and hessian
  ASSERT_NO_THROW(d->getRandomNumber());
  ASSERT_NO_THROW(d->getLogDensityGradient( 0.5 ));
  ASSERT_NO_THROW(d->getLogDensityHessian( 0.5 ));
 }

 TEST(Distrtibutions, ExponentialDistribution)
 {
  knlohmann::json distributionJs;
  Experiment e;
  distribution::univariate::Exponential* d;

  // Creating distribution with an incorrect name
  distributionJs["Type"] = "Distribution/Univariate/Exponential";
  ASSERT_ANY_THROW(d = dynamic_cast<korali::distribution::univariate::Exponential *>(Module::getModule(distributionJs, &e)));

  // Creating distribution correctly now
  distributionJs["Type"] = "Univariate/Exponential";
  ASSERT_NO_THROW(d = dynamic_cast<korali::distribution::univariate::Exponential *>(Module::getModule(distributionJs, &e)));

  //////////////////////////////////////////////////

  // Getting module defaults
  distributionJs["Name"] = "Test";
  ASSERT_NO_THROW(d->applyModuleDefaults(distributionJs));
  ASSERT_NO_THROW(d->applyVariableDefaults());
  auto baseJs = distributionJs;

  // Testing conditional variables
  distributionJs = baseJs;
  distributionJs["Mean"] = "Conditional 1";
  distributionJs["Location"] = "Conditional 2";
  ASSERT_NO_THROW(d->setConfiguration(distributionJs));

  // Testing get configuration method
  d->getConfiguration(distributionJs);
  ASSERT_EQ(distributionJs["Type"].get<std::string>(), "univariate/exponential");
  ASSERT_EQ(distributionJs["Mean"].get<std::string>(), "Conditional 1");
  ASSERT_EQ(distributionJs["Location"].get<std::string>(), "Conditional 2");

  // Testing correct shape
  distributionJs["Mean"] = 2.0;
  distributionJs["Location"] = -0.1;
  ASSERT_NO_THROW(d->setConfiguration(distributionJs));
  ASSERT_NO_THROW(d->updateDistribution());

  // Testing get configuration method
  d->getConfiguration(distributionJs);
  ASSERT_EQ(distributionJs["Type"].get<std::string>(), "univariate/exponential");
  ASSERT_EQ(distributionJs["Mean"].get<double>(), 2.0);
  ASSERT_EQ(distributionJs["Location"].get<double>(), -0.1);

  // Testing get property pointer
  ASSERT_TRUE(d->getPropertyPointer("Mean") != NULL);
  ASSERT_TRUE(d->getPropertyPointer("Location") != NULL);
  ASSERT_ANY_THROW(d->getPropertyPointer("Undefined")); // Distribution not recognized

  /////////////////////////////////////////////////

  // Distributions generated with https://keisan.casio.com/exec/system/1180573226

  // Testing expected density
  EXPECT_NEAR(d->getDensity( 0.00 ), 0.475614712, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.10 ), 0.452418709, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.20 ), 0.430353988, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.30 ), 0.409365377, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.40 ), 0.389400392, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.50 ), 0.370409110, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.60 ), 0.352344045, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.70 ), 0.335160023, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.80 ), 0.318814076, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.90 ), 0.303265330, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.00 ), 0.288474905, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.10 ), 0.274405818, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.20 ), 0.261022888, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.30 ), 0.248292652, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.40 ), 0.236183276, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.50 ), 0.224664482, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.60 ), 0.213707466, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.70 ), 0.203284830, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.80 ), 0.193370512, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.90 ), 0.183939721, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.00 ), 0.174968875, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.10 ), 0.166435542, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.20 ), 0.158318385, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.30 ), 0.150597106, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.40 ), 0.143252398, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.50 ), 0.136265897, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.60 ), 0.129620130, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.70 ), 0.123298482, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.80 ), 0.117285144, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.90 ), 0.111565080, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.00 ), 0.106123987, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.10 ), 0.100948259, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.20 ), 0.096024954, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.30 ), 0.091341762, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.40 ), 0.086886972, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.50 ), 0.082649444, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.60 ), 0.078618583, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.70 ), 0.074784310, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.80 ), 0.071137036, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.90 ), 0.067667642, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.00 ), 0.064367452, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.10 ), 0.061228214, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.20 ), 0.058242079, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.30 ), 0.055401579, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.40 ), 0.052699612, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.50 ), 0.050129422, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.60 ), 0.047684581, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.70 ), 0.045358977, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.80 ), 0.043146793, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.90 ), 0.041042499, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 5.00 ), 0.039040833, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 5.10 ), 0.037136789, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 5.20 ), 0.035325607, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 5.30 ), 0.033602756, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 5.40 ), 0.031963931, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 5.50 ), 0.030405031, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 5.60 ), 0.028922160, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 5.70 ), 0.027511610, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 5.80 ), 0.026169853, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 5.90 ), 0.024893534, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 6.00 ), 0.023679462, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 6.10 ), 0.022524601, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 6.20 ), 0.021426063, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 6.30 ), 0.020381102, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 6.40 ), 0.019387104, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 6.50 ), 0.018441584, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 6.60 ), 0.017542177, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 6.70 ), 0.016686635, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 6.80 ), 0.015872818, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 6.90 ), 0.015098692, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 7.00 ), 0.014362320, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 7.10 ), 0.013661861, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 7.20 ), 0.012995564, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 7.30 ), 0.012361763, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 7.40 ), 0.011758873, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 7.50 ), 0.011185386, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 7.60 ), 0.010639868, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 7.70 ), 0.010120956, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 7.80 ), 0.009627351, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 7.90 ), 0.009157819, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 8.00 ), 0.008711187, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 8.10 ), 0.008286338, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 8.20 ), 0.007882208, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 8.30 ), 0.007497788, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 8.40 ), 0.007132117, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 8.50 ), 0.006784280, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 8.60 ), 0.006453406, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 8.70 ), 0.006138670, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 8.80 ), 0.005839283, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 8.90 ), 0.005554498, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 9.00 ), 0.005283602, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 9.10 ), 0.005025918, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 9.20 ), 0.004780801, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 9.30 ), 0.004547639, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 9.40 ), 0.004325848, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 9.50 ), 0.004114874, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 9.60 ), 0.003914189, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 9.70 ), 0.003723292, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 9.80 ), 0.003541704, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 9.90 ), 0.003368974, PDENSITY_ERROR_TOLERANCE);

  // Testing log density function
  EXPECT_NEAR(d->getLogDensity( 0.00 ), -0.743147180, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.10 ), -0.793147181, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.20 ), -0.843147181, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.30 ), -0.893147181, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.40 ), -0.943147181, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.50 ), -0.993147181, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.60 ), -1.043147180, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.70 ), -1.093147181, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.80 ), -1.143147181, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.90 ), -1.193147180, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 1.00 ), -1.243147181, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 1.10 ), -1.293147180, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 1.20 ), -1.343147180, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 1.30 ), -1.393147181, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 1.40 ), -1.443147180, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 1.50 ), -1.493147180, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 1.60 ), -1.543147180, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 1.70 ), -1.593147180, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 1.80 ), -1.643147181, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 1.90 ), -1.693147180, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.00 ), -1.743147180, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.10 ), -1.793147180, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.20 ), -1.843147180, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.30 ), -1.893147180, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.40 ), -1.943147181, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.50 ), -1.993147181, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.60 ), -2.043147181, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.70 ), -2.093147180, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.80 ), -2.143147180, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.90 ), -2.193147180, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.00 ), -2.243147181, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.10 ), -2.293147181, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.20 ), -2.343147181, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.30 ), -2.393147181, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.40 ), -2.443147181, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.50 ), -2.493147181, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.60 ), -2.543147181, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.70 ), -2.593147181, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.80 ), -2.643147180, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.90 ), -2.693147181, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.00 ), -2.743147180, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.10 ), -2.793147181, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.20 ), -2.843147181, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.30 ), -2.893147181, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.40 ), -2.943147181, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.50 ), -2.993147181, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.60 ), -3.043147181, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.70 ), -3.093147181, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.80 ), -3.143147181, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.90 ), -3.193147181, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 5.00 ), -3.243147181, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 5.10 ), -3.293147180, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 5.20 ), -3.343147181, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 5.30 ), -3.393147181, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 5.40 ), -3.443147181, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 5.50 ), -3.493147181, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 5.60 ), -3.543147180, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 5.70 ), -3.593147180, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 5.80 ), -3.643147180, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 5.90 ), -3.693147181, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 6.00 ), -3.743147180, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 6.10 ), -3.793147180, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 6.20 ), -3.843147181, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 6.30 ), -3.893147181, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 6.40 ), -3.943147180, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 6.50 ), -3.993147181, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 6.60 ), -4.043147181, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 6.70 ), -4.093147181, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 6.80 ), -4.143147180, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 6.90 ), -4.193147181, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 7.00 ), -4.243147180, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 7.10 ), -4.293147181, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 7.20 ), -4.343147181, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 7.30 ), -4.393147180, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 7.40 ), -4.443147180, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 7.50 ), -4.493147180, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 7.60 ), -4.543147180, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 7.70 ), -4.593147181, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 7.80 ), -4.643147181, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 7.90 ), -4.693147180, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 8.00 ), -4.743147181, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 8.10 ), -4.793147181, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 8.20 ), -4.843147180, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 8.30 ), -4.893147181, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 8.40 ), -4.943147180, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 8.50 ), -4.993147181, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 8.60 ), -5.043147180, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 8.70 ), -5.093147180, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 8.80 ), -5.143147181, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 8.90 ), -5.193147181, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 9.00 ), -5.243147181, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 9.10 ), -5.293147180, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 9.20 ), -5.343147181, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 9.30 ), -5.393147181, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 9.40 ), -5.443147180, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 9.50 ), -5.493147180, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 9.60 ), -5.543147180, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 9.70 ), -5.593147180, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 9.80 ), -5.643147180, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 9.90 ), -5.693147180, PDENSITY_ERROR_TOLERANCE);

  // Testing extreme case for log density
  EXPECT_EQ(d->getLogDensity( -0.2 ), -INFINITY);
  EXPECT_EQ(d->getLogDensityGradient( -0.2 ), 0.0);

 // Checking random numbers are within the expected range
  for (size_t i = 0; i < 100; i++)
  {
   double y = d->getRandomNumber();
   EXPECT_TRUE(y >= d->_location);
  }

  // Testing extreme for log density gradient and hessian
  EXPECT_TRUE(d->getLogDensityGradient( 3.0 ) < 0.0);
  EXPECT_EQ(d->getLogDensityHessian( 0.5 ), 0.0);
 }

 TEST(Distrtibutions, GammaDistribution)
 {
  knlohmann::json distributionJs;
  Experiment e;
  distribution::univariate::Gamma* d;

  // Creating distribution with an incorrect name
  distributionJs["Type"] = "Distribution/Univariate/Gamma";
  ASSERT_ANY_THROW(d = dynamic_cast<korali::distribution::univariate::Gamma *>(Module::getModule(distributionJs, &e)));

  // Creating distribution correctly now
  distributionJs["Type"] = "Univariate/Gamma";
  ASSERT_NO_THROW(d = dynamic_cast<korali::distribution::univariate::Gamma *>(Module::getModule(distributionJs, &e)));

  //////////////////////////////////////////////////

  // Getting module defaults
  distributionJs["Name"] = "Test";
  ASSERT_NO_THROW(d->applyModuleDefaults(distributionJs));
  ASSERT_NO_THROW(d->applyVariableDefaults());
  auto baseJs = distributionJs;

  // Testing conditional variables
  distributionJs = baseJs;
  distributionJs["Shape"] = "Conditional 1";
  distributionJs["Scale"] = "Conditional 2";
  ASSERT_NO_THROW(d->setConfiguration(distributionJs));

  // Testing get configuration method
  d->getConfiguration(distributionJs);
  ASSERT_EQ(distributionJs["Type"].get<std::string>(), "univariate/gamma");
  ASSERT_EQ(distributionJs["Shape"].get<std::string>(), "Conditional 1");
  ASSERT_EQ(distributionJs["Scale"].get<std::string>(), "Conditional 2");

  // Testing correct shape
  distributionJs["Shape"] = 0.5;
  distributionJs["Scale"] = 0.5;
  ASSERT_NO_THROW(d->setConfiguration(distributionJs));
  ASSERT_NO_THROW(d->updateDistribution());

  // Testing get configuration method
  d->getConfiguration(distributionJs);
  ASSERT_EQ(distributionJs["Type"].get<std::string>(), "univariate/gamma");
  ASSERT_EQ(distributionJs["Shape"].get<double>(), 0.5);
  ASSERT_EQ(distributionJs["Scale"].get<double>(), 0.5);

  // Testing get property pointer
  ASSERT_TRUE(d->getPropertyPointer("Shape") != NULL);
  ASSERT_TRUE(d->getPropertyPointer("Scale") != NULL);
  ASSERT_ANY_THROW(d->getPropertyPointer("Undefined")); // Distribution not recognized

  // Testing incorrect scale
  d->_shape = 0.5;
  d->_scale = -0.5;
  ASSERT_ANY_THROW(d->updateDistribution());

  // Testing incorrect shape
  d->_shape = -0.5;
  d->_scale = 0.5;
  ASSERT_ANY_THROW(d->updateDistribution());

  // Testing correct scale
  d->_shape = 3.0;
  d->_scale = 1.0;
  ASSERT_NO_THROW(d->updateDistribution());

  /////////////////////////////////////////////////

  // Distributions generated with https://keisan.casio.com/exec/system/1180573226

  // Testing expected density
  EXPECT_NEAR(d->getDensity( 0.00 ), 0.000000000, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.10 ), 0.004524187, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.20 ), 0.016374615, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.30 ), 0.033336820, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.40 ), 0.053625604, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.50 ), 0.075816332, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.60 ), 0.098786095, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.70 ), 0.121663399, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.80 ), 0.143785269, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.90 ), 0.164660712, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.00 ), 0.183939721, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.10 ), 0.201387006, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.20 ), 0.216859833, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.30 ), 0.230289365, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.40 ), 0.241665025, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.50 ), 0.251021430, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.60 ), 0.258427543, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.70 ), 0.263977692, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.80 ), 0.267784199, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.90 ), 0.269971358, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.00 ), 0.270670567, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.10 ), 0.270016424, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.20 ), 0.268143643, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.30 ), 0.265184642, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.40 ), 0.261267706, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.50 ), 0.256515621, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.60 ), 0.251044694, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.70 ), 0.244964094, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.80 ), 0.238375446, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.90 ), 0.231372640, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.00 ), 0.224041808, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.10 ), 0.216461418, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.20 ), 0.208702484, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.30 ), 0.200828847, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.40 ), 0.192897500, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.50 ), 0.184958974, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.60 ), 0.177057722, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.70 ), 0.169232539, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.80 ), 0.161516973, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.90 ), 0.153939737, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.00 ), 0.146525111, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.10 ), 0.139293337, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.20 ), 0.132260988, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.30 ), 0.125441328, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.40 ), 0.118844650, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.50 ), 0.112478590, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.60 ), 0.106348422, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.70 ), 0.100457336, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.80 ), 0.094806686, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.90 ), 0.089396230, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 5.00 ), 0.084224337, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 5.10 ), 0.079288189, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 5.20 ), 0.074583951, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 5.30 ), 0.070106936, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 5.40 ), 0.065851750, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 5.50 ), 0.061812418, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 5.60 ), 0.057982503, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 5.70 ), 0.054355209, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 5.80 ), 0.050923471, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 5.90 ), 0.047680037, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 6.00 ), 0.044617539, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 6.10 ), 0.041728554, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 6.20 ), 0.039005657, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 6.30 ), 0.036441468, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 6.40 ), 0.034028693, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 6.50 ), 0.031760153, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 6.60 ), 0.029628816, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 6.70 ), 0.027627818, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 6.80 ), 0.025750481, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 6.90 ), 0.023990332, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 7.00 ), 0.022341108, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 7.10 ), 0.020796770, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 7.20 ), 0.019351504, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 7.30 ), 0.017999731, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 7.40 ), 0.016736101, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 7.50 ), 0.015555498, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 7.60 ), 0.014453037, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 7.70 ), 0.013424062, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 7.80 ), 0.012464138, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 7.90 ), 0.011569052, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 8.00 ), 0.010734804, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 8.10 ), 0.009957601, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 8.20 ), 0.009233853, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 8.30 ), 0.008560162, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 8.40 ), 0.007933319, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 8.50 ), 0.007350295, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 8.60 ), 0.006808232, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 8.70 ), 0.006304440, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 8.80 ), 0.005836385, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 8.90 ), 0.005401683, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 9.00 ), 0.004998097, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 9.10 ), 0.004623523, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 9.20 ), 0.004275987, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 9.30 ), 0.003953641, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 9.40 ), 0.003654749, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 9.50 ), 0.003377689, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 9.60 ), 0.003120940, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 9.70 ), 0.002883082, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 9.80 ), 0.002662786, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 9.90 ), 0.002458810, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 10.00 ), 0.002269996, PDENSITY_ERROR_TOLERANCE);

  // Testing log density function
  EXPECT_NEAR(d->getLogDensity( 0.10 ), -5.398317367, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.20 ), -4.112023006, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.30 ), -3.401092789, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.40 ), -2.925728644, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.50 ), -2.579441542, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.60 ), -2.314798428, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.70 ), -2.106497069, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.80 ), -1.939434283, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.90 ), -1.803868212, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 1.00 ), -1.693147180, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 1.10 ), -1.602526821, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 1.20 ), -1.528504067, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 1.30 ), -1.468418652, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 1.40 ), -1.420202707, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 1.50 ), -1.382216964, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 1.60 ), -1.353139922, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 1.70 ), -1.331890678, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 1.80 ), -1.317573851, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 1.90 ), -1.309439408, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.00 ), -1.306852819, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.10 ), -1.309272491, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.20 ), -1.316232460, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.30 ), -1.327328935, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.40 ), -1.342209706, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.50 ), -1.360565717, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.60 ), -1.382124290, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.70 ), -1.406643635, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.80 ), -1.433908346, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.90 ), -1.463725707, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.00 ), -1.495922603, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.10 ), -1.530342958, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.20 ), -1.566845561, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.30 ), -1.605302244, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.40 ), -1.645596317, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.50 ), -1.687621243, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.60 ), -1.731279489, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.70 ), -1.776481541, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.80 ), -1.823145047, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.90 ), -1.871194075, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.00 ), -1.920558458, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.10 ), -1.971173233, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.20 ), -2.022978130, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.30 ), -2.075917135, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.40 ), -2.129938098, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.50 ), -2.184992387, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.60 ), -2.241034573, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.70 ), -2.298022163, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.80 ), -2.355915345, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.90 ), -2.414676770, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 5.00 ), -2.474271356, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 5.10 ), -2.534666101, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 5.20 ), -2.595829929, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 5.30 ), -2.657733539, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 5.40 ), -2.720349273, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 5.50 ), -2.783650996, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 5.60 ), -2.847613985, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 5.70 ), -2.912214831, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 5.80 ), -2.977431345, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 5.90 ), -3.043242479, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 6.00 ), -3.109628242, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 6.10 ), -3.176569638, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 6.20 ), -3.244048596, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 6.30 ), -3.312047914, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 6.40 ), -3.380551200, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 6.50 ), -3.449542827, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 6.60 ), -3.519007882, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 6.70 ), -3.588932128, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 6.80 ), -3.659301956, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 6.90 ), -3.730104357, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 7.00 ), -3.801326882, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 7.10 ), -3.872957613, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 7.20 ), -3.944985129, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 7.30 ), -4.017398484, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 7.40 ), -4.090187180, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 7.50 ), -4.163341140, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 7.60 ), -4.236850686, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 7.70 ), -4.310706523, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 7.80 ), -4.384899713, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 7.90 ), -4.459421662, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 8.00 ), -4.534264097, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 8.10 ), -4.609419057, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 8.20 ), -4.684878872, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 8.30 ), -4.760636151, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 8.40 ), -4.836683769, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 8.50 ), -4.913014854, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 8.60 ), -4.989622774, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 8.70 ), -5.066501129, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 8.80 ), -5.143643738, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 8.90 ), -5.221044627, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 9.00 ), -5.298698026, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 9.10 ), -5.376598353, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 9.20 ), -5.454740212, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 9.30 ), -5.533118380, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 9.40 ), -5.611727802, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 9.50 ), -5.690563583, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 9.60 ), -5.769620984, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 9.70 ), -5.848895409, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 9.80 ), -5.928382409, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 9.90 ), -6.008077666, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 10.00 ), -6.087976995, PDENSITY_ERROR_TOLERANCE);

  // Testing extreme cases for log density
  EXPECT_EQ(d->getLogDensity( -0.001 ), -INFINITY);

 // Checking random numbers are within the expected range
  for (size_t i = 0; i < 100; i++)
  {
   double y = d->getRandomNumber();
   EXPECT_TRUE(y >= 0.0);
  }

  // Testing extreme for log density gradient and hessian
  EXPECT_EQ(d->getLogDensityGradient( -0.001 ), 0.0);
  EXPECT_EQ(d->getLogDensityHessian( -0.001 ), 0.0);

  // Normal case for log density gradient and hessian
  ASSERT_NO_THROW(d->getLogDensityGradient( 0.5 ));
  ASSERT_NO_THROW(d->getLogDensityHessian( 0.5 ));
 }

 TEST(Distrtibutions, GeometricDistribution)
 {
  knlohmann::json distributionJs;
  Experiment e;
  distribution::univariate::Geometric* d;

  // Creating distribution with an incorrect name
  distributionJs["Type"] = "Distribution/Univariate/Geometric";
  ASSERT_ANY_THROW(d = dynamic_cast<korali::distribution::univariate::Geometric *>(Module::getModule(distributionJs, &e)));

  // Creating distribution correctly now
  distributionJs["Type"] = "Univariate/Geometric";
  ASSERT_NO_THROW(d = dynamic_cast<korali::distribution::univariate::Geometric *>(Module::getModule(distributionJs, &e)));

  //////////////////////////////////////////////////

  // Getting module defaults
  distributionJs["Name"] = "Test";
  ASSERT_NO_THROW(d->applyModuleDefaults(distributionJs));
  ASSERT_NO_THROW(d->applyVariableDefaults());
  auto baseJs = distributionJs;

  // Testing conditional variables
  distributionJs = baseJs;
  distributionJs["Success Probability"] = "Conditional 1";
  ASSERT_NO_THROW(d->setConfiguration(distributionJs));

  // Testing get configuration method
  d->getConfiguration(distributionJs);
  ASSERT_EQ(distributionJs["Type"].get<std::string>(), "univariate/geometric");
  ASSERT_EQ(distributionJs["Success Probability"].get<std::string>(), "Conditional 1");

  // Testing correct shape
  distributionJs["Success Probability"] = 0.1;
  ASSERT_NO_THROW(d->setConfiguration(distributionJs));
  ASSERT_NO_THROW(d->updateDistribution());

  // Testing get configuration method
  d->getConfiguration(distributionJs);
  ASSERT_EQ(distributionJs["Type"].get<std::string>(), "univariate/geometric");
  ASSERT_EQ(distributionJs["Success Probability"].get<double>(), 0.1);

  // Testing get property pointer
  ASSERT_TRUE(d->getPropertyPointer("Success Probability") != NULL);
  ASSERT_ANY_THROW(d->getPropertyPointer("Undefined")); // Distribution not recognized

  // Testing incorrect scale
  d->_successProbability = 0.1;
  ASSERT_NO_THROW(d->updateDistribution());

  /////////////////////////////////////////////////

  // Distributions generated with https://keisan.casio.com/exec/system/1180573226

  // Testing expected density
  EXPECT_NEAR(d->getDensity( 0.00 ), 0.000000000, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.00 ), 0.100000000, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.00 ), 0.090000000, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.00 ), 0.081000000, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.00 ), 0.072900000, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 5.00 ), 0.065610000, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 6.00 ), 0.059049000, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 7.00 ), 0.053144100, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 8.00 ), 0.047829690, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 9.00 ), 0.043046721, PDENSITY_ERROR_TOLERANCE);

  // Testing log density function
  EXPECT_NEAR(d->getLogDensity( 0.00 ), -2.197224577, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 1.00 ), -2.302585093, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.00 ), -2.407945609, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.00 ), -2.513306124, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.00 ), -2.618666640, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 5.00 ), -2.724027156, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 6.00 ), -2.829387671, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 7.00 ), -2.934748187, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 8.00 ), -3.040108703, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 9.00 ), -3.145469218, PDENSITY_ERROR_TOLERANCE);

  // Testing extreme case for log density
  ASSERT_NO_THROW(d->getLogDensity(1));
  ASSERT_ANY_THROW(d->getLogDensityGradient(1));

 // Checking random numbers are within the expected range
  for (size_t i = 0; i < 100; i++)
  {
   double y = d->getRandomNumber();
   EXPECT_TRUE(y >= 0.0);
  }

  // Testing extreme for log density gradient and hessian
  ASSERT_ANY_THROW(d->getLogDensityGradient(1));
  ASSERT_ANY_THROW(d->getLogDensityHessian(1));
 }

 TEST(Distrtibutions, iGammaDistribution)
  {
   knlohmann::json distributionJs;
   Experiment e;
   distribution::univariate::Igamma* d;

   // Creating distribution with an incorrect name
   distributionJs["Type"] = "Distribution/Univariate/Igamma";
   ASSERT_ANY_THROW(d = dynamic_cast<korali::distribution::univariate::Igamma *>(Module::getModule(distributionJs, &e)));

   // Creating distribution correctly now
   distributionJs["Type"] = "Univariate/Igamma";
   ASSERT_NO_THROW(d = dynamic_cast<korali::distribution::univariate::Igamma *>(Module::getModule(distributionJs, &e)));

   //////////////////////////////////////////////////

   // Getting module defaults
   distributionJs["Name"] = "Test";
   ASSERT_NO_THROW(d->applyModuleDefaults(distributionJs));
   ASSERT_NO_THROW(d->applyVariableDefaults());
   auto baseJs = distributionJs;

   // Testing conditional variables
   distributionJs = baseJs;
   distributionJs["Shape"] = "Conditional 1";
   distributionJs["Scale"] = "Conditional 2";
   ASSERT_NO_THROW(d->setConfiguration(distributionJs));

   // Testing get configuration method
   d->getConfiguration(distributionJs);
   ASSERT_EQ(distributionJs["Type"].get<std::string>(), "univariate/igamma");
   ASSERT_EQ(distributionJs["Shape"].get<std::string>(), "Conditional 1");
   ASSERT_EQ(distributionJs["Scale"].get<std::string>(), "Conditional 2");

   // Testing correct shape
   distributionJs["Shape"] = 1.0;
   distributionJs["Scale"] = 2.0;
   ASSERT_NO_THROW(d->setConfiguration(distributionJs));
   ASSERT_NO_THROW(d->updateDistribution());

   // Testing get configuration method
   d->getConfiguration(distributionJs);
   ASSERT_EQ(distributionJs["Type"].get<std::string>(), "univariate/igamma");
   ASSERT_EQ(distributionJs["Shape"].get<double>(), 1.0);
   ASSERT_EQ(distributionJs["Scale"].get<double>(), 2.0);

   // Testing get property pointer
   ASSERT_TRUE(d->getPropertyPointer("Shape") != NULL);
   ASSERT_TRUE(d->getPropertyPointer("Scale") != NULL);
   ASSERT_ANY_THROW(d->getPropertyPointer("Undefined")); // Distribution not recognized

   // Testing incorrect shape
   d->_shape = -0.5;
   d->_scale = 0.5;
   ASSERT_ANY_THROW(d->updateDistribution());

   // Testing incorrect scale
   d->_shape = 0.5;
   d->_scale = -0.5;
   ASSERT_ANY_THROW(d->updateDistribution());

   // Testing correct scale
   d->_shape = 2.0;
   d->_scale = 1.0;
   ASSERT_NO_THROW(d->updateDistribution());

   /////////////////////////////////////////////////


   // Distributions generated with https://keisan.casio.com/exec/system/1180573226

   // Testing expected density
   EXPECT_EQ(d->getDensity( 0.00 ), -INFINITY);
   EXPECT_NEAR(d->getDensity( 1.00 ), 0.367879441, PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getDensity( 5.00 ), 0.006549846, PDENSITY_ERROR_TOLERANCE);

   // Testing log density function
   EXPECT_EQ(d->getLogDensity( 0.00 ), -INFINITY);
   EXPECT_NEAR(d->getLogDensity( 1.00 ), 0, PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getLogDensity( 5.00 ), -4.02831373, PDENSITY_ERROR_TOLERANCE);

  // Checking random numbers are within the expected range
   for (size_t i = 0; i < 100; i++)
   {
    double y = d->getRandomNumber();
    EXPECT_TRUE(y >= 0.0);
   }

   // Testing extreme for log density gradient and hessian
   EXPECT_EQ(d->getLogDensityGradient( -0.001 ), -INFINITY);
   EXPECT_EQ(d->getLogDensityHessian( -0.001 ), -INFINITY);

   // Normal case for log density gradient and hessian
   ASSERT_NO_THROW(d->getLogDensityGradient( 0.5 ));
   ASSERT_NO_THROW(d->getLogDensityHessian( 0.5 ));
  }

 TEST(Distrtibutions, LaplaceDistribution)
 {
  knlohmann::json distributionJs;
  Experiment e;
  distribution::univariate::Laplace* d;

  // Creating distribution with an incorrect name
  distributionJs["Type"] = "Distribution/Univariate/Laplace";
  ASSERT_ANY_THROW(d = dynamic_cast<korali::distribution::univariate::Laplace *>(Module::getModule(distributionJs, &e)));

  // Creating distribution correctly now
  distributionJs["Type"] = "Univariate/Laplace";
  ASSERT_NO_THROW(d = dynamic_cast<korali::distribution::univariate::Laplace *>(Module::getModule(distributionJs, &e)));

  //////////////////////////////////////////////////

  // Getting module defaults
  distributionJs["Name"] = "Test";
  ASSERT_NO_THROW(d->applyModuleDefaults(distributionJs));
  ASSERT_NO_THROW(d->applyVariableDefaults());
  auto baseJs = distributionJs;

  // Testing conditional variables
  distributionJs = baseJs;
  distributionJs["Width"] = "Conditional 1";
  ASSERT_NO_THROW(d->setConfiguration(distributionJs));

  // Testing get configuration method
  d->getConfiguration(distributionJs);
  ASSERT_EQ(distributionJs["Type"].get<std::string>(), "univariate/laplace");
  ASSERT_EQ(distributionJs["Width"].get<std::string>(), "Conditional 1");

  // Testing correct shape
  distributionJs["Width"] = 0.7;
  ASSERT_NO_THROW(d->setConfiguration(distributionJs));
  ASSERT_NO_THROW(d->updateDistribution());

  // Testing get configuration method
  d->getConfiguration(distributionJs);
  ASSERT_EQ(distributionJs["Type"].get<std::string>(), "univariate/laplace");
  ASSERT_EQ(distributionJs["Width"].get<double>(), 0.7);

  // Testing get property pointer
  ASSERT_TRUE(d->getPropertyPointer("Width") != NULL);
  ASSERT_ANY_THROW(d->getPropertyPointer("Undefined")); // Distribution not recognized

  // Testing incorrect width
  d->_width = -0.5;
  ASSERT_ANY_THROW(d->updateDistribution());

  // Testing correct width
  d->_width = 0.7;
  ASSERT_NO_THROW(d->updateDistribution());

  /////////////////////////////////////////////////

  // Distributions generated with https://keisan.casio.com/exec/system/1180573226

  // Testing expected density
  EXPECT_NEAR(d->getDensity( -5.00 ), 0.000564636, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -4.90 ), 0.000651344, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -4.80 ), 0.000751368, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -4.70 ), 0.000866752, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -4.60 ), 0.000999854, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -4.50 ), 0.001153397, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -4.40 ), 0.001330519, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -4.30 ), 0.001534840, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -4.20 ), 0.001770537, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -4.10 ), 0.002042430, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -4.00 ), 0.002356076, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -3.90 ), 0.002717886, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -3.80 ), 0.003135258, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -3.70 ), 0.003616724, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -3.60 ), 0.004172127, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -3.50 ), 0.004812819, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -3.40 ), 0.005551900, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -3.30 ), 0.006404477, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -3.20 ), 0.007387981, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -3.10 ), 0.008522516, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -3.00 ), 0.009831276, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -2.90 ), 0.011341016, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -2.80 ), 0.013082599, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -2.70 ), 0.015091628, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -2.60 ), 0.017409174, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -2.50 ), 0.020082614, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -2.40 ), 0.023166601, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -2.30 ), 0.026724180, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -2.20 ), 0.030828078, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -2.10 ), 0.035562192, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -2.00 ), 0.041023299, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -1.90 ), 0.047323042, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -1.80 ), 0.054590205, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -1.70 ), 0.062973350, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -1.60 ), 0.072643852, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -1.50 ), 0.083799404, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -1.40 ), 0.096668059, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -1.30 ), 0.111512890, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -1.20 ), 0.128637366, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -1.10 ), 0.148391562, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -1.00 ), 0.171179312, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -0.90 ), 0.197466462, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -0.80 ), 0.227790398, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -0.70 ), 0.262771029, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -0.60 ), 0.303123461, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -0.50 ), 0.349672614, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -0.40 ), 0.403370087, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -0.30 ), 0.465313613, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -0.20 ), 0.536769495, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -0.10 ), 0.619198500, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.00 ), 0.714285714, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.10 ), 0.619198500, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.20 ), 0.536769495, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.30 ), 0.465313613, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.40 ), 0.403370087, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.50 ), 0.349672614, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.60 ), 0.303123461, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.70 ), 0.262771029, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.80 ), 0.227790398, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.90 ), 0.197466462, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.00 ), 0.171179312, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.10 ), 0.148391562, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.20 ), 0.128637366, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.30 ), 0.111512890, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.40 ), 0.096668059, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.50 ), 0.083799404, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.60 ), 0.072643852, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.70 ), 0.062973350, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.80 ), 0.054590205, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.90 ), 0.047323042, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.00 ), 0.041023299, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.10 ), 0.035562192, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.20 ), 0.030828078, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.30 ), 0.026724180, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.40 ), 0.023166601, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.50 ), 0.020082614, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.60 ), 0.017409174, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.70 ), 0.015091628, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.80 ), 0.013082599, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.90 ), 0.011341016, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.00 ), 0.009831276, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.10 ), 0.008522516, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.20 ), 0.007387981, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.30 ), 0.006404477, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.40 ), 0.005551900, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.50 ), 0.004812819, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.60 ), 0.004172127, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.70 ), 0.003616724, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.80 ), 0.003135258, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.90 ), 0.002717886, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.00 ), 0.002356076, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.10 ), 0.002042430, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.20 ), 0.001770537, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.30 ), 0.001534840, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.40 ), 0.001330519, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.50 ), 0.001153397, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.60 ), 0.000999854, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.70 ), 0.000866752, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.80 ), 0.000751368, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.90 ), 0.000651344, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 5.00 ), 0.000564636, PDENSITY_ERROR_TOLERANCE);

  // Testing log density function
  EXPECT_NEAR(d->getLogDensity( -5.00 ), -7.479329380, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -4.90 ), -7.336472237, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -4.80 ), -7.193615094, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -4.70 ), -7.050757951, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -4.60 ), -6.907900808, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -4.50 ), -6.765043665, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -4.40 ), -6.622186523, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -4.30 ), -6.479329380, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -4.20 ), -6.336472237, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -4.10 ), -6.193615094, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -4.00 ), -6.050757951, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -3.90 ), -5.907900808, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -3.80 ), -5.765043665, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -3.70 ), -5.622186522, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -3.60 ), -5.479329380, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -3.50 ), -5.336472237, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -3.40 ), -5.193615094, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -3.30 ), -5.050757951, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -3.20 ), -4.907900808, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -3.10 ), -4.765043665, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -3.00 ), -4.622186522, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -2.90 ), -4.479329380, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -2.80 ), -4.336472236, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -2.70 ), -4.193615094, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -2.60 ), -4.050757951, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -2.50 ), -3.907900808, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -2.40 ), -3.765043665, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -2.30 ), -3.622186522, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -2.20 ), -3.479329380, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -2.10 ), -3.336472237, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -2.00 ), -3.193615094, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -1.90 ), -3.050757951, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -1.80 ), -2.907900808, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -1.70 ), -2.765043665, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -1.60 ), -2.622186522, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -1.50 ), -2.479329380, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -1.40 ), -2.336472237, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -1.30 ), -2.193615094, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -1.20 ), -2.050757951, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -1.10 ), -1.907900808, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -1.00 ), -1.765043665, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -0.90 ), -1.622186522, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -0.80 ), -1.479329379, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -0.70 ), -1.336472237, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -0.60 ), -1.193615094, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -0.50 ), -1.050757951, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -0.40 ), -0.907900808, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -0.30 ), -0.765043665, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -0.20 ), -0.622186522, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -0.10 ), -0.479329380, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.00 ), -0.336472237, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.10 ), -0.479329380, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.20 ), -0.622186522, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.30 ), -0.765043665, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.40 ), -0.907900808, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.50 ), -1.050757951, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.60 ), -1.193615094, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.70 ), -1.336472237, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.80 ), -1.479329379, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.90 ), -1.622186522, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 1.00 ), -1.765043665, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 1.10 ), -1.907900808, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 1.20 ), -2.050757951, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 1.30 ), -2.193615094, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 1.40 ), -2.336472237, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 1.50 ), -2.479329380, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 1.60 ), -2.622186522, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 1.70 ), -2.765043665, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 1.80 ), -2.907900808, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 1.90 ), -3.050757951, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.00 ), -3.193615094, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.10 ), -3.336472237, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.20 ), -3.479329380, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.30 ), -3.622186522, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.40 ), -3.765043665, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.50 ), -3.907900808, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.60 ), -4.050757951, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.70 ), -4.193615094, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.80 ), -4.336472236, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.90 ), -4.479329380, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.00 ), -4.622186522, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.10 ), -4.765043665, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.20 ), -4.907900808, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.30 ), -5.050757951, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.40 ), -5.193615094, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.50 ), -5.336472237, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.60 ), -5.479329380, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.70 ), -5.622186522, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.80 ), -5.765043665, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.90 ), -5.907900808, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.00 ), -6.050757951, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.10 ), -6.193615094, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.20 ), -6.336472237, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.30 ), -6.479329380, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.40 ), -6.622186523, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.50 ), -6.765043665, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.60 ), -6.907900808, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.70 ), -7.050757951, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.80 ), -7.193615094, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.90 ), -7.336472237, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 5.00 ), -7.479329380, PDENSITY_ERROR_TOLERANCE);

  // Testing extreme case for log density
  ASSERT_NO_THROW(d->getLogDensityGradient(-10.0));
  ASSERT_NO_THROW(d->getLogDensityGradient(+10.0));

 // Checking random numbers are within the expected range
  for (size_t i = 0; i < 100; i++) ASSERT_NO_THROW(d->getRandomNumber());

  // Testing extreme for log density gradient and hessian
  ASSERT_NO_THROW(d->getLogDensityGradient( 0.5 ));
  ASSERT_NO_THROW(d->getLogDensityHessian( 0.5 ));
 }

 TEST(Distrtibutions, LogNormalDistribution)
 {
  knlohmann::json distributionJs;
  Experiment e;
  distribution::univariate::LogNormal* d;

  // Creating distribution with an incorrect name
  distributionJs["Type"] = "Distribution/Univariate/LogNormal";
  ASSERT_ANY_THROW(d = dynamic_cast<korali::distribution::univariate::LogNormal *>(Module::getModule(distributionJs, &e)));

  // Creating distribution correctly now
  distributionJs["Type"] = "Univariate/LogNormal";
  ASSERT_NO_THROW(d = dynamic_cast<korali::distribution::univariate::LogNormal *>(Module::getModule(distributionJs, &e)));

  //////////////////////////////////////////////////

  // Getting module defaults
  distributionJs["Name"] = "Test";
  ASSERT_NO_THROW(d->applyModuleDefaults(distributionJs));
  ASSERT_NO_THROW(d->applyVariableDefaults());
  auto baseJs = distributionJs;

  // Testing conditional variables
  distributionJs = baseJs;
  distributionJs["Mu"] = "Conditional 1";
  distributionJs["Sigma"] = "Conditional 2";
  ASSERT_NO_THROW(d->setConfiguration(distributionJs));

  // Testing get configuration method
  d->getConfiguration(distributionJs);
  ASSERT_EQ(distributionJs["Type"].get<std::string>(), "univariate/logNormal");
  ASSERT_EQ(distributionJs["Mu"].get<std::string>(), "Conditional 1");
  ASSERT_EQ(distributionJs["Sigma"].get<std::string>(), "Conditional 2");

  // Testing correct shape
  distributionJs["Mu"] = 0.0;
  distributionJs["Sigma"] = 1.0;
  ASSERT_NO_THROW(d->setConfiguration(distributionJs));
  ASSERT_NO_THROW(d->updateDistribution());

  // Testing get configuration method
  d->getConfiguration(distributionJs);
  ASSERT_EQ(distributionJs["Type"].get<std::string>(), "univariate/logNormal");
  ASSERT_EQ(distributionJs["Mu"].get<double>(), 0.0);
  ASSERT_EQ(distributionJs["Sigma"].get<double>(), 1.0);

  // Testing get property pointer
  ASSERT_TRUE(d->getPropertyPointer("Mu") != NULL);
  ASSERT_TRUE(d->getPropertyPointer("Sigma") != NULL);
  ASSERT_ANY_THROW(d->getPropertyPointer("Undefined")); // Distribution not recognized

  // Testing incorrect width
  d->_sigma = -0.5;
  ASSERT_ANY_THROW(d->updateDistribution());

  // Testing correct width
  d->_sigma = 1.0;
  ASSERT_NO_THROW(d->updateDistribution());

  /////////////////////////////////////////////////

  // Distributions generated with https://keisan.casio.com/exec/system/1180573226

  // Testing expected density
  EXPECT_NEAR(d->getDensity( 0.00 ), 0.000000000, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.10 ), 0.281590189, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.20 ), 0.546267871, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.30 ), 0.644203257, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.40 ), 0.655444168, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.50 ), 0.627496077, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.60 ), 0.583573823, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.70 ), 0.534794832, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.80 ), 0.486415781, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.90 ), 0.440815686, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.00 ), 0.398942280, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.10 ), 0.361031261, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.20 ), 0.326972024, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.30 ), 0.296496371, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.40 ), 0.269276229, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.50 ), 0.244973652, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.60 ), 0.223265447, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.70 ), 0.203854260, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.80 ), 0.186472449, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.90 ), 0.170882238, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.00 ), 0.156874019, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.10 ), 0.144263846, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.20 ), 0.132890686, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.30 ), 0.122613707, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.40 ), 0.113309754, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.50 ), 0.104871067, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.60 ), 0.097203259, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.70 ), 0.090223546, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.80 ), 0.083859205, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.90 ), 0.078046245, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.00 ), 0.072728256, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.10 ), 0.067855420, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.20 ), 0.063383656, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.30 ), 0.059273887, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.40 ), 0.055491406, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.50 ), 0.052005332, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.60 ), 0.048788135, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.70 ), 0.045815229, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.80 ), 0.043064619, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.90 ), 0.040516593, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.00 ), 0.038153457, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.10 ), 0.035959297, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.20 ), 0.033919783, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.30 ), 0.032021989, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.40 ), 0.030254236, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.50 ), 0.028605956, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.60 ), 0.027067575, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.70 ), 0.025630406, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.80 ), 0.024286554, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.90 ), 0.023028838, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 5.00 ), 0.021850715, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 5.10 ), 0.020746215, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 5.20 ), 0.019709889, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 5.30 ), 0.018736751, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 5.40 ), 0.017822239, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 5.50 ), 0.016962171, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 5.60 ), 0.016152709, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 5.70 ), 0.015390329, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 5.80 ), 0.014671790, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 5.90 ), 0.013994109, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 6.00 ), 0.013354538, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 6.10 ), 0.012750543, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 6.20 ), 0.012179782, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 6.30 ), 0.011640095, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 6.40 ), 0.011129482, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 6.50 ), 0.010646092, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 6.60 ), 0.010188210, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 6.70 ), 0.009754246, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 6.80 ), 0.009342725, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 6.90 ), 0.008952276, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 7.00 ), 0.008581626, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 7.10 ), 0.008229590, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 7.20 ), 0.007895064, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 7.30 ), 0.007577022, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 7.40 ), 0.007274504, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 7.50 ), 0.006986618, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 7.60 ), 0.006712529, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 7.70 ), 0.006451458, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 7.80 ), 0.006202676, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 7.90 ), 0.005965502, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 8.00 ), 0.005739296, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 8.10 ), 0.005523463, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 8.20 ), 0.005317442, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 8.30 ), 0.005120707, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 8.40 ), 0.004932766, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 8.50 ), 0.004753157, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 8.60 ), 0.004581444, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 8.70 ), 0.004417220, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 8.80 ), 0.004260099, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 8.90 ), 0.004109722, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 9.00 ), 0.003965747, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 9.10 ), 0.003827854, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 9.20 ), 0.003695741, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 9.30 ), 0.003569124, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 9.40 ), 0.003447734, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 9.50 ), 0.003331316, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 9.60 ), 0.003219633, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 9.70 ), 0.003112458, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 9.80 ), 0.003009577, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 9.90 ), 0.002910789, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 10.00 ), 0.002815902, PDENSITY_ERROR_TOLERANCE);

  // Testing log density function
  EXPECT_NEAR(d->getLogDensity( 0.10 ), -1.267302496, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.20 ), -0.604645818, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.30 ), -0.439740986, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.40 ), -0.422442154, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.50 ), -0.466017860, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.60 ), -0.538584318, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.70 ), -0.625872097, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.80 ), -0.720691504, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.90 ), -0.819128437, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 1.00 ), -0.918938533, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 1.10 ), -1.018790728, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 1.20 ), -1.117880665, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 1.30 ), -1.215720301, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 1.40 ), -1.312017553, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 1.50 ), -1.406604618, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 1.60 ), -1.499393868, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 1.70 ), -1.590349955, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 1.80 ), -1.679471779, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 1.90 ), -1.766780625, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.00 ), -1.852312221, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.10 ), -1.936111389, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.20 ), -2.018228398, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.30 ), -2.098716460, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.40 ), -2.177630026, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.50 ), -2.255023618, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.60 ), -2.330951039, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.70 ), -2.405464849, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.80 ), -2.478616022, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.90 ), -2.550453747, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.00 ), -2.621025302, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.10 ), -2.690376014, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.20 ), -2.758549246, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.30 ), -2.825586432, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.40 ), -2.891527118, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.50 ), -2.956409029, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.60 ), -3.020268137, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.70 ), -3.083138736, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.80 ), -3.145053524, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.90 ), -3.206043675, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.00 ), -3.266138922, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.10 ), -3.325367627, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.20 ), -3.383756856, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.30 ), -3.441332448, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.40 ), -3.498119082, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.50 ), -3.554140338, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.60 ), -3.609418757, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.70 ), -3.663975901, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.80 ), -3.717832400, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.90 ), -3.771008007, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 5.00 ), -3.823521643, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 5.10 ), -3.875391441, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 5.20 ), -3.926634791, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 5.30 ), -3.977268374, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 5.40 ), -4.027308202, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 5.50 ), -4.076769654, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 5.60 ), -4.125667506, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 5.70 ), -4.174015961, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 5.80 ), -4.221828680, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 5.90 ), -4.269118808, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 6.00 ), -4.315899000, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 6.10 ), -4.362181444, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 6.20 ), -4.407977885, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 6.30 ), -4.453299643, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 6.40 ), -4.498157638, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 6.50 ), -4.542562404, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 6.60 ), -4.586524113, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 6.70 ), -4.630052581, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 6.80 ), -4.673157296, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 6.90 ), -4.715847427, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 7.00 ), -4.758131836, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 7.10 ), -4.800019099, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 7.20 ), -4.841517508, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 7.30 ), -4.882635093, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 7.40 ), -4.923379629, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 7.50 ), -4.963758645, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 7.60 ), -5.003779437, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 7.70 ), -5.043449078, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 7.80 ), -5.082774424, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 7.90 ), -5.121762126, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 8.00 ), -5.160418637, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 8.10 ), -5.198750221, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 8.20 ), -5.236762957, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 8.30 ), -5.274462750, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 8.40 ), -5.311855336, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 8.50 ), -5.348946289, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 8.60 ), -5.385741026, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 8.70 ), -5.422244816, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 8.80 ), -5.458462780, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 8.90 ), -5.494399902, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 9.00 ), -5.530061032, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 9.10 ), -5.565450889, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 9.20 ), -5.600574069, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 9.30 ), -5.635435046, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 9.40 ), -5.670038178, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 9.50 ), -5.704387713, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 9.60 ), -5.738487789, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 9.70 ), -5.772342439, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 9.80 ), -5.805955596, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 9.90 ), -5.839331097, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 10.00 ), -5.872472681, PDENSITY_ERROR_TOLERANCE);

  // Testing expected density
  EXPECT_EQ(d->getDensity( 0.0 ), 0.0);

  // Testing log density function
  EXPECT_EQ(d->getLogDensity( 0.0 ), -INFINITY);

 // Checking random numbers are within the expected range
  for (size_t i = 0; i < 100; i++)
  {
   double y = d->getRandomNumber();
   EXPECT_TRUE(y >= 0.0);
  }

  // Testing extreme for log density gradient and hessian
  EXPECT_EQ(d->getLogDensityGradient( 0.0 ), 0.0);
  EXPECT_EQ(d->getLogDensityHessian( 0.0 ), 0.0);

  // Normal case for log density gradient and hessian
  ASSERT_NO_THROW(d->getLogDensityGradient( 0.5 ));
  ASSERT_NO_THROW(d->getLogDensityHessian( 0.5 ));
 }

 TEST(Distrtibutions, NormalDistribution)
 {
  knlohmann::json distributionJs;
  Experiment e;
  distribution::univariate::Normal* d;

  // Creating distribution with an incorrect name
  distributionJs["Type"] = "Distribution/Univariate/Normal";
  ASSERT_ANY_THROW(d = dynamic_cast<korali::distribution::univariate::Normal *>(Module::getModule(distributionJs, &e)));

  // Creating distribution correctly now
  distributionJs["Type"] = "Univariate/Normal";
  ASSERT_NO_THROW(d = dynamic_cast<korali::distribution::univariate::Normal *>(Module::getModule(distributionJs, &e)));

  //////////////////////////////////////////////////

  // Getting module defaults
  distributionJs["Name"] = "Test";
  ASSERT_NO_THROW(d->applyModuleDefaults(distributionJs));
  ASSERT_NO_THROW(d->applyVariableDefaults());
  auto baseJs = distributionJs;

  // Testing conditional variables
  distributionJs = baseJs;
  distributionJs["Mean"] = "Conditional 1";
  distributionJs["Standard Deviation"] = "Conditional 2";
  ASSERT_NO_THROW(d->setConfiguration(distributionJs));

  // Testing get configuration method
  d->getConfiguration(distributionJs);
  ASSERT_EQ(distributionJs["Type"].get<std::string>(), "univariate/normal");
  ASSERT_EQ(distributionJs["Mean"].get<std::string>(), "Conditional 1");
  ASSERT_EQ(distributionJs["Standard Deviation"].get<std::string>(), "Conditional 2");

  // Testing correct shape
  distributionJs["Mean"] = 0.0;
  distributionJs["Standard Deviation"] = 1.0;
  ASSERT_NO_THROW(d->setConfiguration(distributionJs));
  ASSERT_NO_THROW(d->updateDistribution());

  // Testing get configuration method
  d->getConfiguration(distributionJs);
  ASSERT_EQ(distributionJs["Type"].get<std::string>(), "univariate/normal");
  ASSERT_EQ(distributionJs["Mean"].get<double>(), 0.0);
  ASSERT_EQ(distributionJs["Standard Deviation"].get<double>(), 1.0);

  // Testing get property pointer
  ASSERT_TRUE(d->getPropertyPointer("Mean") != NULL);
  ASSERT_TRUE(d->getPropertyPointer("Standard Deviation") != NULL);
  ASSERT_ANY_THROW(d->getPropertyPointer("Undefined")); // Distribution not recognized

  // Testing incorrect sigma
  d->_standardDeviation = -0.5;
  ASSERT_ANY_THROW(d->updateDistribution());

  // Testing correct sigma
  d->_standardDeviation = 1.0;
  ASSERT_NO_THROW(d->updateDistribution());

  /////////////////////////////////////////////////

  // Distributions generated with https://keisan.casio.com/exec/system/1180573226

  // Testing expected density
  EXPECT_NEAR(d->getDensity( -5.00 ), 0.000001487 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -4.90 ), 0.000002439 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -4.80 ), 0.000003961 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -4.70 ), 0.000006370 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -4.60 ), 0.000010141 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -4.50 ), 0.000015984 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -4.40 ), 0.000024942 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -4.30 ), 0.000038535 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -4.20 ), 0.000058943 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -4.10 ), 0.000089262 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -4.00 ), 0.000133830 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -3.90 ), 0.000198655 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -3.80 ), 0.000291947 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -3.70 ), 0.000424780 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -3.60 ), 0.000611902 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -3.50 ), 0.000872683 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -3.40 ), 0.001232219 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -3.30 ), 0.001722569 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -3.20 ), 0.002384088 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -3.10 ), 0.003266819 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -3.00 ), 0.004431848 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -2.90 ), 0.005952532 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -2.80 ), 0.007915452 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -2.70 ), 0.010420935 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -2.60 ), 0.013582969 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -2.50 ), 0.017528300 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -2.40 ), 0.022394530 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -2.30 ), 0.028327038 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -2.20 ), 0.035474593 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -2.10 ), 0.043983596 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -2.00 ), 0.053990967 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -1.90 ), 0.065615815 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -1.80 ), 0.078950158 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -1.70 ), 0.094049077 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -1.60 ), 0.110920835 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -1.50 ), 0.129517596 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -1.40 ), 0.149727466 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -1.30 ), 0.171368592 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -1.20 ), 0.194186055 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -1.10 ), 0.217852177 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -1.00 ), 0.241970725 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -0.90 ), 0.266085250 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -0.80 ), 0.289691553 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -0.70 ), 0.312253933 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -0.60 ), 0.333224603 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -0.50 ), 0.352065327 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -0.40 ), 0.368270140 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -0.30 ), 0.381387816 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -0.20 ), 0.391042694 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( -0.10 ), 0.396952548 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.00 ), 0.398942280 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.10 ), 0.396952548 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.20 ), 0.391042694 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.30 ), 0.381387816 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.40 ), 0.368270140 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.50 ), 0.352065327 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.60 ), 0.333224603 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.70 ), 0.312253933 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.80 ), 0.289691553 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.90 ), 0.266085250 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.00 ), 0.241970725 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.10 ), 0.217852177 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.20 ), 0.194186055 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.30 ), 0.171368592 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.40 ), 0.149727466 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.50 ), 0.129517596 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.60 ), 0.110920835 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.70 ), 0.094049077 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.80 ), 0.078950158 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.90 ), 0.065615815 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.00 ), 0.053990967 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.10 ), 0.043983596 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.20 ), 0.035474593 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.30 ), 0.028327038 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.40 ), 0.022394530 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.50 ), 0.017528300 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.60 ), 0.013582969 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.70 ), 0.010420935 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.80 ), 0.007915452 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.90 ), 0.005952532 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.00 ), 0.004431848 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.10 ), 0.003266819 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.20 ), 0.002384088 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.30 ), 0.001722569 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.40 ), 0.001232219 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.50 ), 0.000872683 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.60 ), 0.000611902 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.70 ), 0.000424780 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.80 ), 0.000291947 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.90 ), 0.000198655 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.00 ), 0.000133830 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.10 ), 0.000089262 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.20 ), 0.000058943 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.30 ), 0.000038535 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.40 ), 0.000024942 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.50 ), 0.000015984 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.60 ), 0.000010141 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.70 ), 0.000006370 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.80 ), 0.000003961 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.90 ), 0.000002439 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 5.00 ), 0.000001487 , PDENSITY_ERROR_TOLERANCE);

  // Testing log density function
  EXPECT_NEAR(d->getLogDensity( -5.00 ), -13.418938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -4.90 ), -12.923938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -4.80 ), -12.438938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -4.70 ), -11.963938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -4.60 ), -11.498938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -4.50 ), -11.043938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -4.40 ), -10.598938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -4.30 ), -10.163938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -4.20 ), -9.738938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -4.10 ), -9.323938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -4.00 ), -8.918938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -3.90 ), -8.523938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -3.80 ), -8.138938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -3.70 ), -7.763938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -3.60 ), -7.398938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -3.50 ), -7.043938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -3.40 ), -6.698938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -3.30 ), -6.363938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -3.20 ), -6.038938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -3.10 ), -5.723938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -3.00 ), -5.418938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -2.90 ), -5.123938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -2.80 ), -4.838938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -2.70 ), -4.563938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -2.60 ), -4.298938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -2.50 ), -4.043938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -2.40 ), -3.798938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -2.30 ), -3.563938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -2.20 ), -3.338938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -2.10 ), -3.123938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -2.00 ), -2.918938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -1.90 ), -2.723938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -1.80 ), -2.538938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -1.70 ), -2.363938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -1.60 ), -2.198938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -1.50 ), -2.043938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -1.40 ), -1.898938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -1.30 ), -1.763938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -1.20 ), -1.638938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -1.10 ), -1.523938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -1.00 ), -1.418938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -0.90 ), -1.323938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -0.80 ), -1.238938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -0.70 ), -1.163938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -0.60 ), -1.098938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -0.50 ), -1.043938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -0.40 ), -0.998938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -0.30 ), -0.963938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -0.20 ), -0.938938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( -0.10 ), -0.923938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.00 ), -0.918938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.10 ), -0.923938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.20 ), -0.938938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.30 ), -0.963938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.40 ), -0.998938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.50 ), -1.043938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.60 ), -1.098938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.70 ), -1.163938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.80 ), -1.238938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 0.90 ), -1.323938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 1.00 ), -1.418938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 1.10 ), -1.523938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 1.20 ), -1.638938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 1.30 ), -1.763938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 1.40 ), -1.898938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 1.50 ), -2.043938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 1.60 ), -2.198938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 1.70 ), -2.363938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 1.80 ), -2.538938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 1.90 ), -2.723938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.00 ), -2.918938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.10 ), -3.123938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.20 ), -3.338938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.30 ), -3.563938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.40 ), -3.798938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.50 ), -4.043938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.60 ), -4.298938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.70 ), -4.563938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.80 ), -4.838938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.90 ), -5.123938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.00 ), -5.418938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.10 ), -5.723938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.20 ), -6.038938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.30 ), -6.363938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.40 ), -6.698938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.50 ), -7.043938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.60 ), -7.398938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.70 ), -7.763938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.80 ), -8.138938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.90 ), -8.523938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.00 ), -8.918938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.10 ), -9.323938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.20 ), -9.738938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.30 ), -10.163938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.40 ), -10.598938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.50 ), -11.043938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.60 ), -11.498938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.70 ), -11.963938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.80 ), -12.438938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.90 ), -12.923938533 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 5.00 ), -13.418938533 , PDENSITY_ERROR_TOLERANCE);

 // Checking random numbers are within the expected range
  for (size_t i = 0; i < 100; i++) ASSERT_NO_THROW(d->getRandomNumber());

  // Normal case for log density gradient and hessian
  ASSERT_NO_THROW(d->getLogDensityGradient( 0.5 ));
  ASSERT_NO_THROW(d->getLogDensityHessian( 0.5 ));
 }

 TEST(Distrtibutions, TruncatedNormalDistribution)
 {
  knlohmann::json distributionJs;
  Experiment e;
  distribution::univariate::TruncatedNormal* d;

  // Creating distribution with an incorrect name
  distributionJs["Type"] = "Distribution/Univariate/TruncatedNormal";
  ASSERT_ANY_THROW(d = dynamic_cast<korali::distribution::univariate::TruncatedNormal *>(Module::getModule(distributionJs, &e)));

  // Creating distribution correctly now
  distributionJs["Type"] = "Univariate/TruncatedNormal";
  ASSERT_NO_THROW(d = dynamic_cast<korali::distribution::univariate::TruncatedNormal *>(Module::getModule(distributionJs, &e)));

  //////////////////////////////////////////////////

  // Getting module defaults
  distributionJs["Name"] = "Test";
  ASSERT_NO_THROW(d->applyModuleDefaults(distributionJs));
  ASSERT_NO_THROW(d->applyVariableDefaults());
  auto baseJs = distributionJs;

  // Testing conditional variables
  distributionJs = baseJs;
  distributionJs["Mu"] = "Conditional 1";
  distributionJs["Sigma"] = "Conditional 2";
  distributionJs["Minimum"] = "Conditional 3";
  distributionJs["Maximum"] = "Conditional 4";
  ASSERT_NO_THROW(d->setConfiguration(distributionJs));

  // Testing get configuration method
  d->getConfiguration(distributionJs);
  ASSERT_EQ(distributionJs["Type"].get<std::string>(), "univariate/truncatedNormal");
  ASSERT_EQ(distributionJs["Mu"].get<std::string>(), "Conditional 1");
  ASSERT_EQ(distributionJs["Sigma"].get<std::string>(), "Conditional 2");
  ASSERT_EQ(distributionJs["Minimum"].get<std::string>(), "Conditional 3");
  ASSERT_EQ(distributionJs["Maximum"].get<std::string>(), "Conditional 4");

  // Testing correct shape
  distributionJs["Mu"] = 100.0;
  distributionJs["Sigma"] = 25.0;
  distributionJs["Minimum"] = 50.0;
  distributionJs["Maximum"] = 150.0;
  ASSERT_NO_THROW(d->setConfiguration(distributionJs));
  ASSERT_NO_THROW(d->updateDistribution());

  // Testing get configuration method
  d->getConfiguration(distributionJs);
  ASSERT_EQ(distributionJs["Type"].get<std::string>(), "univariate/truncatedNormal");
  ASSERT_EQ(distributionJs["Mu"].get<double>(), 100.0);
  ASSERT_EQ(distributionJs["Sigma"].get<double>(), 25.0);
  ASSERT_EQ(distributionJs["Minimum"].get<double>(), 50.0);
  ASSERT_EQ(distributionJs["Maximum"].get<double>(), 150.0);

  // Testing get property pointer
  ASSERT_TRUE(d->getPropertyPointer("Mu") != NULL);
  ASSERT_TRUE(d->getPropertyPointer("Sigma") != NULL);
  ASSERT_TRUE(d->getPropertyPointer("Minimum") != NULL);
  ASSERT_TRUE(d->getPropertyPointer("Maximum") != NULL);
  ASSERT_ANY_THROW(d->getPropertyPointer("Undefined")); // Distribution not recognized

  // Testing incorrect sigma
  d->_sigma = -25.0;
  ASSERT_ANY_THROW(d->updateDistribution());

  // Testing correct sigma
  d->_sigma = 25.0;
  ASSERT_NO_THROW(d->updateDistribution());

  // Testing incorrect minimum/maximum
  d->_minimum = 300.0;
  ASSERT_ANY_THROW(d->updateDistribution());

  // Testing correct minimum/maximum
  d->_minimum = 50.0;
  ASSERT_NO_THROW(d->updateDistribution());

  /////////////////////////////////////////////////

  // Distributions generated with https://keisan.casio.com/exec/system/1180573226

  // Testing expected density
  // To-do: Add more values and improve precision
  EXPECT_NEAR(d->getDensity( 81.63 ), 0.0127629 , 0.001);
  EXPECT_NEAR(d->getDensity( 137.962 ), 0.00527826 , 0.001);
  EXPECT_NEAR(d->getDensity( 122.367 ), 0.0112043 , 0.001);
  EXPECT_NEAR(d->getDensity( 103.704 ), 0.0165359 , 0.001);
  EXPECT_NEAR(d->getDensity( 94.899 ), 0.016374 , 0.001);
  EXPECT_NEAR(d->getDensity( 65.8326 ), 0.00657044 , 0.001);
  EXPECT_NEAR(d->getDensity( 84.5743 ), 0.0138204 , 0.001);
  EXPECT_NEAR(d->getDensity( 71.5672 ), 0.00875626 , 0.001);
  EXPECT_NEAR(d->getDensity( 62.0654 ), 0.00528716 , 0.001);
  EXPECT_NEAR(d->getDensity( 108.155 ), 0.0158521 , 0.001);

  // Testing log density function
  EXPECT_NEAR(d->getLogDensity( 81.63 ), -4.361212754 , 0.001);
  EXPECT_NEAR(d->getLogDensity( 137.96 ), -5.244158781 , 0.001);
  EXPECT_NEAR(d->getLogDensity( 122.37 ), -4.491457646 , 0.001);
  EXPECT_NEAR(d->getLogDensity( 103.70 ), -4.102221504 , 0.001);
  EXPECT_NEAR(d->getLogDensity( 94.90 ), -4.112060568 , 0.001);
  EXPECT_NEAR(d->getLogDensity( 65.83 ), -5.025174478 , 0.001);
  EXPECT_NEAR(d->getLogDensity( 84.57 ), -4.281609518 , 0.001);
  EXPECT_NEAR(d->getLogDensity( 71.57 ), -4.737986406 , 0.001);
  EXPECT_NEAR(d->getLogDensity( 62.07 ), -5.242474039 , 0.001);
  EXPECT_NEAR(d->getLogDensity( 108.16 ), -4.144453295 , 0.001);

  // Checking random numbers are within the expected range
  for (size_t i = 0; i < 100; i++)
  {
   double y = d->getRandomNumber();
   EXPECT_TRUE((y >= d->_minimum) && (y <= d->_maximum));
  }

  // Normal case for log density gradient and hessian
  ASSERT_NO_THROW(d->getLogDensityGradient( 0.5 ));
  ASSERT_NO_THROW(d->getLogDensityHessian( 0.5 ));
 }
 
 TEST(Distrtibutions, PoissonDistribution)
 {
  knlohmann::json distributionJs;
  Experiment e;
  distribution::univariate::Poisson* d;

  // Creating distribution with an incorrect name
  distributionJs["Type"] = "Distribution/Univariate/Poisson";
  ASSERT_ANY_THROW(d = dynamic_cast<korali::distribution::univariate::Poisson*>(Module::getModule(distributionJs, &e)));

  // Creating distribution correctly now
  distributionJs["Type"] = "Univariate/Poisson";
  ASSERT_NO_THROW(d = dynamic_cast<korali::distribution::univariate::Poisson*>(Module::getModule(distributionJs, &e)));

  //////////////////////////////////////////////////

  // Getting module defaults
  distributionJs["Name"] = "Test";
  ASSERT_NO_THROW(d->applyModuleDefaults(distributionJs));
  ASSERT_NO_THROW(d->applyVariableDefaults());
  auto baseJs = distributionJs;

  // Testing incorrect mean
  d->_mean = -16.0;
  ASSERT_ANY_THROW(d->updateDistribution());
  
  // Testing meaningful mean
  distributionJs["Mean"] = 2.0;
  ASSERT_NO_THROW(d->setConfiguration(distributionJs));
  ASSERT_NO_THROW(d->updateDistribution());

  // Testing get property pointer
  ASSERT_TRUE(d->getPropertyPointer("Mean") != NULL);
  
  // Testing get configuration method
  d->getConfiguration(distributionJs);
  ASSERT_EQ(distributionJs["Type"].get<std::string>(), "univariate/poisson");
  ASSERT_EQ(distributionJs["Mean"].get<double>(), 2.0);

  /////////////////////////////////////////////////

  // Testing expected density
  EXPECT_EQ(d->getDensity( -1.0 ), 0.);
  EXPECT_NEAR(d->getDensity( 0.0 ), 0.1353352832, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.0 ), 0.2706705664, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.0 ), 0.2706705664, PDENSITY_ERROR_TOLERANCE);

  // Testing expected log density
  EXPECT_EQ(d->getLogDensity( -1.0 ), -INFINITY);
  EXPECT_NEAR(d->getLogDensity( 0.0 ), -2.0, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 1.0 ), -1.3068528197, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.0 ), -1.3068528197, PDENSITY_ERROR_TOLERANCE);

  // Checking random numbers are within the expected range
  for (size_t i = 0; i < 100; i++)
  {
   double y = d->getRandomNumber();
   EXPECT_TRUE((y >= 0.) && (y <= 1e6));
  }

  // Check that gradient and Hessian not exist
  ASSERT_ANY_THROW(d->getLogDensityGradient( 1.0 ));
  ASSERT_ANY_THROW(d->getLogDensityHessian( 1.0 ));
 }


 TEST(Distrtibutions, UniformDistribution)
 {
  knlohmann::json distributionJs;
  Experiment e;
  distribution::univariate::Uniform* d;

  // Creating distribution with an incorrect name
  distributionJs["Type"] = "Distribution/Univariate/Uniform";
  ASSERT_ANY_THROW(d = dynamic_cast<korali::distribution::univariate::Uniform *>(Module::getModule(distributionJs, &e)));

  // Creating distribution correctly now
  distributionJs["Type"] = "Univariate/Uniform";
  ASSERT_NO_THROW(d = dynamic_cast<korali::distribution::univariate::Uniform *>(Module::getModule(distributionJs, &e)));

  //////////////////////////////////////////////////

  // Getting module defaults
  distributionJs["Name"] = "Test";
  ASSERT_NO_THROW(d->applyModuleDefaults(distributionJs));
  ASSERT_NO_THROW(d->applyVariableDefaults());
  auto baseJs = distributionJs;

  // Testing conditional variables
  distributionJs = baseJs;
  distributionJs["Minimum"] = "Conditional 1";
  distributionJs["Maximum"] = "Conditional 2";
  ASSERT_NO_THROW(d->setConfiguration(distributionJs));

  // Testing get configuration method
  d->getConfiguration(distributionJs);
  ASSERT_EQ(distributionJs["Type"].get<std::string>(), "univariate/uniform");
  ASSERT_EQ(distributionJs["Minimum"].get<std::string>(), "Conditional 1");
  ASSERT_EQ(distributionJs["Maximum"].get<std::string>(), "Conditional 2");

  // Testing correct shape
  distributionJs["Minimum"] = 2.0;
  distributionJs["Maximum"] = 8.0;
  ASSERT_NO_THROW(d->setConfiguration(distributionJs));
  ASSERT_NO_THROW(d->updateDistribution());

  // Testing get configuration method
  d->getConfiguration(distributionJs);
  ASSERT_EQ(distributionJs["Type"].get<std::string>(), "univariate/uniform");
  ASSERT_EQ(distributionJs["Minimum"].get<double>(), 2.0);
  ASSERT_EQ(distributionJs["Maximum"].get<double>(), 8.0);

  // Testing get property pointer
  ASSERT_TRUE(d->getPropertyPointer("Minimum") != NULL);
  ASSERT_TRUE(d->getPropertyPointer("Maximum") != NULL);
  ASSERT_ANY_THROW(d->getPropertyPointer("Undefined")); // Distribution not recognized

  // Testing incorrect minimum/maximum
  d->_minimum = 16.0;
  ASSERT_NO_THROW(d->updateDistribution());
  ASSERT_ANY_THROW(d->getRandomNumber());

  // Testing correct minimum/maximum
  d->_minimum = 2.0;
  ASSERT_NO_THROW(d->updateDistribution());
  ASSERT_NO_THROW(d->getRandomNumber());

  /////////////////////////////////////////////////

  // Distributions generated with https://keisan.casio.com/exec/system/1180573226

  // Testing expected density
  EXPECT_NEAR(d->getDensity( 0 ), 0 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.1 ), 0 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.2 ), 0 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.3 ), 0 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.4 ), 0 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.5 ), 0 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.6 ), 0 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.7 ), 0 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.8 ), 0 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 0.9 ), 0 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.00 ), 0.000000000 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.10 ), 0.000000000 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.20 ), 0.000000000 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.30 ), 0.000000000 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.40 ), 0.000000000 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.50 ), 0.000000000 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.60 ), 0.000000000 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.70 ), 0.000000000 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.80 ), 0.000000000 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1.90 ), 0.000000000 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.00000001 ), 0.166666667 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.10 ), 0.166666667 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.20 ), 0.166666667 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.30 ), 0.166666667 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.40 ), 0.166666667 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.50 ), 0.166666667 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.60 ), 0.166666667 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.70 ), 0.166666667 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.80 ), 0.166666667 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2.90 ), 0.166666667 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.00 ), 0.166666667 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.10 ), 0.166666667 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.20 ), 0.166666667 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.30 ), 0.166666667 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.40 ), 0.166666667 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.50 ), 0.166666667 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.60 ), 0.166666667 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.70 ), 0.166666667 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.80 ), 0.166666667 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 3.90 ), 0.166666667 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.00 ), 0.166666667 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.10 ), 0.166666667 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.20 ), 0.166666667 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.30 ), 0.166666667 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.40 ), 0.166666667 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.50 ), 0.166666667 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.60 ), 0.166666667 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.70 ), 0.166666667 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.80 ), 0.166666667 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4.90 ), 0.166666667 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 5.00 ), 0.166666667 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 5.10 ), 0.166666667 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 5.20 ), 0.166666667 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 5.30 ), 0.166666667 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 5.40 ), 0.166666667 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 5.50 ), 0.166666667 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 5.60 ), 0.166666667 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 5.70 ), 0.166666667 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 5.80 ), 0.166666667 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 5.90 ), 0.166666667 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 6.00 ), 0.166666667 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 6.10 ), 0.166666667 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 6.20 ), 0.166666667 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 6.30 ), 0.166666667 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 6.40 ), 0.166666667 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 6.50 ), 0.166666667 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 6.60 ), 0.166666667 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 6.70 ), 0.166666667 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 6.80 ), 0.166666667 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 6.90 ), 0.166666667 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 7.00 ), 0.166666667 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 7.10 ), 0.166666667 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 7.20 ), 0.166666667 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 7.30 ), 0.166666667 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 7.40 ), 0.166666667 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 7.50 ), 0.166666667 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 7.60 ), 0.166666667 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 7.70 ), 0.166666667 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 7.80 ), 0.166666667 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 7.90 ), 0.166666667 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 8.0000000001 ), 0.000000000 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 8.10 ), 0.000000000 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 8.20 ), 0.000000000 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 8.30 ), 0.000000000 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 8.40 ), 0.000000000 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 8.50 ), 0.000000000 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 8.60 ), 0.000000000 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 8.70 ), 0.000000000 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 8.80 ), 0.000000000 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 8.90 ), 0.000000000 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 9.00 ), 0.000000000 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 9.10 ), 0.000000000 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 9.20 ), 0.000000000 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 9.30 ), 0.000000000 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 9.40 ), 0.000000000 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 9.50 ), 0.000000000 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 9.60 ), 0.000000000 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 9.70 ), 0.000000000 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 9.80 ), 0.000000000 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 9.90 ), 0.000000000 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 10.00 ), 0.000000000 , PDENSITY_ERROR_TOLERANCE);

  // Testing expected log density

  EXPECT_EQ(d->getLogDensity( 0.00 ), -INFINITY);
  EXPECT_EQ(d->getLogDensity( 0.10 ), -INFINITY);
  EXPECT_EQ(d->getLogDensity( 0.20 ), -INFINITY);
  EXPECT_EQ(d->getLogDensity( 0.30 ), -INFINITY);
  EXPECT_EQ(d->getLogDensity( 0.40 ), -INFINITY);
  EXPECT_EQ(d->getLogDensity( 0.50 ), -INFINITY);
  EXPECT_EQ(d->getLogDensity( 0.60 ), -INFINITY);
  EXPECT_EQ(d->getLogDensity( 0.70 ), -INFINITY);
  EXPECT_EQ(d->getLogDensity( 0.80 ), -INFINITY);
  EXPECT_EQ(d->getLogDensity( 0.90 ), -INFINITY);
  EXPECT_EQ(d->getLogDensity( 1.00 ), -INFINITY);
  EXPECT_EQ(d->getLogDensity( 1.10 ), -INFINITY);
  EXPECT_EQ(d->getLogDensity( 1.20 ), -INFINITY);
  EXPECT_EQ(d->getLogDensity( 1.30 ), -INFINITY);
  EXPECT_EQ(d->getLogDensity( 1.40 ), -INFINITY);
  EXPECT_EQ(d->getLogDensity( 1.50 ), -INFINITY);
  EXPECT_EQ(d->getLogDensity( 1.60 ), -INFINITY);
  EXPECT_EQ(d->getLogDensity( 1.70 ), -INFINITY);
  EXPECT_EQ(d->getLogDensity( 1.80 ), -INFINITY);
  EXPECT_EQ(d->getLogDensity( 1.90 ), -INFINITY);
  EXPECT_NEAR(d->getLogDensity( 2.000000001 ), -1.791759469 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.10 ), -1.791759469 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.20 ), -1.791759469 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.30 ), -1.791759469 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.40 ), -1.791759469 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.50 ), -1.791759469 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.60 ), -1.791759469 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.70 ), -1.791759469 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.80 ), -1.791759469 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 2.90 ), -1.791759469 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.00 ), -1.791759469 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.10 ), -1.791759469 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.20 ), -1.791759469 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.30 ), -1.791759469 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.40 ), -1.791759469 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.50 ), -1.791759469 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.60 ), -1.791759469 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.70 ), -1.791759469 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.80 ), -1.791759469 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 3.90 ), -1.791759469 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.00 ), -1.791759469 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.10 ), -1.791759469 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.20 ), -1.791759469 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.30 ), -1.791759469 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.40 ), -1.791759469 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.50 ), -1.791759469 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.60 ), -1.791759469 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.70 ), -1.791759469 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.80 ), -1.791759469 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4.90 ), -1.791759469 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 5.00 ), -1.791759469 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 5.10 ), -1.791759469 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 5.20 ), -1.791759469 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 5.30 ), -1.791759469 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 5.40 ), -1.791759469 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 5.50 ), -1.791759469 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 5.60 ), -1.791759469 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 5.70 ), -1.791759469 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 5.80 ), -1.791759469 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 5.90 ), -1.791759469 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 6.00 ), -1.791759469 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 6.10 ), -1.791759469 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 6.20 ), -1.791759469 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 6.30 ), -1.791759469 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 6.40 ), -1.791759469 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 6.50 ), -1.791759469 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 6.60 ), -1.791759469 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 6.70 ), -1.791759469 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 6.80 ), -1.791759469 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 6.90 ), -1.791759469 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 7.00 ), -1.791759469 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 7.10 ), -1.791759469 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 7.20 ), -1.791759469 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 7.30 ), -1.791759469 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 7.40 ), -1.791759469 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 7.50 ), -1.791759469 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 7.60 ), -1.791759469 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 7.70 ), -1.791759469 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 7.80 ), -1.791759469 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 7.90 ), -1.791759469 , PDENSITY_ERROR_TOLERANCE);
  EXPECT_EQ(d->getLogDensity( 8.00000001 ), -INFINITY);
  EXPECT_EQ(d->getLogDensity( 8.10 ), -INFINITY);
  EXPECT_EQ(d->getLogDensity( 8.20 ), -INFINITY);
  EXPECT_EQ(d->getLogDensity( 8.30 ), -INFINITY);
  EXPECT_EQ(d->getLogDensity( 8.40 ), -INFINITY);
  EXPECT_EQ(d->getLogDensity( 8.50 ), -INFINITY);
  EXPECT_EQ(d->getLogDensity( 8.60 ), -INFINITY);
  EXPECT_EQ(d->getLogDensity( 8.70 ), -INFINITY);
  EXPECT_EQ(d->getLogDensity( 8.80 ), -INFINITY);
  EXPECT_EQ(d->getLogDensity( 8.90 ), -INFINITY);
  EXPECT_EQ(d->getLogDensity( 9.00 ), -INFINITY);
  EXPECT_EQ(d->getLogDensity( 9.10 ), -INFINITY);
  EXPECT_EQ(d->getLogDensity( 9.20 ), -INFINITY);
  EXPECT_EQ(d->getLogDensity( 9.30 ), -INFINITY);
  EXPECT_EQ(d->getLogDensity( 9.40 ), -INFINITY);
  EXPECT_EQ(d->getLogDensity( 9.50 ), -INFINITY);
  EXPECT_EQ(d->getLogDensity( 9.60 ), -INFINITY);
  EXPECT_EQ(d->getLogDensity( 9.70 ), -INFINITY);
  EXPECT_EQ(d->getLogDensity( 9.80 ), -INFINITY);
  EXPECT_EQ(d->getLogDensity( 9.90 ), -INFINITY);
  EXPECT_EQ(d->getLogDensity( 10.00 ), -INFINITY);

  // Checking random numbers are within the expected range
  for (size_t i = 0; i < 100; i++)
  {
   double y = d->getRandomNumber();
   EXPECT_TRUE((y >= d->_minimum) && (y <= d->_maximum));
  }

   // Normal case for log density gradient and hessian
  EXPECT_EQ(d->getLogDensityGradient( 5.0 ), 0.0);
  EXPECT_EQ(d->getLogDensityHessian( 5.0 ), 0.0);
 }
 
 TEST(Distrtibutions, UniformRatioDistribution)
 {
  knlohmann::json distributionJs;
  Experiment e;
  distribution::univariate::UniformRatio* d;

  // Creating distribution with an incorrect name
  distributionJs["Type"] = "Distribution/Univariate/UniformRatio";
  ASSERT_ANY_THROW(d = dynamic_cast<korali::distribution::univariate::UniformRatio *>(Module::getModule(distributionJs, &e)));

  // Creating distribution correctly now
  distributionJs["Type"] = "Univariate/UniformRatio";
  ASSERT_NO_THROW(d = dynamic_cast<korali::distribution::univariate::UniformRatio *>(Module::getModule(distributionJs, &e)));

  //////////////////////////////////////////////////

  // Getting module defaults
  distributionJs["Name"] = "Test";
  ASSERT_NO_THROW(d->applyModuleDefaults(distributionJs));
  ASSERT_NO_THROW(d->applyVariableDefaults());
  auto baseJs = distributionJs;

  // Testing correct shape
  distributionJs["Minimum X"] = 2.0;
  distributionJs["Maximum X"] = 8.0;
  distributionJs["Minimum Y"] = 1.0;
  distributionJs["Maximum Y"] = 9.0;
  ASSERT_NO_THROW(d->setConfiguration(distributionJs));
  ASSERT_NO_THROW(d->updateDistribution());

  // Testing get configuration method
  d->getConfiguration(distributionJs);
  ASSERT_EQ(distributionJs["Type"].get<std::string>(), "univariate/uniformratio");
  ASSERT_EQ(distributionJs["Minimum X"].get<double>(), 2.0);
  ASSERT_EQ(distributionJs["Maximum X"].get<double>(), 8.0);
  ASSERT_EQ(distributionJs["Minimum Y"].get<double>(), 1.0);
  ASSERT_EQ(distributionJs["Maximum Y"].get<double>(), 9.0);

  // Testing get property pointer
  ASSERT_TRUE(d->getPropertyPointer("Minimum X") != NULL);
  ASSERT_TRUE(d->getPropertyPointer("Maximum X") != NULL);
  ASSERT_TRUE(d->getPropertyPointer("Minimum Y") != NULL);
  ASSERT_TRUE(d->getPropertyPointer("Maximum Y") != NULL);
  ASSERT_ANY_THROW(d->getPropertyPointer("Undefined")); // Distribution not recognized

  // Testing incorrect minimum/maximum
  d->_minimumX = 16.0;
  ASSERT_NO_THROW(d->updateDistribution());
  ASSERT_ANY_THROW(d->getRandomNumber());

  // Testing correct minimum/maximum
  d->_minimumX = 2.0;
  ASSERT_NO_THROW(d->updateDistribution());
  ASSERT_NO_THROW(d->getRandomNumber());

  /////////////////////////////////////////////////

  // Testing density
  EXPECT_NEAR(d->getDensity( 0 ), 0000000, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 1. ), 0.625000, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 2. ), 0.156250, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 4. ), 0.031250, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 5. ), 0.016250, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 8. ), 0.000000, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getDensity( 9. ), 0.000000, PDENSITY_ERROR_TOLERANCE);
 
  // Testing log density
  EXPECT_EQ(d->getLogDensity( 0 ), -INFINITY);
  EXPECT_NEAR(d->getLogDensity( 2. ), -1.856298, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensity( 4. ), -3.465736, PDENSITY_ERROR_TOLERANCE);
  EXPECT_EQ(d->getLogDensity( 8. ), -INFINITY);
  EXPECT_EQ(d->getLogDensity( 9. ), -INFINITY);

  // Testing log density gradient
  EXPECT_NEAR(d->getLogDensityGradient( 0 ), 0.000000, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensityGradient( 1. ), -2.000000, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensityGradient( 2. ), -1.000000, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensityGradient( 9. ), 0.000000, PDENSITY_ERROR_TOLERANCE);
  
  // Testing log density Hessian
  EXPECT_NEAR(d->getLogDensityHessian( 1. ), -10.000000, PDENSITY_ERROR_TOLERANCE);
  EXPECT_NEAR(d->getLogDensityHessian( 2. ), -2.500000, PDENSITY_ERROR_TOLERANCE);

 }


 TEST(Distrtibutions, WeibullDistribution)
  {
   knlohmann::json distributionJs;
   Experiment e;
   distribution::univariate::Weibull* d;

   // Creating distribution with an incorrect name
   distributionJs["Type"] = "Distribution/Univariate/Weibull";
   ASSERT_ANY_THROW(d = dynamic_cast<korali::distribution::univariate::Weibull *>(Module::getModule(distributionJs, &e)));

   // Creating distribution correctly now
   distributionJs["Type"] = "Univariate/Weibull";
   ASSERT_NO_THROW(d = dynamic_cast<korali::distribution::univariate::Weibull *>(Module::getModule(distributionJs, &e)));

   //////////////////////////////////////////////////

   // Getting module defaults
   distributionJs["Name"] = "Test";
   ASSERT_NO_THROW(d->applyModuleDefaults(distributionJs));
   ASSERT_NO_THROW(d->applyVariableDefaults());
   auto baseJs = distributionJs;

   // Testing conditional variables
   distributionJs = baseJs;
   distributionJs["Shape"] = "Conditional 1";
   distributionJs["Scale"] = "Conditional 2";
   ASSERT_NO_THROW(d->setConfiguration(distributionJs));

   // Testing get configuration method
   d->getConfiguration(distributionJs);
   ASSERT_EQ(distributionJs["Type"].get<std::string>(), "univariate/weibull");
   ASSERT_EQ(distributionJs["Shape"].get<std::string>(), "Conditional 1");
   ASSERT_EQ(distributionJs["Scale"].get<std::string>(), "Conditional 2");

   // Testing correct shape
   distributionJs["Shape"] = 0.5;
   distributionJs["Scale"] = 0.5;
   ASSERT_NO_THROW(d->setConfiguration(distributionJs));
   ASSERT_NO_THROW(d->updateDistribution());

   // Testing get configuration method
   d->getConfiguration(distributionJs);
   ASSERT_EQ(distributionJs["Type"].get<std::string>(), "univariate/weibull");
   ASSERT_EQ(distributionJs["Shape"].get<double>(), 0.5);
   ASSERT_EQ(distributionJs["Scale"].get<double>(), 0.5);

   // Testing get property pointer
   ASSERT_TRUE(d->getPropertyPointer("Shape") != NULL);
   ASSERT_TRUE(d->getPropertyPointer("Scale") != NULL);
   ASSERT_ANY_THROW(d->getPropertyPointer("Undefined")); // Distribution not recognized

   // Testing incorrect shape
   d->_shape = -0.5;
   d->_scale = 0.5;
   ASSERT_ANY_THROW(d->updateDistribution());

   // Testing incorrect scale
   d->_shape = 0.5;
   d->_scale = -0.5;
   ASSERT_ANY_THROW(d->updateDistribution());

   // Testing correct shape and scale
   d->_shape = 2.0;
   d->_scale = 1.0;
   ASSERT_NO_THROW(d->updateDistribution());


   /////////////////////////////////////////////////

   // Distributions generated with https://keisan.casio.com/exec/system/1180573226

   // Testing expected density
   EXPECT_NEAR(d->getDensity( 0 ), 0 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getDensity( 0.1 ), 0.198009967 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getDensity( 0.2 ), 0.384315776 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getDensity( 0.3 ), 0.548358711 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getDensity( 0.4 ), 0.681715031 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getDensity( 0.5 ), 0.778800783 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getDensity( 0.6 ), 0.837211591 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getDensity( 0.7 ), 0.857676952 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getDensity( 0.8 ), 0.843667879 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getDensity( 0.9 ), 0.800744519 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getDensity( 1.00 ), 0.735758882 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getDensity( 1.10 ), 0.656034015 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getDensity( 1.20 ), 0.568626621 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getDensity( 1.30 ), 0.479750762 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getDensity( 1.40 ), 0.394403579 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getDensity( 1.50 ), 0.316197674 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getDensity( 1.60 ), 0.247375169 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getDensity( 1.70 ), 0.188959123 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getDensity( 1.80 ), 0.140990022 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getDensity( 1.90 ), 0.102797018 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getDensity( 2.00 ), 0.073262556 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getDensity( 2.10 ), 0.051051749 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getDensity( 2.20 ), 0.034791038 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getDensity( 2.30 ), 0.023192097 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getDensity( 2.40 ), 0.015125336 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getDensity( 2.50 ), 0.009652271 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getDensity( 2.60 ), 0.006027992 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getDensity( 2.70 ), 0.003684571 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getDensity( 2.80 ), 0.002204547 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getDensity( 2.90 ), 0.001291253 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getDensity( 3.00 ), 0.000740459 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getDensity( 3.10 ), 0.000415740 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getDensity( 3.20 ), 0.000228562 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getDensity( 3.30 ), 0.000123049 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getDensity( 3.40 ), 0.000064873 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getDensity( 3.50 ), 0.000033496 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getDensity( 3.60 ), 0.000016939 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getDensity( 3.70 ), 0.000008390 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getDensity( 3.80 ), 0.000004070 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getDensity( 3.90 ), 0.000001934 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getDensity( 4.00 ), 0.000000900 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getDensity( 4.10 ), 0.000000411 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getDensity( 4.20 ), 0.000000183 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getDensity( 4.30 ), 0.000000080 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getDensity( 4.40 ), 0.000000034 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getDensity( 4.50 ), 0.000000014 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getDensity( 4.60 ), 0.000000006 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getDensity( 4.70 ), 0.000000002 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getDensity( 4.80 ), 0.000000001 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getDensity( 4.90 ), 0.000000000 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getDensity( 5.00 ), 0.000000000 , PDENSITY_ERROR_TOLERANCE);

   // Testing log density function
   EXPECT_EQ(d->getLogDensity( 0.00 ), -INFINITY);
   EXPECT_NEAR(d->getLogDensity( 0.10 ), -1.619437912 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getLogDensity( 0.20 ), -0.956290732 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getLogDensity( 0.30 ), -0.600825624 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getLogDensity( 0.40 ), -0.383143551 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getLogDensity( 0.50 ), -0.250000000 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getLogDensity( 0.60 ), -0.177678443 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getLogDensity( 0.70 ), -0.153527763 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getLogDensity( 0.80 ), -0.169996371 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getLogDensity( 0.90 ), -0.222213335 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getLogDensity( 1.00 ), -0.306852819 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getLogDensity( 1.10 ), -0.421542640 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getLogDensity( 1.20 ), -0.564531263 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getLogDensity( 1.30 ), -0.734488555 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getLogDensity( 1.40 ), -0.930380583 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getLogDensity( 1.50 ), -1.151387711 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getLogDensity( 1.60 ), -1.396849190 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getLogDensity( 1.70 ), -1.666224568 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getLogDensity( 1.80 ), -1.959066154 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getLogDensity( 1.90 ), -2.274998933 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getLogDensity( 2.00 ), -2.613705639 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getLogDensity( 2.10 ), -2.974915475 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getLogDensity( 2.20 ), -3.358395459 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getLogDensity( 2.30 ), -3.763943696 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getLogDensity( 2.40 ), -4.191384082 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getLogDensity( 2.50 ), -4.640562088 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getLogDensity( 2.60 ), -5.111341374 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getLogDensity( 2.70 ), -5.603601046 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getLogDensity( 2.80 ), -6.117233402 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getLogDensity( 2.90 ), -6.652142083 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getLogDensity( 3.00 ), -7.208240531 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getLogDensity( 3.10 ), -7.785450708 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getLogDensity( 3.20 ), -8.383702010 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getLogDensity( 3.30 ), -9.002930351 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getLogDensity( 3.40 ), -9.643077388 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getLogDensity( 3.50 ), -10.304089851 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getLogDensity( 3.60 ), -10.985918974 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getLogDensity( 3.70 ), -11.688520000 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getLogDensity( 3.80 ), -12.411851753 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getLogDensity( 3.90 ), -13.155876266 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getLogDensity( 4.00 ), -13.920558458 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getLogDensity( 4.10 ), -14.705865846 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getLogDensity( 4.20 ), -15.511768294 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getLogDensity( 4.30 ), -16.338237797 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getLogDensity( 4.40 ), -17.185248279 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getLogDensity( 4.50 ), -18.052775422 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getLogDensity( 4.60 ), -18.940796516 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getLogDensity( 4.70 ), -19.849290311 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getLogDensity( 4.80 ), -20.778236902 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getLogDensity( 4.90 ), -21.727617614 , PDENSITY_ERROR_TOLERANCE);
   EXPECT_NEAR(d->getLogDensity( 5.00 ), -22.697414907 , PDENSITY_ERROR_TOLERANCE);

   // Checking random numbers are within the expected range
   for (size_t i = 0; i < 100; i++)
   {
    double y = d->getRandomNumber();
    EXPECT_TRUE((y >= 0.0));
   }

   // Testing extreme cases for log density gradient and hessian
   EXPECT_EQ(d->getLogDensityGradient( -0.001 ), 0.0);
   EXPECT_EQ(d->getLogDensityHessian( -0.001 ), 0.0);

   // Normal case for log density gradient and hessian
   ASSERT_NO_THROW(d->getLogDensityGradient( 0.5 ));
   ASSERT_NO_THROW(d->getLogDensityHessian( 0.5 ));
  }

 TEST(Distrtibutions, MultivariateNormalDistribution)
  {
   knlohmann::json distributionJs;
   Experiment e;
   distribution::multivariate::Normal* d;

   // Creating distribution with an incorrect name
   distributionJs["Type"] = "Distribution/Multivariate/Normal";
   ASSERT_ANY_THROW(d = dynamic_cast<korali::distribution::multivariate::Normal *>(Module::getModule(distributionJs, &e)));

   // Creating distribution correctly now
   distributionJs["Type"] = "Multivariate/Normal";
   ASSERT_NO_THROW(d = dynamic_cast<korali::distribution::multivariate::Normal *>(Module::getModule(distributionJs, &e)));

   //////////////////////////////////////////////////

   // Getting module defaults
   distributionJs["Name"] = "Test";
   ASSERT_NO_THROW(d->applyModuleDefaults(distributionJs));
   ASSERT_NO_THROW(d->applyVariableDefaults());
   auto baseJs = distributionJs;

   // Testing correct shape
   distributionJs = baseJs;
   distributionJs["Mean Vector"] = std::vector<double>({ 0.5, 0.5, 0.5, 0.5 });
   distributionJs["Sigma"] = std::vector<double>({ 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 });
   ASSERT_NO_THROW(d->setConfiguration(distributionJs));
   ASSERT_NO_THROW(d->updateDistribution());

   // Setting properties directly
   ASSERT_NO_THROW(d->setProperty("Mean Vector", std::vector<double>({ 0.5, 0.5, 0.5, 0.5 })));
   ASSERT_NO_THROW(d->setProperty("Sigma", std::vector<double>({ 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 })));
   ASSERT_ANY_THROW(d->setProperty("Undefined", std::vector<double>({ })));

   // Testing get configuration method
   d->getConfiguration(distributionJs);
   ASSERT_EQ(distributionJs["Type"].get<std::string>(), "multivariate/normal");
   ASSERT_TRUE(distributionJs["Mean Vector"].get<std::vector<double>>() == std::vector<double>({ 0.5, 0.5, 0.5, 0.5 }));
   ASSERT_TRUE(distributionJs["Sigma"].get<std::vector<double>>() == std::vector<double>({ 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 }));

   /////////////////////////////////////////////////

   // Distributions generated with https://keisan.casio.com/exec/system/1180573226

   // Testing correct execution
   double x[] = { 0.3, 0.7, 0.2, 0.1 };
   double res[4];
   ASSERT_NO_THROW(d->getDensity( x, res, 4));
   ASSERT_ANY_THROW(d->getDensity( x, res, 6));

   ASSERT_NO_THROW(d->getLogDensity( x, res, 4));
   ASSERT_ANY_THROW(d->getLogDensity( x, res, 6));

   // Test rng with correct and incorrect sigma vector sizes
   d->_sigma = std::vector<double>({ 0.5, 0.5, 0.5 });
   ASSERT_ANY_THROW(d->updateDistribution());
   d->_sigma = std::vector<double>({ 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 });
   ASSERT_NO_THROW(d->updateDistribution());
   ASSERT_NO_THROW(d->getRandomVector(x, 4));

   distributionJs = baseJs;
   distributionJs["Work Vector"] = "Not a Number";
   ASSERT_ANY_THROW(d->setConfiguration(distributionJs));

   distributionJs = baseJs;
   distributionJs["Work Vector"] = std::vector<double>({ 0.5, 0.5, 0.5, 0.5 });
   ASSERT_NO_THROW(d->setConfiguration(distributionJs));

   distributionJs = baseJs;
   distributionJs.erase("Mean Vector");
   ASSERT_ANY_THROW(d->setConfiguration(distributionJs));

   distributionJs = baseJs;
   distributionJs["Mean Vector"] = "Not a Number";
   ASSERT_ANY_THROW(d->setConfiguration(distributionJs));

   distributionJs = baseJs;
   distributionJs["Mean Vector"] = std::vector<double>({ 0.5, 0.5, 0.5, 0.5 });
   ASSERT_NO_THROW(d->setConfiguration(distributionJs));

   distributionJs = baseJs;
   distributionJs.erase("Sigma");
   ASSERT_ANY_THROW(d->setConfiguration(distributionJs));

   distributionJs = baseJs;
   distributionJs["Sigma"] = "Not a Number";
   ASSERT_ANY_THROW(d->setConfiguration(distributionJs));

   distributionJs = baseJs;
   distributionJs["Sigma"] = std::vector<double>({ 0.5, 0.5, 0.5, 0.5 });
   ASSERT_NO_THROW(d->setConfiguration(distributionJs));

   // To-do: check expected densities
  }

} // namespace
