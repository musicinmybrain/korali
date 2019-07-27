#include "korali.hpp"
#include <chrono>

/************************************************************************/
/*                  Constructor / Destructor Methods                    */
/************************************************************************/

Korali::Variable::Variable()
{
 _a = 1.0;
 _b = 1.0;
 _aux = 0.0;
 _seed = 0;
 _name = "Unnamed";
 _distributionType = "No Distribution";
 _isLogSpace = false;
 _range = gsl_rng_alloc (gsl_rng_default);
}

Korali::Variable::~Variable()
{
 gsl_rng_free(_range);
}

/************************************************************************/
/*                    Configuration Methods                             */
/************************************************************************/
void Korali::Variable::setProperty(std::string propertyName, double value)
{
 bool _recognizedProperty = false;

 if (_distributionType == "Cauchy")
  {
    if (propertyName == "Location") { _a = value; _recognizedProperty = true; }
    if (propertyName == "Scale") { _b = value; _recognizedProperty = true; }
    if (_b > 0) _aux = -gsl_sf_log( _b * M_PI );
  }

  if (_distributionType == "Exponential")
  {
   if (propertyName == "Mean") { _a = value; _recognizedProperty = true; }
   if (propertyName == "Location") { _b = value; _recognizedProperty = true; }
   _aux = 0.0;
  }

  if (_distributionType == "Gamma")
  {
   if (propertyName == "Scale") { _a = value; _recognizedProperty = true; }
   if (propertyName == "Shape") { _b = value; _recognizedProperty = true; }
   if (_b > 0 && _a > 0) _aux = -gsl_sf_lngamma(_b) - _b*log(_a);
  }

  if (_distributionType == "Gaussian")
  {
   if (propertyName == "Mean") { _a = value; _recognizedProperty = true; }
   if (propertyName == "Standard Deviation") { _b = value; _recognizedProperty = true; }
   if (_b > 0) _aux = -0.5*gsl_sf_log(2*M_PI) - gsl_sf_log(_b);
  }

  if (_distributionType == "Laplace")
  {
   if (propertyName == "Mean") { _a = value; _recognizedProperty = true; }
   if (propertyName == "Width") { _b = value; _recognizedProperty = true; }
   if (_b > 0) _aux = -gsl_sf_log(2.*_b);
  }

  if (_distributionType == "Uniform")
  {
   if (propertyName == "Minimum") { _a = value; _recognizedProperty = true; }
   if (propertyName == "Maximum") { _b = value; _recognizedProperty = true; }
   if (_b-_a > 0) _aux = -gsl_sf_log(_b-_a);
  }

  if (_distributionType == "Geometric")
  {
   if (propertyName == "Success Probability") { _a = value; _recognizedProperty = true; }
   _aux = 0.0;
  }

  if (_recognizedProperty == false) koraliError("Incorrect or missing property %s for distribution type %s.\n", propertyName.c_str(), _distributionType.c_str());
}

void Korali::Variable::setDistributionType(std::string distributionType)
{
 bool foundDistributionType = false;
 if (distributionType == "No Distribution")     foundDistributionType = true;
 if (distributionType == "Cauchy")      foundDistributionType = true;
 if (distributionType == "Exponential") foundDistributionType = true;
 if (distributionType == "Gamma")       foundDistributionType = true;
 if (distributionType == "Gaussian")    foundDistributionType = true;
 if (distributionType == "Laplace")     foundDistributionType = true;
 if (distributionType == "Uniform")     foundDistributionType = true;
 if (distributionType == "Geometric")   foundDistributionType = true;
 if (foundDistributionType == false) koraliError("Incorrect or missing distribution %s for parameter %s.\n", distributionType.c_str(), _name.c_str());
 _distributionType = distributionType;
}

void Korali::Variable::setDistribution(nlohmann::json& js)
{
 _seed = consume(js, { "Seed" }, KORALI_NUMBER);
 gsl_rng_set(_range, _seed);

 auto dString = consume(js, { "Type" }, KORALI_STRING, "No Distribution");
 setDistributionType(dString);

 for (auto& property : js.items())
 {
  if (js[property.key()].is_number())
   {
    setProperty(property.key(), property.value());
    js.erase(property.key());
   }
 }
}

void Korali::Variable::getDistribution(nlohmann::json& js)
{
 js["Seed"] = _seed;
 js["Type"] = _distributionType;

 if (_distributionType == "Cauchy")
 {
  js["Location"] = _a;
  js["Scale"] = _b;
 }

 if (_distributionType == "Exponential")
 {
  js["Mean"] = _a;
  js["Location"] = _b;
 }

 if (_distributionType == "Gamma")
 {
  js["Scale"] = _a;
  js["Shape"] = _b;
 }

 if (_distributionType == "Gaussian")
 {
  js["Mean"] = _a;
  js["Standard Deviation"] = _b;
 }

 if (_distributionType == "Laplace")
 {
  js["Mean"] = _a;
  js["Width"] = _b;
 }

 if (_distributionType == "Uniform")
 {
  js["Minimum"] = _a;
  js["Maximum"] = _b;
 }

 if (_distributionType == "Geometric")
 {
  js["Success Probability"] = _a;
 }
}

/************************************************************************/
/*                    Functional Methods                                */
/************************************************************************/

void Korali::Variable::getConfiguration(nlohmann::json& js)
{
 js["Log Space"] = _isLogSpace;
 js["Name"] = _name;
 getDistribution(js["Prior Distribution"]);
 getSolverSettings(js);
}

void Korali::Variable::setConfiguration(nlohmann::json& js)
{
 _isLogSpace = consume(js, { "Log Space"}, KORALI_BOOLEAN, "false");
 _name = consume(js, { "Name" }, KORALI_STRING);

 setDistribution(js["Prior Distribution"]);
 js.erase("Prior Distribution");

 setSolverSettings(js);
}

double Korali::Variable::getDensity(double x)
{
 if (_distributionType == "Cauchy")      { return gsl_ran_cauchy_pdf( x-_a, _b ); }
 if (_distributionType == "Exponential") { return gsl_ran_exponential_pdf(x-_b, _a); }
 if (_distributionType == "Gamma")       { return gsl_ran_gamma_pdf( x, _b, _a ); }
 if (_distributionType == "Gaussian")    { return gsl_ran_gaussian_pdf(x - _a, _b); }
 if (_distributionType == "Laplace")     { return gsl_ran_laplace_pdf( x-_a, _b ); }
 if (_distributionType == "Uniform")     { return gsl_ran_flat_pdf(x, _a, _b); }
 if (_distributionType == "Geometric")   { return gsl_ran_geometric_pdf((int)x, _a); }

 koraliError("Problem requires that variable '%s' has a defined distribution.\n", _name.c_str());
 return 0.0;
};

double Korali::Variable::getLogDensity(double x)
{
 if (_distributionType == "Cauchy")      { return _aux - gsl_sf_log( 1. + gsl_sf_pow_int((x-_a)/_b,2) ); }
 if (_distributionType == "Exponential") { if (x-_b < 0) return -INFINITY; return - log(_a) - (x-_b)/_a; }
 if (_distributionType == "Gamma")       { if(x < 0) return -INFINITY; return _aux + (_b-1)*log(x) - x/_a; }
 if (_distributionType == "Gaussian")    { double d = (x-_a)/_b; return _aux - 0.5*d*d; }
 if (_distributionType == "Laplace")     { return _aux - fabs(x-_a)/_b; }
 if (_distributionType == "Uniform")     { if (x >= _a && x <= _b) return _aux; return -INFINITY; }
 if (_distributionType == "Geometric")   { return log(_a) + (x-1)*log(1.0-_a); }

 koraliError("Problem requires that variable '%s' has a defined distribution.\n", _name.c_str());
 return 0.0;
};

double Korali::Variable::getRandomNumber()
{
 if (_distributionType == "Cauchy")      { return _a + gsl_ran_cauchy(_range, _b); }
 if (_distributionType == "Exponential") { return _b + gsl_ran_exponential(_range, _a); }
 if (_distributionType == "Gamma")       { return gsl_ran_gamma(_range, _b, _a); }
 if (_distributionType == "Gaussian")    { return _a + gsl_ran_gaussian(_range, _b); }
 if (_distributionType == "Laplace")     { return _a + gsl_ran_laplace(_range, _b); }
 if (_distributionType == "Uniform")     { return gsl_ran_flat(_range, _a, _b); }
 if (_distributionType == "Geometric")   { return gsl_ran_geometric(_range, _a); }

 koraliError("Problem requires that variable '%s' has a defined distribution.\n", _name.c_str());
 return 0.0;
};
