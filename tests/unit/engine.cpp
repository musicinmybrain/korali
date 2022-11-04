#include "gtest/gtest.h"
#include "korali.hpp"

namespace
{

 using namespace korali;

 TEST(Engine, badParametersException)
 {
  Engine e;

  // Triggering a configuration parsing error
  e._js["Condruit"] = "Typo";
  ASSERT_ANY_THROW(e.start());

  // Selecting an incorrect conduit
  e._js["Conduit"] = "Incorrect option";
  ASSERT_ANY_THROW(e.start());

  ASSERT_ANY_THROW(e[0] = "Incorrect Key");
 }

} // namespace
