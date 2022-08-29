#pragma once


/** \file
* @brief Implements interface from Boost to STL
******************************************************************************/

#include <vector>
#include <boost/circular_buffer.hpp>

/**
* \namespace korali
* @brief The Korali namespace includes all Korali-specific functions, variables, and modules.
*/
namespace korali
{

/**
 * @brief Copies contents of a boost::circular_buffer into a std::vector
 * @param circularBuffer The boost::circular_buffer
 * @return The std::vector with the contents of the boost::circular_buffer
 */
template<typename T>
std::vector<T> boostToVector( boost::circular_buffer<T> circularBuffer )
{
  const size_t size = circularBuffer.size();
  std::vector<T> vector(size);
  for (size_t i = 0; i < size; i++) vector[i] = circularBuffer[i];
  return vector;
};

} // namespace korali

