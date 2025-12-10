# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "D:/veekay/testbed/cmake-build-debug-visual-studio/_deps/lodepng-src")
  file(MAKE_DIRECTORY "D:/veekay/testbed/cmake-build-debug-visual-studio/_deps/lodepng-src")
endif()
file(MAKE_DIRECTORY
  "D:/veekay/testbed/cmake-build-debug-visual-studio/_deps/lodepng-build"
  "D:/veekay/testbed/cmake-build-debug-visual-studio/_deps/lodepng-subbuild/lodepng-populate-prefix"
  "D:/veekay/testbed/cmake-build-debug-visual-studio/_deps/lodepng-subbuild/lodepng-populate-prefix/tmp"
  "D:/veekay/testbed/cmake-build-debug-visual-studio/_deps/lodepng-subbuild/lodepng-populate-prefix/src/lodepng-populate-stamp"
  "D:/veekay/testbed/cmake-build-debug-visual-studio/_deps/lodepng-subbuild/lodepng-populate-prefix/src"
  "D:/veekay/testbed/cmake-build-debug-visual-studio/_deps/lodepng-subbuild/lodepng-populate-prefix/src/lodepng-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "D:/veekay/testbed/cmake-build-debug-visual-studio/_deps/lodepng-subbuild/lodepng-populate-prefix/src/lodepng-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "D:/veekay/testbed/cmake-build-debug-visual-studio/_deps/lodepng-subbuild/lodepng-populate-prefix/src/lodepng-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
