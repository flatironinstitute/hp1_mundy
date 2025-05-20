// @HEADER
// **********************************************************************************************************************
//
//                                   HP1Mundy: Interphase Chromatin Modeling using MuNDy
//                                             Copyright 2025 Bryce Palmer et al.
//
// Sam is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
//
// Sam is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
// of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along with Mundy. If not, see
// <https://www.gnu.org/licenses/>.
//
// **********************************************************************************************************************
// @HEADER

// C++ core
#include <iostream>

// External
#include <mpi.h>  // for MPI_Comm, MPI_Finalize, etc

// Trilinos
#include <Kokkos_Core.hpp>  // for Kokkos::initialize, Kokkos::finalize

int main(int argc, char** argv) {
  // Initialize MPI and Kokkos
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);

  {
    std::cout << "Welcome to our example application!\n"
                 "I'm pleased to see that you're interested in using Mundy. \n"
                 "\n"
                 "I treasure Mundy, so if you have questions, suggested improvements,\n"
                 "or issues, please don't hesitate to make a GitHub issue or "
                 "pull request.\n"
                 "\n"
                 "Best of luck.\n"
                 "-Bryce Palmer\n"
              << std::endl;
  }

  Kokkos::finalize();
  MPI_Finalize();

  return 0;
}