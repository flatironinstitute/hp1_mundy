// @HEADER
// **********************************************************************************************************************
//
//                                   NgpHP1: Interphase Chromatin Modeling using MuNDy
//                                             Copyright 2025 Bryce Palmer et al.
//
// NgpHP1 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
//
// NgpHP1 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
// of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along with Mundy. If not, see
// <https://www.gnu.org/licenses/>.
//
// **********************************************************************************************************************
// @HEADER

#ifndef MUNDY_NGHP1_CHROMOSOME_POSITION_REGION_GENERATORS_HPP_
#define MUNDY_NGHP1_CHROMOSOME_POSITION_REGION_GENERATORS_HPP_

// External
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <openrand/philox.h>

// C++ core
#include <optional>  // for std::optional
#include <string>    // for std::string
#include <tuple>     // for std::tuple
#include <utility>   // for std::move
#include <vector>    // for std::vector

// Mundy core
#include <mundy_core/throw_assert.hpp>  // for MUNDY_THROW_ASSERT

// Mundy math
#include <mundy_math/Hilbert.hpp>  // for mundy::math::create_hilbert_positions_and_directors
#include <mundy_math/Vector3.hpp>  // for mundy::math::Vector3

// Mundy geom
#include <mundy_geom/distance.hpp>    // for mundy::geom::distance(primA, primB)
#include <mundy_geom/primitives.hpp>  // for all geometric primitives mundy::geom::Point, Line, Sphere, Ellipsoid...

namespace mundy {

namespace ngphp1 {

//! \name Chromatin position and region generators
//@{

void get_chromosome_positions_regions_from_file(
    const std::string& file_name, const unsigned num_chromosomes,
    std::vector<std::vector<mundy::geom::Point<double>>>* all_chromosome_positions,
    std::vector<std::vector<std::string>>* all_chromosome_regions) {
  // The file should be formatted as follows:
  // chromosome_id x y z region
  // 0 x1 y1 z1 H
  // 0 x2 y2 z2 H
  // ...
  // 1 x1 y1 z1 E
  // 1 x2 y2 z2 H
  //
  // chromosome_id should start at 1 and increase by 1 for each new chromosome.
  //
  // And so on for each chromosome. The total number of chromosomes should match the expected total, lest we throw an
  // exception.
  std::ifstream infile(file_name);
  MUNDY_THROW_REQUIRE(infile.is_open(), std::invalid_argument, fmt::format("Could not open file {}", file_name));

  // Read each line. While the chromosome_id is the same, keep adding nodes to the chromosome.
  size_t current_chromosome_id = 1;
  std::string line;
  while (std::getline(infile, line)) {
    std::istringstream iss(line);
    int chromosome_id;
    double x, y, z;
    // Check for our required fields (which doesn't include region)
    if (!(iss >> chromosome_id >> x >> y >> z)) {
      MUNDY_THROW_REQUIRE(false, std::invalid_argument, fmt::format("Could not parse line {}", line));
    }
    // Check for the optional region encoding
    std::optional<std::string> optional_region;
    if (std::string tmp; iss >> tmp) {
      optional_region = std::move(tmp);
    }
    if (static_cast<unsigned>(chromosome_id) != current_chromosome_id) {
      // We are starting a new chromosome
      MUNDY_THROW_REQUIRE(static_cast<unsigned>(chromosome_id) == current_chromosome_id + 1, std::invalid_argument,
                          "Chromosome IDs should be sequential.");
      MUNDY_THROW_REQUIRE(static_cast<unsigned>(chromosome_id) <= num_chromosomes, std::invalid_argument,
                          fmt::format("Chromosome ID {} is greater than the number of chromosomes.", chromosome_id));
      current_chromosome_id = chromosome_id;
    }
    // Add the node to the chromosome
    (*all_chromosome_positions)[current_chromosome_id - 1].emplace_back(x, y, z);
    // Check the optional encoding
    if (optional_region) {
      (*all_chromosome_regions)[current_chromosome_id - 1].emplace_back(*optional_region);
    }
  }
}

// std::vector<std::vector<mundy::geom::Point<double>>> get_chromosome_positions_grid(
//     const unsigned num_chromosomes, const unsigned num_nodes_per_chromosome, const double segment_length) {
//   std::vector<std::vector<mundy::geom::Point<double>>> all_chromosome_positions(num_chromosomes);
//   const mundy::math::Vector3d alignment_dir{0.0, 0.0, 1.0};
//   for (size_t j = 0; j < num_chromosomes; j++) {
//     all_chromosome_positions[j].reserve(num_nodes_per_chromosome);
//     openrand::Philox rng(j, 0);
//     mundy::math::Vector3d start_pos(2.0 * static_cast<double>(j), 0.0, 0.0);
//     for (size_t i = 0; i < num_nodes_per_chromosome; ++i) {
//       const auto pos = start_pos + static_cast<double>(i) * segment_length * alignment_dir;
//       all_chromosome_positions[j].emplace_back(pos);
//     }
//   }

//   return all_chromosome_positions;
// }

// std::vector<std::vector<mundy::geom::Point<double>>> get_chromosome_positions_random_unit_cell(
//     const unsigned num_chromosomes,           //
//     const unsigned num_nodes_per_chromosome,  //
//     const double segment_length,              //
//     const double domain_low[3],               //
//     const double domain_high[3]) {
//   std::vector<std::vector<mundy::geom::Point<double>>> all_chromosome_positions(num_chromosomes);
//   for (size_t j = 0; j < num_chromosomes; j++) {
//     all_chromosome_positions[j].reserve(num_nodes_per_chromosome);

//     // Find a random place within the unit cell with a random orientation for the chain.
//     openrand::Philox rng(j, 0);
//     mundy::math::Vector3d pos_start{rng.uniform<double>(domain_low[0], domain_high[0]),
//                                     rng.uniform<double>(domain_low[1], domain_high[1]),
//                                     rng.uniform<double>(domain_low[2], domain_high[2])};

//     // Find a random unit vector direction
//     const double zrand = rng.rand<double>() - 1.0;
//     const double wrand = std::sqrt(1.0 - zrand * zrand);
//     const double trand = 2.0 * M_PI * rng.rand<double>();
//     mundy::math::Vector3d u_hat{wrand * std::cos(trand), wrand * std::sin(trand), zrand};

//     for (size_t i = 0; i < num_nodes_per_chromosome; ++i) {
//       auto pos = pos_start + static_cast<double>(i) * segment_length * u_hat;
//       all_chromosome_positions[j].emplace_back(pos);
//     }
//   }

//   return all_chromosome_positions;
// }

// std::vector<std::vector<mundy::geom::Point<double>>> get_chromosome_positions_hilbert_random_unit_cell(
//     const unsigned num_chromosomes,           //
//     const unsigned num_nodes_per_chromosome,  //
//     const double segment_length,              //
//     const double domain_low[3],               //
//     const double domain_high[3]) {
//   std::vector<std::vector<mundy::geom::Point<double>>> all_chromosome_positions(num_chromosomes);
//   std::vector<mundy::geom::Point<double>> chromosome_centers_array(num_chromosomes);
//   std::vector<double> chromosome_radii_array(num_chromosomes);
//   for (size_t ichromosome = 0; ichromosome < num_chromosomes; ichromosome++) {
//     // Generate a random unit vector (will be used for creating the location of the nodes, the random position in
//     // the unit cell will be handled later).
//     openrand::Philox rng(ichromosome, 0);
//     const double zrand = rng.rand<double>() - 1.0;
//     const double wrand = std::sqrt(1.0 - zrand * zrand);
//     const double trand = 2.0 * M_PI * rng.rand<double>();
//     mundy::math::Vector3d u_hat(wrand * std::cos(trand), wrand * std::sin(trand), zrand);

//     // Once we have the number of chromosome spheres we can get the hilbert curve set up. This will be at some
//     // orientation and then have sides with a length of initial_chromosome_separation.
//     auto [hilbert_position_array, hilbert_directors] =
//         mundy::math::create_hilbert_positions_and_directors(num_nodes_per_chromosome, u_hat, segment_length);

//     // Create the local positions of the spheres
//     std::vector<mundy::math::Vector3d> sphere_position_array;
//     for (size_t isphere = 0; isphere < num_nodes_per_chromosome; isphere++) {
//       sphere_position_array.push_back(hilbert_position_array[isphere]);
//     }

//     // Figure out where the center of the chromosome is, and its radius, in its own local space
//     mundy::math::Vector3d r_chromosome_center_local(0.0, 0.0, 0.0);
//     double r_max = 0.0;
//     for (size_t i = 0; i < sphere_position_array.size(); i++) {
//       r_chromosome_center_local += sphere_position_array[i];
//     }
//     r_chromosome_center_local /= static_cast<double>(sphere_position_array.size());
//     for (size_t i = 0; i < sphere_position_array.size(); i++) {
//       r_max = std::max(r_max, mundy::math::two_norm(r_chromosome_center_local - sphere_position_array[i]));
//     }

//     // Do max_trials number of insertion attempts to get a random position and orientation within the unit cell that
//     // doesn't overlap with exiting chromosomes.
//     const size_t max_trials = 1000;
//     size_t itrial = 0;
//     bool chromosome_inserted = false;
//     while (itrial <= max_trials) {
//       // Generate a random position within the unit cell.
//       mundy::math::Vector3d r_start(rng.uniform<double>(domain_low[0], domain_high[0]),
//                                     rng.uniform<double>(domain_low[1], domain_high[1]),
//                                     rng.uniform<double>(domain_low[2], domain_high[2]));

//       // Check for overlaps with existing chromosomes
//       bool found_overlap = false;
//       for (size_t jchromosome = 0; jchromosome < chromosome_centers_array.size(); ++jchromosome) {
//         double r_chromosome_distance = mundy::math::two_norm(chromosome_centers_array[jchromosome] - r_start);
//         if (r_chromosome_distance < (r_max + chromosome_radii_array[jchromosome])) {
//           found_overlap = true;
//           break;
//         }
//       }
//       if (found_overlap) {
//         itrial++;
//       } else {
//         chromosome_inserted = true;
//         chromosome_centers_array[ichromosome] = r_start;
//         chromosome_radii_array[ichromosome] = r_max;
//         break;
//       }
//     }
//     MUNDY_THROW_REQUIRE(chromosome_inserted, std::runtime_error,
//                         fmt::format("Failed to insert chromosome after {} trials.", max_trials));

//     // Generate all the positions along the curve due to the placement in the global space
//     const size_t num_nodes_per_chromosome = sphere_position_array.size();
//     for (size_t i = 0; i < num_nodes_per_chromosome; i++) {
//       all_chromosome_positions[ichromosome].emplace_back(chromosome_centers_array.back() + r_chromosome_center_local
//       -
//                                                          sphere_position_array[i]);
//     }
//   }

//   return all_chromosome_positions;
// }
//@}

}  // namespace ngphp1

}  // namespace mundy

//@}

#endif  // MUNDY_NGHP1_CHROMOSOME_POSITION_REGION_GENERATORS_HPP_