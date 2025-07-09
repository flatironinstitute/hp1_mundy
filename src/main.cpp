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

// C++ core
#include <iostream>

// External
#include <openrand/philox.h>

// Trilinos/Kokkos
#include <Kokkos_Core.hpp>
#include <stk_balance/balance.hpp>  // for stk::balance::balanceStkMesh
#include <stk_io/StkMeshIoBroker.hpp>
#include <stk_io/WriteMesh.hpp>
#include <stk_mesh/base/DumpMeshInfo.hpp>  // for stk::mesh::impl::dump_all_mesh_info
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/GetNgpField.hpp>
#include <stk_mesh/base/GetNgpMesh.hpp>
#include <stk_mesh/base/NgpField.hpp>
#include <stk_mesh/base/NgpForEachEntity.hpp>
#include <stk_mesh/base/NgpMesh.hpp>
#include <stk_mesh/base/Selector.hpp>
#include <stk_mesh/base/Types.hpp>         // stk::mesh::EntityRank
#include <stk_topology/topology.hpp>       // stk::topology
#include <stk_util/ngp/NgpSpaces.hpp>      // stk::ngp::ExecSpace, stk::ngp::RangePolicy
#include <stk_util/parallel/Parallel.hpp>  // for stk::parallel_machine_init, stk::parallel_machine_finalize

// Mundy core
#include <mundy_core/throw_assert.hpp>     // for MUNDY_THROW_ASSERT
#include <mundy_geom/distance.hpp>         // for mundy::geom::distance
#include <mundy_geom/primitives.hpp>       // for mundy::geom::Spherocylinder
#include <mundy_math/Quaternion.hpp>       // for mundy::math::Quaternion
#include <mundy_math/Vector3.hpp>          // for mundy::math::Vector3
#include <mundy_mesh/Aggregate.hpp>        // for mundy::mesh::Aggregate
#include <mundy_mesh/BulkData.hpp>         // for mundy::mesh::BulkData
#include <mundy_mesh/DeclareEntities.hpp>  // for mundy::mesh::DeclareEntitiesHelper
#include <mundy_mesh/LinkData.hpp>         // for mundy::mesh::LinkData
#include <mundy_mesh/MeshBuilder.hpp>      // for mundy::mesh::MeshBuilder
#include <mundy_mesh/MetaData.hpp>         // for mundy::mesh::MetaData
#include <mundy_mesh/NgpFieldBLAS.hpp>     // for mundy::mesh::field_copy

// NgpHP1
#include "ChromosomePositionRegionGenerators.hpp"
#include "Parser.hpp"  // for mundy::ngphp1::HP1ParamParser

namespace mundy {

namespace ngphp1 {

// Structures for template accessors and aggregates
struct COORDS {};
struct FORCE {};
struct RNG_COUNTER {};
struct RADIUS {};
struct CHAINID {};
struct LINKED_ENTITIES {};

void run_main(int argc, char **argv) {
  // STK usings
  using stk::mesh::Field;
  using stk::mesh::Part;
  using stk::mesh::Selector;
  using stk::topology::ELEM_RANK;
  using stk::topology::NODE_RANK;

  // Mundy things
  using mesh::BulkData;
  using mesh::DeclareEntitiesHelper;
  using mesh::FieldComponent;
  using mesh::LinkData;
  using mesh::LinkMetaData;
  using mesh::MeshBuilder;
  using mesh::MetaData;
  using mesh::QuaternionFieldComponent;
  using mesh::ScalarFieldComponent;
  using mesh::Vector3FieldComponent;

  // Preprocess
  Teuchos::ParameterList params = HP1ParamParser().parse(argc, argv);
  const auto &sim_params = params.sublist("sim");
  const auto &brownian_motion_params = params.sublist("brownian_motion");
  const auto &backbone_springs_params = params.sublist("backbone_springs");
  const auto &backbone_collision_params = params.sublist("backbone_collision");
  const auto &crosslinker_params = params.sublist("crosslinker");
  const auto &periphery_hydro_params = params.sublist("periphery_hydro");
  const auto &periphery_collision_params = params.sublist("periphery_collision");
  const auto &periphery_binding_params = params.sublist("periphery_binding");
  const auto &active_euchromatin_forces_params = params.sublist("active_euchromatin_forces");
  const auto &neighbor_list_params = params.sublist("neighbor_list");

  // Setup the STK mesh
  MeshBuilder mesh_builder(MPI_COMM_WORLD);
  mesh_builder.set_spatial_dimension(3).set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
  std::shared_ptr<MetaData> meta_data_ptr = mesh_builder.create_meta_data();
  meta_data_ptr->use_simple_fields();
  meta_data_ptr->set_coordinate_field_name("COORDS");
  std::shared_ptr<BulkData> bulk_data_ptr = mesh_builder.create_bulk_data(meta_data_ptr);
  MetaData &meta_data = *meta_data_ptr;
  BulkData &bulk_data = *bulk_data_ptr;

  // Setup the link data (boilerplate)
  LinkMetaData link_meta_data = declare_link_meta_data(meta_data, "ALL_LINKS", NODE_RANK);
  LinkData link_data = declare_link_data(bulk_data, link_meta_data);

  // Declare parts
  // Chromatin vertex spheres for hydrodynamics and vertex type
  auto particle_top = stk::topology::PARTICLE;
  auto beam2_top = stk::topology::BEAM_2;
  auto node_top = stk::topology::NODE;
  auto &spheres_part = meta_data.declare_part("SPHERES", stk::topology::ELEM_RANK);
  auto &e_spheres_part = meta_data.declare_part_with_topology("EUCHROMATIN_SPHERES", particle_top);
  auto &h_spheres_part = meta_data.declare_part_with_topology("HETEROCHROMATIN_SPHERES", particle_top);
  meta_data.declare_part_subset(spheres_part, e_spheres_part);
  meta_data.declare_part_subset(spheres_part, h_spheres_part);
  stk::io::put_assembly_io_part_attribute(spheres_part);
  stk::io::put_io_part_attribute(e_spheres_part);
  stk::io::put_io_part_attribute(h_spheres_part);

  auto &hp1_part = meta_data.declare_part("HP1", stk::topology::ELEM_RANK);
  auto &left_hp1_part = meta_data.declare_part_with_topology("LEFT_HP1", beam2_top);
  auto &doubly_hp1_h_part = meta_data.declare_part_with_topology("DOUBLY_HP1_H", beam2_top);
  auto &doubly_hp1_bs_part = meta_data.declare_part_with_topology("DOUBLY_HP1_BS", beam2_top);
  meta_data.declare_part_subset(hp1_part, left_hp1_part);
  meta_data.declare_part_subset(hp1_part, doubly_hp1_h_part);
  meta_data.declare_part_subset(hp1_part, doubly_hp1_bs_part);
  stk::io::put_assembly_io_part_attribute(hp1_part);
  stk::io::put_io_part_attribute(left_hp1_part);
  stk::io::put_io_part_attribute(doubly_hp1_h_part);
  stk::io::put_io_part_attribute(doubly_hp1_bs_part);

  auto &binding_sites_part = meta_data.declare_part_with_topology("BIND_SITES", node_top);
  stk::io::put_io_part_attribute(
      binding_sites_part);  // This is a node part and might not be compatible with IO unless we add special attributes.

  auto &backbone_segs_part = meta_data.declare_part("BACKBONE_SEGMENTS", stk::topology::ELEM_RANK);
  auto &ee_segs_part = meta_data.declare_part_with_topology("EE_SEGMENTS", beam2_top);
  auto &eh_segs_part = meta_data.declare_part_with_topology("EH_SEGMENTS", beam2_top);
  auto &hh_segs_part = meta_data.declare_part_with_topology("HH_SEGMENTS", beam2_top);
  meta_data.declare_part_subset(backbone_segs_part, ee_segs_part);
  meta_data.declare_part_subset(backbone_segs_part, eh_segs_part);
  meta_data.declare_part_subset(backbone_segs_part, hh_segs_part);
  stk::io::put_assembly_io_part_attribute(backbone_segs_part);
  stk::io::put_io_part_attribute(ee_segs_part);
  stk::io::put_io_part_attribute(eh_segs_part);
  stk::io::put_io_part_attribute(hh_segs_part);

  // Declare all fields
  auto node_rank = stk::topology::NODE_RANK;
  auto element_rank = stk::topology::ELEMENT_RANK;
  auto &node_coords_field = meta_data.declare_field<double>(node_rank, "COORDS");
  auto &node_velocity_field = meta_data.declare_field<double>(node_rank, "VELOCITY");
  auto &node_force_field = meta_data.declare_field<double>(node_rank, "FORCE");
  auto &node_collision_velocity_field = meta_data.declare_field<double>(node_rank, "COLLISION_VELOCITY");
  auto &node_collision_force_field = meta_data.declare_field<double>(node_rank, "COLLISION_FORCE");
  auto &node_rng_field = meta_data.declare_field<unsigned>(node_rank, "RNG_COUNTER");
  auto &node_displacement_since_last_rebuild_field = meta_data.declare_field<double>(node_rank, "OUR_DISPLACEMENT");

  auto &elem_hydrodynamic_radius_field = meta_data.declare_field<double>(element_rank, "HYDRODYNAMIC_RADIUS");
  auto &elem_collision_radius_field = meta_data.declare_field<double>(element_rank, "COLLISION_RADIUS");
  auto &elem_binding_radius_field = meta_data.declare_field<double>(element_rank, "BINDING_RADIUS");

  auto &elem_spring_constant_field = meta_data.declare_field<double>(element_rank, "SPRING_CONSTANT");
  auto &elem_spring_r0_field = meta_data.declare_field<double>(element_rank, "SPRING_R0");

  auto &elem_requires_endpoint_correction_field =
      meta_data.declare_field<unsigned>(element_rank, "REQUIRES_ENDPOINT_CORRECTION");

  auto &elem_binding_rates_field = meta_data.declare_field<double>(element_rank, "BINDING_RATES");
  auto &elem_unbinding_rates_field = meta_data.declare_field<double>(element_rank, "UNBINDING_RATES");
  auto &elem_rng_field = meta_data.declare_field<unsigned>(element_rank, "RNG_COUNTER");
  auto &elem_chain_id_field = meta_data.declare_field<unsigned>(element_rank, "CHAIN_ID");

  auto &elem_e_state_field = meta_data.declare_field<unsigned>(element_rank, "EUCHROMATIN_STATE");
  auto &elem_e_state_change_next_time_field =
      meta_data.declare_field<unsigned>(element_rank, "EUCHROMATIN_STATE_CHANGE_NEXT_TIME");
  auto &elem_e_state_time_field = meta_data.declare_field<unsigned>(element_rank, "EUCHROMATIN_STATE_CHANGE_TIME");

  auto transient_role = Ioss::Field::TRANSIENT;
  stk::io::set_field_role(node_velocity_field, transient_role);
  stk::io::set_field_role(node_force_field, transient_role);
  stk::io::set_field_role(node_collision_velocity_field, transient_role);
  stk::io::set_field_role(node_collision_force_field, transient_role);
  stk::io::set_field_role(node_rng_field, transient_role);
  stk::io::set_field_role(node_displacement_since_last_rebuild_field, transient_role);
  stk::io::set_field_role(elem_hydrodynamic_radius_field, transient_role);
  stk::io::set_field_role(elem_collision_radius_field, transient_role);
  stk::io::set_field_role(elem_binding_radius_field, transient_role);
  stk::io::set_field_role(elem_spring_constant_field, transient_role);
  stk::io::set_field_role(elem_spring_r0_field, transient_role);
  stk::io::set_field_role(elem_binding_rates_field, transient_role);
  stk::io::set_field_role(elem_unbinding_rates_field, transient_role);
  stk::io::set_field_role(elem_rng_field, transient_role);
  stk::io::set_field_role(elem_chain_id_field, transient_role);
  stk::io::set_field_role(elem_e_state_field, transient_role);
  stk::io::set_field_role(elem_e_state_change_next_time_field, transient_role);
  stk::io::set_field_role(elem_e_state_time_field, transient_role);

  auto scalar_io_type = stk::io::FieldOutputType::SCALAR;
  auto vector_3d_io_type = stk::io::FieldOutputType::VECTOR_3D;
  stk::io::set_field_output_type(node_velocity_field, vector_3d_io_type);
  stk::io::set_field_output_type(node_force_field, vector_3d_io_type);
  stk::io::set_field_output_type(node_collision_velocity_field, vector_3d_io_type);
  stk::io::set_field_output_type(node_collision_force_field, vector_3d_io_type);
  stk::io::set_field_output_type(node_rng_field, scalar_io_type);
  stk::io::set_field_output_type(elem_hydrodynamic_radius_field, scalar_io_type);
  stk::io::set_field_output_type(elem_collision_radius_field, scalar_io_type);
  stk::io::set_field_output_type(elem_binding_radius_field, scalar_io_type);
  stk::io::set_field_output_type(elem_spring_constant_field, scalar_io_type);
  stk::io::set_field_output_type(elem_spring_r0_field, scalar_io_type);
  stk::io::set_field_output_type(elem_binding_rates_field, stk::io::FieldOutputType::VECTOR_2D);  // These aren't
  // really Vector2Ds.
  stk::io::set_field_output_type(elem_unbinding_rates_field, stk::io::FieldOutputType::VECTOR_2D);
  stk::io::set_field_output_type(elem_rng_field, scalar_io_type);
  stk::io::set_field_output_type(elem_chain_id_field, scalar_io_type);
  stk::io::set_field_output_type(elem_e_state_field, scalar_io_type);
  stk::io::set_field_output_type(elem_e_state_change_next_time_field, scalar_io_type);
  stk::io::set_field_output_type(elem_e_state_time_field, scalar_io_type);

  // Sew it all together. Start off fields as uninitialized.
  // Give all nodes and elements a random number generator counter.
  stk::mesh::put_field_on_mesh(node_coords_field, meta_data.universal_part(), 3, nullptr);
  stk::mesh::put_field_on_mesh(node_rng_field, meta_data.universal_part(), 1, nullptr);
  stk::mesh::put_field_on_mesh(elem_rng_field, meta_data.universal_part(), 1, nullptr);

  // Heterochromatin and euchromatin spheres are used for hydrodynamics.
  // They move and have forces applied to them. If brownian motion is enabled, they will have a
  // stochastic velocity. Heterochromatin spheres are considered for hp1 binding.
  stk::mesh::put_field_on_mesh(node_velocity_field, spheres_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(node_force_field, spheres_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(node_collision_velocity_field, spheres_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(node_collision_force_field, spheres_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(node_displacement_since_last_rebuild_field, spheres_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(elem_hydrodynamic_radius_field, spheres_part, 1, nullptr);
  stk::mesh::put_field_on_mesh(elem_chain_id_field, spheres_part, 1, nullptr);

  // Backbone segs apply spring forces to their nodes and are used for collisions.
  // The difference between ee, eh, and hh segs is that ee segs can exert an active dipole.
  stk::mesh::put_field_on_mesh(node_force_field, backbone_segs_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(elem_spring_constant_field, backbone_segs_part, 1, nullptr);
  stk::mesh::put_field_on_mesh(elem_spring_r0_field, backbone_segs_part, 1, nullptr);
  stk::mesh::put_field_on_mesh(elem_collision_radius_field, backbone_segs_part, 1, nullptr);
  stk::mesh::put_field_on_mesh(elem_chain_id_field, backbone_segs_part, 1, nullptr);
  stk::mesh::put_field_on_mesh(elem_requires_endpoint_correction_field, backbone_segs_part, 1, nullptr);

  // HP1 crosslinkers are used for binding/unbinding and apply forces to their nodes.
  stk::mesh::put_field_on_mesh(node_force_field, hp1_part, 3, nullptr);
  stk::mesh::put_field_on_mesh(elem_binding_rates_field, hp1_part, 2, nullptr);
  stk::mesh::put_field_on_mesh(elem_unbinding_rates_field, hp1_part, 2, nullptr);
  stk::mesh::put_field_on_mesh(elem_binding_radius_field, hp1_part, 1, nullptr);

  // Bind sites are use for binding/unbinding. They are merely a point in space and have no
  //   inherent field besides node_coords.

  // That's it for the mesh. Commit it's structure and create the bulk data.
  meta_data.commit();

  // Perform restart (optional)
  bool restart_performed = false;
  if (!restart_performed) {
    /* Declare the chromatin and HP1
    //  E : euchromatin spheres
    //  H : heterochromatin spheres
    //  | : crosslinkers
    // ---: backbone springs/backbone segments
    //
    //  |   |                           |   |
    //  H---H---E---E---E---E---E---E---H---H
    //
    // The actual connectivity looks like this:
    //  n : node, s : segment and or spring, c : crosslinker
    //
    // c1_      c3_       c5_       c7_
    // | /      | /       | /       | /
    // n1       n3        n5        n7
    //  \      /  \      /  \      /
    //   s1   s2   s3   s4   s5   s6
    //    \  /      \  /      \  /
    //     n2        n4        n6
    //     | \       | \       | \
    //     c2⎻       c4⎻       c6⎻
    //
    // If you look at this long enough, the pattern is clear.
    //  - One less segment than nodes.
    //  - Same number of crosslinkers as heterochromatin nodes.
    //  - Segment i connects to nodes i and i+1.
    //  - Crosslinker i connects to nodes i and i.
    //
    // We need to use this information to populate the node and element info vectors.
    // Mundy will handle passing off this information to the bulk data. Just make sure that all
    // MPI ranks contain the same node and element info. This way, we can determine which nodes
    // should become shared.
    //
    // Rules (non-exhaustive):
    //  - Neither nodes nor elements need to have parts or fields.
    //  - The rank and type of the fields must be consistent. You can't pass an element field to a node,
    //    nor can you set the value of a field to a different type or size than it was declared as.
    //  - The owner of a node must be the same as one of the elements that connects to it.
    //  - A node connected to an element not on the same rank as the node will be shared with the owner of the
    element.
    //  - Field/Part names are case-sensitive but don't attempt to declare "field_1" and "Field_1" as if
    //    that will give two different fields since STKIO will not be able to distinguish between them.
    //  - A (non-zero) negative node id in the element connection list can be used to indicate that a node should be
    left unassigned.
    //  - All parts need to be able to contain an element of the given topology.
    */

    // Fill the declare entities helper
    mundy::mesh::DeclareEntitiesHelper dec_helper;
    // Start the counts at 1 due to STK not being 0-based indexing
    size_t node_count = 1;
    size_t element_count = 1;

    // Setup the periphery bind sites
    {
      const std::string bind_sites_type = periphery_binding_params.get<std::string>("bind_sites_type");
      if (bind_sites_type == "RANDOM") {
        const size_t num_bind_sites = periphery_binding_params.get<size_t>("num_bind_sites");
        const std::string periphery_shape = periphery_binding_params.get<std::string>("shape");
        openrand::Philox rng(0, 0);
        if (periphery_shape == "SPHERE") {
          const double radius = periphery_binding_params.get<double>("radius");

          for (size_t i = 0; i < num_bind_sites; i++) {
            // Generate a random point on the unit sphere
            const double u1 = rng.rand<double>();
            const double u2 = rng.rand<double>();
            const double theta = 2.0 * M_PI * u1;
            const double phi = std::acos(2.0 * u2 - 1.0);
            double node_coords[3] = {radius * std::sin(phi) * std::cos(theta),  //
                                     radius * std::sin(phi) * std::sin(theta),  //
                                     radius * std::cos(phi)};

            // Declare the node
            dec_helper.create_node()
                .owning_proc(0)                 //
                .id(node_count)                 //
                .add_part(&binding_sites_part)  //
                .add_field_data<double>(&node_coords_field, {node_coords[0], node_coords[1], node_coords[2]});
            node_count++;
          }
        } else if (periphery_shape == "ELLIPSOID") {
          const double a = periphery_binding_params.get<double>("axis_radius1");
          const double b = periphery_binding_params.get<double>("axis_radius2");
          const double c = periphery_binding_params.get<double>("axis_radius3");
          const double inv_mu_max = 1.0 / std::max({b * c, a * c, a * b});
          openrand::Philox rng(0, 0);
          auto keep = [&a, &b, &c, &inv_mu_max, &rng](double x, double y, double z) {
            const double mu_xyz =
                std::sqrt((b * c * x) * (b * c * x) + (a * c * y) * (a * c * y) + (a * b * z) * (a * b * z));
            return inv_mu_max * mu_xyz > rng.rand<double>();
          };

          for (size_t i = 0; i < num_bind_sites; i++) {
            // Rejection sampling to place the periphery binding sites
            double node_coords[3];
            while (true) {
              // Generate a random point on the unit sphere
              const double u1 = rng.rand<double>();
              const double u2 = rng.rand<double>();
              const double theta = 2.0 * M_PI * u1;
              const double phi = std::acos(2.0 * u2 - 1.0);
              node_coords[0] = std::sin(phi) * std::cos(theta);
              node_coords[1] = std::sin(phi) * std::sin(theta);
              node_coords[2] = std::cos(phi);

              // Keep this point with probability proportional to the surface area element
              if (keep(node_coords[0], node_coords[1], node_coords[2])) {
                // Pushforward the point to the ellipsoid
                node_coords[0] *= a;
                node_coords[1] *= b;
                node_coords[2] *= c;
                break;
              }
            }

            // Declare the node
            dec_helper.create_node()
                .owning_proc(0)                 //
                .id(node_count)                 //
                .add_part(&binding_sites_part)  //
                .add_field_data<double>(&node_coords_field, {node_coords[0], node_coords[1], node_coords[2]});
            node_count++;
          }
        }
      }
    }

    // Setup the chromatin fibers
    {
      const size_t num_chromosomes = sim_params.get<size_t>("num_chromosomes");
      // const size_t num_he_blocks = sim_params.get<size_t>("num_hetero_euchromatin_blocks");
      // const size_t num_h_per_block = sim_params.get<size_t>("num_heterochromatin_per_block");
      // const size_t num_e_per_block = sim_params.get<size_t>("num_euchromatin_per_block");
      // const size_t num_nodes_per_chromosome = num_he_blocks * (num_h_per_block + num_e_per_block);
      // const double segment_length = sim_params.get<double>("initial_chromosome_separation");

      std::vector<std::vector<mundy::geom::Point<double>>> all_chromosome_positions(num_chromosomes);
      std::vector<std::vector<std::string>> all_chromosome_regions(num_chromosomes);
      // if (sim_params.get<std::string>("initialization_type") == "GRID") {
      // all_chromosome_positions = get_chromosome_positions_grid(
      //     static_cast<unsigned>(num_chromosomes), static_cast<unsigned>(num_nodes_per_chromosome), segment_length);
      // } else if (sim_params.get<std::string>("initialization_type") == "RANDOM_UNIT_CELL") {
      // auto domain_low = sim_params.get<Teuchos::Array<double>>("domain_low");
      // auto domain_high = sim_params.get<Teuchos::Array<double>>("domain_high");
      // all_chromosome_positions = get_chromosome_positions_random_unit_cell(
      //     static_cast<unsigned>(num_chromosomes), static_cast<unsigned>(num_nodes_per_chromosome), segment_length,
      //     domain_low.getRawPtr(), domain_high.getRawPtr());
      // } else if (sim_params.get<std::string>("initialization_type") == "HILBERT_RANDOM_UNIT_CELL") {
      // auto domain_low = sim_params.get<Teuchos::Array<double>>("domain_low");
      // auto domain_high = sim_params.get<Teuchos::Array<double>>("domain_high");
      // all_chromosome_positions = get_chromosome_positions_hilbert_random_unit_cell(
      //     static_cast<unsigned>(num_chromosomes), static_cast<unsigned>(num_nodes_per_chromosome), segment_length,
      //     domain_low.getRawPtr(), domain_high.getRawPtr());
      // } else
      if (sim_params.get<std::string>("initialization_type") == "FROM_DAT") {
        get_chromosome_positions_regions_from_file(sim_params.get<std::string>("initialize_from_dat_filename"),
                                                   static_cast<unsigned>(num_chromosomes), &all_chromosome_positions,
                                                   &all_chromosome_regions);
      } else {
        MUNDY_THROW_REQUIRE(false, std::invalid_argument, "Invalid initialization type.");
      }

      // XXX: Fix the different regional encoding later!
      // Determine the chromosome regional map for use later, and load from a file if we need to...
      // std::vector<std::string> chromosome_regional_map;
      // if (specify_chromosome_layout) {
      //   std::ifstream layout_file(specify_chromosome_file_);
      //   MUNDY_THROW_REQUIRE(layout_file.is_open(), std::runtime_error,
      //                       "The chromosome layout file " + specify_chromosome_file_ + " could not be opened.");
      //   std::string line;
      //   while (std::getline(layout_file, line)) {
      //     // Remove whitespace from the line
      //     line.erase(remove_if(line.begin(), line.end(), isspace), line.end());
      //     if (!line.empty()) {
      //       chromosome_regional_map.push_back(line);
      //     }
      //   }
      // }

      // Configure the system based on looping over the nodes/sphere for each individual vertex, and adding segments,
      // HP1, where necessary.

      for (size_t f = 0; f < num_chromosomes; f++) {
        // We already have the chromosome positions and regional identities by this point, so use them directly when
        // setting the spheres and segments. Get the current size of the chromosome based on the length of the vector
        // holding all of the information and loop over it.
        auto chromosome_positions = all_chromosome_positions[f];
        auto chromosome_regions = all_chromosome_regions[f];

        for (size_t sphere_local_idx = 0; sphere_local_idx < chromosome_positions.size(); ++sphere_local_idx) {
          // Keep track of variables that may change due to indexing in this segment.
          auto current_node_count = node_count;
          auto current_element_count = element_count;
          // Create the node
          dec_helper.create_node()
              .owning_proc(0)
              .id(node_count)
              .add_field_data<unsigned>(&node_rng_field, 0u)                                            //
              .add_field_data<double>(&node_coords_field, {chromosome_positions[sphere_local_idx][0],   //
                                                           chromosome_positions[sphere_local_idx][1],   //
                                                           chromosome_positions[sphere_local_idx][2]})  //
              .add_field_data<double>(&node_displacement_since_last_rebuild_field, {0.0, 0.0, 0.0})     //
              .add_field_data<double>(&node_velocity_field, {0.0, 0.0, 0.0})                            //
              .add_field_data<double>(&node_force_field, {0.0, 0.0, 0.0})                               //
              .add_field_data<double>(&node_collision_velocity_field, {0.0, 0.0, 0.0})                  //
              .add_field_data<double>(&node_collision_force_field, {0.0, 0.0, 0.0});
          node_count++;

          // Create the sphere
          auto sphere_element = dec_helper.create_element();
          sphere_element
              .owning_proc(0)                     //
              .id(element_count)                  //
              .topology(stk::topology::PARTICLE)  //
              .nodes({current_node_count})        // this refers to the original node index
              .add_field_data<double>(&elem_hydrodynamic_radius_field,
                                      sim_params.get<double>("backbone_sphere_hydrodynamic_radius"))
              .add_field_data<unsigned>(&elem_chain_id_field, f);
          if (chromosome_regions[sphere_local_idx] == "H") {
            sphere_element.add_part(&h_spheres_part);
          } else if (chromosome_regions[sphere_local_idx] == "E") {
            sphere_element.add_part(&e_spheres_part);
          } else {
            MUNDY_THROW_REQUIRE(false, std::invalid_argument,
                                "Invalid chromosome region: " + chromosome_regions[sphere_local_idx]);
          }
          element_count++;

          // Create the backbone segment (unless it is the next to last node in the fiber)
          if (sphere_local_idx < chromosome_positions.size() - 1) {
            auto segment = dec_helper.create_element();
            segment
                .owning_proc(0)                   //
                .id(element_count)                //
                .topology(stk::topology::BEAM_2)  //
                .add_part(&backbone_segs_part)    //
                .nodes({current_node_count,
                        current_node_count + 1})  // this does rely on the nodes being declared sequentially!
                .add_field_data<unsigned>(&elem_chain_id_field, f)
                .add_field_data<unsigned>(&elem_rng_field, 0u);
            element_count++;

            if (sim_params.get<bool>("enable_backbone_collision")) {
              segment.add_field_data<double>(&elem_collision_radius_field,
                                             backbone_collision_params.get<double>("backbone_sphere_collision_radius"));
            }

            if (sim_params.get<bool>("enable_backbone_springs")) {
              segment
                  .add_field_data<double>(&elem_spring_constant_field,
                                          backbone_springs_params.get<double>("spring_constant"))  //
                  .add_field_data<double>(&elem_spring_r0_field, backbone_springs_params.get<double>("spring_r0"));
            }

            // Determine if we need an endpoint correction
            if (sim_params.get<bool>("enable_backbone_collision") || sim_params.get<bool>("enable_backbone_springs")) {
              if (sphere_local_idx == 0 || sphere_local_idx == chromosome_positions.size() - 2) {
                segment.add_field_data<unsigned>(&elem_requires_endpoint_correction_field, 1);
              } else {
                segment.add_field_data<unsigned>(&elem_requires_endpoint_correction_field, 0);
              }
            }

            // Determine if the segment is hh or eh
            if (chromosome_regions[sphere_local_idx] == "H" && chromosome_regions[sphere_local_idx + 1] == "H") {
              segment.add_part(&hh_segs_part);
            } else if (chromosome_regions[sphere_local_idx] == "E" && chromosome_regions[sphere_local_idx + 1] == "E") {
              segment.add_part(&ee_segs_part);
            } else {
              segment.add_part(&eh_segs_part);
            }
          }
        }
      }
    }

    dec_helper.check_consistency(bulk_data);

    // Declare the entities
    bulk_data.modification_begin();
    dec_helper.declare_entities(bulk_data);
    bulk_data.modification_end();

    // Write the mesh to file
    size_t step = 1;  // Step = 0 doesn't write out fields...
    stk::io::write_mesh_with_fields("ngp_hp1.exo", bulk_data, step);
  }

  // XXX Dump all of the mesh info
  stk::mesh::impl::dump_all_mesh_info(bulk_data, std::cout);
}

}  // namespace ngphp1

}  // namespace mundy

///////////////////////////
// Main program          //
///////////////////////////
int main(int argc, char **argv) {
  // Initialize MPI and Kokkos
  stk::parallel_machine_init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  Kokkos::print_configuration(std::cout);

  mundy::ngphp1::run_main(argc, argv);

  Kokkos::finalize();
  stk::parallel_machine_finalize();

  return 0;
}