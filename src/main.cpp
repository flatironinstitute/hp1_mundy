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

namespace mundy {

// Structures for template accessors and aggregates
struct COORDS {};
struct FORCE {};
struct RNG_COUNTER {};
struct RADIUS {};
struct CHAINID {};
struct LINKED_ENTITIES {};

void run_main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <num_time_steps>" << std::endl;
    return;
  }
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

  // Setup the STK mesh (boiler plate)
  MeshBuilder mesh_builder(MPI_COMM_WORLD);
  mesh_builder
      .set_spatial_dimension(3)  //
      .set_entity_rank_names({"NODE", "EDGE", "FACE", "ELEMENT", "CONSTRAINT"});
  std::shared_ptr<MetaData> meta_data_ptr = mesh_builder.create_meta_data();
  meta_data_ptr->use_simple_fields();  // Depreciated in newer versions of STK as they have finished the transition
                                       // to all fields are simple.
  meta_data_ptr->set_coordinate_field_name("COORDS");
  std::shared_ptr<BulkData> bulk_data_ptr = mesh_builder.create_bulk_data(meta_data_ptr);
  MetaData& meta_data = *meta_data_ptr;
  BulkData& bulk_data = *bulk_data_ptr;

  // Setup the link data (boilerplate)
  LinkMetaData link_meta_data = declare_link_meta_data(meta_data, "ALL_LINKS", NODE_RANK);
  LinkData link_data = declare_link_data(bulk_data, link_meta_data);

  // Declare parts
  //   Sphere parts: PARTICLE_TOPOLOGY
  Part& chromatin_sphere_part = meta_data.declare_part_with_topology("CHROMATIN_SPHERE_PART", stk::topology::PARTICLE);
  Part& slink_part = link_meta_data.declare_link_part("SURFACE_LINKS", 2 /* our dimensionality */);

  // Declare all fields
  // Chromatin spheres PARTICLE top
  //  Node fields: (Only what is logical to potentially share with other entities that connect to the node)
  //   - COORDS
  //   - FORCE
  //   - RNG_COUNTER
  //
  //  Elem fields:
  //   - RADIUS
  //   - CHAINID
  //   - RNG_COUNTER
  auto& node_coords_field = meta_data.declare_field<double>(NODE_RANK, "COORDS");
  auto& node_force_field = meta_data.declare_field<double>(NODE_RANK, "FORCE");
  auto& node_rng_counter_field = meta_data.declare_field<unsigned>(NODE_RANK, "RNG_COUNTER");

  auto& elem_radius_field = meta_data.declare_field<double>(ELEM_RANK, "RADIUS");
  auto& elem_chainid_field = meta_data.declare_field<unsigned>(ELEM_RANK, "CHAINID");
  auto& elem_rng_counter_field = meta_data.declare_field<unsigned>(ELEM_RANK, "RNG_COUNTER");

  // Set up fields on the mesh
  // Universal fields
  stk::mesh::put_field_on_mesh(node_coords_field, meta_data.universal_part(), 3, nullptr);
  stk::mesh::put_field_on_mesh(node_force_field, meta_data.universal_part(), 3, nullptr);
  stk::mesh::put_field_on_mesh(node_rng_counter_field, meta_data.universal_part(), 1, nullptr);
  stk::mesh::put_field_on_mesh(elem_rng_counter_field, meta_data.universal_part(), 1, nullptr);
  // Assemble the chromatin_sphere part
  stk::mesh::put_field_on_mesh(elem_radius_field, chromatin_sphere_part, 1, nullptr);
  stk::mesh::put_field_on_mesh(elem_chainid_field, chromatin_sphere_part, 1, nullptr);

  // Declare IO stuff (boilerplate)
  // Part IO
  stk::io::put_io_part_attribute(chromatin_sphere_part);
  // NODE fields
  stk::io::set_field_role(node_coords_field, Ioss::Field::MESH);
  stk::io::set_field_role(node_force_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(node_rng_counter_field, Ioss::Field::TRANSIENT);
  // ELEM fields
  stk::io::set_field_role(elem_radius_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(elem_chainid_field, Ioss::Field::TRANSIENT);
  stk::io::set_field_role(elem_rng_counter_field, Ioss::Field::TRANSIENT);
  // Output types
  stk::io::set_field_output_type(node_coords_field, stk::io::FieldOutputType::VECTOR_3D);
  stk::io::set_field_output_type(node_force_field, stk::io::FieldOutputType::VECTOR_3D);
  stk::io::set_field_output_type(node_rng_counter_field, stk::io::FieldOutputType::SCALAR);
  stk::io::set_field_output_type(elem_radius_field, stk::io::FieldOutputType::SCALAR);
  stk::io::set_field_output_type(elem_chainid_field, stk::io::FieldOutputType::SCALAR);
  stk::io::set_field_output_type(elem_rng_counter_field, stk::io::FieldOutputType::SCALAR);

  // Commit the meta data
  meta_data.commit();

  // Build our accessors and aggregates
  // Create accessors
  auto node_coord_accessor = Vector3FieldComponent(node_coords_field);
  auto node_force_accessor = Vector3FieldComponent(node_force_field);
  auto node_rng_counter_accessor = ScalarFieldComponent(node_rng_counter_field);
  auto elem_radius_accessor = ScalarFieldComponent(elem_radius_field);
  auto elem_chainid_accessor = ScalarFieldComponent(elem_chainid_field);
  auto elem_rng_counter_accessor = ScalarFieldComponent(elem_rng_counter_field);

  auto linked_entities_accessor = FieldComponent(link_meta_data.linked_entities_field());
  // Create aggregates
  auto chromatin_sphere_agg = make_aggregate<stk::topology::PARTICLE>(bulk_data, chromatin_sphere_part)
                                  .add_component<COORDS, NODE_RANK>(node_coord_accessor)
                                  .add_component<FORCE, NODE_RANK>(node_force_accessor)
                                  .add_component<RNG_COUNTER, NODE_RANK>(node_rng_counter_accessor)
                                  .add_component<RADIUS, ELEM_RANK>(elem_radius_accessor)
                                  .add_component<CHAINID, ELEM_RANK>(elem_chainid_accessor)
                                  .add_component<RNG_COUNTER, ELEM_RANK>(elem_rng_counter_accessor);
  auto slink_agg = make_ranked_aggregate<NODE_RANK>(bulk_data, slink_part)
                       .add_component<LINKED_ENTITIES, NODE_RANK>(linked_entities_accessor);

  // Load simulation parameters (XXX)

  // Create entities
  DeclareEntitiesHelper dec_helper;
  unsigned current_node_entity_id = 1;
  unsigned current_elem_entity_id = 1;
  // Create the chromatin spheres at both the node and element level together
  for (unsigned i = 1; i <= 5; ++i) {
    // Create the NODE entity
    dec_helper.create_node()
        .owning_proc(0)
        .id(current_node_entity_id)
        .add_field_data<double>(&node_coords_field, {0.0, 0.0, 0.0})
        .add_field_data<double>(&node_force_field, {0.0, 0.0, 0.0})
        .add_field_data<unsigned>(&node_rng_counter_field, {0});
    // Create the ELEM entity
    dec_helper.create_element()
        .owning_proc(0)
        .id(current_elem_entity_id)
        .topology(stk::topology::PARTICLE)
        .add_part(&chromatin_sphere_part)
        .nodes({current_node_entity_id})
        .add_field_data<unsigned>(&elem_chainid_field, {1})
        .add_field_data<double>(&elem_radius_field, {0.5})
        .add_field_data<unsigned>(&elem_rng_counter_field, {0});
    // Increment counters at the end of the loop
    current_node_entity_id++;
    current_elem_entity_id++;
  }

  // Declare the entities
  dec_helper.check_consistency(bulk_data);
  bulk_data.modification_begin();
  dec_helper.declare_entities(bulk_data);
  bulk_data.modification_end();

  // XXX Dump all of the mesh info
  stk::mesh::impl::dump_all_mesh_info(bulk_data, std::cout);
}

}  // namespace mundy

///////////////////////////
// Main program          //
///////////////////////////
int main(int argc, char** argv) {
  // Initialize MPI and Kokkos
  stk::parallel_machine_init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  Kokkos::print_configuration(std::cout);

  mundy::run_main(argc, argv);

  Kokkos::finalize();
  stk::parallel_machine_finalize();

  return 0;
}