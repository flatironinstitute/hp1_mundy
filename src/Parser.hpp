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

#ifndef MUNDY_NGHP1_HP1_PARAM_PARSER_HPP_
#define MUNDY_NGHP1_HP1_PARAM_PARSER_HPP_

// External
#include <fmt/format.h>
#include <fmt/ostream.h>

// C++ core
#include <iostream>
#include <limits>

// Teuchos
#include <Teuchos_CommandLineProcessor.hpp>  // for Teuchos::CommandLineProcessor
#include <Teuchos_ParameterList.hpp>         // for Teuchos::ParameterList

// Mundy core
#include <mundy_core/OurAnyNumberParameterEntryValidator.hpp>  // for mundy::core::OurAnyNumberParameterEntryValidator
#include <mundy_core/throw_assert.hpp>                         // for MUNDY_THROW_ASSERT

namespace mundy {

namespace ngphp1 {

//! \name Simulation setup/run
//@{

struct HP1ParamParser {
  void print_help_message() {
    std::cout << "#############################################################################################"
              << std::endl;
    std::cout << "To run this code, please pass in --params=<input.yaml> as a command line argument." << std::endl;
    std::cout << std::endl;
    std::cout << "Note, all parameters and sublists in input.yaml must be contained in a single top-level list."
              << std::endl;
    std::cout << "Such as:" << std::endl;
    std::cout << std::endl;
    std::cout << "HP1:" << std::endl;
    std::cout << "  num_time_steps: 1000" << std::endl;
    std::cout << "  timestep_size: 1e-6" << std::endl;
    std::cout << "#############################################################################################"
              << std::endl;
    std::cout << "The valid parameters that can be set in the input file are:" << std::endl;
    Teuchos::ParameterList valid_params = get_valid_params();

    auto print_options =
        Teuchos::ParameterList::PrintOptions().showTypes(false).showDoc(true).showDefault(true).showFlags(false).indent(
            1);
    valid_params.print(std::cout, print_options);
    std::cout << "#############################################################################################"
              << std::endl;
  }

  Teuchos::ParameterList parse(int argc, char **argv) {
    // Parse the command line options to find the input filename
    Teuchos::CommandLineProcessor cmdp(false, true);
    cmdp.setOption("params", &input_parameter_filename_, "The name of the input file.");

    Teuchos::CommandLineProcessor::EParseCommandLineReturn parse_result = cmdp.parse(argc, argv);
    if (parse_result == Teuchos::CommandLineProcessor::PARSE_HELP_PRINTED) {
      print_help_message();

      // Safely exit the program
      // If we print the help message, we don't need to do anything else.
      exit(0);

    } else if (parse_result != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
      throw std::invalid_argument("Failed to parse the command line arguments.");
    }

    // Read, validate, and parse in the parameters from the parameter list.
    try {
      Teuchos::ParameterList param_list = *Teuchos::getParametersFromYamlFile(input_parameter_filename_);
      return parse(param_list);
    } catch (const std::exception &e) {
      std::cerr << "ERROR: Failed to read the input parameter file." << std::endl;
      std::cerr << "During read, the following error occurred: " << e.what() << std::endl;
      std::cerr << "NOTE: This can happen for any number of reasons. Check that the file exists and contains the "
                   "expected parameters."
                << std::endl;
      throw e;
    }

    return Teuchos::ParameterList();
  }

  Teuchos::ParameterList parse(const Teuchos::ParameterList &param_list) {
    // Validate the parameters and set the defaults.
    Teuchos::ParameterList valid_param_list = param_list;
    valid_param_list.validateParametersAndSetDefaults(get_valid_params());
    check_invariants(valid_param_list);
    dump_parameters(valid_param_list);
    return valid_param_list;
  }

  void check_invariants(const Teuchos::ParameterList &valid_param_list) {
    // Check the sim params
    const auto &sim_params = valid_param_list.sublist("sim");
    MUNDY_THROW_REQUIRE(sim_params.get<double>("timestep_size") > 0, std::invalid_argument,
                        "timestep_size must be greater than 0.");
    MUNDY_THROW_REQUIRE(sim_params.get<double>("viscosity") > 0, std::invalid_argument,
                        "viscosity must be greater than 0.");
    MUNDY_THROW_REQUIRE(sim_params.get<double>("initial_chromosome_separation") >= 0, std::invalid_argument,
                        "initial_chromosome_separation must be greater than or equal to 0.");
    MUNDY_THROW_REQUIRE(sim_params.get<bool>("enable_periphery_hydrodynamics")
                            ? sim_params.get<bool>("enable_backbone_n_body_hydrodynamics")
                            : true,
                        std::invalid_argument,
                        "Periphery hydrodynamics requires backbone hydrodynamics to be enabled.");

    MUNDY_THROW_REQUIRE(sim_params.get<std::string>("initialization_type") == "GRID" ||
                            sim_params.get<std::string>("initialization_type") == "RANDOM_UNIT_CELL" ||
                            sim_params.get<std::string>("initialization_type") == "OVERLAP_TEST" ||
                            sim_params.get<std::string>("initialization_type") == "HILBERT_RANDOM_UNIT_CELL" ||
                            sim_params.get<std::string>("initialization_type") == "USHAPE_TEST" ||
                            sim_params.get<std::string>("initialization_type") == "FROM_EXO" ||
                            sim_params.get<std::string>("initialization_type") == "FROM_DAT",
                        std::invalid_argument,
                        fmt::format("Invalid initialization_type ({}). Valid options are GRID, RANDOM_UNIT_CELL, "
                                    "OVERLAP_TEST, HILBERT_RANDOM_UNIT_CELL, USHAPE_TEST, FROM_EXO, FROM_DAT.",
                                    sim_params.get<std::string>("initialization_type")));

    if (sim_params.get<bool>("enable_backbone_springs")) {
      const auto &backbone_spring_params = valid_param_list.sublist("backbone_springs");
      MUNDY_THROW_REQUIRE(backbone_spring_params.get<std::string>("spring_type") == "HOOKEAN" ||
                              backbone_spring_params.get<std::string>("spring_type") == "FENE",
                          std::invalid_argument,
                          fmt::format("Invalid spring_type ({}). Valid options are HOOKEAN and FENE.",
                                      backbone_spring_params.get<std::string>("spring_type")));
      MUNDY_THROW_REQUIRE(backbone_spring_params.get<double>("spring_constant") >= 0, std::invalid_argument,
                          "spring_constant must be non-negative.");
      MUNDY_THROW_REQUIRE(backbone_spring_params.get<double>("spring_r0") >= 0, std::invalid_argument,
                          "max_spring_length must be non-negative.");
    }

    // Check the periphery_hydro params
    if (sim_params.get<bool>("enable_periphery_hydrodynamics")) {
      const auto &periphery_hydro_params = valid_param_list.sublist("periphery_hydro");
      std::string periphery_hydro_shape = periphery_hydro_params.get<std::string>("shape");
      std::string periphery_hydro_quadrature = periphery_hydro_params.get<std::string>("quadrature");
      if (periphery_hydro_quadrature == "GAUSS_LEGENDRE") {
        double periphery_hydro_axis_radius1 = periphery_hydro_params.get<double>("axis_radius1");
        double periphery_hydro_axis_radius2 = periphery_hydro_params.get<double>("axis_radius2");
        double periphery_hydro_axis_radius3 = periphery_hydro_params.get<double>("axis_radius3");
        MUNDY_THROW_REQUIRE(
            (periphery_hydro_shape == "SPHERE") || ((periphery_hydro_shape == "ELLIPSOID") &&
                                                    (periphery_hydro_axis_radius1 == periphery_hydro_axis_radius2) &&
                                                    (periphery_hydro_axis_radius2 == periphery_hydro_axis_radius3) &&
                                                    (periphery_hydro_axis_radius3 == periphery_hydro_axis_radius1)),
            std::invalid_argument, "Gauss-Legendre quadrature is only valid for spherical peripheries.");
      }
    }
  }

  static Teuchos::ParameterList get_valid_params() {
    // Create a paramater entity validator for our large integers to allow for both int and long long.
    auto prefer_size_t = []() {
      if (std::is_same_v<size_t, unsigned short>) {
        return mundy::core::OurAnyNumberParameterEntryValidator::PREFER_UNSIGNED_SHORT;
      } else if (std::is_same_v<size_t, unsigned int>) {
        return mundy::core::OurAnyNumberParameterEntryValidator::PREFER_UNSIGNED_INT;
      } else if (std::is_same_v<size_t, unsigned long>) {
        return mundy::core::OurAnyNumberParameterEntryValidator::PREFER_UNSIGNED_LONG;
      } else if (std::is_same_v<size_t, unsigned long long>) {
        return mundy::core::OurAnyNumberParameterEntryValidator::PREFER_UNSIGNED_LONG_LONG;
      } else {
        throw std::runtime_error("Unknown size_t type.");
        return mundy::core::OurAnyNumberParameterEntryValidator::PREFER_UNSIGNED_INT;
      }
    }();
    const bool allow_all_types_by_default = false;
    mundy::core::OurAnyNumberParameterEntryValidator::AcceptedTypes accept_int(allow_all_types_by_default);
    accept_int.allow_all_integer_types(true);
    auto make_new_validator = [](const auto &preferred_type, const auto &accepted_types) {
      return Teuchos::rcp(new mundy::core::OurAnyNumberParameterEntryValidator(preferred_type, accepted_types));
    };

    static Teuchos::ParameterList valid_parameter_list;

    valid_parameter_list.sublist("sim")
        .set("simid", 1, "Simulation ID for RNG seeding.", make_new_validator(prefer_size_t, accept_int))
        .set("num_time_steps", 100, "Number of time steps.", make_new_validator(prefer_size_t, accept_int))
        .set("timestep_size", 0.001, "Time step size.")
        .set("viscosity", 1.0, "Viscosity.")
        // Initialization
        .set("num_chromosomes", static_cast<size_t>(1), "Number of chromosomes.",
             make_new_validator(prefer_size_t, accept_int))
        .set("num_hetero_euchromatin_blocks", static_cast<size_t>(2),
             "Number of heterochromatin/euchromatin blocks per chain.", make_new_validator(prefer_size_t, accept_int))
        .set("num_euchromatin_per_block", static_cast<size_t>(1), "Number of euchromatin beads per block.",
             make_new_validator(prefer_size_t, accept_int))
        .set("num_heterochromatin_per_block", static_cast<size_t>(1), "Number of heterochromatin beads per block.",
             make_new_validator(prefer_size_t, accept_int))
        .set("hydro_type", std::string("RPYC"),
             "Hydrodynamic kernel type. Valid options are 'STOKES', 'RPY' or 'RPYC'.")
        .set("hydro_update_frequency", static_cast<size_t>(1),
             "The frequency at which we should update the background fluid velocity field. If not 1, the old velocity "
             "will be reused for the next update_frequency steps.",
             make_new_validator(prefer_size_t, accept_int))
        .set("backbone_sphere_hydrodynamic_radius", 0.5,
             "Backbone sphere hydrodynamic radius. Even if n-body hydrodynamics is disabled, we still have "
             "self-interaction.")
        .set("check_max_speed_pre_position_update", false, "Check max speed before updating positions.")
        .set("max_allowable_speed", std::numeric_limits<double>::max(),
             "Maximum allowable speed (only used if "
             "check_max_speed_pre_position_update is true).")
        .set("initial_chromosome_separation", 1.0, "Initial chromosome separation.")
        .set("initialization_type", std::string("GRID"),
             "Initialization_type. Valid options are GRID, RANDOM_UNIT_CELL, "
             "OVERLAP_TEST, HILBERT_RANDOM_UNIT_CELL, USHAPE_TEST, "
             "FROM_EXO, FROM_DAT.")
        .set("initialize_from_exo_filename", std::string("HP1"),
             "Exo file to initialize from if initialization_type is FROM_EXO.")
        .set("initialize_from_dat_filename", std::string("HP1_pos.dat"),
             "Dat file to initialize from if initialization_type is FROM_DAT.")
        .set<Teuchos::Array<double>>(
            "domain_low", Teuchos::tuple<double>(0.0, 0.0, 0.0),
            "Lower left corner of the unit cell. (Only used if initialization_type involves a 'UNIT_CELL').")
        .set<Teuchos::Array<double>>(
            "domain_high", Teuchos::tuple<double>(10.0, 10.0, 10.0),
            "Upper right corner of the unit cell. (Only used if initialization_type involves a 'UNIT_CELL').")
        .set("loadbalance_post_initialization", false, "If we should load balance post-initialization or not.")
        .set("specify_chromosome_layout", false,
             "If true, we will use the specified chromosome layout from the input file.")
        .set("specify_chromosome_file", std::string("chromosome_layout.dat"),
             "If specify_chromosome_layout is true, this is the file containing the chromosome layout.")
        // IO
        .set("io_frequency", static_cast<size_t>(10), "Number of timesteps between writing output.",
             make_new_validator(prefer_size_t, accept_int))
        .set("log_frequency", static_cast<size_t>(10), "Number of timesteps between logging.",
             make_new_validator(prefer_size_t, accept_int))
        .set("output_filename", std::string("HP1"), "Output filename.")
        .set("enable_continuation_if_available", true,
             "Enable continuing a previous simulation if an output file already exists.")
        // Control flags
        .set("enable_brownian_motion", true, "Enable chromatin Brownian motion.")
        .set("enable_backbone_springs", true, "Enable backbone springs.")
        .set("enable_backbone_collision", true, "Enable backbone collision.")
        .set("enable_backbone_n_body_hydrodynamics", true, "Enable backbone N-body hydrodynamics.")
        .set("enable_crosslinkers", true, "Enable crosslinkers.")
        .set("enable_periphery_collision", true, "Enable periphery collision.")
        .set("enable_periphery_hydrodynamics", true, "Enable periphery hydrodynamics.")
        .set("enable_periphery_binding", true, "Enable periphery binding.")
        .set("enable_active_euchromatin_forces", true, "Enable active euchromatin forces.");

    valid_parameter_list.sublist("brownian_motion").set("kt", 1.0, "Temperature kT for Brownian Motion.");

    valid_parameter_list.sublist("backbone_springs")
        .set("spring_type", std::string("HOOKEAN"), "Chromatin spring type. Valid options are HOOKEAN or FENE.")
        .set("spring_constant", 100.0, "Chromatin spring constant.")
        .set("spring_r0", 1.0, "Chromatin rest length (HOOKEAN) or rmax (FENE).");

    valid_parameter_list.sublist("backbone_collision")
        .set("backbone_sphere_collision_radius", 0.5, "Backbone sphere collision radius (as so aptly named).")
        .set("max_allowable_overlap", 1e-4, "Maximum allowable overlap between spheres post-collision resolution.")
        .set("backbone_collision_type", std::string("WCA"),
             "Collision type. Valid options are 'WCA' or 'COMPLEMENTARITY'.")
        .set("max_collision_iterations", static_cast<size_t>(10000),
             "Maximum number of collision iterations. If this is reached, an error will be thrown.",
             make_new_validator(prefer_size_t, accept_int))
        .set("backbone_wca_epsilon", 1.0, "Backbone WCA epsilon.")
        .set("backbone_wca_sigma", 1.0, "Backbone WCA sigma.")
        .set("backbone_wca_cutoff", 1.122462048309373, "Backbone WCA cutoff.");

    valid_parameter_list.sublist("crosslinker")
        .set("spring_type", std::string("HOOKEAN"), "Crosslinker spring type. Valid options are HOOKEAN or FENE.")
        .set("kt", 1.0, "Temperature kT for crosslinkers.")
        .set("spring_constant", 10.0, "Crosslinker spring constant.")
        .set("spring_r0", 2.5, "Crosslinker rest length.")
        .set("left_binding_rate", 1.0, "Crosslinker left binding rate.")
        .set("right_binding_rate", 1.0, "Crosslinker right binding rate.")
        .set("left_unbinding_rate", 1.0, "Crosslinker left unbinding rate.")
        .set("right_unbinding_rate", 1.0, "Crosslinker right unbinding rate.");

    valid_parameter_list.sublist("periphery_hydro")
        .set("check_max_periphery_overlap", false, "Check max periphery overlap.")
        .set("max_allowed_periphery_overlap", 1e-6, "Maximum allowed periphery overlap.")
        .set("shape", std::string("SPHERE"), "Periphery hydrodynamic shape. Valid options are SPHERE or ELLIPSOID.")
        .set("radius", 5.0, "Periphery radius (only used if periphery_shape is SPHERE).")
        .set("axis_radius1", 5.0, "Periphery axis length 1 (only used if periphery_shape is ELLIPSOID).")
        .set("axis_radius2", 5.0, "Periphery axis length 2 (only used if periphery_shape is ELLIPSOID).")
        .set("axis_radius3", 5.0, "Periphery axis length 3 (only used if periphery_shape is ELLIPSOID).")
        .set("quadrature", std::string("GAUSS_LEGENDRE"),
             "Periphery quadrature. Valid options are GAUSS_LEGENDRE or "
             "FROM_FILE.")
        .set("spectral_order", static_cast<size_t>(32),
             "Periphery spectral order (only used if periphery is spherical is Gauss-Legendre quadrature).",
             make_new_validator(prefer_size_t, accept_int))
        .set("num_quadrature_points", static_cast<size_t>(1000),
             "Periphery number of quadrature points (only used if quadrature type is FROM_FILE). Number of points in "
             "the files must match this quantity.",
             make_new_validator(prefer_size_t, accept_int))
        .set("quadrature_points_filename", std::string("hp1_periphery_hydro_quadrature_points.dat"),
             "Periphery quadrature points filename (only used if quadrature type is FROM_FILE).")
        .set("quadrature_weights_filename", std::string("hp1_periphery_hydro_quadrature_weights.dat"),
             "Periphery quadrature weights filename (only used if quadrature type is FROM_FILE).")
        .set("quadrature_normals_filename", std::string("hp1_periphery_hydro_quadrature_normals.dat"),
             "Periphery quadrature normals filename (only used if quadrature type is FROM_FILE).");

    valid_parameter_list.sublist("periphery_collision")
        .set("shape", std::string("SPHERE"), "Periphery collision shape. Valid options are SPHERE or ELLIPSOID.")
        .set("radius", 5.0, "Periphery radius (only used if periphery_shape is SPHERE).")
        .set("axis_radius1", 5.0, "Periphery axis length 1 (only used if periphery_shape is ELLIPSOID).")
        .set("axis_radius2", 5.0, "Periphery axis length 2 (only used if periphery_shape is ELLIPSOID).")
        .set("axis_radius3", 5.0, "Periphery axis length 3 (only used if periphery_shape is ELLIPSOID).")
        .set("collision_spring_constant", 1000.0, "Periphery collision spring constant.");

    valid_parameter_list.sublist("periphery_binding")
        .set("binding_rate", 1.0, "Periphery binding rate.")
        .set("unbinding_rate", 1.0, "Periphery unbinding rate.")
        .set("spring_constant", 1000.0, "Periphery spring constant.")
        .set("spring_r0", 1.0, "Periphery spring rest length.")
        .set("bind_sites_type", std::string("RANDOM"),
             "Periphery bind sites type. Valid options are RANDOM or FROM_FILE.")
        .set("shape", std::string("SPHERE"),
             "The shape of the binding site locations. Only used if bind_sites_type is RANDOM. Valid options are "
             "SPHERE or ELLIPSOID.")
        .set("radius", 5.0, "Periphery radius (only used if periphery_shape is SPHERE).")
        .set("axis_radius1", 5.0, "Periphery axis length 1 (only used if periphery_shape is ELLIPSOID).")
        .set("axis_radius2", 5.0, "Periphery axis length 2 (only used if periphery_shape is ELLIPSOID).")
        .set("axis_radius3", 5.0, "Periphery axis length 3 (only used if periphery_shape is ELLIPSOID).")
        .set("num_bind_sites", static_cast<size_t>(1000),
             "Periphery number of binding sites (only used if periphery_binding_sites_type is RANDOM and periphery "
             "has spherical or ellipsoidal shape).",
             make_new_validator(prefer_size_t, accept_int))
        .set("bind_site_locations_filename", std::string("periphery_bind_sites.dat"),
             "Periphery binding sites filename (only used if periphery_binding_sites_type is FROM_FILE).");

    valid_parameter_list.sublist("active_euchromatin_forces")
        .set("force_sigma", 1.0, "Active euchromatin force sigma.")
        .set("kon", 1.0, "Active euchromatin force kon.")
        .set("koff", 1.0, "Active euchromatin force koff.");

    valid_parameter_list.sublist("neighbor_list").set("skin_distance", 1.0, "Neighbor list skin distance.");

    return valid_parameter_list;
  }

  void dump_parameters(const Teuchos::ParameterList &valid_param_list) {
    if (stk::parallel_machine_rank(MPI_COMM_WORLD) == 0) {
      std::cout << "##################################################" << std::endl;
      std::cout << "INPUT PARAMETERS:" << std::endl;

      std::cout << std::endl;
      const auto &sim_params = valid_param_list.sublist("sim");
      std::cout << "SIMULATION:" << std::endl;
      std::cout << "  simid:           " << sim_params.get<size_t>("simid") << std::endl;
      std::cout << "  num_time_steps:  " << sim_params.get<size_t>("num_time_steps") << std::endl;
      std::cout << "  timestep_size:   " << sim_params.get<double>("timestep_size") << std::endl;
      std::cout << "  viscosity:       " << sim_params.get<double>("viscosity") << std::endl;
      std::cout << "  num_chromosomes: " << sim_params.get<size_t>("num_chromosomes") << std::endl;
      std::cout << "  num_hetero_euchromatin_blocks:      " << sim_params.get<size_t>("num_hetero_euchromatin_blocks")
                << std::endl;
      std::cout << "  num_euchromatin_per_block: " << sim_params.get<size_t>("num_euchromatin_per_block") << std::endl;
      std::cout << "  num_heterochromatin_per_block:  " << sim_params.get<size_t>("num_heterochromatin_per_block")
                << std::endl;
      std::cout << "  backbone_sphere_hydrodynamic_radius: "
                << sim_params.get<double>("backbone_sphere_hydrodynamic_radius") << std::endl;
      std::cout << "  initial_chromosome_separation:   " << sim_params.get<double>("initial_chromosome_separation")
                << std::endl;
      std::cout << "  initialization_type:             " << sim_params.get<std::string>("initialization_type")
                << std::endl;
      if (sim_params.get<std::string>("initialization_type") == "FROM_EXO") {
        std::cout << "  initialize_from_file_filename: " << sim_params.get<std::string>("initialize_from_exo_filename")
                  << std::endl;
      }

      if (sim_params.get<std::string>("initialization_type") == "FROM_DAT") {
        std::cout << "  initialize_from_file_filename: " << sim_params.get<std::string>("initialize_from_dat_filename")
                  << std::endl;
      }

      if (sim_params.get<bool>("specify_chromosome_layout")) {
        std::cout << "  specify_chromosome_layout: " << sim_params.get<bool>("specify_chromosome_layout") << std::endl;
        std::cout << "  specify_chromosome_file:   " << sim_params.get<std::string>("specify_chromosome_file")
                  << std::endl;
      }

      if ((sim_params.get<std::string>("initialization_type") == "RANDOM_UNIT_CELL") ||
          (sim_params.get<std::string>("initialization_type") == "HILBERT_RANDOM_UNIT_CELL")) {
        auto domain_low = sim_params.get<Teuchos::Array<double>>("domain_low");
        auto domain_high = sim_params.get<Teuchos::Array<double>>("domain_high");
        std::cout << "  domain_low: {" << domain_low[0] << ", " << domain_low[1] << ", " << domain_low[2] << "}"
                  << std::endl;
        std::cout << "  domain_high: {" << domain_high[0] << ", " << domain_high[1] << ", " << domain_high[2] << "}"
                  << std::endl;
      }

      std::cout << "  loadbalance_post_initialization: " << sim_params.get<bool>("loadbalance_post_initialization")
                << std::endl;
      std::cout << "  check_max_speed_pre_position_update: "
                << sim_params.get<bool>("check_max_speed_pre_position_update") << std::endl;
      if (sim_params.get<bool>("check_max_speed_pre_position_update")) {
        std::cout << "  max_allowable_speed: " << sim_params.get<double>("max_allowable_speed") << std::endl;
      }
      std::cout << std::endl;

      std::cout << "IO:" << std::endl;
      std::cout << "  io_frequency:    " << sim_params.get<size_t>("io_frequency") << std::endl;
      std::cout << "  log_frequency:   " << sim_params.get<size_t>("log_frequency") << std::endl;
      std::cout << "  output_filename: " << sim_params.get<std::string>("output_filename") << std::endl;
      std::cout << "  enable_continuation_if_available: " << sim_params.get<bool>("enable_continuation_if_available")
                << std::endl;
      std::cout << std::endl;

      std::cout << "CONTROL FLAGS:" << std::endl;
      std::cout << "  enable_brownian_motion: " << sim_params.get<bool>("enable_brownian_motion") << std::endl;
      std::cout << "  enable_backbone_springs:          " << sim_params.get<bool>("enable_backbone_springs")
                << std::endl;
      std::cout << "  enable_backbone_collision:        " << sim_params.get<bool>("enable_backbone_collision")
                << std::endl;
      std::cout << "  enable_backbone_n_body_hydrodynamics:    "
                << sim_params.get<bool>("enable_backbone_n_body_hydrodynamics") << std::endl;
      std::cout << "  enable_crosslinkers:              " << sim_params.get<bool>("enable_crosslinkers") << std::endl;
      std::cout << "  enable_periphery_hydrodynamics:   " << sim_params.get<bool>("enable_periphery_hydrodynamics")
                << std::endl;
      std::cout << "  enable_periphery_collision:       " << sim_params.get<bool>("enable_periphery_collision")
                << std::endl;
      std::cout << "  enable_periphery_binding:         " << sim_params.get<bool>("enable_periphery_binding")
                << std::endl;
      std::cout << "  enable_active_euchromatin_forces: " << sim_params.get<bool>("enable_active_euchromatin_forces")
                << std::endl;

      if (sim_params.get<bool>("enable_brownian_motion")) {
        const auto &brownian_motion_params = valid_param_list.sublist("brownian_motion");

        std::cout << std::endl;
        std::cout << "BROWNIAN MOTION:" << std::endl;
        std::cout << "  kt: " << brownian_motion_params.get<double>("kt") << std::endl;
      }

      if (sim_params.get<bool>("enable_backbone_springs")) {
        const auto &backbone_springs_params = valid_param_list.sublist("backbone_springs");

        std::cout << std::endl;
        std::cout << "BACKBONE SPRINGS:" << std::endl;
        std::cout << "  spring_type:      " << backbone_springs_params.get<std::string>("spring_type") << std::endl;
        std::cout << "  spring_constant:  " << backbone_springs_params.get<double>("spring_constant") << std::endl;
        if (backbone_springs_params.get<std::string>("spring_type") == "HOOKEAN") {
          std::cout << "  spring_r0 (rest_length): " << backbone_springs_params.get<double>("spring_r0") << std::endl;
        } else if (backbone_springs_params.get<std::string>("spring_type") == "FENE") {
          std::cout << "  spring_r0 (r_max):       " << backbone_springs_params.get<double>("spring_r0") << std::endl;
        }
      }

      if (sim_params.get<bool>("enable_backbone_collision")) {
        const auto &backbone_collision_params = valid_param_list.sublist("backbone_collision");

        std::cout << std::endl;
        std::cout << "BACKBONE COLLISION:" << std::endl;
        std::cout << "  backbone_sphere_collision_radius: "
                  << backbone_collision_params.get<double>("backbone_sphere_collision_radius") << std::endl;
        std::cout << "  max_allowable_overlap: " << backbone_collision_params.get<double>("max_allowable_overlap")
                  << std::endl;
        std::cout << "  max_collision_iterations: " << backbone_collision_params.get<size_t>("max_collision_iterations")
                  << std::endl;
      }

      if (sim_params.get<bool>("enable_crosslinkers")) {
        const auto &crosslinker_params = valid_param_list.sublist("crosslinker");

        std::cout << std::endl;
        std::cout << "CROSSLINKERS:" << std::endl;
        std::cout << "  spring_type: " << crosslinker_params.get<std::string>("spring_type") << std::endl;
        std::cout << "  kt: " << crosslinker_params.get<double>("kt") << std::endl;
        std::cout << "  spring_constant: " << crosslinker_params.get<double>("spring_constant") << std::endl;
        std::cout << "  spring_r0: " << crosslinker_params.get<double>("spring_r0") << std::endl;
        std::cout << "  left_binding_rate: " << crosslinker_params.get<double>("left_binding_rate") << std::endl;
        std::cout << "  right_binding_rate: " << crosslinker_params.get<double>("right_binding_rate") << std::endl;
        std::cout << "  left_unbinding_rate: " << crosslinker_params.get<double>("left_unbinding_rate") << std::endl;
        std::cout << "  right_unbinding_rate: " << crosslinker_params.get<double>("right_unbinding_rate") << std::endl;
      }

      if (sim_params.get<bool>("enable_periphery_hydrodynamics")) {
        const auto &periphery_hydro_params = valid_param_list.sublist("periphery_hydro");

        std::cout << std::endl;
        std::cout << "PERIPHERY HYDRODYNAMICS:" << std::endl;
        std::cout << "  check_max_periphery_overlap: "
                  << periphery_hydro_params.get<bool>("check_max_periphery_overlap") << std::endl;
        if (periphery_hydro_params.get<bool>("check_max_periphery_overlap")) {
          std::cout << "  max_allowed_periphery_overlap: "
                    << periphery_hydro_params.get<double>("max_allowed_periphery_overlap") << std::endl;
        }
        if (periphery_hydro_params.get<std::string>("shape") == "SPHERE") {
          std::cout << "  shape: SPHERE" << std::endl;
          std::cout << "  radius: " << periphery_hydro_params.get<double>("radius") << std::endl;
        } else if (periphery_hydro_params.get<std::string>("shape") == "ELLIPSOID") {
          std::cout << "  shape: ELLIPSOID" << std::endl;
          std::cout << "  axis_radius1: " << periphery_hydro_params.get<double>("axis_radius1") << std::endl;
          std::cout << "  axis_radius2: " << periphery_hydro_params.get<double>("axis_radius2") << std::endl;
          std::cout << "  axis_radius3: " << periphery_hydro_params.get<double>("axis_radius3") << std::endl;
        }
        if (periphery_hydro_params.get<std::string>("quadrature") == "GAUSS_LEGENDRE") {
          std::cout << "  quadrature: GAUSS_LEGENDRE" << std::endl;
          std::cout << "  spectral_order: " << periphery_hydro_params.get<size_t>("spectral_order") << std::endl;
        } else if (periphery_hydro_params.get<std::string>("quadrature") == "FROM_FILE") {
          std::cout << "  quadrature: FROM_FILE" << std::endl;
          std::cout << "  num_quadrature_points: " << periphery_hydro_params.get<size_t>("num_quadrature_points")
                    << std::endl;
          std::cout << "  quadrature_points_filename: "
                    << periphery_hydro_params.get<std::string>("quadrature_points_filename") << std::endl;
          std::cout << "  quadrature_weights_filename: "
                    << periphery_hydro_params.get<std::string>("quadrature_weights_filename") << std::endl;
          std::cout << "  quadrature_normals_filename: "
                    << periphery_hydro_params.get<std::string>("quadrature_normals_filename") << std::endl;
        }
      }

      if (sim_params.get<bool>("enable_periphery_collision")) {
        const auto &periphery_collision_params = valid_param_list.sublist("periphery_collision");

        std::cout << std::endl;
        std::cout << "PERIPHERY COLLISION:" << std::endl;
        if (periphery_collision_params.get<std::string>("shape") == "SPHERE") {
          std::cout << "  shape: SPHERE" << std::endl;
          std::cout << "  radius: " << periphery_collision_params.get<double>("radius") << std::endl;
        } else if (periphery_collision_params.get<std::string>("shape") == "ELLIPSOID") {
          std::cout << "  shape: ELLIPSOID" << std::endl;
          std::cout << "  axis_radius1: " << periphery_collision_params.get<double>("axis_radius1") << std::endl;
          std::cout << "  axis_radius2: " << periphery_collision_params.get<double>("axis_radius2") << std::endl;
          std::cout << "  axis_radius3: " << periphery_collision_params.get<double>("axis_radius3") << std::endl;
        }
        std::cout << "  collision_spring_constant: "
                  << periphery_collision_params.get<double>("collision_spring_constant") << std::endl;
      }

      if (sim_params.get<bool>("enable_periphery_binding")) {
        const auto &periphery_binding_params = valid_param_list.sublist("periphery_binding");

        std::cout << std::endl;
        std::cout << "PERIPHERY BINDING:" << std::endl;
        std::cout << "  binding_rate: " << periphery_binding_params.get<double>("binding_rate") << std::endl;
        std::cout << "  unbinding_rate: " << periphery_binding_params.get<double>("unbinding_rate") << std::endl;
        std::cout << "  spring_constant: " << periphery_binding_params.get<double>("spring_constant") << std::endl;
        std::cout << "  spring_r0: " << periphery_binding_params.get<double>("spring_r0") << std::endl;
        if (periphery_binding_params.get<std::string>("bind_sites_type") == "RANDOM") {
          std::cout << "  bind_sites_type: RANDOM" << std::endl;
          if (periphery_binding_params.get<std::string>("shape") == "SPHERE") {
            std::cout << "  shape: SPHERE" << std::endl;
            std::cout << "  radius: " << periphery_binding_params.get<double>("radius") << std::endl;
          } else if (periphery_binding_params.get<std::string>("shape") == "ELLIPSOID") {
            std::cout << "  shape: ELLIPSOID" << std::endl;
            std::cout << "  axis_radius1: " << periphery_binding_params.get<double>("axis_radius1") << std::endl;
            std::cout << "  axis_radius2: " << periphery_binding_params.get<double>("axis_radius2") << std::endl;
            std::cout << "  axis_radius3: " << periphery_binding_params.get<double>("axis_radius3") << std::endl;
          }

          std::cout << "  num_bind_sites: " << periphery_binding_params.get<size_t>("num_bind_sites") << std::endl;
        } else if (periphery_binding_params.get<std::string>("bind_sites_type") == "FROM_FILE") {
          std::cout << "  bind_sites_type: FROM_FILE" << std::endl;
          std::cout << "  bind_site_locations_filename: "
                    << periphery_binding_params.get<std::string>("bind_site_locations_filename") << std::endl;
        }
      }

      if (sim_params.get<bool>("enable_active_euchromatin_forces")) {
        const auto &active_euchromatin_forces_params = valid_param_list.sublist("active_euchromatin_forces");

        std::cout << std::endl;
        std::cout << "ACTIVE EUCHROMATIN FORCES:" << std::endl;
        std::cout << "  force_sigma: " << active_euchromatin_forces_params.get<double>("force_sigma") << std::endl;
        std::cout << "  kon: " << active_euchromatin_forces_params.get<double>("kon") << std::endl;
        std::cout << "  koff: " << active_euchromatin_forces_params.get<double>("koff") << std::endl;
      }

      std::cout << std::endl;

      std::cout << "NEIGHBOR LIST:" << std::endl;
      const auto &neighbor_list_params = valid_param_list.sublist("neighbor_list");
      std::cout << "  skin_distance: " << neighbor_list_params.get<double>("skin_distance") << std::endl;
      std::cout << "##################################################" << std::endl;
    }
  }

 private:
  /// \brief Default parameter filename if none is provided.
  std::string input_parameter_filename_ = "ngp_hp1.yaml";
};  // class HP1ParamParser

}  // namespace ngphp1

}  // namespace mundy

//@}

#endif  // MUNDY_NGP_HP1_PARSER_HPP_