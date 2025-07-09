# @HEADER
# **********************************************************************************************************************
#
#                                   NgpHP1: Interphase Chromatin Modeling using MuNDy
#                                             Copyright 2025 Bryce Palmer et al.
#
# NgpHP1 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# NgpHP1 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Mundy. If not, see
# <https://www.gnu.org/licenses/>.
#
# **********************************************************************************************************************
# @HEADER

###############################################################################
# Global setup options
###############################################################################
cmake_minimum_required(VERSION 3.23.0 FATAL_ERROR)

# Get and declare PROJECT_NAME
include("${CMAKE_CURRENT_SOURCE_DIR}/ProjectName.cmake")

# Get the current git hash
set(${PROJECT_NAME}_VERSION 0.0.1)

execute_process(
    COMMAND git describe --abbrev=8 --dirty --always
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE ${PROJECT_NAME}_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message("${PROJECT_NAME} version: v${${PROJECT_NAME}_VERSION} (${${PROJECT_NAME}_COMMIT})")

project(
    ${PROJECT_NAME}
    VERSION ${${PROJECT_NAME}_VERSION}
    LANGUAGES CXX
)

add_definitions("-D${PROJECT_NAME}_VERSION=\"${${PROJECT_NAME}_VERSION}\"")
add_definitions("-D${PROJECT_NAME}_COMMIT=\"${${PROJECT_NAME}_COMMIT}\"")

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")


###############################################################################
# Options
###############################################################################
option(${PROJECT_NAME}_ENABLE_UNIT_TESTS "Enable unit testing" ON)
option(${PROJECT_NAME}_ENABLE_ALL_WARNINGS "Show most warnings for most compilers" ON)
option(${PROJECT_NAME}_ENABLE_WERROR "Warnings are errors" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED)

###############################################################################
# EXE/LIBRARY
###############################################################################
set(${PROJECT_NAME}_exe_name "${PROJECT_NAME}.X")
add_executable(${${PROJECT_NAME}_exe_name} ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)
add_library(${PROJECT_NAME} "")
# add_library(${PROJECT_NAME}::all_libs INTERFACE)

###############################################################################
# MPI
###############################################################################
find_package(MPI REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC $<$<BOOL:${MPI_CXX_FOUND}>:MPI::MPI_CXX>)


###############################################################################
# Mundy
###############################################################################
find_package(Mundy REQUIRED HINTS ${Mundy_DIR})
target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC ${Mundy_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PUBLIC ${Mundy_LIBRARIES})
add_definitions(${Mundy_DEFINITIONS})

message("Found Mundy = ${Mundy_DIR}")
message("Mundy version:      ${Mundy_VERSION}")
message("Mundy include dirs: ${Mundy_INCLUDE_DIRS}")
message("Mundy libraries:    ${Mundy_LIBRARIES}")
message("Mundy definitions:  ${Mundy_DEFINITIONS}")


###############################################################################
# Targets
###############################################################################
message(STATUS "CMAKE_SYSTEM_NAME = ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_CXX_COMPILER_ID = ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")

# Add -restrict to Intel CXX compiler
target_compile_options(${PROJECT_NAME} PUBLIC $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:Intel>>:-restrict>)

# Logic for handling warnings
if(${PROJECT_NAME}_ENABLE_ALL_WARNINGS)
    # GCC, Clang, and Intel seem to accept these
    list(APPEND ${PROJECT_NAME}_CXX_FLAGS "-Wall" "-Wextra" "-pedantic")

    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
        # Intel always reports some diagnostics we don't necessarily care about
        list(APPEND ${PROJECT_NAME}_CXX_FLAGS "-diag-disable:11074,11076")
    endif()

    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 7.0)
        # Avoid notes about -faligned-new with GCC > 7
        list(APPEND ${PROJECT_NAME}_CXX_FLAGS "-faligned-new")
    endif()
endif()

# Add our extra flags according to language
separate_arguments(${PROJECT_NAME}_CXX_FLAGS)
target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${${PROJECT_NAME}_CXX_FLAGS}>)
target_compile_options(${PROJECT_NAME} PRIVATE $<$<BOOL:${ENABLE_WERROR}>:-Werror>)

# Add the include directories for the library (we'll use a single src instead of splitting into src/include)
target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/src>
    $<INSTALL_INTERFACE:src>)

# Link our exe to our library and our alias to our library
target_link_libraries(${${PROJECT_NAME}_exe_name} PRIVATE ${PROJECT_NAME})
# target_link_libraries(${PROJECT_NAME}::all_libs INTERFACE ${PROJECT_NAME})

# Add the subdirectories for further processing
add_subdirectory(src)


###############################################################################
# Testing (optional)
###############################################################################
if(${PROJECT_NAME}_ENABLE_UNIT_TESTS)
    enable_testing()
    set(utest_ex_name "${PROJECT_NAME}_unit_tests.X")
    add_executable(${utest_ex_name} ${CMAKE_CURRENT_SOURCE_DIR}/tests/GTestMain.cpp)

    # Unit tests are a separate executable and depend on ${PROJECT_NAME} lib and GoogleTest/GoogleMock
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        release-1.12.1
    )

    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    # Link against gtest_main and mundy
    target_link_libraries(${utest_ex_name} PRIVATE 
        ${PROJECT_NAME}
        GTest::gtest 
        GTest::gtest_main 
        GTest::gmock 
        GTest::gmock_main)
    target_include_directories(${utest_ex_name} PRIVATE 
        "${CMAKE_SOURCE_DIR}/tests"
        "${CMAKE_SOURCE_DIR}/tests/unit_tests"
        "${googletest_SOURCE_DIR}/googlemock/include"
        "${googletest_SOURCE_DIR}/googletest/include")

    add_subdirectory(tests)
endif()


###############################################################################
# Installation
###############################################################################

# Create two aliases for the library ${PROJECT_NAME} and ${PROJECT_NAME}::all_libs.
# The latter makes us TriBITS compliant and is a nice way to standardized aliases.
install(TARGETS ${${PROJECT_NAME}_exe_name} ${PROJECT_NAME}
        EXPORT "${PROJECT_NAME}Targets"
        RUNTIME DESTINATION bin
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib)
install(DIRECTORY src/ DESTINATION include)
# install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include DESTINATION include)
# install(TARGETS ${PROJECT_NAME}::all_libs EXPORT "${PROJECT_NAME}Targets")

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

export(
    TARGETS ${PROJECT_NAME}
    NAMESPACE ${PROJECT_NAME}::
    FILE ${PROJECT_NAME}Targets.cmake)
install(
    EXPORT ${PROJECT_NAME}Targets
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION lib/cmake/${PROJECT_NAME})
configure_package_config_file(
    cmake/${PROJECT_NAME}Config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    INSTALL_DESTINATION lib/cmake/${PROJECT_NAME})
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    DESTINATION lib/cmake/${PROJECT_NAME})
