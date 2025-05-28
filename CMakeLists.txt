cmake_minimum_required(VERSION 3.15...3.28)

project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX
)

# Compiler flags
add_library(diamx_compiler_flags INTERFACE)
target_compile_features(diamx_compiler_flags INTERFACE cxx_std_11)

add_subdirectory(diamx/nest)

target_compile_definitions(nestDiamx PRIVATE VERSION_INFO=${PROJECT_VERSION})

# The install directory is the output (wheel) directory
# install(TARGETS nestDiamx DESTINATION .)
install(TARGETS nestDiamx DESTINATION ${SKBUILD_PROJECT_NAME})