# cmake_minimum_required(VERSION 3.15...3.28)

# project(diamx_lz_model VERSION 1.0)

# Fetch NEST library
include(FetchContent)
FetchContent_Declare(
  nest
  GIT_REPOSITORY https://github.com/NESTCollaboration/nest.git
  GIT_TAG        2418fef3fa87170a14e7db07d23ebeab1eb19bbb # Version 2.4.0
)
FetchContent_MakeAvailable(nest)

# lz_model libraray
add_library(nestDiamxCore STATIC src/nestDiamxCore.cpp)
# target_link_libraries(nestDiamxCore PUBLIC NESTCore diamx_compiler_flags)
target_link_libraries(nestDiamxCore PUBLIC NESTCore)
target_include_directories(nestDiamxCore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(nestDiamxCore PROPERTIES POSITION_INDEPENDENT_CODE ON)

# diamx.nest python bindings
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(nestDiamx MODULE src/nestDiamxPy.cpp)
target_link_libraries(nestDiamx PUBLIC nestDiamxCore)

# test executable
# add_executable(test src/test.cpp)
# target_link_libraries(test PUBLIC nestDiamxCore)