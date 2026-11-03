cmake_minimum_required(VERSION 3.18)
project(ekf_delta LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 120)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDAToolkit REQUIRED)
set(Python3_EXECUTABLE "/home/msy/miniconda3/envs/rover_slam/bin/python")
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
set(pybind11_DIR "/home/msy/miniconda3/envs/rover_slam/lib/python3.10/site-packages/pybind11/share/cmake/pybind11")
find_package(pybind11 REQUIRED)
find_package(OpenMP REQUIRED)

set(PARENT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(QOE_CORE "${PARENT_DIR}/build/libqoe_core.so")
if(NOT EXISTS "${QOE_CORE}")
  message(FATAL_ERROR "missing ${QOE_CORE}; build the parent nonsingularPredictDA first")
endif()

pybind11_add_module(ekf_delta src/ekf_delta.cu ${PARENT_DIR}/src/kepler.cpp ${PARENT_DIR}/src/elements.cpp)
target_include_directories(ekf_delta PRIVATE "${PARENT_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}/include" /usr/include/eigen3)
target_link_libraries(ekf_delta PRIVATE "${QOE_CORE}" OpenMP::OpenMP_CXX CUDA::cudart CUDA::cublas CUDA::cusolver)
target_compile_options(ekf_delta PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
set_target_properties(ekf_delta PROPERTIES BUILD_RPATH "${PARENT_DIR}/build")

