#************************************************#
#                                                #
#       this cmakelist build for  ctesting       #
#                                                #
#************************************************#
cmake_minimum_required(VERSION 3.8.0)

# Set project name
project(qoe)

# Set  C++ standard version
set(CMAKE_CXX_STANDARD 17)
include_directories(/usr/include/eigen3)
# include_directories(/home/msy/miniconda3/include/eigen3)
# find_package(Eigen REQUIRED)

# 找到 Python 和 pybind11 包
# set(PYBIND11_PYTHON_VERSION 3.12)
# set(Python_ROOT_DIR "/home/msy/miniconda3/bin/python3.12")
# set(PYBIND11_INCLUDE_DIRS "/home/msy/miniconda3/include/python3.12")
# include_directories(${PYTHON_INCLUDE_DIRS})

set(Python3_EXECUTABLE "/home/msy/miniconda3/envs/rover_slam/bin/python")
# set(Python3_EXECUTABLE "/usr/bin/python3.8")
# set(Python3_EXECUTABLE "/home/msy/miniconda3/bin/python3.12")

find_package(Python3 COMPONENTS Interpreter Development)
set(pybind11_DIR /home/msy/miniconda3/envs/rover_slam/lib/python3.10/site-packages/pybind11/share/cmake/pybind11)
# set(pybind11_DIR /home/msy/.local/lib/python3.8/site-packages/pybind11/share/cmake/pybind11)
# set(pybind11_DIR  /home/msy/miniconda3/lib/python3.12/site-packages/pybind11/share/cmake/pybind11)
find_package(pybind11 REQUIRED)
# 本地 DACE 已 build 到 ../dace/build（见 dace/README.md），否则回退系统 /usr/local
set(DACE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../dace/build" CACHE PATH "DACE prefix")
if(NOT EXISTS "${DACE_ROOT}/libdace.so")
  set(DACE_ROOT "/usr/local")
endif()
message(STATUS "qoe 使用 DACE_ROOT = ${DACE_ROOT}")
# dace_DIR：build 布局为 <root>/dace，install 布局为 <root>/lib/cmake/dace
set(dace_DIR "${DACE_ROOT}/dace" CACHE PATH "dace config dir" FORCE)
if(NOT EXISTS "${dace_DIR}/daceConfig.cmake")
  set(dace_DIR "${DACE_ROOT}/lib/cmake/dace" CACHE PATH "dace config dir" FORCE)
endif()
find_package(dace REQUIRED)
find_package(OpenMP REQUIRED)
# 运行时库目录（build 布局 libdace.so 在 DACE_ROOT 下，install 布局在 lib/ 下）
if(EXISTS "${DACE_ROOT}/libdace.so")
  set(DACE_LIBDIR "${DACE_ROOT}")
else()
  set(DACE_LIBDIR "${DACE_ROOT}/lib")
endif()
# Set src file list

# aux_source_directory (${PROJECT_SOURCE_DIR}/src/elements.cpp 
#     ${PROJECT_SOURCE_DIR}/src/integrater.cpp
#     ${PROJECT_SOURCE_DIR}/src/kepler.cpp
#     ${PROJECT_SOURCE_DIR}/src/main.cpp
#     SRC_LIST)
set(SRC_LIST src/elements.cpp src/integrater.cpp src/kepler.cpp src/dastate.cpp src/main.cpp)

# Set Include h file path
list(APPEND PYLIB_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include)
# 包含头文件目录
include_directories(${PROJECT_SOURCE_DIR}/include)

INCLUDE_DIRECTORIES(${PYLIB_INCLUDE_DIRS})

#
# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
# Set output directory for the shared library
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build)

# Add pybind11 module
pybind11_add_module(${PROJECT_NAME} ${SRC_LIST})
# 链接 dace 库与 OpenMP 到 qoe 模块
target_link_libraries(${PROJECT_NAME} PRIVATE dace::dace OpenMP::OpenMP_CXX)
set_target_properties(${PROJECT_NAME} PROPERTIES BUILD_RPATH "${DACE_LIBDIR}")
# 设置编译目标
# add_executable(da src/dastate.cpp)
# target_link_libraries(da dace)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -funroll-loops")
add_library(dastate SHARED src/dastate.cpp)
target_link_libraries(dastate dace::dace OpenMP::OpenMP_CXX)
set_target_properties(dastate PROPERTIES BUILD_RPATH "${DACE_LIBDIR}")
add_executable(test src/test.cpp src/kepler.cpp src/elements.cpp src/integrater.cpp)
target_link_libraries(test dastate)
set_target_properties(test PROPERTIES BUILD_RPATH "${DACE_LIBDIR}")
