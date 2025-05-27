#************************************************#
#                                                #
#       this cmakelist build for  ctesting       #
#                                                #
#************************************************#
cmake_minimum_required(VERSION 3.8.0)

# Set project name
project(qoe)

# Set  C++ standard version
set(CMAKE_CXX_STANDARD 14)
include_directories(/usr/include/eigen3)
# include_directories(/home/msy/miniconda3/include/eigen3)
# find_package(Eigen REQUIRED)

# 找到 Python 和 pybind11 包
# set(PYBIND11_PYTHON_VERSION 3.12)
# set(Python_ROOT_DIR "/home/msy/miniconda3/bin/python3.12")
# set(PYBIND11_INCLUDE_DIRS "/home/msy/miniconda3/include/python3.12")
# include_directories(${PYTHON_INCLUDE_DIRS})

set(Python3_EXECUTABLE "/usr/bin/python3.8")
# set(Python3_EXECUTABLE "/home/msy/miniconda3/bin/python3.12")

find_package(Python3 COMPONENTS Interpreter Development)
set(pybind11_DIR /home/msy/.local/lib/python3.8/site-packages/pybind11/share/cmake/pybind11)
# set(pybind11_DIR  /home/msy/miniconda3/lib/python3.12/site-packages/pybind11/share/cmake/pybind11)
find_package(pybind11 REQUIRED)
find_package(dace REQUIRED)
# 查找 dace 库
find_library(DACE_LIB dace PATHS /usr/local/lib)

include_directories("/usr/local/include/dace")
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
# 链接 dace 库到 qoe 模块
target_link_libraries(${PROJECT_NAME} PRIVATE ${DACE_LIB})
target_include_directories(${PROJECT_NAME} PRIVATE /usr/local/include/dace)
# 设置编译目标
# add_executable(da src/dastate.cpp)
# target_link_libraries(da dace)

set(CMAKE_BUILD_TYPE Debug)
add_library(dastate SHARED src/dastate.cpp)
target_link_libraries(dastate ${DACE_LIB})
add_executable(test src/test.cpp src/kepler.cpp src/elements.cpp src/integrater.cpp)
target_link_libraries(test dastate)


