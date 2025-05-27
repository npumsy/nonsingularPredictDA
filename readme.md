## Intro
This is orbit propagater designed for precise orbit determination. It's not well tested, and to be developed, you should test on your own. The author has limmited effort to continue, so it's made public.  

Feature:
- using Orbit Element and cardesian state
- OE of classic orbit element and nonsingular element, considering transform of osculating and mean element, 
- state covariance, jacobian from element to $r,v$ vectors.
- state transition matrix, using lagrange analytical method
- ODE propogation with RK8(7) and RK4(5), variable step or fixed step
- C++ library and python libray `qoe` used in script

Some (dastate.cpp) based on differential algebra library [DACE](https://github.com/dacelib/dace), provide for high order ODE propagation of state error and STM.

Some (mainly kepler.cpp) based on [pykep](https://esa.github.io/pykep/), transfer of orbit element, and 

referenced to book of LiuLin,HouXiyun, Nanjing University, 《轨道力学基础》

## usage
```
python3 ../script/testcov.py 
```
result:
Osculatimg Orbital Elements by RV:
 6.9366e+06   0.0350005 0.000674293 4.82366e-05     1.26494   0.0169624
E_n1 = -0.036439
E_n1 = -0.044185
E_n1 = -0.036423
E_n1 = -0.036439
Mean Orbital Elements converted:
6.92796e+06   0.0349392 0.000135393 2.10863e-07     1.26474   0.0169479
----------------------
Osculatimg Orbital Elements converted:
 6.9366e+06   0.0350005 0.000674293 4.82366e-05     1.26494   0.0169624
===================
E_n1 = -0.036439
Position and Velocity Vector:
6.92544e+06      190433      230987    -303.939      2277.9      7229.1
Conversion errpo:
3.336e-06
6.30389e-07
9.66247e-08
4.32522e-10
2.54249e-09
-6.58201e-09

J234 force: (m/s^2)                     -0.0113171      -0.000311193    -0.00111071
RKdouble---Drag a_x, a_y, a_z(m/s^2)=   1.07131e-09,    -6.54819e-09,   -2.67007e-08,
RKdouble---total acc a_x, a_y, a_z(m/s^2)=      -8.29887,       -0.22820,         -0.27753,
DA---J234: (m/s^2)=                     -0.0113171,  -0.000311193,   -0.00111071
DA-- a_x, a_y, a_z(m/s^2)=                      -8.29887,       -0.228199,        -0.277529
DA---drag: (m/s^2)=                     1.07131e-09,    -6.54819e-09,   -2.67007e-08

### testode
```
python3 ./script/testode.py
```
result: 
DA d state (6x1):
0000    6925443.9520, 190432.6240, 230986.9010, -303.93854, 2277.90445, 7229.09828
0010    6921989.6884, 213199.8038, 303262.5598, -386.90722, 2275.48603, 7225.88874
0020    6917705.9636, 235941.4358, 375501.7816, -469.82911, 2272.79496, 7221.81096
STM error;

[ 1.00026494e+03 -9.97759010e+03 -1.99780164e+01  2.56829786e-02  1.24092986e+00 -9.97583299e-01]
[-3.28345987e+03  1.27640419e+04  7.22192438e+04 -8.28962117e+01 -1.45013948e+00 -5.07535913e+00]
正向递推erv0求误差演化 erv1(6x1):
0010    1000.0696, -9989.3976, -9.9950, 0.01371, 1.12048, -0.99895

Backward prop state(6x1):
0000    6925443.9520, 190432.6241, 230986.9010, -303.93854, 2277.90445, 7229.09828
反向递推erv1误差得出erv0(6x1): [ 1.29723921e-05  7.90925696e-05 -2.07737903e-05 -2.83454631e-06
 -4.50224570e-06  2.74389367e-06]
DA反向估计 estimate(6x1):
0000    999.9987, -10000.0000, -0.0000, 0.00026, 0.99998, -1.00000

t0加误差状态正向递推的结果，反向递推恢复t0误差(6x1):
 [ 1.00000000e+03 -1.00000000e+04 -2.31235463e-05 -4.82789915e-08
  9.99999998e-01 -1.00000000e+00]
Backward prop estimate(6x1): [-1.31054781e-03  3.98564443e-05 -7.42875272e-05  2.61788959e-04
  3.19708806e-06  1.29809960e-05]

## install
The user must change the related lines in CMake file, especilly :
```
find_library(DACE_LIB dace PATHS /usr/local/lib)
```
to your dace.so libyrary directory.

**使用前需要配置好dace.so库的位置，以及比如你所要依赖的python可执行文件的位置、pybind库的位置。**
### dependency
- eigen
- pybind11
- dace
- python

### build
项目根目录下运行以下命令以确保模块正确生成：

```sh
mkdir -p build
cd build
cmake ..
make
```
这样，生成的共享库模块应该位于 build 目录中，并且命名为 qoe.cpython-38-x86_64-linux-gnu.so。

C++ 运行测试可执行文件 build/test；

或者通过python运行测试。
确保 PYTHONPATH 包含生成的模块路径，然后运行你的测试脚本：

```sh
export PYTHONPATH=$(pwd)/build:$PYTHONPATH
cd ../tests
python3 test_my_module.py
```
这样，你的 Python 测试脚本应该能够找到并导入 qoe 模块。

