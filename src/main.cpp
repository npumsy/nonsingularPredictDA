/************************************************************************
 * Copyright (C) 2024 Meng Siyang
 * Author: Meng Siyang
 * Description:
 *   计算卫星轨道
 *   Purpose:
 *         Build Python API by pybind11.
 *
 *************************************************************************/

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include <pybind11/eigen.h>

#include "elements.h"
#include "kepler.h"
#include "integrater.h"
#include "dastates.h"

using namespace Eigen;
namespace py = pybind11;

// 绑定代码，使用pybind11
PYBIND11_MODULE(qoe, m)
{

     m.doc() = R"pbdoc(
        SpaceDSL is a astrodynamics simulation library. This library is Written by C++.
        The purpose is to provide an open framework for astronaut dynamics enthusiasts,
        and more freely to achieve astrodynamics simulation.
        The project is open under the MIT protocol, and it is also for freer purposes.
        The project is built with CMake and can be used on Windows, Linux and Mac OS.
        This library can compiled into static library, dynamic library and Python library.
    )pbdoc";
     // **************************SpConst.h**************************
     m.def("PI", []()
           { return M_PI; });

     m.def("GM_Earth", []()
           { return osculating::MU; });

     m.def("EarthRadius", []()
           { return osculating::RE; });
     
     m.def("Earth_J2", []()
           { return osculating::J2; });
     m.def("Earth_J3", []()
           { return osculating::J3; });
     m.def("Earth_J4", []()
           { return osculating::J4; });
      
      py::class_<DensityInterpolator>(m, "DensityInterpolator")
        .def(py::init<const std::string&>(), R"pbdoc(
            构造函数，从给定的CSV文件中读取数据并初始化插值器。
            参数:
            filename : str
                CSV文件路径，包含高度和密度数据。

            )pbdoc")
        .def("getRho", &DensityInterpolator::getRho, R"pbdoc(
            成员函数，根据目标高度进行线性插值计算。
            参数:
            target_height : float
                目标高度，单位为千米。
            返回:
            float
                目标高度处的大气密度，单位为千克/立方米。
            异常:
            std::out_of_range
                如果目标高度超出数据范围。
            )pbdoc");

     // **************************SpOrbitParam.h**************************

     m.def("propagate_lagrangian", &kep3::propagate_lagrangian,
           py::arg("pos0"), py::arg("vel0"), py::arg("tof"), py::arg("mu"), py::arg("stm"), py::arg("Phi0f"),
           R"pbdoc(
          Propagate an initial Cartesian state for a time t assuming a central body and keplerian motion.
          
          Parameters
          ----------
          pos0 : numpy.ndarray
              Initial position vector (3x1).
          vel0 : numpy.ndarray
              Initial velocity vector (3x1).
          tof : float
              Time of flight.
          mu : float
              Gravitational parameter.
          stm : bool
              State transition matrix flag.
          Phi0f : numpy.ndarray
              State transition matrix (6x6).

          Returns
          -------
          numpy.ndarray
              Propagated state vector (6x1).
          )pbdoc");
     m.def("rv2OEOsc", &osculating::rv2OEOsc);
     m.def("OEOsc2rv", &osculating::OEOsc2rv,py::arg("ICSc"), py::arg("MaxIt"), py::arg("eps"),
      R"pbdoc(
            使用 OEOsc2rv 函数计算位置-速度向量 x。
            Parameters
                  numpy.ndarray(6x1) 瞬时轨道元素 OE
                  int 最大迭代次数 MaxIt 
                  double 容差 epsl。

            Returns : numpy.ndarray(6x1)
                  包含位置和速度的 6 元素向量 x。
                  OE: (a,平纬度幅角 M + w,ecos(w),esin(w),i,RAAN)
            )pbdoc");
     m.def("OEOsc2OEMeanEU", &osculating::OEOsc2OEMeanEU);
     m.def("OEMeanEU2OEOsc", &osculating::OEMeanEU2OEOsc);
     m.def("testOEosc", &osculating::testOEosc);
     m.def("testOEMean", &osculating::testOEMean);
     m.def("testAllosc", &osculating::testAll);

//      递推轨道
      m.def("intJ234DragRV_RK4Step", intJ234DragRV_RK4Step, py::arg("rv0"), 
            py::arg("tf"),
            py::arg("rhoCdA_m")= 1.4281E-12,
            py::arg("J234")=true,
            py::arg("step")=1.0,
          R"pbdoc(
              递推位置和速度.
              Parameters
                  RV0 :
                        Array of position and velocity (6x1).
                  tf:
                  rhoCdA_m : rho*Cd*A/m
                  step :     
              Returns: 
                  RVf :
                        Array of position and velocity (6x1).
          )pbdoc");
      // kep3::Vector6d EigenwarpOrbitJ234DragODE(kep3::Vector6d &rv0, double t, double arg1)
      // Vector6d EigenwarpIntOrbitJ234DragODE(const Vector6d &rv0, double t0,double rhoCdA_m,bool J234)
      m.def("EigenwarpDAOrbitJ234DragODE", EigenwarpDAOrbitJ234DragODE, py::arg("rv0"), 
            py::arg("t0"),
            py::arg("scale_rhoCdA_m")=1,
            py::arg("J234")=true);
// rho0 =3.245746e-4 kg/km^3
// A 2m^2 = 2e-6km^2
// Cd 2.2
// m 1000 kg
      m.def("EigenwarpIntOrbitJ234DragODE", EigenwarpIntOrbitJ234DragODE, py::arg("rv0"), 
            py::arg("t0"),
            py::arg("rhoCdA_m")=1.42812824E-12,// km-1, 改成单位m，需要乘1e3
            py::arg("J234")=true);
      py::class_<NominalErrorProp>(m, "NominalErrorProp")
            .def(py::init<const kep3::Vector6d&, int>(),py::arg("rv0"),
                  py::arg("order")=1, R"pbdoc(
                  构造函数，初始化误差传播模型。
                  参数:
                        rv0 : kep3::Vector6d                        初始位置和速度矢量。
                        order : int                        误差传播的阶数。
            )pbdoc")
            .def("~NominalErrorProp", [](NominalErrorProp& self) { delete &self; }, R"pbdoc(
                  析构函数。
            )pbdoc")
            .def("updateX0", &NominalErrorProp::updateX0,
                  py::arg("rv0"))
            .def("propNomJ234Drag", &NominalErrorProp::propNomJ234Drag,
                  py::arg("Phi0f"),
                  py::arg("tf"),
                  py::arg("givePhi")=true,
                  py::arg("step")=10.0,
             R"pbdoc(
                  成员函数，计算一阶和二阶誉差传播。
                  参数:
                        Phi0f : Eigen::Ref<Eigen::Matrix<double, 6, 6>>                        6x6状态转移矩阵。
                        tf :                         终端时间。
                        givePhi : bool, optional                        是否返回状态转移矩阵。
                        step : double, optional                        积分步长。
                  返回:
                        kep3::Vector6d                        终端状态误差。
            )pbdoc")
            .def("evaldXf", &NominalErrorProp::evaldXf,
                  py::arg("drv0"), 
                  py::arg("scale_rhoCdA_m")=1.0,
            R"pbdoc(
                  成员函数，计算终端状态误差。
                  参数:
                        drv0 : kep3::Vector6d                        初始状态误差。
                        scale_rhoCdA_m : double, optional                        缩放因子。
                  返回:
                        kep3::Vector6d                        终端状态误差。
            )pbdoc")
            .def("bkpropNomJ234Drag", &NominalErrorProp::bkpropNomJ234Drag,
            // 输入仍是按tp>t0来算，不能提供负数
                  py::arg("Phi0f"),
                  py::arg("tp"),
                  py::arg("givePhi")=true,
                  py::arg("step")=10.0
            )
            .def("evaldXp", &NominalErrorProp::evaldXp,
                  py::arg("drv0"), 
                  py::arg("scale_rhoCdA_m")=1.0);
      m.def("daJ234DragRV_RK4Step", daJ234DragRV_RK4Step, py::arg("rv0"), 
            py::arg("Phi0f"),
            py::arg("tf"),
            py::arg("scale_rhoCdA_m")=1,
            py::arg("givePhi")=false,
            py::arg("step")=10.0,
            py::arg("order")=1,
            py::arg("scale_derive")=1,
          R"pbdoc(
              利用多项式代数求解，RK4递推位置和速度.并求解状态转移矩阵
              Parameters
                  RV0 :
                        Array of position and velocity (6x1).
                  Phi0f :
                        State Transition matrix (6x6).
                  tf:
                  scalerhoCdA_m : 几倍乘以 rho*Cd*A/m
                  step :  
                  order: 多项式展开的阶数   
              Returns: 
                  RVf :
                        Array of position and velocity (6x1).
          )pbdoc");
       m.def("OscElemsLongpropagate", &osculating::OscElemsLongpropagate, py::arg("tf"), py::arg("OEm"), 
            py::arg("RE") = osculating::RE, 
            py::arg("mu") = osculating::MU, 
            py::arg("tol") = 1e-9, 
            py::arg("J2") = osculating::J2, 
            py::arg("J3") = osculating::J3, 
            py::arg("J4") = osculating::J4,
            R"pbdoc(
           平均轨道要素长期摄动影响. 考虑J2、J3、J4摄动
            Parameters
            Osc :
                  OE: (a,平纬度幅角 M + w,ecos(w),esin(w),i,RAAN) (6x1).
                  delta t
            Returns: numpy.ndarray
                  Array of mean elements (6x1).
            )pbdoc");
      // 
// double rho0 = 3.003075e-4, double CDA_m=0.0044,
      m.def("OscElemsLongDrag_BCrho0", &osculating::OscElemsLongDrag_BCrho0, py::arg("OEm"),  py::arg("dt"),
            py::arg("rp0") =  535e3,
            py::arg("H0") =65.35644970323516e3 , 
            py::arg("mu") = osculating::MU, 
            R"pbdoc(
           大气阻力摄动平均轨道要素长期. 阻力除以Beta
            Parameters
            Osc :
                  Array of mean elements (6x1).
                  delta t
            Returns: 
                  考虑时间的长期变化。 (6x1).
            )pbdoc");
      m.def("Dragbeta",&osculating::Dragbeta,
            py::arg("rho0") = 3.003075e-4,
            py::arg("CDA_m")=4.4e-9,
            R"pbdoc(
            轨道参数估计时，相乘的一项。基准参数可查询atmo1976. 单位为km
            )pbdoc");
            
      m.def("Jacobian_RV2OscElems", &osculating::Jacobian_RV2OscElems, py::arg("OE"), py::arg("RV"),
          py::arg("mu")=osculating::MU, 
          py::arg("tol"),
          R"pbdoc(
              计算从位置和速度相对于轨道要素的雅可比矩阵.
              Parameters
              OE :
                  Array of orbital elements (6x1).
              x :
                  State vector (6x1).
              mu :      Gravitational parameter.
              tol :     Tolerance for 求解Kepler方程
              Returns: 
                  雅可比矩阵 (6x6).
          )pbdoc");

    m.def("STM_ONsElemsWarppedByCOE", &osculating::STM_ONsElemsWarppedByCOE, py::arg("OE"), py::arg("tf"),
          py::arg("Re") = osculating::RE, 
          py::arg("mu") = osculating::MU, 
          py::arg("J2") = osculating::J2, 
          R"pbdoc(
              计算第一类非奇异轨道要素的J2摄动的状态转移矩阵. 内部计算仍然是经典轨道要素的
              Parameters
              OE :
                  Array of orbital elements (6x1).
              tf :
                  Final time.
              Re :      Earth radius.
              mu :      Gravitational parameter.
              J2 :      Second zonal harmonic.
              Returns: 
                  状态转移矩阵 (6x6).
          )pbdoc");
      


// **************************SpOrbitParam.h**************************
// oscstm::
     m.def("OE2Osc", &oscstm::OE2Osc,
          R"pbdoc(
          Convert orbital elements to osculating elements. numpy.ndarray(6x1).
          Parameter
               OE: (a,平纬度幅角 M + w,ecos(w),esin(w),i,RAAN) (6x1).
          Returns
               Osc: (a,纬度幅角 f + w,ecos(w),esin(w),i,RAAN) (6x1).
          )pbdoc");

    m.def("Osc2OE", &oscstm::Osc2OE,
          R"pbdoc(
          Convert osculating elements to orbital elements. numpy.ndarray(6x1).
          Parameters
          Osc :
              Osc: (a,纬度幅角 f + w,ecos(w),esin(w),i,RAAN) (6x1).
          Returns
              OE: (a,平纬度幅角 M + w,ecos(w),esin(w),i,RAAN) (6x1).
          )pbdoc");
      m.def("OscMeanElemspropagate", &oscstm::OscMeanElemspropagate, 
            py::arg("J2") = osculating::J2, 
            py::arg("t"),
             py::arg("ICSc"),
            py::arg("RE") = osculating::RE, 
            py::arg("mu") = osculating::MU, 
            py::arg("tol") = 1e-9, 
            R"pbdoc(
           平均轨道要素长期摄动影响. 考虑J2
            Parameters
            Osc :
                  Array of mean elements (6x1).
                  delta t
            Returns: numpy.ndarray
                  Array of mean elements (6x1).
            )pbdoc");
    
      m.def("lam2theta", [](double lambda, double q1, double q2, double Tol) {
            double F;
            double theta = oscstm::lam2theta(lambda, q1, q2, Tol, F);
            return std::make_tuple(theta, F);
        },
        py::arg("lambda"), py::arg("q1"), py::arg("q2"), py::arg("Tol"),
        R"pbdoc(
        Calculate true longitude theta and eccentric longitude F from mean longitude lambda.

        Parameters
            lambda : float
                  Mean longitude.
            q1 : float
                  e * cos(w).
            q2 : float
                  e * sin(w).
            Tol : float
                  Tolerance.
        Returns
            Tuple[float, float]
                  True longitude theta and eccentric longitude F.
        )pbdoc");

    m.def("theta2lam", &oscstm::theta2lam,
          py::arg("a"), py::arg("theta"), py::arg("q1"), py::arg("q2"),
          R"pbdoc(
          Calculate mean longitude lambda from true longitude theta.

          Parameters
            a : float
                  Semi-major axis.
            theta : float
                  True longitude.
            q1 : float
                  e * cos(w).
            q2 : float
                  e * sin(w).

          Returns
              Mean longitude lambda.
          )pbdoc");

      // 下面这几个函数来自Gim-Alfriend相对运动解析解，需要进一步调试
      m.def("OscMeanElemsSTM", &oscstm::OscMeanElemsSTM,
          py::arg("J2"), py::arg("t"), py::arg("ICSc"), py::arg("Re"), py::arg("mu"), py::arg("tol"),
          R"pbdoc(
          Calculate the state transition matrix for mean non-singular variables with perturbation by J2.
            Gim-Alfriend主轨道要素的STM,

          Parameters
            J2 : float
                  J2 perturbation coefficient.
            t : numpy.ndarray
                  Array of times.(t0,tf)
            ICSc : numpy.ndarray
                  Array of initial conditions (6x1).
            Re : float
                  Earth radius.
            mu : float
                  Gravitational parameter.
            tol : float
                  Tolerance.

          Returns : numpy.ndarray(6x6)
               state transition matrix.
          )pbdoc");
      m.def("DMeanToOsculatingElements", &oscstm::DMeanToOsculatingElements,
          py::arg("J2"), py::arg("meanElems"), py::arg("Re"), py::arg("mu"), py::arg("DJ2"),
          R"pbdoc(
          Convert mean orbital elements to osculating elements with perturbation by J2.
          
          Parameters
            J2 : float
                  J2 perturbation coefficient.
            meanElems : numpy.ndarray
                  Array of mean orbital elements (6x1).
            Re : float
                  Earth radius.
            mu : float
                  Gravitational parameter.
            DJ2 : numpy.ndarray (6x6)
                  transformation matrix (output).

          Returns: numpy.ndarray(6x1).          
              Array of osculating orbital elements 

          Notes
            The function calculates the formation matrix D_J2 in closed form between mean and osculating 
            new set of elements with the perturbation by only J2.
          )pbdoc");
      m.def("OscMeanToOsculatingElements", &oscstm::OscMeanToOsculatingElements,
          py::arg("J2"), py::arg("meanElems"), py::arg("Re"), py::arg("mu"),
          R"pbdoc(
          Convert mean orbital elements to osculating elements with perturbation by J2.
          
          Parameters
            J2 : float
                  J2 perturbation coefficient.
            meanElems : numpy.ndarray
                  Array of mean orbital elements (6x1).
            Re : float
                  Earth radius.
            mu : float
                  Gravitational parameter.

          Returns: numpy.ndarray(6x1).          
              Array of osculating orbital elements 

          Notes
            Form mean to osculating element with the perturbation by only J2
          )pbdoc");

// test 

     m.def("testlam2theta", &oscstm::testlam2theta);
     m.def("testDensity", &testDensity);
     m.def("testSigmaMat", &oscstm::testSigmaMat);
     m.def("testSigmaInverseMatrix", &oscstm::testSigmaInverseMatrix);
     m.def("testOscMeanToOsculatingElements", &oscstm::testOscMeanToOsculatingElements);
     m.def("testOscMeanSTM", &oscstm::testOscMeanSTM);

     
#ifdef VERSION_INFO
     m.attr("__version__") = VERSION_INFO;
#else
     m.attr("__version__") = "dev";
#endif
}

