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

#include <cstring>
#include <stdexcept>
#include <string>
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

     m.def("pertAccelECI", [](const Vector6d &rv, double kappa, const std::vector<double> &thetas, int lmax){
         return pertAccelECI(rv, kappa, thetas, lmax);
       }, py::arg("rv"), py::arg("kappa") = 1.0, py::arg("thetas") = std::vector<double>{}, py::arg("lmax") = 2,
          "摄动加速度（去二体，ECI，m/s²）：J234+阻力(κ)+三体+SH(θ)");

     m.def("setThirdBody", [](bool on){ setThirdBody(on); }, py::arg("on"),
           "三体（日/月）确定性摄动开关");
     m.def("setPropEpoch", [](double mjd0){ setPropEpoch(mjd0); }, py::arg("mjd0"),
           "传播起点绝对历元（MJD）");

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
     m.def("gveStepNoeBatch", [](const std::vector<kep3::Vector6d> &oe0s, double t0, double dt,
                                 int nthreads, double beta){
            std::vector<kep3::Vector6d> oef;
            { py::gil_scoped_release release; kep3::gveStepNoeBatch(oe0s, t0, dt, nthreads, beta, oef); }
            return oef;
         }, py::arg("oe0s"), py::arg("t0"), py::arg("dt"), py::arg("nthreads") = 16, py::arg("beta") = 1.0);
     m.def("rv2OEOsc", &osculating::rv2OEOsc);
     m.def("oe2eciPos", &osculating::oe2eciPos, py::arg("OE"), py::arg("MaxIt") = 100, py::arg("eps") = 1e-5,
           "快速 qoe→ECI 位置（与 OEOsc2rv 的 PQW→ECI 公式逐位一致）");
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
      // 非奇异要素（QOE）版误差传播：状态/返回值均为要素 (a[m], u=M+ω, ex, ey, i, Ω[rad])，高斯变分方程递推。
      m.def("noeGveRhs", &noeGveRhs, py::arg("oe"), py::arg("beta")=1.0, py::arg("t")=0.0,
            "高斯变分方程 RHS（非奇异要素 [a,u,ex,ey,i,Om]，m/rad）：摄动=J234+阻力(β)+三体");
      m.def("noeOsc2rv", &noeOsc2rv, py::arg("oe"), py::arg("MaxIt")=100, py::arg("epsl")=1e-12,
            "非奇异要素→rv（m, m/s），与 osculating::OEOsc2rv 同一公式");
      m.def("stateTransferGVEBatch", [](const std::vector<Vector6d> &oes, double tf, int nthreads, double step){
            std::vector<Vector6d> oef; std::vector<double> Phi;
            { py::gil_scoped_release release; stateTransferGVEBatch(oes, tf, nthreads, oef, Phi, step); }
            return py::make_tuple(oef, Phi);
            }, py::arg("oes"), py::arg("tf"), py::arg("nthreads")=16, py::arg("step")=10.0,
            "批量 GVE 一步传播：返回 (oe_f, Phi=∂oe_f/∂oe_0 展平 n×36)");
      m.def("noeOsc2rvJacBatch", [](const std::vector<Vector6d> &oes, int nthreads){
            std::vector<double> J;
            { py::gil_scoped_release release; noeOsc2rvJacBatch(oes, nthreads, J); }
            return J;
            }, py::arg("oes"), py::arg("nthreads")=16,
            "批量 ∂(r,v)/∂oe（展平 n×36，行主序）");
      m.def("noeOsc2rvBatch", [](const std::vector<Vector6d> &oes, int nthreads){
            std::vector<double> RV;
            { py::gil_scoped_release release; noeOsc2rvBatch(oes, nthreads, RV); }
            return RV;
            }, py::arg("oes"), py::arg("nthreads")=16,
            "批量 非奇异要素→rv（展平 n×6，行主序）");
      m.def("rv2OEOscBatch", [](const std::vector<Vector6d> &rvs, int nthreads){
            std::vector<double> OE;
            { py::gil_scoped_release release; rv2OEOscBatch(rvs, nthreads, OE); }
            return OE;
            }, py::arg("rvs"), py::arg("nthreads")=16,
            "批量 rv→非奇异要素（展平 n×6，行主序）");

      // ---- 快速 numpy 入口（避免逐星 std::vector<Vector6d> 的 Python/pybind 转换开销）----
      auto _v6_from_np = [](py::array_t<double, py::array::c_style | py::array::forcecast> a,
                            const char *what){
            auto buf = a.request();
            if(buf.ndim != 2 || buf.shape[1] != 6)
                throw std::runtime_error(std::string(what) + " 需 (n,6) float64 数组");
            const std::size_t n = (std::size_t)buf.shape[0];
            std::vector<Vector6d> v(n);
            const double *p = static_cast<const double *>(buf.ptr);
            for(std::size_t i = 0; i < n; ++i)
                for(int c = 0; c < 6; ++c) v[i](c) = p[i*6 + c];
            return v;
      };
      m.def("stateTransferBatchFlat",
            [&](py::array_t<double, py::array::c_style | py::array::forcecast> x, double dt, int nthreads){
            std::vector<Vector6d> v = _v6_from_np(x, "stateTransferBatchFlat");
            std::vector<Vector6d> xf; std::vector<double> Phi;
            { py::gil_scoped_release release; stateTransferBatch(v, dt, nthreads, xf, Phi); }
            const std::size_t n = v.size();
            py::array_t<double> XF({n, (std::size_t)6}), PH({n, (std::size_t)36});
            double *px = XF.mutable_data(), *pp = PH.mutable_data();
            for(std::size_t i = 0; i < n; ++i){
                for(int c = 0; c < 6; ++c) px[i*6 + c] = xf[i](c);
                for(int c = 0; c < 36; ++c) pp[i*36 + c] = Phi[i*36 + c];
            }
            return py::make_tuple(XF, PH);
            }, py::arg("x"), py::arg("dt"), py::arg("nthreads")=16,
            "批量单步（全动力学状态 + 解析二体 STM）：(n,6) numpy → ((n,6),(n,36))");
      m.def("noeOsc2rvBatchFlat",
            [&](py::array_t<double, py::array::c_style | py::array::forcecast> oe, int nthreads){
            std::vector<Vector6d> v = _v6_from_np(oe, "noeOsc2rvBatchFlat");
            std::vector<double> RV;
            { py::gil_scoped_release release; noeOsc2rvBatch(v, nthreads, RV); }
            const std::size_t n = v.size();
            py::array_t<double> O({n, (std::size_t)6});
            std::memcpy(O.mutable_data(), RV.data(), n*6*sizeof(double));
            return O;
            }, py::arg("oe"), py::arg("nthreads")=16, "批量 QOE→rv（(n,6) numpy）");
      m.def("noeOsc2rvJacBatchFlat",
            [&](py::array_t<double, py::array::c_style | py::array::forcecast> oe, int nthreads){
            std::vector<Vector6d> v = _v6_from_np(oe, "noeOsc2rvJacBatchFlat");
            std::vector<double> J;
            { py::gil_scoped_release release; noeOsc2rvJacBatch(v, nthreads, J); }
            const std::size_t n = v.size();
            py::array_t<double> JJ({n, (std::size_t)6, (std::size_t)6});
            std::memcpy(JJ.mutable_data(), J.data(), n*36*sizeof(double));
            return JJ;
            }, py::arg("oe"), py::arg("nthreads")=16, "批量 ∂(r,v)/∂oe（(n,6,6) numpy）");
      m.def("gveStepNoeBatchFlat",
            [&](py::array_t<double, py::array::c_style | py::array::forcecast> oe, double t0, double dt,
                int nthreads, double beta){
            std::vector<Vector6d> v = _v6_from_np(oe, "gveStepNoeBatchFlat");
            std::vector<Vector6d> of;
            { py::gil_scoped_release release; kep3::gveStepNoeBatch(v, t0, dt, nthreads, beta, of); }
            const std::size_t n = v.size();
            py::array_t<double> O({n, (std::size_t)6});
            double *p = O.mutable_data();
            for(std::size_t i = 0; i < n; ++i)
                for(int c = 0; c < 6; ++c) p[i*6 + c] = of[i](c);
            return O;
            }, py::arg("oe"), py::arg("t0"), py::arg("dt"), py::arg("nthreads")=16, py::arg("beta")=1.0,
            "批量 GVE 单步 3/8-RK4（CUDA）：(n,6) numpy → (n,6) numpy");
      m.def("gveGetTiming", &kep3::gveGetTiming,
            "GVE 单步细粒度计时 [pack, h2d, kernel, d2h]（秒，累计）");
      m.def("gveResetTiming", &kep3::gveResetTiming, "清零 GVE 单步细粒度计时");
      m.def("gvePropagateNoeBatchFlat",
            [&](py::array_t<double, py::array::c_style | py::array::forcecast> oe, int nfr,
                double dt, int nthreads, double beta){
            std::vector<Vector6d> v = _v6_from_np(oe, "gvePropagateNoeBatchFlat");
            const int n = (int)v.size();
            std::vector<double> rv_all;
            { py::gil_scoped_release release; kep3::gvePropagateNoeBatch(v, nfr, dt, nthreads, beta, rv_all); }
            py::array_t<double> O({(std::size_t)nfr, (std::size_t)n, (std::size_t)6});
            std::memcpy(O.mutable_data(), rv_all.data(), rv_all.size()*sizeof(double));
            return O;
            }, py::arg("oe"), py::arg("nfr"), py::arg("dt"), py::arg("nthreads")=16, py::arg("beta")=1.0,
            "整弧多帧 GVE（CUDA）：(n,6) → rv (nfr,n,6)");
      m.def("stmFoldGpuFlat",
            [&](py::array_t<double, py::array::c_style | py::array::forcecast> rv0, int nfr, double dt,
                py::array_t<double, py::array::c_style | py::array::forcecast> A0, int nthreads){
            auto b0 = rv0.request(); auto ba = A0.request();
            if(b0.ndim != 2 || b0.shape[1] != 6) throw std::runtime_error("rv0 需 (n,6)");
            if(ba.ndim != 2 || ba.shape[1] != 36) throw std::runtime_error("A0 需 (n,36)");
            const std::size_t n = (std::size_t)b0.shape[0];
            const double *p0 = static_cast<const double *>(b0.ptr);
            std::vector<Vector6d> rv(n);
            for(std::size_t i = 0; i < n; ++i)
                for(int c = 0; c < 6; ++c) rv[i](c) = p0[i*6 + c];
            std::vector<double> a0(n*36);
            std::memcpy(a0.data(), ba.ptr, n*36*sizeof(double));
            std::vector<double> out;
            { py::gil_scoped_release release; kep3::stmFoldGpuBatch(rv, nfr, dt, a0, nthreads, out); }
            py::array_t<double> O({(std::size_t)nfr, n, (std::size_t)3, (std::size_t)6});
            std::memcpy(O.mutable_data(), out.data(), out.size()*sizeof(double));
            return O;
            }, py::arg("rv0"), py::arg("nfr"), py::arg("dt"), py::arg("A0"), py::arg("nthreads")=16,
            "解析两体 STM 折叠（GPU）：(n,6),nfr,dt,(n,36) → A (nfr,n,3,6)");
      m.def("cartStepBatchFlat",
            [&](py::array_t<double, py::array::c_style | py::array::forcecast> x, double dt,
                int nthreads, double beta){
            std::vector<Vector6d> v = _v6_from_np(x, "cartStepBatchFlat");
            std::vector<Vector6d> of;
            { py::gil_scoped_release release; kep3::cartStepBatch(v, dt, nthreads, beta, of); }
            py::array_t<double> O({v.size(), (std::size_t)6});
            double *p = O.mutable_data();
            for(std::size_t i = 0; i < v.size(); ++i)
                for(int c = 0; c < 6; ++c) p[i*6 + c] = of[i](c);
            return O;
            }, py::arg("x"), py::arg("dt"), py::arg("nthreads")=16, py::arg("beta")=1.0,
            "笛卡尔全动力学单步 3/8-RK4（CUDA）：(n,6) → (n,6)");
      m.def("cartPropagateBatchFlat",
            [&](py::array_t<double, py::array::c_style | py::array::forcecast> x, int nfr, double dt,
                int nthreads, double beta){
            std::vector<Vector6d> v = _v6_from_np(x, "cartPropagateBatchFlat");
            const int n = (int)v.size();
            std::vector<double> x_all;
            { py::gil_scoped_release release; kep3::cartPropagateBatch(v, nfr, dt, nthreads, beta, x_all); }
            py::array_t<double> O({(std::size_t)nfr, (std::size_t)n, (std::size_t)6});
            std::memcpy(O.mutable_data(), x_all.data(), x_all.size()*sizeof(double));
            return O;
            }, py::arg("x"), py::arg("nfr"), py::arg("dt"), py::arg("nthreads")=16, py::arg("beta")=1.0,
            "笛卡尔全动力学整弧多帧（CUDA）：(n,6) → (nfr,n,6)");
      m.def("stateStmFoldFlat",
            [&](py::array_t<double, py::array::c_style | py::array::forcecast> rv0,
                py::array_t<double, py::array::c_style | py::array::forcecast> dts,
                py::array_t<double, py::array::c_style | py::array::forcecast> A0, int nthreads){
            auto b0 = rv0.request(); auto bd = dts.request(); auto ba = A0.request();
            if(b0.ndim != 2 || b0.shape[1] != 6) throw std::runtime_error("rv0 需 (n,6)");
            if(ba.ndim != 2 || ba.shape[1] != 36) throw std::runtime_error("A0 需 (n,36)");
            const std::size_t n = (std::size_t)b0.shape[0], nf = (std::size_t)bd.shape[0];
            const double *p0 = static_cast<const double *>(b0.ptr);
            std::vector<Vector6d> rv(n);
            for(std::size_t i = 0; i < n; ++i)
                for(int c = 0; c < 6; ++c) rv[i](c) = p0[i*6 + c];
            std::vector<double> dtv(nf);
            std::memcpy(dtv.data(), bd.ptr, nf*sizeof(double));
            std::vector<double> a0(n*36);
            std::memcpy(a0.data(), ba.ptr, n*36*sizeof(double));
            std::vector<double> out;
            { py::gil_scoped_release release; stateStmFoldBatch(rv, dtv, a0, nthreads, out); }
            py::array_t<double> O({nf, n, (std::size_t)3, (std::size_t)6});
            std::memcpy(O.mutable_data(), out.data(), out.size()*sizeof(double));
            return O;
            }, py::arg("rv0"), py::arg("dts"), py::arg("A0"), py::arg("nthreads")=16,
            "解析两体 STM 折叠：(n,6),(nfr,),(n,36) → A (nfr,n,3,6)");
      py::class_<NominalErrorPropNOE>(m, "NominalErrorPropNOE")
            .def(py::init<const Vector6d&, int>(), py::arg("oe0"), py::arg("order")=1, R"pbdoc(
                  构造函数：非奇异要素 (a[m], u=M+ω, ex, ey, i, Ω[rad]) 的一阶 DA 误差传播（GVE 递推）。
            )pbdoc")
            .def("updateX0", &NominalErrorPropNOE::updateX0, py::arg("oe0"))
            .def("propNomJ234Drag", &NominalErrorPropNOE::propNomJ234Drag,
                  py::arg("Phi0f"), py::arg("tf"), py::arg("givePhi")=true, py::arg("step")=10.0,
                  "正向递推：返回终端要素 oe_f（m, rad）；givePhi 时写 Phi0f = ∂oe_f/∂oe_0")
            .def("bkpropNomJ234Drag", &NominalErrorPropNOE::bkpropNomJ234Drag,
                  py::arg("Phi0f"), py::arg("tp"), py::arg("givePhi")=true, py::arg("step")=10.0,
                  "反向递推（tp 为正向时长，物理时间递减）：返回终端要素；givePhi 时写 ∂/∂oe_0")
            .def("evaldXf", &NominalErrorPropNOE::evaldXf, py::arg("doe0"), py::arg("scale_rhoCdA_m")=1.0,
                  "在一阶 Taylor 模型上由 d oe_0 求终端要素")
            .def("evaldXp", &NominalErrorPropNOE::evaldXp, py::arg("doe0"), py::arg("scale_rhoCdA_m")=1.0);
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
       // 增广状态 DA 传播：把阻力参数 kappa 升为第 7 个 DA 变量，导出 6 个输出状态的密集泰勒系数，
       // 用于可微 Learning（系数对展开中心的导数由更高一阶系数给出，不改 DACE 内核）。
       m.def("daJ234DragAugCoeffs", [](const Vector6d &rv0, double kappa0, double tf, int order, double step){
            std::vector<double> coeffs;
            std::vector<std::vector<unsigned int>> mons;
            Vector6d rvf = daJ234DragAugCoeffs(rv0, kappa0, tf, order, step, coeffs, mons);
            return py::make_tuple(rvf, coeffs, mons);
       }, py::arg("rv0"), py::arg("kappa0"), py::arg("tf"), py::arg("order")=2, py::arg("step")=1.0,
          R"pbdoc(
               增广状态 (x, kappa) 的 DA 传播，导出密集泰勒系数。
               Parameters
                   rv0 : 初始位置速度 (6x1), 单位 m。
                   kappa0 : 阻力缩放参数 (标量)。
                   tf : 传播时长 (s)。
                   order : DA 阶数。
                   step : RK4 步长 (s)。
               Returns:
                   (rvf, coeffs, mons):
                     rvf    : 终端状态 (6x1), 单位 m。
                     coeffs : 长度 6*nmono，按 [输出状态 i][单项式 k] 展平。
                     mons   : nmono 个 7 维指数向量。
          )pbdoc");
       // 整星座批处理（线程安全 double 传播 + OpenMP，释放 GIL）：返回 (xf, sens=∂x_f/∂κ)。
       m.def("daJ234DragBatchD", [](const std::vector<Vector6d> &rv0s,
                                    const std::vector<double> &kappas, double tf, double step, double dk, int nthreads){
            std::vector<Vector6d> sens;
            std::vector<Vector6d> xf;
            {
                py::gil_scoped_release release;
                xf = daJ234DragBatchD(rv0s, kappas, tf, step, dk, sens, nthreads);
            }
            return py::make_tuple(xf, sens);
       }, py::arg("rv0s"), py::arg("kappas"), py::arg("tf"), py::arg("step")=10.0,
          py::arg("dk")=1e-4, py::arg("nthreads")=0,
          R"pbdoc(
               整星座 J234+阻力 double 批传播（线程安全、OpenMP、释放 GIL）。
               Returns: (xf, sens)，sens=∂x_f/∂κ（有限差分）。
          )pbdoc");
       // 多历元阻力灵敏度：每星一次连续积分、各 tfs 输出 xf 与 ∂x/∂κ（展平 [N,K,6]）。
       m.def("daJ234DragMultiEpochBatch", [](const std::vector<Vector6d> &rv0s,
                                            const std::vector<double> &kappas,
                                            const std::vector<double> &tfs,
                                            double step, double dk, int nthreads){
             std::vector<double> xf, sens;
             { py::gil_scoped_release release;
               daJ234DragMultiEpochBatch(rv0s, kappas, tfs, step, dk, nthreads, xf, sens); }
             return py::make_tuple(xf, sens);
       }, py::arg("rv0s"), py::arg("kappas"), py::arg("tfs"), py::arg("step")=10.0,
          py::arg("dk")=1e-4, py::arg("nthreads")=0,
          R"pbdoc(
               多历元阻力灵敏度：每星一次连续积分，在 tfs 各时间历元输出 xf 与 sens=∂x/∂κ。
               返回展平列表 (xf, sens)，索引 [i*K*6 + k*6 + c]。
          )pbdoc");
       // C++ 批量单步 StateTransfer：一步 3/8 RK4（全 TBPfull 动力学）+ 解析二体变分 STM。
       m.def("stateTransferBatch", [](const std::vector<Vector6d> &rv0s, double dt, int nthreads){
            std::vector<Vector6d> xf; std::vector<double> Phi;
            { py::gil_scoped_release release; stateTransferBatch(rv0s, dt, nthreads, xf, Phi); }
            return py::make_tuple(xf, Phi);
       }, py::arg("rv0s"), py::arg("dt"), py::arg("nthreads")=0,
          R"pbdoc(
                C++ 批量单步 StateTransfer：x(dt) 一步 RK4（全动力学）；Phi=∂x(dt)/∂x0 解析二体变分。
          )pbdoc");
       // 整星座批传播（线程安全 double + OpenMP，释放 GIL）：位置相关残差场（RBF/SH），
       // 返回 (xf, Jt=∂x/∂θ [N,6,m], Jx=∂x/∂x0 [N,6,6])。场参数由入参设定。
       m.def("daFieldBatchDRBF", [](const std::vector<Vector6d> &rv0s,
                                    const std::vector<double> &thetas,
                                    const std::vector<std::array<double,3>> &centers, double s,
                                    double tf, double step, double dk, int nthreads){
            setRBFParams(centers, s);
            std::vector<double> Jt, Jx; std::vector<Vector6d> xf;
            { py::gil_scoped_release release;
              xf = daFieldBatchD(rv0s, thetas, tf, step, dk, nthreads, Jt, Jx); }
            return py::make_tuple(xf, Jt, Jx);
       }, py::arg("rv0s"), py::arg("thetas"), py::arg("centers"), py::arg("s"),
          py::arg("tf"), py::arg("step")=10.0, py::arg("dk")=1e-4, py::arg("nthreads")=0);
       m.def("daFieldBatchDSH", [](const std::vector<Vector6d> &rv0s,
                                   const std::vector<double> &thetas, int lmax,
                                   double tf, double step, double dk, int nthreads){
            setSHParams(lmax);
            std::vector<double> Jt, Jx; std::vector<Vector6d> xf;
            { py::gil_scoped_release release;
              xf = daFieldBatchD(rv0s, thetas, tf, step, dk, nthreads, Jt, Jx); }
            return py::make_tuple(xf, Jt, Jx);
       }, py::arg("rv0s"), py::arg("thetas"), py::arg("lmax"),
          py::arg("tf"), py::arg("step")=10.0, py::arg("dk")=1e-4, py::arg("nthreads")=0);
       // 多历元批传播（SH）：每星一次连续积分，返回 (xf[N,K,6]展平, Jt[N,K,6,m], Jx[N,K,6,6])。
       m.def("daFieldMultiEpochBatchDSH", [](const std::vector<Vector6d> &rv0s,
                                             const std::vector<double> &thetas, int lmax,
                                             const std::vector<double> &tfs, double step,
                                             double dk, int nthreads){
            setSHParams(lmax);
            std::vector<double> xf, Jt, Jx;
            { py::gil_scoped_release release;
              daFieldMultiEpochBatchD(rv0s, thetas, tfs, step, dk, nthreads, xf, Jt, Jx); }
            return py::make_tuple(xf, Jt, Jx);
       }, py::arg("rv0s"), py::arg("thetas"), py::arg("lmax"), py::arg("tfs"),
          py::arg("step")=10.0, py::arg("dk")=1e-4, py::arg("nthreads")=0);

       m.def("daFieldMultiEpochBatchDSH2", [](const std::vector<Vector6d> &rv0s,
                                             const std::vector<double> &thetas, int lmax,
                                             const std::vector<double> &tfs, double step,
                                             int nthreads){
            setSHParams(lmax);
            std::vector<double> xf, Jt, Jx, Hess;
            { py::gil_scoped_release release;
              daFieldMultiEpochBatchDA2(rv0s, thetas, tfs, 2, step, nthreads, xf, Jt, Jx, Hess); }
            return py::make_tuple(xf, Jt, Jx, Hess);
       }, py::arg("rv0s"), py::arg("thetas"), py::arg("lmax"), py::arg("tfs"),
          py::arg("step")=10.0, py::arg("nthreads")=0,
          "解析批量多历元 order=2：返回 (xf, Jt, Jx, Hess[...,36])。");

       m.def("daFieldMultiEpochBatchDRBF", [](const std::vector<Vector6d> &rv0s,
                                              const std::vector<double> &thetas,
                                              const std::vector<std::array<double,3>> &centers, double s,
                                              const std::vector<double> &tfs, double step,
                                              double dk, int nthreads){
            setRBFParams(centers, s);
            std::vector<double> xf, Jt, Jx;
            { py::gil_scoped_release release;
              daFieldMultiEpochBatchD(rv0s, thetas, tfs, step, dk, nthreads, xf, Jt, Jx); }
            return py::make_tuple(xf, Jt, Jx);
       }, py::arg("rv0s"), py::arg("thetas"), py::arg("centers"), py::arg("s"), py::arg("tfs"),
          py::arg("step")=10.0, py::arg("dk")=1e-4, py::arg("nthreads")=0);
       // 通用多参数增广状态 DA 传播：theta(1..m) 为阻力项的独立乘性因子（多参数可微 Learning）。
       m.def("daAugCoeffs", [](const Vector6d &rv0, const std::vector<double> &params,
                               double tf, int order, double step){
            std::vector<double> coeffs;
            std::vector<std::vector<unsigned int>> mons;
            Vector6d rvf = daAugCoeffs(rv0, params, tf, order, step, coeffs, mons);
            return py::make_tuple(rvf, coeffs, mons);
       }, py::arg("rv0"), py::arg("params"), py::arg("tf"), py::arg("order")=2, py::arg("step")=1.0,
          R"pbdoc(
               通用增广状态 (x, theta) 的 DA 传播，theta 为阻力项独立乘性因子，导出密集泰勒系数。
               Returns: (rvf, coeffs, mons)，coeffs 长度 6*nmono，mons 为 nmono 个 (6+m) 维指数向量。
          )pbdoc");
       // 位置相关 RBF 引力异常场：theta(1..m) 为可学习势系数，centers/s 固定。
       m.def("daAugRBFCoeffs", [](const Vector6d &rv0, const std::vector<double> &thetas,
                                  const std::vector<std::array<double,3>> &centers, double s,
                                  double tf, int order, double step){
            std::vector<double> coeffs;
            std::vector<std::vector<unsigned int>> mons;
            Vector6d rvf = daAugRBFCoeffs(rv0, thetas, centers, s, tf, order, step, coeffs, mons);
            return py::make_tuple(rvf, coeffs, mons);
       }, py::arg("rv0"), py::arg("thetas"), py::arg("centers"), py::arg("s"),
          py::arg("tf"), py::arg("order")=2, py::arg("step")=1.0,
           R"pbdoc(
               位置相关 RBF 引力异常场的增广 DA 传播。
               Returns: (rvf, coeffs, mons)，mons 为 nmono 个 (6+m) 维指数向量。
          )pbdoc");
       // 低阶非带谐球谐引力异常场（笛卡尔实球谐）：theta 为 C_lm,S_lm（l=2..lmax, m=1..l）。
       m.def("daAugSHCoeffs", [](const Vector6d &rv0, const std::vector<double> &thetas, int lmax,
                                 double tf, int order, double step){
            std::vector<double> coeffs;
            std::vector<std::vector<unsigned int>> mons;
            Vector6d rvf = daAugSHCoeffs(rv0, thetas, lmax, tf, order, step, coeffs, mons);
            return py::make_tuple(rvf, coeffs, mons);
       }, py::arg("rv0"), py::arg("thetas"), py::arg("lmax"), py::arg("tf"),
          py::arg("order")=2, py::arg("step")=1.0,
          R"pbdoc(
               低阶非带谐球谐（笛卡尔实球谐）引力异常场的增广 DA 传播。
               thetas 顺序：l=2..lmax, m=1..l, 每个 (C_lm,S_lm)；个数 = lmax(lmax+1)-2。
               Returns: (rvf, coeffs, mons)。
          )pbdoc");
       // 球谐异常场数值自检：给定位置(m)、系数、lmax，返回残差加速度(m/s^2)。
       m.def("shResidualAccel", [](const Vector6d &rv_m, const std::vector<double> &thetas, int lmax){
            return shResidualAccel(rv_m, thetas, lmax);
       }, py::arg("rv_m"), py::arg("thetas"), py::arg("lmax"),
          "低阶球谐异常场的残差加速度（数值，m/s^2），供物理自检。");
       // 残差加速度（m/s^2）访问器，供 A_{,θ} 的 FD 自检。
       m.def("fieldResidualAccelRBF", [](const Vector6d &rv_m, const std::vector<double> &thetas,
                                         const std::vector<std::array<double,3>> &centers, double s){
            setRBFParams(centers, s);
            return fieldResidualAccel(rv_m, thetas);
       }, py::arg("rv_m"), py::arg("thetas"), py::arg("centers"), py::arg("s"));
       m.def("fieldResidualAccelSH", [](const Vector6d &rv_m, const std::vector<double> &thetas, int lmax){
            setSHParams(lmax);
            return fieldResidualAccel(rv_m, thetas);
       }, py::arg("rv_m"), py::arg("thetas"), py::arg("lmax"));
       // A_{,θ_k} = ∂²f/∂x∂θ_k（力场基对状态的 Jacobian），供积分伴随使用与 FD 自检。
       m.def("fieldBasisJacobianRBF", [](const Vector6d &rv_km, const std::vector<double> &thetas,
                                         const std::vector<std::array<double,3>> &centers, double s){
            setRBFParams(centers, s);
            std::vector<double> dBdx;
            fieldBasisJacobian(rv_km.data(), (int)thetas.size(), dBdx);
            return dBdx;
       }, py::arg("rv_km"), py::arg("thetas"), py::arg("centers"), py::arg("s"),
          "RBF 力场的 A_{,θ_k}=∂b_k/∂x（长度 m*36，[k][i*6+j]）。");
       m.def("fieldBasisJacobianSH", [](const Vector6d &rv_km, int m, int lmax){
            setSHParams(lmax);
            std::vector<double> dBdx;
            fieldBasisJacobian(rv_km.data(), m, dBdx);
            return dBdx;
       }, py::arg("rv_km"), py::arg("m"), py::arg("lmax"),
          "球谐力场的 A_{,θ_k}=∂b_k/∂x（长度 m*36，[k][i*6+j]）。");
       // 路径 A：记录型标量 + tape 的 RBF 流传播与反向。
       py::class_<RecordFlow>(m, "RecordFlow").def_readonly("xf", &RecordFlow::xf);
       m.def("daRecordFlowRBF", [](const Vector6d &rv0, const std::vector<double> &thetas,
                                   const std::vector<std::array<double,3>> &centers, double s,
                                   double tf, double step){
            return daRecordFlowRBF(rv0, thetas, centers, s, tf, step);
       }, py::arg("rv0"), py::arg("thetas"), py::arg("centers"), py::arg("s"),
          py::arg("tf"), py::arg("step"),
          "路径 A：以记录型标量实例化 TBPfull_rbf<Scalar> 并积分，返回 xf(m) 与 tape 句柄。");
       m.def("daRecordFlowSH", [](const Vector6d &rv0, const std::vector<double> &thetas, int lmax,
                                  double tf, double step){
            return daRecordFlowSH(rv0, thetas, lmax, tf, step);
       }, py::arg("rv0"), py::arg("thetas"), py::arg("lmax"), py::arg("tf"), py::arg("step"),
          "路径 A（球谐力场）：记录型标量实例化 TBPfull_field<Scalar> 并积分。");
       // 多历元一阶算子：一次积分出各历元状态与一阶 Jacobian（order=1，对 m 线性）。
       m.def("daFieldMultiEpochRBF", [](const Vector6d &rv0, const std::vector<double> &thetas,
                                        const std::vector<std::array<double,3>> &centers, double s,
                                        const std::vector<double> &tfs, int order, double step){
            setRBFParams(centers, s);
            std::vector<Vector6d> rvf; std::vector<double> J;
            daFieldMultiEpoch(rv0, thetas, tfs, order, step, rvf, J);
            return py::make_tuple(rvf, J);
       }, py::arg("rv0"), py::arg("thetas"), py::arg("centers"), py::arg("s"),
          py::arg("tfs"), py::arg("order")=1, py::arg("step")=10.0,
          "多历元 RBF 一阶算子：返回 ([x_f^(k)], Jflat[k,6,6+m])。");
       m.def("daFieldMultiEpochSH", [](const Vector6d &rv0, const std::vector<double> &thetas,
                                       int lmax, const std::vector<double> &tfs, int order, double step){
            setSHParams(lmax);
            std::vector<Vector6d> rvf; std::vector<double> J;
            daFieldMultiEpoch(rv0, thetas, tfs, order, step, rvf, J);
            return py::make_tuple(rvf, J);
       }, py::arg("rv0"), py::arg("thetas"), py::arg("lmax"),
          py::arg("tfs"), py::arg("order")=1, py::arg("step")=10.0,
          "多历元球谐一阶算子：返回 ([x_f^(k)], Jflat[k,6,6+m])。");
       // D1：多历元稠密泰勒系数导出（供 torch 侧精确非线性求值/二阶）。
       m.def("daFieldMultiEpochCoeffsRBF", [](const Vector6d &rv0, const std::vector<double> &thetas,
                                              const std::vector<std::array<double,3>> &centers, double s,
                                              const std::vector<double> &tfs, int order, double step){
            setRBFParams(centers, s);
            std::vector<double> rvf, coeffs; std::vector<std::vector<unsigned int>> mons;
            daFieldMultiEpochCoeffs(rv0, thetas, tfs, order, step, rvf, coeffs, mons);
            return py::make_tuple(rvf, coeffs, mons);
       }, py::arg("rv0"), py::arg("thetas"), py::arg("centers"), py::arg("s"),
          py::arg("tfs"), py::arg("order")=1, py::arg("step")=10.0,
          "多历元 RBF 稠密系数：返回 (rvf[K*6](m), coeffs[K*6*nmono], mons[nmono][N])。");
       m.def("daFieldMultiEpochCoeffsSH", [](const Vector6d &rv0, const std::vector<double> &thetas,
                                             int lmax, const std::vector<double> &tfs, int order, double step){
            setSHParams(lmax);
            std::vector<double> rvf, coeffs; std::vector<std::vector<unsigned int>> mons;
            daFieldMultiEpochCoeffs(rv0, thetas, tfs, order, step, rvf, coeffs, mons);
            return py::make_tuple(rvf, coeffs, mons);
       }, py::arg("rv0"), py::arg("thetas"), py::arg("lmax"),
          py::arg("tfs"), py::arg("order")=1, py::arg("step")=10.0,
          "多历元球谐稠密系数：返回 (rvf[K*6](m), coeffs[K*6*nmono], mons[nmono][N])。");
       // 批量并行【解析】DA 多历元展开（WITH_PTHREAD + OpenMP）
       m.def("daFieldMultiEpochBatchDARBF", [](const std::vector<Vector6d> &rv0s,
                                               const std::vector<double> &thetas,
                                               const std::vector<std::array<double,3>> &centers, double s,
                                               const std::vector<double> &tfs, int order, double step, int nthreads){
            setRBFParams(centers, s);
            std::vector<double> xf, Jt, Jx;
            daFieldMultiEpochBatchDA(rv0s, thetas, tfs, order, step, nthreads, xf, Jt, Jx);
            return py::make_tuple(xf, Jt, Jx);
       }, py::arg("rv0s"), py::arg("thetas"), py::arg("centers"), py::arg("s"),
          py::arg("tfs"), py::arg("order")=1, py::arg("step")=10.0, py::arg("nthreads")=0,
          "批量并行解析 DA（RBF）：返回 (xf[N*K*6](m), Jt[N*K*6*m], Jx[N*K*6*6])。");
       m.def("daFieldMultiEpochBatchDASH", [](const std::vector<Vector6d> &rv0s,
                                              const std::vector<double> &thetas, int lmax,
                                              const std::vector<double> &tfs, int order, double step, int nthreads){
            setSHParams(lmax);
            std::vector<double> xf, Jt, Jx;
            daFieldMultiEpochBatchDA(rv0s, thetas, tfs, order, step, nthreads, xf, Jt, Jx);
            return py::make_tuple(xf, Jt, Jx);
       }, py::arg("rv0s"), py::arg("thetas"), py::arg("lmax"),
          py::arg("tfs"), py::arg("order")=1, py::arg("step")=10.0, py::arg("nthreads")=0,
          "批量并行解析 DA（SH）：返回 (xf[N*K*6](m), Jt[N*K*6*m], Jx[N*K*6*6])。");
       // 变分灵敏度多历元算子：DA 只管状态(N=6)给 A，θ 用变分矩阵（不进 DA，对 m 线性）。
       m.def("daVarMultiEpochRBF", [](const Vector6d &rv0, const std::vector<double> &thetas,
                                      const std::vector<std::array<double,3>> &centers, double s,
                                      const std::vector<double> &tfs, double step){
            setRBFParams(centers, s);
            std::vector<Vector6d> rvf; std::vector<double> J;
            daVarMultiEpoch(rv0, thetas, tfs, step, rvf, J);
            return py::make_tuple(rvf, J);
       }, py::arg("rv0"), py::arg("thetas"), py::arg("centers"), py::arg("s"),
          py::arg("tfs"), py::arg("step")=10.0, "变分灵敏度多历元 RBF 算子。");
       m.def("daVarMultiEpochSH", [](const Vector6d &rv0, const std::vector<double> &thetas,
                                     int lmax, const std::vector<double> &tfs, double step){
            setSHParams(lmax);
            std::vector<Vector6d> rvf; std::vector<double> J;
            daVarMultiEpoch(rv0, thetas, tfs, step, rvf, J);
            return py::make_tuple(rvf, J);
       }, py::arg("rv0"), py::arg("thetas"), py::arg("lmax"),
          py::arg("tfs"), py::arg("step")=10.0, "变分灵敏度多历元球谐算子。");
       // 批量并行【解析变分】（线程局部 DACE(1,6) + double [x,Φ,S]）
       m.def("daVarMultiEpochBatchPRBF", [](const std::vector<Vector6d> &rv0s,
                                            const std::vector<double> &thetas,
                                            const std::vector<std::array<double,3>> &centers, double s,
                                            const std::vector<double> &tfs, double step, int nthreads){
            setRBFParams(centers, s);
            std::vector<double> xf, Jt, Jx;
            daVarMultiEpochBatchP(rv0s, thetas, tfs, step, nthreads, xf, Jt, Jx);
            return py::make_tuple(xf, Jt, Jx);
       }, py::arg("rv0s"), py::arg("thetas"), py::arg("centers"), py::arg("s"),
          py::arg("tfs"), py::arg("step")=10.0, py::arg("nthreads")=0,
          "批量并行解析变分（RBF）：返回 (xf[N*K*6](m), Jt[N*K*6*m], Jx[N*K*6*6])。");
       m.def("daVarMultiEpochBatchPSH", [](const std::vector<Vector6d> &rv0s,
                                           const std::vector<double> &thetas, int lmax,
                                           const std::vector<double> &tfs, double step, int nthreads){
            setSHParams(lmax);
            std::vector<double> xf, Jt, Jx;
            daVarMultiEpochBatchP(rv0s, thetas, tfs, step, nthreads, xf, Jt, Jx);
            return py::make_tuple(xf, Jt, Jx);
       }, py::arg("rv0s"), py::arg("thetas"), py::arg("lmax"),
          py::arg("tfs"), py::arg("step")=10.0, py::arg("nthreads")=0,
          "批量并行解析变分（SH）：返回 (xf[N*K*6](m), Jt[N*K*6*m], Jx[N*K*6*6])。");
       m.def("daVarMultiEpochBatchPSKC", [](const std::vector<Vector6d> &rv0s,
                                            const std::vector<double> &thetas, int lmax,
                                            const std::vector<double> &tfs, double step, int nthreads){
            setSHParams(lmax);
            std::vector<double> xf, Jt, Jx, Jk;
            daVarMultiEpochBatchPSKC(rv0s, thetas, lmax, tfs, step, nthreads, xf, Jt, Jx, Jk);
            return py::make_tuple(xf, Jt, Jx, Jk);
       }, py::arg("rv0s"), py::arg("thetas"), py::arg("lmax"),
          py::arg("tfs"), py::arg("step")=10.0, py::arg("nthreads")=0,
          "批量解析变分 + 解析 κ：返回 (xf[N*K*6], Jt[N*K*6*m]=∂x/∂θ, Jx[N*K*6*6]=∂x/∂x0, Jk[N*K*6]=∂x/∂κ)。");
       m.def("daRecordFlowBackward", [](const RecordFlow &rf, const std::vector<double> &grad){
            Vector6d gx; std::vector<double> gp;
            daRecordFlowBackward(rf, grad, gx, gp);
            return py::make_tuple(gx, gp);
       }, py::arg("flow"), py::arg("grad"),
          "路径 A 反向：给定 ∂L/∂xf(m)，返回 (∂L/∂x0(m), ∂L/∂θ)。");
       // 积分伴随：前向出各历元 (x_f, Φ) 并缓存阶段；反向为离散 RK4 转置（含 deep）。
       py::class_<DeepFlow>(m, "DeepFlow")
            .def_readonly("rvf", &DeepFlow::rvf)
            .def_readonly("PhiEpoch", &DeepFlow::PhiEpoch)
            .def_readonly("tfs", &DeepFlow::tfs);
       m.def("daDeepForwardRBF", [](const Vector6d &rv0, const std::vector<double> &thetas,
                                    const std::vector<std::array<double,3>> &centers, double s,
                                    const std::vector<double> &tfs, double step){
            setRBFParams(centers, s);
            DeepFlow fl; daDeepForward(rv0, thetas, tfs, step, fl);
            return fl;
       }, py::arg("rv0"), py::arg("thetas"), py::arg("centers"), py::arg("s"),
          py::arg("tfs"), py::arg("step")=10.0, "积分伴随前向（RBF）：返回 DeepFlow。");
       m.def("daDeepForwardSH", [](const Vector6d &rv0, const std::vector<double> &thetas, int lmax,
                                   const std::vector<double> &tfs, double step){
            setSHParams(lmax);
            DeepFlow fl; daDeepForward(rv0, thetas, tfs, step, fl);
            return fl;
       }, py::arg("rv0"), py::arg("thetas"), py::arg("lmax"),
          py::arg("tfs"), py::arg("step")=10.0, "积分伴随前向（球谐）：返回 DeepFlow。");
       m.def("daDeepBackwardRBF", [](const DeepFlow &fl, const std::vector<Vector6d> &gx,
                                     const std::vector<double> &gP,
                                     const std::vector<std::array<double,3>> &centers, double s){
            setRBFParams(centers, s);
            Vector6d gx0; std::vector<double> gt;
            daDeepBackward(fl, gx, gP, gx0, gt);
            return py::make_tuple(gx0, gt);
       }, py::arg("flow"), py::arg("gx_epoch"), py::arg("gPhi_epoch"),
          py::arg("centers"), py::arg("s"), "积分伴随反向（RBF）：返回 (∂L/∂x0, ∂L/∂θ)。");
       m.def("daDeepBackwardSH", [](const DeepFlow &fl, const std::vector<Vector6d> &gx,
                                    const std::vector<double> &gP, int lmax){
            setSHParams(lmax);
            Vector6d gx0; std::vector<double> gt;
            daDeepBackward(fl, gx, gP, gx0, gt);
            return py::make_tuple(gx0, gt);
       }, py::arg("flow"), py::arg("gx_epoch"), py::arg("gPhi_epoch"), py::arg("lmax"),
          "积分伴随反向（球谐）：返回 (∂L/∂x0, ∂L/∂θ)。");
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

