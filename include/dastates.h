#ifndef __DA_ODE_H__
#define __DA_ODE_H__

#include <dace/dace.h>
#include <cmath>
#include <fstream>
// #include "kepler.h"
#include <Eigen/Core>
#define printEachStepPertub_da false
#define printTotalAccleration_da false
typedef Eigen::Matrix<double, 6,1> Vector6d;
using namespace std; 
using namespace DACE;

namespace bddd{
    const double MU = 3.986004415e14;  // Gravitational parameter (m^3/s^2)
    const double RE = 6378.137e3;      // Earth's radius (m)
    const double J2 = 1082.626690598e-6;    // J2 harmonic
    const double J3 = 2.532435345754e-6 ;    // J3 harmonic
    // const double J3 = 0 ;    // J3 harmonic
    const double J4 = 1.619331205072e-6 ;    // J4 harmonic
    // const double J4 = 0 ;    // J4 harmonic
    const double epsilon = 3.35281317789691e-3;
    const double OMEGA_EARTH = 7.2921159e-5; // 地球自转角速度，单位为rad/s
}

// rho0 =3.245746e-4 kg/km^3
// A 2m^2 = 2e-6km^2
// Cd 2.2
// m 1000
// km-1, 改成单位m，需要乘1e3
template<typename T>
AlgebraicVector<T> TBPfull(AlgebraicVector<T> x, double t,T beta, 
                            double mu = bddd::MU/1e9, double Re =bddd::RE/1e3, double rhoCdA_m = 1.42812824E-12,double h0 = 530,double H0=65.18534) ;

// 增广状态版 RHS：x = [r(3), v(3), kappa]，kappa 为阻力缩放参数（第 7 个状态，导数恒为 0），
// 阻力项用 x[6] 代替 TBPfull 的 double beta 参数。返回 7 维时间导数。
template<typename T>
AlgebraicVector<T> TBPfull_param(AlgebraicVector<T> x, double t, double arg1);

// 增广状态 DA 传播：把阻力参数 kappa 升为第 7 个 DA 变量（kappa' = 0），
// 在 7 个变量上积分到 order 阶，导出 6 个输出状态的密集泰勒系数。
// 系数对展开中心（x0, kappa）的导数由更高一阶系数给出（移位恒等式），
// 故不需反向 tape / 伴随，也不改 DACE 内核。
// coeffs: 长度 6*nmono，按 [输出状态 i][单项式 k] 排布；
// mons:   nmono 个指数向量，每个长度 7（与 dadiff 单调一致：按总阶递增、同阶字典序）。
Vector6d daJ234DragAugCoeffs(const Vector6d &rv0, double kappa0, double tf,
                             int order, double step,
                             std::vector<double> &coeffs,
                             std::vector<std::vector<unsigned int>> &mons);
// 内部运算单位为km，和大气密度*面积的单位一样，
Vector6d EigenwarpDAOrbitJ234DragODE(const Vector6d &rv0, double t, double arg1,bool J234);
// Exercise 6.2.1: 3/8 rule RK4 integrator
template<typename T> T rk4( T x0, double t0, double t1, T (*f)(T,double,double) ,double arg1, double hmax);
template<typename T> T rk4b( T x0, double t0, double t1, T (*f)(T,double,double) ,double arg1, double hmax);
// 函数内部计算单位为km
Vector6d daJ234DragRV_RK4Step(const Vector6d &rv0, Eigen::Ref<Eigen::Matrix<double, 6, 6>> Phi0f, double tf,
                                    double scale_rhoCdA_m=1,bool givePhi=false, double step=1.0,int order=1,double scale=0.01);

class NominalErrorProp{
    public: 
        NominalErrorProp(const Vector6d &rv0,int order=1);
        ~NominalErrorProp();
        void updateX0(const Vector6d &rv0);
        Vector6d propNomJ234Drag( Eigen::Ref<Eigen::Matrix<double, 6, 6>> Phi0f, double tf,bool givePhi=true, double step=1.0);
        Vector6d bkpropNomJ234Drag( Eigen::Ref<Eigen::Matrix<double, 6, 6>> Phi0f, double tf,bool givePhi=true, double step=1.0);
        Vector6d evaldXf(const Vector6d &drv0,double scale_rhoCdA_m=1.0);
        Vector6d evaldXp(const Vector6d &drv0,double scale_rhoCdA_m=1.0);
    private:
        AlgebraicVector<DA>  xf,x0,xp;
};

class NominalErrorPropNOE{
    public: 
        NominalErrorPropNOE(const Vector6d &oe0,int order=1);
        ~NominalErrorPropNOE();
        void updateX0(const Vector6d &oe0);
        Vector6d propNomJ234Drag( Eigen::Ref<Eigen::Matrix<double, 6, 6>> Phi0f, double tf,bool givePhi=true, double step=1.0);
        Vector6d bkpropNomJ234Drag( Eigen::Ref<Eigen::Matrix<double, 6, 6>> Phi0f, double tf,bool givePhi=true, double step=1.0);
        Vector6d evaldXf(const Vector6d &doe0,double scale_rhoCdA_m=1.0);
        Vector6d evaldXp(const Vector6d &doe0,double scale_rhoCdA_m=1.0);
    private:
        AlgebraicVector<DA>  xf,x0,xp;
};
#endif

