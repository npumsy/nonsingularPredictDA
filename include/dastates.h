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
AlgebraicVector<T> TBPfull(AlgebraicVector<T> x, double t,double beta, 
                            double mu = bddd::MU/1e9, double Re =bddd::RE/1e3, double rhoCdA_m = 1.42812824E-12,double h0 = 530,double H0=65.18534) ;
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

