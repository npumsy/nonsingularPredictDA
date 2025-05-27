#include <functional>
#include <Eigen/Dense>
// #define BOOST_MATH_INSTRUMENT true

#ifndef ICATT_KEPLER_H
#define ICATT_KEPLER_H
namespace kepler {
    double period(double sma, double mu);

    double newton(double x0, std::function<double(double)> const &func, std::function<double(double)> const &deriv,
                  int maxiter, double tol);

    double mean2ecc(double M, double ecc);

    double ecc2true(double E, double ecc);
    void benchmark(int times);

    using namespace Eigen;
    /*State: Computes the satellite state vector from osculating Keplerian
       elements for elliptic orbits

    Inputs:
    gm        Gravitational coefficient
    Kep       Keplerian elements (a,e,i,Omega,omega,M) with
                a      Semimajor axis 
                e      Eccentricity 
                i      Inclination [rad]
                Omega  Longitude of the ascending node [rad]
                omega  Argument of pericenter  [rad]
                M      Mean anomaly at epoch [rad]
    dt        Time since epoch

    Output:
    Y         State vector (x,y,z,vx,vy,vz)

    Notes:
    The semimajor axis a=Kep(0), dt and gm must be given in consistent
    units, e.g. [m], [s] and [m^3/s^2]. The resulting units of length and
    velocity are implied by the units of gm, e.g. [m] and [m/s].

    Last modified:   2018/01/27   M. Mahooti
    */
    // Computes the satellite state vector from osculating Keplerian elements for elliptic orbits
    Eigen::VectorXd State(double gm, const Eigen::VectorXd& eles, double dt);
    // Rotation matrices
    Eigen::Matrix3d R_x(double angle);
    Eigen::Matrix3d R_y(double angle);
    Eigen::Matrix3d R_z(double angle);
}
namespace kep3
{
    typedef Eigen::Matrix<double, 6,1> Vector6d;
    typedef Eigen::Matrix<double, 1,6> Vector6dr;
    using Eigen::Vector3d;
    using Eigen::RowVector3d;
    typedef Eigen::Matrix<double, 6,6> mat66;
    typedef Eigen::Matrix<double, 3,6> mat36;
    typedef Eigen::Matrix3d mat33;

// In terms of the eccentric anomaly difference (DE)
// -------------------------------------------
inline double kepDE(double DE, double DM, double sigma0, double sqrta, double a, double R)
{
    return -DM + DE + sigma0 / sqrta * (1 - std::cos(DE)) - (1 - R / a) * std::sin(DE);
}

inline double d_kepDE(double DE, double sigma0, double sqrta, double a, double R)
{
    return 1 + sigma0 / sqrta * std::sin(DE) - (1 - R / a) * std::cos(DE);
}
// 假设已经定义了 v0 和 dv0
inline Vector6dr _dot(const Vector3d& v0, const mat36& dv0) {
    return v0.transpose() * dv0;
}
inline mat36 _dot(const RowVector3d& v0, const Vector6dr& dv0) {
    return v0.transpose() * dv0;
}
inline double sign(double x) {
    if (x == 0)
        return 0;
    else {
        return (x > 0) - (x < 0);
    }
}
struct Cf1f2{double f,df;};
Cf1f2 kepDE_dKepDE(double DE, double DM_cropped, double sigma0, double sqrta, double a, double R0);

/// Lagrangian propagation
/**
 * This function propagates an initial Cartesian state for a time t assuming a
 * central body and a keplerian motion. Lagrange coefficients are used as basic
 * numerical technique. All units systems can be used, as long
 * as the input parameters are all expressed in the same system.
 */
Vector6d propagate_lagrangian(
    const Vector3d &pos0, const Vector3d &vel0, 
    const double tof, const double mu, bool stm,
    Eigen::Ref<Eigen::Matrix<double, 6, 6>> Phi0f);

template <class F, class T>
T newton_raphson_iterate(F f, T guess, T min, T max, int digits, uint& max_iter);
// template double newton_raphson_iterate(Cf1f2 (*f)(double), 
//                                 double guess, double min, double max, int digits, uint& max_iter);

template <class F, class T>
void handle_zero_derivative(F f,
                            T& last_f0,
                            const T& f0,
                            T& delta,
                            T& result,
                            T& guess,
                            const T& min,
                            const T& max);
// Here we take the lagrangian coefficient expressions for rf and vf as function of r0 and v0, and manually,
// differentiate it to obtain the state transition matrix.
mat66 stm_lagrangian(const Vector3d &pos0, const Vector3d &vel0, double tof, // NOLINT
                                      double mu,                                                        // NOLINT
                                      double R0, double Rf, double energy,                              // NOLINT
                                      double sigma0,                                                    // NOLINT
                                      double a, double s0, double c0,                                   // NOLINT
                                      double DX, double F, double G, double Ft, double Gt);


}

namespace osculating{
    // https://github.com/decenter2021/osculating2mean/
    // Vallado, D.A., 1997. Fundamentals of astrodynamics and applications. McGraw-Hill.
        // Constants
const double MU = 3.986004415e14;  // Gravitational parameter (m^3/s^2)
const double RE = 6378.137e3;      // Earth's radius (m)
const double J2 = 1082.626690598e-6;    // J2 harmonic
const double J3 = 2.532435345754e-6 ;    // J3 harmonic
// const double J3 = 0 ;    // J3 harmonic
const double J4 = 1.619331205072e-6 ;    // J4 harmonic
// const double J4 = 0 ;    // J4 harmonic
const double epsilon = 3.35281317789691e-3;
const double OMEGA_EARTH = 7.2921159e-5; // 地球自转角速度，单位为rad/s
// % Non-singular orbital elements are employed: 
//      a: semi-major axis
//      lambda: mean anomaly + argument of perigee
//      ex: e*cos(argument of perigee)
//      ey: e*sin(argument of perigee)
//      i: inclination
//      Omega: longitude of ascending node
    // 球谐地球重力势将振荡轨道单元转换为平均轨道单元。

// 输入: 接受一个 6 元素向量 x，包含位置和速度。
// 计算轨道元素: 使用与 MATLAB 代码相同的公式计算轨道元素。
// 输出: 返回一个包含轨道元素的 6 元素向量。
Eigen::VectorXd rv2OEOsc(const Eigen::VectorXd &x);

// 使用 OEOsc2rv 函数计算位置-速度向量 x。
// 输入: 轨道元素 OE，最大迭代次数 MaxIt 和容差 epsl。
// 输出: 包含位置和速度的 6 元素向量 x。
Eigen::VectorXd OEOsc2rv(const Eigen::VectorXd &OE, int MaxIt, double epsl);
// 实现第一类虚变量的贝塞尔函数
double modified_bessel_first_kind(int n, double x);
double gammaFun(int n);
double factorial(int n);
// 输入: 平均近点角 M，离心率 e，最大迭代次数 MaxIt 和容差 epsl。
// 输出: 偏近点角 E。
double KepEqtnE(double M, double e, int MaxIt , double epsl );

void testOEosc();

//输入: 非奇异轨道元素向量 OEMean。
// 输出: Eckstein-Ustinov 摄动的 6 元素向量 EUPerturbations。
Eigen::VectorXd EcksteinUstinovPerturbations(const Eigen::VectorXd &OEMean);


// 初始化 mean 轨道元素 OEMean 为 osculating 元素。
//  Input: osculating OE: a, u (mean anomaly + arg perigee), ex, ey, i, longitude of asceding node
//  Output: mean OE:      a, u (mean anomaly + arg perigee), ex, ey, i, longitude of asceding node
Eigen::VectorXd OEOsc2OEMeanEU(const Eigen::VectorXd &OEosc, int MaxIt = 100, double epslPos = 1e-3, double epslVel = 1e-4);

// 转化 osculating 元素。为mean 轨道元素 OEMean 
Eigen::VectorXd OEMeanEU2OEOsc(const Eigen::VectorXd &OEMean);

// 由于 $J_2$ 引起的 Eckstein-Ustinov 一阶扰动,从平均轨道元素转换为振荡轨道元素
int testOEMean();

int testAll();

// 平均轨道要素的长期变化
// OE: a, lambda(Mean anomaly + arg perigee), ex, ey, i, longitude of asceding node
Eigen::VectorXd OscElemsLongpropagate( double tf, const Eigen::VectorXd OEm, double RE, double mu, double tol,double J2,double J3,double J4);

// 大气阻力除以这2项
// double rho0 = 3.003075e-4, double CDA_m=0.0044,
Eigen::VectorXd OscElemsLongDrag_BCrho0( const Eigen::VectorXd OEm,double dt,    double rp0, double H0, double mu);
inline double Dragbeta(double rho0 = 3.003075e-4, double CDA_m=0.0044){
    return rho0*CDA_m;
}
//  Input: osculating OE: a, u (mean anomaly + arg perigee), ex, ey, i, longitude of asceding node
Eigen::MatrixXd Jacobian_RV2OscElems(const Eigen::VectorXd OE, const Eigen::VectorXd &x,  double mu, double tol);

Eigen::Matrix3d PQW2GCRF(double Omega, double omega, double i);
Eigen::MatrixXd STM_ONsElemsWarppedByCOE(const Eigen::VectorXd OE, double tf, double Re, double mu, double J2);

int testOEMeanSTM();
}

namespace oscstm{
// OE   平纬度幅角 M + w
// Osc  纬度幅角   f + w
// 这两种轨道要素在圆轨道下仍然是奇异的
// Input
    // double a = OE(0), u = OE(1);
    // double ex = OE(2), double ey = OE(3);
    // double i = OE(4), double Omega = OE(5);
// Output
    // double a0 = ICSc(0), argLat0 = ICSc(1), i0 = ICSc(2);
    // double q10 = ICSc(3), q20 = ICSc(4), RAAN0 = ICSc(5);
Eigen::VectorXd OE2Osc(const Eigen::VectorXd &OE);
Eigen::VectorXd Osc2OE(const Eigen::VectorXd &Osc);

// Calculation of the state transition matrix
// for the mean non-singular variables with perturbation by J2
// input :
//    t_in(1) = t0
//    t_in(2) = t
//    ICs_c(1)= a0,    ICs_c(2)= theta0,    ICs_c(3)= i0
//    ICs_c(4)= q10,    ICs_c(5)= q20,    ICs_c(6)= Omega0
// output :
//    6x6 state transition matrix, phi_J2
//    cond_c(1)= a,    cond_c(2)= theta,    cond_c(3)= i
//    cond_c(4)= q1,    cond_c(5)= q2,    cond_c(6)= Omega
Eigen::MatrixXd OscMeanElemsSTM(double J2, double tf, const Eigen::VectorXd ICSc, double Re, double mu, double tol);

// 平均轨道要素的长期变化
// OE: a, theta(True anomaly + arg perigee), ex, ey, i, longitude of asceding node
Eigen::VectorXd OscMeanElemspropagate(double J2, double tf, const Eigen::VectorXd ICSc, double Re, double mu, double tol);


// Calculation of true longitude theta = f + w
// from mean longitude lambda = M + w
// input :
//    lambda = mean longitude = M + w
//    q1 = e * cos(w)
//    q2 = e * sin(w)
// output :
//    theta = true longitude = f + w
//    F = eccentric longitude = E + w
double lam2theta(double lambda, double q1, double q2, double Tol, double &F);

// Calculation of mean longitude lambda = M + w
// from true longitude theta = f + w
// input :
//    a = semi major axis
//    theta = true longitude
//    q1 = e * cos(w)
//    q2 = e * sin(w)
// output :
//    lambda = mean longitude
double theta2lam(double a, double theta, double q1, double q2);

// form mean to osculating element with the perturbation by only J2
// input :
//    mean_c(1) = a_mean
//    mean_c(2) = theta_mean
//    mean_c(3) = i_mean
//    mean_c(4) = q1_mean
//    mean_c(5) = q2_mean
//    mean_c(6) = Omega_mean
// output :
//    osc
Eigen::VectorXd OscMeanToOsculatingElements(double J2, Eigen::VectorXd meanElems, double Re, double mu);
// formation matrix D_J2 in closed form between mean and osculating new set of elements
// with the perturbation by only J2
// form mean to osculating element with the perturbation by only J2
// input :
//    mean_c(1) = a_mean
//    mean_c(2) = theta_mean
//    mean_c(3) = i_mean
//    mean_c(4) = q1_mean
//    mean_c(5) = q2_mean
//    mean_c(6) = Omega_mean
// output :
//    D_J2 = I + (-J2*Re^2)*(D_lp+D_sp1+D_sp2) = 6x6 transformation matrix D_J2
//    osc
Eigen::VectorXd DMeanToOsculatingElements(double J2, Eigen::VectorXd meanElems, double Re, double mu,Eigen::Ref<Eigen::Matrix<double, 6, 6>> DJ2);


// System_matrix, Sigma in osculating element with perturbation by J2
// input :
//    elems(1)= a
//    elems(2)= theta
//    elems(3)= i
//    elems(4)= q1
//    elems(5)= q2
//    elems(6)= Omega
// output :
//    6x6 system matrix
//    Sigma = A + (3*J2*Re^2)*B
Eigen::MatrixXd SigmaMatrix(double J2, Eigen::VectorXd elems, double Re, double mu);


// Calculation of the inverse of system_matrix at t0, inv_AA_BB_t0
// with perturbed osculating elements by J2
// input :
//    elems(1)= a0
//    elems(2)= theta0
//    elems(3)= i0
//    elems(4)= q10
//    elems(5)= q20
//    elems(6)= Omega0
// output :
//    6x6 inverse of system matrix
//    SigmaInverse = inv(T)*(T*inv(A+gamma*B))
Eigen::MatrixXd SigmaInverseMatrix(double J2, Eigen::VectorXd elems, double Re, double mu, double tol);

void printmat66(Eigen::MatrixXd mat);
int testSigmaInverseMatrix();
int testOscMeanToOsculatingElements();
int testSigmaMat();

int testlam2theta();
int testOscMeanSTM();


}

#endif //ICATT_KEPLER_H
