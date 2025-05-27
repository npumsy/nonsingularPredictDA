#ifndef __INT_H__
#define __INT_H__
#include <functional>
#include <Eigen/Dense>
#define printEachStepPertub false
#define printTotalAccleration false
/* Include Files */
typedef signed char int8_T;
typedef unsigned char uint8_T;
typedef double Real;
typedef int Index;

typedef void (*DerivFunction)(double, const double*, double*);

const Real tableau[8][7]{
  {       0.0,           0.0,            0.0,            0.0,             0.0,        0.0,      0.0 },
  {   1.0/4.0,       1.0/4.0,            0.0,            0.0,             0.0,        0.0,      0.0 },
  {   3.0/8.0,      3.0/32.0,       9.0/32.0,            0.0,             0.0,        0.0,      0.0 },
  { 12.0/13.0, 1932.0/2197.0, -7200.0/2197.0,  7296.0/2197.0,             0.0,        0.0,      0.0 },
  {       1.0,   439.0/216.0,           -8.0,   3680.0/513.0,   -845.0/4104.0,        0.0,      0.0 },
  {   1.0/2.0,     -8.0/27.0,            2.0, -3544.0/2565.0,   1859.0/4104.0, -11.0/40.0,      0.0 },
  {       0.0,    16.0/135.0,            0.0, 6656.0/12825.0, 28561.0/56430.0,  -9.0/50.0, 2.0/55.0 },
  {       0.0,    25.0/216.0,            0.0,  1408.0/2565.0,   2197.0/4104.0,   -1.0/5.0,      0.0 }
};

#define B(row,col) (tableau[row-1][col-1])
#define FOREACH for (Index j = 0; j < N; j++)
#define TK (x[0])
#define XK (x[j+1])
#define YK (y[j])
#define K1 (k1[j])
#define K2 (k2[j])
#define K3 (k3[j])
#define K4 (k4[j])
#define K5 (k5[j])
#define K6 (k6[j])

/*
* @brief 单步求解形如 y'=f(x,y) 的常微分方程。
* @param void RK45(void (*func)(double, double[], double[]), 指向求解的常微分方程 f(x, y) 的函数指针
* @param[in&out] x 函数在初始点 t 处的值,。
* @param[in] t0 积分时间起点。
* @param[in] dt 给定步长。
* @param[out] errorEstimate 误差，用于步长控制。
* @return 无
*/
// template <int N>
void RK45(DerivFunction func, double x[], double t, double dt,  double errorEstimate[],uint8_T N);
// template <int N>
void RKF87(DerivFunction func, double y[], double x0, double h, double out[],uint8_T N);

const int ATTEMPTS = 12;
const double MIN_SCALE_FACTOR = 0.125;
const double MAX_SCALE_FACTOR = 4.0;

// Wrapper function for RK45 with adaptive step size
/*
* @brief 自适应步长的数值积分算法，用于求解形如 y'=f(x,y) 的常微分方程。
* @param Deriv 指向求解的常微分方程 f(x, y) 的函数指针，其原型为 void Deriv(double x, double y[], double dy[])
* @param[in] y0 初始条件，即函数在初始点 x 处的值。
* @param[in] t0 积分起点。
* @param[in] h_initial 初始步长。
* @param[in] tf 积分终点。
* @param[in] tolerance 误差容限。
* @param[out] y_f 在积分终点 xmax 处的函数值，即解的近似值。
* @param[out] flag 状态码，表示积分的结果状态。通常，0 表示成功，其他值表示错误或警告的特定情况。
* @param[out] h_next 下一个步长，即在下一次积分时应使用的步长。
* @return 无
*/
// 数组个数 uint8_T N 弃用template <int N> 
void Runge_Kutta_Fehlberg_7_8(DerivFunction func, const double y0[], double x, double h, double xmax, double tolerance,
                double y[], int& out, double& h_next,uint8_T N);
// 定步长的rk4阶算法
void Runge_Kutta_4(std::function<void(double, const double[], double[])> Deriv,
                   const double y0[], double t0, double h, double tf, double y_f[], uint8_T N);
// template <int N>
void Runge_Kutta_45(DerivFunction func, const double y0[], double t0, double h_initial, double tf, double tolerance, 
                double y_f[], int& flag, double& h_next,uint8_T N);
void Runge_Kutta_45(std::function<void(double, const double[], double[])> func, const double y0[], double t0, double h_initial, double tf, double tolerance, 
                double y_f[], int& flag, double& h_next,uint8_T N);
inline double norm(double err[],int N){
    double sum=0;
    for(int i=0;i<N;i++)sum+=(err[i]*err[i]);
    return sqrt(sum);
}
typedef Eigen::Matrix<double, 6,1> Vector6d;
typedef Eigen::Matrix<double, 1,6> Vector6dr;
typedef Eigen::Matrix<float, 6,1> Vector6f;
typedef Eigen::Matrix<float, 1,6> Vector6fr;
typedef Eigen::RowVector3f Vec3fr;
typedef Eigen::Vector3f Vec3f;
typedef Eigen::Matrix<double, 6,6> mat66;
typedef Eigen::Matrix<double, 3,6> mat36;
typedef Eigen::Matrix3d mat33;

/// @brief 单步求解轨道,内部计算单位km
/// @param t0 
/// @param rv0 设定状态初值
/// @param tf 
/// @return    到达时刻的状态
Vector6d intJ234DragRV_RK4Step(const Vector6d &rv0, double tf,double rhoCdA_m = 1.42812824E-12,bool J234=true, double step=1.0);
Vector6d EigenwarpIntOrbitJ234DragODE(const Vector6d &rv0, double tf,double rhoCdA_m = 1.42812824E-12,bool J234=true);
void yprime (double t, const double y[6],double rhoCdA_m,bool J234, double yp[6]);
Eigen::Vector3d J234_pert2(const Eigen::Vector3d& posECI, double r);

// rho0 =3.245746e-4 kg/km^3
// A 2m^2 = 2e-6km^2
// Cd 2.2
// m 1000
// km-1, 改成单位m，需要乘1e3
Eigen::Vector3d Drag_pert3(const Eigen::Vector3d& vel, const Eigen::Vector3d& pos, double h,double rhoCdA_m = 1.42812824E-12,double rp0 = 530,double H0=65.18534);
// python3 script/testDensity.py 
// 65.18534134312125
// rho0 0.0003245746
// 2.2*2/1000*3.2457e-4*1e-9
#endif