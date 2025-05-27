#include "integrater.h"
#include <cmath>
#include <iostream>
#include <functional>
#include "kepler.h"
/// @brief 单步求解轨道
/// @param t0 
/// @param rv0 设定状态初值
/// @param tf 
/// @return    到达时刻的状态
Vector6d intJ234DragRV_RK4Step(const Vector6d &rv0, double tf,double rhoCdA_m,bool J234, double step){
    const uint8_T N = 6;

    int flag;
    Vector6d rvf;
    std::function<void(double ,const double*, double*)> yprime2 = 
                std::bind(yprime, std::placeholders::_1,std::placeholders::_2, rhoCdA_m, J234, std::placeholders::_3);
    Runge_Kutta_4(yprime2, rv0.data(), 0, tf, step, rvf.data(), N);
    return rvf;
}
Vector6d EigenwarpIntOrbitJ234DragODE(const Vector6d &rv0, double t0,double rhoCdA_m,bool J234){
    const uint8_T N = 6;
    Vector6d rvf;
    yprime(t0, rv0.data(), rhoCdA_m, J234, rvf.data());
    return rvf;
}
void yprime(double t, const double y[6], double rhoCdA_m,bool J234, double yp[6])
{
    double mu =osculating::MU/1e9;    // 引力常数 km^3/s^2
    double Re = osculating::RE/1e3;

    double r = std::sqrt(y[0] /10000* y[0] + y[1] /10000* y[1] + y[2] /10000* y[2])*100;
    // r/=1e3;
    int i;
    for (i = 0; i < 3; i++)
    {
        yp[i] = y[i + 3];
    }
    double common_factor = -mu / pow(r,3);
    double f_eci[3]={0,0,0};
    Eigen::Vector3d a1(0,0,0);
    if(J234){
        a1 = J234_pert2(Eigen::Vector3d(y[0],y[1],y[2]),r);
        for (i = 0; i < 3; i++) f_eci[i]=a1(i);
    }
    a1 = Drag_pert3(Eigen::Vector3d(y[3],y[4],y[5]),Eigen::Vector3d(y[0],y[1],y[2]),(r-Re),rhoCdA_m);
    for (i = 0; i < 3; i++) f_eci[i]+=a1(i);
// if(printEachStepPertub)std::cout<<"Drag force:(m/s^2) "<< (1e3*a1).transpose()<< std::endl;

    yp[3] =common_factor * y[0]  + f_eci[0];
    yp[4] = common_factor * y[1]  + f_eci[1];
    yp[5] = common_factor * y[2]  + f_eci[2];
    if(printTotalAccleration)
        printf("RKdouble---total acc a_x, a_y, a_z(m/s^2)= %1.5f,\t%1.5f,\t%1.5f,\n",1e3*yp[3], 1e3*yp[4], 1e3*yp[5]);
}
Eigen::Vector3d J234_pert2_(const Eigen::Vector3d& posECI, double r) {
    double x = posECI(0);
    double y = posECI(1);
    double z = posECI(2);
    
    double r2 = r * r;
    double r4 = r2 * r2;
    double r3 = r2 * r;

    double z2_r2 = (z / r) * (z / r);
    double z3_r3 = z2_r2*z/r;
    double z4_r4 = z2_r2*z2_r2;

    double Re2_r2 = std::pow(osculating::RE, 2) / r2;
    double Re3_r3 = Re2_r2 * (osculating::RE / r);
    double Re4_r4 = std::pow(Re2_r2, 2);

    Eigen::Vector3d a_J2 = -1.5 * osculating::J2 * osculating::MU * Re2_r2 / r3 * ((1 - 5 * z2_r2) * posECI + 2 * z2_r2 * Eigen::Vector3d(0, 0, 1));
    Eigen::Vector3d a_J3 = -0.5 * osculating::J3 * osculating::MU * Re3_r3 / r4 * (5 * (7 * z3_r3 - 3 * z / r) * posECI + 3 * z * (1 - 7 * z2_r2) * Eigen::Vector3d(0, 0, 1));
    Eigen::Vector3d a_J4 = 5.0 / 8.0 * osculating::J4 * osculating::MU * Re4_r4 / r3 * (35 * (z4_r4 - 6 * z2_r2 + 3) * posECI + 8 * z * (3 - 7 * z2_r2) * Eigen::Vector3d(0, 0, 1) / r2);
    
    return a_J2 + a_J3 + a_J4;
}

Eigen::Vector3d J234_pert2(const Eigen::Vector3d& posECI, double r) {
    double x = posECI(0);
    double y = posECI(1);
    double z = posECI(2);
    
    // return a_J2 + a_J3 + a_J4;
    double r2 = r*r;
    double r3 = r2 * r;
    double Re_r = (osculating::RE/1e3 / r);
    double Re2_r2 = Re_r*Re_r;
    double Re3_r3 = Re2_r2 * Re_r;
    double Re4_r4 = Re3_r3 * Re_r;
    double z_r = z/r;
    double z2_r2 = z_r * z_r;
    double z3_r3 = z2_r2 * z_r;
    double z4_r4 = z3_r3 * z_r;

    double common_factor = -osculating::MU/1e9  / r3;

    double J2_term = (3.0 / 2.0) * osculating::J2 * Re2_r2 * (1.0 - 5.0 * z2_r2);
    double J3_term = (5.0 / 2.0) * osculating::J3 * Re3_r3 * (3.0 * z_r - 7.0 * z3_r3);
    double J4_term = (5.0 / 8.0) * osculating::J4 * Re4_r4 * (3.0 - 42.0 * z2_r2 + 63.0 * z4_r4);

    Eigen::Vector3d acc_J234;
    acc_J234(0) = common_factor * x * ( J2_term + J3_term - J4_term);
    acc_J234(1) = common_factor * y * ( J2_term + J3_term - J4_term);

    J2_term = (3.0 / 2.0) * osculating::J2 * Re2_r2 * (3.0 - 5.0 * z2_r2);
    J3_term = (5.0 / 2.0) * osculating::J3 * Re3_r3 * (6.0 *z_r - 7.0 * z3_r3 - (3.0 / 5.0) * r / z);
    J4_term = (5.0 / 8.0) * osculating::J4 * Re4_r4 * (15.0 - 70.0 * z2_r2 + 63.0 * z4_r4);

    acc_J234(2) = common_factor * z * ( J2_term + J3_term - J4_term);

    if(printEachStepPertub)std::cout<<"J234 force: (m/s^2)"<< (1e3*(acc_J234) ).transpose()<< std::endl;
    
    // if(1)
    //     for(int i=0;i<3;i++){
    //         printf("RKdouble ERROR---a_J2, a_J3, a_J4(m/s^2)= %1.5e,\t%1.5e,\t%1.5e,\n",a_J2[i],a_J3[i],a_J4[i]);
        //     }
    return acc_J234;
    
    // return a_J2 ;
}

Eigen::Vector3d J234_pert2__(const Eigen::Vector3d& posECI, double r) {
    double x = posECI(0);
    double y = posECI(1);
    double z = posECI(2);
    
    double r2 = r*r;
    double r3 = r2 * r;
    double r5 = r3 * r2;
    double Re_r = (osculating::RE / r);
    double Re2_r2 = Re_r*Re_r;
    double Re3_r3 = Re2_r2 * Re_r;
    double Re4_r4 = Re3_r3 * Re_r;
    double z_r = z/r;
    double z2_r2 = (z / r) * (z / r);
    double z3_r3 = z2_r2 * z_r;
    double z4_r4 = z3_r3 * z_r;

    double common_factor = osculating::MU  / r3;

    // J2项摄动加速度
    Eigen::Vector3d a_J2 = osculating::J2 * common_factor * Re2_r2 *
                           Eigen::Vector3d(
                               x * (5 * z2_r2 - 1),
                               y * (5* z2_r2 - 1),
                               z * (5 * z2_r2 - 3)
                           );

    // J3项摄动加速度
    Eigen::Vector3d a_J3 = osculating::J3 *common_factor * Re3_r3/r*
                           Eigen::Vector3d(
                               x * (10 * z3_r3 - 3 * z_r),
                               y * (10 * z3_r3 - 3 * z_r),
                               z * (4 * z3_r3 - 1.5 * z_r) * z
                           );

    // J4项摄动加速度
    Eigen::Vector3d a_J4 = osculating::J4 * common_factor * Re4_r4 *
                           Eigen::Vector3d(
                               x * (35 * z4_r4 - 30 * z2_r2 + 3),
                               y * (35 * z4_r4 - 30 * z2_r2 + 3),
                               z * (35 * z4_r4 - 15 * z2_r2 + 1) * z
                           );
    // 总加速度
    Eigen::Vector3d acc_J234_ = a_J2 + a_J3 + a_J4;
    return acc_J234_;
    
    // return a_J2 ;
}

Eigen::Vector3d Drag_pert3(const Eigen::Vector3d& vel, const Eigen::Vector3d& pos, double h, double rhoCdA_m,double h0,double H0) {
    Eigen::Vector3d omega_earth(0, 0, osculating::OMEGA_EARTH);
    Eigen::Vector3d rel_vel = vel - omega_earth.cross(pos);

    double v = rel_vel.norm();
    Eigen::Vector3d drag_acc = -0.5 * rhoCdA_m *exp(-(h-h0)/H0)* v * rel_vel;
//     printf("RKdouble---v, v_x, v_y, v_z(m/s^2)=%1.5e,\t %1.5e,\t%1.5e,\t%1.5e,\n",v,rel_vel(0),rel_vel(1),rel_vel(2));
if(printEachStepPertub)printf("RKdouble---Drag a_x, a_y, a_z(m/s^2)= %1.5e,\t%1.5e,\t%1.5e,\n",1e3*drag_acc(0),1e3*drag_acc(1),1e3*drag_acc(2));
    
    
    // if(1)
    //     for(int i=0;i<3;i++){
    //         printf("a_drag(m/s^2)= %1.5e\n",drag_acc[i]);
    //     }
    return drag_acc;
}

// bool CONFIGB_TXT_ARRAY[7]= {
//     #include "config_bool.txt"
//     };
// bool now_debugging =CONFIGB_TXT_ARRAY[0];
// const float CONFIGD_TXT_ARRAY[10]= {
//         #include "config_float.txt"
//     };
// float err_max = CONFIGD_TXT_ARRAY[0];// km
bool now_debugging = false;
float err_max = 1e-5;// km
/*
 * REFERENCE1: https://github.com/recalon/rkf_calon/rkf_calon.cpp
 *  REFERENCE2: https://github.com/lcsamaro/rkf45/blob/master/main.cpp
 *  const Real tableau[8][7]{
 *    {       0.0,           0.0,            0.0,            0.0,             0.0,        0.0,      0.0 },
 *    {   1.0/4.0,       1.0/4.0,            0.0,            0.0,             0.0,        0.0,      0.0 },
 *    {   3.0/8.0,      3.0/32.0,       9.0/32.0,            0.0,             0.0,        0.0,      0.0 },
 *    { 12.0/13.0, 1932.0/2197.0, -7200.0/2197.0,  7296.0/2197.0,             0.0,        0.0,      0.0 },
 *    {       1.0,   439.0/216.0,           -8.0,   3680.0/513.0,   -845.0/4104.0,        0.0,      0.0 },
 *    {   1.0/2.0,     -8.0/27.0,            2.0, -3544.0/2565.0,   1859.0/4104.0, -11.0/40.0,      0.0 },
 *    {       0.0,    16.0/135.0,            0.0, 6656.0/12825.0, 28561.0/56430.0,  -9.0/50.0, 2.0/55.0 },
 *    {       0.0,    25.0/216.0,            0.0,  1408.0/2565.0,   2197.0/4104.0,   -1.0/5.0,      0.0 }
 *  };
/*
* @brief 单步求解形如 y'=f(x,y) 的常微分方程。
* @param void RK45(void (*func)(double, double[], double[]), 指向求解的常微分方程 f(x, y) 的函数指针
* @param[in&out] x 函数在初始点 t 处的值,。
* @param[in] t0 积分时间起点。
* @param[in] dt 给定步长。
* @param[out] errorEstimate 误差，用于步长控制。
* @return 无
*/
void RK45(DerivFunction func, double x[], double t, double dt, double errorEstimate[], uint8_T N) {
  double y[N + 1], dx[N],rk4[N],rk5[N];
/*
Butcher Tableau
time | c
-----+-----
     | 5 order
     | 4 order
*/

  double k1[N], k2[N], k3[N], k4[N], k5[N], k6[N];

  y[0] = t;
  for (Index i = 0; i < N; i++) y[i + 1] = x[i];

  func(t, x, k1);

  const double h = dt;

  TK = t;
  FOREACH XK = YK;
  FOREACH K1 = h * k1[j];

  TK = t + B(2, 1) * h;
  FOREACH XK = YK + B(2, 2) * K1;
  func(TK, x, k2);
  FOREACH K2 = h * k2[j];

  TK = t + B(3, 1) * h;
  FOREACH XK = YK + B(3, 2) * K1 + B(3, 3) * K2;
  func(TK, x, k3);
  FOREACH K3 = h * k3[j];

  TK = t + B(4, 1) * h;
  FOREACH XK = YK + B(4, 2) * K1 + B(4, 3) * K2 + B(4, 4) * K3;
  func(TK, x, k4);
  FOREACH K4 = h * k4[j];

  TK = t + B(5, 1) * h;
  FOREACH XK = YK+ B(5, 2) * K1 + B(5, 3) * K2 + B(5, 4) * K3 + B(5, 5) * K4;
  func(TK, x, k5);
  FOREACH K5 = h * k5[j];

  TK = t + B(6, 1) * h;
  FOREACH XK = YK + B(6, 2) * K1 + B(6, 3) * K2 + B(6, 4) * K3 + B(6, 5) * K4 + B(6, 6) * K5;
  func(TK, x, k6);
  FOREACH K6 = h * k6[j];

for (Index j = 0; j < N; j++){
    rk4[j]=YK+ (B(8,2)*K1 + B(8,4)*K3 + B(8,5)*K4 + B(8,6)*K5);
    dx[j] = (B(7, 2)*K1 + B(7, 4)*K3 + B(7, 5)*K4 + B(7, 6)*K5 + B(7, 7)*K6);
    rk5[j] = YK + dx[j];

    errorEstimate[j] = abs((rk5[j] - rk4[j])/ (x[j] * err_max + dt * abs(K1) * err_max));
    y[j + 1] = rk5[j];
}
    FOREACH x[j] = y[j + 1];
//   FOREACH dx[j] = B(7, 1) * K1 + B(7, 2) * K2 + B(7, 3) * K3 + B(7, 4) * K4 + B(7, 5) * K5 + B(7, 6) * K6;

//   FOREACH XK = YK + dx[j];

//   func(t + dt, x, k[0]);
//   FOREACH errorEstimate[j] = abs(dx[j] / (x[j] * err_max + dt * abs(k[j][0]) * err_max));
}
/*
void RK45(DerivFunction func, double x[], double t, double dt, double errorEstimate[], uint8_T N) 
{
    // RK45 implementation

    // Calculate k1
    double k1[N];
    func(t, x, k1);

    // Calculate k2
    double x2[N];
    for (uint8_T i = 0; i < N; i++) {
        x2[i] = x[i] + 0.5 * dt * k1[i];
    }
    double k2[N];
    func(t + 0.5 * dt, x2, k2);

    // Calculate k3
    double x3[N];
    for (uint8_T i = 0; i < N; i++) {
        x3[i] = x[i] + 0.5 * dt * k2[i];
    }
    double k3[N];
    func(t + 0.5 * dt, x3, k3);

    // Calculate k4
    double x4[N];
    for (uint8_T i = 0; i < N; i++) {
        x4[i] = x[i] + dt * k3[i];
    }
    double k4[N];
    func(t + dt, x4, k4);

    // Calculate k5
    double x5[N];
    for (uint8_T i = 0; i < N; i++) {
        x5[i] = x[i] + dt * (0.25 * k1[i] + 0.75 * k4[i]);
    }
    double k5[N];
    func(t + dt, x5, k5);

    // Calculate k6
    double x6[N];
    for (uint8_T i = 0; i < N; i++) {
        x6[i] = x[i] + dt * (3.0 / 7.0 * k1[i] + 2.0 / 7.0 * k2[i] + 12.0 / 7.0 * k3[i] + 9.0 / 7.0 * k4[i]);
    }
    double k6[N];
    func(t + dt, x6, k6);

    // Calculate error estimate
    double error[N];
    for (uint8_T i = 0; i < N; i++) {
        error[i] = dt * (1.0 / 90.0 * k1[i] - 3.0 / 20.0 * k3[i] + 3.0 / 2.0 * k4[i] - 4.0 / 45.0 * k5[i] + 8.0 / 45.0 * k6[i]);
    }

    // Update error estimate array
    // std::memcpy(errorEstimate, error, N * sizeof(double));
    for (uint8_T i = 0; i < N; i++) {errorEstimate[i] = error[i];}
}
*/


/*Description:                                                              
The Runge-Kutta-Fehlberg method is an adaptive procedure for approxi-  
mating the solution of the differential equation y'(x) = f(x,y) with   
initial condition y(x0) = c.  This implementation evaluates f(x,y)     
thirteen times per step using embedded seventh order and eight order   
Runge-Kutta estimates to estimate the not only the solution but also   
the error.                                                             
The next step size is then calculated using the preassigned tolerance  
and error estimate.                                                    
For step i+1,                                                          
    y[i+1] = y[i] +  h * (41/840 * k1 + 34/105 * k6 + 9/35 * k7         
                    + 9/35 * k8 + 9/280 * k9 + 9/280 k10 + 41/840 k11 ) 
where                                                                  
    k1 = f( x[i],y[i] ),                                                   
    k2 = f( x[i]+2h/27, y[i] + 2h*k1/27),                                  
    k3 = f( x[i]+h/9, y[i]+h/36*( k1 + 3 k2) ),                            
    k4 = f( x[i]+h/6, y[i]+h/24*( k1 + 3 k3) ),                            
    k5 = f( x[i]+5h/12, y[i]+h/48*(20 k1 - 75 k3 + 75 k4)),                
    k6 = f( x[i]+h/2, y[i]+h/20*( k1 + 5 k4 + 4 k5 ) ),                    
    k7 = f( x[i]+5h/6, y[i]+h/108*( -25 k1 + 125 k4 - 260 k5 + 250 k6 ) ), 
    k8 = f( x[i]+h/6, y[i]+h*( 31/300 k1 + 61/225 k5 - 2/9 k6              
                                                            + 13/900 K7) )  
    k9 = f( x[i]+2h/3, y[i]+h*( 2 k1 - 53/6 k4 + 704/45 k5 - 107/9 k6      
                                                        + 67/90 k7 + 3 k8) ), 
    k10 = f( x[i]+h/3, y[i]+h*( -91/108 k1 + 23/108 k4 - 976/135 k5        
                            + 311/54 k6 - 19/60 k7 + 17/6 K8 - 1/12 k9) ), 
    k11 = f( x[i]+h, y[i]+h*( 2383/4100 k1 - 341/164 k4 + 4496/1025 k5     
            - 301/82 k6 + 2133/4100 k7 + 45/82 K8 + 45/164 k9 + 18/41 k10) )  
    k12 = f( x[i], y[i]+h*( 3/205 k1 - 6/41 k6 - 3/205 k7 - 3/41 K8        
                                                    + 3/41 k9 + 6/41 k10) )  
    k13 = f( x[i]+h, y[i]+h*( -1777/4100 k1 - 341/164 k4 + 4496/1025 k5    
                        - 289/82 k6 + 2193/4100 k7 + 51/82 K8 + 33/164 k9 +   
                                                        12/41 k10 + k12) )  
    x[i+1] = x[i] + h.                                                     
                                                                        
The error is estimated to be                                           
    err = -41/840 * h * ( k1 + k11 - k12 - k13)                         
The step size h is then scaled by the scale factor                     
    scale = 0.8 * | epsilon * y[i] / [err * (xmax - x[0])] | ^ 1/7     
The scale factor is further constrained 0.125 < scale < 4.0.           
The new step size is h := scale * h.      
*/ 
void RKF87(void (*func)(double, const double[], double[]), double y[], double x0, double h, double err[],uint8_T N) {
    const double c_1_11 = 41.0 / 840.0, c6 = 34.0 / 105.0, c_7_8 = 9.0 / 35.0, c_9_10 = 9.0 / 280.0;

    const double a2 = 2.0 / 27.0,  a3 = 1.0 / 9.0, a4 = 1.0 / 6.0, a5 = 5.0 / 12.0, a6 = 1.0 / 2.0, a7 = 5.0 / 6.0, a8 = 1.0 / 6.0, a9 = 2.0 / 3.0, a10 = 1.0 / 3.0;

    const double b31 = 1.0 / 36.0, b32 = 3.0 / 36.0, b41 = 1.0 / 24.0, b43 = 3.0 / 24.0, b51 = 20.0 / 48.0, b53 = -75.0 / 48.0, b54 = 75.0 / 48.0, b61 = 1.0 / 20.0, b64 = 5.0 / 20.0, b65 = 4.0 / 20.0, 
    b71 = -25.0 / 108.0, b74 =  125.0 / 108.0, b75 = -260.0 / 108.0, b76 =  250.0 / 108.0, b81 = 31.0/300.0, b85 = 61.0/225.0, b86 = -2.0/9.0, b87 = 13.0/900.0, b91 = 2.0, b94 = -53.0/6.0, 
    b95 = 704.0 / 45.0, b96 = -107.0 / 9.0, b97 = 67.0 / 90.0, b98 = 3.0, b10_1 = -91.0 / 108.0, b10_4 = 23.0 / 108.0, b10_5 = -976.0 / 135.0, b10_6 = 311.0 / 54.0, b10_7 = -19.0 / 60.0,
    b10_8 = 17.0 / 6.0, b10_9 = -1.0 / 12.0, b11_1 = 2383.0 / 4100.0, b11_4 = -341.0 / 164.0, b11_5 = 4496.0 / 1025.0, b11_6 = -301.0 / 82.0, b11_7 = 2133.0 / 4100.0, b11_8 = 45.0 / 82.0,
    b11_9 = 45.0 / 164.0, b11_10 = 18.0 / 41.0, b12_1 = 3.0 / 205.0, b12_6 = - 6.0 / 41.0, b12_7 = - 3.0 / 205.0, b12_8 = - 3.0 / 41.0, b12_9 = 3.0 / 41.0, b12_10 = 6.0 / 41.0, b13_1 = -1777.0 / 4100.0,
    b13_4 = -341.0 / 164.0, b13_5 = 4496.0 / 1025.0, b13_6 = -289.0 / 82.0, b13_7 = 2193.0 / 4100.0, b13_8 = 51.0 / 82.0, b13_9 = 33.0 / 164.0, b13_10 = 12.0 / 41.0;

    double k1[N], k2[N], k3[N], k4[N], k5[N], k6[N], k7[N], k8[N], k9[N], k10[N], k11[N], k12[N], k13[N];
    double err_factor = -41.0 / 840.0;
    double h2_7 = a2 * h;

    func(x0, y, k1);

    double temp1[N];
    for (int i = 0; i < N; ++i) {
        temp1[i] = y[i] + h2_7 * k1[i];
    }

    func(x0 + h2_7, temp1, k2);

    double temp2[N];
    for (int i = 0; i < N; ++i) {
        temp2[i] = y[i] + h * (b31 * k1[i] + b32 * k2[i]);
    }

    func(x0 + a3 * h, temp2, k3);

    double temp3[N];
    for (int i = 0; i < N; ++i) {
        temp3[i] = y[i] + h * (b41 * k1[i] + b43 * k3[i]);
    }

    func(x0 + a4 * h, temp3, k4);

    double temp4[N];
    for (int i = 0; i < N; ++i) {
        temp4[i] = y[i] + h * (b51 * k1[i] + b53 * k3[i] + b54 * k4[i]);
    }

    func(x0 + a5 * h, temp4, k5);

    double temp5[N];
    for (int i = 0; i < N; ++i) {
        temp5[i] = y[i] + h * (b61 * k1[i] + b64 * k4[i] + b65 * k5[i]);
    }

    func(x0 + a6 * h, temp5, k6);

    double temp6[N];
    for (int i = 0; i < N; ++i) {
        temp6[i] = y[i] + h * (b71 * k1[i] + b74 * k4[i] + b75 * k5[i] + b76 * k6[i]);
    }

    func(x0 + a7 * h, temp6, k7);

    double temp7[N];
    for (int i = 0; i < N; ++i) {
        temp7[i] = y[i] + h * (b81 * k1[i] + b85 * k5[i] + b86 * k6[i] + b87 * k7[i]);
    }

    func(x0 + a8 * h, temp7, k8);

    double temp8[N];
    for (int i = 0; i < N; ++i) {
        temp8[i] = y[i] + h * (b91 * k1[i] + b94 * k4[i] + b95 * k5[i] + b96 * k6[i] + b97 * k7[i] + b98 * k8[i]);
    }

    func(x0 + a9 * h, temp8, k9);

    double temp9[N];
    for (int i = 0; i < N; ++i) {
        temp9[i] = y[i] + h * (b10_1 * k1[i] + b10_4 * k4[i] + b10_5 * k5[i] + b10_6 * k6[i] + b10_7 * k7[i] + b10_8 * k8[i] + b10_9 * k9[i]);
    }

    func(x0 + a10 * h, temp9, k10);

    double temp10[N];
    for (int i = 0; i < N; ++i) {
        temp10[i] = y[i] + h * (b11_1 * k1[i] + b11_4 * k4[i] + b11_5 * k5[i] + b11_6 * k6[i] + b11_7 * k7[i] + b11_8 * k8[i] + b11_9 * k9[i] + b11_10 * k10[i]);
    }

    func(x0 + h, temp10, k11);

    double temp11[N];
    for (int i = 0; i < N; ++i)
    {
        temp11[i] = y[i] + h * (b12_1 * k1[i] + b12_6 * k6[i] + b12_7 * k7[i] + b12_8 * k8[i] + b12_9 * k9[i] + b12_10 * k10[i]);
    }

    func(x0, temp11, k12);

    double temp12[N];
    for (int i = 0; i < N; ++i) {
        temp12[i] = y[i] + h * (b13_1 * k1[i] + b13_4 * k4[i] + b13_5 * k5[i] + b13_6 * k6[i] + b13_7 * k7[i] + b13_8 * k8[i] + b13_9 * k9[i] + b13_10 * k10[i] + k12[i]);
    }

    func(x0 + h, temp12, k13);

    for (int i = 0; i < N; ++i) {
        y[i] += h * (c_1_11 * (k1[i] + k11[i]) + c6 * k6[i] + c_7_8 * (k7[i] + k8[i]) + c_9_10 * (k9[i] + k10[i]));
        err[i] = err_factor * (k1[i] + k11[i] - k12[i] - k13[i]);
    }
}

template<typename T>T min(T x,T y){return x<y?x:y;}
template<typename T>T max(T x,T y){return x>y?x:y;}
template<typename T>T abs(T x){return x>0?x:-x;}
template<typename T>T sqr(T x){return x*x;}
template <int N> double norm(double x[N]){
    double r;
    for(int i=0;i<N;i++)r+=x[i]*x[i];
    return sqrt(r);
}
/*
* @brief 自适应步长的数值积分算法，用于求解形如 y'=f(x,y) 的常微分方程。
* @param[in] y0 初始条件，即函数在初始点 x 处的值。
* @param[in] x 积分起点。
* @param[in] h 初始步长。
* @param[in] xmax 积分终点。
* @param[in] tolerance 误差容限。
* @param[out] y 在积分终点 xmax 处的函数值，即解的近似值。
* @param[out] out 状态码，表示积分的结果状态。通常，0 表示成功，其他值表示错误或警告的特定情况。
* @param[out] h_next 下一个步长，即在下一次积分时应使用的步长。
* @return 无
*/
void Runge_Kutta_Fehlberg_7_8(void (*func)(double, const double[], double[]), 
    const double y0[], double x, double h, double xmax, double tolerance,
    double y[], int& flag, double& h_next,uint8_T N) {
    int last_interval = 0;

    // 验证步长h为正，并且积分上限大于初始点
    if (xmax < x || h <= 0.0) {
        for (int i = 0; i < N; ++i) {
            y[i] = y0[i];
        }
        flag = -2;
        h_next = h;
        return;
    }

    // 如果积分上限与初始点相同，直接返回初始点的值
    h_next = h;
    for (int i = 0; i < N; ++i) {
        y[i] = y0[i];
    }
    if (xmax == x) {
        flag = 0;
        h_next = h;
        return;
    }

    // 确保步长h不大于积分区间的长度
    if (h > (xmax - x)) {
        h = xmax - x;
        last_interval = 1;
    }

    // 将误差容限重新定义为单位长度的误差容限
    tolerance = tolerance / (xmax - x);

    double temp_y0[N];
    double temp_y[N];
    for (int i = 0; i < N; ++i) {
        temp_y0[i] = y0[i];
        temp_y[i] = y0[i];
    }
    while (x < xmax) {
        double scale = 1.0;
        int i;
        for (i = 0; i < ATTEMPTS; ++i) {
            double err[N];
            RKF87(func, temp_y, x, h, err,N);
            double err_norm = norm(err,N);
            if (err_norm == 0.0) {
                scale = MAX_SCALE_FACTOR;
                break;
            }
            double yy = (norm(temp_y0,N) == 0.0) ? tolerance : norm(temp_y0,N);
            scale = 0.8 * std::pow((tolerance * yy / err_norm), 1.0 / 7.0);
            scale = min(max(scale, MIN_SCALE_FACTOR), MAX_SCALE_FACTOR);
            if (err_norm < (tolerance * yy)) {
                break;
            }
            h = h * scale;
            if (x + h > xmax) {
                h = xmax - x;
            } else if (x + h + 0.5 * h > xmax) {
                h = 0.5 * h;
            }
        }
        if (i >= ATTEMPTS) {
            h_next = h * scale;
            flag = -1;
            return;
        }
        for (int i = 0; i < N; ++i) {
            temp_y0[i] = temp_y[i];
        }
        x = x + h;
        h = h * scale;
        h_next = h;
        if (last_interval) {
            break;
        }
        if (x + h > xmax) {
            last_interval = 1;
            h = xmax - x;
        }
        else if (x + h + 0.5 * h > xmax) {
            h = 0.5 * h;
        }
    }
    for (int i = 0; i < N; ++i) {
        y[i] = temp_y0[i];
    }
    flag = 0;
}
// 辅助函数：将两个数组按元素相加
void addVectors(const double v1[], const double v2[], double result[], int N) {
    for (int i = 0; i < N; ++i) {
        result[i] = v1[i] + v2[i];
    }
}
double h_max=600,h_min = 0.1;
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
void Runge_Kutta_45(void (*Deriv)(double, const double[], double[]),
                const double y0[], double t0, double h_initial, double tf, double tolerance, 
                double y_f[], int& flag, double& h_next,uint8_T N)
{   const double SAFETY = 0.9;
    const int MAX_ITER = 1000;

    double t = t0;
    double h = h_initial;
    h_next = h_initial;

    double y[N];
    for (uint8_T i = 0; i < N; ++i) {
        y[i] = y0[i];
    }

    double k1[N], k2[N], k3[N], k4[N], k5[N], k6[N];
    double errorEstimate[N], e_max,delta;

    int iter = 0;
    while (t < tf && iter < MAX_ITER) {
        // 计算 k1
        (*Deriv)(t, y, k1);

        // 计算 k2
        double y_temp[N];
        for (uint8_T i = 0; i < N; ++i) {
            y_temp[i] = y[i] + h* k1[i] / 4.0;
        }
        (*Deriv)(t + h / 4.0, y_temp, k2);

        // 计算 k3
        for (uint8_T i = 0; i < N; ++i) {
            y_temp[i] = y[i] + 3.0 * h* k1[i] / 32.0 + 9.0 * h* k2[i] / 32.0;
        }
        (*Deriv)(t + 3.0 * h / 8.0, y_temp, k3);

        // 计算 k4
        for (uint8_T i = 0; i < N; ++i) {
            y_temp[i] = y[i] + 1932.0 * h* k1[i] / 2197.0 - 7200.0 *h*  k2[i] / 2197.0 + 7296.0 * h* k3[i] / 2197.0;
        }
        (*Deriv)(t + 12.0 * h / 13.0, y_temp, k4);

        // 计算 k5
        for (uint8_T i = 0; i < N; ++i) {
            y_temp[i] = y[i] + 439.0 *h*  k1[i] / 216.0 - 8.0 *h*  k2[i] + 3680.0 * h* k3[i] / 513.0 - 845.0 *h*  k4[i] / 4104.0;
        }
        (*Deriv)(t + h, y_temp, k5);

        // 计算 k6
        for (uint8_T i = 0; i < N; ++i) {
            y_temp[i] = y[i] - 8.0 * h* k1[i] / 27.0 + 2.0 *h*  k2[i] - 3544.0 *h*  k3[i] / 2565.0 + 1859.0 *h*  k4[i] / 4104.0 - 11.0 * h* k5[i] / 40.0;
        }
        (*Deriv)(t + h / 2.0, y_temp, k6);

        // 计算误差估计
        for (uint8_T i = 0; i < N; ++i) {
            errorEstimate[i] = std::abs(h* k1[i] / 360.0 - 128.0 * h* k3[i] / 4275.0 - 2197.0 *h*  k4[i] / 75240.0 +h*  k5[i] / 50.0 + 2.0 * h* k6[i] / 55.0);
        }
        // if(N>1){e2 = std::abs(h* k1[1] / 360.0 - 1408.0 *h*  k3[1] / 2565.0 - 2197.0 *h*  k4[1] / 4104.0 - h* k5[1] / 5.0);
        //     if(N>2){e3 = std::abs(-h* k1[2] / 360.0 + 2197.0 * h* k4[2] /75240.0 +h*  k5[2] / 50.0 + 2.0 * h* k6[2] / 55.0);
        //         if(N>3)e4 = std::abs(-h* k1[3] / 360.0 + 256.0 * h* k3[3] / 1771.0 + 2197.0 *  h* k4[3] / 75240.0 -h*  k5[3] / 50.0 + 2.0 * h* k6[3] / 55.0);
        //     }
        // }
        // 计算误差估计的最大值
        e_max = 0.0;
        for (uint8_T i = 0; i < N; ++i){
            e_max =  max(errorEstimate[i], e_max);
        }
        
        
        // 计算下一步的步长
        delta = tolerance / (2.0 * (e_max + 1e-10));delta = SAFETY * std::pow(delta, 0.2);
        if (delta <= 0.1) {
            h_next = h* 0.1;
        } else if (delta >= 4.0) {
            h_next = h* 4.0;
        } else {
            if(delta>1.2||delta<0.8)h_next = h* delta;
            else h_next= h;
        }
        if(now_debugging){std::cout<<"t="<<t<<"\th="<<h<<"\tcal_error="<<e_max<<"\tscale step="<<delta<<"\t x[i] = ";for(int i=0;i<N;i++)std::cout<<y[i]<<' ';std::cout<<std::endl;}
        
        if (e_max <= tolerance) {
            // 当前步长可接受，更新 t 和 y，并计算下一步的步长
            t += h;
            for (uint8_T i = 0; i < N; ++i) {
                // y[i] += h* k1[i] / 360.0 + 128.0 * h* k3[i] / 4275.0 + 2197.0 * h* k4[i] / 75240.0 + h* k5[i] / 50.0 + 2.0 * h* k6[i] / 55.0;
                y[i]+= h * (((k1[i] * 25.0 / 216.0 + k3[i] * 1408.0 / 2565.0) + k4[i] * 2197.0 / 4104.0) - k5[i] / 5.0);
            }
        }
        h=min(max(h_min,h_next),h_max);
        // Adjust step size if necessary
        if (t + h > tf) {
            h = tf - t; // Adjust step size to exactly reach tf
        }
        iter++;
    }

    if (iter >= MAX_ITER) {
        // 达到最大迭代次数，可能发生错误
        flag = -1;
    } else {
        // 成功完成
        flag = 0;
    }

    // 将最终结果保存在 y_f 中
    for (uint8_T i = 0; i < N; ++i) {
        y_f[i] = y[i];
    }
}

void Runge_Kutta_4(std::function<void(double, const double[], double[])> Deriv,
                   const double y0[], double t0, double h, double tf, double y_f[], uint8_T N) {
    double t = t0;
    double y[N];
    for (uint8_T i = 0; i < N; ++i) {
        y[i] = y0[i];
    }

    double k1[N], k2[N], k3[N], k4[N];
    double y_temp[N];

    while (t < tf) {
        // 计算 k1
        Deriv(t, y, k1);

        // 计算 k2
        for (uint8_T i = 0; i < N; ++i) {
            y_temp[i] = y[i] + h * k1[i] / 2.0;
        }
        Deriv(t + h / 2.0, y_temp, k2);

        // 计算 k3
        for (uint8_T i = 0; i < N; ++i) {
            y_temp[i] = y[i] + h * k2[i] / 2.0;
        }
        Deriv(t + h / 2.0, y_temp, k3);

        // 计算 k4
        for (uint8_T i = 0; i < N; ++i) {
            y_temp[i] = y[i] + h * k3[i];
        }
        Deriv(t + h, y_temp, k4);

        // 更新 y 值
        for (uint8_T i = 0; i < N; ++i) {
            y[i] += h * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
        }

        t += h;

        // 确保最后一步不超过 tf
        if (t + h > tf) {
            h = tf - t;
        }
    }

    // 将最终结果保存在 y_f 中
    for (uint8_T i = 0; i < N; ++i) {
        y_f[i] = y[i];
    }
}

void Runge_Kutta_45(std::function<void(double, const double[], double[])> Deriv,
                const double y0[], double t0, double h_initial, double tf, double tolerance, 
                double y_f[], int& flag, double& h_next,uint8_T N)
{   const double SAFETY = 0.9;
    const int MAX_ITER = 1000;

    double t = t0;
    double h = h_initial;
    h_next = h_initial;

    double y[N];
    for (uint8_T i = 0; i < N; ++i) {
        y[i] = y0[i];
    }

    double k1[N], k2[N], k3[N], k4[N], k5[N], k6[N];
    double errorEstimate[N], e_max,delta;

    int iter = 0;
    while (t < tf && iter < MAX_ITER) {
        // 计算 k1
        Deriv(t, y, k1);

        // 计算 k2
        double y_temp[N];
        for (uint8_T i = 0; i < N; ++i) {
            y_temp[i] = y[i] + h* k1[i] / 4.0;
        }
        Deriv(t + h / 4.0, y_temp, k2);

        // 计算 k3
        for (uint8_T i = 0; i < N; ++i) {
            y_temp[i] = y[i] + 3.0 * h* k1[i] / 32.0 + 9.0 * h* k2[i] / 32.0;
        }
        Deriv(t + 3.0 * h / 8.0, y_temp, k3);

        // 计算 k4
        for (uint8_T i = 0; i < N; ++i) {
            y_temp[i] = y[i] + 1932.0 * h* k1[i] / 2197.0 - 7200.0 *h*  k2[i] / 2197.0 + 7296.0 * h* k3[i] / 2197.0;
        }
        Deriv(t + 12.0 * h / 13.0, y_temp, k4);

        // 计算 k5
        for (uint8_T i = 0; i < N; ++i) {
            y_temp[i] = y[i] + 439.0 *h*  k1[i] / 216.0 - 8.0 *h*  k2[i] + 3680.0 * h* k3[i] / 513.0 - 845.0 *h*  k4[i] / 4104.0;
        }
        Deriv(t + h, y_temp, k5);

        // 计算 k6
        for (uint8_T i = 0; i < N; ++i) {
            y_temp[i] = y[i] - 8.0 * h* k1[i] / 27.0 + 2.0 *h*  k2[i] - 3544.0 *h*  k3[i] / 2565.0 + 1859.0 *h*  k4[i] / 4104.0 - 11.0 * h* k5[i] / 40.0;
        }
        Deriv(t + h / 2.0, y_temp, k6);

        // 计算误差估计
        for (uint8_T i = 0; i < N; ++i) {
            errorEstimate[i] = std::abs(h* k1[i] / 360.0 - 128.0 * h* k3[i] / 4275.0 - 2197.0 *h*  k4[i] / 75240.0 +h*  k5[i] / 50.0 + 2.0 * h* k6[i] / 55.0);
        }
        // if(N>1){e2 = std::abs(h* k1[1] / 360.0 - 1408.0 *h*  k3[1] / 2565.0 - 2197.0 *h*  k4[1] / 4104.0 - h* k5[1] / 5.0);
        //     if(N>2){e3 = std::abs(-h* k1[2] / 360.0 + 2197.0 * h* k4[2] /75240.0 +h*  k5[2] / 50.0 + 2.0 * h* k6[2] / 55.0);
        //         if(N>3)e4 = std::abs(-h* k1[3] / 360.0 + 256.0 * h* k3[3] / 1771.0 + 2197.0 *  h* k4[3] / 75240.0 -h*  k5[3] / 50.0 + 2.0 * h* k6[3] / 55.0);
        //     }
        // }
        // 计算误差估计的最大值
        e_max = 0.0;
        for (uint8_T i = 0; i < N; ++i){
            e_max =  max(errorEstimate[i], e_max);
        }
        
        
        // 计算下一步的步长
        delta = tolerance / (2.0 * (e_max + 1e-10));delta = SAFETY * std::pow(delta, 0.2);
        if (delta <= 0.1) {
            h_next = h* 0.1;
        } else if (delta >= 4.0) {
            h_next = h* 4.0;
        } else {
            if(delta>1.2||delta<0.8)h_next = h* delta;
            else h_next= h;
        }
        if(now_debugging){std::cout<<"t="<<t<<"\th="<<h<<"\tcal_error="<<e_max<<"\tscale step="<<delta<<"\t x[i] = ";for(int i=0;i<N;i++)std::cout<<y[i]<<' ';std::cout<<std::endl;}
        
        if (e_max <= tolerance) {
            // 当前步长可接受，更新 t 和 y，并计算下一步的步长
            t += h;
            for (uint8_T i = 0; i < N; ++i) {
                // y[i] += h* k1[i] / 360.0 + 128.0 * h* k3[i] / 4275.0 + 2197.0 * h* k4[i] / 75240.0 + h* k5[i] / 50.0 + 2.0 * h* k6[i] / 55.0;
                y[i]+= h * (((k1[i] * 25.0 / 216.0 + k3[i] * 1408.0 / 2565.0) + k4[i] * 2197.0 / 4104.0) - k5[i] / 5.0);
            }
        }
        h=min(max(h_min,h_next),h_max);
        // Adjust step size if necessary
        if (t + h > tf) {
            h = tf - t; // Adjust step size to exactly reach tf
        }
        iter++;
    }

    if (iter >= MAX_ITER) {
        // 达到最大迭代次数，可能发生错误
        flag = -1;
    } else {
        // 成功完成
        flag = 0;
    }

    // 将最终结果保存在 y_f 中
    for (uint8_T i = 0; i < N; ++i) {
        y_f[i] = y[i];
    }
}

