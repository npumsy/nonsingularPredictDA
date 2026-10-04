#include "dastates.h"
#include <Eigen/Core>
// #include <Eigen/Dense>
#include <functional>
#include <vector>
#include <array>
using namespace std; 
using namespace DACE;

// --- 位置相关残差力场：统一接口（RBF / 低阶球谐），供 double / DA / 记录标量三条路径共用 ---
// 残差加速度 a_res = grad U（RBF: U=sum_k theta_k exp(-|r-c_k|^2/(2 s^2))；球谐见 sh_solid）。
namespace {
enum FieldKind { FIELD_RBF = 0, FIELD_SH = 1 };
struct FieldSpec {
    int kind = FIELD_RBF;
    int lmax = 2;                              // 球谐最大阶
    std::vector<std::array<double,3>> centers; // RBF 中心（km）
    double s = 1.0;                            // RBF 宽度（km）
};
FieldSpec g_field;
} // namespace
void setRBFParams(const std::vector<std::array<double,3>> &centers, double s){
    g_field.kind = FIELD_RBF;
    g_field.centers = centers;
    g_field.s = s;
}
void setSHParams(int lmax){
    g_field.kind = FIELD_SH;
    g_field.lmax = lmax;
}


template<typename T>
AlgebraicVector<T> TBPfullwarp(AlgebraicVector<T> x, double t, double arg1){
    return TBPfull(x,t ,T(arg1));
}
// Exercise 6.2.1: 3/8 rule RK4 integrator
template<typename T> T rk4( T x0, double t0, double t1, T (*f)(T,double,double) ,double arg1, double hmax)
{
	int steps = ceil( (t1-t0)/hmax );
	double h = (t1-t0)/steps;
    double t = t0;

    T k1, k2, k3, k4;
	for( int i = 0; i < steps; i++ )
	{
        k1 = f( x0, t ,arg1);
        k2 = f( x0 + h*k1/3.0, t + h/3.0 ,arg1 );
        k3 = f( x0 + h*(-k1/3.0 + k2), t + 2.0*h/3.0 ,arg1);
        k4 = f( x0 + h*(k1 - k2 + k3), t + h ,arg1);
        x0 = x0 + h*(k1 + 3*k2 + 3*k3 +k4)/8.0;
		t += h;
	}

    return x0;
}
template<typename T> T rk4b( T x0, double t0, double t1, T (*f)(T,double,double) ,double arg1, double hmax)
{
	int steps = ceil( (t1-t0)/hmax );
	double h = (t1-t0)/steps;
    double t = t0;

    T k1, k2, k3, k4;
	for( int i = 0; i < steps; i++ )
	{
        k1 = f( x0, t ,arg1);
        k2 = f( x0 - h*k1/3.0, t - h/3.0 ,arg1 );
        k3 = f( x0 - h*(-k1/3.0 + k2), t - 2.0*h/3.0 ,arg1);
        k4 = f( x0 - h*(k1 - k2 + k3), t - h ,arg1);
        x0 = x0 - h*(k1 + 3*k2 + 3*k3 +k4)/8.0;
		t += h;
	}

    return x0;
}
NominalErrorProp::NominalErrorProp(const Vector6d &rv0,int order){
    const int N = 6;
     DA::init( order, N );       // initialize DACE for 1st-order computations in 2 variables

    x0 = AlgebraicVector<DA>(6);
    xf = AlgebraicVector<DA>(6);
    xp = AlgebraicVector<DA>(6);
   
    for(int i=0;i<6;i++)x0[i]=rv0(i)/1e3 + DA(i+1);

    DA::pushTO( 1 );    // only first order derivative needed

}
void NominalErrorProp::updateX0(const Vector6d &rv0){
    for(int i=0;i<6;i++)x0[i]=rv0(i)/1e3 + DA(i+1);
}
Vector6d NominalErrorProp::propNomJ234Drag( Eigen::Ref<Eigen::Matrix<double, 6, 6>> Phi0f, double tf,
                                    bool givePhi, double step){
    xf = rk4( x0, 0, tf, TBPfullwarp ,1.0 ,step);
    
    if(givePhi)
        for( int i = 0; i < 6; i++ )
        {
            for( int j = 1; j <= 6; j++ )
            {
                Phi0f(i,j-1)=cons(xf[i].deriv(j));
            }
        }
    Vector6d rvf;
    for(int i=0;i<6;i++)rvf[i]=cons(xf[i])*1e3;
    return rvf;
}
Vector6d NominalErrorProp::bkpropNomJ234Drag( Eigen::Ref<Eigen::Matrix<double, 6, 6>> Phi0f, double tp,
                                    bool givePhi, double step){
    xp = rk4b( x0, 0, tp, TBPfullwarp ,1.0 ,step);
    
    if(givePhi)
        for( int i = 0; i < 6; i++ )
        {
            for( int j = 1; j <= 6; j++ )
            {
                Phi0f(i,j-1)=cons(xp[i].deriv(j));
            }
        }
    Vector6d rvf;
    for(int i=0;i<6;i++)rvf[i]=cons(xp[i])*1e3;
    return rvf;
}
// // 函数返回一个 6x6x6 的张量
// Eigen::Tensor<double, 6> returnTensor() {
//     // 创建一个 3x4x5 的张量
//     Eigen::Tensor<double, 6> tensor(6,6,6);

//     // 初始化张量的值
//     for (int i = 0; i < 3; ++i) {
//         for (int j = 0; j < 4; ++j) {
//             for (int k = 0; k < 5; ++k) {
//                 tensor(i, j, k) = i * 100 + j * 10 + k;
//             }
//         }
//     }

//     return tensor;
// }
Vector6d NominalErrorProp::evaldXf(const Vector6d &drv0,double scale_rhoCdA_m){
    Vector6d drvf;
    AlgebraicVector<double> Deltax0(6),Deltaxf(6);
    for(int i=0;i<6;i++){Deltax0[i]=drv0[i]/1e3;}
    Deltaxf=xf.eval(Deltax0)*1e3;
    for(int i=0;i<6;i++)drvf[i]=Deltaxf[i];
    return drvf;
}
Vector6d NominalErrorProp::evaldXp(const Vector6d &drv0,double scale_rhoCdA_m){
    Vector6d drvf;
    AlgebraicVector<double> Deltax0(6),Deltaxf(6);
    for(int i=0;i<6;i++){Deltax0[i]=drv0[i]/1e3;}
    Deltaxf=xp.eval(Deltax0)*1e3;
    for(int i=0;i<6;i++)drvf[i]=Deltaxf[i];
    return drvf;
}
NominalErrorProp::~NominalErrorProp(){
    DA::popTO( );
}


template<typename T> AlgebraicVector<T> TBP( AlgebraicVector<T> x, double t )
{
    
    AlgebraicVector<T> pos(3), res(6);
    
    pos[0] = x[0]; pos[1] = x[1]; pos[2] = x[2];
    
    T r = pos.vnorm();
    
    const double mu = 398600; // km^3/s^2
    
    res[0] = x[3];
    res[1] = x[4];
    res[2] = x[5];
    
    res[3] = -mu*pos[0]/(r*r*r);
    res[4] = -mu*pos[1]/(r*r*r);
    res[5] = -mu*pos[2]/(r*r*r);
    
    return res;
    
}

template<typename T>
AlgebraicVector<T> TBPfull(AlgebraicVector<T> x, double t,T beta, double mu, double Re, double rhoCdA_m,double h0,double H0) {
    AlgebraicVector<T> pos(3),vel(3), res(6);
    
    pos[0] = x[0]; pos[1] = x[1]; pos[2] = x[2];
    // vel[0]=x[3];vel[1]=x[4];vel[2]=x[5];
    
    T r = pos.vnorm();
    T z_r =pos[2] / r;
    T z2_r2 = z_r*z_r;
    T z3_r3 = z2_r2 * z_r;
    T z4_r4 = z3_r3 * z_r;
    T Re_r = Re/r;
    T Re2_r2 = Re_r * Re_r;
    T Re3_r3 = Re2_r2 * Re_r;
    T Re4_r4 = Re3_r3 * Re_r;

    T common_factor = -mu /pow(r,3);

    T J2_term = (3.0 / 2.0) * bddd::J2 * Re2_r2 * (1.0 - 5.0 * z2_r2);
    T J3_term = (5.0 / 2.0) * bddd::J3 * Re3_r3 * (3.0 * z_r - 7.0 * z3_r3);
    T J4_term = (5.0 / 8.0) * bddd::J4 * Re4_r4 * (3.0 - 42.0 * z2_r2 + 63.0 * z4_r4);

    res[3] = common_factor * pos[0] * (1.0 + J2_term + J3_term - J4_term);
    res[4] = common_factor * pos[1] * (1.0 + J2_term + J3_term - J4_term);


    J2_term = (3.0 / 2.0) * bddd::J2 * Re2_r2 * (3.0 - 5.0 * z2_r2);
    J3_term = (5.0 / 2.0) * bddd::J3 * Re3_r3 * (6.0 * z_r- 7.0 * z3_r3 - (3.0 / 5.0) * r / pos[2]);
    J4_term = (5.0 / 8.0) * bddd::J4 * Re4_r4 * (15.0 - 70.0 * z2_r2 + 63.0 * z4_r4);

    res[5] = common_factor * pos[2] * (1.0 + J2_term + J3_term - J4_term);
 if(printEachStepPertub_da)cout<<"DA---J234: (m/s^2)="<<cons(res[3]-common_factor * pos[0] )*1e3<<",\t"<<cons(res[4]-common_factor * pos[1] )*1e3<<",\t"<<cons(res[5]-common_factor * pos[2] )*1e3<<endl;
    // cout<<"DA---a_x, a_y, a_z(m/s^2)="<<cons(res[3]-common_factor * pos[0] )*1e3<<",\t"<<cons(res[4]-common_factor * pos[1] )*1e3<<",\t"<<cons(res[5]-common_factor * pos[2] )*1e3<<endl;

    // Calculate drag acceleration
    // T v = vel.vnorm(); // velocity magnitude
    AlgebraicVector<T> rel_vel = {x[3] +bddd::OMEGA_EARTH * x[1], x[4] -bddd::OMEGA_EARTH * x[0], x[5]}; // Earth rotation correction
    T v = rel_vel.vnorm();
    AlgebraicVector<T> drag_acc = {
        -0.5 * rhoCdA_m * beta*exp(-(r-Re - h0) / H0) * v * rel_vel[0],
        -0.5 * rhoCdA_m * beta*exp(-(r-Re - h0) / H0) * v * rel_vel[1],
        -0.5 * rhoCdA_m * beta*exp(-(r-Re - h0) / H0) * v * rel_vel[2]
    };
// cout<<"DA---v, v_x,v_y, v_z(m/s^2)="<<cons(v)*1e3<<",\t"<<cons(rel_vel[0])*1e3<<",\t"<<cons(rel_vel[1])*1e3<<",\t"<<cons(rel_vel[2])*1e3<<endl;

    // Combine accelerations
    res[0] = x[3];
    res[1] = x[4];
    res[2] = x[5];
    res[3] +=drag_acc[0];
    res[4] +=drag_acc[1];
    res[5] +=drag_acc[2];

 if(printTotalAccleration_da)cout<<"DA-- a_x, a_y, a_z(m/s^2)="<<1e3*cons(res[3])<<",\t"<<1e3*cons(res[4])<<",\t"<<1e3*cons(res[5])<<endl;
  if(printEachStepPertub_da)cout<<"DA---drag: (m/s^2)="<<cons(drag_acc[0])*1e3<<",\t"<<cons(drag_acc[1])*1e3<<",\t"<<cons(drag_acc[2])*1e3<<endl;

    return res;
}


// 增广状态 RHS：x = [r(3), v(3), kappa]，kappa 为第 7 个状态（导数恒为 0）。
template<typename T>
AlgebraicVector<T> TBPfull_param(AlgebraicVector<T> x, double t, double arg1){
    AlgebraicVector<T> x6(6);
    for(int i=0;i<6;i++) x6[i]=x[i];
    AlgebraicVector<T> r6 = TBPfull(x6, t, x[6]);
    AlgebraicVector<T> r(7);
    for(int i=0;i<6;i++) r[i]=r6[i];
    r[6] = T(0.0);
    return r;
}

namespace {
void enumerate_monomials_uv(unsigned int nv, unsigned int order,
                            std::vector<std::vector<unsigned int>> &out){
    out.clear();
    std::vector<unsigned int> e(nv, 0u);
    std::function<void(unsigned int, unsigned int)> rec =
        [&](unsigned int idx, unsigned int rem){
            if(idx == nv-1){ e[idx]=rem; out.push_back(e); return; }
            for(unsigned int k=0;k<=rem;k++){ e[idx]=k; rec(idx+1, rem-k); }
        };
    for(unsigned int tot=0; tot<=order; ++tot) rec(0, tot);
}
} // namespace

Vector6d daJ234DragAugCoeffs(const Vector6d &rv0, double kappa0, double tf,
                             int order, double step,
                             std::vector<double> &coeffs,
                             std::vector<std::vector<unsigned int>> &mons){
    const int N = 7;
    DA::init(order, N);
    DA::setEps(0.0);   // 不做 fabs(c)<=eps 的系数丢弃，否则阻力等小灵敏度会被截掉
    AlgebraicVector<DA> x(7);
    for(int i=0;i<6;i++) x[i] = rv0(i)/1e3 + DA(i+1);
    x[6] = kappa0 + DA(7);

    // 保留 order 阶（不做 pushTO(1)），以便用更高一阶系数得到中心敏感度。
    x = rk4(x, 0, tf, TBPfull_param, 0.0, step);

    enumerate_monomials_uv(7u, (unsigned int)order, mons);
    const std::size_t nmono = mons.size();
    coeffs.assign(6*nmono, 0.0);
    for(int i=0;i<6;i++){
        for(std::size_t k=0;k<nmono;k++){
            coeffs[(std::size_t)i*nmono + k] = x[i].getCoefficient(mons[k]);
        }
    }
    Vector6d rvf;
    for(int i=0;i<6;i++) rvf[i]=cons(x[i])*1e3;
    return rvf;
}

// 通用增广 RHS：x = [rv(6), theta(m)]，阻力项乘上 Π theta_k（各参数为独立乘性因子）。
template<typename T>
AlgebraicVector<T> TBPfull_aug(AlgebraicVector<T> x, double t, double arg1){
    const int m = (int)x.size() - 6;
    T scale = T(1.0);
    for(int k=0;k<m;k++) scale = scale * x[6+k];
    AlgebraicVector<T> x6(6);
    for(int i=0;i<6;i++) x6[i]=x[i];
    AlgebraicVector<T> r6 = TBPfull(x6, t, scale);
    AlgebraicVector<T> r(6+m);
    for(int i=0;i<6;i++) r[i]=r6[i];
    for(int k=0;k<m;k++) r[6+k]=T(0.0);
    return r;
}

Vector6d daAugCoeffs(const Vector6d &rv0, const std::vector<double> &params,
                     double tf, int order, double step,
                     std::vector<double> &coeffs,
                     std::vector<std::vector<unsigned int>> &mons){
    const int m = (int)params.size();
    const int N = 6 + m;
    DA::init(order, N);
    DA::setEps(0.0);
    AlgebraicVector<DA> x(N);
    for(int i=0;i<6;i++) x[i] = rv0(i)/1e3 + DA(i+1);
    for(int k=0;k<m;k++) x[6+k] = params[k] + DA(7+k);

    x = rk4(x, 0, tf, TBPfull_aug, 0.0, step);

    enumerate_monomials_uv((unsigned int)N, (unsigned int)order, mons);
    const std::size_t nmono = mons.size();
    coeffs.assign(6*nmono, 0.0);
    for(int i=0;i<6;i++)
        for(std::size_t k=0;k<nmono;k++)
            coeffs[(std::size_t)i*nmono + k] = x[i].getCoefficient(mons[k]);
    Vector6d rvf;
    for(int i=0;i<6;i++) rvf[i]=cons(x[i])*1e3;
    return rvf;
}

// 位置相关残差力场的增广 RHS（统一接口；力场种类由 setRBFParams/setSHParams 设定）。
template<typename T> AlgebraicVector<T> TBPfull_field(AlgebraicVector<T>, double, double);

template<typename T>
AlgebraicVector<T> TBPfull_rbf(AlgebraicVector<T> x, double t, double arg1){
    return TBPfull_field<T>(x, t, arg1);
}

Vector6d daAugRBFCoeffs(const Vector6d &rv0, const std::vector<double> &thetas,
                        const std::vector<std::array<double,3>> &centers, double s,
                        double tf, int order, double step,
                        std::vector<double> &coeffs,
                        std::vector<std::vector<unsigned int>> &mons){
    setRBFParams(centers, s);
    const int m = (int)thetas.size();
    const int N = 6 + m;
    DA::init(order, N);
    DA::setEps(0.0);
    AlgebraicVector<DA> x(N);
    for(int i=0;i<6;i++) x[i] = rv0(i)/1e3 + DA(i+1);
    for(int k=0;k<m;k++) x[6+k] = thetas[k] + DA(7+k);

    x = rk4(x, 0, tf, TBPfull_rbf, 0.0, step);

    enumerate_monomials_uv((unsigned int)N, (unsigned int)order, mons);
    const std::size_t nmono = mons.size();
    coeffs.assign(6*nmono, 0.0);
    for(int i=0;i<6;i++)
        for(std::size_t k=0;k<nmono;k++)
            coeffs[(std::size_t)i*nmono + k] = x[i].getCoefficient(mons[k]);
    Vector6d rvf;
    for(int i=0;i<6;i++) rvf[i]=cons(x[i])*1e3;
    return rvf;
}

// --- 低阶非带谐球谐异常场（笛卡尔实球谐，regular solid harmonics）---------
namespace {
// A_lm = r^l P_lm(sin phi) cos(m lambda), B_lm = ... sin ...（m>=1, l=2,3）
// 及其笛卡尔梯度，均为 (l 次) 多项式。
template<typename T>
void sh_solid(int l,int m,const T&x,const T&y,const T&z,
              T&A,T&B,T&gAx,T&gAy,T&gAz,T&gBx,T&gBy,T&gBz){
    if(l==2 && m==1){
        A = 3.0*x*z; B = 3.0*y*z;
        gAx = 3.0*z; gAy = T(0.0); gAz = 3.0*x;
        gBx = T(0.0); gBy = 3.0*z; gBz = 3.0*y;
    } else if(l==2 && m==2){
        A = 3.0*(x*x - y*y); B = 6.0*x*y;
        gAx = 6.0*x; gAy = -6.0*y; gAz = T(0.0);
        gBx = 6.0*y; gBy = 6.0*x; gBz = T(0.0);
    } else if(l==3 && m==1){
        T q = 5.0*z*z - (x*x + y*y + z*z);
        A = 1.5*x*q; B = 1.5*y*q;
        gAx = 1.5*(4.0*z*z - 3.0*x*x - y*y); gAy = -3.0*x*y; gAz = 12.0*x*z;
        gBx = -3.0*x*y; gBy = 1.5*(4.0*z*z - x*x - 3.0*y*y); gBz = 12.0*y*z;
    } else if(l==3 && m==2){
        T xy2 = x*x - y*y;
        A = 15.0*z*xy2; B = 30.0*x*y*z;
        gAx = 30.0*x*z; gAy = -30.0*y*z; gAz = 15.0*xy2;
        gBx = 30.0*y*z; gBy = 30.0*x*z; gBz = 30.0*x*y;
    } else if(l==3 && m==3){
        A = 15.0*(x*x*x - 3.0*x*y*y); B = 15.0*(3.0*x*x*y - y*y*y);
        gAx = 45.0*(x*x - y*y); gAy = -90.0*x*y; gAz = T(0.0);
        gBx = 90.0*x*y; gBy = 45.0*(x*x - y*y); gBz = T(0.0);
    } else {
        A=T(0.0); B=T(0.0); gAx=T(0.0); gAy=T(0.0); gAz=T(0.0);
        gBx=T(0.0); gBy=T(0.0); gBz=T(0.0);
    }
}
} // namespace

// 单个基（第 kk 个 θ 对应的加速度基 b_kk，已除去 θ_kk 因子）；x 单位 km。
// RBF:  b_k = (w/s^2) d；球谐: 第 kk 个系数（C 或 S）对应的势梯度项。
template<typename T>
AlgebraicVector<T> basisAccel(const AlgebraicVector<T>& x, int kk){
    const T X=x[0], Y=x[1], Z=x[2];
    AlgebraicVector<T> a(3); a[0]=T(0.0); a[1]=T(0.0); a[2]=T(0.0);
    if(g_field.kind == FIELD_SH){
        const double mu = bddd::MU/1e9;   // km^3/s^2
        const double Re = bddd::RE/1e3;   // km
        const T rn = sqrt(X*X+Y*Y+Z*Z);
        const T r2 = rn*rn;
        int k = 0;
        for(int l=2; l<=g_field.lmax; ++l){
            const T rp  = T(1.0)/pow(rn, 2*l+1);   // r^{-(2l+1)}
            const T rp3 = rp/r2;                   // r^{-(2l+3)}
            const double Re_l = std::pow(Re, l);
            const double c2l1 = (double)(2*l+1);
            for(int mm=1; mm<=l; ++mm){
                if(k==kk || k+1==kk){
                    T A,B,gAx,gAy,gAz,gBx,gBy,gBz;
                    sh_solid(l,mm,X,Y,Z,A,B,gAx,gAy,gAz,gBx,gBy,gBz);
                    const bool isC = (k==kk);
                    const T val = isC ? A  : B;
                    const T gx  = isC ? gAx : gBx;
                    const T gy  = isC ? gAy : gBy;
                    const T gz  = isC ? gAz : gBz;
                    const T cf  = mu*Re_l;
                    a[0] = cf*(rp*gx - c2l1*rp3*X*val);
                    a[1] = cf*(rp*gy - c2l1*rp3*Y*val);
                    a[2] = cf*(rp*gz - c2l1*rp3*Z*val);
                    return a;
                }
                k += 2;
            }
        }
    } else {   // RBF: b_k = (w/s^2)(r-c_k)
        const double s2 = g_field.s*g_field.s;
        const T dx = X - g_field.centers[kk][0];
        const T dy = Y - g_field.centers[kk][1];
        const T dz = Z - g_field.centers[kk][2];
        const T w  = exp(-(dx*dx+dy*dy+dz*dz)/(2.0*s2));
        const T c  = w/s2;
        a[0] = c*dx; a[1] = c*dy; a[2] = c*dz;   // -grad U（U=θ_k exp(...)）
    }
    return a;
}

// 统一残差加速度 a_res(r; θ)（3 维）= sum_k θ_k b_k(x)；θ 取自增广状态 x[6..]。
template<typename T>
AlgebraicVector<T> fieldAccel(const AlgebraicVector<T>& x, int m){
    AlgebraicVector<T> a(3); a[0]=T(0.0); a[1]=T(0.0); a[2]=T(0.0);
    for(int k=0;k<m;k++){
        AlgebraicVector<T> bk = basisAccel(x, k);
        a[0] += x[6+k]*bk[0]; a[1] += x[6+k]*bk[1]; a[2] += x[6+k]*bk[2];
    }
    return a;
}

// 位置相关残差力场的增广 RHS：x=[rv(6),θ(m)]，力场由 g_field 设定（RBF 或球谐）。
template<typename T>
AlgebraicVector<T> TBPfull_field(AlgebraicVector<T> x, double t, double arg1){
    const int m = (int)x.size() - 6;
    AlgebraicVector<T> x6(6);
    for(int i=0;i<6;i++) x6[i]=x[i];
    AlgebraicVector<T> r6 = TBPfull(x6, t, T(1.0));   // 二体 + J234 + 标称阻力
    AlgebraicVector<T> r(6+m);
    for(int i=0;i<6;i++) r[i]=r6[i];
    AlgebraicVector<T> a = fieldAccel(x, m);
    r[3] += a[0]; r[4] += a[1]; r[5] += a[2];
    for(int k=0;k<m;k++) r[6+k]=T(0.0);
    return r;
}

template<typename T>
AlgebraicVector<T> TBPfull_sh(AlgebraicVector<T> x, double t, double arg1){
    return TBPfull_field<T>(x, t, arg1);
}

Vector6d daAugSHCoeffs(const Vector6d &rv0, const std::vector<double> &thetas,
                       int lmax, double tf, int order, double step,
                       std::vector<double> &coeffs,
                       std::vector<std::vector<unsigned int>> &mons){
    setSHParams(lmax);
    const int m = (int)thetas.size();
    const int N = 6 + m;
    DA::init(order, N);
    DA::setEps(0.0);
    AlgebraicVector<DA> x(N);
    for(int i=0;i<6;i++) x[i] = rv0(i)/1e3 + DA(i+1);
    for(int k=0;k<m;k++) x[6+k] = thetas[k] + DA(7+k);

    x = rk4(x, 0, tf, TBPfull_sh, 0.0, step);

    enumerate_monomials_uv((unsigned int)N, (unsigned int)order, mons);
    const std::size_t nmono = mons.size();
    coeffs.assign(6*nmono, 0.0);
    for(int i=0;i<6;i++)
        for(std::size_t k=0;k<nmono;k++)
            coeffs[(std::size_t)i*nmono + k] = x[i].getCoefficient(mons[k]);
    Vector6d rvf;
    for(int i=0;i<6;i++) rvf[i]=cons(x[i])*1e3;
    return rvf;
}

Vector6d shResidualAccel(const Vector6d &rv_m, const std::vector<double> &thetas, int lmax){
    setSHParams(lmax);
    const int m = (int)thetas.size();
    AlgebraicVector<double> x(6+m);
    for(int i=0;i<6;i++) x[i]=rv_m[i]/1e3;   // km
    for(int k=0;k<m;k++) x[6+k]=thetas[k];
    AlgebraicVector<double> rs = TBPfull_sh(x, 0.0, 1.0);
    AlgebraicVector<double> x6(6);
    for(int i=0;i<6;i++) x6[i]=x[i];
    AlgebraicVector<double> rb = TBPfull(x6, 0.0, 1.0);
    Vector6d out;
    for(int i=0;i<3;i++) out[i]=(rs[3+i]-rb[3+i])*1e3;   // km/s^2 -> m/s^2
    for(int i=3;i<6;i++) out[i]=0.0;
    return out;
}

// --- 路径 A：记录型标量的流传播（力场由 g_field 设定，RBF/球谐共用）----------
static RecordFlow daRecordFlowField(const Vector6d &rv0, const std::vector<double> &thetas,
                                    double tf, double step){
    const int m = (int)thetas.size();
    auto tape = std::make_shared<DACE::RecordTape>();
    DACE::active_tape() = tape.get();

    AlgebraicVector<DACE::Scalar> x(6+m);
    RecordFlow rf;
    rf.leaf_x0.resize(6);
    rf.leaf_p.resize(m);
    for(int i=0;i<6;i++){ x[i] = DACE::make_leaf(rv0(i)/1e3); rf.leaf_x0[i]=x[i].node; }
    for(int k=0;k<m;k++){ x[6+k] = DACE::make_leaf(thetas[k]); rf.leaf_p[k]=x[6+k].node; }

    // 以 Scalar 实例化既有 RHS + 积分器，自动记录整条计算图。
    x = rk4(x, 0, tf, TBPfull_field, 0.0, step);

    rf.out.resize(6);
    for(int i=0;i<6;i++){ rf.xf[i]=x[i].v*1e3; rf.out[i]=x[i].node; }
    DACE::active_tape() = nullptr;
    rf.tape = tape;
    return rf;
}

RecordFlow daRecordFlowRBF(const Vector6d &rv0, const std::vector<double> &thetas,
                           const std::vector<std::array<double,3>> &centers, double s,
                           double tf, double step){
    setRBFParams(centers, s);
    return daRecordFlowField(rv0, thetas, tf, step);
}

RecordFlow daRecordFlowSH(const Vector6d &rv0, const std::vector<double> &thetas,
                          int lmax, double tf, double step){
    setSHParams(lmax);
    return daRecordFlowField(rv0, thetas, tf, step);
}

// 多历元一阶可微算子：一次积分，在 tfs[k] 处取状态与一阶 Jacobian（order=1，对 m 线性）。
void daFieldMultiEpoch(const Vector6d &rv0_m, const std::vector<double> &thetas,
                       const std::vector<double> &tfs, int order, double step,
                       std::vector<Vector6d> &rvf, std::vector<double> &Jflat){
    const int m = (int)thetas.size();
    const int N = 6 + m;
    DA::init(order, N);
    DA::setEps(0.0);
    AlgebraicVector<DA> x(N);
    for(int i=0;i<6;i++) x[i] = rv0_m(i)/1e3 + DA(i+1);
    for(int k=0;k<m;k++) x[6+k] = thetas[k] + DA(7+k);

    // 预建一阶指数向量，避免内层重复分配。
    std::vector<std::vector<unsigned int>> e1(N);
    for(int j=0;j<N;j++){ e1[j].assign(N, 0u); e1[j][j] = 1u; }

    const int K = (int)tfs.size();
    rvf.assign(K, Vector6d());
    Jflat.assign((std::size_t)K*6*(6+m), 0.0);
    double tcur = 0.0;
    for(int kk=0; kk<K; ++kk){
        x = rk4(x, tcur, tfs[kk], TBPfull_field, 0.0, step);   // 连续积分，状态续用
        tcur = tfs[kk];
        for(int i=0;i<6;i++) rvf[kk][i] = cons(x[i])*1e3;
        for(int i=0;i<6;i++){
            for(int j=0;j<6;j++)
                Jflat[((std::size_t)kk*6+i)*(6+m)+j] = x[i].getCoefficient(e1[j]);
            for(int k=0;k<m;k++)
                Jflat[((std::size_t)kk*6+i)*(6+m)+6+k] = x[i].getCoefficient(e1[6+k])*1e3;
        }
    }
}

// A = ∂f/∂x (6×6) at (rv_km, θ)：DA order=1、N=6 求一次 RHS 的 Jacobian（不含 θ 的 DA）。
static void fieldStateJacobian(const double rv_km[6], const std::vector<double> &th, double A[36]){
    const int m = (int)th.size();
    DA::init(1, 6);
    DA::setEps(0.0);
    AlgebraicVector<DA> xa(6+m);
    for(int i=0;i<6;i++) xa[i] = rv_km[i] + DA(i+1);
    for(int k=0;k<m;k++) xa[6+k] = DA(th[k]);   // θ 作为常数
    AlgebraicVector<DA> f = TBPfull_field(xa, 0.0, 1.0);
    std::vector<unsigned int> e(6, 0u);
    for(int i=0;i<6;i++)
        for(int j=0;j<6;j++){ e.assign(6,0u); e[j]=1u; A[i*6+j] = f[i].getCoefficient(e); }
}

// B = ∂f/∂θ (6×m) at rv_km：力场对每个 θ 的基（只加速度行非零）。θ 不进 DA。
static void fieldBasis(const double r[3], int m, std::vector<double> &B){
    B.assign(6*m, 0.0);
    const double X=r[0], Y=r[1], Z=r[2];
    if(g_field.kind == FIELD_SH){
        const double mu = bddd::MU/1e9, Re = bddd::RE/1e3;
        const double rn = std::sqrt(X*X+Y*Y+Z*Z), r2 = rn*rn;
        int k = 0;
        for(int l=2; l<=g_field.lmax; ++l){
            const double rp = 1.0/std::pow(rn, 2*l+1), rp3 = rp/r2;
            const double Re_l = std::pow(Re, l), c2l1 = (double)(2*l+1);
            for(int mm=1; mm<=l; ++mm){
                double A,Bv,gAx,gAy,gAz,gBx,gBy,gBz;
                sh_solid(l,mm,X,Y,Z,A,Bv,gAx,gAy,gAz,gBx,gBy,gBz);
                B[3*m+k]   = mu*Re_l*(rp*gAx - c2l1*rp3*X*A);
                B[4*m+k]   = mu*Re_l*(rp*gAy - c2l1*rp3*Y*A);
                B[5*m+k]   = mu*Re_l*(rp*gAz - c2l1*rp3*Z*A);
                B[3*m+k+1] = mu*Re_l*(rp*gBx - c2l1*rp3*X*Bv);
                B[4*m+k+1] = mu*Re_l*(rp*gBy - c2l1*rp3*Y*Bv);
                B[5*m+k+1] = mu*Re_l*(rp*gBz - c2l1*rp3*Z*Bv);
                k += 2;
            }
        }
    } else {
        const double s2 = g_field.s*g_field.s;
        for(int k=0;k<m;k++){
            const double dx=X-g_field.centers[k][0], dy=Y-g_field.centers[k][1], dz=Z-g_field.centers[k][2];
            const double w = std::exp(-(dx*dx+dy*dy+dz*dz)/(2.0*s2))/s2;
            B[3*m+k]=w*dx; B[4*m+k]=w*dy; B[5*m+k]=w*dz;
        }
    }
}

// 残差加速度 a_res(r;θ)（m/s^2，前 3 维）；力场由 setRBFParams/setSHParams 设定。
Vector6d fieldResidualAccel(const Vector6d &rv_m, const std::vector<double> &thetas){
    const int m = (int)thetas.size();
    AlgebraicVector<double> x(6+m);
    for(int i=0;i<6;i++) x[i]=rv_m[i]/1e3;   // km
    for(int k=0;k<m;k++) x[6+k]=thetas[k];
    AlgebraicVector<double> a = fieldAccel(x, m);
    Vector6d out; out.setZero();
    for(int i=0;i<3;i++) out[i]=a[i]*1e3;    // km/s^2 -> m/s^2
    return out;
}

// ∂b_k/∂x（6x6，每个 k）——即 A_{,θ_k} = ∂²f/∂x∂θ_k。落到 dBdx[k*36 + i*6 + j]。
// RBF 用解析式（O(m)，快）；球谐回退到逐基 order-1 DA（m 小）。
// RBF: b_k = α w d, α=1/s², w=exp(-|d|²α/2) → ∂b_i/∂r_j = α w (δ_ij - α d_i d_j)。
void fieldBasisJacobian(const double r_km[6], int m, std::vector<double>& dBdx){
    dBdx.assign((std::size_t)m*36, 0.0);
    if(g_field.kind == FIELD_RBF){
        const double X=r_km[0], Y=r_km[1], Z=r_km[2];
        const double a = 1.0/(g_field.s*g_field.s);
        for(int k=0;k<m;k++){
            const double dx=X-g_field.centers[k][0], dy=Y-g_field.centers[k][1], dz=Z-g_field.centers[k][2];
            const double w = std::exp(-(dx*dx+dy*dy+dz*dz)*0.5*a);
            const double aw = a*w;
            const double d[3]={dx,dy,dz};
            for(int i=0;i<3;i++) for(int j=0;j<3;j++)
                dBdx[(std::size_t)k*36 + (3+i)*6 + j] = aw*((i==j?1.0:0.0) - a*d[i]*d[j]);
        }
        return;
    }
    DA::init(1, 6);
    DA::setEps(0.0);
    AlgebraicVector<DA> xa(6 + m);
    for(int i=0;i<6;i++) xa[i] = r_km[i] + DA(i+1);
    for(int k=0;k<m;k++) xa[6+k] = DA(0.0);
    std::vector<unsigned int> e(6, 0u);
    for(int k=0;k<m;k++){
        AlgebraicVector<DA> bk = basisAccel(xa, k);
        for(int i=0;i<3;i++){
            const int row = 3 + i;   // 只有加速度行（3..5）非零
            for(int j=0;j<6;j++){
                e.assign(6,0u); e[j]=1u;
                dBdx[(std::size_t)k*36 + row*6 + j] = bk[i].getCoefficient(e);
            }
        }
    }
}

// 变分灵敏度多历元：积分 [x(6); Φ(36); S(6m)]，A 用 DA(N=6)，B 解析；θ 不进 DA。
void daVarMultiEpoch(const Vector6d &rv0_m, const std::vector<double> &thetas,
                     const std::vector<double> &tfs, double step,
                     std::vector<Vector6d> &rvf, std::vector<double> &Jflat){
    const int m = (int)thetas.size();
    const int nA = 6 + 36 + 6*m;
    std::vector<double> y(nA, 0.0), k1(nA), k2(nA), k3(nA), k4(nA), yt(nA);
    for(int i=0;i<6;i++) y[i] = rv0_m[i]/1e3;                 // x (km)
    for(int i=0;i<6;i++) y[6 + i*6 + i] = 1.0;                // Φ = I
    // S = 0
    std::vector<double> B(6*m);
    auto dydt = [&](const std::vector<double> &yy, std::vector<double> &out){
        const double *x = &yy[0], *Phi = &yy[6], *S = &yy[42];
        AlgebraicVector<double> xa(6+m);
        for(int i=0;i<6;i++) xa[i]=x[i];
        for(int k=0;k<m;k++) xa[6+k]=thetas[k];
        AlgebraicVector<double> f = TBPfull_field(xa, 0.0, 1.0);
        for(int i=0;i<6;i++) out[i]=f[i];
        double A[36]; fieldStateJacobian(x, thetas, A);
        for(int i=0;i<6;i++) for(int j=0;j<6;j++){
            double s=0; for(int q=0;q<6;q++) s += A[i*6+q]*Phi[q*6+j]; out[6+i*6+j]=s; }
        fieldBasis(x, m, B);
        for(int i=0;i<6;i++) for(int k=0;k<m;k++){
            double s=B[i*m+k]; for(int q=0;q<6;q++) s += A[i*6+q]*S[q*m+k]; out[42+i*m+k]=s; }
    };
    const int K = (int)tfs.size();
    rvf.assign(K, Vector6d());
    Jflat.assign((std::size_t)K*6*(6+m), 0.0);
    double tcur = 0.0;
    for(int kk=0; kk<K; ++kk){
        const double H = tfs[kk]-tcur;
        const int ns = std::max(1, (int)std::ceil(H/step));
        const double h = H/ns;
        for(int s=0;s<ns;s++){
            dydt(y,k1);
            for(int q=0;q<nA;q++) yt[q]=y[q]+h*k1[q]/3.0;                 dydt(yt,k2);
            for(int q=0;q<nA;q++) yt[q]=y[q]+h*(-k1[q]/3.0+k2[q]);        dydt(yt,k3);
            for(int q=0;q<nA;q++) yt[q]=y[q]+h*(k1[q]-k2[q]+k3[q]);       dydt(yt,k4);
            for(int q=0;q<nA;q++) y[q]+=h*(k1[q]+3*k2[q]+3*k3[q]+k4[q])/8.0;
        }
        tcur = tfs[kk];
        for(int i=0;i<6;i++) rvf[kk][i]=y[i]*1e3;
        for(int i=0;i<6;i++) for(int j=0;j<6;j++)
            Jflat[((std::size_t)kk*6+i)*(6+m)+j] = y[6+i*6+j];
        for(int i=0;i<6;i++) for(int k=0;k<m;k++)
            Jflat[((std::size_t)kk*6+i)*(6+m)+6+k] = y[42+i*m+k]*1e3;
    }
}

void daRecordFlowBackward(const RecordFlow &rf, const std::vector<double> &grad,
                          Vector6d &gx, std::vector<double> &gp){
    const int m = (int)rf.leaf_p.size();
    std::vector<double> seeds(6);
    for(int i=0;i<6;i++) seeds[i] = grad[i]*1e3;   // 输出以 m 给出，节点值为 km
    const std::vector<double> g = rf.tape->backward(rf.out, seeds);
    for(int i=0;i<6;i++) gx[i] = g[rf.leaf_x0[i]]/1e3;   // 叶为 km，返回对 m 的梯度
    gp.resize(m);
    for(int k=0;k<m;k++) gp[k] = g[rf.leaf_p[k]];
}

// =================== 积分伴随：大 m deep 反向（离散 RK4 转置） ===================
// 扩展系统 z=[x(6); Φ(36)]，θ 不进 DA（N=6）。前向/反向都用与 rk4 相同的 3/8 RK4，
// 故反向严格等于离散前向算子的转置（与 FD 一致）。μ 布局 [μx(6); μΦ(36)]。

// θ-无关的 TBPfull（N=6）的 f、A=∂f/∂x、H[i][j][l]=∂²f_i/∂x_j∂x_l（DA，N=6）。
static void tbpFHA(const double x[6], bool needH, double f[6], double A[36], double H[216]){
    DA::init(needH?2u:1u, 6); DA::setEps(0.0);
    AlgebraicVector<DA> xa(6);
    for(int i=0;i<6;i++) xa[i]=x[i]+DA(i+1);
    AlgebraicVector<DA> fd = TBPfull(xa, 0.0, DA(1.0));
    std::vector<unsigned int> e(6,0u);
    for(int i=0;i<6;i++){
        if(f) f[i]=cons(fd[i]);
        for(int j=0;j<6;j++){ e.assign(6,0u); e[j]=1u; A[i*6+j]=fd[i].getCoefficient(e); }
        if(needH) for(int j=0;j<6;j++) for(int l=0;l<6;l++){
            e.assign(6,0u); e[j]+=1u; e[l]+=1u;
            H[(i*6+j)*6+l]=(j==l?2.0:1.0)*fd[i].getCoefficient(e);
        }
    }
}

// RBF 力场的解析 a=Σθ_k b_k、Aa=Σθ_k ∂b_k/∂x、Ha=Σθ_k ∂²b_k/∂x²（O(m)，无 DA）。
static void rbfFieldDerivs(const double x[6], const std::vector<double>& th, int m,
                           double a[3], double Aa[18], double Ha[108], bool needH){
    a[0]=a[1]=a[2]=0.0;
    if(Aa) for(int q=0;q<18;q++) Aa[q]=0.0;
    if(needH) for(int q=0;q<108;q++) Ha[q]=0.0;
    const double X=x[0],Y=x[1],Z=x[2];
    const double al=1.0/(g_field.s*g_field.s);
    for(int k=0;k<m;k++){
        const double tk=th[k];
        const double dx=X-g_field.centers[k][0], dy=Y-g_field.centers[k][1], dz=Z-g_field.centers[k][2];
        const double w=std::exp(-(dx*dx+dy*dy+dz*dz)*0.5*al);
        const double aw=al*w;
        const double d[3]={dx,dy,dz};
        for(int i=0;i<3;i++) a[i]+= tk*aw*d[i];
        for(int i=0;i<3;i++) for(int j=0;j<3;j++)
            Aa[i*6+j]+= tk*aw*((i==j?1.0:0.0)-al*d[i]*d[j]);
        if(needH) for(int i=0;i<3;i++) for(int j=0;j<3;j++) for(int l=0;l<3;l++){
            double term = -al*d[l]*((i==j?1.0:0.0)-al*d[i]*d[j])
                          -al*((i==l?1.0:0.0)*d[j]+d[i]*(j==l?1.0:0.0));
            Ha[(i*6+j)*6+l]+= tk*aw*term;
        }
    }
}

static void deepRhs(const double x[6], const std::vector<double>& th, int m,
                    double f[6], double A[36]){
    if(g_field.kind == FIELD_RBF){
        tbpFHA(x, false, f, A, nullptr);
        double a[3], Aa[18];
        rbfFieldDerivs(x, th, m, a, Aa, nullptr, false);
        f[3]+=a[0]; f[4]+=a[1]; f[5]+=a[2];
        for(int i=0;i<3;i++) for(int j=0;j<3;j++) A[(3+i)*6+j]+=Aa[i*6+j];
        return;
    }
    DA::init(1, 6); DA::setEps(0.0);
    AlgebraicVector<DA> xa(6+m);
    for(int i=0;i<6;i++) xa[i]=x[i]+DA(i+1);
    for(int k=0;k<m;k++) xa[6+k]=DA(th[k]);
    AlgebraicVector<DA> fd = TBPfull_field(xa, 0.0, 1.0);
    std::vector<unsigned int> e(6,0u);
    for(int i=0;i<6;i++){
        f[i]=cons(fd[i]);
        for(int j=0;j<6;j++){ e.assign(6,0u); e[j]=1u; A[i*6+j]=fd[i].getCoefficient(e); }
    }
}

// 一次 order-2 DA 同时给出 A=∂f/∂x（一阶）与 H[i][j][l]=∂²f_i/∂x_j∂x_l（二阶）。
static void deepHessA(const double x[6], const std::vector<double>& th, int m,
                      double H[216], double A[36]){
    if(g_field.kind == FIELD_RBF){
        double ft[6]; tbpFHA(x, true, ft, A, H);
        double a[3], Aa[18], Ha[108];
        rbfFieldDerivs(x, th, m, a, Aa, Ha, true);
        for(int i=0;i<3;i++) for(int j=0;j<3;j++)
            A[(3+i)*6+j]+=Aa[i*6+j];
        for(int i=0;i<3;i++) for(int j=0;j<3;j++) for(int l=0;l<3;l++)
            H[((3+i)*6+j)*6+l]+=Ha[(i*6+j)*6+l];
        return;
    }
    DA::init(2, 6); DA::setEps(0.0);
    AlgebraicVector<DA> xa(6+m);
    for(int i=0;i<6;i++) xa[i]=x[i]+DA(i+1);
    for(int k=0;k<m;k++) xa[6+k]=DA(th[k]);
    AlgebraicVector<DA> fd = TBPfull_field(xa, 0.0, 1.0);
    std::vector<unsigned int> e(6,0u);
    for(int i=0;i<6;i++){
        for(int j=0;j<6;j++){ e.assign(6,0u); e[j]=1u; A[i*6+j]=fd[i].getCoefficient(e); }
        for(int j=0;j<6;j++) for(int l=0;l<6;l++){
            e.assign(6,0u); e[j]+=1u; e[l]+=1u;
            H[(i*6+j)*6+l] = (j==l? 2.0 : 1.0)*fd[i].getCoefficient(e);
        }
    }
}

static void mat36(const double A[36], const double P[36], double out[36]){
    for(int i=0;i<6;i++) for(int q=0;q<6;q++){
        double s=0; for(int p=0;p<6;p++) s+=A[i*6+p]*P[p*6+q]; out[i*6+q]=s;
    }
}

// muIn 已是该阶段 k_i 的总伴随（权重已含在 mk 中），故此处不再乘 c_i。
// 一次遍历 RBF 力场：加权 a,Aa,Ha 与逐基 B(6m)=b_k、BJ(36m)=∂b_k/∂x（无 DA）。
static void rbfAllFieldDerivs(const double x[6], const std::vector<double>& th, int m,
                              double a[3], double Aa[18], double Ha[108],
                              std::vector<double>& B, std::vector<double>& BJ, bool needH){
    a[0]=a[1]=a[2]=0.0;
    for(int q=0;q<18;q++) Aa[q]=0.0;
    if(needH) for(int q=0;q<108;q++) Ha[q]=0.0;
    B.assign((std::size_t)6*m, 0.0);
    BJ.assign((std::size_t)36*m, 0.0);
    const double X=x[0],Y=x[1],Z=x[2], al=1.0/(g_field.s*g_field.s);
    for(int k=0;k<m;k++){
        const double tk=th[k];
        const double dx=X-g_field.centers[k][0], dy=Y-g_field.centers[k][1], dz=Z-g_field.centers[k][2];
        const double w=std::exp(-(dx*dx+dy*dy+dz*dz)*0.5*al), aw=al*w;
        const double d[3]={dx,dy,dz};
        const double bx=aw*dx, by=aw*dy, bz=aw*dz;
        B[(std::size_t)3*m+k]=bx; B[(std::size_t)4*m+k]=by; B[(std::size_t)5*m+k]=bz;
        a[0]+=tk*bx; a[1]+=tk*by; a[2]+=tk*bz;
        for(int i=0;i<3;i++) for(int j=0;j<3;j++){
            const double bj=aw*((i==j?1.0:0.0)-al*d[i]*d[j]);
            BJ[(std::size_t)k*36 + (3+i)*6+j]=bj;
            Aa[i*6+j]+= tk*bj;
        }
        if(needH) for(int i=0;i<3;i++) for(int j=0;j<3;j++) for(int l=0;l<3;l++){
            const double term=-al*d[l]*((i==j?1.0:0.0)-al*d[i]*d[j])
                              -al*((i==l?1.0:0.0)*d[j]+d[i]*(j==l?1.0:0.0));
            Ha[(i*6+j)*6+l]+= tk*aw*term;
        }
    }
}

static void deepJfT(const double x[6], const double Phi[36], const double muIn[42],
                    const std::vector<double>& th, int m,
                    std::vector<double>& gtheta, double muOut[42]){
    double A[36], H[216];
    static thread_local std::vector<double> Bv, Aq;   // 复用缓冲，避免每阶段分配
    if(g_field.kind == FIELD_RBF){
        double ft[6]; tbpFHA(x, true, ft, A, H);
        double a[3], Aa[18], Ha[108];
        rbfAllFieldDerivs(x, th, m, a, Aa, Ha, Bv, Aq, true);   // 一次遍历出 B、BJ，兼 a/Aa/Ha
        for(int i=0;i<3;i++) for(int j=0;j<3;j++) A[(3+i)*6+j]+=Aa[i*6+j];
        for(int i=0;i<3;i++) for(int j=0;j<3;j++) for(int l=0;l<3;l++)
            H[((3+i)*6+j)*6+l]+=Ha[(i*6+j)*6+l];
    } else {
        deepHessA(x, th, m, H, A);
        fieldBasis(x, m, Bv);
        fieldBasisJacobian(x, m, Aq);
    }
    for(int i=0;i<6;i++) for(int q=0;q<6;q++){
        double s=0; for(int p=0;p<6;p++) s+=A[p*6+i]*muIn[6+p*6+q];   // Aᵀ μΦ
        muOut[6+i*6+q]=s;
    }
    double W[36];                                                    // W[i][j]=Σq μΦ[i][q] Φ[j][q]
    for(int i=0;i<6;i++) for(int j=0;j<6;j++){
        double s=0; for(int q=0;q<6;q++) s+=muIn[6+i*6+q]*Phi[j*6+q]; W[i*6+j]=s;
    }
    for(int k=0;k<m;k++){
        double direct=0; for(int i=0;i<6;i++) direct+=Bv[i*m+k]*muIn[i];
        double deep=0;   for(int i=0;i<6;i++) for(int j=0;j<6;j++) deep+=W[i*6+j]*Aq[(std::size_t)k*36+i*6+j];
        gtheta[k]+= direct+deep;
    }
    for(int l=0;l<6;l++){
        double s=0; for(int i=0;i<6;i++) for(int j=0;j<6;j++) s+=W[i*6+j]*H[(i*6+j)*6+l];
        double ax=0; for(int p=0;p<6;p++) ax+=A[p*6+l]*muIn[p];
        muOut[l]=ax+s;                                               // Aᵀ μx + Cᵀ μΦ
    }
}

static void deepStepTranspose(const DeepFlow& fl, int si, double mu[42], std::vector<double>& gtheta){
    const double h=fl.hs[si];
    const double* xs=&fl.xs[(std::size_t)si*24];
    const double* Ps=&fl.Phis[(std::size_t)si*144];
    double mk[4][42];
    for(int q=0;q<42;q++){ mk[0][q]=(h/8.0)*mu[q]; mk[1][q]=3*(h/8.0)*mu[q];
                           mk[2][q]=3*(h/8.0)*mu[q]; mk[3][q]=(h/8.0)*mu[q]; }
    double my[42]; for(int q=0;q<42;q++) my[q]=mu[q];   // 显式恒等项 ∂y'/∂y=I
    double mx[42];
    deepJfT(xs+18, Ps+108, mk[3], fl.thetas, fl.m, gtheta, mx);   // k4 = F(p4)
    for(int q=0;q<42;q++){ my[q]+=mx[q]; mk[0][q]+=h*mx[q]; mk[1][q]+=-h*mx[q]; mk[2][q]+=h*mx[q]; }
    deepJfT(xs+12, Ps+72,  mk[2], fl.thetas, fl.m, gtheta, mx);   // k3 = F(p3), p3=y+h(k2-k1/3)
    for(int q=0;q<42;q++){ my[q]+=mx[q]; mk[1][q]+=h*mx[q]; mk[0][q]+=-(h/3.0)*mx[q]; }
    deepJfT(xs+6,  Ps+36,  mk[1], fl.thetas, fl.m, gtheta, mx);   // k2 = F(p2)
    for(int q=0;q<42;q++){ my[q]+=mx[q]; mk[0][q]+=(h/3.0)*mx[q]; }
    deepJfT(xs+0,  Ps+0,   mk[0], fl.thetas, fl.m, gtheta, mx);   // k1 = F(p1)
    for(int q=0;q<42;q++){ my[q]+=mx[q]; mu[q]=my[q]; }
}

void daDeepForward(const Vector6d &rv0_km, const std::vector<double> &thetas,
                   const std::vector<double> &tfs, double step, DeepFlow &fl){
    const int m=(int)thetas.size();
    fl.m=m; fl.thetas=thetas; fl.tfs=tfs;
    fl.xs.clear(); fl.Phis.clear(); fl.hs.clear(); fl.tend.clear();
    const int K=(int)tfs.size();
    fl.rvf.assign(K,Vector6d()); fl.PhiEpoch.assign((std::size_t)K*36,0.0);
    fl.epoch_step.assign(K,-1);
    double x[6]; for(int i=0;i<6;i++) x[i]=rv0_km[i];
    double Phi[36]; for(int q=0;q<36;q++) Phi[q]=0.0; for(int i=0;i<6;i++) Phi[i*6+i]=1.0;
    double tcur=0.0;
    for(int kk=0;kk<K;kk++){
        double H=tfs[kk]-tcur; int ns=std::max(1,(int)std::ceil(H/step)); double h=H/ns;
        for(int s=0;s<ns;s++){
            double f1[6],A1[36]; deepRhs(x,thetas,m,f1,A1);
            double x2[6]; for(int i=0;i<6;i++) x2[i]=x[i]+(h/3.0)*f1[i];
            double f2[6],A2[36]; deepRhs(x2,thetas,m,f2,A2);
            double x3[6]; for(int i=0;i<6;i++) x3[i]=x[i]+h*(-f1[i]/3.0+f2[i]);
            double f3[6],A3[36]; deepRhs(x3,thetas,m,f3,A3);
            double x4[6]; for(int i=0;i<6;i++) x4[i]=x[i]+h*(f1[i]-f2[i]+f3[i]);
            double f4[6],A4[36]; deepRhs(x4,thetas,m,f4,A4);
            double kP1[36]; mat36(A1,Phi,kP1);
            double P2[36]; for(int q=0;q<36;q++) P2[q]=Phi[q]+(h/3.0)*kP1[q];
            double kP2[36]; mat36(A2,P2,kP2);
            double P3[36]; for(int q=0;q<36;q++) P3[q]=Phi[q]+h*(-kP1[q]/3.0+kP2[q]);
            double kP3[36]; mat36(A3,P3,kP3);
            double P4[36]; for(int q=0;q<36;q++) P4[q]=Phi[q]+h*(kP1[q]-kP2[q]+kP3[q]);
            double kP4[36]; mat36(A4,P4,kP4);
            const double* XS[4]={x,x2,x3,x4}; const double* PS[4]={Phi,P2,P3,P4};
            for(int st=0;st<4;st++){
                for(int i=0;i<6;i++) fl.xs.push_back(XS[st][i]);
                for(int q=0;q<36;q++) fl.Phis.push_back(PS[st][q]);
            }
            fl.hs.push_back(h); fl.tend.push_back(tcur+h);
            for(int i=0;i<6;i++) x[i]+=(h/8.0)*(f1[i]+3*f2[i]+3*f3[i]+f4[i]);
            for(int q=0;q<36;q++) Phi[q]+=(h/8.0)*(kP1[q]+3*kP2[q]+3*kP3[q]+kP4[q]);
            tcur+=h;
        }
        for(int i=0;i<6;i++) fl.rvf[kk][i]=x[i];
        for(int q=0;q<36;q++) fl.PhiEpoch[(std::size_t)kk*36+q]=Phi[q];
        fl.epoch_step[kk]=(int)fl.hs.size()-1;
    }
}

void daDeepBackward(const DeepFlow &fl, const std::vector<Vector6d> &gx_epoch,
                    const std::vector<double> &gPhi_epoch,
                    Vector6d &gx0, std::vector<double> &gtheta){
    const int K=(int)fl.tfs.size();
    gtheta.assign(fl.m,0.0);
    double mu[42]; for(int q=0;q<42;q++) mu[q]=0.0;
    int cur=(int)fl.hs.size()-1;
    for(int i=0;i<6;i++) mu[i]+=gx_epoch[K-1][i];
    for(int q=0;q<36;q++) mu[6+q]+=gPhi_epoch[(std::size_t)(K-1)*36+q];
    for(int kk=K-1; kk>=0; --kk){
        int stop=(kk>0)? fl.epoch_step[kk-1] : -1;
        while(cur>stop){ deepStepTranspose(fl,cur,mu,gtheta); cur--; }
        if(kk>0){
            for(int i=0;i<6;i++) mu[i]+=gx_epoch[kk-1][i];
            for(int q=0;q<36;q++) mu[6+q]+=gPhi_epoch[(std::size_t)(kk-1)*36+q];
        }
    }
    for(int i=0;i<6;i++) gx0[i]=mu[i];
}

Vector6d EigenwarpDAOrbitJ234DragODE(const Vector6d &rv0, double t, double arg1,bool J234){
    AlgebraicVector<double> x(6),dx(6);
    for(int i=0;i<6;i++)x[i]=rv0[i]/1e3;
    dx =  TBPfull(x,t ,arg1);
    Vector6d rdx(6);
    for(int i=0;i<6;i++)rdx[i]=dx[i]*1e3;
    return rdx;
}

void ex6_2_3(double T=10 )
{
    AlgebraicVector<double> x0(6);
    AlgebraicVector<DA>  xf(6);
    x0[0] = 6716.3932; 
    x0[1] = -1389.0295; 
    x0[2] = -992.5427; 
    x0[3] = 1.48411; 
    x0[4] = 2.06038; 
    x0[5] = 7.15010;

    double scale=0.01;
    AlgebraicVector<DA> x = x0 + scale*AlgebraicVector<DA>::identity( );

    DA::pushTO( 1 );    // only first order computation needed

    x = rk4( x, 0, T, TBPfullwarp ,0,1.0);

    cout << "Exercise 6.2.3: CR3BP STM" << endl;
    cout.precision( 6 );
    cout << cons(x);

    Eigen::MatrixXd Phi0f(6,6);

    for( int i = 0; i < 6; i++ )
    {
        for( int j = 1; j <= 6; j++ )
        {
            cout << cons(x[i].deriv(j))/scale << "  ";
            Phi0f(i,j-1)=cons(x[i].deriv(j))/scale;
        }
        cout << endl;
    }
    cout << endl;
    cout<<"Eigen\n";
    cout<<Phi0f<<endl;
    DA::popTO( );
}
// 函数内部计算单位为km
Vector6d daJ234DragRV_RK4Step(const Vector6d &rv0, Eigen::Ref<Eigen::Matrix<double, 6, 6>> Phi0f, double tf,
                                    double scale_rhoCdA_m,bool givePhi, double step,int order,double scale){
    const int N = 6;
    DA::init( order, N );       // initialize DACE for 1st-order computations in 2 variables
    AlgebraicVector<double> x0(6);
    for(int i=0;i<6;i++)x0[i]=rv0(i)/1e3;

    AlgebraicVector<DA>  xf(6);

    AlgebraicVector<DA> x = x0 + scale*AlgebraicVector<DA>::identity( );

    DA::pushTO( 1 );    // only first order derivative needed
    x = rk4( x, 0, tf, TBPfullwarp ,scale_rhoCdA_m,step);

    Vector6d rvf;
    for(int i=0;i<6;i++)rvf[i]=cons(x[i])*1e3;
if(givePhi)
    for( int i = 0; i < 6; i++ )
    {
        for( int j = 1; j <= 6; j++ )
        {
            Phi0f(i,j-1)=cons(x[i].deriv(j))/scale;
        }
    }
    DA::popTO( );
    return rvf;
}
int main_(int argc, char *argv[])
{
    int order = 3;
    double tf=10;
    if (argc > 1) {
        try {
            order= std::stoi(argv[1]);
            if(argc>2){
                tf= std::stoi(argv[2]);
            }
        } catch (const std::invalid_argument& ia) {
            std::cerr << "Invalid argument, please provide an integer." << std::endl;
        }
    } 
    DA::init( order, 6 );       // initialize DACE for 1st-order computations in 2 variables

    ex6_2_3( tf);
    return 0;
}

