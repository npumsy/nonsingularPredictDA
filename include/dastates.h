#ifndef __DA_ODE_H__
#define __DA_ODE_H__

#include <dace/dace.h>
#include <cmath>
#include <fstream>
#include <array>
#include <memory>
#include "RecordingScalar.h"
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

// 整星座批处理（线程安全，无 DACE）：每星 2 次 double 传播（κ、κ+dk），FD 出 ∂x_f/∂κ。
// kappas 长度 1（广播）或 N（每星一个）。返回终端状态 xf；sens 写入 ∂x_f/∂κ。
// nthreads<=0 用默认线程数。可 OpenMP 并行。
std::vector<Vector6d> daJ234DragBatchD(const std::vector<Vector6d> &rv0s,
                                       const std::vector<double> &kappas,
                                       double tf, double step, double dk,
                                       std::vector<Vector6d> &sens, int nthreads);

// 通用增广状态版：x = [r(3), v(3), theta(1..m)]，每个 theta_k 都是阻力项的独立乘性因子
// （如大气密度倍率、阻力系数 Cd、面质比 A/m），RK4 时 theta_k' = 0。
// 用于多参数（较大 m）的可微 Learning：pybind 接口 daAugCoeffs 暴露。
template<typename T>
AlgebraicVector<T> TBPfull_aug(AlgebraicVector<T> x, double t, double arg1);

Vector6d daAugCoeffs(const Vector6d &rv0, const std::vector<double> &params,
                     double tf, int order, double step,
                     std::vector<double> &coeffs,
                     std::vector<std::vector<unsigned int>> &mons);

// 位置相关引力异常场（RBF 势）：势 U(r)=sum_k theta_k exp(-|r-c_k|^2/(2 s^2))，
// 异常加速度 = -grad U。theta(1..m) 为待学习系数（m=中心数），中心 c_k 与宽度 s 固定。
template<typename T>
AlgebraicVector<T> TBPfull_rbf(AlgebraicVector<T> x, double t, double arg1);

Vector6d daAugRBFCoeffs(const Vector6d &rv0, const std::vector<double> &thetas,
                        const std::vector<std::array<double,3>> &centers, double s,
                        double tf, int order, double step,
                        std::vector<double> &coeffs,
                        std::vector<std::vector<unsigned int>> &mons);

// 位置相关引力异常场（低阶非带谐球谐势）：用笛卡尔实球谐(regular solid harmonics)表示，
// 只学非带谐系数 C_lm,S_lm (m>=1)；带谐 J_l 已由 TBPfull 显式处理。
// 势 U = mu * sum_{l=2}^{L} sum_{m=1}^{l} Re^l / r^{2l+1} * (C_lm A_lm + S_lm B_lm)，
// A_lm=r^l P_lm(sin phi)cos(m lambda)、B_lm=...sin...，异常加速度 a = grad U。
// 参数顺序：l=2..L、m=1..l、每个 (C_lm,S_lm)；个数 = L(L+1)-2（L=2→4，L=3→10）。
template<typename T>
AlgebraicVector<T> TBPfull_sh(AlgebraicVector<T> x, double t, double arg1);

Vector6d daAugSHCoeffs(const Vector6d &rv0, const std::vector<double> &thetas,
                       int lmax, double tf, int order, double step,
                       std::vector<double> &coeffs,
                       std::vector<std::vector<unsigned int>> &mons);

// 设定统一力场（RBF / 球谐），供下列多历元算子使用。
void setRBFParams(const std::vector<std::array<double,3>> &centers, double s);
void setSHParams(int lmax);

// 多历元一阶可微算子：一次积分到各 tf，取状态与一阶 Jacobian [∂x/∂x0 (6) | ∂x/∂θ (m)]。
// 只需 order=1（对 m 线性，无二项式爆炸），供 PyTorch 训练一次前向、backward 仅做矩阵乘。
// rvf[k] : 第 k 个历元末态（m）；Jflat 按 [k][输出 i][列 (6+m)] 行主序（∂x_f(m)/∂x0(m)、∂x_f(m)/∂θ）。
void daFieldMultiEpoch(const Vector6d &rv0_m, const std::vector<double> &thetas,
                       const std::vector<double> &tfs, int order, double step,
                       std::vector<Vector6d> &rvf, std::vector<double> &Jflat);

// 变分灵敏度多历元算子：DA 只作用于状态（N=6）求 A=∂f/∂x；积分增广 [x, Φ=∂x/∂x0, S=∂x/∂θ]，
// θ **不进 DA**，代价对 m 线性。一次前向到各 tf；Jflat 同 daFieldMultiEpoch 排布。
void daVarMultiEpoch(const Vector6d &rv0_m, const std::vector<double> &thetas,
                     const std::vector<double> &tfs, double step,
                     std::vector<Vector6d> &rvf, std::vector<double> &Jflat);

// 残差加速度 a_res(r;θ)（m/s^2，前 3 维）——供 fieldBasisJacobian 的 FD 自检。
Vector6d fieldResidualAccel(const Vector6d &rv_m, const std::vector<double> &thetas);

// ∂b_k/∂x（6x6，每个 k）——A_{,θ_k} = ∂²f/∂x∂θ_k（力场基对状态的 Jacobian）。
// 力场由 setRBFParams/setSHParams 设定；dBdx 长度 m*36，排布 [k][i*6+j]。
void fieldBasisJacobian(const double r_km[6], int m, std::vector<double>& dBdx);

// 数值求低阶球谐异常场的残差加速度（仅位置相关部分，供物理自检）：输入位置 m、系数、lmax，
// 返回 m/s^2 的 6 维向量（前 3 为加速度，后 3 为 0）。
Vector6d shResidualAccel(const Vector6d &rv_m, const std::vector<double> &thetas, int lmax);

// 积分伴随（大 m deep Learning）：扩展系统 z=[x(6); Φ(36)]，θ 不进 DA（N=6）。
// 前向用与 rk4 相同的 3/8 RK4 并记录各阶段；反向为离散 RK4 的转置，给出
// ∂L/∂θ（含 deep ∂Φ/∂θ）与 ∂L/∂x0=λ(0)。力场由 setRBFParams/setSHParams 设定。
struct DeepFlow {
    int m = 0;
    std::vector<double> thetas;
    std::vector<double> xs;      // [nsteps][4][6] 各 RK4 阶段的 x (km)
    std::vector<double> Phis;    // [nsteps][4][36] 各阶段的 Φ
    std::vector<double> hs;      // [nsteps]
    std::vector<double> tend;    // [nsteps]
    std::vector<int> epoch_step; // [K] 每个历元所在步（步末时刻=历元时刻）
    std::vector<Vector6d> rvf;   // [K] 历元末态 (km)
    std::vector<double> PhiEpoch;// [K*36]
    std::vector<double> tfs;     // [K]
};
void daDeepForward(const Vector6d &rv0_km, const std::vector<double> &thetas,
                   const std::vector<double> &tfs, double step, DeepFlow &fl);
// gx_epoch: [K] 对 x_f 的种子；gPhi_epoch: [K*36] 对 Φ 的种子。
void daDeepBackward(const DeepFlow &fl, const std::vector<Vector6d> &gx_epoch,
                    const std::vector<double> &gPhi_epoch,
                    Vector6d &gx0, std::vector<double> &gtheta);

// 路径 A：记录型标量（DACE::Scalar + DACE::RecordTape）的流传播（RBF 力场）。
// 以 Scalar 实例化既有 template<T> RHS 即自动记录整条积分图；正向返回 xf(m) 与 tape/叶节点/输出节点，
// 反向用 RecordTape 得到 ∂L/∂x0、∂L/∂θ（代价 ~ 图规模，与 m 无关）。
struct RecordFlow {
    Vector6d xf;
    std::shared_ptr<DACE::RecordTape> tape;
    std::vector<int> leaf_x0;
    std::vector<int> leaf_p;
    std::vector<int> out;
};
RecordFlow daRecordFlowRBF(const Vector6d &rv0, const std::vector<double> &thetas,
                           const std::vector<std::array<double,3>> &centers, double s,
                           double tf, double step);
RecordFlow daRecordFlowSH(const Vector6d &rv0, const std::vector<double> &thetas,
                          int lmax, double tf, double step);
void daRecordFlowBackward(const RecordFlow &rf, const std::vector<double> &grad,
                          Vector6d &gx, std::vector<double> &gp);
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

