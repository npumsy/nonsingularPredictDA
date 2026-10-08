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
    const double J3 = -2.532435345754e-6 ;   // J3 harmonic（doc/HG-iod.md l89-109 取负号）
    // const double J3 = 0 ;    // J3 harmonic
    const double J4 = -1.619331205072e-6 ;   // J4 harmonic（doc/HG-iod.md l89-109 取负号）
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

// 整星座批处理：**解析**（κ 进 DA 第 7 变量）出 x_f 与 ∂x_f/∂κ（非 FD；`dk` 忽略）。
// kappas 长度 1（广播）或 N（每星一个）。返回终端状态 xf；sens 写入 ∂x_f/∂κ。
// nthreads<=0 用默认线程数。可 OpenMP 并行。
// 批 double 位置相关残差力场传播（场参数由 setRBFParams/setSHParams 设定）；返回 xf，
// 并 FD 出 ∂x_f/∂θ（Jt, N*6*m）与 ∂x_f/∂x0（Jx, N*6*6）。
std::vector<Vector6d> daFieldBatchD(const std::vector<Vector6d> &rv0s,
                                    const std::vector<double> &thetas,
                                    double tf, double step, double dk, int nthreads,
                                    std::vector<double> &Jt, std::vector<double> &Jx);
// 多历元批传播（每星一次连续积分，记录各历元状态与 ∂x/∂θ、∂x/∂x0）；场参数由 setRBF/SHParams 设定。
void daFieldMultiEpochBatchD(const std::vector<Vector6d> &rv0s,
                             const std::vector<double> &thetas,
                             const std::vector<double> &tfs, double step, double dk, int nthreads,
                             std::vector<double> &xf, std::vector<double> &Jt, std::vector<double> &Jx);
// 解析（变分）多历元批：逐星调 daVarMultiEpoch（A=∂f/∂x 由 DA(N=6) 精确给出，非 FD），
// 返回 xf(n*K*6) 与 Jx(n*K*6*6)=∂x/∂x0。θ 为空时即参考场（J234+阻力）。
void daVarMultiEpochBatch(const std::vector<Vector6d> &rv0s,
                          const std::vector<double> &thetas,
                          const std::vector<double> &tfs, double step, int nthreads,
                          std::vector<double> &xf, std::vector<double> &Jx);
// 批量并行【解析变分】：A=∂f/∂x 由线程局部 DACE(1,6) 精确给，[x,Φ,S] 以 double 积分；
// 返回 xf(N*K*6,m)、Jt(N*K*6*m)、Jx(N*K*6*6)；场参数由 setRBF/SHParams 预设。nthreads<=0 默认。
void daVarMultiEpochBatchP(const std::vector<Vector6d> &rv0s,
                           const std::vector<double> &thetas,
                           const std::vector<double> &tfs, double step, int nthreads,
                           std::vector<double> &xf, std::vector<double> &Jt, std::vector<double> &Jx);

// 批量解析变分 + **κ 列（解析，非 FD）**：xf(N*K*6)、Jt(N*K*6*m)=∂x/∂θ、Jx(N*K*6*6)=∂x/∂x0、Jk(N*K*6)=∂x/∂κ。
// κ 走变分增广（dK/dt=A·K+Fκ，Fκ=名义阻力），θ 不进 DA。场参数由 setSHParams 预设。nthreads<=0 默认。
void daVarMultiEpochBatchPSKC(const std::vector<Vector6d> &rv0s,
                              const std::vector<double> &thetas, int lmax,
                              const std::vector<double> &tfs, double step, int nthreads,
                              std::vector<double> &xf, std::vector<double> &Jt,
                              std::vector<double> &Jx, std::vector<double> &Jk);
std::vector<Vector6d> daJ234DragBatchD(const std::vector<Vector6d> &rv0s,
                                       const std::vector<double> &kappas,
                                       double tf, double step, double dk,
                                       std::vector<Vector6d> &sens, int nthreads);

// 多历元阻力灵敏度：每星**一次连续积分**，各历元输出 xf 与 **解析** ∂x/∂κ（κ 进 DA 线性系数，非 FD）。
// xf/sens 展平为 [i*K*6 + k*6 + c]（i=星、k=历元、c=状态分量）。线程安全、OpenMP、释放 GIL。
void daJ234DragMultiEpochBatch(const std::vector<Vector6d> &rv0s,
                               const std::vector<double> &kappas,
                               const std::vector<double> &tfs,
                               double step, double dk, int nthreads,
                               std::vector<double> &xf, std::vector<double> &sens);

// C++ 批量单步 StateTransfer：**一步 3/8 RK4**（全 TBPfull 动力学：二体+J234+阻力）给 x(dt)；
// STM 用**解析二体变分** A=∂f_two/∂x（沿同一步 RK4 积分，非 FD）。纯 double、OpenMP、释放 GIL。
// 单位 m / m·s⁻¹；Phi 展平 [i*36 + 6*row + col]，无量纲。
void stateTransferBatch(const std::vector<Vector6d> &rv0s, double dt, int nthreads,
                        std::vector<Vector6d> &xf, std::vector<double> &Phi);

// 解析两体 STM 折叠（一次算完整弧）：A[f] = (Φ_rv(t_f)·A0)[0:3, :]；Aout 展平 nfr*n*18。
void stateStmFoldBatch(const std::vector<Vector6d> &rv0s, const std::vector<double> &dts,
                       const std::vector<double> &A0flat, int nthreads, std::vector<double> &Aout);

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

// 三体（日/月）确定性摄动开关 + 传播起点绝对历元（MJD）。开启后 forceAccel/TBPfull 内叠加日月第三体。
void setThirdBody(bool on);
void setPropEpoch(double mjd0);

// 多历元一阶可微算子：一次积分到各 tf，取状态与一阶 Jacobian [∂x/∂x0 (6) | ∂x/∂θ (m)]。
// 只需 order=1（对 m 线性，无二项式爆炸），供 PyTorch 训练一次前向、backward 仅做矩阵乘。
// rvf[k] : 第 k 个历元末态（m）；Jflat 按 [k][输出 i][列 (6+m)] 行主序（∂x_f(m)/∂x0(m)、∂x_f(m)/∂θ）。
void daFieldMultiEpoch(const Vector6d &rv0_m, const std::vector<double> &thetas,
                       const std::vector<double> &tfs, int order, double step,
                       std::vector<Vector6d> &rvf, std::vector<double> &Jflat);

// D1：多历元**稠密泰勒系数**导出（一次连续积分，每个 tf 记录 6 个状态的 (mons, coeffs)）。
// 用于在 Python/torch 侧对任意 (x0, θ) 做**精确非线性**求值（不重跑 DA），并支持 order>=2 二阶项。
// 注意：状态 DA 内部单位为 km；coeffs 为该 km 状态的泰勒系数，rvf 为 m。
// coeffs 排布 [k][输出状态 i][单式项]，长度 K*6*nmono；mons 长度 nmono，每个长度 N=6+m。
void daFieldMultiEpochCoeffs(const Vector6d &rv0_m, const std::vector<double> &thetas,
                             const std::vector<double> &tfs, int order, double step,
                             std::vector<double> &rvf, std::vector<double> &coeffs,
                             std::vector<std::vector<unsigned int>> &mons);

// 批量并行【解析】DA 多历元展开（DACE WITH_PTHREAD + OpenMP）：主线程 DA::init 一次，
// 每线程 daceInitializeThread，一次连续积分给各 tf 的 x、Φ=∂x/∂x0、∂x/∂θ（解析系数，非 FD）。
// 场参数由 setRBFParams/setSHParams 预先设定。排布同 daFieldMultiEpochBatchD：
// xf(N*K*6, m)、Jt(N*K*6*m, m/θ)、Jx(N*K*6*6)。nthreads<=0 用默认。
void daFieldMultiEpochBatchDA(const std::vector<Vector6d> &rv0s,
                              const std::vector<double> &thetas,
                              const std::vector<double> &tfs, int order, double step, int nthreads,
                              std::vector<double> &xf, std::vector<double> &Jt, std::vector<double> &Jx);

// 同 daFieldMultiEpochBatchDA，但额外输出**二阶 Hessian**（对 x0，N=6 部分；order 须≥2）。
// Hess 排布 (N*K*6, 36)：[(i*K+k)*6+c]*36 + a*6+b = ∂²x_c/∂δ_a∂δ_b（归一化单位、对称）。
void daFieldMultiEpochBatchDA2(const std::vector<Vector6d> &rv0s,
                               const std::vector<double> &thetas,
                               const std::vector<double> &tfs, int order, double step, int nthreads,
                               std::vector<double> &xf, std::vector<double> &Jt, std::vector<double> &Jx,
                               std::vector<double> &Hess);

// 变分灵敏度多历元算子：DA 只作用于状态（N=6）求 A=∂f/∂x；积分增广 [x, Φ=∂x/∂x0, S=∂x/∂θ]，
// θ **不进 DA**，代价对 m 线性。一次前向到各 tf；Jflat 同 daFieldMultiEpoch 排布。
void daVarMultiEpoch(const Vector6d &rv0_m, const std::vector<double> &thetas,
                     const std::vector<double> &tfs, double step,
                     std::vector<Vector6d> &rvf, std::vector<double> &Jflat);

// 残差加速度 a_res(r;θ)（m/s^2，前 3 维）——供 fieldBasisJacobian 的 FD 自检。
Vector6d fieldResidualAccel(const Vector6d &rv_m, const std::vector<double> &thetas);

// 摄动加速度（去二体，ECI，m/s²）：J234 + 阻力(乘 κ) + 三体(若开) + SH(θ)。供非奇异要素 Gauss 变分方程。
Vector6d pertAccelECI(const Vector6d &rv_m, double kappa, const std::vector<double> &thetas, int lmax);

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

// 非奇异要素 (a, u=M+ω, e_x=e cosω, e_y=e sinω, i, Ω) 的一阶 DA 误差传播。
// 状态口径：a 为 **m**，角度为 **rad**（与 `qoe` 其余入口一致；内部 rv 亦为 m）。
// 递推方程 = 高斯变分方程（去二体摄动 a_R/a_T/a_N），摄动取 TBPfull（二体+J234+阻力，三体随全局开关），
// 与 `script/core/da_engine.py::_gve_rhs` 同一组公式（dM 的系数取经典 GVE 的 √(1−e²)/(h e)）。
// θ 不进（同 NominalErrorProp，纯 J234+阻力）。
// 注：e ≈ 0 时 ω 不定，取 0（与 `osculating::OEOsc2rv` 同口径）；真实轨道 e~1e-3 不受影响。
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

// 非奇异要素→rv（m, m/s）：与 `osculating::OEOsc2rv` **同一公式**（T=double 时 rel ~1e-15）。
// 模板化是为让 GVE 的 DA 递推能对要素求导；此 double 实例供 gtest / Python 对拍。
Vector6d noeOsc2rv(const Vector6d &oe, int MaxIt = 100, double epsl = 1e-12);
// 高斯变分方程 RHS（非奇异要素 [a,u,ex,ey,i,Om]，m/rad）；beta 为阻力乘性因子（κ），t 为传播时间（三体）。
Vector6d noeGveRhs(const Vector6d &oe, double beta = 1.0, double t = 0.0);
// 批量并行 GVE 一步传播（EKF 用）：oe_f (n×6) 与 Phi=∂oe_f/∂oe_0 (n×36, 行主序)。
void stateTransferGVEBatch(const std::vector<Vector6d> &oes, double tf, int nthreads,
                           std::vector<Vector6d> &oef, std::vector<double> &Phi, double step = 10.0);
// 批量并行 ∂(r,v)/∂oe（复用 DA 可微的 OEOsc2rvT），返回 n×36 行主序。
void noeOsc2rvJacBatch(const std::vector<Vector6d> &oes, int nthreads, std::vector<double> &J);
// 批量并行 非奇异要素→rv（纯 double），返回 n×6 行主序。
void noeOsc2rvBatch(const std::vector<Vector6d> &oes, int nthreads, std::vector<double> &RV);
// 批量并行 rv→非奇异要素，返回 n×6 行主序。
void rv2OEOscBatch(const std::vector<Vector6d> &rvs, int nthreads, std::vector<double> &OE);
#endif

