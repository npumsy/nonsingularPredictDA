#ifndef QOEJOPT_H
#define QOEJOPT_H

// qoejopt：joint 单发全局联合的 C++/CUDA 内环核心（P4.1）。
//
// 设计：device 常驻装配。Python 侧只负责
//   - 一次 setup：把结构（COO / ISL / GTS 拓扑）与观测（D/W/Q）拷成 device 缓冲；
//   - 每次 assemble：给 X(n,6) 与 rv0/A0（由 qoe 的扁平入口算好，避免 qoejopt 依赖 DACE），
//     核内完成 GVE 多帧 + 两体 STM 折叠 + 装配，A_val/b 直接写进 Python 提供的 torch cuda 缓冲
//     （data_ptr，int），得到 res2；P/A 中间量只驻显存，**不落 host**。
// Python 保留 thin LM 循环 + Theseus Baspacho（外环 θ 学习沿用 BaspachoSolveFunction 原语链式）。
//
// 实现见 src/gve_step.cu（复用同 TU 的 device 助手，无需 -rdc）与 src/qoejopt.cpp（pybind）。

#include <cstdint>
#include <vector>
#include "kepler.h"   // kep3::Vector6d

namespace qoejopt {

// joint 问题上下文：结构/观测/scratch 全部 device 常驻（由 qoejopt.cpp 的 setup 分配）。
// 单发窗口（win_assemble_ss）复用同一 Ctx，置 nfr=L、prior_mode/cov_mode=1：
//   - cov_mode：ISL/GTS 行用 d_P0（每星 6×6 边界先验）白化 1/√(σ²+J P0 Jᵀ)；
//   - prior_mode：以 d_prior_fac/d_zprior 的每星 6×6 信息先验替代标量 damp。
struct Ctx {
    int n = 0, nfr = 0, E = 0, G = 0, ncol = 0, nnz = 0, nrows = 0;
    int off_isl = 0, off_gts = 0, off_damp = 0;
    double dt = 0.0, s_isl = 1.0, s_gts = 1.0, issq = 0.0, sigma_a = 0.0;
    int prior_mode = 0, cov_mode = 0;
    int stm_gve = 0;            // 1=joint 的 d_A 用同源全动力学变分 Φ_oe（`stmFoldGveKernel`），0=二体
    int sub_steps = 1;          // >1：每输出帧=一段，内部 sub_steps 个 dt/sub_steps 子步（大步长段级递推）

    // device 结构/观测（setup 上传）
    int *d_isl_i = nullptr, *d_isl_jj = nullptr, *d_gts_a = nullptr;
    double *d_isl_D = nullptr, *d_isl_W = nullptr;
    double *d_gts_D = nullptr, *d_gts_W = nullptr, *d_gts_Q = nullptr;
    double *d_ctr = nullptr;

    // 单发窗口：每星 6×6 边界先验（白化 + 信息先验）
    double *d_P0 = nullptr;         // (n,36) 边界先验协方差
    double *d_prior_fac = nullptr;  // (n,36) A_prior=chol(P0^-1)^T（行 a 全 6 列）
    double *d_zprior = nullptr;     // (n,6)
    double *d_cov = nullptr;        // (nfr,n,36) 每帧 QOE 协方差（ssCov）
    double *d_phi = nullptr;        // (nfr,n,36) 每帧 QOE 变分 STM Φ_f（ssCov）
    // 逐帧协方差滤波（rv 空间，单发 ncol=6N，iEKF）：边界 P0、P_k^-、P_k^+、上一轮 P^+
    double *d_P0rv = nullptr;       // (n,36)
    double *d_Pm = nullptr;         // (nfr,n,36) P_k^-
    double *d_Pp = nullptr;         // (nfr,n,36) P_k^+
    double *d_PpPrev = nullptr;     // (nfr,n,36) 上一轮 P_k^+
    int *d_adj_j = nullptr;         // (nfr*n,adjD) 每 (帧,星) 的邻居星（-1 空）
    double *d_adj_w = nullptr;      // (nfr*n,adjD) 邻边可见权重
    int adjD = 0;

    // device scratch（setup 分配）
    double *d_X = nullptr, *d_rv0 = nullptr, *d_A0 = nullptr;
    double *d_P = nullptr,   // rv 全帧 (nfr*n*6)
           *d_A = nullptr,   // 位置 Jacobian 折叠 (nfr*n*18)
           *d_phi_all = nullptr,  // 每帧 Φ_oe=∂oe_f/∂oe_0 (nfr*n*36，窗口 P⁻ 跨窗递推用)
           *d_Pw = nullptr,       // 窗口 P 递推 scratch (n*36)
           *d_res = nullptr; // res2 累加器 (1)
};

// 单发窗口：更新观测缓冲（每窗 D/W/Q；结构不变）。
void setObsDevice(Ctx &c, const double *hIslD, const double *hIslW,
                  const double *hGtsD, const double *hGtsW, const double *hGtsQ);
// 单发窗口：更新每星 6×6 边界先验（res2 白化用 d_P0；信息先验见 d_prior_fac/d_zprior）。
void setPriorDevice(Ctx &c, const double *hPriorFac, const double *hZprior, const double *hP0);
// 逐帧协方差滤波：设置 rv 空间边界 P0（6×6/星）。
void setCov0Device(Ctx &c, const double *hP0rv);
// 逐帧协方差滤波：设置每 (帧,星) 的 ISL 邻接表（避免协方差核 O(E) 扫描）。
void setAdjDevice(Ctx &c, const int *hAdjJ, const double *hAdjW, int D, int total);
// 逐帧协方差滤波：mode=1 串行扫描（迭代 1）；mode=2 并行 Jacobi（迭代 ≥2，读上一轮 P^+）。
void ssFrameCovDevice(Ctx &c, int mode);
// 单发窗口：从边界 oe0(n,6) 二体递推 nfr 步（状态与 STM 同源），输出每帧 rv(nfr,n,6) 与
// **rv 空间** 6×6 协方差 P(nfr,n,36)=Φ_rv P0rv Φ_rvᵀ + Q（Q 在 rv 空间；QOE 映射在边界由调用方做）。
void ssCovDevice(const Ctx &c, const double *hX, double *hCov, double *hRv);

// 展开：d_X -> d_P（GVE 多帧 rv）。实现于 gve_step.cu。
void jointExpandDevice(Ctx &c);

// 独立批量 QOE→rv(+Jacobian)（网络训练可微链路）：h_X (n,6) → h_rv (n,6)、h_J (n,36)。
// 设备缓冲位常驻按需扩容（gve_step.cu 实现，复用 oeJacKernel）。
void oe2rvJacHost(const double *h_X, int n, double *h_rv, double *h_J);

// 两体 STM 折叠：d_rv0,d_A0 -> d_A（nfr*n*18）。
void jointStmFoldDevice(Ctx &c);

// 展开 + 装配：写 A_val/b（device 指针，可空表示不写），返回 visible-ISL 残差平方和 res2。
double jointAssembleInto(Ctx &c, double *A_val, double *b);

// 仅装配：用当前 d_P/d_A/d_X 写 A_val/b（不重展开），返回 res2。线搜索接受后刷新用。
double jointAssembleCachedInto(Ctx &c, double *A_val, double *b);

// 展开(GVE)+STM 折叠 + 算 res2（线搜索用；同时刷新 d_P/d_A/d_X，供随后 assemble_cached）。
double jointIslRes2(Ctx &c);

// 把 d_P (nfr*n*6) 拷回 host。
void jointCopyOut(const Ctx &c, double *host_rv);
// 整弧单发：由 QOE 先验 d_P0 与先验均值 d_zprior 算 rv 边界 d_P0rv=J0 P0 J0ᵀ（J0=∂rv/∂oe）。
void jointSetCov0(Ctx &c);

// K 帧单发窗口：P⁻ 跨窗递推 P_next_i = Φ_{s,i} P_i Φ_{s,i}ᵀ + Q_i（均在 QOE 空间；
// Φ 取 d_phi_all 的第 stride 帧=∂oe_{stride}/∂oe_0，Q 由调用方按 rv→oe 映射后传入）。
// hPp/hQ/hPn 均为 (n,36) host；需先 setup 且已 win_fold（d_phi_all 就绪）。
void winPriorPropDevice(Ctx &c, const double *hPp, const double *hQ, int stride, double *hPn);

// 可观测性：装配 H=AᵀA（ncol×ncol，ISL + prior 行）到 d_H；需先 expand（d_P/d_A 就绪）。
void obsFormHDevice(Ctx &c, double *d_H);
// H 对称特征分解（cuSOLVER syevd）：返回全谱升序 eval(ncol) 与最小 k 个特征向量 evec(ncol,k)。
void obsEigDevice(Ctx &c, const double *d_H, int k, std::vector<double> &eval,
                  std::vector<double> &evec);
// 逐星有效 Jacobian Gram 的 6 个特征（奇异值²，升序）：h_sv2 (n,6)。需先 expand。
void obsPerStarDevice(Ctx &c, double *h_sv2);

// 细粒度计时（秒，累计）：[expand(GVE), stmfold, assemble, res2]。
void jointResetTiming();
std::vector<double> jointGetTiming();

// P4.2 前向敏度：S_f = ∂oe_f/∂κ（变分方程 dS/dt = (∂f/∂oe)S + ∂f/∂κ，与状态同一 3/8-RK4；
// ∂f/∂oe、∂f/∂κ 用设备端中心差分，κ=逐星阻力因子 beta）。输出 oe_all 与 S_all（均 nfr*n*6）。
void jointStateSensBatch(const Ctx &c, const double *x_host, double beta, double *oe_all, double *S_all);
void dbgJac(const double *oe_host, int n, double beta, double *Aout, double *Fout);

// 单步 GVE + 全动力学变分 STM（Φ_oe=∂oe(dt)/∂oe(0)，设备端中心差分 Jacobian）；供 EKF-qoe 预测。
// d_oe (n×6) → d_oef (n×6) 与 d_Phi (n×36)。
void gveStmDevice(const double *d_oe, int n, double dt, double beta, double *d_oef, double *d_Phi);
// 优化版（6 线程/星，按变分列拆分；数值与 gveStmDevice 一致）
void gveStmDeviceGrp(const double *d_oe, int n, double dt, double beta, double *d_oef,
                     double *d_Phi);

// 一阶同源 STM：Φ ≈ I + A·Δt（A=∂f/∂oe 解析，段首求一次）。FLS 的 dyn 因子雅可比用。
void gveStmIADev(const double *d_oe, int n, double dt, double beta, double *d_oef, double *d_Phi);

// GVE 控制矩阵 B=∂(oe_dot)/∂a（ECI 3 轴，每星 6×3）：FLS 过程噪声 x_dot=A·x+B·u。d_B=(n,6,3)。
void gveCtrlDev(const double *d_oe, int n, double *d_B);

// 窗内 dyn 二因子（3N 并行）：x_lo/x_hi (nseg,n,6) → res(nseg,n,6)、Φ(nseg,n,36)。
void gveDynFactorDev(const double *d_xlo, const double *d_xhi, int nseg, int n,
                     double dt, double beta, double *d_res, double *d_Phi);

// FLS 逐星边缘先验行写入（n 星，pnode 节点，A_val/b 的 off_pri/base_b 处）。
void winPriorAsmDevice(const double *d_X, const double *d_pri_fac, const double *d_zp, int n,
                       int pnode, int off_pri, int base_b, double *A_val, double *b);

// FLS 每步几何（全 CUDA）：QOE(n,6,device) → rv(n,6)、∂p/∂oe 前三行((n,18))。
void oeRvJacDevice(const double *d_oe, int n, double *d_rv, double *d_Ap);

// FLS 逐星边缘（P1'，每星一线程）：装配值/列均衡 + 三组行表（Λ00/Λ11 贡献、Λ10 配对）→ Lp、Pmarg。
void flsMargDevice(const double *d_Aval, const double *d_sc,
                   const int64_t *d00_p, const int64_t *d00_s, const int64_t *d00_c,
                   const int64_t *d11_p, const int64_t *d11_s, const int64_t *d11_c,
                   const int64_t *d_prs1, const int64_t *d_prs0, int n,
                   double *d_Lp, double *d_Pm);

// 同源全动力学 STM 折叠：d_A(nfr,n,18)=∂p_f/∂oe_0（链式变分 Φ_oe，`gveStmStep_dev`）。
void stmFoldGveDevice(const double *d_oe, int n, int nfr, double dt, double beta, double *d_A);

// P4.2/CUDA：整弧 RBF θ 前向敏度 S_f = ∂oe_f/∂θ（nfr,n,6,m）；centers/s 需先 qoe.setRbfCuda 预设。
// 变分方程 dS/dt = (∂f/∂oe)S + ∂f/∂θ（∂f/∂θ_k = RHS(ap=第 k 基加速度)），与状态同一 3/8-RK4。
void gveRbfSensBatch(const std::vector<kep3::Vector6d> &oe0s, int nfr, double dt, double beta, int m,
                     std::vector<double> &oe_all, std::vector<double> &S_all);

// ===================== L 节点批量窗口（每帧一节点）device 常驻装配 =====================
// 变量 = L 个节点 × n 星 × 6（QOE），**ncol = 6NL**；因子 = 每帧 ISL/GTS + 帧间 dyn 连续性
// （单步 φ、Φ）+ damp。ISL/GTS/damp 复用单发 `joint*` 核（令 nfr=L）；dyn 用 `winDynAsmKernel`。
struct WinCtx {
    int n = 0, L = 0, E = 0, G = 0, ncol = 0, nnz = 0, nrows = 0;
    int off_isl = 0, off_gts = 0, off_dyn = 0, off_damp = 0;
    double dt = 0.0, s_isl = 1.0, s_gts = 1.0, issq = 0.0;
    int prior = 0;                                             // 1 = 末组为每星 6×6 先验行（win_assemble 不写 damp）
    int q_perstar = 0;                                         // 1 = qingt 为 (n,36) 按星索引（否则共享 6×6）
    int *d_isl_i = nullptr, *d_isl_jj = nullptr, *d_gts_a = nullptr;
    double *d_isl_D = nullptr, *d_isl_W = nullptr;
    double *d_gts_D = nullptr, *d_gts_W = nullptr, *d_gts_Q = nullptr;
    double *d_ctr = nullptr, *d_qingt = nullptr;
    double *d_X = nullptr, *d_rv = nullptr, *d_Ap = nullptr;   // 节点 QOE / rv / ∂p/∂oe(18)
    double *d_end = nullptr, *d_phi = nullptr;                 // ((L-1)*n) 单步 φ 末态(QOE) 与 Φ_oe(36)
    double *d_res = nullptr;
};

// L 节点批量窗口装配：host 传 X(L*n,6,QOE)、rv(L*n,6)、Ap(L*n,18)=∂p/∂oe；写 A_val/b(device)；返回 ISL res2。
double winAssembleDevice(WinCtx &c, const double *hX, const double *hRv, const double *hAp,
                         double *A_val, double *b);
void winResetTiming();
std::vector<double> winGetTiming();
// 仅算 ISL res2（线搜索用）：假定 WinCtx.d_rv 已刷新；不做 A/b 装配、不算 STM。
double winRes2Only(WinCtx &c);

// FLS 逐帧消去（sweep，scaled 空间）：每星一线程，装本帧 S_m,ii = ISL@m+GTS@m+dynNext 本帧侧
// +dynPrev 上帧侧+先验(m=0)−Δ_{m-1} → 6×6 逆 → 输出 dlt_out = −Cᵀ S_m,ii⁻¹ C（C = A_mᵀ A_n，
// A_m/A_n = dynNext 行两侧 6×6）与 gam_out = Cᵀ y（y = 本帧解）。m = L−1（末帧）时不写 Δ/γ。
void flsFrameSweepDevice(const double *d_Aval, const double *d_sc, int n, int L, int E, int G,
                         int m, int off_isl, int off_gts, int off_dyn, int off_damp,
                         const int64_t *isl_p, const int64_t *isl_s, const int64_t *isl_c,
                         const int64_t *gts_p, const int64_t *gts_s, const int64_t *gts_c,
                         const double *d_dlt_prev, const double *d_y,
                         double *d_dlt_out, double *d_gam_out);
// 回代：out = C_m·z（C_m = dyn(m→m+1) 行两侧 6×6 的耦合；scaled 空间）。
void flsCouplingDevice(const double *d_Aval, const double *d_sc, int n, int m, int off_dyn,
                       const double *d_z, double *d_out);

// FLS L=1（EKF 式单帧交接）：每星 P⁺=(prior+ISL+GTS)⁻¹（scaled→物理）→ P⁻=ΦP⁺Φᵀ+Q → 白化 Lp＝chol⁻ᵀ(P⁻)。
// d00 行表复用（prior@0+ISL@0+GTS@0+dyn），核内跳过每星最后一条（dyn(0→1)，L=1 无该行）。
// d_phi=(n,6,6) 单步同源 Φ；d_linv=(n,6,6) 为 Q^{-1/2}（Linv = Lq⁻¹，Q=Linv⁻¹Linv⁻ᵀ）。
void flsEkfHandoffDevice(const double *d_Aval, const double *d_sc,
                         const int64_t *d00_p, const int64_t *d00_s, const int64_t *d00_c,
                         const double *d_phi, const double *d_linv, int n,
                         double *d_Lp, double *d_Pm);

}  // namespace qoejopt

#endif  // QOEJOPT_H
