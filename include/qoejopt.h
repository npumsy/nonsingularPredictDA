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

namespace qoejopt {

// joint 问题上下文：结构/观测/scratch 全部 device 常驻（由 qoejopt.cpp 的 setup 分配）。
struct Ctx {
    int n = 0, nfr = 0, E = 0, G = 0, ncol = 0, nnz = 0, nrows = 0;
    int off_isl = 0, off_gts = 0, off_damp = 0;
    double dt = 0.0, s_isl = 1.0, s_gts = 1.0, issq = 0.0;

    // device 结构/观测（setup 上传）
    int *d_isl_i = nullptr, *d_isl_jj = nullptr, *d_gts_a = nullptr;
    double *d_isl_D = nullptr, *d_isl_W = nullptr;
    double *d_gts_D = nullptr, *d_gts_W = nullptr, *d_gts_Q = nullptr;
    double *d_ctr = nullptr;

    // device scratch（setup 分配）
    double *d_X = nullptr, *d_rv0 = nullptr, *d_A0 = nullptr;
    double *d_P = nullptr,   // rv 全帧 (nfr*n*6)
           *d_A = nullptr,   // 位置 Jacobian 折叠 (nfr*n*18)
           *d_res = nullptr; // res2 累加器 (1)
};

// 展开：d_X -> d_P（GVE 多帧 rv）。实现于 gve_step.cu。
void jointExpandDevice(Ctx &c);

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

// 细粒度计时（秒，累计）：[expand(GVE), stmfold, assemble, res2]。
void jointResetTiming();
std::vector<double> jointGetTiming();

// P4.2 前向敏度：S_f = ∂oe_f/∂κ（变分方程 dS/dt = (∂f/∂oe)S + ∂f/∂κ，与状态同一 3/8-RK4；
// ∂f/∂oe、∂f/∂κ 用设备端中心差分，κ=逐星阻力因子 beta）。输出 oe_all 与 S_all（均 nfr*n*6）。
void jointStateSensBatch(const Ctx &c, const double *x_host, double beta, double *oe_all, double *S_all);

// 单步 GVE + 全动力学变分 STM（Φ_oe=∂oe(dt)/∂oe(0)，设备端中心差分 Jacobian）；供 EKF-qoe 预测。
// d_oe (n×6) → d_oef (n×6) 与 d_Phi (n×36)。
void gveStmDevice(const double *d_oe, int n, double dt, double beta, double *d_oef, double *d_Phi);

}  // namespace qoejopt

#endif  // QOEJOPT_H
