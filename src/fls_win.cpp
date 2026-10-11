// FLS 固定滞后平滑 C++ 核心：factor 构造 / 滑窗结构 / Schur 边缘化全部在 C++。
// 依赖：qoe_core 的 GVE 核（gve_step.cu）+ BaSpaCho（theseus，源码编入）+ torch。
// 本轮为骨架：确认 Baspacho/torch 在 C++ 里编过并可互通（含 add_MtM/factor/solve/compute_diag_blocks 全链）。
// 注意：调用 torch C++ 前，Python 侧必须先 `import torch`（否则 torch python 侧未初始化 → 段错误）。
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <memory>
#include <vector>

#include <Eigen/Dense>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "dastates.h"
#include "extlib/baspacho_solver.h"
#include "kepler.h"
#include "qoejopt.h"

namespace py = pybind11;

// 探针：确认 torch 链接正常（返回一个 torch 张量）。
static torch::Tensor fls_probe(int64_t n) {
    auto t = torch::zeros({n, 6}, torch::TensorOptions().dtype(torch::kFloat64));
    return t;
}

// 探针：复刻 validate_qoejopt 的规范结构（2 星、1 条 ISL 行 + 每星 6 行先验），
// 走完 SymbolicDecomposition → add_MtM → factor → solve → compute_diag_blocks(6)，
// 与 Eigen 稠密 AᵀA 的逆对角块 + 解对拍。返回 {块误差, 参考幅值, 解误差}。
static std::vector<double> fls_baspacho_probe() {
    auto opts = torch::TensorOptions().dtype(torch::kInt64);
    auto optsD = torch::TensorOptions().dtype(torch::kFloat64);
    const char *dev = "cuda";
    auto td = torch::Device(dev);
    // 符号结构（块级 CSR，CPU 张量）：2 个 6 维块、满 2×2 图案
    auto paramSize = torch::tensor({6, 6}, opts);
    auto bptrs = torch::tensor({0, 2, 4}, opts);
    auto binds = torch::tensor({0, 1, 0, 1}, opts);
    SymbolicDecomposition sym(paramSize, bptrs, binds, dev);
    auto num = sym.createNumericDecomposition(1);
    // A 标量 CSR：行 0 = ISL（块0 六列 + 块1 六列）；行 1..12 = 每星 6 行先验（各 6 非零）
    std::vector<int64_t> rp{0}, ci;
    for (int k = 0; k < 12; ++k) ci.push_back(k);
    rp.push_back((int64_t)ci.size());
    for (int i = 0; i < 2; ++i)
        for (int a = 0; a < 6; ++a) {
            for (int k = 0; k < 6; ++k) ci.push_back(i * 6 + k);
            rp.push_back((int64_t)ci.size());
        }
    const int64_t nnz = (int64_t)ci.size();                 // 12 + 72 = 84
    auto ptrs = torch::tensor(rp, opts).to(td);
    auto inds = torch::tensor(ci, opts).to(td);
    auto val = (torch::rand({1, nnz}, optsD) * 0.5 + 0.5).to(td);
    num.add_MtM(val, ptrs, inds);
    num.factor();
    auto x = torch::ones({1, 12}, optsD).to(td);
    num.solve(x);
    auto blk = num.compute_diag_blocks(6).cpu();            // (2,6,6)，span 序
    auto perm = num.param_to_span().cpu();                  // 调序 → 调用序
    // 稠密参考（Eigen）
    auto val_cpu = val.cpu();
    auto vc = val_cpu.accessor<double, 2>();
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(13, 12);
    for (int64_t r = 0; r < 13; ++r)
        for (int64_t j = rp[r]; j < rp[r + 1]; ++j) A(r, ci[j]) = vc[0][j];
    Eigen::MatrixXd Hi = (A.transpose() * A).inverse();
    auto bc = blk.accessor<double, 3>();
    auto pc = perm.accessor<int64_t, 1>();
    auto x_cpu = x.cpu();
    auto xc = x_cpu.accessor<double, 2>();
    double err = 0.0, refmax = 0.0, xerr = 0.0;
    for (int b = 0; b < 2; ++b) {
        int ib = (int)pc[b];
        for (int p = 0; p < 6; ++p)
            for (int q = 0; q < 6; ++q) {
                double ref = Hi(b * 6 + p, b * 6 + q);
                err = std::max(err, std::abs(bc[ib][p][q] - ref));
                refmax = std::max(refmax, std::abs(ref));
            }
    }
    for (int r = 0; r < 12; ++r) {
        double ref = 0.0;
        for (int c = 0; c < 12; ++c) ref += Hi(r, c);
        xerr = std::max(xerr, std::abs(xc[0][r] - ref));
    }
    return {err, refmax, xerr};
}

// ===================== FLS 滑窗结构（C++，等价 py `_lift_struct`/`_lift_block_struct`） =====================
// 行组顺序：ISL(L*E × 12) → GTS(L*G × 6) → dyn((L-1)*n*6 × 12) → 先验(L*n*6 × 6) 或 damp(ncol × 1)。
struct FlsStruct {
    int n = 0, L = 0, E = 0, G = 0;
    int ncol = 0, nrows = 0, nnz = 0;
    int nISL = 0, nGTS = 0, nDYN = 0, nPRI = 0;
    int off_isl = 0, off_gts = 0, off_dyn = 0, off_damp = 0;
    double issq = 0.0;
    std::vector<int32_t> A_rp, A_ci;
    std::vector<int64_t> bptrs, binds;
};

static FlsStruct fls_struct_impl(int n, int L,
                                 const std::vector<int> &isl_i, const std::vector<int> &isl_j,
                                 const std::vector<int> &gts, double damp, bool prior) {
    FlsStruct s;
    s.n = n; s.L = L; s.E = (int)isl_i.size(); s.G = (int)gts.size();
    s.nISL = L * s.E; s.nGTS = L * s.G; s.nDYN = (L - 1) * n * 6;
    s.nPRI = prior ? L * n * 6 : 0;
    s.nrows = s.nISL + s.nGTS + s.nDYN + s.nPRI + (prior ? 0 : (L * n * 6));
    s.ncol = L * n * 6;
    s.off_isl = 0;
    s.off_gts = s.nISL * 12;
    s.off_dyn = s.nISL * 12 + s.nGTS * 6;
    s.off_damp = s.nISL * 12 + s.nGTS * 6 + s.nDYN * 12;
    s.issq = std::sqrt(damp);
    auto &rp = s.A_rp; auto &ci = s.A_ci;
    rp.reserve((size_t)s.nrows + 1); rp.push_back(0);
    for (int k = 0; k < L; ++k)
        for (int e = 0; e < s.E; ++e) {
            const int b0 = (k * n + isl_i[e]) * 6, b1 = (k * n + isl_j[e]) * 6;
            for (int c = 0; c < 6; ++c) ci.push_back(b0 + c);
            for (int c = 0; c < 6; ++c) ci.push_back(b1 + c);
            rp.push_back((int32_t)ci.size());
        }
    for (int k = 0; k < L; ++k)
        for (int g = 0; g < s.G; ++g) {
            const int b = (k * n + gts[g]) * 6;
            for (int c = 0; c < 6; ++c) ci.push_back(b + c);
            rp.push_back((int32_t)ci.size());
        }
    for (int k = 1; k < L; ++k)
        for (int i = 0; i < n; ++i) {
            const int b0 = ((k - 1) * n + i) * 6, b1 = (k * n + i) * 6;
            for (int comp = 0; comp < 6; ++comp) {              // 每 (k,i) 6 行（QOE 分量各一行）
                for (int c = 0; c < 6; ++c) ci.push_back(b0 + c);
                for (int c = 0; c < 6; ++c) ci.push_back(b1 + c);
                rp.push_back((int32_t)ci.size());
            }
        }
    if (prior) {
        for (int r = 0; r < L * n * 6; ++r) {
            const int b = (r / 6) * 6;
            for (int c = 0; c < 6; ++c) ci.push_back(b + c);
            rp.push_back((int32_t)ci.size());
        }
    } else {
        for (int c = 0; c < s.ncol; ++c) {
            ci.push_back(c);
            rp.push_back((int32_t)ci.size());
        }
    }
    s.nnz = (int)ci.size();
    const int nblk = L * n;
    std::vector<std::vector<int>> adj(nblk);
    auto addb = [&](int a, int b) { if (a != b) { adj[a].push_back(b); adj[b].push_back(a); } };
    for (int k = 0; k < L; ++k)
        for (int e = 0; e < s.E; ++e) addb(k * n + isl_i[e], k * n + isl_j[e]);
    for (int k = 1; k < L; ++k)
        for (int i = 0; i < n; ++i) addb((k - 1) * n + i, k * n + i);
    s.bptrs.reserve((size_t)nblk + 1); s.bptrs.push_back(0);
    for (int i = 0; i < nblk; ++i) {
        auto &v = adj[i];
        v.push_back(i);
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
        for (int c : v) s.binds.push_back(c);
        s.bptrs.push_back((int64_t)s.binds.size());
    }
    return s;
}

// ===================== FLS 逐帧消去（sweep）用单帧结构（信息形式，l430-433 选项 2） =====================
// 行组：ISL(E×12) → GTS(G×6) → dynNext(n*6×6) → dynPrev(n*6×6) → 先验(n*6×6)；
// 消去修正（−Δ_{m-1}）不经行、用 NumericDecomposition.add_M 直接进数值矩阵。
struct FlsFrameStruct {
    int n = 0, E = 0, G = 0;
    int ncol = 0, nrows = 0, nnz = 0;
    int off_isl = 0, off_gts = 0, off_dynN = 0, off_dynP = 0, off_prior = 0;  // A_val 值偏移
    int r_isl = 0, r_gts = 0, r_dynN = 0, r_dynP = 0, r_prior = 0;            // b 行基
    std::vector<int32_t> A_rp, A_ci;
    std::vector<int64_t> bptrs, binds;
};

static FlsFrameStruct fls_frame_impl(int n, const std::vector<int> &isl_i,
                                     const std::vector<int> &isl_j, const std::vector<int> &gts) {
    FlsFrameStruct s;
    s.n = n; s.E = (int)isl_i.size(); s.G = (int)gts.size();
    s.ncol = n * 6;
    auto &rp = s.A_rp; auto &ci = s.A_ci;
    rp.push_back(0);
    for (int e = 0; e < s.E; ++e) {                             // ISL@m
        const int b0 = isl_i[e] * 6, b1 = isl_j[e] * 6;
        for (int c = 0; c < 6; ++c) ci.push_back(b0 + c);
        for (int c = 0; c < 6; ++c) ci.push_back(b1 + c);
        rp.push_back((int32_t)ci.size());
    }
    s.off_isl = 0; s.r_isl = 0; s.nrows = s.E;
    s.off_gts = (int)ci.size(); s.r_gts = s.nrows; s.nrows += s.G;
    for (int g = 0; g < s.G; ++g) {                             // GTS@m
        const int b = gts[g] * 6;
        for (int c = 0; c < 6; ++c) ci.push_back(b + c);
        rp.push_back((int32_t)ci.size());
    }
    s.off_dynN = (int)ci.size(); s.r_dynN = s.nrows; s.nrows += n * 6;
    for (int i = 0; i < n; ++i)                                 // dynNext（本帧的 −qΦ 侧）
        for (int comp = 0; comp < 6; ++comp) {
            for (int c = 0; c < 6; ++c) ci.push_back(i * 6 + c);
            rp.push_back((int32_t)ci.size());
        }
    s.off_dynP = (int)ci.size(); s.r_dynP = s.nrows; s.nrows += n * 6;
    for (int i = 0; i < n; ++i)                                 // dynPrev（上帧的 +q 侧）
        for (int comp = 0; comp < 6; ++comp) {
            for (int c = 0; c < 6; ++c) ci.push_back(i * 6 + c);
            rp.push_back((int32_t)ci.size());
        }
    s.off_prior = (int)ci.size(); s.r_prior = s.nrows; s.nrows += n * 6;
    for (int i = 0; i < n; ++i)                                 // 先验
        for (int comp = 0; comp < 6; ++comp) {
            for (int c = 0; c < 6; ++c) ci.push_back(i * 6 + c);
            rp.push_back((int32_t)ci.size());
        }
    s.nnz = (int)ci.size();
    std::vector<std::vector<int>> adj(n);
    for (int e = 0; e < s.E; ++e) {
        const int a = isl_i[e], b = isl_j[e];
        if (a != b) { adj[a].push_back(b); adj[b].push_back(a); }
    }
    s.bptrs.push_back(0);
    for (int i = 0; i < n; ++i) {
        auto &v = adj[i];
        v.push_back(i);
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
        for (int c : v) s.binds.push_back(c);
        s.bptrs.push_back((int64_t)s.binds.size());
    }
    return s;
}

// py 门面（结构对拍用）。
static py::dict fls_build_struct(int n, int L,
                                 py::array_t<int, py::array::c_style | py::array::forcecast> ei,
                                 py::array_t<int, py::array::c_style | py::array::forcecast> ej,
                                 py::array_t<int, py::array::c_style | py::array::forcecast> gts,
                                 double damp) {
    auto req_i = ei.request(); auto req_j = ej.request(); auto req_g = gts.request();
    const int *pi = static_cast<const int *>(req_i.ptr);
    const int *pj = static_cast<const int *>(req_j.ptr);
    const int *pg = static_cast<const int *>(req_g.ptr);
    std::vector<int> vi(pi, pi + req_i.size), vj(pj, pj + req_j.size), vg(pg, pg + req_g.size);
    FlsStruct s = fls_struct_impl(n, L, vi, vj, vg, damp, true);
    py::dict d;
    d["n"] = s.n; d["L"] = s.L; d["E"] = s.E; d["G"] = s.G;
    d["ncol"] = s.ncol; d["nrows"] = s.nrows; d["nnz"] = s.nnz;
    d["nISL"] = s.nISL; d["nGTS"] = s.nGTS; d["nDYN"] = s.nDYN;
    d["off_isl"] = s.off_isl; d["off_gts"] = s.off_gts; d["off_dyn"] = s.off_dyn;
    d["off_damp"] = s.off_damp; d["issq"] = s.issq;
    d["A_rp"] = py::array_t<int32_t>({(py::ssize_t)s.A_rp.size()}, s.A_rp.data());
    d["A_ci"] = py::array_t<int32_t>({(py::ssize_t)s.A_ci.size()}, s.A_ci.data());
    d["bptrs"] = py::array_t<int64_t>({(py::ssize_t)s.bptrs.size()}, s.bptrs.data());
    d["binds"] = py::array_t<int64_t>({(py::ssize_t)s.binds.size()}, s.binds.data());
    return d;
}

// QOE retraction（角度 wrap、i/e/a 合法化），镜像 py `_retract_qoe_np`。
static void retract_qoe(std::vector<double> &X, int nnode, int n) {
    const double PI = 3.14159265358979323846;
    for (int r = 0; r < nnode * n; ++r) {
        double *x = &X[(size_t)r * 6];
        x[1] = std::fmod(x[1], 2 * PI); if (x[1] < 0) x[1] += 2 * PI;
        x[5] = std::fmod(x[5], 2 * PI); if (x[5] < 0) x[5] += 2 * PI;
        x[4] = std::min(std::max(x[4], 1e-6), PI - 1e-6);
        const double e = std::hypot(x[2], x[3]);
        if (e > 0.999) { x[2] *= 0.999 / e; x[3] *= 0.999 / e; }
        x[0] = std::min(std::max(x[0], 1e5), 1e8);
    }
}

// 稳定白化因子：Q(6×6/星) → q 使 qᵀq = Q⁻¹。
// 对 chol(Q) 做下三角回代（q = L⁻¹），**不显式求 Q⁻¹**（Q⁻¹ 病态会放大到 1e37 级，见 doc/experiment.md l1233）。
static std::vector<double> qinv_sqrt(const std::vector<double> &Q) {
    const int n = (int)(Q.size() / 36);
    std::vector<double> out((size_t)n * 36, 0.0);
    for (int i = 0; i < n; ++i) {
        Eigen::Matrix<double, 6, 6> Qm;
        for (int a = 0; a < 6; ++a)
            for (int b = 0; b < 6; ++b) Qm(a, b) = Q[(size_t)i * 36 + a * 6 + b];
        Qm += Eigen::Matrix<double, 6, 6>::Identity() * (Qm.trace() * 1e-12 + 1e-300);  // 同 proc_Q_qoe 口径正则
        Eigen::LLT<Eigen::Matrix<double, 6, 6>> llt(Qm);
        Eigen::Matrix<double, 6, 6> L = llt.matrixL();
        Eigen::Matrix<double, 6, 6> Linv =
            L.triangularView<Eigen::Lower>().solve(Eigen::Matrix<double, 6, 6>::Identity());
        for (int a = 0; a < 6; ++a)
            for (int b = 0; b < 6; ++b) out[(size_t)i * 36 + a * 6 + b] = Linv(a, b);
    }
    return out;
}

// ===================== FlsWindow：固定滞后平滑（L 历元窗，全 C++） =====================
// 符号（与你 doc/batchSmoother.md 的体系一致）：
//   x_k = X[k]   窗口第 k 个历元的状态（QOE，6/星）；L = 窗历元数；dt = Δt；
//   Φ = Phi      x_k → x_{k+1} 的 STM（装配时由 qoe_core 的 gveStmStep 现算，3/8-RK4 变分）；
//   Q 的每星白化因子 Q_isqrt = Q^{-1/2}（GVE 口径、δu=1e-7 RTN）；
//   Λ0⁻ = Lambda0  保留历元上的边缘化先验信息；z_prior 先验中心；
//   Lp   = Λ0 的白化因子（行式 Λ0 = LpᵀLp，即 Lp = chol(Λ0) 的行因子）；
//   P_marg  保留历元每星 6×6 Schur 边缘协方差。
// 本轮实现：结构 + 观测/状态上传 + 装配（ISL/GTS/dyn/先验），A_val/b 与 py `_lift_assemble` 对拍。
class FlsWindow {
  public:
    FlsWindow(int n, int L, int E, int G, double dt, double sigma_isl, double sigma_gts,
              std::vector<int> isl_i, std::vector<int> isl_j, std::vector<int> gts,
              std::vector<double> Q, double damp, bool prior)
        : n_(n), L_(L), E_(E), G_(G), dt_(dt), sigma_isl_(sigma_isl), sigma_gts_(sigma_gts),
          isl_i_(std::move(isl_i)), isl_j_(std::move(isl_j)), gts_(std::move(gts)),
          prior_(prior) {
        S_ = fls_struct_impl(n_, L_, isl_i_, isl_j_, gts_, damp, prior_);
        // 观测表按历元平铺（与 py `_lift_struct` 的 isl_i/isl_jj/gts_a 布局一致：长度 L*E / L*G）
        std::vector<int> isl_i_t, isl_j_t, gts_t;
        for (int k = 0; k < L_; ++k) {
            isl_i_t.insert(isl_i_t.end(), isl_i_.begin(), isl_i_.end());
            isl_j_t.insert(isl_j_t.end(), isl_j_.begin(), isl_j_.end());
            gts_t.insert(gts_t.end(), gts_.begin(), gts_.end());
        }
        d_isl_i_ = upi(isl_i_t); d_isl_jj_ = upi(isl_j_t); d_gts_a_ = upi(gts_t);
        d_qingt_ = upd(qinv_sqrt(Q));                       // 白化 qᵀq = Q⁻¹（稳定：chol+三角回代）
        d_isl_D_ = mal(E_ * L_); d_isl_W_ = mal(E_ * L_);
        d_gts_D_ = mal((size_t)G_ * L_); d_gts_W_ = mal((size_t)G_ * L_);
        d_gts_Q_ = mal((size_t)G_ * L_ * 3); d_ctr_ = mal((size_t)n_ * 18);
        d_Xa_ = mal((size_t)L_ * n_ * 6); d_rv_ = mal((size_t)L_ * n_ * 6);
        d_Ap_ = mal((size_t)L_ * n_ * 18);
        const size_t nDyn_alloc = (size_t)std::max(L_ - 1, 1);   // L=1 交接也需 1 个区间的 φ/Φ
        d_end_ = mal(nDyn_alloc * n_ * 6); d_phi_ = mal(nDyn_alloc * n_ * 36);
        d_res_ = mal(1);
        d_rvGeo_ = mal((size_t)L_ * n_ * 6);
        d_ApGeo_ = mal((size_t)L_ * n_ * 18);
        auto optsD = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA);
        auto optsI = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
        A_rp_t_ = torch::tensor(S_.A_rp, torch::TensorOptions().dtype(torch::kInt64)).to(optsI);
        A_ci_t_ = torch::tensor(S_.A_ci, torch::TensorOptions().dtype(torch::kInt64)).to(optsI);
        A_val_ = torch::zeros({S_.nnz}, optsD);
        b_ = torch::zeros({S_.nrows}, optsD);
        X_ = torch::zeros({L_, n_, 6}, optsD);              // x_k 队列
        rv_ = torch::zeros({L_, n_, 6}, optsD);
        Ap_ = torch::zeros({L_, n_, 18}, optsD);
        Lambda0_ = torch::zeros({n_, 6, 6}, optsD);         // Λ0⁻
        Lp_ = torch::zeros({n_, 6, 6}, optsD);              // Λ0 白化因子
        z_prior_ = torch::zeros({n_, 6}, optsD);
        hX_.assign((size_t)L_ * n_ * 6, 0.0);
        hRv_.assign((size_t)L_ * n_ * 6, 0.0);
        hAp_.assign((size_t)L_ * n_ * 18, 0.0);
        qoejopt::WinCtx &c = wc_;
        c.n = n_; c.L = L_; c.E = E_; c.G = G_;
        c.ncol = S_.ncol; c.nnz = S_.nnz; c.nrows = S_.nrows;
        c.off_isl = S_.off_isl; c.off_gts = S_.off_gts; c.off_dyn = S_.off_dyn; c.off_damp = S_.off_damp;
        c.dt = dt_; c.s_isl = 1.0 / sigma_isl_; c.s_gts = 1.0 / sigma_gts_; c.issq = S_.issq;
        c.prior = prior_ ? 1 : 0;
        c.q_perstar = 1;
        c.d_isl_i = d_isl_i_; c.d_isl_jj = d_isl_jj_; c.d_gts_a = d_gts_a_;
        c.d_isl_D = d_isl_D_; c.d_isl_W = d_isl_W_;
        c.d_gts_D = d_gts_D_; c.d_gts_W = d_gts_W_; c.d_gts_Q = d_gts_Q_;
        c.d_ctr = d_ctr_; c.d_qingt = d_qingt_;
        c.d_X = d_Xa_; c.d_rv = d_rv_; c.d_Ap = d_Ap_;
        c.d_end = d_end_; c.d_phi = d_phi_; c.d_res = d_res_;
        // FLS 自算几何 / 求解所需
        Pm_host_.assign((size_t)n_ * 36, 0.0);
        Lp_host_.assign((size_t)n_ * 36, 0.0);
        zp_host_.assign((size_t)n_ * 6, 0.0);
        auto optsIcpu = torch::TensorOptions().dtype(torch::kInt64);
        auto paramSize = torch::full({(int64_t)L_ * n_}, 6, optsIcpu);
        auto bptr_t = torch::from_blob(S_.bptrs.data(), {(py::ssize_t)S_.bptrs.size()}, optsIcpu).clone();
        auto bind_t = torch::from_blob(S_.binds.data(), {(py::ssize_t)S_.binds.size()}, optsIcpu).clone();
        sym_ = std::make_unique<SymbolicDecomposition>(paramSize, bptr_t, bind_t, "cuda");
        // ---- P1'（逐星 Schur）：逐星行表（device int64，一次构造）----
        {
            std::vector<int64_t> p00{0}, s00, c00, p11{0}, s11, c11, ps1, ps0;
            for (int i = 0; i < n_; ++i) {
                s00.push_back((int64_t)S_.off_damp + (int64_t)i * 36);        // prior@0
                c00.push_back((int64_t)i * 6);
                for (int e = 0; e < E_; ++e) {                                // ISL@0（关联到 i 的一侧）
                    if (isl_i_[e] == i) {
                        s00.push_back((int64_t)S_.off_isl + (int64_t)e * 12);
                        c00.push_back((int64_t)i * 6);
                    }
                    if (isl_j_[e] == i) {
                        s00.push_back((int64_t)S_.off_isl + (int64_t)e * 12 + 6);
                        c00.push_back((int64_t)i * 6);
                    }
                }
                for (int g = 0; g < G_; ++g)                                  // GTS@0
                    if (gts_[g] == i) {
                        s00.push_back((int64_t)S_.off_gts + (int64_t)g * 6);
                        c00.push_back((int64_t)i * 6);
                    }
                s00.push_back((int64_t)S_.off_dyn + (int64_t)i * 72);         // dyn(0→1) 的节点 0（每行 12 非零）
                c00.push_back((int64_t)i * 6);
                p00.push_back((int64_t)s00.size());
                for (int e = 0; e < E_; ++e) {                                // ISL@1
                    if (isl_i_[e] == i) {
                        s11.push_back((int64_t)S_.off_isl + ((int64_t)E_ + e) * 12);
                        c11.push_back((int64_t)(n_ + i) * 6);
                    }
                    if (isl_j_[e] == i) {
                        s11.push_back((int64_t)S_.off_isl + ((int64_t)E_ + e) * 12 + 6);
                        c11.push_back((int64_t)(n_ + i) * 6);
                    }
                }
                for (int g = 0; g < G_; ++g)                                  // GTS@1
                    if (gts_[g] == i) {
                        s11.push_back((int64_t)S_.off_gts + ((int64_t)G_ + g) * 6);
                        c11.push_back((int64_t)(n_ + i) * 6);
                    }
                s11.push_back((int64_t)S_.off_dyn + (int64_t)i * 72 + 6);     // dyn(0→1) 的节点 1
                c11.push_back((int64_t)(n_ + i) * 6);
                // 说明：曾把 dyn(1→2)@1 当作节点 1 的信息（L≥3），实为"把 frame 2 的动力学当硬观测"的
                // 不一致处理：先验过度自信 → 长弧上 gauge 漂移（L=4 127 帧 Kabsch 1800+ m）。改为保守的
                // 成对（0,1）边缘：丢弃未来帧信息，先验略松但一致（L=2 无此项，原本即精确）。
                p11.push_back((int64_t)s11.size());
                for (int k = 0; k < 6; ++k) {                                 // Λ10 配对（dyn(0→1) 的 6 行）
                    ps1.push_back((int64_t)S_.off_dyn + (int64_t)(i * 6 + k) * 12 + 6);
                    ps0.push_back((int64_t)S_.off_dyn + (int64_t)(i * 6 + k) * 12);
                }
            }
            d00_p_ = torch::tensor(p00, optsI);
            d00_s_ = torch::tensor(s00, optsI);
            d00_c_ = torch::tensor(c00, optsI);
            d11_p_ = torch::tensor(p11, optsI);
            d11_s_ = torch::tensor(s11, optsI);
            d11_c_ = torch::tensor(c11, optsI);
            pr_s1_ = torch::tensor(ps1, optsI);
            pr_s0_ = torch::tensor(ps0, optsI);
            d_PmMarg_ = mal((size_t)n_ * 36);
        }
        {
            std::vector<int64_t> rnz;
            rnz.reserve(S_.nnz);
            for (int r = 0; r < S_.nrows; ++r)
                for (int j = S_.A_rp[r]; j < S_.A_rp[r + 1]; ++j) rnz.push_back(r);
            row_nz_ = torch::tensor(rnz, optsI);
        }
        // ---- 逐帧消去（sweep，l430-433 选项 2）：单帧结构/符号 + 单帧缓冲 + 逐星相对行表 ----
        {
            F_ = fls_frame_impl(n_, isl_i_, isl_j_, gts_);
            auto optsIcpu2 = torch::TensorOptions().dtype(torch::kInt64);
            auto bpF = torch::from_blob(F_.bptrs.data(), {(py::ssize_t)F_.bptrs.size()},
                                        optsIcpu2).clone();
            auto biF = torch::from_blob(F_.binds.data(), {(py::ssize_t)F_.binds.size()},
                                        optsIcpu2).clone();
            auto psF = torch::full({(int64_t)n_}, 6, optsIcpu2);
            symF_ = std::make_unique<SymbolicDecomposition>(psF, bpF, biF, "cuda");
            ArpF_t_ = torch::tensor(F_.A_rp, torch::TensorOptions().dtype(torch::kInt64)).to(optsI);
            AciF_t_ = torch::tensor(F_.A_ci, torch::TensorOptions().dtype(torch::kInt64)).to(optsI);
            std::vector<int64_t> rnz;
            rnz.reserve(F_.nnz);
            for (int r = 0; r < F_.nrows; ++r)
                for (int j = F_.A_rp[r]; j < F_.A_rp[r + 1]; ++j) rnz.push_back(r);
            rnzF_t_ = torch::tensor(rnz, optsI);
            Af_t_ = torch::zeros({F_.nnz}, optsD);
            bf_t_ = torch::zeros({F_.nrows}, optsD);
            eta_t_ = torch::zeros({(int64_t)n_ * 6}, optsD);
            yF_t_ = torch::zeros({(int64_t)L_ * n_ * 6}, optsD);
            dlt_t_ = torch::zeros({(int64_t)n_ * 36}, optsD);
            gam_t_ = torch::zeros({(int64_t)n_ * 6}, optsD);
            tmp_t_ = torch::zeros({(int64_t)n_ * 6}, optsD);
            {   // 首窗弱弹簧先验（Lp=1e-8·I）与中心缓冲
                auto eye = torch::eye(6, torch::TensorOptions().dtype(torch::kFloat64)).to(optsD);
                weak_Lp_ = (eye * 1e-8).repeat({(int64_t)n_, 1, 1}).reshape({(int64_t)n_ * 36})
                               .contiguous();
                zw_t_ = torch::zeros({(int64_t)n_ * 6}, optsD);
            }
            {   // 消去修正图案：块对角（每行 6 项；add_M 只取 c<=r）
                std::vector<int64_t> rpC{0}, ciC;
                for (int r = 0; r < n_ * 6; ++r) {
                    const int i0 = r / 6;
                    for (int c = 0; c < 6; ++c) ciC.push_back((int64_t)i0 * 6 + c);
                    rpC.push_back((int64_t)ciC.size());
                }
                rpC_t_ = torch::tensor(rpC, optsI);
                ciC_t_ = torch::tensor(ciC, optsI);
            }
            {   // 逐星相对行表（帧内基址；列 = 帧内局部基）
                std::vector<int64_t> fp{0}, fs, fc, gp{0}, gs, gc;
                for (int i = 0; i < n_; ++i) {
                    for (int e = 0; e < E_; ++e) {
                        if (isl_i_[e] == i) {
                            fs.push_back((int64_t)e * 12); fc.push_back((int64_t)i * 6);
                        }
                        if (isl_j_[e] == i) {
                            fs.push_back((int64_t)e * 12 + 6); fc.push_back((int64_t)i * 6);
                        }
                    }
                    fp.push_back((int64_t)fs.size());
                    for (int g = 0; g < G_; ++g)
                        if (gts_[g] == i) {
                            gs.push_back((int64_t)g * 6); gc.push_back((int64_t)i * 6);
                        }
                    gp.push_back((int64_t)gs.size());
                }
                fis_p_ = torch::tensor(fp, optsI); fis_s_ = torch::tensor(fs, optsI);
                fis_c_ = torch::tensor(fc, optsI);
                fgt_p_ = torch::tensor(gp, optsI); fgt_s_ = torch::tensor(gs, optsI);
                fgt_c_ = torch::tensor(gc, optsI);
            }
            // 逐帧消去（doc l430-433 选项 2）为默认求解路径；FLS_SWEEP=0 可退回全窗联合因子。
            const char *sw = std::getenv("FLS_SWEEP");
            sweep_ = !(sw && std::atoi(sw) == 0);
        }
    }
    ~FlsWindow() {
        cudaFree(d_isl_i_); cudaFree(d_isl_jj_); cudaFree(d_gts_a_); cudaFree(d_qingt_);
        cudaFree(d_isl_D_); cudaFree(d_isl_W_); cudaFree(d_gts_D_); cudaFree(d_gts_W_);
        cudaFree(d_gts_Q_); cudaFree(d_ctr_);
        cudaFree(d_Xa_); cudaFree(d_rv_); cudaFree(d_Ap_);
        cudaFree(d_end_); cudaFree(d_phi_); cudaFree(d_res_);
        cudaFree(d_rvGeo_); cudaFree(d_ApGeo_); cudaFree(d_PmMarg_);
    }

    // 每步观测（**C++ 原生**）：ISL 距离/权重 (L*E)；GTS 距离/权重 (L*G)、单位向量 (L*G,3)；ctr (n,6,3)。
    void set_obs_raw(const double *D_isl, const double *W_isl, const double *D_gts,
                     const double *W_gts, const double *Q_gts, const double *ctr) {
        cudaMemcpy(d_isl_D_, D_isl, (size_t)E_ * L_ * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_isl_W_, W_isl, (size_t)E_ * L_ * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_gts_D_, D_gts, (size_t)G_ * L_ * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_gts_W_, W_gts, (size_t)G_ * L_ * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_gts_Q_, Q_gts, (size_t)G_ * L_ * 3 * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_ctr_, ctr, (size_t)n_ * 18 * sizeof(double), cudaMemcpyHostToDevice);
    }
    // py 薄包装（只做尺寸检查 + 取裸指针）
    void set_obs(py::array_t<double, py::array::c_style | py::array::forcecast> D_isl,
                 py::array_t<double, py::array::c_style | py::array::forcecast> W_isl,
                 py::array_t<double, py::array::c_style | py::array::forcecast> D_gts,
                 py::array_t<double, py::array::c_style | py::array::forcecast> W_gts,
                 py::array_t<double, py::array::c_style | py::array::forcecast> Q_gts,
                 py::array_t<double, py::array::c_style | py::array::forcecast> ctr) {
        set_obs_raw((double *)D_isl.request().ptr, (double *)W_isl.request().ptr,
                    (double *)D_gts.request().ptr, (double *)W_gts.request().ptr,
                    (double *)Q_gts.request().ptr, (double *)ctr.request().ptr);
    }

    // 上传窗口状态 x_k（(L,n,6)）；rv 与 ∂p/∂oe 由 C++ 内部 `geo_all()` 现算（无需外部传）。
    void set_X_raw(const double *X) {
        std::copy(X, X + (size_t)L_ * n_ * 6, hX_.begin());
        geo_all();
    }
    void set_X(py::array_t<double, py::array::c_style | py::array::forcecast> X) {
        auto cx = X.request();
        if ((size_t)cx.size != (size_t)L_ * n_ * 6)
            throw std::runtime_error("FlsWindow.set_X: X 需 (L,n,6)");
        set_X_raw((double *)cx.ptr);
    }

    // 设置边缘化先验 Λ0（白化因子 Lp (n,6,6)）与先验中心 z_prior (n,6)。
    void set_prior_raw(const double *Lp, const double *z_prior) {
        cudaMemcpy(Lp_.data_ptr<double>(), Lp, (size_t)n_ * 36 * sizeof(double),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(z_prior_.data_ptr<double>(), z_prior, (size_t)n_ * 6 * sizeof(double),
                   cudaMemcpyHostToDevice);
        has_prior_ = true;
    }
    void set_prior(py::array_t<double, py::array::c_style | py::array::forcecast> Lp,
                   py::array_t<double, py::array::c_style | py::array::forcecast> z_prior) {
        set_prior_raw((double *)Lp.request().ptr, (double *)z_prior.request().ptr);
    }

    // 装配 A_val/b（ISL/GTS/dyn + 先验行）；返回 ISL 残差平方和。
    double assemble() {
        double res2 = qoejopt::winAssembleDevice(wc_, hX_.data(), hRv_.data(), hAp_.data(),
                                                 A_val_.data_ptr<double>(), b_.data_ptr<double>());
        if (prior_ && has_prior_)
            qoejopt::winPriorAsmDevice(wc_.d_X, Lp_.data_ptr<double>(), z_prior_.data_ptr<double>(),
                                       n_, 0, S_.off_damp, S_.nISL + S_.nGTS + S_.nDYN,
                                       A_val_.data_ptr<double>(), b_.data_ptr<double>());
        else if (prior_) {
            // 首窗（无边缘先验）：写"弱弹簧"先验行（Lp=1e-8·I、中心=当前窗首），避免零行块使
            // Baspacho 因子出现零主元 → 首步解 NaN（随后被线搜索静默拒绝）。
            cudaMemcpy(wc_.d_X, hX_.data(), (size_t)L_ * n_ * 6 * sizeof(double),
                       cudaMemcpyHostToDevice);
            cudaMemcpy(zw_t_.data_ptr<double>(), hX_.data(), (size_t)n_ * 6 * sizeof(double),
                       cudaMemcpyHostToDevice);
            qoejopt::winPriorAsmDevice(wc_.d_X, weak_Lp_.data_ptr<double>(),
                                       zw_t_.data_ptr<double>(), n_, 0, S_.off_damp,
                                       S_.nISL + S_.nGTS + S_.nDYN, A_val_.data_ptr<double>(),
                                       b_.data_ptr<double>());
        }
        return res2;
    }
    // 仅算 ISL res2（GN 线搜索用）：刷新几何并送 rv 到装配区；不做 A/b 装配、不算 dyn STM。
    double res2_only() {
        geo_all();
        cudaMemcpy(wc_.d_rv, hRv_.data(), (size_t)L_ * n_ * 6 * sizeof(double),
                   cudaMemcpyHostToDevice);
        return qoejopt::winRes2Only(wc_);
    }

    torch::Tensor A_val() const { return A_val_.cpu(); }
    torch::Tensor b() const { return b_.cpu(); }
    int ncol() const { return S_.ncol; }
    int nrows() const { return S_.nrows; }
    int nnz() const { return S_.nnz; }

    // 纯 C++ 递推一步（不依赖 py）：x_new_in 为空时由上一历元 x_{L-1} 经 GVE 前推 1 帧（dt）。
    // 队列左移 → 全窗 rv/Ap → 装配 + GN（列均衡/Baspacho/线搜索）→
    // Schur 边缘保留历元（节点 1）→ P_marg → Λ0 白化 Lp、先验中心 z_prior。
    void step(const double *x_new_in, int steps) {
        using clk = std::chrono::steady_clock;
        auto _now = [] { return std::chrono::duration<double>(clk::now().time_since_epoch()).count(); };
        double t0 = _now(), t1 = 0, t_pred = 0, t_geo = 0, t_asm = 0, t_fac = 0, t_solve = 0,
               t_ls = 0, t_schur = 0, t_gn = 0;
        dbg_mtm = dbg_fac = dbg_solve = 0;      // 每步归因（GN 内：列均衡/因子/求解）
        qoejopt::winResetTiming();              // 装配内归因（dyn STM / 装配核 / res2）
        // ⓪ 新历元 + 队列左移
        std::vector<double> xn((size_t)n_ * 6, 0.0);
        if (x_new_in) {
            std::copy(x_new_in, x_new_in + (size_t)n_ * 6, xn.begin());
        } else {
            // 复用既有 C++ GVE 批量递推（N 星并行）：x_{L-1} → x_L
            std::vector<Vector6d> oe0(n_), oef;
            for (int i = 0; i < n_; ++i)
                for (int c = 0; c < 6; ++c) oe0[i](c) = hX_[((size_t)(L_ - 1) * n_ + i) * 6 + c];
            kep3::gveStepNoeBatch(oe0, 0.0, dt_, 16, 1.0, oef);
            for (int i = 0; i < n_; ++i)
                for (int c = 0; c < 6; ++c) xn[(size_t)i * 6 + c] = oef[i](c);
        }
        t1 = _now(); t_pred = t1 - t0;
        for (int k = 0; k + 1 < L_; ++k)
            std::copy(hX_.begin() + (size_t)(k + 1) * n_ * 6, hX_.begin() + (size_t)(k + 2) * n_ * 6,
                      hX_.begin() + (size_t)k * n_ * 6);
        std::copy(xn.begin(), xn.end(), hX_.end() - (size_t)n_ * 6);
        t1 = _now(); geo_all(); t_geo += _now() - t1;
        // ① GN（线搜索用 assemble 返回的 ISL res2）
        t1 = _now(); double res2 = assemble(); t_asm += _now() - t1;
        t1 = _now();
        int gn_it = 0;   // GN 实际迭代数（诊断）
        for (int it = 0; it < std::max(1, steps); ++it) {
            ++gn_it;
            double tm = _now();
            auto mtm = scaled_AtA_Atb();
            dbg_mtm += _now() - tm;
            torch::Tensor dcpu;
            if (sweep_) {   // 信息形式逐帧消去（l430-433 选项 2）：每帧 6N 因子 + 逐星 Schur 修正 + 回代
                const int64_t nc = (int64_t)n_ * 6;
                tm = _now();
                std::vector<NumericDecomposition> nds;
                nds.reserve(L_);
                for (int m = 0; m < L_; ++m) {
                    double _ta = _now(), _tb;
                    gather_frame_(m, mtm.first);
                    _tb = _now(); double t_gath = _tb - _ta; _ta = _tb;
                    eta_t_.zero_();
                    eta_t_.index_add_(0, AciF_t_, Af_t_ * bf_t_.index_select(0, rnzF_t_));
                    if (m > 0) eta_t_.sub_(gam_t_.view({nc}));  // −γ_{m-1}
                    _tb = _now(); double t_eta = _tb - _ta; _ta = _tb;
                    auto nd = symF_->createNumericDecomposition(1);
                    _tb = _now(); double t_cre = _tb - _ta; _ta = _tb;
                    nd.add_MtM(Af_t_.view({1, (int64_t)F_.nnz}), ArpF_t_, AciF_t_);
                    _tb = _now(); double t_mtm = _tb - _ta; _ta = _tb;
                    if (m > 0)
                        nd.add_M(dlt_t_.view({1, (int64_t)n_ * 36}), rpC_t_, ciC_t_);  // −Δ_{m-1}
                    _tb = _now(); double t_add = _tb - _ta; _ta = _tb;
                    nd.factor();
                    _tb = _now(); double t_fac = _tb - _ta; _ta = _tb;
                    auto yv = eta_t_.view({1, nc}).clone();
                    nd.solve(yv);
                    _tb = _now(); double t_slv = _tb - _ta; _ta = _tb;
                    yF_t_.narrow(0, (int64_t)m * nc, nc).copy_(yv.view({nc}));
                    if (m < L_ - 1)
                        qoejopt::flsFrameSweepDevice(
                            mtm.first.data_ptr<double>(), mtm_scl_.data_ptr<double>(), n_, L_, E_,
                            G_, m, S_.off_isl, S_.off_gts, S_.off_dyn, S_.off_damp,
                            fis_p_.data_ptr<int64_t>(), fis_s_.data_ptr<int64_t>(),
                            fis_c_.data_ptr<int64_t>(), fgt_p_.data_ptr<int64_t>(),
                            fgt_s_.data_ptr<int64_t>(), fgt_c_.data_ptr<int64_t>(),
                            m > 0 ? dlt_t_.data_ptr<double>() : nullptr,
                            yF_t_.data_ptr<double>() + (int64_t)m * nc,
                            dlt_t_.data_ptr<double>(), gam_t_.data_ptr<double>());
                    _tb = _now();
                    if (getenv("FLS_SWEEPDBG"))
                        fprintf(stderr,
                                "[sweep] m=%d gather=%.2f eta=%.2f create=%.2f mtm=%.2f "
                                "addM=%.2f factor=%.2f solve=%.2f kern=%.2f ms\n",
                                m, 1e3 * t_gath, 1e3 * t_eta, 1e3 * t_cre, 1e3 * t_mtm,
                                1e3 * t_add, 1e3 * t_fac, 1e3 * t_slv, 1e3 * (_tb - _ta));
                    nds.push_back(std::move(nd));
                }
                dbg_fac += _now() - tm;
                tm = _now();
                for (int m = L_ - 2; m >= 0; --m) {   // 回代 δ_m = y_m − S_m⁻¹(C_m δ_{m+1})
                    qoejopt::flsCouplingDevice(
                        mtm.first.data_ptr<double>(), mtm_scl_.data_ptr<double>(), n_, m,
                        S_.off_dyn, yF_t_.data_ptr<double>() + (int64_t)(m + 1) * nc,
                        tmp_t_.data_ptr<double>());
                    auto zz = tmp_t_.view({1, nc}).clone();   // z = S_m⁻¹(C_m δ_{m+1})
                    nds[m].solve(zz);
                    yF_t_.narrow(0, (int64_t)m * nc, nc).sub_(zz.view({nc}));
                }
                dbg_solve += _now() - tm;
                dcpu = (yF_t_.view({L_, nc}) * mtm_scl_.view({L_, nc})).view({-1}).cpu();
            } else {
                tm = _now();
                auto nd = sym_->createNumericDecomposition(1);
                nd.add_MtM(mtm.first.reshape({1, S_.nnz}), A_rp_t_, A_ci_t_);
                nd.factor();
                dbg_fac += _now() - tm;
                tm = _now();
                auto xv = mtm.second.reshape({1, S_.ncol}).clone();
                nd.solve(xv);
                dbg_solve += _now() - tm;
                dcpu = (xv.reshape({S_.ncol}) * mtm_scl_).cpu();
            }
            auto dc = dcpu.accessor<double, 1>();
            const auto X0 = hX_;
            double alpha = 1.0;
            bool acc = false;
            double dcmax = 0.0, dcn2 = 0.0;
            int64_t dcnf = 0;
            for (int64_t q = 0; q < S_.ncol; ++q) {
                if (!std::isfinite(dc[q])) { ++dcnf; continue; }
                dcmax = std::max(dcmax, std::fabs(dc[q]));
                dcn2 += dc[q] * dc[q];
            }
            if (getenv("FLS_LSDBG") && gn_it <= 2) {
                double scmin = 1e300, scmax = 0.0;
                {
                    auto sc_cpu = mtm_scl_.to(torch::kCPU);
                    auto scv = sc_cpu.accessor<double, 1>();
                    for (int64_t q = 0; q < S_.ncol; ++q) {
                        scmin = std::min(scmin, scv[q]);
                        scmax = std::max(scmax, scv[q]);
                    }
                }
                auto a_cpu = A_val_.to(torch::kCPU);
                auto av = a_cpu.accessor<double, 1>();
                int64_t anf = 0;
                double amax = 0.0;
                for (int64_t q = 0; q < S_.nnz; ++q) {
                    if (!std::isfinite(av[q])) { ++anf; continue; }
                    amax = std::max(amax, std::fabs(av[q]));
                }
                auto b_cpu = b_.to(torch::kCPU);
                auto bv = b_cpu.accessor<double, 1>();
                int64_t bnf = 0;
                double bmax = 0.0;
                for (int64_t q = 0; q < S_.nrows; ++q) {
                    if (!std::isfinite(bv[q])) { ++bnf; continue; }
                    bmax = std::max(bmax, std::fabs(bv[q]));
                }
                fprintf(stderr,
                        "[fls/dc] it=%d |dc|max=%.3e |dc|2=%.3e nf=%lld sc=[%.2e,%.2e] res2=%.6e "
                        "A(nf=%lld,max=%.2e) b(nf=%lld,max=%.2e)\n",
                        gn_it, dcmax, std::sqrt(dcn2), (long long)dcnf, scmin, scmax, res2,
                        (long long)anf, amax, (long long)bnf, bmax);
            }
            for (int ls = 0; ls < 4; ++ls) {
                for (size_t i = 0; i < hX_.size(); ++i) hX_[i] = X0[i] + alpha * dc[(int64_t)i];
                retract_qoe(hX_, L_, n_);
                t1 = _now(); const double rn = res2_only(); t_asm += _now() - t1;
                if (getenv("FLS_LSDBG") && gn_it <= 2)
                    fprintf(stderr, "[fls/ls] it=%d ls=%d alpha=%.4g res2=%.8e rn=%.8e |dc|max=%.3e\n",
                            gn_it, ls, alpha, res2, rn, dcmax);
                if (std::isfinite(rn) && rn < res2) {
                    res2 = rn; acc = true;
                    t1 = _now(); assemble(); t_asm += _now() - t1;   // 接受后完整装配（供下一轮 GN/边缘化）
                    break;
                }
                alpha *= 0.5;
            }
            if (!acc) { hX_ = X0; geo_all(); break; }
        }
        t_gn = _now() - t1;
        res2_ = res2;
        // ② 先验交接：L≥2 → 节点 1 的每星 6×6 Schur 边缘；L=1 → EKF 式 P⁻=ΦP⁺Φᵀ+Q 白化。
        t1 = _now();
        if (L_ == 1) {
            double tm = _now();
            qoejopt::gveStmDeviceGrp(wc_.d_X, n_, dt_, 1.0, d_end_, d_phi_);   // 单步同源 Φ
            qoejopt::flsEkfHandoffDevice(A_val_.data_ptr<double>(), mtm_scl_.data_ptr<double>(),
                                         d00_p_.data_ptr<int64_t>(), d00_s_.data_ptr<int64_t>(),
                                         d00_c_.data_ptr<int64_t>(), d_phi_, d_qingt_, n_,
                                         Lp_.data_ptr<double>(), d_PmMarg_);
            double t2 = _now(); dbg_marg = t2 - tm;
            cudaMemcpy(Lp_host_.data(), Lp_.data_ptr<double>(), (size_t)n_ * 36 * sizeof(double),
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(Pm_host_.data(), d_PmMarg_, (size_t)n_ * 36 * sizeof(double),
                       cudaMemcpyDeviceToHost);
            double t3 = _now(); dbg_d2h = t3 - t2;
            {
                std::vector<Vector6d> oe0(n_), oef;
                for (int i = 0; i < n_; ++i)
                    for (int c = 0; c < 6; ++c) oe0[i](c) = hX_[(size_t)i * 6 + c];
                kep3::gveStepNoeBatch(oe0, 0.0, dt_, 16, 1.0, oef);
                for (int i = 0; i < n_; ++i)
                    for (int c = 0; c < 6; ++c) zp_host_[(size_t)i * 6 + c] = oef[i](c);
            }
            cudaMemcpy(z_prior_.data_ptr<double>(), zp_host_.data(),
                       (size_t)n_ * 6 * sizeof(double), cudaMemcpyHostToDevice);
            dbg_rest = _now() - t3;
            t_fac += _now() - t1;
            t1 = _now();
            has_prior_ = true;
        } else {
            // P1'（逐星 Schur，M1b：Λ00⁻¹≈块对角）：**每星一线程的专用核**（gather→Λ 组装→两次 6×6 逆
            // →对称 Jacobi→Lp/Pmarg），全程列均衡空间，末尾换回物理空间。行表 ctor 里预存。
            double tm = _now();
            qoejopt::flsMargDevice(A_val_.data_ptr<double>(), mtm_scl_.data_ptr<double>(),
                                   d00_p_.data_ptr<int64_t>(), d00_s_.data_ptr<int64_t>(),
                                   d00_c_.data_ptr<int64_t>(),
                                   d11_p_.data_ptr<int64_t>(), d11_s_.data_ptr<int64_t>(),
                                   d11_c_.data_ptr<int64_t>(),
                                   pr_s1_.data_ptr<int64_t>(), pr_s0_.data_ptr<int64_t>(),
                                   n_, Lp_.data_ptr<double>(), d_PmMarg_);
            double t2 = _now(); dbg_marg = t2 - tm;
            cudaMemcpy(Lp_host_.data(), Lp_.data_ptr<double>(), (size_t)n_ * 36 * sizeof(double),
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(Pm_host_.data(), d_PmMarg_, (size_t)n_ * 36 * sizeof(double),
                       cudaMemcpyDeviceToHost);
            double t3 = _now(); dbg_d2h = t3 - t2;
            std::copy(hX_.begin() + (size_t)n_ * 6, hX_.begin() + (size_t)2 * n_ * 6,
                      zp_host_.begin());
            cudaMemcpy(z_prior_.data_ptr<double>(), hX_.data() + (size_t)n_ * 6,
                       (size_t)n_ * 6 * sizeof(double), cudaMemcpyHostToDevice);
            dbg_rest = _now() - t3;
            t_fac += _now() - t1;
            t1 = _now();
            has_prior_ = true;
        }
        t_ls += _now() - t1;
        t_schur = _now() - t0 - t_pred - t_geo - t_asm - t_gn;
        dbg_pred = t_pred; dbg_geo = t_geo; dbg_asm = t_asm; dbg_gn = t_gn;
        dbg_total = _now() - t0;
        static int _dbg_n = 0;
        if (getenv("FLS_STEPDBG") && _dbg_n++ < 3) {
            auto wt = qoejopt::winGetTiming();
            fprintf(stderr, "[fls/step/t] pred=%.3f geo=%.3f asm=%.3f gn=%.3f | GN 均衡=%.3f "
                    "因子=%.3f 求解=%.3f | 核=%.3f D2H=%.3f 余=%.3f | asm内 dyn=%.3f 核=%.3f "
                    "res2=%.3f | it=%d sweep=%d | total=%.3f s\n",
                    t_pred, t_geo, t_asm, t_gn, dbg_mtm, dbg_fac, dbg_solve,
                    dbg_marg, dbg_d2h, dbg_rest, wt[0], wt[1], wt[2], gn_it, sweep_ ? 1 : 0,
                    _now() - t0);
        }
    }
    void step(int steps) { step(nullptr, steps); }
    // py 薄 wrapper：x_new=None 时由 GVE 前推；返回更新后的窗口状态 X(L,n,6)。
    py::array_t<double> step_py(py::object x_new, int steps) {
        if (x_new.is_none()) {
            step(nullptr, steps);
        } else {
            auto a = x_new.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
            auto r = a.request();
            if ((size_t)r.size != (size_t)n_ * 6)
                throw std::runtime_error("FlsWindow.step: x_new 需 (n,6)");
            step((double *)r.ptr, steps);
        }
        return X_host();
    }
    py::array_t<double> X_host() const {
        py::array_t<double> out({(py::ssize_t)L_, (py::ssize_t)n_, (py::ssize_t)6});
        std::copy(hX_.begin(), hX_.end(), out.mutable_data());
        return out;
    }
    py::array_t<double> P_marg_host() const {
        py::array_t<double> out({(py::ssize_t)n_, (py::ssize_t)6, (py::ssize_t)6});
        std::copy(Pm_host_.begin(), Pm_host_.end(), out.mutable_data());
        return out;
    }
    py::array_t<double> Lp_host() const {
        py::array_t<double> out({(py::ssize_t)n_, (py::ssize_t)6, (py::ssize_t)6});
        std::copy(Lp_host_.begin(), Lp_host_.end(), out.mutable_data());
        return out;
    }
    py::array_t<double> z_prior_host() const {
        py::array_t<double> out({(py::ssize_t)n_, (py::ssize_t)6});
        std::copy(zp_host_.begin(), zp_host_.end(), out.mutable_data());
        return out;
    }
    double res2() const { return res2_; }
    // 逐帧消去（sweep）开关：True 时 GN 因子/求解走信息形式逐帧消去（Baspacho 只解 6N 单帧）。
    void set_sweep(bool on) { sweep_ = on; }
    bool sweep() const { return sweep_; }
    // C++ 原生访问（gtest 用，不经 py）
    const std::vector<double> &X_ref() const { return hX_; }
    const std::vector<double> &P_marg_ref() const { return Pm_host_; }
    const std::vector<double> &Lp_ref() const { return Lp_host_; }
    // 单步耗时归因：{pred, geo, asm, gn, schur核, schur D2H, schur 其余, 全步}
    std::vector<double> timings() const { return {dbg_pred, dbg_geo, dbg_asm, dbg_gn,
                                                  dbg_marg, dbg_d2h, dbg_rest, dbg_total,
                                                  dbg_mtm, dbg_fac, dbg_solve}; }
    py::array_t<double> sc_host() const {
        auto c = mtm_scl_.to(torch::kCPU);
        py::array_t<double> out({(py::ssize_t)c.numel()});
        std::copy(c.data_ptr<double>(), c.data_ptr<double>() + c.numel(), out.mutable_data());
        return out;
    }

  private:
    // 全窗几何（**全 CUDA**）：QOE → rv 与 ∂p/∂oe；避开宿主 DACE 的反复 init / 线程态开销。
    void geo_all() {
        cudaMemcpy(d_Xa_, hX_.data(), (size_t)L_ * n_ * 6 * sizeof(double), cudaMemcpyHostToDevice);
        qoejopt::oeRvJacDevice(d_Xa_, L_ * n_, d_rvGeo_, d_ApGeo_);
        cudaMemcpy(hRv_.data(), d_rvGeo_, (size_t)L_ * n_ * 6 * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(hAp_.data(), d_ApGeo_, (size_t)L_ * n_ * 18 * sizeof(double), cudaMemcpyDeviceToHost);
    }
    // 列均衡（1/√列能量，上限 1e6）+ Aᵀb
    std::pair<torch::Tensor, torch::Tensor> scaled_AtA_Atb() {
        auto A2 = A_val_ * A_val_;
        auto cn = torch::zeros({S_.ncol}, A_val_.options());
        cn.index_add_(0, A_ci_t_, A2);
        mtm_scl_ = torch::clamp(1.0 / torch::sqrt(torch::clamp_min(cn, 1e-30)), 0.0, 1e6);
        auto Av_s = A_val_ * mtm_scl_.index_select(0, A_ci_t_);
        auto Atb = torch::zeros({S_.ncol}, A_val_.options());
        Atb.index_add_(0, A_ci_t_, Av_s * b_.index_select(0, row_nz_));
        return {Av_s, Atb};
    }
    // 帧 m 的行缓冲 gather（从窗口 scaled A 值 Av_s 与 b_ 按行组拷贝；同组行序一致）。
    void gather_frame_(int m, const torch::Tensor &Av_s) {
        const int64_t nE12 = (int64_t)E_ * 12, nG6 = (int64_t)G_ * 6, nD = (int64_t)n_ * 6;
        Af_t_.narrow(0, F_.off_isl, nE12)
            .copy_(Av_s.narrow(0, (int64_t)S_.off_isl + (int64_t)m * nE12, nE12));
        Af_t_.narrow(0, F_.off_gts, nG6)
            .copy_(Av_s.narrow(0, (int64_t)S_.off_gts + (int64_t)m * nG6, nG6));
        if (m < L_ - 1)
            Af_t_.narrow(0, F_.off_dynN, nD * 6)
                .copy_(Av_s.narrow(0, (int64_t)S_.off_dyn + (int64_t)m * nD * 12, nD * 12)
                           .reshape({nD, 12}).narrow(1, 0, 6).reshape({-1}));
        else
            Af_t_.narrow(0, F_.off_dynN, nD * 6).zero_();
        if (m >= 1)
            Af_t_.narrow(0, F_.off_dynP, nD * 6)
                .copy_(Av_s.narrow(0, (int64_t)S_.off_dyn + (int64_t)(m - 1) * nD * 12, nD * 12)
                           .reshape({nD, 12}).narrow(1, 6, 6).reshape({-1}));
        else
            Af_t_.narrow(0, F_.off_dynP, nD * 6).zero_();
        if (m == 0)
            Af_t_.narrow(0, F_.off_prior, nD * 6)
                .copy_(Av_s.narrow(0, (int64_t)S_.off_damp, nD * 6));
        else
            Af_t_.narrow(0, F_.off_prior, nD * 6).zero_();
        bf_t_.narrow(0, F_.r_isl, E_).copy_(b_.narrow(0, (int64_t)m * E_, E_));
        bf_t_.narrow(0, F_.r_gts, G_).copy_(b_.narrow(0, S_.nISL + (int64_t)m * G_, G_));
        if (m < L_ - 1)
            bf_t_.narrow(0, F_.r_dynN, nD).copy_(
                b_.narrow(0, S_.nISL + S_.nGTS + (int64_t)m * nD, nD));
        else
            bf_t_.narrow(0, F_.r_dynN, nD).zero_();
        if (m >= 1)
            bf_t_.narrow(0, F_.r_dynP, nD).copy_(
                b_.narrow(0, S_.nISL + S_.nGTS + (int64_t)(m - 1) * nD, nD));
        else
            bf_t_.narrow(0, F_.r_dynP, nD).zero_();
        if (m == 0)
            bf_t_.narrow(0, F_.r_prior, nD).copy_(
                b_.narrow(0, S_.nISL + S_.nGTS + S_.nDYN, nD));
        else
            bf_t_.narrow(0, F_.r_prior, nD).zero_();
    }

    static double *mal(size_t cnt) { double *p = nullptr; cudaMalloc(&p, cnt * sizeof(double)); return p; }
    static int *upip(const std::vector<int> &v) { int *p = nullptr; cudaMalloc(&p, v.size() * sizeof(int)); cudaMemcpy(p, v.data(), v.size() * sizeof(int), cudaMemcpyHostToDevice); return p; }
    static double *updp(const std::vector<double> &v) { double *p = nullptr; cudaMalloc(&p, v.size() * sizeof(double)); cudaMemcpy(p, v.data(), v.size() * sizeof(double), cudaMemcpyHostToDevice); return p; }
    int *upi(const std::vector<int> &v) { return upip(v); }
    double *upd(const std::vector<double> &v) { return updp(v); }

    int n_, L_, E_, G_;
    double dt_, sigma_isl_, sigma_gts_;
    std::vector<int> isl_i_, isl_j_, gts_;
    bool prior_ = true, has_prior_ = false;
    FlsStruct S_;
    qoejopt::WinCtx wc_;
    int *d_isl_i_ = nullptr, *d_isl_jj_ = nullptr, *d_gts_a_ = nullptr;
    double *d_qingt_ = nullptr, *d_isl_D_ = nullptr, *d_isl_W_ = nullptr;
    double *d_gts_D_ = nullptr, *d_gts_W_ = nullptr, *d_gts_Q_ = nullptr, *d_ctr_ = nullptr;
    double *d_Xa_ = nullptr, *d_rv_ = nullptr, *d_Ap_ = nullptr;
    double *d_end_ = nullptr, *d_phi_ = nullptr, *d_res_ = nullptr, *d_Pm_ = nullptr;
    torch::Tensor A_rp_t_, A_ci_t_, A_val_, b_;
    torch::Tensor X_, rv_, Ap_;                             // x_k / 几何 / ∂p∂oe
    torch::Tensor Lambda0_, Lp_, z_prior_;                  // Λ0⁻ / 白化 / 先验中心
    std::vector<double> hX_, hRv_, hAp_;
    // FLS 求解/边缘化
    std::unique_ptr<SymbolicDecomposition> sym_;            // CUDA（GN 用）
    torch::Tensor row_nz_, mtm_scl_;                        // 非零行号 / 列均衡因子
    double *d_rvGeo_ = nullptr, *d_ApGeo_ = nullptr;        // 几何 scratch（全 CUDA）
    // P1'（逐星 Schur）：逐星行表（device int64；给 flsMargDevice）
    torch::Tensor d00_p_, d00_s_, d00_c_, d11_p_, d11_s_, d11_c_, pr_s1_, pr_s0_;
    double *d_PmMarg_ = nullptr;                            // Pmarg scratch (n*36)
    // 单步耗时归因（timings()）
    double dbg_pred = 0, dbg_geo = 0, dbg_asm = 0, dbg_gn = 0, dbg_marg = 0, dbg_d2h = 0,
           dbg_rest = 0, dbg_total = 0, dbg_mtm = 0, dbg_fac = 0, dbg_solve = 0;
    std::vector<double> Pm_host_, Lp_host_, zp_host_;       // P^marg / Lp / z_prior（host 副本）
    double res2_ = 0.0;
    // 逐帧消去（sweep）：单帧结构/符号 + 缓冲 + 逐星相对行表 + 修正图案
    FlsFrameStruct F_;
    std::unique_ptr<SymbolicDecomposition> symF_;
    torch::Tensor ArpF_t_, AciF_t_, rnzF_t_, rpC_t_, ciC_t_;
    torch::Tensor Af_t_, bf_t_, eta_t_, yF_t_, dlt_t_, gam_t_, tmp_t_, weak_Lp_, zw_t_;
    torch::Tensor fis_p_, fis_s_, fis_c_, fgt_p_, fgt_s_, fgt_c_;
    bool sweep_ = false;
};

// 探针：直接调宿主 C++ 接口算 rv/Ap（隔离测试重复调用/states）
static py::tuple fls_geom_probe(py::array_t<double, py::array::c_style | py::array::forcecast> X) {
    auto r = X.request();
    const int total = (int)(r.size / 6);
    const double *px = static_cast<const double *>(r.ptr);
    std::vector<Vector6d> oes(total);
    for (int j = 0; j < total; ++j)
        for (int c = 0; c < 6; ++c) oes[j](c) = px[(size_t)j * 6 + c];
    std::vector<double> RV((size_t)total * 6), J6((size_t)total * 36);
    noeOsc2rvBatch(oes, 16, RV);
    noeOsc2rvJacBatch(oes, 16, J6);
    py::array_t<double> rv_out({(py::ssize_t)total, (py::ssize_t)6});
    std::copy(RV.begin(), RV.end(), rv_out.mutable_data());
    py::array_t<double> J_out({(py::ssize_t)total, (py::ssize_t)6, (py::ssize_t)6});
    std::copy(J6.begin(), J6.end(), J_out.mutable_data());
    return py::make_tuple(rv_out, J_out);
}

PYBIND11_MODULE(qoefls, m) {
    m.doc() = "FLS 固定滞后平滑 C++ 核心（factor/滑窗/Schur 边缘化）";
    m.def("fls_probe", &fls_probe, py::arg("n"), "torch+BaSpaCho 链接探针");
    m.def("fls_baspacho_probe", &fls_baspacho_probe,
          "BaSpaCho C++ 全链探针（add_MtM/factor/solve/compute_diag_blocks vs 稠密）");
    m.def("fls_build_struct", &fls_build_struct, py::arg("n"), py::arg("L"),
          py::arg("edges_i"), py::arg("edges_j"), py::arg("gts"), py::arg("damp"),
          "FLS 滑窗装配结构（CSR + 块图案），等价 py _lift_struct(prior=True)/_lift_block_struct");
    m.def("fls_geom_probe", &fls_geom_probe, py::arg("X"),
          "探针：QOE(n,6) → (rv(n,6), ∂(r,v)/∂oe(n,6,6))，直调宿主 C++ 接口");
    py::class_<FlsWindow>(m, "FlsWindow")
        .def(py::init([](int n, int L, int E, int G, double dt, double s1, double s2,
                         py::array_t<int, py::array::c_style | py::array::forcecast> ei,
                         py::array_t<int, py::array::c_style | py::array::forcecast> ej,
                         py::array_t<int, py::array::c_style | py::array::forcecast> gts,
                         py::array_t<double, py::array::c_style | py::array::forcecast> Q,
                         double damp, bool prior) {
            auto ri = ei.request(); auto rj = ej.request();
            auto rg = gts.request(); auto rq = Q.request();
            return new FlsWindow(n, L, E, G, dt, s1, s2,
                                 std::vector<int>((int *)ri.ptr, (int *)ri.ptr + ri.size),
                                 std::vector<int>((int *)rj.ptr, (int *)rj.ptr + rj.size),
                                 std::vector<int>((int *)rg.ptr, (int *)rg.ptr + rg.size),
                                 std::vector<double>((double *)rq.ptr, (double *)rq.ptr + rq.size),
                                 damp, prior);
        }), py::return_value_policy::take_ownership,
            py::arg("n"), py::arg("L"), py::arg("E"), py::arg("G"), py::arg("dt"),
            py::arg("sigma_isl"), py::arg("sigma_gts"), py::arg("isl_i"), py::arg("isl_j"),
            py::arg("gts"), py::arg("Q"), py::arg("damp") = 1e-6, py::arg("prior") = true)
        .def("set_obs", &FlsWindow::set_obs, py::arg("D_isl"), py::arg("W_isl"),
             py::arg("D_gts"), py::arg("W_gts"), py::arg("Q_gts"), py::arg("ctr"))
        .def("set_X", &FlsWindow::set_X, py::arg("X"))
        .def("set_prior", &FlsWindow::set_prior, py::arg("Lp"), py::arg("z_prior"))
        .def("set_sweep", &FlsWindow::set_sweep, py::arg("on"),
             "GN 因子/求解改走信息形式逐帧消去（Baspacho 只解 6N 单帧；边际/质量应与全窗一致）")
        .def("sweep", &FlsWindow::sweep, "逐帧消去是否开启")
        .def("assemble", &FlsWindow::assemble, "装配 A_val/b（ISL/GTS/dyn/先验），返回 ISL res2")
        .def("A_val", &FlsWindow::A_val)
        .def("b", &FlsWindow::b)
        .def("ncol", &FlsWindow::ncol)
        .def("nrows", &FlsWindow::nrows)
        .def("nnz", &FlsWindow::nnz)
        .def("step", &FlsWindow::step_py, py::arg("x_new") = py::none(), py::arg("steps") = 5,
             "推进一步（py wrapper；纯 C++ 版为 FlsWindow::step）："
             "GVE 前推/队列左移 → 装配+GN → Schur 边缘 → 先验白化；返回 X(L,n,6)")
        .def("X_host", &FlsWindow::X_host, "窗口状态 X(L,n,6)")
        .def("P_marg_host", &FlsWindow::P_marg_host, "保留历元每星 6×6 Schur 边缘 P^marg (n,6,6)")
        .def("Lp_host", &FlsWindow::Lp_host, "Λ0 白化因子 Lp (n,6,6)")
        .def("z_prior_host", &FlsWindow::z_prior_host, "先验中心 z_prior (n,6)")
        .def("res2", &FlsWindow::res2, "上次装配的 ISL 残差平方和")
        .def("timings", &FlsWindow::timings,
             "单步耗时归因 [pred, geo, asm, gn, schur核, schur D2H, schur 其余, 全步] (s)")
        .def("sc_host", &FlsWindow::sc_host, "GN 末次列均衡因子 sc（调试/参考口径一致用）");
}
