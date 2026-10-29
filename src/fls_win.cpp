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
#include <limits>
#include <string>
#include <memory>
#include <queue>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "dastates.h"
#include "extlib/baspacho_solver.h"
#include "extlib/cusolver_lu_solver.h"
#include "kepler.h"
#include "qoejopt.h"
#include "fls_lists.h"
#include "fls_local_handoff.h"
#include "fls_gauge_householder.h"
#include "fls_gauge_lamwin.h"

namespace py = pybind11;

// 窗口 gauge 求解的 scaled 空间对角正则（与历史 solve_batch 口径一致）。
static constexpr double kGaugeSolveEps = 1e-9;

// G^T G 的正则 eps：平移列 (~N) 与旋转列 (~N r^2) 量级差 ~1e13，
// 不能用全局 trace 相对 eps（会淹没平移块）。这里取最小对角元的相对量，
// 再叠加 trace 的 1e-18 仅用于秩亏方向；两者都不改变正常块。
static double gauge_eps_from_gram(const Eigen::MatrixXd &Sg) {
    const double tr = std::max(Sg.trace(), 0.0);
    double mind = 0.0;
    if (Sg.rows() > 0) {
        mind = Sg.diagonal().minCoeff();
        if (!(mind > 0.0)) mind = 0.0;
    }
    return std::max(std::max(mind * 1e-12, tr * 1e-18), 1e-300);
}

// 对称矩阵的均衡 SVD 伪逆：ds=1/sqrt(|diag|)，使 Ds A Ds 的对角为 ±1，
// 再按相对阈值 sv > max(sv)*1e-12 截断。rank loss 只截断、不抛错；
// 非有限输入才返回 false。输出 Ainv = Ds * pinv(Ds A Ds) * Ds，
// 回填 rank/cond/trunc 诊断。注意 ds 必须取 1/sqrt(|diag|)：若取 sqrt(|diag|)，
// 均衡矩阵的条件数会被平方，1e-12 截断会误伤 1e-6 相对量级的真实方向。
struct GaugePinvDiag {
    int rank = 0;
    int trunc = 0;
    double cond = std::numeric_limits<double>::infinity();      // 原始 A 的 cond
    double cond_bal = std::numeric_limits<double>::infinity();  // 均衡 A 的 cond
};

// 末次 solve_batch 的逐轮诊断（只读，不影响 solve 行为）。
// dx_* 是线搜索最终尝试/接受的参数增量；accepted=false 时实际位移为 0，
// 因此另存 dx_trial_* 供观察被拒绝的大步尝试。cost_after 是最后一次
// 线搜索评估的 merit；cost_after_refresh 是随后 assemble 刷新 Pg_eff 后
// 实际带入下一轮的 res2（gauge 路径二者可能略有差异）。
struct SolveIterDiag {
    int iter = -1;
    double dx_norm = 0.0;
    double dx_max = 0.0;
    double dx_trial_norm = 0.0;
    double dx_trial_max = 0.0;
    double alpha = 0.0;
    double cost_before = 0.0;
    double cost_after = 0.0;
    double cost_after_refresh = 0.0;
    bool accepted = false;
    int ls_trials = 0;
};

static bool svd_pinv_balanced_(const Eigen::MatrixXd &A, Eigen::MatrixXd &Ainv,
                               GaugePinvDiag *diag) {
    if (A.rows() != A.cols() || A.rows() <= 0 || !A.allFinite()) return false;
    const int m = (int)A.rows();
    Eigen::VectorXd ds(m);
    for (int i = 0; i < m; ++i)
        ds(i) = 1.0 / std::sqrt(std::max(std::abs(A(i, i)), 1e-300));
    Eigen::MatrixXd Ab = ds.asDiagonal() * A * ds.asDiagonal();
    Ab = 0.5 * (Ab + Ab.transpose());
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(Ab, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const auto &sv = svd.singularValues();
    if (sv.size() == 0 || !sv.allFinite()) return false;
    const double smax = sv(0);
    const double cut = smax * 1e-12;
    int rank = 0;
    Eigen::MatrixXd pinv = Eigen::MatrixXd::Zero(m, m);
    for (int i = 0; i < sv.size(); ++i) {
        if (sv(i) > cut && sv(i) > 0.0) {
            ++rank;
            pinv.noalias() +=
                (svd.matrixV().col(i) * svd.matrixU().col(i).transpose()) / sv(i);
        }
    }
    Ainv = ds.asDiagonal() * pinv * ds.asDiagonal();
    Ainv = 0.5 * (Ainv + Ainv.transpose());
    if (!Ainv.allFinite()) return false;
    if (diag) {
        diag->rank = rank;
        diag->trunc = m - rank;
        diag->cond_bal = (smax > 0.0)
                             ? smax / std::max(sv(sv.size() - 1), 1e-300)
                             : std::numeric_limits<double>::infinity();
        Eigen::JacobiSVD<Eigen::MatrixXd> svd_raw(A);
        const auto &svr = svd_raw.singularValues();
        diag->cond = (svr.size() > 0 && svr(0) > 0.0)
                         ? svr(0) / std::max(svr(svr.size() - 1), 1e-300)
                         : std::numeric_limits<double>::infinity();
    }
    return true;
}

// FNV-1a 64：给 Woodbury 的 D^-1 做内容指纹，便于验证
// marginalize_first_local_gauge 与 P_node_host_fast_gauge 用的是同一 D。
static uint64_t fnv1a64(const void *data, size_t nbytes) {
    const unsigned char *p = static_cast<const unsigned char *>(data);
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < nbytes; ++i) {
        h ^= (uint64_t)p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

// Eigen（列主序）→ numpy（C 行主序）拷贝，供诊断/参考路径返回矩阵。
static py::array_t<double> eigen_to_np_rowmajor(const Eigen::MatrixXd &M) {
    py::array_t<double> out({(py::ssize_t)M.rows(), (py::ssize_t)M.cols()});
    auto *p = out.mutable_data();
    for (int i = 0; i < M.rows(); ++i)
        for (int j = 0; j < M.cols(); ++j) p[(size_t)i * M.cols() + j] = M(i, j);
    return out;
}

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
                                 const std::vector<int> &gts, double damp, bool prior,
                                 int prior_node = -1, int r_cross = 1) {
    FlsStruct s;
    s.n = n; s.L = L; s.E = (int)isl_i.size(); s.G = (int)gts.size();
    s.nISL = L * s.E; s.nGTS = L * s.G; s.nDYN = (L - 1) * n * 6;
    s.nPRI = prior ? ((prior_node >= 0) ? n * 6 : L * n * 6) : 0;
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
        const int r0 = (prior_node >= 0) ? prior_node * n * 6 : 0;
        const int r1 = (prior_node >= 0) ? (prior_node + 1) * n * 6 : L * n * 6;
        for (int r = r0; r < r1; ++r) {
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
    // r_cross>=2：把同历元 ISL 图的 r 跳邻域也加入块结构，供 set_prior_cross
    // 注入局部联合边界先验的跨星块（默认 1 = 只到 1 跳，保持旧结构）。
    if (r_cross >= 2) {
        std::vector<std::vector<int>> g((size_t)n);
        for (int e = 0; e < s.E; ++e) {
            const int a = isl_i[e], b = isl_j[e];
            if (a != b) { g[(size_t)a].push_back(b); g[(size_t)b].push_back(a); }
        }
        for (int k = 0; k < L; ++k)
            for (int u = 0; u < n; ++u) {
                std::vector<int> dist((size_t)n, -1);
                std::queue<int> q; q.push(u); dist[(size_t)u] = 0;
                while (!q.empty()) {
                    const int x = q.front(); q.pop();
                    if (dist[(size_t)x] >= r_cross) continue;
                    for (int y : g[(size_t)x])
                        if (dist[(size_t)y] < 0) {
                            dist[(size_t)y] = dist[(size_t)x] + 1;
                            q.push(y);
                        }
                }
                for (int v = 0; v < n; ++v)
                    if (v != u && dist[(size_t)v] > 0) addb(k * n + u, k * n + v);
            }
    }
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
                                 double damp, int prior_node = -1, int r_cross = 1) {
    auto req_i = ei.request(); auto req_j = ej.request(); auto req_g = gts.request();
    const int *pi = static_cast<const int *>(req_i.ptr);
    const int *pj = static_cast<const int *>(req_j.ptr);
    const int *pg = static_cast<const int *>(req_g.ptr);
    std::vector<int> vi(pi, pi + req_i.size), vj(pj, pj + req_j.size), vg(pg, pg + req_g.size);
    FlsStruct s = fls_struct_impl(n, L, vi, vj, vg, damp, true, prior_node, r_cross);
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
              std::vector<double> Q, double damp, bool prior, int prior_node = -1,
              int dyn_sub = 1, int r_cross = 1)
        : n_(n), L_(L), E_(E), G_(G), dt_(dt), sigma_isl_(sigma_isl), sigma_gts_(sigma_gts),
          isl_i_(std::move(isl_i)), isl_j_(std::move(isl_j)), gts_(std::move(gts)),
          prior_(prior), prior_node_(prior_node), dyn_sub_(dyn_sub),
          dyn_sub_cfg_(dyn_sub > 0 ? dyn_sub : 1), r_cross_(r_cross) {
        S_ = fls_struct_impl(n_, L_, isl_i_, isl_j_, gts_, damp, prior_, prior_node_, r_cross_);
        // GTS→Pg 的节点范围默认全部节点 [0,L)；滑窗第二窗起由
        // set_gts_pg_node_range(L-1, L) 只让新进入的末节点贡献 G 分量。
        gts_pg_node_first_ = 0;
        gts_pg_node_last_ = L_;
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
        d_Xcost_ = mal((size_t)L_ * n_ * 6);
        d_rvCost_ = mal((size_t)L_ * n_ * 6);
        d_ApCost_ = mal((size_t)L_ * n_ * 18);
        const size_t nDyn_alloc = (size_t)std::max(L_ - 1, 1);   // L=1 交接也需 1 个区间的 φ/Φ
        d_ekf_host_.assign((size_t)n_ * 6, 1.0);                 // 固定每星缩放 D（set_P_init 冻结）
        d_ekf_scale_ = mal((size_t)n_ * 6);
        cudaMemcpy(d_ekf_scale_, d_ekf_host_.data(), (size_t)n_ * 6 * sizeof(double),
                   cudaMemcpyHostToDevice);
        d_end_ = mal(nDyn_alloc * n_ * 6); d_phi_ = mal(nDyn_alloc * n_ * 36);
        d_state_sub_ = mal(nDyn_alloc * n_ * 6); d_state_b_ = mal(nDyn_alloc * n_ * 6);
        d_phi_acc_ = mal(nDyn_alloc * n_ * 36); d_phi_out_ = mal(nDyn_alloc * n_ * 36);
        d_phi_step_ = mal(nDyn_alloc * n_ * 36);
        d_qingt_key_ = mal(nDyn_alloc * n_ * 36);
        d_ekf_xref_ = mal(nDyn_alloc * n_ * 6);
        d_ekf_xpred_ = mal(nDyn_alloc * n_ * 6);
        d_ekf_phi_ = mal(nDyn_alloc * n_ * 36);
        d_ekf_Pprev_ = mal(nDyn_alloc * n_ * 36);
        d_ekf_Ppred_ = mal(nDyn_alloc * n_ * 36);
        d_ekf_qeff_ = mal(nDyn_alloc * n_ * 36);
        d_Pplus_ = mal((size_t)n_ * 36);
        d_KH_ = mal((size_t)n_ * 36);
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
        Lp_ = torch::zeros({n_, 6, 6}, optsD);              // 节点0 白化因子（兼容旧接口）
        z_prior_ = torch::zeros({n_, 6}, optsD);
        Lp_all_ = torch::zeros({L_, n_, 6, 6}, optsD);      // 每节点 6×6 先验（关键帧窗口）
        z_prior_all_ = torch::zeros({L_, n_, 6}, optsD);
        hX_.assign((size_t)L_ * n_ * 6, 0.0);
        cost_ref_.assign((size_t)L_ * n_ * 6, 0.0);   // D7 cost 参考点，初始与 hX_ 一致
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
        c.dyn_sub = dyn_sub_;
        c.d_state_sub = d_state_sub_; c.d_state_b = d_state_b_;
        c.d_phi_acc = d_phi_acc_; c.d_phi_out = d_phi_out_; c.d_phi_step = d_phi_step_;
        c.d_dyn_xref = d_ekf_xref_;
        c.gauge_k = 0;
        c.d_gauge_Ga = c.d_gauge_Gb = c.d_gauge_C = nullptr;
        c.d_gauge_corr = nullptr;
        // FLS 自算几何 / 求解所需
        Pm_host_.assign((size_t)n_ * 36, 0.0);
        T_host_.assign((size_t)n_ * 36, 0.0);
        ekf_T_chain_.assign((size_t)n_ * 36, 0.0);
        phi_host_.assign((size_t)n_ * 36, 0.0);
        ekf_jnorm_host_.assign((size_t)n_, 0.0);
        ekf_jmin_host_.assign((size_t)n_, 0.0);
        for (int i = 0; i < n_; ++i)
            for (int a = 0; a < 6; ++a)
                for (int b = 0; b < 6; ++b) {
                    T_host_[(size_t)i * 36 + a * 6 + b] = (a == b) ? 1.0 : 0.0;
                    ekf_T_chain_[(size_t)i * 36 + a * 6 + b] = (a == b) ? 1.0 : 0.0;
                }
        Lp_host_.assign((size_t)n_ * 36, 0.0);
        zp_host_.assign((size_t)n_ * 6, 0.0);
        auto optsIcpu = torch::TensorOptions().dtype(torch::kInt64);
        auto paramSize = torch::full({(int64_t)L_ * n_}, 6, optsIcpu);
        auto bptr_t = torch::from_blob(S_.bptrs.data(), {(py::ssize_t)S_.bptrs.size()}, optsIcpu).clone();
        auto bind_t = torch::from_blob(S_.binds.data(), {(py::ssize_t)S_.binds.size()}, optsIcpu).clone();
        // FLS_SYM_UPPER=1：符号结构只保留上三角块。BaSpaCho 的 AMD/fill 路径
        // 假定输入是单三角，全对称输入会产生重复项并虚增 fill。
        const bool sym_upper = std::getenv("FLS_SYM_UPPER") != nullptr;
        auto upper_struct = [](const std::vector<int64_t> &bp, const std::vector<int64_t> &bi) {
            std::vector<int64_t> up{0}, ui;
            for (size_t r = 0; r + 1 < bp.size(); ++r) {
                for (int64_t q = bp[r]; q < bp[r + 1]; ++q)
                    if (bi[(size_t)q] >= (int64_t)r) ui.push_back(bi[(size_t)q]);
                up.push_back((int64_t)ui.size());
            }
            return std::make_pair(up, ui);
        };
        if (sym_upper) {
            auto us = upper_struct(S_.bptrs, S_.binds);
            bptr_t = torch::tensor(us.first, optsIcpu);
            bind_t = torch::tensor(us.second, optsIcpu);
        }
        // 调试/调优：FLS_ELIM_MODE=nodes 强制按节点 0→L-1 顺序消去；
        // autoL 以第 L 节点为分界（前 L-1 节点按给定顺序，其余自适应）。
        {
            const char *em = std::getenv("FLS_ELIM_MODE");
            const int emode = em ? std::atoi(em) : 0;
            if (emode > 0 && L_ > 1) {
                std::vector<int64_t> ranges{0};
                const int nfr = std::min(emode, L_);
                for (int k = 1; k <= nfr; ++k) ranges.push_back((int64_t)k * n_);
                sym_ = std::make_unique<SymbolicDecomposition>(
                    paramSize, bptr_t, bind_t, "cuda", ranges, std::vector<int64_t>{},
                    /*AddFillForGivenElims=*/2, /*findSparseEliminationRanges=*/false);
                fprintf(stderr, "[fls/sym] FLS_ELIM_MODE=%d ranges[0..%d]=%lld\n", emode,
                        (int)ranges.size() - 1, (long long)ranges.back());
            } else {
                // FLS_KF_ELIMLAST=k：把最后 k 个帧的节点强制放到消去序末尾
                // （BaSpaCho elimLastIds，仍需 AddFillComplete）。用于量化
                // 「时间分层图 + 全局 AMD」与「逐帧块序」的 fill 差。
                const char *el = std::getenv("FLS_KF_ELIMLAST");
                const int klast = el ? std::atoi(el) : 0;
                if (klast > 0 && L_ > 1) {
                    std::vector<int64_t> last;
                    for (int k = std::max(0, L_ - klast); k < L_; ++k)
                        for (int i = 0; i < n_; ++i) last.push_back((int64_t)k * n_ + i);
                    sym_ = std::make_unique<SymbolicDecomposition>(
                        paramSize, bptr_t, bind_t, "cuda", std::vector<int64_t>{}, last,
                        /*AddFillComplete=*/0, /*findSparseEliminationRanges=*/true);
                    fprintf(stderr, "[fls/sym] FLS_KF_ELIMLAST=%d nlast=%d\n", klast,
                            (int)last.size());
                } else {
                    sym_ = std::make_unique<SymbolicDecomposition>(paramSize, bptr_t, bind_t, "cuda");
                }
            }
        }
        symc_ = std::make_unique<SymbolicDecomposition>(paramSize, bptr_t, bind_t, "cpu");
        // ---- P1'（逐星 Schur）：行表由 fls_lists.h 统一构造（与 gtest 同源）----
        {
            FlsLists lists = fls_make_lists(n_, L_, E_, G_, isl_i_, isl_j_, gts_,
                                            S_.off_isl, S_.off_gts, S_.off_dyn, S_.off_damp);
            d00_p_ = torch::tensor(lists.d00_p, optsI);
            d00_s_ = torch::tensor(lists.d00_s, optsI);
            d00_c_ = torch::tensor(lists.d00_c, optsI);
            d11_p_ = torch::tensor(lists.d11_p, optsI);
            d11_s_ = torch::tensor(lists.d11_s, optsI);
            d11_c_ = torch::tensor(lists.d11_c, optsI);
            pr_s1_ = torch::tensor(lists.pr_s1, optsI);
            pr_s0_ = torch::tensor(lists.pr_s0, optsI);
            d_PmMarg_ = mal((size_t)n_ * 36);
        }
        {
            std::vector<int64_t> rnz;
            rnz.reserve(S_.nnz);
            for (int r = 0; r < S_.nrows; ++r)
                for (int j = S_.A_rp[r]; j < S_.A_rp[r + 1]; ++j) rnz.push_back(r);
            row_nz_ = torch::tensor(rnz, optsI);
            std::vector<int64_t> rpE{(int64_t)0}, ciE;
            for (int64_t q = 0; q < S_.ncol; ++q) { ciE.push_back(q); rpE.push_back(q + 1); }
            auto optsIcpuE = torch::TensorOptions().dtype(torch::kInt64);
            auto rpE_t = torch::from_blob(rpE.data(), {(py::ssize_t)rpE.size()}, optsIcpuE).clone();
            auto ciE_t = torch::from_blob(ciE.data(), {(py::ssize_t)ciE.size()}, optsIcpuE).clone();
            rp_eps_t_ = rpE_t.to(optsI);
            ci_eps_t_ = ciE_t.to(optsI);
        }
        // ---- 逐帧消去（sweep，l430-433 选项 2）：单帧结构/符号 + 单帧缓冲 + 逐星相对行表 ----
        {
            F_ = fls_frame_impl(n_, isl_i_, isl_j_, gts_);
            auto optsIcpu2 = torch::TensorOptions().dtype(torch::kInt64);
            auto bpF = torch::from_blob(F_.bptrs.data(), {(py::ssize_t)F_.bptrs.size()},
                                        optsIcpu2).clone();
            auto biF = torch::from_blob(F_.binds.data(), {(py::ssize_t)F_.binds.size()},
                                        optsIcpu2).clone();
            if (sym_upper) {
                auto us = upper_struct(F_.bptrs, F_.binds);
                bpF = torch::tensor(us.first, optsIcpu2);
                biF = torch::tensor(us.second, optsIcpu2);
            }
            auto psF = torch::full({(int64_t)n_}, 6, optsIcpu2);
            symF_ = std::make_unique<SymbolicDecomposition>(psF, bpF, biF, "cuda");
            // ---- 星主序联合因子（KF_CHAIN）：每星 6L 变量一块，块图沿用 F_ 的 ISL 星图 ----
            if (L_ > 1) {
                auto psS = torch::full({(int64_t)n_}, (int64_t)6 * L_, optsIcpu2);
                symStar_ = std::make_unique<SymbolicDecomposition>(psS, bpF, biF, "cuda");
                std::vector<int64_t> perm((size_t)S_.ncol), inv((size_t)S_.ncol);
                for (int k = 0; k < L_; ++k)
                    for (int i = 0; i < n_; ++i)
                        for (int c = 0; c < 6; ++c) {
                            const int64_t oldi = ((int64_t)k * n_ + i) * 6 + c;
                            const int64_t newi = ((int64_t)i * L_ + k) * 6 + c;
                            perm[(size_t)oldi] = newi;
                            inv[(size_t)newi] = oldi;
                        }
                col_perm_t_ = torch::tensor(perm, torch::TensorOptions().dtype(torch::kInt64))
                                  .to(optsI);
                inv_perm_t_ = torch::tensor(inv, torch::TensorOptions().dtype(torch::kInt64))
                                  .to(optsI);
                auto ci_cpu = A_ci_t_.to(torch::kCPU);
                auto cia = ci_cpu.accessor<int64_t, 1>();
                std::vector<int64_t> cistar((size_t)S_.nnz);
                for (int64_t q = 0; q < S_.nnz; ++q)
                    cistar[(size_t)q] = perm[(size_t)cia[q]];
                A_ci_star_t_ =
                    torch::tensor(cistar, torch::TensorOptions().dtype(torch::kInt64)).to(optsI);
            }
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
            const char *gm = std::getenv("FLS_MARG_GLOBAL");
            global_marg_ = (gm && std::atoi(gm) != 0);
        }
    }
    ~FlsWindow() {
        cudaFree(d_isl_i_); cudaFree(d_isl_jj_); cudaFree(d_gts_a_); cudaFree(d_qingt_);
        cudaFree(d_isl_D_); cudaFree(d_isl_W_); cudaFree(d_gts_D_); cudaFree(d_gts_W_);
        cudaFree(d_gts_Q_); cudaFree(d_ctr_);
        cudaFree(d_Xa_); cudaFree(d_rv_); cudaFree(d_Ap_);
        cudaFree(d_Xcost_); cudaFree(d_rvCost_); cudaFree(d_ApCost_);
        cudaFree(d_end_); cudaFree(d_phi_); cudaFree(d_res_); cudaFree(d_ekf_scale_);
        cudaFree(d_state_sub_); cudaFree(d_state_b_);
        cudaFree(d_phi_acc_); cudaFree(d_phi_out_); cudaFree(d_phi_step_);
        cudaFree(d_qingt_key_); cudaFree(d_ekf_xref_); cudaFree(d_ekf_xpred_);
        cudaFree(d_ekf_phi_); cudaFree(d_ekf_Pprev_); cudaFree(d_ekf_Ppred_);
        cudaFree(d_ekf_qeff_); cudaFree(d_Pplus_); cudaFree(d_KH_);
        cudaFree(d_rvGeo_); cudaFree(d_ApGeo_); cudaFree(d_PmMarg_);
        cudaFree(d_gauge_Ga_); cudaFree(d_gauge_Gb_); cudaFree(d_gauge_C_);
        cudaFree(d_gauge_corr_);
    }

    // 固定 EKF 区间 dyn：输入 x_ref/x_pred/Phi_q/P_prev/P_pred (L-1,n,6/36)；
    // Q_eff 与白化 q=chol(Q_eff)^-T 在 CUDA 核内计算，Python 只传原始量。
    void set_dyn_ekf_raw(const double *xref, const double *xpred, const double *phi,
                         const double *Pprev, const double *Ppred) {
        const int nk = (L_ - 1) * n_;
        cudaMemcpy(d_ekf_xref_, xref, (size_t)nk * 6 * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_ekf_xpred_, xpred, (size_t)nk * 6 * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_ekf_phi_, phi, (size_t)nk * 36 * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_ekf_Pprev_, Pprev, (size_t)nk * 36 * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_ekf_Ppred_, Ppred, (size_t)nk * 36 * sizeof(double), cudaMemcpyHostToDevice);
        qoejopt::ekfDynFactorDevice(d_ekf_xref_, d_ekf_xpred_, d_ekf_phi_,
                                    d_ekf_Pprev_, d_ekf_Ppred_, L_ - 1, n_,
                                    d_end_, d_qingt_key_);
        dyn_sub_ = 0;
        wc_.dyn_sub = 0;
        wc_.d_qingt = d_qingt_key_;
        wc_.d_phi = d_ekf_phi_;      // dyn_sub=0 时装配核不再重算 STM，必须指向上传的 EKF Φ；
        wc_.q_perstar = 2;           // 否则 dyn 行读到未初始化缓冲（窗口发散根因）
        note_dyn_interval_inputs_(xref, phi);
    }
    void set_dyn_ekf(py::array_t<double, py::array::c_style | py::array::forcecast> xref,
                     py::array_t<double, py::array::c_style | py::array::forcecast> xpred,
                     py::array_t<double, py::array::c_style | py::array::forcecast> phi,
                     py::array_t<double, py::array::c_style | py::array::forcecast> Pprev,
                     py::array_t<double, py::array::c_style | py::array::forcecast> Ppred) {
        const size_t nk = (size_t)(L_ - 1) * n_;
        auto a = xref.request(); auto b = xpred.request(); auto c = phi.request();
        auto d = Pprev.request(); auto e = Ppred.request();
        if ((size_t)a.size != nk * 6 || (size_t)b.size != nk * 6 ||
            (size_t)c.size != nk * 36 || (size_t)d.size != nk * 36 ||
            (size_t)e.size != nk * 36)
            throw std::runtime_error("set_dyn_ekf: shapes must be (L-1,n,6/6/36/36/36)");
        set_dyn_ekf_raw((double *)a.ptr, (double *)b.ptr, (double *)c.ptr,
                        (double *)d.ptr, (double *)e.ptr);
    }
    void set_dyn_ekf_qeff(py::array_t<double, py::array::c_style | py::array::forcecast> xref,
                          py::array_t<double, py::array::c_style | py::array::forcecast> xpred,
                          py::array_t<double, py::array::c_style | py::array::forcecast> phi,
                          py::array_t<double, py::array::c_style | py::array::forcecast> qeff) {
        const size_t nk = (size_t)(L_ - 1) * n_;
        auto a = xref.request(); auto b = xpred.request(); auto c = phi.request();
        auto e = qeff.request();
        if ((size_t)a.size != nk * 6 || (size_t)b.size != nk * 6 ||
            (size_t)c.size != nk * 36 || (size_t)e.size != nk * 36)
            throw std::runtime_error("set_dyn_ekf_qeff: shapes must be (L-1,n,6/6/36/36)");
        cudaMemcpy(d_ekf_xref_, (double *)a.ptr, nk * 6 * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_ekf_xpred_, (double *)b.ptr, nk * 6 * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_ekf_phi_, (double *)c.ptr, nk * 36 * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_ekf_qeff_, (double *)e.ptr, nk * 36 * sizeof(double), cudaMemcpyHostToDevice);
        qoejopt::ekfDynQeffDevice(d_ekf_xref_, d_ekf_xpred_, d_ekf_phi_, d_ekf_qeff_,
                                  L_ - 1, n_, d_end_, d_qingt_key_);
        dyn_sub_ = 0;
        wc_.dyn_sub = 0;
        wc_.d_qingt = d_qingt_key_;
        wc_.d_phi = d_ekf_phi_;      // 同 set_dyn_ekf_raw：固定 EKF 区间必须接线上传的 Φ
        wc_.q_perstar = 2;
        note_dyn_interval_inputs_((double *)a.ptr, (double *)c.ptr);
    }

    // ---- 全 N gauge 修正（固定 EKF dyn）：T' = T + Gb C Gaᵀ ----
    // Python 提供每区间 Ga/Gb (6N,k)、C(k,k)。C 已含 (I−Gbᵀ T Ga)(GaᵀGa+eps I)^-1，
    // C++ 只做 g=Gaᵀ(X_a−x_ref)（6N 全星座点积）与 corr=Gb(C g)，并把 corr 作为
    // dyn 预测量从残差中扣除：e = x_b − x_pred − T'(x_a−x_ref)。
    // 注意：现有 dyn Jacobian 结构固定为 12 非零/行（−Φ_i, I），本实现只修正
    // 残差/预测；T' 的稠密低秩列耦合不进 A_val（见 dyn_gauge_diag 的诊断）。
    void note_dyn_interval_inputs_(const double *xref, const double *phi) {
        const size_t nk = (size_t)(L_ - 1) * n_;
        if (nk > 0) {
            dyn_xref_host_.assign(xref, xref + nk * 6);
            dyn_phi_host_.assign(phi, phi + nk * 36);
        } else {
            dyn_xref_host_.clear();
            dyn_phi_host_.clear();
        }
        has_dyn_xref_ = (nk > 0);
        has_dyn_phi_ = (nk > 0);
        wc_.d_dyn_xref = d_ekf_xref_;
        if (has_dyn_gauge_) dyn_gauge_metric_ok_ = false;   // 惰性：诊断时才重算 M
    }
    void sync_dyn_gauge_ctx_() {
        wc_.gauge_k = has_dyn_gauge_ ? dyn_gauge_k_ : 0;
        wc_.d_gauge_Ga = has_dyn_gauge_ ? d_gauge_Ga_ : nullptr;
        wc_.d_gauge_Gb = has_dyn_gauge_ ? d_gauge_Gb_ : nullptr;
        wc_.d_gauge_C = has_dyn_gauge_ ? d_gauge_C_ : nullptr;
        wc_.d_gauge_corr = has_dyn_gauge_ ? d_gauge_corr_ : nullptr;
        wc_.d_dyn_xref = d_ekf_xref_;
    }
    void free_dyn_gauge_dev_() {
        if (d_gauge_Ga_) { cudaFree(d_gauge_Ga_); d_gauge_Ga_ = nullptr; }
        if (d_gauge_Gb_) { cudaFree(d_gauge_Gb_); d_gauge_Gb_ = nullptr; }
        if (d_gauge_C_) { cudaFree(d_gauge_C_); d_gauge_C_ = nullptr; }
        if (d_gauge_corr_) { cudaFree(d_gauge_corr_); d_gauge_corr_ = nullptr; }
        wc_.gauge_k = 0;
        wc_.d_gauge_Ga = wc_.d_gauge_Gb = wc_.d_gauge_C = nullptr;
        wc_.d_gauge_corr = nullptr;
    }
    // Ga/Gb: (L−1,6N,k) 或单区间 (6N,k)；C: (L−1,k,k) 或 (k,k)。k≤64（设备归约上限）。
    // 逻辑限制：当前实现只把 Gb·C·Gaᵀ(X−xref) 从残差里减掉，dyn 行的
    // Jacobian 仍是 [-Φ,I]，代价与梯度不同源；继续使用会得到错误下降方向。
    // 正确做法是把该低秩项同时写进代价与 Jacobian（或并入 Pg 账本），
    // 在完成之前显式拒绝，避免静默产生错误解。
    void set_dyn_gauge_corr(py::array_t<double, py::array::c_style | py::array::forcecast> Ga,
                            py::array_t<double, py::array::c_style | py::array::forcecast> Gb,
                            py::array_t<double, py::array::c_style | py::array::forcecast> C) {
        throw std::runtime_error(
            "set_dyn_gauge_corr: 残差修正未进入 Jacobian（代价/梯度不同源），"
            "已禁用；请改用固定 EKF dyn 或把该低秩项并入 Pg 账本");
        if (L_ < 2)
            throw std::runtime_error("set_dyn_gauge_corr: 需 L>=2（没有 din 区间）");
        const int nkint = L_ - 1;
        const int n6 = n_ * 6;
        auto a = Ga.request(); auto b = Gb.request(); auto c = C.request();
        auto shape_k = [&](const py::buffer_info &bi, const char *name) -> int64_t {
            if (bi.ndim == 3) {
                if (bi.shape[0] != nkint && bi.shape[0] != L_)
                    throw std::runtime_error(std::string("set_dyn_gauge_corr: ") + name +
                                             " 首维需 (L-1) 或 L");
                if (bi.shape[1] != n6)
                    throw std::runtime_error(std::string("set_dyn_gauge_corr: ") + name +
                                             " 第二维需 6N");
                return bi.shape[2];
            }
            if (bi.ndim == 2) {
                if (nkint != 1)
                    throw std::runtime_error(std::string("set_dyn_gauge_corr: ") + name +
                                             " 为二维时只允许单区间 L=2");
                if (bi.shape[0] != n6)
                    throw std::runtime_error(std::string("set_dyn_gauge_corr: ") + name +
                                             " 首维需 6N");
                return bi.shape[1];
            }
            throw std::runtime_error(std::string("set_dyn_gauge_corr: ") + name +
                                     " 需二维或三维数组");
        };
        const int64_t ka = shape_k(a, "Ga");
        const int64_t kb = shape_k(b, "Gb");
        if (ka != kb || ka <= 0 || ka > 64)
            throw std::runtime_error("set_dyn_gauge_corr: Ga/Gb 的 k 必须一致且 1..64");
        const int k = (int)ka;
        if (c.ndim == 3) {
            if (c.shape[0] != nkint && c.shape[0] != L_)
                throw std::runtime_error("set_dyn_gauge_corr: C 首维需 (L-1) 或 L");
            if (c.shape[1] != k || c.shape[2] != k)
                throw std::runtime_error("set_dyn_gauge_corr: C 需 (...,k,k)");
        } else if (c.ndim == 2) {
            if (nkint != 1 || c.shape[0] != k || c.shape[1] != k)
                throw std::runtime_error("set_dyn_gauge_corr: 二维 C 需 (k,k) 且 L=2");
        } else {
            throw std::runtime_error("set_dyn_gauge_corr: C 需二维或三维数组");
        }
        std::vector<double> ga((size_t)nkint * n6 * k), gb((size_t)nkint * n6 * k),
                            cc((size_t)nkint * k * k);
        const bool ga3 = (a.ndim == 3), gb3 = (b.ndim == 3), c3 = (c.ndim == 3);
        // Ga/Gb 的首维若为 L（含两端时间基），只消费前 L−1 个区间。
        for (int iv = 0; iv < nkint; ++iv)
            for (int q = 0; q < n6 * k; ++q) {
                const size_t src = (ga3 ? (size_t)iv * n6 * k : 0) + (size_t)q;
                ga[(size_t)iv * n6 * k + q] = ((double *)a.ptr)[src];
                gb[(size_t)iv * n6 * k + q] = ((double *)b.ptr)[gb3 ? src : (size_t)q];
            }
        for (int iv = 0; iv < nkint; ++iv)
            for (int q = 0; q < k * k; ++q) {
                const size_t src = (c3 ? (size_t)iv * k * k : 0) + (size_t)q;
                cc[(size_t)iv * k * k + q] = ((double *)c.ptr)[src];
            }
        for (size_t q = 0; q < ga.size(); ++q)
            if (!std::isfinite(ga[q]) || !std::isfinite(gb[q]))
                throw std::runtime_error("set_dyn_gauge_corr: Ga/Gb 含非有限值");
        for (size_t q = 0; q < cc.size(); ++q)
            if (!std::isfinite(cc[q]))
                throw std::runtime_error("set_dyn_gauge_corr: C 含非有限值");

        free_dyn_gauge_dev_();
        dyn_gauge_ga_host_ = std::move(ga);
        dyn_gauge_gb_host_ = std::move(gb);
        dyn_gauge_c_host_ = std::move(cc);
        dyn_gauge_k_ = k;
        has_dyn_gauge_ = true;
        const size_t nGa = (size_t)nkint * n6 * k;
        const size_t nC = (size_t)nkint * k * k;
        d_gauge_Ga_ = mal(nGa);
        d_gauge_Gb_ = mal(nGa);
        d_gauge_C_ = mal(nC);
        d_gauge_corr_ = mal((size_t)nkint * n6);
        cudaMemcpy(d_gauge_Ga_, dyn_gauge_ga_host_.data(), nGa * sizeof(double),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_gauge_Gb_, dyn_gauge_gb_host_.data(), nGa * sizeof(double),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_gauge_C_, dyn_gauge_c_host_.data(), nC * sizeof(double),
                   cudaMemcpyHostToDevice);
        cudaMemset(d_gauge_corr_, 0, (size_t)nkint * n6 * sizeof(double));
        sync_dyn_gauge_ctx_();
        dyn_gauge_metric_ok_ = false;   // 惰性：dyn_gauge_diag 才计算 M/M'
    }
    void clear_dyn_gauge_corr() {
        free_dyn_gauge_dev_();
        has_dyn_gauge_ = false;
        dyn_gauge_k_ = 0;
        dyn_gauge_ga_host_.clear();
        dyn_gauge_gb_host_.clear();
        dyn_gauge_c_host_.clear();
        dyn_gauge_M_host_.clear();
        dyn_gauge_Mp_host_.clear();
        dyn_gauge_fro_host_.clear();
        dyn_gauge_metric_ok_ = false;
    }
    bool has_dyn_gauge_corr() const { return has_dyn_gauge_; }
    // 静态诊断：M=Gbᵀ T Ga、M'=Gbᵀ T' Ga、||Gb C Gaᵀ||_F（T 为逐星 Φ）。
    // C++ 不重算 C 所需的 M；这里仅在显式诊断时计算 M。
    void recompute_dyn_gauge_metric_() {
        dyn_gauge_M_host_.clear();
        dyn_gauge_Mp_host_.clear();
        dyn_gauge_fro_host_.clear();
        dyn_gauge_metric_ok_ = false;
        if (!has_dyn_gauge_ || !has_dyn_phi_ || L_ < 2) return;
        const int k = dyn_gauge_k_, n6 = n_ * 6, nkint = L_ - 1;
        dyn_gauge_M_host_.assign((size_t)nkint * k * k, 0.0);
        dyn_gauge_Mp_host_.assign((size_t)nkint * k * k, 0.0);
        dyn_gauge_fro_host_.assign((size_t)nkint, 0.0);
        for (int iv = 0; iv < nkint; ++iv) {
            Eigen::MatrixXd Ga(n6, k), Gb(n6, k), C(k, k);
            for (int a = 0; a < n6; ++a)
                for (int j = 0; j < k; ++j) {
                    Ga(a, j) = dyn_gauge_ga_host_[(size_t)iv * n6 * k + (size_t)a * k + j];
                    Gb(a, j) = dyn_gauge_gb_host_[(size_t)iv * n6 * k + (size_t)a * k + j];
                }
            for (int a = 0; a < k; ++a)
                for (int b = 0; b < k; ++b)
                    C(a, b) = dyn_gauge_c_host_[(size_t)iv * k * k + (size_t)a * k + b];
            Eigen::MatrixXd GtTG = Eigen::MatrixXd::Zero(k, k);
            for (int i = 0; i < n_; ++i) {
                Eigen::Matrix<double, 6, 6> Phi;
                for (int a = 0; a < 6; ++a)
                    for (int b = 0; b < 6; ++b)
                        Phi(a, b) = dyn_phi_host_[((size_t)iv * n_ + i) * 36 + a * 6 + b];
                const Eigen::MatrixXd gai = Ga.block(i * 6, 0, 6, k);
                const Eigen::MatrixXd gbi = Gb.block(i * 6, 0, 6, k);
                GtTG.noalias() += gbi.transpose() * Phi * gai;
            }
            const Eigen::MatrixXd GtGa = Ga.transpose() * Ga;
            const Eigen::MatrixXd GtGb = Gb.transpose() * Gb;
            const Eigen::MatrixXd M = GtTG;
            const Eigen::MatrixXd Mp = M + GtGb * C * GtGa;
            const Eigen::MatrixXd Mc = C.transpose() * GtGb * C;
            const double fro2 = (Mc.cwiseProduct(GtGa)).sum();
            for (int a = 0; a < k; ++a)
                for (int b = 0; b < k; ++b) {
                    dyn_gauge_M_host_[(size_t)iv * k * k + (size_t)a * k + b] = M(a, b);
                    dyn_gauge_Mp_host_[(size_t)iv * k * k + (size_t)a * k + b] = Mp(a, b);
                }
            dyn_gauge_fro_host_[iv] = std::sqrt(std::max(fro2, 0.0));
        }
        dyn_gauge_metric_ok_ = true;
    }
    // 诊断 dict：G_bᵀT'G_a（M/Mp）、||Gb C Gaᵀ||_F、当前 X 上 ||corr|| 与 g=Gaᵀ(X_a−x_ref)。
    py::dict dyn_gauge_diag() {
        if (!has_dyn_gauge_)
            throw std::runtime_error("dyn_gauge_diag: 未设置 set_dyn_gauge_corr");
        if (!has_dyn_phi_)
            throw std::runtime_error("dyn_gauge_diag: 需先 set_dyn_ekf/set_dyn_ekf_qeff");
        if (!dyn_gauge_metric_ok_) recompute_dyn_gauge_metric_();
        if (!dyn_gauge_metric_ok_ || dyn_gauge_M_host_.empty())
            throw std::runtime_error("dyn_gauge_diag: M/M' 诊断未就绪");
        const int k = dyn_gauge_k_, n6 = n_ * 6, nkint = L_ - 1;
        std::vector<double> g((size_t)nkint * k, std::nan("")),
                            cn((size_t)nkint, std::nan(""));
        if (has_dyn_xref_) {
            for (int iv = 0; iv < nkint; ++iv) {
                Eigen::MatrixXd Ga(n6, k), Gb(n6, k), C(k, k);
                for (int a = 0; a < n6; ++a)
                    for (int j = 0; j < k; ++j) {
                        Ga(a, j) = dyn_gauge_ga_host_[(size_t)iv * n6 * k + (size_t)a * k + j];
                        Gb(a, j) = dyn_gauge_gb_host_[(size_t)iv * n6 * k + (size_t)a * k + j];
                    }
                for (int a = 0; a < k; ++a)
                    for (int b = 0; b < k; ++b)
                        C(a, b) = dyn_gauge_c_host_[(size_t)iv * k * k + (size_t)a * k + b];
                Eigen::VectorXd dx(n6);
                for (int a = 0; a < n6; ++a)
                    dx(a) = hX_[(size_t)iv * n6 + a] - dyn_xref_host_[(size_t)iv * n6 + a];
                const Eigen::VectorXd gv = Ga.transpose() * dx;
                const Eigen::VectorXd corr = Gb * (C * gv);
                for (int j = 0; j < k; ++j) g[(size_t)iv * k + j] = gv(j);
                cn[iv] = corr.norm();
            }
        }
        py::array_t<double> M({(py::ssize_t)nkint, (py::ssize_t)k, (py::ssize_t)k});
        py::array_t<double> Mp({(py::ssize_t)nkint, (py::ssize_t)k, (py::ssize_t)k});
        py::array_t<double> fro((py::ssize_t)nkint), corr_norm((py::ssize_t)nkint);
        py::array_t<double> garr({(py::ssize_t)nkint, (py::ssize_t)k});
        std::copy(dyn_gauge_M_host_.begin(), dyn_gauge_M_host_.end(), M.mutable_data());
        std::copy(dyn_gauge_Mp_host_.begin(), dyn_gauge_Mp_host_.end(), Mp.mutable_data());
        std::copy(dyn_gauge_fro_host_.begin(), dyn_gauge_fro_host_.end(), fro.mutable_data());
        std::copy(cn.begin(), cn.end(), corr_norm.mutable_data());
        std::copy(g.begin(), g.end(), garr.mutable_data());
        py::dict d;
        d["k"] = k;
        d["intervals"] = nkint;
        d["M"] = M;
        d["Mprime"] = Mp;
        d["corr_fro"] = fro;
        d["corr_state_norm"] = corr_norm;
        d["g"] = garr;
        d["has_xref"] = has_dyn_xref_;
        return d;
    }

    // L=1 EKF 交接：初始化上一帧 P（否则 step 的 P_before 为 0，Jupd/T 全 NaN）。
    // 物理过程噪声 σ_a（m/s²）：Q_rv=diag(0.5σa dt², σa dt)，再用 J_q→rv 转 QOE。
    void set_Q_phys(double sigma_a) {
        std::vector<Vector6d> oe(n_), rv;
        std::vector<double> J;
        for (int i = 0; i < n_; ++i)
            for (int c = 0; c < 6; ++c)
                oe[i](c) = hX_[(size_t)i * 6 + c];
        noeOsc2rvJacBatch(oe, 16, J);
        const double dt = dt_;
        const double qp = std::pow(0.5 * sigma_a * dt * dt, 2);
        const double qv = std::pow(sigma_a * dt, 2);
        std::vector<double> Q((size_t)n_ * 36, 0.0);
        for (int i = 0; i < n_; ++i) {
            Eigen::Matrix<double, 6, 6> Ji;
            for (int a = 0; a < 6; ++a)
                for (int b = 0; b < 6; ++b) Ji(a, b) = J[(size_t)i * 36 + a * 6 + b];
            Eigen::Matrix<double, 6, 6> Qr = Eigen::Matrix<double, 6, 6>::Zero();
            for (int a = 0; a < 3; ++a) Qr(a, a) = qp;
            for (int a = 3; a < 6; ++a) Qr(a, a) = qv;
            Eigen::Matrix<double, 6, 6> Jinv = Ji.inverse();
            Eigen::Matrix<double, 6, 6> Qq = Jinv * Qr * Jinv.transpose();
            Qq = 0.5 * (Qq + Qq.transpose());
            for (int a = 0; a < 6; ++a)
                for (int b = 0; b < 6; ++b) Q[(size_t)i * 36 + a * 6 + b] = Qq(a, b);
        }
        auto qd = qinv_sqrt(Q);
        qphys_Q_host_ = Q;
        cudaMemcpy(d_qingt_, qd.data(), (size_t)n_ * 36 * sizeof(double), cudaMemcpyHostToDevice);
        qphys_sigma_ = sigma_a;
        // 完整 GVE 模式切换：不能只换 Q 而保留上一窗 EKF 的 dyn_sub/q_perstar/Phi/x_ref。
        apply_gve_mode_();
    }

    // GVE/EKF 模式审计：确认 q/phi/dyn_sub/x_ref 是否真的指向当前模式。
    py::dict dyn_mode_diag() {
        py::dict d;
        d["dyn_sub"] = dyn_sub_;
        d["wc_dyn_sub"] = wc_.dyn_sub;
        d["q_perstar"] = wc_.q_perstar;
        d["q_is_phys"] = (wc_.d_qingt == d_qingt_);
        d["q_is_ekf"] = (wc_.d_qingt == d_qingt_key_);
        d["phi_is_gve"] = (wc_.d_phi == d_phi_);
        d["phi_is_ekf"] = (wc_.d_phi == d_ekf_phi_);
        d["has_dyn_xref"] = has_dyn_xref_;
        d["dyn_xref_null"] = (wc_.d_dyn_xref == nullptr);
        d["dt"] = dt_;
        d["sigma_a"] = qphys_sigma_;
        d["ekf_qphys_valid"] = ekf_qphys_valid_;
        d["has_dyn_gauge"] = has_dyn_gauge_;
        const size_t nq = (size_t)n_ * 36;
        std::vector<double> qh(nq);
        cudaMemcpy(qh.data(), d_qingt_, nq * sizeof(double), cudaMemcpyDeviceToHost);
        double qwmin = std::numeric_limits<double>::infinity();
        double qwmax = -std::numeric_limits<double>::infinity();
        for (double v : qh) { qwmin = std::min(qwmin, v); qwmax = std::max(qwmax, v); }
        d["qwhite_min"] = qwmin;
        d["qwhite_max"] = qwmax;
        if ((size_t)qphys_Q_host_.size() == nq) {
            double qmin = std::numeric_limits<double>::infinity();
            double qmax = -std::numeric_limits<double>::infinity();
            double lmin = std::numeric_limits<double>::infinity();
            double lmax = -std::numeric_limits<double>::infinity();
            double qa = 0.0, qa_max = 0.0;
            for (int i = 0; i < n_; ++i) {
                Eigen::Matrix<double, 6, 6> Qm;
                for (int a = 0; a < 6; ++a)
                    for (int b = 0; b < 6; ++b) {
                        const double v = qphys_Q_host_[(size_t)i * 36 + a * 6 + b];
                        Qm(a, b) = v;
                        qmin = std::min(qmin, v); qmax = std::max(qmax, v);
                    }
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> es(Qm);
                const auto w = es.eigenvalues();
                lmin = std::min(lmin, w.minCoeff()); lmax = std::max(lmax, w.maxCoeff());
                qa += Qm(0, 0); qa_max = std::max(qa_max, Qm(0, 0));
            }
            d["Q_min"] = qmin;
            d["Q_max"] = qmax;
            d["Q_eig_min"] = lmin;
            d["Q_eig_max"] = lmax;
            d["Q_a_mean"] = qa / std::max(1, n_);
            d["Q_a_max"] = qa_max;
            d["sqrtQ_a_max"] = std::sqrt(std::max(0.0, qa_max));
        }
        return d;
    }

    // 物理 GVE Q（Python proc_Q_qoe 的 GVE 口径）直接上传，避免
    // set_Q_phys 里 J^{-1} Q_rv J^{-T} 在 QOE 奇异点产生非 PSD Q。
    void set_Q_phys_arr(py::array_t<double, py::array::c_style | py::array::forcecast> Q,
                        double sigma_a) {
        auto q = Q.request();
        if ((size_t)q.size != (size_t)n_ * 36)
            throw std::runtime_error("set_Q_phys_arr: Q 需 (n,6,6)");
        std::vector<double> qh((double *)q.ptr, (double *)q.ptr + (size_t)n_ * 36);
        const char *fs = std::getenv("KF_QPHYS_FLOOR");
        const double qfloor = (fs && *fs) ? std::atof(fs) : 1e-8;
        for (size_t i = 0; i < (size_t)n_; ++i) {
            Eigen::Matrix<double, 6, 6> Qm;
            for (int a = 0; a < 6; ++a)
                for (int b = 0; b < 6; ++b) Qm(a, b) = qh[i * 36 + a * 6 + b];
            Qm = 0.5 * (Qm + Qm.transpose());
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> es(Qm);
            Eigen::Matrix<double, 6, 1> w = es.eigenvalues();
            for (int a = 0; a < 6; ++a) w(a) = std::max(w(a), qfloor);
            Qm = es.eigenvectors() * w.asDiagonal() * es.eigenvectors().transpose();
            for (int a = 0; a < 6; ++a)
                for (int b = 0; b < 6; ++b) qh[i * 36 + a * 6 + b] = Qm(a, b);
        }
        auto qd = qinv_sqrt(qh);
        qphys_Q_host_ = qh;
        cudaMemcpy(d_qingt_, qd.data(), (size_t)n_ * 36 * sizeof(double), cudaMemcpyHostToDevice);
        qphys_sigma_ = sigma_a;
        apply_gve_mode_();
    }
    py::array_t<double> get_Q_phys() const {
        py::array_t<double> out({(py::ssize_t)n_, (py::ssize_t)6, (py::ssize_t)6});
        if ((size_t)qphys_Q_host_.size() == (size_t)n_ * 36)
            std::copy(qphys_Q_host_.begin(), qphys_Q_host_.end(), out.mutable_data());
        return out;
    }
    void apply_gve_mode_() {
        dyn_sub_ = std::max(1, dyn_sub_cfg_);
        wc_.dyn_sub = dyn_sub_;
        wc_.q_perstar = 1;
        wc_.d_qingt = d_qingt_;
        wc_.d_phi = d_phi_;
        wc_.d_dyn_xref = nullptr;
        has_dyn_xref_ = false;
        has_dyn_phi_ = false;
        ekf_qphys_valid_ = false;
        ekf_cache_valid_ = false;
        free_dyn_gauge_dev_();
        has_dyn_gauge_ = false;
        dyn_gauge_k_ = 0;
        dyn_gauge_metric_ok_ = false;
    }

    void set_P_init(py::array_t<double, py::array::c_style | py::array::forcecast> P) {
        auto a = P.request();
        if ((size_t)a.size != (size_t)n_ * 36)
            throw std::runtime_error("set_P_init: P 需 (n,6,6)");
        std::copy((double *)a.ptr, (double *)a.ptr + n_ * 36, Pm_host_.begin());
        cudaMemcpy(d_PmMarg_, a.ptr, (size_t)n_ * 36 * sizeof(double),
                   cudaMemcpyHostToDevice);
        // 冻结 L=1 EKF 的固定每星缩放 D=diag(√diag(P0))：核内全部协方差运算在
        // P_s=D⁻¹PD⁻¹ 空间进行（QOE 混合量纲 cond~1e14 会让原空间运算失效）。
        const double *Pp = (const double *)a.ptr;
        for (int i = 0; i < n_; ++i)
            for (int c = 0; c < 6; ++c)
                d_ekf_host_[(size_t)i * 6 + c] =
                    std::sqrt(std::max(Pp[(size_t)i * 36 + c * 6 + c], 1e-300));
        cudaMemcpy(d_ekf_scale_, d_ekf_host_.data(), (size_t)n_ * 6 * sizeof(double),
                   cudaMemcpyHostToDevice);
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
        cost_ref_ = hX_;   // 新的装配点：cost 投影基点与当前 X 同步
        has_x_ = true;
        // 兼容 set_prior_multi 先于 set_X 的调用顺序：若先验已入但还没有
        // x_first，则以紧随其后的第一次 set_X 作为 x_first。之后若 z 与它
        // 不一致，fej_check_frozen_ 会在 marginalize 前报 prior_center_not_first_x。
        if (has_prior_ && multi_prior_ && !fej_first_x_valid_ &&
            (int)fej_prior_z0_.size() == n_ * 6) {
            fej_first_x_ = hX_;
            fej_first_x_valid_ = true;
        }
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
        prior_gauge_projected_ = false;   // 新 prior 默认未投影；set_prior_projected 会再置位
        prior_proj_lowrank_ = false;      // 旧 A0/B0 随 prior 失效，需重新 project_prior_gauge
        prior_proj_external_ = false;
        prior_proj_A0_host_.clear();
        prior_proj_B0_host_.clear();
        prior_proj_S0_host_.clear();
        prior_proj_S0inv_host_.clear();
    }
    void set_prior(py::array_t<double, py::array::c_style | py::array::forcecast> Lp,
                   py::array_t<double, py::array::c_style | py::array::forcecast> z_prior) {
        set_prior_raw((double *)Lp.request().ptr, (double *)z_prior.request().ptr);
        multi_prior_ = false;
    }
    // 每节点 6×6 先验：Lp (L,n,6,6)、z_prior (L,n,6)。用于关键帧窗口把 EKF/上一窗
    // 的每帧信息量直接注入各节点；节点 k 的先验行写在该节点自己的 6N 列上。
    void set_prior_multi_raw(const double *Lp, const double *z_prior) {
        cudaMemcpy(Lp_all_.data_ptr<double>(), Lp, (size_t)L_ * n_ * 36 * sizeof(double),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(z_prior_all_.data_ptr<double>(), z_prior, (size_t)L_ * n_ * 6 * sizeof(double),
                   cudaMemcpyHostToDevice);
        Lp_all_host_.assign(Lp, Lp + (size_t)L_ * n_ * 36);
        z_prior_all_host_.assign(z_prior, z_prior + (size_t)L_ * n_ * 6);
        multi_prior_ = true; has_prior_ = true;
        prior_gauge_projected_ = false;
        prior_proj_lowrank_ = false;
        prior_proj_external_ = false;
        prior_proj_A0_host_.clear();
        prior_proj_B0_host_.clear();
        prior_proj_S0_host_.clear();
        prior_proj_S0inv_host_.clear();
        // D3/FEJ：set_prior_multi 标记本窗先验首次入窗。此刻的 X 就是 x_first；
        // 节点0 的 Lp/z 作为冻结快照，供 marginalize_* 检查先验没有被重线性化。
        fej_first_x_valid_ = has_x_ && (hX_.size() == (size_t)L_ * n_ * 6);
        if (fej_first_x_valid_)
            fej_first_x_ = hX_;
        else
            fej_first_x_.clear();
        fej_prior_z0_.assign(z_prior, z_prior + (size_t)n_ * 6);
        fej_prior_Lp0_.assign(Lp, Lp + (size_t)n_ * 36);
        // 多节点联合先验：快照所有 L 个节点，FEJ 检查不再只看 node0。
        fej_prior_z_all_.assign(z_prior, z_prior + (size_t)L_ * n_ * 6);
        fej_prior_Lp_all_.assign(Lp, Lp + (size_t)L_ * n_ * 36);
        fej_x_at_solve_valid_ = false;
        fej_x_at_solve_.clear();
        fej_g_frozen_ = has_gauge_;
        fej_G_hash_frozen_ = has_gauge_ ? gauge_G_hash_() : 0;
        fej_G_hash_at_solve_ = 0;
        fej_prior_center_dev_ = 0.0;
        fej_prior_Lp_dev_ = 0.0;
        fej_solve_start_dev_ = 0.0;
        fej_x_drift_ = 0.0;
        fej_last_marg_check_ok_ = false;
        fej_last_marg_check_reason_ = "not_checked";
    }
    void set_prior_multi(py::array_t<double, py::array::c_style | py::array::forcecast> Lp,
                         py::array_t<double, py::array::c_style | py::array::forcecast> z_prior) {
        auto a = Lp.request(); auto b = z_prior.request();
        if ((size_t)a.size != (size_t)L_ * n_ * 36 || (size_t)b.size != (size_t)L_ * n_ * 6)
            throw std::runtime_error("set_prior_multi: Lp 需 (L,n,6,6)，z_prior 需 (L,n,6)");
        set_prior_multi_raw((double *)a.ptr, (double *)b.ptr);
        // 每次新先验都先清空跨块；避免上一窗的块残留。
        clear_prior_cross_();
    }

    // 局部联合边界先验的跨变量块：val (B,6,6) 为**物理**信息 Λ_ij，
    // bi/bj (B,) 为变量编号（0..L*n-1，已按本窗节点重排）。跨窗只带物理量；
    // 进因子/RHS/cost 时用当前窗口 sc 转 scaled（S Λ S）。对角线不在
    // 这里，仍由 set_prior_multi 的 Lp 负责。矩阵按对称展开，块只存一次。
    void set_prior_cross(py::array_t<double, py::array::c_style | py::array::forcecast> val,
                         py::array_t<int, py::array::c_style | py::array::forcecast> bi,
                         py::array_t<int, py::array::c_style | py::array::forcecast> bj) {
        auto rv = val.request(); auto ri = bi.request(); auto rj = bj.request();
        const int64_t B = ri.size;
        if (rv.ndim != 3 || rv.shape[0] != B || rv.shape[1] != 6 || rv.shape[2] != 6 ||
            rj.size != B)
            throw std::runtime_error("set_prior_cross: val 需 (B,6,6)，bi/bj 需 (B,)");
        clear_prior_cross_();
        if (B == 0) return;
        const int nvar = L_ * n_;
        const double *pv = static_cast<const double *>(rv.ptr);
        const int *pi = static_cast<const int *>(ri.ptr);
        const int *pj = static_cast<const int *>(rj.ptr);
        cross_bi_.assign(pi, pi + B);
        cross_bj_.assign(pj, pj + B);
        // 物理坐标：跨块不随窗口 sc 变；进因子/代价时再乘当前 sc。
        cross_val_.assign(pv, pv + (size_t)B * 36);
        for (int64_t q = 0; q < B; ++q) {
            if (cross_bi_[q] < 0 || cross_bi_[q] >= nvar ||
                cross_bj_[q] < 0 || cross_bj_[q] >= nvar ||
                cross_bi_[q] == cross_bj_[q])
                throw std::runtime_error("set_prior_cross: bi/bj 越界或相同");
        }
        for (int64_t q = 0; q < B; ++q) {
            double *v = &cross_val_[(size_t)q * 36];
            // val 是 (i,j) 的上三角块，本身一般不对称；全矩阵的对称性由
            // build_prior_cross_csr_ 在 (j,i) 写入转置块保证。这里若把
            // 6×6 块强行对称化，会改变真实互信息并造成 Hessian/RHS 不同源。
            for (int a = 0; a < 36; ++a) {
                if (!std::isfinite(v[a]))
                    throw std::runtime_error("set_prior_cross: val 含非有限值");
            }
        }
        // 全局块对角占优保险：逐对 PSD 不足以保证带环的全局块稀疏先验
        // 半正定。用 Gershgorin 下界 s_g=0.9·min_i λmin(A_i)/max_i Σ_j||C_ij||
        // 统一收缩 cross，先保证 K⪰0，再做逐对细化。默认关闭：全局
        // Gershgorin 在弱对角先验下会把 cross 压到 1e-25 量级，等价于
        // 删除联合先验并导致窗口发散；需要时显式 KF_CROSS_GLOBAL=1。
        const char *cg_env = std::getenv("KF_CROSS_GLOBAL");
        const bool cross_global = (cg_env != nullptr) && (std::atoi(cg_env) != 0);
        if (cross_global && prior_ &&
            (int)Lp_all_host_.size() == L_ * n_ * 36) {
            std::vector<double> row_norm((size_t)L_ * n_, 0.0);
            double min_eig = std::numeric_limits<double>::infinity();
            for (int64_t q = 0; q < B; ++q) {
                const int vi = cross_bi_[q], vj = cross_bj_[q];
                Eigen::Matrix<double, 6, 6> Li;
                for (int a = 0; a < 36; ++a)
                    Li.data()[a] = Lp_all_host_[(size_t)vi * 36 + a];
                Eigen::Matrix<double, 6, 6> A = Li.transpose() * Li;
                A = 0.5 * (A + A.transpose());
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> ea(A);
                min_eig = std::min(min_eig, ea.eigenvalues().minCoeff());
                Eigen::Matrix<double, 6, 6> C;
                for (int a = 0; a < 36; ++a)
                    C.data()[a] = cross_val_[(size_t)q * 36 + a];
                const double nrm = C.norm();
                row_norm[(size_t)vi] += nrm;
                row_norm[(size_t)vj] += nrm;
            }
            double max_row = 0.0;
            for (double v : row_norm) max_row = std::max(max_row, v);
            if (max_row > 0.0) {
                double sg = (std::isfinite(min_eig) && min_eig > 0.0)
                                ? 0.9 * min_eig / max_row : 0.0;
                sg = std::min(1.0, sg);
                if (sg < 1.0) {
                    for (double &v : cross_val_) v *= sg;
                    fprintf(stderr,
                            "[kf/cross] 全局占优收缩 s_g=%.3e "
                            "(min_eig=%.3e max_row=%.3e)\n", sg, min_eig, max_row);
                }
            }
        }
        // 逐对 PSD 收缩：跨块 C 与两端对角 A=Lp_iᵀLp_i、B=Lp_jᵀLp_j
        // 必须满足 B - Cᵀ A⁻¹ C ⪰ 0。两端对角来自不同小球时直接拼可能
        // 不定；按 ρ=λmax(B^-1/2 CᵀA⁻¹C B^-1/2) 收缩 C，保证每对 PSD。
        int n_scaled = 0;
        if (prior_ && (int)Lp_all_host_.size() == L_ * n_ * 36) {
            for (int64_t q = 0; q < B; ++q) {
                const int vi = cross_bi_[q], vj = cross_bj_[q];
                Eigen::Matrix<double, 6, 6> Li, Lj;
                for (int a = 0; a < 36; ++a) {
                    Li.data()[a] = Lp_all_host_[(size_t)vi * 36 + a];
                    Lj.data()[a] = Lp_all_host_[(size_t)vj * 36 + a];
                }
                Eigen::Matrix<double, 6, 6> A = Li.transpose() * Li;
                Eigen::Matrix<double, 6, 6> Bm = Lj.transpose() * Lj;
                Eigen::Matrix<double, 6, 6> C;
                for (int a = 0; a < 36; ++a)
                    C.data()[a] = cross_val_[(size_t)q * 36 + a];
                auto pinv_psd = [](const Eigen::Matrix<double, 6, 6> &M) {
                    Eigen::Matrix<double, 6, 6> S = 0.5 * (M + M.transpose());
                    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> es(S);
                    const auto &w = es.eigenvalues();
                    const auto &V = es.eigenvectors();
                    const double fl = std::max(std::max(0.0, w.maxCoeff()) * 1e-12, 1e-300);
                    return std::make_pair(
                        Eigen::Matrix<double, 6, 6>(V * w.cwiseMax(fl).cwiseInverse()
                                                       .asDiagonal() * V.transpose()),
                        Eigen::Matrix<double, 6, 6>(V * w.cwiseMax(fl).cwiseSqrt()
                                                       .asDiagonal() * V.transpose()));
                };
                auto [Ainv, Ainv_sqrt] = pinv_psd(A);
                auto [Binv, Binv_sqrt] = pinv_psd(Bm);
                Eigen::Matrix<double, 6, 6> R = Binv_sqrt * C.transpose() * Ainv * C *
                                                Binv_sqrt;
                R = 0.5 * (R + R.transpose());
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> er(R);
                const double rho = std::max(0.0, er.eigenvalues().maxCoeff());
                if (rho > 1.0) {
                    const double s = 0.999 / std::sqrt(rho);
                    for (int a = 0; a < 36; ++a)
                        cross_val_[(size_t)q * 36 + a] *= s;
                    ++n_scaled;
                }
            }
            if (n_scaled)
                fprintf(stderr, "[kf/cross] 逐对 PSD 收缩 %d/%lld 块\n",
                        n_scaled, (long long)B);
        }
        bool any_nonzero = false;
        for (double v : cross_val_)
            if (v != 0.0) { any_nonzero = true; break; }
        if (!any_nonzero) { clear_prior_cross_(); return; }
        build_prior_cross_csr_();
        cross_active_ = true;
        fej_cross_val_ = cross_val_;
        fej_cross_hash_ = fnv1a64(cross_val_.data(), cross_val_.size() * sizeof(double));
    }

    void clear_prior_cross_() {
        cross_bi_.clear(); cross_bj_.clear(); cross_val_.clear();
        cross_rp_.clear(); cross_ci_.clear(); cross_vals_.clear();
        cross_entry_blk_.clear(); cross_entry_ab_.clear(); cross_entry_row_.clear();
        cross_active_ = false;
        fej_cross_val_.clear();
        fej_cross_hash_ = 0;
        cross_rp_t_ = torch::Tensor();
        cross_ci_t_ = torch::Tensor();
        cross_vals_t_ = torch::Tensor();
    }

    void build_prior_cross_csr_() {
        const int64_t ncol = S_.ncol;
        struct Ent { int col; int blk; int ab; };
        std::vector<std::vector<Ent>> rows((size_t)ncol);
        for (size_t q = 0; q < cross_bi_.size(); ++q) {
            const int vi = cross_bi_[q], vj = cross_bj_[q];
            for (int a = 0; a < 6; ++a)
                for (int b = 0; b < 6; ++b) {
                    // val 只存 (i,j) 上三角块 C；全矩阵对称展开时
                    // (j,i) 块必须是 C^T，不能把 C 原样再写一遍。
                    rows[(size_t)vi * 6 + a].push_back({vj * 6 + b, (int)q, a * 6 + b});
                    rows[(size_t)vj * 6 + b].push_back({vi * 6 + a, (int)q, b * 6 + a});
                }
        }
        cross_rp_.assign(1, 0);
        cross_ci_.clear(); cross_entry_blk_.clear();
        cross_entry_ab_.clear(); cross_entry_row_.clear();
        for (int64_t r = 0; r < ncol; ++r) {
            auto &v = rows[(size_t)r];
            std::sort(v.begin(), v.end(),
                      [](const Ent &a, const Ent &b) { return a.col < b.col; });
            for (const auto &e : v) {
                cross_ci_.push_back(e.col);
                cross_entry_blk_.push_back(e.blk);
                cross_entry_ab_.push_back(e.ab);
                cross_entry_row_.push_back((int)r);
            }
            cross_rp_.push_back((int64_t)cross_ci_.size());
        }
        auto optsI = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
        auto optsIcpu = torch::TensorOptions().dtype(torch::kInt64);
        cross_rp_t_ = torch::from_blob(cross_rp_.data(),
                                       {(py::ssize_t)cross_rp_.size()}, optsIcpu)
                          .clone().to(optsI);
        cross_ci_t_ = torch::from_blob(cross_ci_.data(),
                                       {(py::ssize_t)cross_ci_.size()}, optsIcpu)
                          .clone().to(optsI);
    }

    // 用当前窗口的 sc 把物理跨块 Λ_ij 变成 scaled 矩阵 D Λ_ij D。
    void cross_scaled_vals_(const double *sc, std::vector<double> &vals) const {
        vals.resize(cross_ci_.size());
        for (size_t e = 0; e < cross_ci_.size(); ++e) {
            const int r = cross_entry_row_[e], c = cross_ci_[e];
            const int q = cross_entry_blk_[e], ab = cross_entry_ab_[e];
            vals[e] = sc[r] * cross_val_[(size_t)q * 36 + ab] * sc[c];
        }
    }

    void add_cross_to_nd_(NumericDecomposition &nd) {
        if (!cross_active_ || cross_ci_.empty()) return;
        auto sc_cpu = mtm_scl_.to(torch::kCPU).contiguous();
        std::vector<double> vals;
        cross_scaled_vals_(sc_cpu.data_ptr<double>(), vals);
        auto vv = torch::from_blob(vals.data(), {(py::ssize_t)vals.size()},
                                   torch::TensorOptions().dtype(torch::kFloat64))
                      .clone().to(torch::kCUDA);
        nd.add_M(vv.view({1, (int64_t)vals.size()}), cross_rp_t_, cross_ci_t_);
    }
    void add_cross_to_nd_cpu_(NumericDecomposition &nd) const {
        if (!cross_active_ || cross_ci_.empty()) return;
        auto sc_cpu = mtm_scl_.to(torch::kCPU).contiguous();
        std::vector<double> vals;
        cross_scaled_vals_(sc_cpu.data_ptr<double>(), vals);
        auto optsI = torch::TensorOptions().dtype(torch::kInt64);
        auto optsD = torch::TensorOptions().dtype(torch::kFloat64);
        auto rp = torch::from_blob((void *)cross_rp_.data(),
                                   {(py::ssize_t)cross_rp_.size()}, optsI).clone();
        auto ci = torch::from_blob((void *)cross_ci_.data(),
                                   {(py::ssize_t)cross_ci_.size()}, optsI).clone();
        auto vv = torch::from_blob(vals.data(), {(py::ssize_t)vals.size()}, optsD).clone();
        nd.add_M(vv.view({1, (int64_t)vals.size()}), rp, ci);
    }

    // 跨块先验的 RHS（scaled）：行 i 加 sc_i Λ_ij (z_j-x_j)，
    // 行 j 加 sc_j Λ_ij^T (z_i-x_i)。
    void add_cross_rhs_(const std::vector<double> &sc, std::vector<double> &rhs) const {
        if (!cross_active_ || (int)sc.size() != L_ * n_ * 6) return;
        rhs.assign((size_t)S_.ncol, 0.0);
        for (size_t q = 0; q < cross_bi_.size(); ++q) {
            const int vi = cross_bi_[q], vj = cross_bj_[q];
            const double *v = &cross_val_[q * 36];
            double ei[6], ej[6];
            for (int a = 0; a < 6; ++a) {
                const int64_t gi = (int64_t)vi * 6 + a, gj = (int64_t)vj * 6 + a;
                ei[a] = z_prior_all_host_[(size_t)gi] - hX_[(size_t)gi];
                ej[a] = z_prior_all_host_[(size_t)gj] - hX_[(size_t)gj];
            }
            for (int a = 0; a < 6; ++a) {
                double si = 0.0;
                for (int b = 0; b < 6; ++b) si += v[a * 6 + b] * ej[b];
                rhs[(size_t)vi * 6 + a] += sc[(size_t)(vi * 6 + a)] * si;
            }
            for (int b = 0; b < 6; ++b) {
                double sj = 0.0;
                for (int a = 0; a < 6; ++a) sj += v[a * 6 + b] * ei[a];
                rhs[(size_t)vj * 6 + b] += sc[(size_t)(vj * 6 + b)] * sj;
            }
        }
    }

    // 跨块先验代价（物理残差）：每块只存一次，故乘 2。
    double cross_cost_() const {
        if (!cross_active_) return 0.0;
        double c = 0.0;
        for (size_t q = 0; q < cross_bi_.size(); ++q) {
            const int vi = cross_bi_[q], vj = cross_bj_[q];
            const double *v = &cross_val_[q * 36];
            double ei[6], ej[6];
            for (int a = 0; a < 6; ++a) {
                const int64_t gi = (int64_t)vi * 6 + a, gj = (int64_t)vj * 6 + a;
                ei[a] = z_prior_all_host_[(size_t)gi] - hX_[(size_t)gi];
                ej[a] = z_prior_all_host_[(size_t)gj] - hX_[(size_t)gj];
            }
            double t = 0.0;
            for (int a = 0; a < 6; ++a)
                for (int b = 0; b < 6; ++b) t += ei[a] * v[a * 6 + b] * ej[b];
            c += 2.0 * t;
        }
        return c;
    }

    // 装配 A_val/b（ISL/GTS/dyn + 先验行）；返回 ISL 残差平方和。
    double assemble() {
        double res2 = qoejopt::winAssembleDevice(wc_, hX_.data(), hRv_.data(), hAp_.data(),
                                                 A_val_.data_ptr<double>(), b_.data_ptr<double>());
        if (prior_ && has_prior_ && multi_prior_) {
            // 每节点先验：节点 k 的 6N 行写在 (off_damp + k*n*36, base_b + k*n*6)。
            // prior_node_ >= 0 时结构里只建了该节点的 prior 行。
            const int k0 = (prior_node_ >= 0) ? prior_node_ : 0;
            const int k1 = (prior_node_ >= 0) ? prior_node_ + 1 : L_;
            for (int k = k0; k < k1; ++k)
                qoejopt::winPriorAsmDevice(
                    wc_.d_X, Lp_all_.data_ptr<double>() + (size_t)k * n_ * 36,
                    z_prior_all_.data_ptr<double>() + (size_t)k * n_ * 6, n_, k,
                    S_.off_damp + (int64_t)k * n_ * 36,
                    S_.nISL + S_.nGTS + S_.nDYN + (int64_t)k * n_ * 6,
                    A_val_.data_ptr<double>(), b_.data_ptr<double>());
        } else if (prior_ && has_prior_)
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
        // 步骤 5：gauge 开启时从原始 GTS 行累加 Λ_g 进 Pg；本地 A 行保持原始 v，
        // 窗口 Woodbury 的 (I-Pi) 自动给出 J_perp。每次 assemble 由 CUDA 重写
        // 原始 GTS 行后重算 Λ_g，天然幂等，不会像“先改行再缓存”那样跨迭代变陈旧。
        apply_gts_gauge_split_();
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

    // 全代价（ISL+GTS+dyn+prior）：供线搜索使用；prior 只计 cost_node 一个节点。
    // 关键帧路径 Lp_all 只有 node0 非零；若多节点 prior 都非零，本核不计其余节点。
    //
    // D7 route (A)：c(x)=Σρ(Px)^2+(G^T x-z)^T Pg_eff^-1 (G^T x-z)，
    // P=I-Pi=I-G S^-1 G^T。在临时 device 缓冲上算 Px 的几何（不改 wc_.d_X），
    // gauge 项用 Pg_eff^-1；Step-4 的 -e·de 保持原口径。
    double total_cost() {
        geo_all();
        const bool prior_on = prior_ && has_prior_;
        // 单节点 prior 在 Lp_/z_prior_；set_prior_multi 才用 Lp_all_/z_prior_all_。
        const double *lp = prior_on ? (multi_prior_ ? Lp_all_.data_ptr<double>()
                                                   : Lp_.data_ptr<double>()) : nullptr;
        const double *zp = prior_on ? (multi_prior_ ? z_prior_all_.data_ptr<double>()
                                                   : z_prior_.data_ptr<double>()) : nullptr;
        // 非 multi 的 Lp_/z_prior_ 只有 node0，必须传 0；multi 时按 prior_node_ 选。
        const int cost_node = (multi_prior_ && prior_node_ > 0) ? prior_node_ : 0;
        const int k = gauge_k_;
        const int64_t nc = (int64_t)S_.ncol;
        // D7 cost 参考点：solve_batch 入口的装配/线性化点（线搜索内不更新）。
        // 尺寸异常时回退到当前 X，避免未初始化缓冲参与投影。
        const double *xref =
            ((int64_t)cost_ref_.size() == nc) ? cost_ref_.data() : hX_.data();
        double ref_sq = 0.0;
        for (int64_t a = 0; a < nc; ++a) ref_sq += xref[a] * xref[a];
        const double ref_norm = std::sqrt(ref_sq);

        const bool gauge_eff_ok =
            has_gauge_ && k > 0 &&
            (int)Pg_eff_inv_host_.size() == k * k &&
            (int)zg_host_.size() == k &&
            (int)Qg_host_.size() == (int)(nc * k) &&
            d_Xcost_ != nullptr && d_rvCost_ != nullptr && d_ApCost_ != nullptr;
        const bool gauge_pg_ok =
            has_gauge_ && k > 0 &&
            (int)Pg_inv_host_.size() == k * k &&
            (int)zg_host_.size() == k &&
            (int)Qg_host_.size() == (int)(nc * k);

        bool projected = false;
        double px_dev = 0.0, gxz = 0.0, px_gxz = 0.0;
        Eigen::MatrixXd Gmat, Qg;
        Eigen::VectorXd gxl(k);
        if (gauge_eff_ok) {
            Gmat.resize(nc, k);
            for (int64_t a = 0; a < nc; ++a)
                for (int j = 0; j < k; ++j)
                    Gmat(a, j) = Qg_host_[(size_t)a * k + j];
            Eigen::MatrixXd Sg = Gmat.transpose() * Gmat;
            Sg = 0.5 * (Sg + Sg.transpose());
            const double epsg = gauge_eps_from_gram(Sg);
            const Eigen::MatrixXd S =
                Sg + epsg * Eigen::MatrixXd::Identity(k, k);
            Eigen::LLT<Eigen::MatrixXd> llt_s(S);
            if (llt_s.info() == Eigen::Success) {
                const Eigen::MatrixXd Sinv =
                    llt_s.solve(Eigen::MatrixXd::Identity(k, k));
                for (int j = 0; j < k; ++j) {
                    double s = 0.0;
                    for (int64_t a = 0; a < nc; ++a)
                        s += Gmat(a, j) * hX_[(size_t)a];
                    gxl(j) = s;
                }
                // 参考点投影：d=x-ref，xp=ref+P d，P=I-G S^-1 G^T。
                // 不投影绝对 x；δ=0（x=ref）时 xp=ref，cost 回到原始残差尺度。
                Eigen::VectorXd dref(nc);
                for (int64_t a = 0; a < nc; ++a)
                    dref(a) = hX_[(size_t)a] - xref[a];
                const Eigen::VectorXd pi_d =
                    Gmat * (Sinv * (Gmat.transpose() * dref));
                std::vector<double> xp((size_t)(nc), 0.0);
                double px_sq = 0.0;
                for (int64_t a = 0; a < nc; ++a) {
                    xp[(size_t)a] = xref[a] + dref(a) - pi_d(a);
                    const double dx = xp[(size_t)a] - hX_[(size_t)a];
                    px_sq += dx * dx;
                }
                px_dev = std::sqrt(px_sq);
                cudaMemcpy(d_Xcost_, xp.data(), (size_t)nc * sizeof(double),
                           cudaMemcpyHostToDevice);
                qoejopt::oeRvJacDevice(d_Xcost_, (int)(nc / 6), d_rvCost_, d_ApCost_);
                Eigen::VectorXd gxl_p(k);
                for (int j = 0; j < k; ++j) {
                    double s = 0.0;
                    for (int64_t a = 0; a < nc; ++a)
                        s += Gmat(a, j) * xp[(size_t)a];
                    gxl_p(j) = s;
                }
                Eigen::VectorXd eg(k);
                for (int j = 0; j < k; ++j)
                    eg(j) = gxl(j) - zg_host_[(size_t)j];
                gxz = eg.norm();
                px_gxz = (gxl_p - Eigen::Map<const Eigen::VectorXd>(
                                       zg_host_.data(), k)).norm();
                Qg.resize(k, k);
                for (int a = 0; a < k; ++a)
                    for (int b = 0; b < k; ++b)
                        Qg(a, b) = Pg_eff_inv_host_[(size_t)a * k + b];
                Qg = 0.5 * (Qg + Qg.transpose());
                projected = true;
            }
        }

        double geom = 0.0;
        // 多节点先验：CUDA 核只支持单 prior_node，这里先只算几何，再在宿主侧
        // 按 assemble() 的同一 Lp/z/行序累加全部节点的 prior 残差。单节点路径
        // 保持原 kernel 口径不变。
        const int kernel_prior_on =
            (prior_on && (!multi_prior_ || prior_node_ >= 0)) ? 1 : 0;
        if (projected) {
            qoejopt::WinCtx cc = wc_;
            cc.d_X = d_Xcost_;
            cc.d_rv = d_rvCost_;
            cc.d_Ap = d_ApCost_;
            geom = qoejopt::winTotalCostDevice(cc, lp, zp, kernel_prior_on, cost_node);
        } else {
            // 无 gauge / Pg_eff 未就绪：保持原口径（不投影）。
            cudaMemcpy(wc_.d_X, hX_.data(), (size_t)L_ * n_ * 6 * sizeof(double),
                       cudaMemcpyHostToDevice);
            cudaMemcpy(wc_.d_rv, hRv_.data(), (size_t)L_ * n_ * 6 * sizeof(double),
                       cudaMemcpyHostToDevice);
            cudaMemcpy(wc_.d_Ap, hAp_.data(), (size_t)L_ * n_ * 18 * sizeof(double),
                       cudaMemcpyHostToDevice);
            geom = qoejopt::winTotalCostDevice(wc_, lp, zp, kernel_prior_on, cost_node);
        }
        if (prior_on && multi_prior_ && prior_node_ < 0 &&
            (int)Lp_all_host_.size() == L_ * n_ * 36 &&
            (int)z_prior_all_host_.size() == L_ * n_ * 6) {
            // 与 winPriorRes2Kernel 同口径：b_a = Σ_j F[a,j] (z_j-x_j)，cost=Σ b_a²。
            double pc = 0.0;
            for (int kk = 0; kk < L_; ++kk)
                for (int ii = 0; ii < n_; ++ii) {
                    const double *F = &Lp_all_host_[((size_t)kk * n_ + ii) * 36];
                    const double *zp_i = &z_prior_all_host_[((size_t)kk * n_ + ii) * 6];
                    const double *x_i = &hX_[((size_t)kk * n_ + ii) * 6];
                    double e[6];
                    for (int c = 0; c < 6; ++c) e[c] = zp_i[c] - x_i[c];
                    for (int a = 0; a < 6; ++a) {
                        double s = 0.0;
                        for (int j = 0; j < 6; ++j) s += F[a * 6 + j] * e[j];
                        pc += s * s;
                    }
                }
            geom += pc;
        }
        // 局部联合边界先验的跨变量块代价（与 add_M/RHS 同一 scaled 口径）。
        if (cross_active_) geom += cross_cost_();
        double cost = geom;
        double prior_corr = 0.0;
        bool prior_corr_applied = false;
        // winTotalCostDevice 返回 Σ b²（各核都不带 0.5，见 gve_step.cu 的
        // win*Res2Kernel），而 GN 法方程对应 0.5·Σ b² 口径。线搜索只比较同一
        // 口径的代价，因此附加项必须同样乘 2：
        //   步骤 4：物理 prior 行是未投影的 Λ0；投影后代价减 e^T ΔΛ e
        //           （不是 0.5 e^T ΔΛ e，否则相对权重会差一倍）。
        if (prior_on && prior_proj_lowrank_ && has_gauge_) {
            Eigen::VectorXd e, de;
            if (prior_projection_vectors_(e, de)) {
                prior_corr = -e.dot(de);
                cost += prior_corr;
                prior_corr_applied = true;
            }
        }
        // 步骤 3/5：gauge 先验 (G^T x-z)^T Pg_eff^-1 (G^T x-z) 的 2 倍口径。
        // (A) 下几何残差已在 Px 上算，GTS 的 G 分量被 P 剥掉，再由 Pg_eff^-1
        // 加回（Λ_g 只进一次）；w=Pg_eff^-1(z-G^T x) 与 solve_batch 的 RHS 同源。
        double gauge_cost = 0.0;
        bool used_pg_eff = false, used_pg_inv = false;
        if (projected) {
            Eigen::VectorXd eg(k);
            for (int j = 0; j < k; ++j)
                eg(j) = gxl(j) - zg_host_[(size_t)j];
            gauge_cost = eg.dot(Qg * eg);
            cost += gauge_cost;
            used_pg_eff = true;
        } else if (gauge_pg_ok) {
            // 显式 fallback：Pg_eff 未就绪时保持旧口径（不投影 + Pg^-1）。
            Eigen::MatrixXd Q(k, k);
            for (int a = 0; a < k; ++a)
                for (int b = 0; b < k; ++b)
                    Q(a, b) = Pg_inv_host_[(size_t)a * k + b];
            Q = 0.5 * (Q + Q.transpose());
            Eigen::VectorXd eg(k);
            for (int j = 0; j < k; ++j) {
                double s = 0.0;
                for (int64_t a = 0; a < nc; ++a)
                    s += Qg_host_[(size_t)a * k + j] * hX_[(size_t)a];
                eg(j) = s - zg_host_[(size_t)j];
            }
            gxz = eg.norm();
            px_gxz = gxz;   // fallback 不投影：Px=x
            gauge_cost = eg.dot(Q * eg);
            cost += gauge_cost;
            used_pg_inv = true;
        }
        // 代价分解诊断（cost_gauge_diag 读取；不改 solve 状态）。
        cost_last_total_ = cost;
        cost_last_geom_ = geom;
        cost_last_gauge_ = gauge_cost;
        cost_last_prior_proj_ = prior_corr;
        cost_last_prior_proj_applied_ = prior_corr_applied;
        cost_last_px_dev_ = px_dev;
        cost_last_gxz_ = gxz;
        cost_last_pxgxz_ = px_gxz;
        cost_last_cost_ref_norm_ = ref_norm;
        cost_last_projected_ = projected;
        cost_last_pg_eff_used_ = used_pg_eff;
        cost_last_pg_inv_used_ = used_pg_inv;
        return cost;
    }
    // D7 诊断：最近一次 total_cost 的分解与投影口径。
    py::dict cost_gauge_diag() {
        total_cost();
        py::dict d;
        d["total"] = cost_last_total_;
        d["geom"] = cost_last_geom_;
        d["gauge"] = cost_last_gauge_;
        d["prior_proj"] = cost_last_prior_proj_;
        d["prior_proj_applied"] = cost_last_prior_proj_applied_;
        d["Px_minus_x_norm"] = cost_last_px_dev_;
        d["cost_ref_norm"] = cost_last_cost_ref_norm_;
        d["Gt_x_minus_z_norm"] = cost_last_gxz_;
        d["Gt_Px_minus_z_norm"] = cost_last_pxgxz_;
        d["uses_Pg_eff_inv"] = cost_last_pg_eff_used_;
        d["uses_Pg_inv"] = cost_last_pg_inv_used_;
        d["pg_used"] = cost_last_pg_eff_used_ ? "Pg_eff_inv"
                       : (cost_last_pg_inv_used_ ? "Pg_inv" : "none");
        d["d7_cost_projected"] = cost_last_projected_;
        d["has_gauge"] = has_gauge_;
        d["k"] = gauge_k_;
        if (has_gauge_ && gauge_k_ > 0) {
            auto trace_of = [](const std::vector<double> &v, int k) {
                double s = 0.0;
                for (int i = 0; i < k && (int)v.size() >= k * k; ++i)
                    s += v[(size_t)i * k + i];
                return s;
            };
            auto norm_of = [](const std::vector<double> &v) {
                double s = 0.0;
                for (double x : v) s += x * x;
                return std::sqrt(s);
            };
            d["Pg_eff_trace"] = trace_of(Pg_eff_host_, gauge_k_);
            d["Pg_trace"] = trace_of(Pg_host_, gauge_k_);
            d["Pg_eff_inv_norm"] = norm_of(Pg_eff_inv_host_);
            d["Pg_inv_norm"] = norm_of(Pg_inv_host_);
        }
        return d;
    }
    // pybind 诊断入口：在**当前 X** 上返回 total_cost()，用于有限差分/梯度一致性检查。
    // 不改变任何状态、不触发 solve；只刷新几何与代价（与线搜索同一口径）。
    double total_cost_host() { return total_cost(); }
    // D7 诊断：当前 cost 参考点 (L,n,6)，即最近一次 solve_batch 入口的装配点。
    py::array_t<double> cost_ref_host() const {
        py::array_t<double> out({L_, n_, 6});
        double *p = out.mutable_data();
        const size_t nval = (size_t)L_ * n_ * 6;
        if (cost_ref_.size() == nval)
            std::copy(cost_ref_.begin(), cost_ref_.end(), p);
        else
            std::copy(hX_.begin(), hX_.begin() + nval, p);
        return out;
    }

    // 固定关键帧 batch：不 push 新帧、不左移，只对当前 X(L,n,6) 做 GN。
    // 关键帧窗口用：先 set_X/set_obs/set_prior_multi，再 solve_batch；随后 marginalize_first。
    void solve_batch(int steps) {
        const bool swdbg = getenv("FLS_SWEEP_DBG") || getenv("FLS_SWEEPDBG");
        cost_ref_ = hX_;   // D7：本窗装配/线性化参考点（线搜索内不再更新）
        last_solve_iters_.clear();
        last_solve_valid_ = false;
        // D3/FEJ：记录首次求解起点。marginalize_* 会检查该点等于 x_first，
        // 且 G 从此刻到边缘化未变；x 之后的漂移不当作 FEJ 违规。
        if (!fej_x_at_solve_valid_ && has_x_ && (int)hX_.size() == L_ * n_ * 6) {
            fej_x_at_solve_ = hX_;
            fej_x_at_solve_valid_ = true;
            fej_G_hash_at_solve_ = gauge_G_hash_();
        }
        geo_all();
        assemble();
        if (getenv("FLS_SWEEP_BENCH")) sweep_bench_();
        double res2 = total_cost();                    // 全代价线搜索：ISL+GTS+dyn+prior
        int gn_it = 0;
        for (int it = 0; it < std::max(1, steps); ++it) {
            ++gn_it;
            const double cost_before = res2;
            auto mtm = scaled_AtA_Atb();
            torch::Tensor dcpu;
            // gauge 开启时强制全窗因子：逐帧 sweep 的因子只覆盖单帧 6N，
            // 无法承载跨帧的 G_win 投影与 rank-2k Woodbury；gauge 是可选路径，
            // 这里用一次全窗分解保证与 doc 的 Λ_win 数学等价。
            if (sweep_ && !has_gauge_ && !cross_active_) {
                const int64_t nc = (int64_t)n_ * 6;
                const bool sw_dbg = swdbg;
                // FLS_SWEEP_DIAG=1：旧逐星对角近似（S^{-1}_{ii}≈S_{ii}^{-1}，且耦合反向），
                // 仅保留作对照；默认走精确块三对角消去。
                // 固定 EKF dyn（q_perstar=2/dyn_sub=0）下该近似不可用：每星 6×6 对角块
                // S_ii 的软方向能量只来自 ISL/GTS（相对 dyn 项被列均衡压到 ~1e-18 量级），
                // S_ii^{-1} 量级爆炸使 −CᵀS_ii^{-1}C 远超真实 Schur 修正，下一帧 Hessian
                // 失去正定 → 因子的解非有限 → 线搜索 4 次全拒（last_accepted=0、dev=0、
                // ISL 停在窗入口值）。这里直接改走精确 Schur（同一份 A 数据，无额外代价）。
                bool sw_diag = getenv("FLS_SWEEP_DIAG") != nullptr;
                const bool ekf_dyn = (dyn_sub_ == 0) || (wc_.q_perstar == 2);
                if (sw_diag && ekf_dyn) {
                    static bool warned = false;
                    if (!warned) {
                        warned = true;
                        fprintf(stderr,
                                "[sweep/diag] 固定 EKF dyn 布局下逐星对角近似不稳定（实测解 NaN），"
                                "自动改用精确 Schur（FLS_SWEEP_DIAG 忽略）\n");
                    }
                    sw_diag = false;
                }
                auto optsD = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA);
                auto _sw_now = [] {
                    return std::chrono::duration<double>(
                               std::chrono::steady_clock::now().time_since_epoch()).count();
                };
                const double t_sw0 = _sw_now();
                std::vector<NumericDecomposition> nds;
                nds.reserve(L_);
                // C_m = H_{m,m+1} = A_mᵀ A_{m+1}（逐星 6×6，来自 EKF dyn 行）。
                torch::Tensor Cs;
                if (L_ > 1) Cs = torch::zeros({L_ - 1, (int64_t)n_, 6, 6}, optsD);
                for (int m = 0; m < L_; ++m) {
                    double _ta = _sw_now(), _tb;
                    gather_frame_(m, mtm.first);
                    _tb = _sw_now(); const double t_gath = _tb - _ta; _ta = _tb;
                    eta_t_.zero_();
                    eta_t_.index_add_(0, AciF_t_, Af_t_ * bf_t_.index_select(0, rnzF_t_));
                    if (m > 0) eta_t_.sub_(gam_t_.view({nc}));   // −C_{m-1}ᵀ y_{m-1}
                    _tb = _sw_now(); const double t_eta = _tb - _ta; _ta = _tb;
                    auto nd = symF_->createNumericDecomposition(1);
                    _tb = _sw_now(); const double t_cre = _tb - _ta; _ta = _tb;
                    nd.add_MtM(Af_t_.view({1, (int64_t)F_.nnz}), ArpF_t_, AciF_t_);
                    _tb = _sw_now(); const double t_mtm = _tb - _ta; _ta = _tb;
                    // dlt_t_ 由上一帧尾部写入（旧路径为逐星近似，精确路径为
                    // −Cᵀ(D⁻¹)_{ii}C），两路都在此加入当前帧 Hessian。
                    if (m > 0) nd.add_M(dlt_t_.view({1, (int64_t)n_ * 36}), rpC_t_, ciC_t_);
                    _tb = _sw_now(); const double t_add = _tb - _ta; _ta = _tb;
                    nd.factor();
                    _tb = _sw_now(); const double t_fac = _tb - _ta; _ta = _tb;
                    auto yv = eta_t_.view({1, nc}).clone();
                    nd.solve(yv);
                    _tb = _sw_now(); const double t_slv = _tb - _ta; _ta = _tb;
                    yF_t_.narrow(0, (int64_t)m * nc, nc).copy_(yv.view({nc}));
                    double t_kern = 0.0, t_sel = 0.0;
                    if (m < L_ - 1) {
                        if (sw_diag) {
                            const double _tk = _sw_now();
                            qoejopt::flsFrameSweepDevice(
                                mtm.first.data_ptr<double>(), mtm_scl_.data_ptr<double>(), n_, L_, E_,
                                G_, m, S_.off_isl, S_.off_gts, S_.off_dyn, S_.off_damp,
                                fis_p_.data_ptr<int64_t>(), fis_s_.data_ptr<int64_t>(),
                                fis_c_.data_ptr<int64_t>(), fgt_p_.data_ptr<int64_t>(),
                                fgt_s_.data_ptr<int64_t>(), fgt_c_.data_ptr<int64_t>(),
                                m > 0 ? dlt_t_.data_ptr<double>() : nullptr,
                                yF_t_.data_ptr<double>() + (int64_t)m * nc,
                                dlt_t_.data_ptr<double>(), gam_t_.data_ptr<double>());
                            t_kern = _sw_now() - _tk;
                        } else {
                            // 精确 Schur：C_m 由本区间 dyn 行（已列均衡）直接组装。
                            auto dyn = mtm.first
                                           .narrow(0, (int64_t)S_.off_dyn +
                                                          (int64_t)m * nc * 12,
                                                   nc * 12)
                                           .reshape({(int64_t)n_, 6, 12});
                            auto Cm = torch::bmm(dyn.narrow(2, 0, 6).transpose(1, 2),
                                                 dyn.narrow(2, 6, 6));
                            Cs[m].copy_(Cm);
                            // γ_m = C_mᵀ y_m（前代 RHS 的跨帧项）。
                            gam_t_.copy_(torch::bmm(Cm.transpose(1, 2),
                                                    yF_t_.narrow(0, (int64_t)m * nc, nc)
                                                        .view({(int64_t)n_, 6, 1}))
                                             .view({nc}));
                            // Δ_m = −C_mᵀ (D_m⁻¹)_{ii} C_m：用整帧因子对角块的
                            // selected inverse（含 ISL 跨星耦合），与联合因子逐位同解。
                            const double _ts = _sw_now();
                            auto sb = nd.compute_diag_blocks(6);          // CPU, span 序
                            auto perm = nd.param_to_span();               // param→span
                            auto sinv = sb.index_select(0, perm).to(optsD);
                            if (getenv("FLS_SWEEP_CHK") && m == 0) {
                                auto x6 = torch::zeros({1, 6 * nc}, optsD);
                                for (int r = 0; r < 6; ++r)
                                    x6[0][(int64_t)r * nc + r] = 1.0;
                                nd.solve_multi(x6, 6);
                                auto s0 = sinv[0].cpu();
                                double emax = 0.0;
                                auto s0a = s0.accessor<double, 2>();
                                auto x6c = x6.cpu();
                                auto x6a = x6c.accessor<double, 2>();
                                for (int a = 0; a < 6; ++a)
                                    for (int b = 0; b < 6; ++b) {
                                        const double v = x6a[0][(int64_t)b * nc + a];
                                        emax = std::max(emax, std::fabs(v - s0a[a][b]));
                                    }
                                fprintf(stderr, "[sweep/chk] selinv vs 6-RHS solve: max=%.3e\n",
                                        emax);
                            }
                            dlt_t_.copy_(
                                torch::bmm(torch::bmm(Cm.transpose(1, 2), sinv), Cm)
                                    .neg()
                                    .reshape({(int64_t)n_ * 36}));
                            t_sel = _sw_now() - _ts;
                        }
                    }
                    if (sw_dbg) {
                        // 逐帧诊断：本帧 dlt/解的量级与非有限计数（NaN 定位用）。
                        auto dl = dlt_t_.to(torch::kCPU);
                        auto dacc = dl.accessor<double, 1>();
                        double dmx = 0.0;
                        int64_t dnf = 0;
                        for (int64_t q = 0; q < (int64_t)n_ * 36; ++q) {
                            if (!std::isfinite(dacc[q])) { ++dnf; continue; }
                            dmx = std::max(dmx, std::fabs(dacc[q]));
                        }
                        auto yf = yF_t_.narrow(0, (int64_t)m * nc, nc).to(torch::kCPU);
                        auto yacc = yf.accessor<double, 1>();
                        double ymx = 0.0;
                        int64_t ynf = 0;
                        for (int64_t q = 0; q < nc; ++q) {
                            if (!std::isfinite(yacc[q])) { ++ynf; continue; }
                            ymx = std::max(ymx, std::fabs(yacc[q]));
                        }
                        fprintf(stderr,
                                "[sweep] m=%d gather=%.2f eta=%.2f create=%.2f mtm=%.2f "
                                "addM=%.2f factor=%.2f solve=%.2f kern=%.2f selinv=%.2f | "
                                "dlt: nf=%lld/%lld max=%.3e | y: nf=%lld/%lld max=%.3e\n",
                                m, 1e3 * t_gath, 1e3 * t_eta, 1e3 * t_cre, 1e3 * t_mtm,
                                1e3 * t_add, 1e3 * t_fac, 1e3 * t_slv, 1e3 * t_kern, 1e3 * t_sel,
                                (long long)dnf, (long long)n_ * 36, dmx, (long long)ynf,
                                (long long)nc, ymx);
                    }
                    nds.push_back(std::move(nd));
                }
                const double t_fwd = _sw_now() - t_sw0;
                double t_bwd = 0.0;
                if (L_ > 1) {
                    const double _tb0 = _sw_now();
                    for (int m = L_ - 2; m >= 0; --m) {
                        // δ_m = y_m − D_m⁻¹ C_m δ_{m+1}
                        tmp_t_.copy_(torch::bmm(
                            Cs[m], yF_t_.narrow(0, (int64_t)(m + 1) * nc, nc)
                                       .view({(int64_t)n_, 6, 1})).view({nc}));
                        auto zz = tmp_t_.view({1, nc}).clone();
                        nds[m].solve(zz);
                        yF_t_.narrow(0, (int64_t)m * nc, nc).sub_(zz.view({nc}));
                    }
                    t_bwd = _sw_now() - _tb0;
                }
                if (sw_dbg) {
                    // 残差校验：联合 scaled 法方程 H_s Y = b_s 是否被块三对角解满足。
                    // 说明见 sweep_scaled_resid_（相消感知的相对残差）。
                    const double rn = sweep_scaled_resid_(mtm.first, mtm.second,
                                                          yF_t_.view({S_.ncol}), t_fwd, t_bwd,
                                                          (int)sw_diag);
                    (void)rn;
                }
                dcpu = (yF_t_.view({L_, nc}) * mtm_scl_.view({L_, nc})).view({-1}).cpu();
            } else {
                auto optsD = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA);
                auto _jn = [] {
                    return std::chrono::duration<double>(
                               std::chrono::steady_clock::now().time_since_epoch()).count();
                };
                // cross 块按原列序给；chain 的星主序列置换会使 bi/bj 错位，
                // 有 cross 时禁用 chain（与 sweep 一致）。
                const bool chain = getenv("KF_CHAIN") && !has_gauge_ &&
                                   symStar_ != nullptr && !cross_active_;
                const double _j0 = _jn();
                nd_ = chain
                          ? std::make_unique<NumericDecomposition>(
                                symStar_->createNumericDecomposition(1))
                          : std::make_unique<NumericDecomposition>(
                                sym_->createNumericDecomposition(1));
                const double _j1 = _jn();
                torch::Tensor rhs, scl;
                if (chain) {
                    // 星主序：块图 = ISL 星图，dyn/先验链落在每星 6L 块内。
                    nd_->add_MtM(mtm.first.reshape({1, S_.nnz}), A_rp_t_, A_ci_star_t_);
                    rhs = mtm.second.index_select(0, inv_perm_t_).reshape({1, S_.ncol}).clone();
                    scl = mtm_scl_.index_select(0, inv_perm_t_);
                } else {
                    nd_->add_MtM(mtm.first.reshape({1, S_.nnz}), A_rp_t_, A_ci_t_);
                    rhs = mtm.second.reshape({1, S_.ncol}).clone();
                    scl = mtm_scl_;
                }
                if (has_gauge_) {
                    auto ev = torch::full({1, (int64_t)S_.ncol}, kGaugeSolveEps, optsD);
                    nd_->add_M(ev, rp_eps_t_, ci_eps_t_);
                }
                // 关键：cross 先验必须进入 GN Hessian，否则 H/g/cost 不同源。
                add_cross_to_nd_(*nd_);
                const double _j2 = _jn();
                nd_->factor();
                const double _j3 = _jn();
                nd_->solve(rhs);
                // factor/solve 兜底：cross 坐标或全局不定时，先清 cross、
                // 用对角先验重建 factor 重解一次；再失败才抛，不把 NaN 送到 D。
                bool fin_rhs = torch::isfinite(rhs).all().item<bool>();
                if (!fin_rhs && cross_active_) {
                    fprintf(stderr, "[kf/cross] cross_fallback=1 "
                                    "(joint solve 非有限；清 cross 重解)\n");
                    clear_prior_cross_();
                    mtm = scaled_AtA_Atb();
                    nd_ = std::make_unique<NumericDecomposition>(
                        sym_->createNumericDecomposition(1));
                    nd_->add_MtM(mtm.first.reshape({1, S_.nnz}), A_rp_t_, A_ci_t_);
                    if (has_gauge_) {
                        auto ev = torch::full({1, (int64_t)S_.ncol},
                                              kGaugeSolveEps, optsD);
                        nd_->add_M(ev, rp_eps_t_, ci_eps_t_);
                    }
                    nd_->factor();
                    rhs = mtm.second.reshape({1, S_.ncol}).clone();
                    scl = mtm_scl_;
                    nd_->solve(rhs);
                    fin_rhs = torch::isfinite(rhs).all().item<bool>();
                }
                if (!fin_rhs)
                    throw std::runtime_error(
                        "solve_batch: factor/solve 非有限（cross 已回退）");
                const double _j4 = _jn();
                if (chain)
                    dcpu = (rhs.reshape({S_.ncol}) * scl)
                               .index_select(0, col_perm_t_)
                               .cpu();
                else
                    dcpu = (rhs.reshape({S_.ncol}) * scl).cpu();
                const double _j5 = _jn();
                if (swdbg)
                    fprintf(stderr,
                            "[sweep] joint chain=%d create=%.2f mtm=%.2f factor=%.2f "
                            "solve=%.2f d2h=%.2f ms data=%lld\n",
                            (int)chain, 1e3 * (_j1 - _j0), 1e3 * (_j2 - _j1),
                            1e3 * (_j3 - _j2), 1e3 * (_j4 - _j3), 1e3 * (_j5 - _j4),
                            (long long)nd_->data_size());
                if (swdbg && !has_gauge_) {
                    // 与 sweep 同口径的残差（scl 还原到 scaled 解 y = dc / scl）。
                    auto scl_full = chain ? scl.index_select(0, col_perm_t_) : scl;
                    auto Yj = (dcpu / scl_full.to(torch::kCPU)).to(optsD);
                    sweep_scaled_resid_(mtm.first, mtm.second, Yj, 1e3 * (_j3 - _j2),
                                        1e3 * (_j4 - _j3), -1);
                }
            }
            std::vector<double> dcv((size_t)S_.ncol, 0.0);
            {
                auto dc = dcpu.accessor<double, 1>();
                for (int64_t q = 0; q < S_.ncol; ++q) dcv[(size_t)q] = dc[q];
            }
            if (has_gauge_ && gauge_k_ > 0) {
                // D7 route (A)：cost 在 Px 上计、gauge 项用 Pg_eff^-1；RHS 同步：
                //   bproj = P^T Atb = Atb - G S^-1 G^T Atb,
                //   w     = Pg_eff^-1 (z - G^T x),
                //   δ     = K^-1 bproj + X w - Utot D^-1 Utot^T (bproj + G w)
                //         = dcv - X cS + X w - Utot D^-1 Utot^T (Atb + G(w-cS)),
                //   cS = S^-1 G^T Atb。
                // Hwin=(I-Pi)K(I-Pi)+G Pg_eff^-1 G^T 不变（D 仍按 Pg_eff^-1 构造）。
                GaugeWoodburyData gw;
                if (!build_gauge_woodbury_(nd_.get(), mtm_scl_, false, gw))
                    throw std::runtime_error(
                        "solve_batch: gauge Woodbury D 不可逆（检查 G/Pg 尺度）");
                const int k = gw.k;
                const int64_t nc = (int64_t)S_.ncol;
                Eigen::MatrixXd Gmat(nc, k);
                for (int64_t a = 0; a < nc; ++a)
                    for (int j = 0; j < k; ++j) Gmat(a, j) = Qg_host_[(size_t)a * k + j];
                auto b_cpu = mtm.second.to(torch::kCPU).contiguous();
                auto sc_cpu = mtm_scl_.to(torch::kCPU).contiguous();
                const double *bp = b_cpu.data_ptr<double>();
                const double *sp = sc_cpu.data_ptr<double>();
                Eigen::VectorXd atb(nc);
                for (int64_t i = 0; i < nc; ++i) atb(i) = bp[i] / sp[i];
                Eigen::VectorXd gxl(k), zp(k);
                for (int j = 0; j < k; ++j) zp(j) = zg_host_[(size_t)j];
                for (int a = 0; a < k; ++a) {
                    double sxl = 0.0;
                    for (int64_t i = 0; i < nc; ++i)
                        sxl += Gmat(i, a) * hX_[(size_t)i];
                    gxl(a) = sxl;
                }
                const Eigen::VectorXd cS = gw.Sinv * (Gmat.transpose() * atb);
                const Eigen::VectorXd w = gw.Q * (zp - gxl);
                const Eigen::VectorXd rhs_vec = atb + Gmat * (w - cS);
                const Eigen::VectorXd rhs = gw.Utot.transpose() * rhs_vec;
                const Eigen::VectorXd ag = gw.Dinv * rhs;
                if (!ag.allFinite() || !w.allFinite() || !cS.allFinite())
                    throw std::runtime_error("solve_batch: gauge Woodbury 解非有限");
                const Eigen::VectorXd Xw = gw.Utot.leftCols(k) * (w - cS);
                const Eigen::VectorXd Ua = gw.Utot * ag;
                for (int64_t i = 0; i < nc; ++i)
                    dcv[(size_t)i] += Xw(i) - Ua(i);
            }
            if (getenv("FLS_DBG")) {
                double dmax = 0.0; int64_t dnf = 0;
                for (int64_t q = 0; q < S_.ncol; ++q) {
                    if (!std::isfinite(dcv[(size_t)q])) { ++dnf; continue; }
                    dmax = std::max(dmax, std::fabs(dcv[(size_t)q]));
                }
                fprintf(stderr, "[fls/sb] it=%d |dc|max=%.3e nf=%lld/%lld sweep=%d gauge=%d\n",
                        gn_it, dmax, (long long)dnf, (long long)S_.ncol, (int)sweep_,
                        (int)has_gauge_);
            }
            if (getenv("FLS_SWEEP_CMP") && sweep_ && !has_gauge_) {
                // 诊断：同一线性化点上与联合 BaspaCho 解逐分量对比。
                auto ndj = sym_->createNumericDecomposition(1);
                ndj.add_MtM(mtm.first.reshape({1, S_.nnz}), A_rp_t_, A_ci_t_);
                ndj.factor();
                auto xj = mtm.second.reshape({1, S_.ncol}).clone();
                ndj.solve(xj);
                auto dj = (xj.reshape({S_.ncol}) * mtm_scl_).cpu().contiguous();
                const double *jp = dj.data_ptr<double>();
                double dmax = 0.0, n2 = 0.0, j2 = 0.0, dot = 0.0;
                for (int64_t q = 0; q < S_.ncol; ++q) {
                    const double d = dcv[(size_t)q] - jp[q];
                    dmax = std::max(dmax, std::fabs(d));
                    n2 += dcv[(size_t)q] * dcv[(size_t)q];
                    j2 += jp[q] * jp[q];
                    dot += dcv[(size_t)q] * jp[q];
                }
                const double cosv = dot / (std::sqrt(n2 * j2) + 1e-300);
                fprintf(stderr,
                        "[sweep/cmp] it=%d max|ds-dj|=%.3e |ds-dj|/|dj|=%.3e cos=%.6f\n",
                        gn_it, dmax, std::sqrt(std::max(0.0, n2 + j2 - 2.0 * dot)) /
                                            (std::sqrt(j2) + 1e-300), cosv);
            }
            const auto X0 = hX_;
            double alpha = 1.0;
            double alpha_used = 1.0;
            double cost_trial = res2;
            int ls_trials = 0;
            bool acc = false;
            for (int ls = 0; ls < 4; ++ls) {
                ++ls_trials;
                alpha_used = alpha;
                for (size_t i = 0; i < hX_.size(); ++i) hX_[i] = X0[i] + alpha * dcv[i];
                retract_qoe(hX_, L_, n_);
                const double rn = total_cost();
                cost_trial = rn;
                if (getenv("FLS_DBG"))
                    fprintf(stderr, "[flsdbg] it=%d ls=%d res2=%.6e rn=%.6e alpha=%.3g\n",
                            it, ls, res2, rn, alpha);
                if (std::isfinite(rn) && rn < res2) {
                    res2 = rn; acc = true;
                    assemble();
                    // 每次重装后把投影代价的参考点前移到新的线性化点：
                    // total_cost() 用 xp=xref+(I-Pi)(x-xref)，而 A/b 在 hX_ 上；
                    // 若 xref 停在窗口入口，第二轮 GN 的代价与导数不同源。
                    cost_ref_ = hX_;
                    // assemble 会按新 X 重算 Λ_g → Pg_eff^-1；cost 的 gauge 项
                    // 依赖 Pg_eff^-1，必须刷新 merit 基线，否则下一轮线搜索用
                    // 的是旧 Pg_eff 的代价（D7 同源头）。
                    if (has_gauge_) {
                        const double rn2 = total_cost();
                        if (std::isfinite(rn2)) res2 = rn2;
                    }
                    break;
                }
                alpha *= 0.5;
            }
            // 逐轮诊断：dx 用参数增量（retract 之前）度量；accepted=false 时
            // 实际位移为 0，另记 dx_trial_* 以保留被拒绝的大步尝试。
            {
                SolveIterDiag rec;
                rec.iter = it;
                rec.cost_before = cost_before;
                rec.cost_after = cost_trial;
                rec.cost_after_refresh = res2;
                rec.accepted = acc;
                rec.ls_trials = ls_trials;
                rec.alpha = alpha_used;
                double dn = 0.0, dm = 0.0;
                for (size_t i = 0; i < dcv.size(); ++i) {
                    const double v = dcv[i];
                    if (!std::isfinite(v)) { dn = dm = std::numeric_limits<double>::infinity(); break; }
                    dn += v * v;
                    dm = std::max(dm, std::fabs(v));
                }
                dn = std::isfinite(dn) ? std::sqrt(dn) : dn;
                rec.dx_trial_norm = alpha_used * dn;
                rec.dx_trial_max = alpha_used * dm;
                rec.dx_norm = acc ? rec.dx_trial_norm : 0.0;
                rec.dx_max = acc ? rec.dx_trial_max : 0.0;
                last_solve_iters_.push_back(rec);
                last_solve_valid_ = true;
                if (swdbg)
                    fprintf(stderr,
                            "[sweep/it] it=%d accepted=%d alpha=%.3g dx_norm=%.3e "
                            "cost=%.6e -> %.6e\n",
                            it, (int)acc, alpha_used, rec.dx_trial_norm, cost_before,
                            cost_trial);
            }
            if (!acc) { hX_ = X0; geo_all(); break; }
        }
        res2_ = res2;
        assemble();
        cost_ref_ = hX_;           // 求解终点与本窗后处理/边缘化的代价参考点同源
        scaled_AtA_Atb();          // 刷新 mtm_scl_，供边缘化/selected inverse
    }

    // 末次 solve_batch 的逐轮位移/代价（只读诊断）。iters 为每轮记录；
    // last_* 为最后一轮摘要。尚未求解时 valid=false，last_* 返回 NaN/false，
    // 便于 Python 侧在大步长后统一记录 G_first 与当前 K 的不匹配。
    py::dict last_solve_diag() const {
        py::dict d;
        d["valid"] = last_solve_valid_;
        d["n_iters"] = (int)last_solve_iters_.size();
        py::list iters;
        for (const auto &r : last_solve_iters_) {
            py::dict q;
            q["iter"] = r.iter;
            q["dx_norm"] = r.dx_norm;
            q["dx_max"] = r.dx_max;
            q["dx_trial_norm"] = r.dx_trial_norm;
            q["dx_trial_max"] = r.dx_trial_max;
            q["alpha"] = r.alpha;
            q["cost_before"] = r.cost_before;
            q["cost_after"] = r.cost_after;
            q["cost_after_refresh"] = r.cost_after_refresh;
            q["accepted"] = r.accepted;
            q["ls_trials"] = r.ls_trials;
            iters.append(q);
        }
        d["iters"] = iters;
        const double nan = std::numeric_limits<double>::quiet_NaN();
        if (last_solve_iters_.empty()) {
            d["last_iter"] = -1;
            d["last_dx_norm"] = nan;
            d["last_dx_max"] = nan;
            d["last_dx_trial_norm"] = nan;
            d["last_alpha"] = nan;
            d["last_cost_before"] = nan;
            d["last_cost_after"] = nan;
            d["last_cost_after_refresh"] = nan;
            d["last_accepted"] = false;
            d["last_ls_trials"] = 0;
        } else {
            const auto &r = last_solve_iters_.back();
            d["last_iter"] = r.iter;
            d["last_dx_norm"] = r.dx_norm;
            d["last_dx_max"] = r.dx_max;
            d["last_dx_trial_norm"] = r.dx_trial_norm;
            d["last_alpha"] = r.alpha;
            d["last_cost_before"] = r.cost_before;
            d["last_cost_after"] = r.cost_after;
            d["last_cost_after_refresh"] = r.cost_after_refresh;
            d["last_accepted"] = r.accepted;
            d["last_ls_trials"] = r.ls_trials;
        }
        return d;
    }

    // 消 node0（保留 node1），生成下一窗 node0 的先验 Lp/Pm/z_prior。
    // 与 step 的 L>=2 分支同口径（r=0 逐星 Schur；先用 flsMargDevice）。
    void marginalize_first() {
        if (L_ < 2) throw std::runtime_error("marginalize_first: 需 L>=2");
        enforce_fej_("marginalize_first");
        if (prior_proj_lowrank_)
            fprintf(stderr, "[marg] 警告: 非 gauge 边缘化未含步骤4先验投影的 rank-2k 修正；"
                            "gauge 路径请用 marginalize_first_local_gauge\n");
        assemble();
        scaled_AtA_Atb();
        qoejopt::flsMargDevice(A_val_.data_ptr<double>(), mtm_scl_.data_ptr<double>(),
                               d00_p_.data_ptr<int64_t>(), d00_s_.data_ptr<int64_t>(),
                               d00_c_.data_ptr<int64_t>(),
                               d11_p_.data_ptr<int64_t>(), d11_s_.data_ptr<int64_t>(),
                               d11_c_.data_ptr<int64_t>(),
                               pr_s1_.data_ptr<int64_t>(), pr_s0_.data_ptr<int64_t>(),
                               n_, Lp_.data_ptr<double>(), d_PmMarg_);
        cudaMemcpy(Lp_host_.data(), Lp_.data_ptr<double>(), (size_t)n_ * 36 * sizeof(double),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(Pm_host_.data(), d_PmMarg_, (size_t)n_ * 36 * sizeof(double),
                   cudaMemcpyDeviceToHost);
        std::copy(hX_.begin() + (size_t)n_ * 6, hX_.begin() + (size_t)2 * n_ * 6,
                  zp_host_.begin());
        cudaMemcpy(z_prior_.data_ptr<double>(), hX_.data() + (size_t)n_ * 6,
                   (size_t)n_ * 6 * sizeof(double), cudaMemcpyHostToDevice);
        has_prior_ = true; multi_prior_ = false;
        prior_gauge_projected_ = false;
        prior_proj_lowrank_ = false;      // Lp_ 已重写，旧 A0/B0 失效
        prior_proj_A0_host_.clear();
        prior_proj_B0_host_.clear();
        prior_proj_S0_host_.clear();
        prior_proj_S0inv_host_.clear();
    }

    // 窗口级低秩 gauge 先验：G 是物理 QOE（平移列=1 m、旋转列=1 rad），
    // **不做 QR/SVD 正交化**；条件数由 S=G^T G+eps I 的 Woodbury 处理。
    // Pg 是物理 gauge 坐标 δg 的协方差，信息项为 G Pg^-1 G^T；
    // 跨窗由 Python 传入上一窗返回的 Pg_next（本函数只存储，不重置 Pg）。
    void set_gauge_prior(py::array_t<double, py::array::c_style | py::array::forcecast> G,
                         py::array_t<double, py::array::c_style | py::array::forcecast> Pg,
                         py::array_t<double, py::array::c_style | py::array::forcecast> z) {
        auto g = G.request(); auto p = Pg.request(); auto q = z.request();
        const int k = (int)p.shape[0];
        if (g.ndim != 2 || (int)g.shape[0] != S_.ncol || (int)g.shape[1] != k ||
            p.ndim != 2 || p.shape[1] != k || (size_t)q.size != (size_t)k)
            throw std::runtime_error("set_gauge_prior: G(ncol,k), Pg(k,k), z(k)");
        gauge_k_ = k;
        Qg_host_.assign((double *)g.ptr, (double *)g.ptr + (size_t)S_.ncol * k);
        Eigen::MatrixXd P(k, k);
        for (int a = 0; a < k; ++a)
            for (int b = 0; b < k; ++b) P(a, b) = ((double *)p.ptr)[a * k + b];
        P = 0.5 * (P + P.transpose());
        Pg_host_.assign((double *)P.data(), (double *)P.data() + k * k);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(P);
        Eigen::VectorXd w = es.eigenvalues();
        // 只抬非正特征值（数值舍入负值）；正的旋转块 1e-8 不能被平移块
        // 1e6 的全局相对下限抬到 1e-6（doc 的 Pg 尺度 bug）。
        const double fl = std::max(w.maxCoeff() * 1e-14, 1e-300);
        for (int i = 0; i < k; ++i)
            if (!(w(i) > 0.0)) w(i) = fl;
        Eigen::MatrixXd Pi = es.eigenvectors() * w.cwiseInverse().asDiagonal() *
                             es.eigenvectors().transpose();
        Pg_inv_host_.assign((double *)Pi.data(), (double *)Pi.data() + k * k);
        zg_host_.assign((double *)q.ptr, (double *)q.ptr + k);
        Pg_info_host_.assign((size_t)k * k, 0.0);
        gauge_xg_host_.clear();
        gauge_hg_host_.clear();
        has_gauge_ = true;
        gauge_woodbury_host_valid_ = false;   // G/Pg 已换，旧 Woodbury 诊断失效
        // G 基改变后，旧的首节点投影数据（A0/B0/S0）不再对应本窗 G0；调用方须重新
        // project_prior_gauge。project_prior_gauge 本身在 set_gauge_prior 之后调用。
        prior_gauge_projected_ = false;
        prior_proj_lowrank_ = false;
        prior_proj_A0_host_.clear();
        prior_proj_B0_host_.clear();
        prior_proj_S0_host_.clear();
        prior_proj_S0inv_host_.clear();
        update_pg_eff_();
        // D3/FEJ：G_win 由调用方在 x_first 构造并传入；C++ 只负责从此刻起冻结。
        // 若 set_prior_multi 已建立本窗 x_first，则把当前 G 记为冻结基；后续再
        // set_gauge_prior 会改变 hash，marginalize 的 FEJ 检查会显式报错。
        if (fej_first_x_valid_ && !fej_g_frozen_) {
            fej_G_hash_frozen_ = gauge_G_hash_();
            fej_g_frozen_ = true;
        }
    }

    // 步骤 4：首节点先验的精确 gauge 投影。
    //   Λ0_proj=(I-Pi_{G0})Λ0(I-Pi_{G0})，Pi_{G0}=G0 S0^-1 G0^T，S0=G0^T G0+eps I。
    // Λ0_proj = Λ0 - ΔΛ，ΔΛ = A0 S0^-1 G0^T + G0 S0^-1 A0^T
    //                       - G0 S0^-1 B0 S0^-1 G0^T，
    // 其中 A0=Λ0 G0、B0=G0^T A0。稀疏 prior 行表只能存每星 6×6 块，而 ΔΛ 的
    // 交叉星块是 rank-2k；本方法从节点0已有的 Lp_i（Λ0_i=Lp_i^T Lp_i）构造
    // A0/B0/S0，并在 solve_batch/marginalize_first_local_gauge 的 Woodbury 小矩阵里
    // 追加 -P ΔΛ P 的 rank-2k 项（P 为窗口 gauge 投影），同时修正 Atb 与总代价。
    //
    // 该 rank-2k 路径是**精确代数**，不是近似：
    //   P = I - G_win S_win^-1 G_win^T,  Y=P A0,  Z=P G0,
    //   -P ΔΛ P = -[Y,Z] E [Y,Z]^T,
    //   E = [[0, S0^-1],[S0^-1, -S0^-1 B0 S0^-1]],  C2=-E,
    //   C2^-1 = [[-B0, -S0],[-S0, 0]]  （Woodbury 的 D22 正是 C2^-1+[Y,Z]^T K^-1[Y,Z]）。
    // 唯一近似是 S0/S_win 的 eps 正则（gauge_eps_from_gram）和浮点舍入；没有
    // 低秩减法产生的“不定小负特征值”。doc 要求的 Householder 对照路径见
    // project_prior_gauge_householder（小 n 验证；不进入生产 N=1152）。
    void project_prior_gauge(py::array_t<double, py::array::c_style | py::array::forcecast> G0) {
        if (!has_gauge_ || gauge_k_ <= 0)
            throw std::runtime_error("project_prior_gauge: 需先 set_gauge_prior");
        if (!has_prior_)
            throw std::runtime_error("project_prior_gauge: 需先 set_prior/set_prior_multi");
        if (prior_node_ > 0)
            throw std::runtime_error(
                "project_prior_gauge: 仅支持首节点先验（prior_node<=0）");
        if (prior_proj_external_)
            throw std::runtime_error(
                "project_prior_gauge: prior 已由 set_prior_projected 外部投影，禁止二次投影");
        // 先失效旧入口：本次若在中途校验失败，不得让上一轮的 A0/B0 继续生效。
        prior_gauge_projected_ = false;
        prior_proj_lowrank_ = false;
        prior_proj_A0_host_.clear();
        prior_proj_B0_host_.clear();
        prior_proj_S0_host_.clear();
        prior_proj_S0inv_host_.clear();
        // B0 诊断：每次入口先重置；即使因近奇异跳过投影，也保留本次数值供 getter 读取。
        prior_proj_B0_cond_ = -1.0;
        prior_proj_B0_min_eig_ = 0.0;
        prior_proj_B0_max_eig_ = 0.0;
        prior_proj_guard_reason_.clear();
        auto g = G0.request();
        const int k = gauge_k_;
        if (g.ndim != 2 || (int)g.shape[0] != n_ * 6 || (int)g.shape[1] != k)
            throw std::runtime_error("project_prior_gauge: G0 需 (n*6,k)");
        prior_proj_G0_host_.assign((double *)g.ptr, (double *)g.ptr + (size_t)n_ * 6 * k);
        Eigen::MatrixXd Gm(n_ * 6, k);
        double gmax = 0.0;
        for (int a = 0; a < n_ * 6; ++a)
            for (int j = 0; j < k; ++j) {
                Gm(a, j) = prior_proj_G0_host_[(size_t)a * k + j];
                gmax = std::max(gmax, std::abs(Gm(a, j)));
            }
        if (!Gm.allFinite())
            throw std::runtime_error("project_prior_gauge: G0 含非有限值");
        // 步骤4 的 G0 必须是窗口 G_win 的首节点块（doc: G0∈R^{6N×k} 是 G_win 在 K0
        // 上的分量）。若调用方传了别的基，修正与 solve 的 P_win 不一致，必须显式报错。
        if ((int)Qg_host_.size() < n_ * 6 * k)
            throw std::runtime_error("project_prior_gauge: set_gauge_prior 的 G 尺寸不足");
        double gdev = 0.0;
        for (int a = 0; a < n_ * 6 * k; ++a)
            gdev = std::max(gdev, std::abs(prior_proj_G0_host_[(size_t)a]
                                           - Qg_host_[(size_t)a]));
        if (gdev > 1e-8 * std::max(1.0, gmax))
            throw std::runtime_error(
                "project_prior_gauge: G0 与 set_gauge_prior 的首节点块不一致");
        // 节点0 prior 的每星白化因子 Lp_i：multi_prior_ 时取 Lp_all_ 的 k=0 切片，
        // 否则用单节点 Lp_。Λ0 为块对角，故 A0/B0 可按星累加而不物化 6N×6N。
        const torch::Tensor &Lp_src = multi_prior_ ? Lp_all_ : Lp_;
        std::vector<double> lp((size_t)n_ * 36);
        cudaMemcpy(lp.data(), Lp_src.data_ptr<double>(), (size_t)n_ * 36 * sizeof(double),
                   cudaMemcpyDeviceToHost);
        Eigen::MatrixXd A0 = Eigen::MatrixXd::Zero(n_ * 6, k);
        Eigen::MatrixXd B0 = Eigen::MatrixXd::Zero(k, k);
        for (int i = 0; i < n_; ++i) {
            Eigen::Matrix<double, 6, 6> Li, Lam;
            for (int a = 0; a < 6; ++a)
                for (int b = 0; b < 6; ++b) Li(a, b) = lp[(size_t)i * 36 + a * 6 + b];
            Lam = Li.transpose() * Li;
            for (int a = 0; a < 6; ++a)
                for (int j = 0; j < k; ++j) {
                    double s = 0.0;
                    for (int b = 0; b < 6; ++b) s += Lam(a, b) * Gm(i * 6 + b, j);
                    A0(i * 6 + a, j) = s;
                }
            for (int a = 0; a < k; ++a)
                for (int b = 0; b < k; ++b) {
                    double s = 0.0;
                    for (int c = 0; c < 6; ++c)
                        s += Gm(i * 6 + c, a) * A0(i * 6 + c, b);
                    B0(a, b) += s;
                }
        }
        B0 = 0.5 * (B0 + B0.transpose());
        if (!A0.allFinite() || !B0.allFinite())
            throw std::runtime_error("project_prior_gauge: A0/B0 含非有限值");
        // B0 条件数 guard：节点0先验在 G0 方向信息弱时 B0 近奇异，直接 LLT/低秩
        // 修正会把舍入误差放大成错误梯度。此时不中止整窗，只跳过步骤4投影。
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(B0);
        if (es.info() != Eigen::Success) {
            prior_proj_guard_reason_ = "B0_eigen_failed";
            fprintf(stderr,
                    "[prior/gauge] B0=G0^T Λ0 G0 特征分解失败；跳过步骤4投影"
                    "（prior_proj_lowrank_=false）\n");
            return;
        }
        const double bmax = es.eigenvalues().maxCoeff();
        const double bmin = es.eigenvalues().minCoeff();
        prior_proj_B0_max_eig_ = bmax;
        prior_proj_B0_min_eig_ = bmin;
        prior_proj_B0_cond_ = (bmax > 0.0 && bmin > 0.0)
                                  ? bmax / bmin
                                  : std::numeric_limits<double>::infinity();
        const double b0_guard = std::max(1e-12 * std::max(bmax, 0.0), 1e-300);
        // G0 本身是否满秩仍按原口径检查；B0 近奇异只跳过投影，不中止整窗。
        Eigen::MatrixXd S0 = Gm.transpose() * Gm;
        S0 = 0.5 * (S0 + S0.transpose());
        S0 += gauge_eps_from_gram(S0) * Eigen::MatrixXd::Identity(k, k);
        Eigen::LLT<Eigen::MatrixXd> llt(S0);
        if (llt.info() != Eigen::Success)
            throw std::runtime_error("project_prior_gauge: G0 列不满秩（G0^T G0 非正定）");
        if (!(bmin > b0_guard) || prior_proj_B0_cond_ > 1e12) {
            prior_proj_guard_reason_ = "B0_near_singular";
            fprintf(stderr,
                    "[prior/gauge] B0=G0^T Λ0 G0 近奇异: min_eig=%.3e max_eig=%.3e "
                    "cond=%.3e（guard: min_eig>max(1e-12*max_eig,1e-300) 且 cond<=1e12）；"
                    "跳过步骤4投影（prior_proj_lowrank_=false）\n",
                    bmin, bmax, prior_proj_B0_cond_);
            return;
        }
        Eigen::MatrixXd S0inv = llt.solve(Eigen::MatrixXd::Identity(k, k));
        // Eigen 默认列主序；host 缓存统一按 (row-major) [a*k+j] 存放，供
        // build/prior_projection_vectors_ 用 [a*k+j] 读取。绝不能直接 assign
        // A0.data()（列主序），否则 k>1 时 A0/B0/S0 会被转置读取。
        prior_proj_A0_host_.assign((size_t)n_ * 6 * k, 0.0);
        for (int a = 0; a < n_ * 6; ++a)
            for (int j = 0; j < k; ++j)
                prior_proj_A0_host_[(size_t)a * k + j] = A0(a, j);
        prior_proj_B0_host_.assign((size_t)k * k, 0.0);
        prior_proj_S0_host_.assign((size_t)k * k, 0.0);
        prior_proj_S0inv_host_.assign((size_t)k * k, 0.0);
        for (int a = 0; a < k; ++a)
            for (int b = 0; b < k; ++b) {
                prior_proj_B0_host_[(size_t)a * k + b] = B0(a, b);
                prior_proj_S0_host_[(size_t)a * k + b] = S0(a, b);
                prior_proj_S0inv_host_[(size_t)a * k + b] = S0inv(a, b);
            }
        prior_gauge_projected_ = true;
        prior_proj_lowrank_ = true;
    }
    // 小窗口参考路径：Householder 显式 gauge 投影（doc §"若修完 Pg 仍需投影"）。
    //   G0=Q0 R0（Eigen HouseholderQR），Z=Q^T 使 Z Q0=[I_k;0]，
    //   Lambda_tilde=Z Lambda0 Z^T，前 k 行/列清零或替换 gauge 块，
    //   Lambda_proj=Z^T Lambda_tilde Z，再加 eps=1e-9||Lambda0||。
    // 与生产 rank-2k 路径的关系见 include/fls_gauge_householder.h 顶部：
    //   生产：Lambda0 - dLam，dLam=A0 S0^-1 G0^T+...（S0=G0^T G0+eps I）；
    //   本路径：正交变换 + 分块赋值，避免低秩减法的负特征值。
    // 返回 dict 含 Lambda0/Lambda0_proj/Lambda0_rank2k/误差与 doc 门禁；
    // 不写回 prior_proj_*，不改 solve 状态（纯参考/验证）。
    // 仅允许 n*6 <= max_ncol（默认 600）；N=1152 的 6912 维 dense Z 不在此路径。
    py::dict project_prior_gauge_householder(
            py::array_t<double, py::array::c_style | py::array::forcecast> G0,
            int max_ncol = 600, int gauge_block_mode = 0) {
        if (!has_gauge_ || gauge_k_ <= 0)
            throw std::runtime_error("project_prior_gauge_householder: 需先 set_gauge_prior");
        if (!has_prior_)
            throw std::runtime_error("project_prior_gauge_householder: 需先 set_prior");
        if (prior_node_ > 0)
            throw std::runtime_error(
                "project_prior_gauge_householder: 仅支持首节点先验（prior_node<=0）");
        const int n6 = n_ * 6;
        if (n6 > max_ncol)
            throw std::runtime_error(
                "project_prior_gauge_householder: n*6 超过 max_ncol（dense Z 仅小窗验证）");
        const int k = gauge_k_;
        auto g = G0.request();
        if (g.ndim != 2 || (int)g.shape[0] != n6 || (int)g.shape[1] != k)
            throw std::runtime_error("project_prior_gauge_householder: G0 需 (n*6,k)");
        Eigen::MatrixXd Gm(n6, k);
        double gmax = 0.0, gdev = 0.0;
        for (int a = 0; a < n6; ++a)
            for (int j = 0; j < k; ++j) {
                Gm(a, j) = ((double *)g.ptr)[(size_t)a * k + j];
                gmax = std::max(gmax, std::abs(Gm(a, j)));
                gdev = std::max(gdev, std::abs(Gm(a, j) - Qg_host_[(size_t)a * k + j]));
            }
        if (!Gm.allFinite())
            throw std::runtime_error("project_prior_gauge_householder: G0 含非有限值");
        if ((int)Qg_host_.size() < n6 * k || gdev > 1e-8 * std::max(1.0, gmax))
            throw std::runtime_error(
                "project_prior_gauge_householder: G0 与 set_gauge_prior 的首节点块不一致");
        const torch::Tensor &Lp_src = multi_prior_ ? Lp_all_ : Lp_;
        std::vector<double> lp((size_t)n_ * 36);
        cudaMemcpy(lp.data(), Lp_src.data_ptr<double>(), (size_t)n_ * 36 * sizeof(double),
                   cudaMemcpyDeviceToHost);
        Eigen::MatrixXd Lam0 = Eigen::MatrixXd::Zero(n6, n6);
        for (int i = 0; i < n_; ++i) {
            Eigen::Matrix<double, 6, 6> Li, Lam;
            for (int a = 0; a < 6; ++a)
                for (int b = 0; b < 6; ++b) Li(a, b) = lp[(size_t)i * 36 + a * 6 + b];
            Lam = Li.transpose() * Li;
            Lam0.block<6, 6>(i * 6, i * 6) = Lam;
        }
        Eigen::MatrixXd Pg(k, k);
        for (int a = 0; a < k; ++a)
            for (int b = 0; b < k; ++b) Pg(a, b) = Pg_host_[(size_t)a * k + b];
        const Eigen::MatrixXd *pgp =
            ((int)Pg_host_.size() == k * k) ? &Pg : nullptr;
        const fls::HouseholderProjection hp =
            fls::householder_project_prior(Lam0, Gm, pgp, gauge_block_mode, true);
        if (hp.B0_status != 0) {
            // B0 近奇异/非有限时不中止整窗，也不返回可能被舍入放大的投影矩阵。
            fprintf(stderr,
                    "[prior/gauge/householder] B0=G0^T Λ0 G0 %s: min_eig=%.3e "
                    "max_eig=%.3e cond=%.3e；返回 applied=0（不应用投影）\n",
                    hp.B0_status == 1 ? "近奇异" : "非有限/特征分解失败",
                    hp.B0_min_eig, hp.B0_max_eig, hp.B0_cond);
            py::dict d0;
            d0["n"] = hp.n;
            d0["k"] = hp.k;
            d0["mode"] = hp.gauge_block_mode;
            d0["applied"] = 0;
            d0["reason"] = hp.B0_status == 1 ? "B0_near_singular" : "B0_invalid";
            d0["B0_cond"] = hp.B0_cond;
            d0["B0_min_eig"] = hp.B0_min_eig;
            d0["B0_max_eig"] = hp.B0_max_eig;
            return d0;
        }
        py::dict d;
        d["applied"] = 1;
        d["reason"] = "";
        d["B0_cond"] = hp.B0_cond;
        d["B0_min_eig"] = hp.B0_min_eig;
        d["B0_max_eig"] = hp.B0_max_eig;
        d["n"] = hp.n;
        d["k"] = hp.k;
        d["mode"] = hp.gauge_block_mode;
        d["Lambda0"] = eigen_to_np_rowmajor(hp.Lambda0);
        d["Lambda0_proj"] = eigen_to_np_rowmajor(hp.Lambda_proj);
        d["Lambda0_rank2k"] = eigen_to_np_rowmajor(hp.Lambda_rank2k);
        d["Lambda0_expected"] = eigen_to_np_rowmajor(hp.Lambda_expected);
        d["gauge_term"] = eigen_to_np_rowmajor(hp.gauge_term);
        d["Q0"] = eigen_to_np_rowmajor(hp.Q0);
        d["R0"] = eigen_to_np_rowmajor(hp.R0);
        d["eps"] = hp.eps;
        d["eps_gram"] = hp.eps_gram;
        d["err_rel"] = hp.err_rel;
        d["gate_gauge"] = hp.gate_gauge;
        d["gate_perp"] = hp.gate_perp;
        d["lmin_total"] = hp.lmin_total;
        d["qr_err"] = hp.qr_err;
        return d;
    }
    // 接受调用方已投影好的逐星 prior（Lp,z）。该入口只能表示可块对角化的近似
    // 投影；此时不再叠加本类的 A0/B0 低秩修正，避免二次投影。
    void set_prior_projected(py::array_t<double, py::array::c_style | py::array::forcecast> Lp,
                             py::array_t<double, py::array::c_style | py::array::forcecast> z) {
        set_prior(Lp, z);
        prior_gauge_projected_ = true;
        prior_proj_lowrank_ = false;
        prior_proj_external_ = true;   // 已外部投影，禁止再 project_prior_gauge 二次投影
    }

    // 局部联合边界先验：对每个重叠变量 (k,i) 取“同一颗星前后时刻 + 同节点
    // 一圈邻居星”的小球（节点 k-1,k,k+1；星 i∪N_r(i)），用球内全部
    // ISL/GTS/dyn/当前先验（含跨块）求局部后验。返回：
    //   Pm_all  — 球内目标变量的完整边际协方差（物理，含 gauge，诊断用）
    //   Lp_all  — 去 gauge 后目标 6×6 条件信息 Λ_phys 的 Cholesky（LpᵀLp=Λ）
    //   cross   — 去 gauge 后球内 off-diagonal 6×6 信息块（scaled，成对存一次）
    // 块归属：只保留 target_gvar < other_gvar 的块，保证每个变量对只记一次。
    void local_prior_blocks_(int r, const Eigen::MatrixXd &Pg,
                             const std::vector<double> &Gmat,
                             const std::vector<double> &sc, const double *av,
                             std::vector<double> &Pm_all,
                             std::vector<double> &Lp_all,
                             std::vector<int> &cbi, std::vector<int> &cbj,
                             std::vector<double> &cval) {
        const int64_t nc = (int64_t)S_.ncol;
        const int kk = gauge_k_;
        const int nt = std::max(0, L_ - 1) * n_;
        Pm_all.assign((size_t)nt * 36, 0.0);
        Lp_all.assign((size_t)nt * 36, 0.0);
        cbi.clear(); cbj.clear(); cval.clear();
        std::vector<std::vector<int>> nbr((size_t)n_);
        for (int e = 0; e < E_; ++e) {
            const int a = isl_i_[e], b = isl_j_[e];
            if (a != b) { nbr[a].push_back(b); nbr[b].push_back(a); }
        }
        const int row_prior0 = S_.nISL + S_.nGTS + S_.nDYN;
        const int ntasks = std::max(0, L_ - 1) * n_;
#pragma omp parallel
        {
            std::vector<int> loc((size_t)L_ * n_, -1);
            std::vector<int> gvars, stars;
            std::vector<int> lbi, lbj;
            std::vector<double> lval;
            std::vector<double> H;
#pragma omp for schedule(dynamic)
            for (int t = 0; t < ntasks; ++t) {
                const int k = 1 + t / n_;
                const int i = t % n_;
                stars.clear();
                stars.push_back(i);
                {
                    std::vector<int> frontier{i};
                    std::vector<char> in_star((size_t)n_, 0);
                    in_star[(size_t)i] = 1;
                    for (int lev = 0; lev < r; ++lev) {
                        std::vector<int> next;
                        for (int ss : frontier)
                            for (int jj : nbr[(size_t)ss])
                                if (!in_star[(size_t)jj]) {
                                    in_star[(size_t)jj] = 1;
                                    stars.push_back(jj);
                                    next.push_back(jj);
                                }
                        frontier.swap(next);
                        if (frontier.empty()) break;
                    }
                }
                std::sort(stars.begin(), stars.end());
                stars.erase(std::unique(stars.begin(), stars.end()), stars.end());
                const int nd0 = std::max(0, k - 1);
                const int nd1 = std::min(L_ - 1, k + 1);
                gvars.clear();
                for (int node = nd0; node <= nd1; ++node) {
                    if (node == k) {
                        for (int s : stars) gvars.push_back(node * n_ + s);
                    } else {
                        gvars.push_back(node * n_ + i);
                    }
                }
                const int M = 6 * (int)gvars.size();
                for (int p = 0; p < (int)gvars.size(); ++p) loc[(size_t)gvars[p]] = p;
                const int tpos = loc[(size_t)(k * n_ + i)];
                H.assign((size_t)M * M, 0.0);
                std::vector<double> rv(M, 0.0);
                auto add_row = [&](int row) {
                    std::fill(rv.begin(), rv.end(), 0.0);
                    for (int64_t j = S_.A_rp[row]; j < S_.A_rp[row + 1]; ++j) {
                        const int c = S_.A_ci[j];
                        const int p = loc[(size_t)(c / 6)];
                        if (p < 0) return;
                        rv[p * 6 + c % 6] += av[j] * sc[(size_t)c];
                    }
                    for (int a = 0; a < M; ++a) {
                        if (rv[a] == 0.0) continue;
                        for (int b = 0; b < M; ++b)
                            H[(size_t)a * M + b] += rv[a] * rv[b];
                    }
                };
                // ISL：目标节点 k 内两端星都在球内的边。
                for (int e = 0; e < E_; ++e)
                    if (loc[(size_t)(k * n_ + isl_i_[e])] >= 0 &&
                        loc[(size_t)(k * n_ + isl_j_[e])] >= 0)
                        add_row(k * E_ + e);
                // GTS：球内各节点、球内星。
                for (int node = nd0; node <= nd1; ++node)
                    for (int g = 0; g < G_; ++g)
                        if (loc[(size_t)(node * n_ + gts_[g])] >= 0)
                            add_row(node * G_ + g);
                // dyn：同星前后两段。
                for (int ps = k - 1; ps <= k; ++ps) {
                    if (ps < 0 || ps + 1 >= L_) continue;
                    for (int comp = 0; comp < 6; ++comp)
                        add_row(S_.nISL + S_.nGTS +
                                (ps * n_ + i) * 6 + comp);
                }
                // 当前对角先验行。
                if (prior_) {
                    for (int gv : gvars)
                        for (int comp = 0; comp < 6; ++comp)
                            add_row(row_prior0 + gv * 6 + comp);
                }
                // 当前跨块先验。
                for (size_t q = 0; q < cross_bi_.size(); ++q) {
                    const int pi = loc[(size_t)cross_bi_[q]];
                    const int pj = loc[(size_t)cross_bj_[q]];
                    if (pi < 0 || pj < 0) continue;
                    const double *v = &cross_val_[q * 36];
                    for (int a = 0; a < 6; ++a)
                        for (int b = 0; b < 6; ++b) {
                            const double vv = v[a * 6 + b];
                            H[(size_t)(pi * 6 + a) * M + (pj * 6 + b)] += vv;
                            H[(size_t)(pj * 6 + b) * M + (pi * 6 + a)] += vv;
                        }
                }
                double tr = 0.0;
                for (int a = 0; a < M; ++a) tr += H[(size_t)a * M + a];
                const double eps = std::max(tr * 1e-12 / std::max(M, 1), 1e-300);
                for (int a = 0; a < M; ++a) H[(size_t)a * M + a] += eps;
                Eigen::MatrixXd He(M, M);
                for (int a = 0; a < M; ++a)
                    for (int b = 0; b < M; ++b)
                        He(a, b) = H[(size_t)a * M + b];
                He = 0.5 * (He + He.transpose());
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(He);
                const auto &ew = es.eigenvalues();
                const auto &EV = es.eigenvectors();
                const double wmax = std::max(0.0, ew.maxCoeff());
                const double wfl = std::max(wmax * 1e-12, 1e-300);
                Eigen::MatrixXd Ps = EV * (ew.cwiseMax(wfl).array().inverse().matrix()).asDiagonal() *
                                     EV.transpose();
                Ps = 0.5 * (Ps + Ps.transpose());
                Eigen::MatrixXd Pperp = Ps;
                if (kk > 0 && (int)Gmat.size() == (int)(nc * kk)) {
                    Eigen::MatrixXd Gs(M, kk);
                    for (int p = 0; p < (int)gvars.size(); ++p)
                        for (int a = 0; a < 6; ++a) {
                            const int64_t g = (int64_t)gvars[p] * 6 + a;
                            for (int j = 0; j < kk; ++j)
                                Gs(p * 6 + a, j) = Gmat[(size_t)g * kk + j] /
                                                   std::max(sc[(size_t)g], 1e-300);
                        }
                    Pperp = Ps - Gs * Pg * Gs.transpose();
                    Pperp = 0.5 * (Pperp + Pperp.transpose());
                }
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> ep(Pperp);
                const auto &pw = ep.eigenvalues();
                const auto &PV = ep.eigenvectors();
                const double pwmax = std::max(0.0, pw.maxCoeff());
                const double pwfl = std::max(pwmax * 1e-12, 1e-300);
                Eigen::MatrixXd Lam = PV * (pw.cwiseMax(pwfl).array().inverse().matrix()).asDiagonal() *
                                      PV.transpose();
                Lam = 0.5 * (Lam + Lam.transpose());
                // 目标完整边际协方差（含 gauge）。
                const int64_t tg = (int64_t)(k * n_ + i) * 6;
                Eigen::Matrix<double, 6, 6> Pt = Ps.block<6, 6>(tpos * 6, tpos * 6);
                for (int a = 0; a < 6; ++a)
                    for (int b = 0; b < 6; ++b)
                        Pm_all[(size_t)t * 36 + a * 6 + b] =
                            Pt(a, b) * sc[(size_t)(tg + a)] * sc[(size_t)(tg + b)];
                // 目标去 gauge 条件信息 Λ_phys 的 Cholesky（LpᵀLp=Λ）。
                Eigen::Matrix<double, 6, 6> Lt = Lam.block<6, 6>(tpos * 6, tpos * 6);
                for (int a = 0; a < 6; ++a)
                    for (int b = 0; b < 6; ++b)
                        Lt(a, b) /= std::max(sc[(size_t)(tg + a)], 1e-300) *
                                    std::max(sc[(size_t)(tg + b)], 1e-300);
                Lt = 0.5 * (Lt + Lt.transpose());
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> el(Lt);
                const auto &lw = el.eigenvalues();
                const auto &LV = el.eigenvectors();
                const double lfl = std::max(std::max(0.0, lw.maxCoeff()) * 1e-12, 1e-300);
                // Lp 必须满足 LpᵀLp=Λ（信息），Lt 已是物理信息；再取逆会把
                // 先验信息变成协方差，和 cross 块/下一窗口径相反。
                Eigen::Matrix<double, 6, 6> Lpd =
                    (LV * lw.cwiseMax(lfl).asDiagonal() * LV.transpose());
                Lpd = 0.5 * (Lpd + Lpd.transpose());
                Eigen::LLT<Eigen::Matrix<double, 6, 6>> llt(Lpd);
                Eigen::Matrix<double, 6, 6> Lp;
                if (llt.info() == Eigen::Success) Lp = llt.matrixU();
                else Lp = Lpd;
                for (int a = 0; a < 6; ++a)
                    for (int b = 0; b < 6; ++b)
                        Lp_all[(size_t)t * 36 + a * 6 + b] = Lp(a, b);
                // 跨块：只保留 target_gvar < other_gvar。
                // 跨块来自本球条件信息，而另一端对角来自它自己的球；两者
                // 直接拼成全局块稀疏先验可能不定。先按保守系数收缩跨块，
                // 保留方向、牺牲幅度；KF_CROSS_SCALE=1 可做完整对照。
                static const double cross_scale = [] {
                    const char *e = std::getenv("KF_CROSS_SCALE");
                    return e ? std::atof(e) : 0.5;
                }();
                for (int p = 0; p < (int)gvars.size(); ++p) {
                    if (p == tpos) continue;
                    const int og = gvars[p];
                    if (og <= k * n_ + i) continue;
                    Eigen::Matrix<double, 6, 6> B =
                        cross_scale * Lam.block<6, 6>(tpos * 6, p * 6);
                    // 输出物理信息 Λ_phys = sc^-1 Λ_s sc^-1；下一窗用自己的
                    // sc 重新缩放，避免跨窗 sc 变化导致跨块量级错位。
                    const int64_t tg6 = (int64_t)(k * n_ + i) * 6;
                    const int64_t og6 = (int64_t)og * 6;
                    lbi.push_back(k * n_ + i);
                    lbj.push_back(og);
                    for (int a = 0; a < 6; ++a)
                        for (int b = 0; b < 6; ++b)
                            lval.push_back(B(a, b) /
                                           (std::max(sc[(size_t)(tg6 + a)], 1e-300) *
                                            std::max(sc[(size_t)(og6 + b)], 1e-300)));
                }
                for (int gv : gvars) loc[(size_t)gv] = -1;
            }
#pragma omp critical
            {
                cbi.insert(cbi.end(), lbi.begin(), lbi.end());
                cbj.insert(cbj.end(), lbj.begin(), lbj.end());
                cval.insert(cval.end(), lval.begin(), lval.end());
            }
        }
        if (getenv("FLS_MARGDBG"))
            fprintf(stderr, "[marg-local] block priors: targets=%d cross=%d\n",
                    ntasks, (int)cbi.size());
    }

    // 局部 r 邻域 Schur 核：对每个星 i 取 B_i={i}∪N_r(i)，用两帧局部块
    // H00/H01/H11 的 Schur 补 S=H11−H01ᵀH00⁻¹H01，取 node1 目标星 6×6，
    // 返回物理 QOE 空间的 [K⁻¹]_11 块 (n*36)。调用前必须已完成 assemble()
    // 与 scaled_AtA_Atb()；本函数不重复装配，供 gauge/非 gauge 边缘化共享
    // 同一 A/sc。av/sc 非空时直接复用调用方已搬到 host 的缓冲，避免重复 D2H。
    // 每星只写自己的 H 区段，OpenMP 并行不改变逐星结果。
    std::vector<double> local_schur_node1_pk_(int r, const double *av_in = nullptr,
                                              const double *sc_in = nullptr) {
        torch::Tensor A_hold, sc_hold;
        const double *av = av_in;
        const double *sc = sc_in;
        if (av == nullptr) {
            A_hold = A_val_.to(torch::kCPU).contiguous();
            av = A_hold.data_ptr<double>();
        }
        if (sc == nullptr) {
            sc_hold = mtm_scl_.to(torch::kCPU).contiguous();
            sc = sc_hold.data_ptr<double>();
        }

        std::vector<std::vector<int>> nbr((size_t)n_);
        for (int e = 0; e < E_; ++e) {
            const int a = isl_i_[e], b = isl_j_[e];
            if (a != b) { nbr[a].push_back(b); nbr[b].push_back(a); }
        }
        std::vector<std::vector<int>> ball((size_t)n_);
        int m_max = 6;
        for (int i = 0; i < n_; ++i) {
            std::vector<int> B{i};
            std::vector<char> in((size_t)n_, 0);
            in[i] = 1;
            std::vector<int> frontier{i};
            for (int lev = 0; lev < r; ++lev) {
                std::vector<int> next;
                for (int s : frontier)
                    for (int j : nbr[s])
                        if (!in[j]) { in[j] = 1; B.push_back(j); next.push_back(j); }
                frontier.swap(next);
            }
            std::sort(B.begin() + 1, B.end());
            m_max = std::max(m_max, 6 * (int)B.size());
            ball[i] = std::move(B);
        }
        if (m_max > 78)
            throw std::runtime_error("marginalize_first_local: r=2 邻域超过 78 维，请用 r=1");
        if (std::getenv("FLS_MARGDBG"))
            fprintf(stderr, "[marg-local] n=%d L=%d E=%d G=%d mm=%d nrows=%d nnz=%d\n",
                    n_, L_, E_, G_, m_max, S_.nrows, S_.nnz);

        const int mm = m_max;
        std::vector<double> H00((size_t)n_ * mm * mm, 0.0), H01((size_t)n_ * mm * mm, 0.0),
            H11((size_t)n_ * mm * mm, 0.0);
        const int row_prior0 = S_.nISL + S_.nGTS + S_.nDYN;
        const int n6 = 6 * n_;

        // 每个星的局部邻域装配完全独立；按星并行，避免 L=7/1152 时
        // Python/C++ 串行遍历所有边成为 marginalize 主耗时。
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < n_; ++i) {
            const auto &B = ball[i];
            const int M = 6 * (int)B.size();
            std::vector<int> loc((size_t)n_, -1);
            for (int p = 0; p < (int)B.size(); ++p) loc[B[p]] = p;
            double *h00 = H00.data() + (size_t)i * mm * mm;
            double *h01 = H01.data() + (size_t)i * mm * mm;
            double *h11 = H11.data() + (size_t)i * mm * mm;
            auto add_row = [&](int row) {
                std::vector<std::pair<int, double>> nz;
                for (int64_t j = S_.A_rp[row]; j < S_.A_rp[row + 1]; ++j) {
                    const int c = S_.A_ci[j];
                    const int node = c / n6, star = (c % n6) / 6, comp = c % 6;
                    if (node > 1) continue;
                    const int p = loc[star];
                    if (p < 0) continue;
                    const int idx = node * M + p * 6 + comp;
                    const double v = av[j] * sc[c];
                    bool hit = false;
                    for (auto &q : nz) if (q.first == idx) { q.second += v; hit = true; break; }
                    if (!hit) nz.emplace_back(idx, v);
                }
                for (const auto &a : nz)
                    for (const auto &b : nz) {
                        const int na = a.first / M, nb = b.first / M;
                        const int ia = a.first % M, ib = b.first % M;
                        if (na == 0 && nb == 0) h00[ia * mm + ib] += a.second * b.second;
                        else if (na == 1 && nb == 1) h11[ia * mm + ib] += a.second * b.second;
                        else if (na == 0 && nb == 1) h01[ia * mm + ib] += a.second * b.second;
                    }
            };
            // 局部块只含 node0/node1 两帧变量，因此只装配完全落在这两帧内的因子。
            // k>=2 的 ISL/GTS 行、dyn(k→k+1),k>=1 行都还带 node>=2 的列，按 node>1
            // 跳过列后会变成缺列的残缺行，等价于把 node2 当成固定节点，注入虚假信息。
            const int kf = std::min(L_, 2);   // node0、node1 两帧
            // ISL：仅 k=0/1（L>1）且两端都在 B_i 的行；这些行只含 node0/node1 列
            for (int k = 0; k < kf; ++k)
                for (int e = 0; e < E_; ++e)
                    if (loc[isl_i_[e]] >= 0 && loc[isl_j_[e]] >= 0)
                        add_row(k * E_ + e);
            // GTS：仅 k=0/1 且星在 B_i 的行；同样只含 node0/node1 列
            for (int k = 0; k < kf; ++k)
                for (int g = 0; g < G_; ++g)
                    if (loc[gts_[g]] >= 0) add_row(S_.nISL + k * G_ + g);
            // dyn：只取 dyn(0→1)；dyn(k→k+1),k>=1 含 node2 列，不能进局部块。
            // 先验行是逐节点一元因子：k>=2 的先验行在 node>1 投影后整行为空，
            // 不会形成残缺行，故先验仍按 kp0..kp1 处理。
            const int kp0 = (prior_node_ >= 0) ? prior_node_ : 0;
            const int kp1 = (prior_node_ >= 0) ? prior_node_ + 1 : L_;
            for (int s : B) {
                for (int comp = 0; comp < 6; ++comp) {
                    add_row(S_.nISL + S_.nGTS + (0 * n_ + s) * 6 + comp);
                    for (int k = kp0; k < kp1; ++k)
                        add_row(row_prior0 + (k * n_ + s) * 6 + comp);
                }
            }
            // padding：补齐的块用单位阵，保证 S 在补齐维上为 I
            for (int a = M; a < mm; ++a) { h00[a * mm + a] = 1.0; h11[a * mm + a] = 1.0; }
        }

        if (std::getenv("FLS_MARGDBG")) {
            int nbad0 = 0, nbad1 = 0; double w0min = 1e300, w1min = 1e300;
            for (int i = 0; i < n_; ++i) {
                const int M = 6 * (int)ball[i].size();
                Eigen::MatrixXd A0(M, M), A1(M, M);
                for (int a = 0; a < M; ++a)
                    for (int b = 0; b < M; ++b) {
                        A0(a, b) = H00[(size_t)i * mm * mm + a * mm + b];
                        A1(a, b) = H11[(size_t)i * mm * mm + a * mm + b];
                    }
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> e0(A0), e1(A1);
                const double m0 = e0.eigenvalues().minCoeff(), m1 = e1.eigenvalues().minCoeff();
                w0min = std::min(w0min, m0); w1min = std::min(w1min, m1);
                if (!(m0 > 0.0)) ++nbad0;
                if (!(m1 > 0.0)) ++nbad1;
            }
            fprintf(stderr, "[marg-local] H assembled nbad0=%d nbad1=%d w0min=%.3e w1min=%.3e\n",
                    nbad0, nbad1, w0min, w1min);
        }
        std::vector<double> Psc((size_t)n_ * 36, 0.0);
        qoejopt::flsLocalSchurBatch(H00.data(), H01.data(), H11.data(), n_, mm, Psc.data());
        if (std::getenv("FLS_MARGDBG"))
            fprintf(stderr, "[marg-local] kernel done\n");
        std::vector<double> PK((size_t)n_ * 36, 0.0);
        for (int i = 0; i < n_; ++i) {
            const int64_t b1 = (int64_t)(n_ + i) * 6;
            Eigen::Matrix<double, 6, 6> P;
            for (int a = 0; a < 6; ++a)
                for (int c = 0; c < 6; ++c)
                    P(a, c) = Psc[(size_t)i * 36 + a * 6 + c] * sc[b1 + a] * sc[b1 + c];
            P = 0.5 * (P + P.transpose());
            for (int a = 0; a < 6; ++a)
                for (int c = 0; c < 6; ++c)
                    PK[(size_t)i * 36 + a * 6 + c] = P(a, c);
        }
        return PK;
    }

    // 关键帧边缘化（r=1/r=2 局部 Schur）：调用 local_schur_node1_pk_ 取
    // node1 每星 6×6 的局部 Schur 核，返回物理 QOE 空间 Pm (n,6,6)；
    // 同时更新 Pm_host_/z_prior_。
    // 关键帧边缘化（r=1/r=2 局部 Schur）：调用 local_schur_node1_pk_ 取
    // node1 每星 6×6 的局部 Schur 核，返回物理 QOE 空间 Pm (n,6,6)；
    // 同时更新 Pm_host_/z_prior_。
    //
    // 尾部必须把新的 Pm 也写进单节点先验 Lp_（与 marginalize_first 的
    // flsMargDevice 同口径：Pm 与 Lp 同一组 PSD-floor 特征值严格互逆）。
    // 否则 multi_prior_ 切回 false 后，任何后续 assemble() 会走
    // `prior_ && has_prior_` 分支、用从未被写过的零 Lp_ 覆盖先验行，使 A/b
    // 静默丢掉 node0 先验（gauge_lamwin_host / solve_rhs 等诊断会读到无先验的 K）。
    void refresh_single_prior_from_pm_() {
        const double *pm = Pm_host_.data();
        std::vector<double> lp_host((size_t)n_ * 36, 0.0);
        for (int i = 0; i < n_; ++i) {
            Eigen::Matrix<double, 6, 6> P;
            for (int a = 0; a < 6; ++a)
                for (int b = 0; b < 6; ++b)
                    P(a, b) = pm[(size_t)i * 36 + a * 6 + b];
            P = fls::psd_floor_6x6(P);
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> es(P);
            Eigen::Matrix<double, 6, 1> w = es.eigenvalues();
            const double wfl =
                std::max(w.maxCoeff() * 1e-14, 1e-300);
            for (int a = 0; a < 6; ++a)
                if (!(w(a) > 0.0)) w(a) = wfl;
            const Eigen::Matrix<double, 6, 6> Lp =
                (es.eigenvectors() * w.cwiseSqrt().cwiseInverse().asDiagonal())
                    .transpose();
            for (int a = 0; a < 36; ++a) lp_host[(size_t)i * 36 + a] = Lp.data()[a];
        }
        cudaMemcpy(Lp_.data_ptr<double>(), lp_host.data(),
                   (size_t)n_ * 36 * sizeof(double), cudaMemcpyHostToDevice);
    }

    py::array_t<double> marginalize_first_local(int r) {
        if (L_ < 2) throw std::runtime_error("marginalize_first_local: 需 L>=2");
        if (r < 1 || r > 2) throw std::runtime_error("marginalize_first_local: r 仅支持 1/2");
        if (prior_proj_lowrank_)
            fprintf(stderr, "[marg-local] 警告: 非 gauge 局部 Schur 未含步骤4先验投影的 "
                            "rank-2k 修正；gauge 路径请用 marginalize_first_local_gauge\n");
        assemble();
        scaled_AtA_Atb();
        const std::vector<double> PK = local_schur_node1_pk_(r);
        py::array_t<double> out({(py::ssize_t)n_, (py::ssize_t)6, (py::ssize_t)6});
        auto *op = out.mutable_data();
        Pm_host_.assign((size_t)n_ * 36, 0.0);
        for (int i = 0; i < n_; ++i) {
            Eigen::Matrix<double, 6, 6> P;
            for (int a = 0; a < 6; ++a)
                for (int c = 0; c < 6; ++c)
                    P(a, c) = PK[(size_t)i * 36 + a * 6 + c];
            for (int a = 0; a < 6; ++a)
                for (int c = 0; c < 6; ++c) {
                    op[((size_t)i * 6 + a) * 6 + c] = P(a, c);
                    Pm_host_[(size_t)i * 36 + a * 6 + c] = P(a, c);
                }
        }
        std::copy(hX_.begin() + (size_t)n_ * 6, hX_.begin() + (size_t)2 * n_ * 6,
                  zp_host_.begin());
        cudaMemcpy(z_prior_.data_ptr<double>(), hX_.data() + (size_t)n_ * 6,
                   (size_t)n_ * 6 * sizeof(double), cudaMemcpyHostToDevice);
        has_prior_ = true; multi_prior_ = false;
        prior_gauge_projected_ = false;
        prior_proj_lowrank_ = false;      // prior 来源切换/中心更新，旧 A0/B0 不再对应
        prior_proj_A0_host_.clear();
        prior_proj_B0_host_.clear();
        prior_proj_S0_host_.clear();
        prior_proj_S0inv_host_.clear();
        refresh_single_prior_from_pm_();  // 与 Pm 严格互逆，避免后续 assemble 写零先验行
        return out;
    }

    // gauge 版关键帧边缘化：在整窗 gauge 先验
    //   Hwin = (I-Pi)K(I-Pi) + G Pg_eff^-1 G^T
    // 上消去 node0，返回 (Pm, Pg_next)：
    //   Pm      = node1 每星 6×6 后验（物理 QOE，含 gauge 低秩项）
    //   Pg_next = 剩余窗（node1..L-1）的 gauge 协方差，坐标 = 剩余 G 行
    //             = U_r^T Hwin^-1 U_r，U_r 为 G_rest 的整窗嵌入（node0 行置零）。
    //   Hwin 的 Pg_eff 只含 gts_pg_node_range 内的 GTS G 信息（通常首窗全部、
    //   滑窗仅新末节点）；因此 Pg_next 已经是“下一窗先验 Pg”的正确后验，
    //   下一窗只需把新进入节点的 GTS G 信息再加进 Pg，不能重复累加旧节点。
    //   坐标基是当前窗冻结 G 的 node1..L-1 行；first_node=True 时每节点 G
    //   块相同，故与下一窗整窗 G 用同一组 6 个系数。若下一窗在 6×6 层用新
    //   首节点重建旋转列，调用方必须先做换基 Pg_new = T^-1 Pg_next T^-T；
    //   C++ 不猜基、不做隐式换基。
    // Woodbury 只在小矩阵上做，不显式构造 ncol×ncol 投影矩阵：
    //   [Hwin^-1]_rr = [K^-1]_rr - Utot_r D^-1 Utot_r^T,
    //   Utot=K^-1 U=[X,G]（步骤4时再并 [K^-1 Y,K^-1 Z]），
    //   D=C^-1+U^T K^-1 U（见 build_gauge_woodbury_）。
    // [K^-1]_rr 取剩余所有节点的 selected inverse 对角块（exact），再补
    // 同一 Woodbury 的 gauge 修正；Pm_all 供下一窗对全部重叠节点做联合
    // 边界先验。r 仅为兼容签名保留（当前忽略）。
    py::dict marginalize_first_local_gauge(int r, bool exact_compare = false) {
        if (L_ < 2) throw std::runtime_error("marginalize_first_local_gauge: 需 L>=2");
        if (r < 1 || r > 2)
            throw std::runtime_error("marginalize_first_local_gauge: r 仅支持 1/2");
        if (!has_gauge_ || gauge_k_ <= 0)
            throw std::runtime_error("marginalize_first_local_gauge: 需先 set_gauge_prior");
        // D3/FEJ：节点0 先验行 (Lp,z) 与 G_win 必须在首次入窗 x_first 冻结；
        // 本检查在 assemble（会用最终 X 重写 ISL/GTS/dyn 行）之前执行。之后的
        // assemble 只按既有约定重线性化 ISL/GTS/dyn；节点0 先验的 H=ΣLp^T Lp
        // 与 G 不变，先验 b=Lp(z-x) 的变化不影响 Pm/Pg_next。
        enforce_fej_("marginalize_first_local_gauge");
        assemble();
        auto mtm_marg = scaled_AtA_Atb();
        // D7 route (A) 的 RHS 投影（bproj=P^T b, w=Pg_eff^-1(z-G^T x)）只
        // 作用于 solve 的梯度侧；本函数只消费 Hwin^-1 的 rr 块
        // （局部 r Schur + D^-1），不读 Atb/b，因此这里无需再改 RHS。
        const int k = gauge_k_;
        const int64_t nc = (int64_t)S_.ncol;
        auto sc_cpu = mtm_scl_.to(torch::kCPU).contiguous();
        const int64_t order = (int64_t)nc;
        // 联合边界先验的 selected inverse 与 Woodbury 共用同一份 CUDA factor。
        auto nd = sym_->createNumericDecomposition(1);
        nd.add_MtM(mtm_marg.first.reshape({1, S_.nnz}), A_rp_t_, A_ci_t_);
        add_cross_to_nd_(nd);
        {
            auto ev = torch::full({1, order}, kGaugeSolveEps,
                                  torch::TensorOptions().dtype(torch::kFloat64)
                                      .device(torch::kCUDA));
            nd.add_M(ev, rp_eps_t_, ci_eps_t_);
        }
        nd.factor();
        GaugeWoodburyData gw;
        // cpu=false：nd 是 CUDA 数值分解（sym_），build_gauge_woodbury_ 内的
        // K^-1 solve 必须在 CUDA 上取 RHS，否则 Baspacho 报 "Expected
        // x.device().is_cuda()" 并使本函数永远失败（Pg_next / gauge Pm 全丢）。
        const torch::Device dev_solve = mtm_scl_.device();
        if (!build_gauge_woodbury_(&nd, mtm_scl_, false, gw))
            throw std::runtime_error(
                "marginalize_first_local_gauge: gauge Woodbury D 不可逆");
        const int ku = gw.ku;
        Eigen::MatrixXd Gmat(nc, k);
        for (int64_t a = 0; a < nc; ++a)
            for (int j = 0; j < k; ++j) Gmat(a, j) = Qg_host_[(size_t)a * k + j];

        // ---- Pg_next = U_r^T Hwin^-1 U_r ----
        Eigen::MatrixXd Ur = Eigen::MatrixXd::Zero(nc, k);
        for (int node = 1; node < L_; ++node)
            for (int star = 0; star < n_; ++star) {
                const int64_t b = (int64_t)(node * n_ + star) * 6;
                for (int a = 0; a < 6; ++a)
                    for (int j = 0; j < k; ++j) Ur(b + a, j) = Gmat(b + a, j);
            }
        std::vector<double> ur_row((size_t)nc * k, 0.0);
        for (int64_t a = 0; a < nc; ++a)
            for (int j = 0; j < k; ++j) ur_row[(size_t)a * k + j] = Ur(a, j);
        auto Ur_t = torch::from_blob(ur_row.data(), {nc, (int64_t)k},
                                     torch::TensorOptions().dtype(torch::kFloat64))
                        .clone().to(dev_solve);
        auto sc_dev = mtm_scl_.contiguous();   // 与 nd/Ur_t 同 device（CUDA）
        auto Bs = Ur_t * sc_dev.view({nc, 1});
        Eigen::MatrixXd Yr(nc, k);
        for (int j = 0; j < k; ++j) {
            auto yj = Bs.select(1, j).reshape({1, nc}).clone();
            nd.solve(yj);
            auto xj = (yj.reshape({nc}) * sc_dev).cpu().contiguous();
            const double *p = xj.data_ptr<double>();
            for (int64_t a = 0; a < nc; ++a) Yr(a, j) = p[a];
        }
        Eigen::MatrixXd Y6 = Eigen::MatrixXd::Zero(k, k);
        for (int node = 1; node < L_; ++node)
            for (int star = 0; star < n_; ++star) {
                const int64_t b = (int64_t)(node * n_ + star) * 6;
                for (int a = 0; a < 6; ++a)
                    for (int c = 0; c < k; ++c) {
                        const double g = Gmat(b + a, c);
                        for (int j = 0; j < k; ++j) Y6(c, j) += g * Yr(b + a, j);
                    }
            }
        Y6 = 0.5 * (Y6 + Y6.transpose());
        // Btot = U_r^T Utot（U_r 为剩余窗 G 行嵌入：node0 行置零）。
        // 不拆 [X,G] 再逐块对称化：Utot 的增广列（步骤4）本就不对称，
        // Pg=Y6-Btot D^-1 Btot^T 对任意 Btot 都对称，只需最后对称化 Pg。
        Eigen::MatrixXd Btot = Ur.transpose() * gw.Utot;
        Eigen::MatrixXd Pg = Y6 - Btot * gw.Dinv * Btot.transpose();
        Pg = 0.5 * (Pg + Pg.transpose());
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Pg);
        Eigen::VectorXd w = es.eigenvalues();
        const double pg_lmin = w.minCoeff();
        if (pg_lmin < -1e-10 * std::max(std::abs(w.maxCoeff()), 1.0))
            fprintf(stderr, "[marg-gauge] 警告: Pg_next floor 前 λmin=%.3e<0 "
                            "(λmax=%.3e)；floor 仅吸收数值负值，请检查 D/Pg 尺度\n",
                    pg_lmin, w.maxCoeff());
        // Pg_next 同时含平移(~1e6)与旋转(~1e-8)：只抬非正特征值，
        // 不能用全局相对下限把旋转协方差抬大。
        const double fl = std::max(std::max(w.maxCoeff(), 0.0) * 1e-14, 1e-300);
        for (int i = 0; i < k; ++i)
            if (!(w(i) > 0.0)) w(i) = fl;
        Pg = es.eigenvectors() * w.asDiagonal() * es.eigenvectors().transpose();
        Pg = 0.5 * (Pg + Pg.transpose());
        // GTS Fisher 的零空间：Pg_next 在该子空间必须保持输入 Pg（不是被
        // Schur 一起压小）。观测子空间保留上面的 rank-r 收缩。这里只替换
        // null-null 块，不改观测子空间，也不改 Pg_eff 的 window 求解。
        if ((int)Pg_info_host_.size() == k * k && (int)Pg_host_.size() == k * k) {
            Eigen::MatrixXd Lg(k, k);
            for (int a = 0; a < k; ++a)
                for (int b = 0; b < k; ++b)
                    Lg(a, b) = Pg_info_host_[(size_t)a * k + b];
            Lg = 0.5 * (Lg + Lg.transpose());
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> el(Lg);
            const Eigen::VectorXd lw = el.eigenvalues();
            const double lcut = std::max(
                (lw.size() ? lw.cwiseAbs().maxCoeff() : 0.0) * 1e-12, 1e-300);
            std::vector<int> nidx;
            for (int i = 0; i < k; ++i)
                if (!(lw(i) > lcut)) nidx.push_back(i);
            if (!nidx.empty()) {
                Eigen::MatrixXd N(k, (int)nidx.size());
                for (int j = 0; j < (int)nidx.size(); ++j)
                    N.col(j) = el.eigenvectors().col(nidx[(size_t)j]);
                Eigen::MatrixXd Pgin(k, k);
                for (int a = 0; a < k; ++a)
                    for (int b = 0; b < k; ++b)
                        Pgin(a, b) = Pg_host_[(size_t)a * k + b];
                const Eigen::MatrixXd Dn =
                    N.transpose() * (Pgin - Pg) * N;
                const double tr0 = Pg.trace();
                const double trn = (N.transpose() * Pgin * N).trace();
                Pg.noalias() += N * Dn * N.transpose();
                Pg = 0.5 * (Pg + Pg.transpose());
                fprintf(stderr,
                        "[pg/null] rank=%d null=%d tr_before=%.3e "
                        "tr_null_in=%.3e tr_after=%.3e\n",
                        k - (int)nidx.size(), (int)nidx.size(), tr0, trn,
                        Pg.trace());
            }
        }
        py::array_t<double> Pgout({(py::ssize_t)k, (py::ssize_t)k});
        std::copy(Pg.data(), Pg.data() + (size_t)k * k, Pgout.mutable_data());

        // 默认：node1 的 local-r Schur（生产快路径，~0.75 s/窗）。
        // KF_JOINT_DIAG_EXACT=1：计算剩余所有节点的 exact selected-inverse
        // 对角块（~15 s/窗，仅验证；仍不含跨节点块，不能当生产）。
        const bool joint_exact = exact_compare || [] {
            const char *e = std::getenv("KF_JOINT_DIAG_EXACT");
            return e != nullptr && std::atoi(e) != 0;
        }();
        py::array_t<double> out({(py::ssize_t)n_, (py::ssize_t)6, (py::ssize_t)6});
        py::array_t<double> Pm_all_out, Lp_all_out;
        py::array_t<double> Pm_all_local_out, Lp_all_local_out;
        std::vector<int> cross_bi_v, cross_bj_v;
        std::vector<double> cross_val_v;
        auto *op = out.mutable_data();
        Pm_host_.assign((size_t)n_ * 36, 0.0);
        int nbad_pm = 0;
        double pm_lmin = 0.0;
        if (joint_exact) {
            Pm_all_out = py::array_t<double>(
                {(py::ssize_t)(L_ - 1), (py::ssize_t)n_,
                 (py::ssize_t)6, (py::ssize_t)6});
            auto *opall = Pm_all_out.mutable_data();
            auto Pd = nd.compute_diag_blocks_frame0(6, (int64_t)L_ * n_)
                          .cpu().contiguous();
            const double *pp = Pd.data_ptr<double>();
            const double *sp = sc_cpu.data_ptr<double>();
            for (int node = 1; node < L_; ++node) {
                for (int i = 0; i < n_; ++i) {
                    const int64_t b = (int64_t)node * n_ + i;
                    Eigen::Matrix<double, 6, 6> PK;
                    for (int a = 0; a < 6; ++a)
                        for (int c = 0; c < 6; ++c)
                            PK(a, c) = pp[(b * 6 + a) * 6 + c] *
                                       sp[b * 6 + a] * sp[b * 6 + c];
                    PK = 0.5 * (PK + PK.transpose());
                    Eigen::MatrixXd Ai(6, ku);
                    for (int a = 0; a < 6; ++a)
                        for (int j = 0; j < ku; ++j) Ai(a, j) = gw.Utot(b * 6 + a, j);
                    Eigen::Matrix<double, 6, 6> corr = Ai * gw.Dinv * Ai.transpose();
                    Eigen::Matrix<double, 6, 6> Praw = PK - corr;
                    Praw = 0.5 * (Praw + Praw.transpose());
                    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> esp(Praw);
                    const double lmin = esp.eigenvalues().minCoeff();
                    pm_lmin = std::min(pm_lmin, lmin);
                    const double lscale =
                        std::max(std::abs(Praw.diagonal().maxCoeff()), 1e-300);
                    if (lmin < -1e-10 * lscale) ++nbad_pm;
                    Eigen::Matrix<double, 6, 6> P = fls::psd_floor_6x6(Praw);
                    const size_t off = (size_t)(node - 1) * n_ + i;
                    for (int a = 0; a < 6; ++a)
                        for (int c = 0; c < 6; ++c)
                            opall[(off * 6 + a) * 6 + c] = P(a, c);
                    if (node == 1) {
                        for (int a = 0; a < 6; ++a)
                            for (int c = 0; c < 6; ++c) {
                                op[((size_t)i * 6 + a) * 6 + c] = P(a, c);
                                Pm_host_[(size_t)i * 36 + a * 6 + c] = P(a, c);
                            }
                    }
                }
            }
        } else {
            // 局部联合边界先验：每个重叠变量的小球联合后验。
            // Pm_all（完整边际，诊断/Puse）、Lp_all（去 gauge 条件信息）、
            // cross（球内 off-diagonal，scaled，成对存一次）。
            auto A_cpu = A_val_.to(torch::kCPU).contiguous();
            std::vector<double> Pm_all_v, Lp_all_v;
            local_prior_blocks_(r, Pg, Qg_host_, std::vector<double>(
                                    sc_cpu.data_ptr<double>(),
                                    sc_cpu.data_ptr<double>() + S_.ncol),
                                A_cpu.data_ptr<double>(), Pm_all_v, Lp_all_v,
                                cross_bi_v, cross_bj_v, cross_val_v);
            Pm_all_out = py::array_t<double>(
                {(py::ssize_t)(L_ - 1), (py::ssize_t)n_,
                 (py::ssize_t)6, (py::ssize_t)6});
            Lp_all_out = py::array_t<double>(
                {(py::ssize_t)(L_ - 1), (py::ssize_t)n_,
                 (py::ssize_t)6, (py::ssize_t)6});
            auto *opall = Pm_all_out.mutable_data();
            auto *oplp = Lp_all_out.mutable_data();
            for (size_t q = 0; q < Pm_all_v.size(); ++q) {
                opall[q] = Pm_all_v[q];
                oplp[q] = Lp_all_v[q];
            }
            for (int i = 0; i < n_; ++i) {
                for (int a = 0; a < 6; ++a)
                    for (int c = 0; c < 6; ++c) {
                        const double v = Pm_all_v[(size_t)i * 36 + a * 6 + c];
                        op[((size_t)i * 6 + a) * 6 + c] = v;
                        Pm_host_[(size_t)i * 36 + a * 6 + c] = v;
                    }
            }
        }
        // S1 诊断：同一窗同时算 exact selected-inverse 对角块与 local r 球块，
        // 不覆盖 exact Pm/Pm_host；只返回 local 的 Pm_all/Lp_all/cross 供对拍。
        if (exact_compare) {
            auto A_cpu_l = A_val_.to(torch::kCPU).contiguous();
            std::vector<double> Pm_l, Lp_l;
            local_prior_blocks_(r, Pg, Qg_host_, std::vector<double>(
                                    sc_cpu.data_ptr<double>(),
                                    sc_cpu.data_ptr<double>() + S_.ncol),
                                A_cpu_l.data_ptr<double>(), Pm_l, Lp_l,
                                cross_bi_v, cross_bj_v, cross_val_v);
            Pm_all_local_out = py::array_t<double>(
                {(py::ssize_t)(L_ - 1), (py::ssize_t)n_,
                 (py::ssize_t)6, (py::ssize_t)6});
            Lp_all_local_out = py::array_t<double>(
                {(py::ssize_t)(L_ - 1), (py::ssize_t)n_,
                 (py::ssize_t)6, (py::ssize_t)6});
            std::copy(Pm_l.begin(), Pm_l.end(), Pm_all_local_out.mutable_data());
            std::copy(Lp_l.begin(), Lp_l.end(), Lp_all_local_out.mutable_data());
        }
        if (nbad_pm)
            fprintf(stderr, "[marg-gauge] 警告: %d 个节点块 floor 前 "
                            "λmin<-1e-10·scale (min=%.3e)；psd_floor 会掩盖该负特征值\n",
                    nbad_pm, pm_lmin);


        std::copy(hX_.begin() + (size_t)n_ * 6, hX_.begin() + (size_t)2 * n_ * 6,
                  zp_host_.begin());
        cudaMemcpy(z_prior_.data_ptr<double>(), hX_.data() + (size_t)n_ * 6,
                   (size_t)n_ * 6 * sizeof(double), cudaMemcpyHostToDevice);
        has_prior_ = true; multi_prior_ = false;
        prior_gauge_projected_ = false;
        prior_proj_lowrank_ = false;
        prior_proj_A0_host_.clear();
        prior_proj_B0_host_.clear();
        prior_proj_S0_host_.clear();
        prior_proj_S0inv_host_.clear();
        refresh_single_prior_from_pm_();  // node1 兼容路径；下一窗由 set_prior_multi 覆盖全部节点
        py::dict out_dict;
        out_dict["Pm"] = out;
        out_dict["Pg"] = Pgout;
        out_dict["Pm_all"] = Pm_all_out;
        auto _add_cross = [&]() {
            py::array_t<int> cbi_out({(py::ssize_t)cross_bi_v.size()});
            py::array_t<int> cbj_out({(py::ssize_t)cross_bj_v.size()});
            py::array_t<double> cval_out({(py::ssize_t)cross_bi_v.size(),
                                          (py::ssize_t)6, (py::ssize_t)6});
            if (!cross_bi_v.empty()) {
                std::copy(cross_bi_v.begin(), cross_bi_v.end(), cbi_out.mutable_data());
                std::copy(cross_bj_v.begin(), cross_bj_v.end(), cbj_out.mutable_data());
                std::copy(cross_val_v.begin(), cross_val_v.end(), cval_out.mutable_data());
            }
            out_dict["cross_bi"] = cbi_out;
            out_dict["cross_bj"] = cbj_out;
            out_dict["cross_val"] = cval_out;
        };
        if (!joint_exact || exact_compare) {
            out_dict["Lp_all"] = Lp_all_out;
            _add_cross();
        }
        if (exact_compare) {
            out_dict["Pm_all_local"] = Pm_all_local_out;
            out_dict["Lp_all_local"] = Lp_all_local_out;
        }
        return out_dict;
    }

    // 解 H X = U（物理 QOE 的列空间）；U=(ncol,k)，用于低秩 gauge 协方差修正。
    py::array_t<double> solve_rhs(py::array_t<double, py::array::c_style | py::array::forcecast> U) {
        auto u = U.request();
        if (u.ndim != 2 || (int)u.shape[0] != S_.ncol)
            throw std::runtime_error("solve_rhs: U 需 (ncol,k)");
        const int k = (int)u.shape[1];
        assemble();
        auto mtm = scaled_AtA_Atb();
        auto Av = mtm.first.to(torch::kCPU).contiguous();
        auto rp = A_rp_t_.to(torch::kCPU);
        auto ci = A_ci_t_.to(torch::kCPU);
        auto sc = mtm_scl_.to(torch::kCPU).contiguous();
        auto B = torch::from_blob((double *)u.ptr, {(int64_t)S_.ncol, (int64_t)k},
                                  torch::TensorOptions().dtype(torch::kFloat64)).clone()
                     * sc.view({(int64_t)S_.ncol, 1});
        auto nd = symc_->createNumericDecomposition(1);
        nd.add_MtM(Av.reshape({1, S_.nnz}), rp, ci);
        add_cross_to_nd_cpu_(nd);
        nd.factor();
        py::array_t<double> out({(py::ssize_t)S_.ncol, (py::ssize_t)k});
        auto *op = out.mutable_data();
        for (int j = 0; j < k; ++j) {
            auto yj = B.select(1, j).reshape({1, (int64_t)S_.ncol}).clone();
            nd.solve(yj);
            auto xj = yj.reshape({(int64_t)S_.ncol}) * sc;
            const double *p = xj.data_ptr<double>();
            for (int64_t a = 0; a < S_.ncol; ++a) op[a * k + j] = p[a];
        }
        return out;
    }

    // 精确节点联合信息 Λ_node = (Σ_node)^{-1}，Σ_node 为 H^{-1} 的 node-node 块。
    // 用 solve_multi 对单位右端做多右端求解；selected inverse 只在因子填充模式上
    // 精确，拿不到跨星长程块。返回物理 QOE 空间的 (6n,6n) 稠密阵。
    // 仅用于方法论验证/小窗口；6N=6912 时单次峰值约 2 GB。
    py::array_t<double> joint_node_info(int node, double eps = 0.0) {
        if (node < 0 || node >= L_) throw std::runtime_error("joint_node_info: node 越界");
        assemble();
        auto mtm = scaled_AtA_Atb();
        const int64_t ncol = S_.ncol;
        const int64_t q = (int64_t)n_ * 6;
        if (q > 12288) throw std::runtime_error("joint_node_info: 6N > 12288，拒绝分配");
        const int64_t off = (int64_t)node * q;
        auto optsD = torch::TensorOptions().dtype(torch::kFloat64);
        const bool cuda = torch::cuda::is_available();
        auto sc_cpu = mtm_scl_.to(torch::kCPU).contiguous();

        // scaled 单位右端：B[:, r] = D e_{off+r}
        auto X = torch::zeros({1, q * ncol}, optsD);
        {
            auto xa = X.accessor<double, 2>();
            auto sa = sc_cpu.accessor<double, 1>();
            for (int64_t r = 0; r < q; ++r) xa[0][r * ncol + (off + r)] = sa[off + r];
        }
        if (cuda) {
            X = X.to(torch::kCUDA);
            auto Av = mtm.first.to(torch::kCUDA).contiguous();
            auto rp = A_rp_t_.to(torch::kCUDA);
            auto ci = A_ci_t_.to(torch::kCUDA);
            auto nd = sym_->createNumericDecomposition(1);
            nd.add_MtM(Av.reshape({1, S_.nnz}), rp, ci);
            add_cross_to_nd_(nd);
            if (eps > 0.0) {                      // 物理对角正则（scaled: eps·sc²）
                auto ev_cpu = torch::zeros({1, (int64_t)ncol},
                                           torch::TensorOptions().dtype(torch::kFloat64));
                auto sa = sc_cpu.accessor<double, 1>();
                auto ea = ev_cpu.accessor<double, 2>();
                for (int64_t a = 0; a < ncol; ++a) ea[0][a] = eps * sa[a] * sa[a];
                nd.add_M(ev_cpu.to(torch::kCUDA), rp_eps_t_, ci_eps_t_);
            }
            nd.factor();
            nd.solve_multi(X, q);
        } else {
            auto Av = mtm.first.to(torch::kCPU).contiguous();
            auto rp = A_rp_t_.to(torch::kCPU);
            auto ci = A_ci_t_.to(torch::kCPU);
            auto nd = symc_->createNumericDecomposition(1);
            nd.add_MtM(Av.reshape({1, S_.nnz}), rp, ci);
            add_cross_to_nd_cpu_(nd);
            if (eps > 0.0) {
                auto ev = torch::zeros({1, (int64_t)ncol},
                                       torch::TensorOptions().dtype(torch::kFloat64));
                auto sa = sc_cpu.accessor<double, 1>();
                auto ea = ev.accessor<double, 2>();
                for (int64_t a = 0; a < ncol; ++a) ea[0][a] = eps * sa[a] * sa[a];
                nd.add_M(ev, rp_eps_t_.to(torch::kCPU), ci_eps_t_.to(torch::kCPU));
            }
            nd.factor();
            nd.solve_multi(X, q);
        }
        auto Xc = X.to(torch::kCPU).contiguous();
        const double *xp = Xc.data_ptr<double>();
        const double *sp = sc_cpu.data_ptr<double>();
        auto Sig = torch::zeros({q, q}, optsD);
        {
            auto sg = Sig.accessor<double, 2>();
            for (int64_t r = 0; r < q; ++r)
                for (int64_t a = 0; a < q; ++a)
                    sg[a][r] = sp[off + a] * xp[r * ncol + (off + a)];
        }
        Sig = 0.5 * (Sig + Sig.transpose(0, 1));
        torch::Tensor Lam;
        if (cuda) {
            auto Sg = Sig.to(torch::kCUDA);
            Lam = torch::linalg_inv(Sg).to(torch::kCPU);
        } else {
            Lam = torch::linalg_inv(Sig);
        }
        Lam = 0.5 * (Lam + Lam.transpose(0, 1));
        py::array_t<double> out({(py::ssize_t)q, (py::ssize_t)q});
        std::memcpy(out.mutable_data(), Lam.contiguous().data_ptr<double>(),
                    sizeof(double) * (size_t)q * (size_t)q);
        return out;
    }

    // 含过程噪声的精确信息预测：Λ⁺ = (Φ Σ Φᵀ + Q)^{-1}，Σ = Λ^{-1}。
    // Lam (6N,6N) 物理信息；Phi/Q (n,6,6) 逐星。返回 (6N,6N) 物理信息。
    py::array_t<double> info_predict(
        py::array_t<double, py::array::c_style | py::array::forcecast> Lam,
        py::array_t<double, py::array::c_style | py::array::forcecast> Phi,
        py::array_t<double, py::array::c_style | py::array::forcecast> Q) {
        auto rl = Lam.request(); auto rp = Phi.request(); auto rq = Q.request();
        const int64_t q = (int64_t)n_ * 6;
        if (rl.ndim != 2 || rl.shape[0] != q || rl.shape[1] != q)
            throw std::runtime_error("info_predict: Lam 需 (6N,6N)");
        if (rp.ndim != 3 || rp.shape[0] != n_ || rp.shape[1] != 6 || rp.shape[2] != 6)
            throw std::runtime_error("info_predict: Phi 需 (n,6,6)");
        if (rq.ndim != 3 || rq.shape[0] != n_ || rq.shape[1] != 6 || rq.shape[2] != 6)
            throw std::runtime_error("info_predict: Q 需 (n,6,6)");
        if (q > 12288) throw std::runtime_error("info_predict: 6N > 12288，拒绝分配");
        auto optsD = torch::TensorOptions().dtype(torch::kFloat64);
        const bool cuda = torch::cuda::is_available();
        auto dev = cuda ? torch::kCUDA : torch::kCPU;
        auto optsI = torch::TensorOptions().dtype(torch::kInt64).device(dev);
        auto T = [&](const void *p, std::initializer_list<int64_t> sh) {
            return torch::from_blob((void *)p, sh, optsD).clone().to(dev);
        };
        auto Lm = T(rl.ptr, {q, q});
        auto Ph = T(rp.ptr, {n_, 6, 6});
        auto Qm = T(rq.ptr, {n_, 6, 6});
        auto idx = torch::arange(n_, optsI).view({n_, 1, 1});
        auto a6 = torch::arange(6, optsI);
        auto rr = (idx * 6 + a6.view({1, 6, 1})).expand({n_, 6, 6});
        auto cc = (idx * 6 + a6.view({1, 1, 6})).expand({n_, 6, 6});
        auto Pb = torch::zeros({q, q}, optsD.device(dev));
        auto Qb = torch::zeros({q, q}, optsD.device(dev));
        Pb.index_put_({rr, cc}, Ph);
        Qb.index_put_({rr, cc}, Qm);
        auto Sig = torch::linalg_inv(Lm);
        auto Sp = Pb.matmul(Sig).matmul(Pb.transpose(0, 1)) + Qb;
        auto Lp = torch::linalg_inv(Sp);
        Lp = 0.5 * (Lp + Lp.transpose(0, 1));
        auto Lc = Lp.to(torch::kCPU).contiguous();
        py::array_t<double> out({(py::ssize_t)q, (py::ssize_t)q});
        std::memcpy(out.mutable_data(), Lc.data_ptr<double>(),
                    sizeof(double) * (size_t)q * (size_t)q);
        return out;
    }

    // 局部联合信息：以 (node, center) 为心、r 跳 ISL 邻域 S，取
    //   Σ_S = [H^{-1}]_{SS},  Λ_S = Σ_S^{-1},
    // 返回 (center,j) 的物理信息跨块 (B,6,6) 与 bi/bj（变量号 node*n+j），
    // 直接供 set_prior_cross 注入。S 的列用 solve_multi 求，不做全局稠密逆。
    py::tuple local_cross_info(int node, int center, int r = 1, double eps = 0.0) {
        if (node < 0 || node >= L_ || center < 0 || center >= n_ || r < 1)
            throw std::runtime_error("local_cross_info: 参数越界");
        assemble();
        auto mtm = scaled_AtA_Atb();
        const int64_t ncol = S_.ncol;
        std::vector<std::vector<int>> g((size_t)n_);
        for (int e = 0; e < E_; ++e) {
            const int a = isl_i_[e], b = isl_j_[e];
            if (a != b) { g[(size_t)a].push_back(b); g[(size_t)b].push_back(a); }
        }
        std::vector<int> dist((size_t)n_, -1);
        std::queue<int> qu; qu.push(center); dist[(size_t)center] = 0;
        while (!qu.empty()) {
            const int x = qu.front(); qu.pop();
            if (dist[(size_t)x] >= r) continue;
            for (int y : g[(size_t)x])
                if (dist[(size_t)y] < 0) { dist[(size_t)y] = dist[(size_t)x] + 1; qu.push(y); }
        }
        std::vector<int> Sset;
        for (int i = 0; i < n_; ++i) if (dist[(size_t)i] >= 0) Sset.push_back(i);
        const int64_t m = (int64_t)Sset.size() * 6;
        const int64_t off = (int64_t)node * n_ * 6;
        if (m > 12288) throw std::runtime_error("local_cross_info: 邻域太大");
        auto optsD = torch::TensorOptions().dtype(torch::kFloat64);
        const bool cuda = torch::cuda::is_available();
        auto sc_cpu = mtm_scl_.to(torch::kCPU).contiguous();
        auto X = torch::zeros({1, m * ncol}, optsD);
        {
            auto xa = X.accessor<double, 2>();
            auto sa = sc_cpu.accessor<double, 1>();
            for (int64_t rcol = 0; rcol < m; ++rcol) {
                const int64_t a = off + (int64_t)Sset[(size_t)(rcol / 6)] * 6 + (rcol % 6);
                xa[0][rcol * ncol + a] = sa[a];
            }
        }
        auto apply_eps = [&](NumericDecomposition &nd, bool is_cuda) {
            if (!(eps > 0.0)) return;
            auto ev_cpu = torch::zeros({1, (int64_t)ncol}, optsD);
            auto ea = ev_cpu.accessor<double, 2>();
            auto sa = sc_cpu.accessor<double, 1>();
            for (int64_t a = 0; a < ncol; ++a) ea[0][a] = eps * sa[a] * sa[a];
            if (is_cuda) nd.add_M(ev_cpu.to(torch::kCUDA), rp_eps_t_, ci_eps_t_);
            else nd.add_M(ev_cpu, rp_eps_t_.to(torch::kCPU), ci_eps_t_.to(torch::kCPU));
        };
        if (cuda) {
            X = X.to(torch::kCUDA);
            auto Av = mtm.first.to(torch::kCUDA).contiguous();
            auto rp = A_rp_t_.to(torch::kCUDA);
            auto ci = A_ci_t_.to(torch::kCUDA);
            auto nd = sym_->createNumericDecomposition(1);
            nd.add_MtM(Av.reshape({1, S_.nnz}), rp, ci);
            add_cross_to_nd_(nd);
            apply_eps(nd, true);
            nd.factor();
            nd.solve_multi(X, m);
        } else {
            auto Av = mtm.first.to(torch::kCPU).contiguous();
            auto rp = A_rp_t_.to(torch::kCPU);
            auto ci = A_ci_t_.to(torch::kCPU);
            auto nd = symc_->createNumericDecomposition(1);
            nd.add_MtM(Av.reshape({1, S_.nnz}), rp, ci);
            add_cross_to_nd_cpu_(nd);
            apply_eps(nd, false);
            nd.factor();
            nd.solve_multi(X, m);
        }
        auto Xc = X.to(torch::kCPU).contiguous();
        const double *xp = Xc.data_ptr<double>();
        const double *sp = sc_cpu.data_ptr<double>();
        auto Sig = torch::zeros({m, m}, optsD);
        {
            auto sg = Sig.accessor<double, 2>();
            for (int64_t rcol = 0; rcol < m; ++rcol)
                for (int64_t arow = 0; arow < m; ++arow) {
                    const int64_t a = off + (int64_t)Sset[(size_t)(arow / 6)] * 6 + (arow % 6);
                    sg[arow][rcol] = sp[a] * xp[rcol * ncol + a];
                }
        }
        Sig = 0.5 * (Sig + Sig.transpose(0, 1));
        torch::Tensor Lam;
        if (cuda) Lam = torch::linalg_inv(Sig.to(torch::kCUDA)).to(torch::kCPU);
        else Lam = torch::linalg_inv(Sig);
        Lam = 0.5 * (Lam + Lam.transpose(0, 1));
        {   // PSD 投影：局部逆在极端条件数下会出负特征值，按相对下限钳制
            auto ev = torch::linalg_eigh(Lam);
            auto w = std::get<0>(ev), V = std::get<1>(ev);
            auto fl = torch::clamp_min(w.max(), 1e-300) * 1e-12;
            w = torch::clamp_min(w, fl);
            Lam = V.matmul(torch::diag(w)).matmul(V.transpose(0, 1));
            Lam = 0.5 * (Lam + Lam.transpose(0, 1));
        }
        const double *lp = Lam.contiguous().data_ptr<double>();
        const int64_t cbase = (int64_t)(std::find(Sset.begin(), Sset.end(), center) - Sset.begin()) * 6;
        std::vector<double> vals;
        std::vector<int> bi, bj;
        for (size_t jj = 0; jj < Sset.size(); ++jj) {
            if (Sset[jj] == center) continue;
            const int64_t jbase = (int64_t)jj * 6;
            for (int a = 0; a < 6; ++a)
                for (int b = 0; b < 6; ++b)
                    vals.push_back(lp[(cbase + a) * m + (jbase + b)]);
            bi.push_back(node * n_ + center);
            bj.push_back(node * n_ + Sset[jj]);
        }
        const int64_t B = (int64_t)bi.size();
        py::array_t<double> val({(py::ssize_t)B, (py::ssize_t)6, (py::ssize_t)6});
        std::memcpy(val.mutable_data(), vals.data(), sizeof(double) * vals.size());
        py::array_t<int> ai({(py::ssize_t)B}), aj({(py::ssize_t)B});
        std::memcpy(ai.mutable_data(), bi.data(), sizeof(int) * bi.size());
        std::memcpy(aj.mutable_data(), bj.data(), sizeof(int) * bj.size());
        return py::make_tuple(val, ai, aj);
    }

    // 多中心局部联合信息的全局组装：一次分解，逐中心求 Λ_{S_r(c)}；
    // 每个无序对 (i,j) 只由较小端点 i 的邻域提供，避免重叠重复计数。
    // 返回 (val(B,6,6), bi(B,), bj(B,), s_g)；s_g 是 Gershgorin 收缩系数，
    // global_shrink!=0 时把 val 乘以 s_g，保证“对角先验+跨块”严格对角占优。
    py::tuple local_cross_info_global(int node, int r = 1, double eps = 0.0,
                                      int global_shrink = 0) {
        if (node < 0 || node >= L_ || r < 1)
            throw std::runtime_error("local_cross_info_global: 参数越界");
        assemble();
        auto mtm = scaled_AtA_Atb();
        const int64_t ncol = S_.ncol;
        const int64_t off = (int64_t)node * n_ * 6;
        auto optsD = torch::TensorOptions().dtype(torch::kFloat64);
        const bool cuda = torch::cuda::is_available();
        auto sc_cpu = mtm_scl_.to(torch::kCPU).contiguous();
        std::vector<std::vector<int>> g((size_t)n_);
        for (int e = 0; e < E_; ++e) {
            const int a = isl_i_[e], b = isl_j_[e];
            if (a != b) { g[(size_t)a].push_back(b); g[(size_t)b].push_back(a); }
        }
        auto apply_eps = [&](NumericDecomposition &nd, bool is_cuda) {
            if (!(eps > 0.0)) return;
            auto ev_cpu = torch::zeros({1, (int64_t)ncol}, optsD);
            auto ea = ev_cpu.accessor<double, 2>();
            auto sa = sc_cpu.accessor<double, 1>();
            for (int64_t a = 0; a < ncol; ++a) ea[0][a] = eps * sa[a] * sa[a];
            if (is_cuda) nd.add_M(ev_cpu.to(torch::kCUDA), rp_eps_t_, ci_eps_t_);
            else nd.add_M(ev_cpu, rp_eps_t_.to(torch::kCPU), ci_eps_t_.to(torch::kCPU));
        };
        std::unique_ptr<NumericDecomposition> nd;
        if (cuda) {
            auto Av = mtm.first.to(torch::kCUDA).contiguous();
            auto rp = A_rp_t_.to(torch::kCUDA);
            auto ci = A_ci_t_.to(torch::kCUDA);
            nd = std::make_unique<NumericDecomposition>(sym_->createNumericDecomposition(1));
            nd->add_MtM(Av.reshape({1, S_.nnz}), rp, ci);
            add_cross_to_nd_(*nd);
            apply_eps(*nd, true);
        } else {
            auto Av = mtm.first.to(torch::kCPU).contiguous();
            auto rp = A_rp_t_.to(torch::kCPU);
            auto ci = A_ci_t_.to(torch::kCPU);
            nd = std::make_unique<NumericDecomposition>(symc_->createNumericDecomposition(1));
            nd->add_MtM(Av.reshape({1, S_.nnz}), rp, ci);
            add_cross_to_nd_cpu_(*nd);
            apply_eps(*nd, false);
        }
        nd->factor();

        std::vector<double> vals;
        std::vector<int> bi, bj;
        std::vector<double> dvals((size_t)n_ * 36, 0.0);   // 与跨块同源的逐星对角信息
        // 按批合并 r 跳邻域的多右端：每个中心一次 D2H 改为每批一次，
        // 因子只建一次；CUDA 侧用一次 solve_multi 解整批 RHS。
        const int kCenterBatch = 1;
        const double *sp = sc_cpu.data_ptr<double>();
        for (int c0 = 0; c0 < n_; c0 += kCenterBatch) {
            const int c1 = std::min(n_, c0 + kCenterBatch);
            std::vector<std::vector<int>> neigh((size_t)(c1 - c0));
            int64_t total_m = 0;
            for (int center = c0; center < c1; ++center) {
                std::vector<int> dist((size_t)n_, -1);
                std::queue<int> qu; qu.push(center); dist[(size_t)center] = 0;
                while (!qu.empty()) {
                    const int x = qu.front(); qu.pop();
                    if (dist[(size_t)x] >= r) continue;
                    for (int y : g[(size_t)x])
                        if (dist[(size_t)y] < 0) {
                            dist[(size_t)y] = dist[(size_t)x] + 1; qu.push(y);
                        }
                }
                auto &S = neigh[(size_t)(center - c0)];
                for (int i = 0; i < n_; ++i) if (dist[(size_t)i] >= 0) S.push_back(i);
                total_m += (int64_t)S.size() * 6;
            }
            if (total_m <= 0) continue;
            auto X = torch::zeros({1, total_m * ncol}, optsD);
            {
                auto xa = X.accessor<double, 2>();
                int64_t rbase = 0;
                for (const auto &S : neigh) {
                    const int64_t m = (int64_t)S.size() * 6;
                    for (int64_t rcol = 0; rcol < m; ++rcol) {
                        const int64_t a = off + (int64_t)S[(size_t)(rcol / 6)] * 6 + (rcol % 6);
                        xa[0][(rbase + rcol) * ncol + a] = sp[a];
                    }
                    rbase += m;
                }
            }
            if (cuda) X = X.to(torch::kCUDA);
            {
                auto Xmut = X;
                nd->solve_multi(Xmut, total_m);
            }
            auto Xc = X.to(torch::kCPU).contiguous();
            const double *xp = Xc.data_ptr<double>();
            int64_t rbase = 0;
            for (int center = c0; center < c1; ++center) {
                const auto &Sset = neigh[(size_t)(center - c0)];
                if (Sset.size() < 2) { rbase += (int64_t)Sset.size() * 6; continue; }
                const int64_t m = (int64_t)Sset.size() * 6;
                auto Sig = torch::zeros({m, m}, optsD);
                {
                    auto sg = Sig.accessor<double, 2>();
                    for (int64_t rcol = 0; rcol < m; ++rcol)
                        for (int64_t arow = 0; arow < m; ++arow) {
                            const int64_t a = off + (int64_t)Sset[(size_t)(arow / 6)] * 6 + (arow % 6);
                            sg[arow][rcol] = sp[a] * xp[(rbase + rcol) * ncol + a];
                        }
                }
                Sig = 0.5 * (Sig + Sig.transpose(0, 1));
                torch::Tensor Lam;
                if (cuda) Lam = torch::linalg_inv(Sig.to(torch::kCUDA)).to(torch::kCPU);
                else Lam = torch::linalg_inv(Sig);
                Lam = 0.5 * (Lam + Lam.transpose(0, 1));
                {   // PSD 投影（与 local_cross_info 同口径）
                    auto ev = torch::linalg_eigh(Lam);
                    auto w = std::get<0>(ev), V = std::get<1>(ev);
                    auto fl = torch::clamp_min(w.max(), 1e-300) * 1e-12;
                    w = torch::clamp_min(w, fl);
                    Lam = V.matmul(torch::diag(w)).matmul(V.transpose(0, 1));
                    Lam = 0.5 * (Lam + Lam.transpose(0, 1));
                }
                const double *lp = Lam.contiguous().data_ptr<double>();
                const int64_t cbase =
                    (int64_t)(std::find(Sset.begin(), Sset.end(), center) - Sset.begin()) * 6;
                for (int a = 0; a < 6; ++a)
                    for (int b = 0; b < 6; ++b)
                        dvals[(size_t)center * 36 + a * 6 + b] =
                            lp[(cbase + a) * m + (cbase + b)];
                for (size_t jj = 0; jj < Sset.size(); ++jj) {
                    if (Sset[jj] <= center) continue;      // 较小端点拥有该对
                    const int64_t jbase = (int64_t)jj * 6;
                    for (int a = 0; a < 6; ++a)
                        for (int b = 0; b < 6; ++b)
                            vals.push_back(lp[(cbase + a) * m + (jbase + b)]);
                    bi.push_back(node * n_ + center);
                    bj.push_back(node * n_ + Sset[jj]);
                }
                rbase += m;
            }
        }
        // Gershgorin：对角先验 A_i=Lp_i^T Lp_i 与跨块行范数 r_i，
        // s_g=min_i 0.9 λmin(A_i)/r_i；s_g<1 时全局矩阵可能不定。
        double s_g = 1.0;
        if (!bi.empty()) {
            std::vector<double> rn((size_t)n_, 0.0), lmin((size_t)n_, 0.0);
            for (int i = 0; i < n_; ++i) {
                Eigen::Matrix<double, 6, 6> A;
                for (int a = 0; a < 36; ++a) A.data()[a] = dvals[(size_t)i * 36 + a];
                A = 0.5 * (A + A.transpose());
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> es(A);
                lmin[(size_t)i] = es.eigenvalues().minCoeff();
            }
            for (size_t q = 0; q < bi.size(); ++q) {
                Eigen::Matrix<double, 6, 6> C;
                for (int a = 0; a < 36; ++a) C.data()[a] = vals[q * 36 + a];
                Eigen::JacobiSVD<Eigen::Matrix<double, 6, 6>> sv(C);
                const double nrm = sv.singularValues()(0);
                const int i = bi[q] - node * n_, j = bj[q] - node * n_;
                if (i >= 0 && i < n_) rn[(size_t)i] += nrm;
                if (j >= 0 && j < n_) rn[(size_t)j] += nrm;
            }
            for (int i = 0; i < n_; ++i)
                if (rn[(size_t)i] > 0.0 && lmin[(size_t)i] > 0.0)
                    s_g = std::min(s_g, 0.9 * lmin[(size_t)i] / rn[(size_t)i]);
            s_g = std::max(0.0, std::min(1.0, s_g));
            if (global_shrink)
                for (double &v : vals) v *= s_g;
        }
        const int64_t B = (int64_t)bi.size();
        py::array_t<double> val({(py::ssize_t)B, (py::ssize_t)6, (py::ssize_t)6});
        if (B) std::memcpy(val.mutable_data(), vals.data(), sizeof(double) * vals.size());
        py::array_t<int> ai({(py::ssize_t)B}), aj({(py::ssize_t)B});
        if (B) {
            std::memcpy(ai.mutable_data(), bi.data(), sizeof(int) * bi.size());
            std::memcpy(aj.mutable_data(), bj.data(), sizeof(int) * bj.size());
        }
        py::array_t<double> dval({(py::ssize_t)n_, (py::ssize_t)6, (py::ssize_t)6});
        std::memcpy(dval.mutable_data(), dvals.data(), sizeof(double) * dvals.size());
        return py::make_tuple(val, ai, aj, s_g, dval);
    }

    // 不相交 r 跳簇的局部联合信息：贪心覆盖把星集划成互不相交的簇，
    // 每个簇用一次多右端解取 Σ_C=[H^{-1}]_{CC}，求逆并 PSD 投影；
    // 簇间跨块置零，组装矩阵天然块对角 PSD。返回
    //   (val(B,6,6), bi(B,), bj(B,), diag(n,6,6), cluster_id(n,))
    py::tuple local_cross_info_partition(int node, int r = 1, double eps = 0.0) {
        if (node < 0 || node >= L_ || r < 1)
            throw std::runtime_error("local_cross_info_partition: 参数越界");
        assemble();
        auto mtm = scaled_AtA_Atb();
        const int64_t ncol = S_.ncol;
        const int64_t off = (int64_t)node * n_ * 6;
        const int64_t total_m = (int64_t)n_ * 6;
        auto optsD = torch::TensorOptions().dtype(torch::kFloat64);
        const bool cuda = torch::cuda::is_available();
        auto sc_cpu = mtm_scl_.to(torch::kCPU).contiguous();
        std::vector<std::vector<int>> g((size_t)n_);
        for (int e = 0; e < E_; ++e) {
            const int a = isl_i_[e], b = isl_j_[e];
            if (a != b) { g[(size_t)a].push_back(b); g[(size_t)b].push_back(a); }
        }
        // 贪心划分：每个未分配中心取 r 跳邻域内未分配的星组成一个簇
        std::vector<int> assigned((size_t)n_, -1), cid((size_t)n_, -1);
        std::vector<std::vector<int>> clusters;
        for (int c = 0; c < n_; ++c) {
            if (assigned[(size_t)c] >= 0) continue;
            std::vector<int> dist((size_t)n_, -1);
            std::queue<int> qu; qu.push(c); dist[(size_t)c] = 0;
            while (!qu.empty()) {
                const int x = qu.front(); qu.pop();
                if (dist[(size_t)x] >= r) continue;
                for (int y : g[(size_t)x])
                    if (dist[(size_t)y] < 0) {
                        dist[(size_t)y] = dist[(size_t)x] + 1; qu.push(y);
                    }
            }
            std::vector<int> C;
            for (int i = 0; i < n_; ++i)
                if (dist[(size_t)i] >= 0 && assigned[(size_t)i] < 0) C.push_back(i);
            if (C.empty()) C.push_back(c);
            const int id = (int)clusters.size();
            for (int i : C) { assigned[(size_t)i] = id; cid[(size_t)i] = id; }
            clusters.push_back(std::move(C));
        }
        auto apply_eps = [&](NumericDecomposition &nd, bool is_cuda) {
            if (!(eps > 0.0)) return;
            auto ev_cpu = torch::zeros({1, ncol}, optsD);   // add_M 需要整窗 ncol 长度
            auto ea = ev_cpu.accessor<double, 2>();
            auto sa = sc_cpu.accessor<double, 1>();
            for (int64_t a = 0; a < ncol; ++a) ea[0][a] = eps * sa[a] * sa[a];
            if (is_cuda) nd.add_M(ev_cpu.to(torch::kCUDA), rp_eps_t_, ci_eps_t_);
            else nd.add_M(ev_cpu, rp_eps_t_.to(torch::kCPU), ci_eps_t_.to(torch::kCPU));
        };
        std::unique_ptr<NumericDecomposition> nd;
        if (cuda) {
            auto Av = mtm.first.to(torch::kCUDA).contiguous();
            auto rp = A_rp_t_.to(torch::kCUDA);
            auto ci = A_ci_t_.to(torch::kCUDA);
            nd = std::make_unique<NumericDecomposition>(sym_->createNumericDecomposition(1));
            nd->add_MtM(Av.reshape({1, S_.nnz}), rp, ci);
            add_cross_to_nd_(*nd);
            apply_eps(*nd, true);
        } else {
            auto Av = mtm.first.to(torch::kCPU).contiguous();
            auto rp = A_rp_t_.to(torch::kCPU);
            auto ci = A_ci_t_.to(torch::kCPU);
            nd = std::make_unique<NumericDecomposition>(symc_->createNumericDecomposition(1));
            nd->add_MtM(Av.reshape({1, S_.nnz}), rp, ci);
            add_cross_to_nd_cpu_(*nd);
            apply_eps(*nd, false);
        }
        nd->factor();
        auto X = torch::zeros({1, total_m * ncol}, optsD);
        {
            auto xa = X.accessor<double, 2>();
            auto sa = sc_cpu.accessor<double, 1>();
            for (int64_t rcol = 0; rcol < total_m; ++rcol)
                xa[0][rcol * ncol + (off + rcol)] = sa[off + rcol];
        }
        if (cuda) X = X.to(torch::kCUDA);
        { auto Xmut = X; nd->solve_multi(Xmut, total_m); }
        auto Xc = X.to(torch::kCPU).contiguous();
        const double *xp = Xc.data_ptr<double>();
        const double *sp = sc_cpu.data_ptr<double>();

        std::vector<double> vals, dvals((size_t)n_ * 36, 0.0);
        std::vector<int> bi, bj;
        for (const auto &C : clusters) {
            const int64_t m = (int64_t)C.size() * 6;
            auto Sig = torch::zeros({m, m}, optsD);
            {
                auto sg = Sig.accessor<double, 2>();
                for (int64_t rcol = 0; rcol < m; ++rcol)
                    for (int64_t arow = 0; arow < m; ++arow) {
                        const int64_t a = off + (int64_t)C[(size_t)(arow / 6)] * 6 + (arow % 6);
                        const int64_t rglobal = (int64_t)C[(size_t)(rcol / 6)] * 6 + (rcol % 6);
                        sg[arow][rcol] = sp[a] * xp[rglobal * ncol + a];
                    }
            }
            Sig = 0.5 * (Sig + Sig.transpose(0, 1));
            torch::Tensor Lam;
            if (cuda) Lam = torch::linalg_inv(Sig.to(torch::kCUDA)).to(torch::kCPU);
            else Lam = torch::linalg_inv(Sig);
            Lam = 0.5 * (Lam + Lam.transpose(0, 1));
            {   // PSD 投影（相对下限）
                auto ev = torch::linalg_eigh(Lam);
                auto w = std::get<0>(ev), V = std::get<1>(ev);
                auto fl = torch::clamp_min(w.max(), 1e-300) * 1e-12;
                w = torch::clamp_min(w, fl);
                Lam = V.matmul(torch::diag(w)).matmul(V.transpose(0, 1));
                Lam = 0.5 * (Lam + Lam.transpose(0, 1));
            }
            const double *lp = Lam.contiguous().data_ptr<double>();
            for (size_t ii = 0; ii < C.size(); ++ii) {
                const int gi = C[ii];
                for (int a = 0; a < 6; ++a)
                    for (int b = 0; b < 6; ++b)
                        dvals[(size_t)gi * 36 + a * 6 + b] =
                            lp[((int64_t)ii * 6 + a) * m + (int64_t)ii * 6 + b];
                for (size_t jj = ii + 1; jj < C.size(); ++jj) {
                    const int gj = C[jj];
                    for (int a = 0; a < 6; ++a)
                        for (int b = 0; b < 6; ++b)
                            vals.push_back(lp[((int64_t)ii * 6 + a) * m + (int64_t)jj * 6 + b]);
                    bi.push_back(node * n_ + gi);
                    bj.push_back(node * n_ + gj);
                }
            }
        }
        const int64_t B = (int64_t)bi.size();
        py::array_t<double> val({(py::ssize_t)B, (py::ssize_t)6, (py::ssize_t)6});
        if (B) std::memcpy(val.mutable_data(), vals.data(), sizeof(double) * vals.size());
        py::array_t<int> ai({(py::ssize_t)B}), aj({(py::ssize_t)B});
        if (B) {
            std::memcpy(ai.mutable_data(), bi.data(), sizeof(int) * bi.size());
            std::memcpy(aj.mutable_data(), bj.data(), sizeof(int) * bj.size());
        }
        py::array_t<double> dval({(py::ssize_t)n_, (py::ssize_t)6, (py::ssize_t)6});
        std::memcpy(dval.mutable_data(), dvals.data(), sizeof(double) * dvals.size());
        py::array_t<int> cida({(py::ssize_t)n_});
        std::memcpy(cida.mutable_data(), cid.data(), sizeof(int) * cid.size());
        return py::make_tuple(val, ai, aj, dval, cida);
    }

    // 含过程噪声的增广求解（式 (12)）：解
    //   [H_new+Q^-1  B; B^T  -M] [δ; y] = [g; 0],
    // 等价于 (H_new + Λ⁺) δ = g，Λ⁺=(Φ H_prev^-1 Φ^T+Q)^-1。
    // H_new/H_prev 为物理 6N×6N 稠密信息；Phi/Q 为 (n,6,6)。
    // 当前只支持小 N（稠密 CSR 模式）；生产稀疏模式后续实现。
    py::array_t<double> augmented_solve(
        py::array_t<double, py::array::c_style | py::array::forcecast> H_new,
        py::array_t<double, py::array::c_style | py::array::forcecast> H_prev,
        py::array_t<double, py::array::c_style | py::array::forcecast> Phi,
        py::array_t<double, py::array::c_style | py::array::forcecast> Q,
        py::array_t<double, py::array::c_style | py::array::forcecast> g) {
        auto rn = H_new.request(); auto rp = H_prev.request();
        auto rph = Phi.request(); auto rq = Q.request(); auto rg = g.request();
        const int64_t q = (int64_t)n_ * 6;
        if (rn.ndim != 2 || rn.shape[0] != q || rn.shape[1] != q)
            throw std::runtime_error("augmented_solve: H_new 需 (6N,6N)");
        if (rp.ndim != 2 || rp.shape[0] != q || rp.shape[1] != q)
            throw std::runtime_error("augmented_solve: H_prev 需 (6N,6N)");
        if (rph.ndim != 3 || rph.shape[0] != n_ || rph.shape[1] != 6 || rph.shape[2] != 6)
            throw std::runtime_error("augmented_solve: Phi 需 (n,6,6)");
        if (rq.ndim != 3 || rq.shape[0] != n_ || rq.shape[1] != 6 || rq.shape[2] != 6)
            throw std::runtime_error("augmented_solve: Q 需 (n,6,6)");
        if (rg.ndim != 1 || rg.shape[0] != q)
            throw std::runtime_error("augmented_solve: g 需 (6N,)");
        if (q > 384) throw std::runtime_error("augmented_solve: 当前仅支持 6N<=384（稠密模式）");
        auto optsD = torch::TensorOptions().dtype(torch::kFloat64);
        auto dev = torch::kCUDA;
        auto T = [&](const void *p, std::initializer_list<int64_t> sh) {
            return torch::from_blob((void *)p, sh, optsD).clone();
        };
        auto Hn = T(rn.ptr, {q, q});
        auto Hp = T(rp.ptr, {q, q});
        auto Ph = T(rph.ptr, {n_, 6, 6});
        auto Qm = T(rq.ptr, {n_, 6, 6});
        auto gg = T(rg.ptr, {q});
        auto optsI32 = torch::TensorOptions().dtype(torch::kInt);
        auto idx = torch::arange(n_, optsI32).view({n_, 1, 1});
        auto a6 = torch::arange(6, optsI32);
        auto rr = (idx * 6 + a6.view({1, 6, 1})).expand({n_, 6, 6});
        auto cc = (idx * 6 + a6.view({1, 1, 6})).expand({n_, 6, 6});
        auto Qinv_b = torch::zeros({q, q}, optsD);
        auto Ph_b = torch::zeros({q, q}, optsD);
        // 逐星 6×6 逆与块对角嵌入（小 N）
        for (int i = 0; i < n_; ++i) {
            auto Qi = Qm[i];
            auto Pi = torch::linalg_inv(0.5 * (Qi + Qi.transpose(0, 1)));
            Qinv_b.index_put_({rr[i], cc[i]}, Pi);
            Ph_b.index_put_({rr[i], cc[i]}, Ph[i]);
        }
        auto B = Qinv_b.matmul(Ph_b);
        auto M = Hp + Ph_b.transpose(0, 1).matmul(B);
        const int64_t N = 2 * q;
        auto Aug = torch::zeros({N, N}, optsD);
        Aug.slice(0, 0, q).slice(1, 0, q).copy_(Hn + Qinv_b);
        Aug.slice(0, 0, q).slice(1, q, N).copy_(B);
        Aug.slice(0, q, N).slice(1, 0, q).copy_(B.transpose(0, 1));
        Aug.slice(0, q, N).slice(1, q, N).copy_(-M);
        // 稠密 → CSR（小 N：保留完整模式，避免漏掉结构零）
        auto rowPtr = torch::empty({N + 1}, optsI32);
        auto colInd = torch::empty({N * N}, optsI32);
        auto vals = torch::empty({1, N * N}, optsD);
        {
            auto rpa = rowPtr.accessor<int, 1>();
            auto cia = colInd.accessor<int, 1>();
            auto va = vals.accessor<double, 2>();
            auto Aa = Aug.accessor<double, 2>();
            int64_t p = 0;
            rpa[0] = 0;
            for (int64_t i = 0; i < N; ++i) {
                for (int64_t j = 0; j < N; ++j) {
                    cia[p] = (int)j;
                    va[0][p] = Aa[i][j];
                    ++p;
                }
                rpa[i + 1] = (int)p;
            }
        }
        CusolverLUSolver lu(1, N, rowPtr.to(torch::kCUDA), colInd.to(torch::kCUDA), AMD);
        auto sing = lu.factor(vals.to(torch::kCUDA));
        if (!sing.empty() && sing[0] >= 0)
            throw std::runtime_error("augmented_solve: LU 奇异");
        auto rhs = torch::zeros({1, N}, optsD);
        rhs.slice(1, 0, q) = gg;
        rhs = rhs.to(torch::kCUDA);
        lu.solve(rhs);
        auto sol = rhs.to(torch::kCPU).contiguous();
        py::array_t<double> out({(py::ssize_t)q});
        std::memcpy(out.mutable_data(), sol.data_ptr<double>(), sizeof(double) * q);
        return out;
    }

    // 含 Q 的增广系统（块稀疏 CSR 版）：输入 H_new/H_prev 的 CSR（物理 6N×6N），
    // 按 [H_new+Q^-1, B; B^T, -(H_prev+Phi^T Q^-1 Phi)] 组装 2N×2N 的 CSR，
    // 再走 Theseus 的 cuSOLVER RF LU。支持 N=1152（2N=13824）。
    py::array_t<double> augmented_solve_sparse(
        py::array_t<int, py::array::c_style | py::array::forcecast> Hn_rp,
        py::array_t<int, py::array::c_style | py::array::forcecast> Hn_ci,
        py::array_t<double, py::array::c_style | py::array::forcecast> Hn_val,
        py::array_t<int, py::array::c_style | py::array::forcecast> Hp_rp,
        py::array_t<int, py::array::c_style | py::array::forcecast> Hp_ci,
        py::array_t<double, py::array::c_style | py::array::forcecast> Hp_val,
        py::array_t<double, py::array::c_style | py::array::forcecast> Phi,
        py::array_t<double, py::array::c_style | py::array::forcecast> Q,
        py::array_t<double, py::array::c_style | py::array::forcecast> g) {
        auto rnrp = Hn_rp.request(); auto rnci = Hn_ci.request(); auto rnv = Hn_val.request();
        auto rprp = Hp_rp.request(); auto rpci = Hp_ci.request(); auto rpv = Hp_val.request();
        auto rph = Phi.request(); auto rq = Q.request(); auto rg = g.request();
        const int64_t q = (int64_t)n_ * 6;
        if (rnrp.ndim != 1 || rnrp.shape[0] != q + 1 ||
            rnv.ndim != 1 || rnv.shape[0] != rnci.shape[0])
            throw std::runtime_error("augmented_solve_sparse: H_new CSR 形状不符");
        if (rprp.ndim != 1 || rprp.shape[0] != q + 1 ||
            rpv.ndim != 1 || rpv.shape[0] != rpci.shape[0])
            throw std::runtime_error("augmented_solve_sparse: H_prev CSR 形状不符");
        if (rph.ndim != 3 || rph.shape[0] != n_ || rph.shape[1] != 6 || rph.shape[2] != 6)
            throw std::runtime_error("augmented_solve_sparse: Phi 需 (n,6,6)");
        if (rq.ndim != 3 || rq.shape[0] != n_ || rq.shape[1] != 6 || rq.shape[2] != 6)
            throw std::runtime_error("augmented_solve_sparse: Q 需 (n,6,6)");
        if (rg.ndim != 1 || rg.shape[0] != q)
            throw std::runtime_error("augmented_solve_sparse: g 需 (6N,)");
        const int *hnrp = (const int *)rnrp.ptr, *hnci = (const int *)rnci.ptr;
        const double *hnv = (const double *)rnv.ptr;
        const int *hprp = (const int *)rprp.ptr, *hpci = (const int *)rpci.ptr;
        const double *hpv = (const double *)rpv.ptr;
        const double *phi = (const double *)rph.ptr, *qv = (const double *)rq.ptr;
        const double *gv = (const double *)rg.ptr;
        const int64_t N = 2 * q;
        std::vector<std::tuple<int, int, double>> trips;
        trips.reserve((size_t)(rnv.shape[0] + rpv.shape[0] + (int64_t)n_ * 3 * 36) + 64);
        auto addt = [&](int r, int c, double v) { trips.emplace_back(r, c, v); };
        // 左上：H_new；右下：-H_prev
        for (int64_t r = 0; r < q; ++r) {
            for (int64_t p = hnrp[r]; p < hnrp[r + 1]; ++p)
                addt((int)r, hnci[p], hnv[p]);
            for (int64_t p = hprp[r]; p < hprp[r + 1]; ++p)
                addt((int)(r + q), (int)(hpci[p] + q), -hpv[p]);
        }
        // 逐星：Q^-1、B=Q^-1 Phi、-(Phi^T Q^-1 Phi)
        for (int i = 0; i < n_; ++i) {
            Eigen::Matrix<double, 6, 6> Qi, Pi, Phi_i, Qinv, B, Md;
            for (int a = 0; a < 36; ++a) {
                Qi.data()[a] = qv[(size_t)i * 36 + a];
                Phi_i.data()[a] = phi[(size_t)i * 36 + a];
            }
            Qi = 0.5 * (Qi + Qi.transpose());
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> es(Qi);
            const auto &w = es.eigenvalues();
            const auto &V = es.eigenvectors();
            const double fl = std::max(std::max(0.0, w.maxCoeff()) * 1e-12, 1e-300);
            Qinv = V * w.cwiseMax(fl).cwiseInverse().asDiagonal() * V.transpose();
            B = Qinv * Phi_i;
            Md = Phi_i.transpose() * Qinv * Phi_i;
            const int r0 = i * 6;
            for (int a = 0; a < 6; ++a)
                for (int b = 0; b < 6; ++b) {
                    addt(r0 + a, r0 + b, Qinv(a, b));
                    addt(r0 + a, (int)q + r0 + b, B(a, b));
                    addt((int)q + r0 + a, r0 + b, B(b, a));
                    addt((int)q + r0 + a, (int)q + r0 + b, -Md(a, b));
                }
        }
        std::sort(trips.begin(), trips.end(),
                  [](const auto &a, const auto &b) {
                      return std::get<0>(a) < std::get<0>(b) ||
                             (std::get<0>(a) == std::get<0>(b) && std::get<1>(a) < std::get<1>(b));
                  });
        std::vector<int> rp((size_t)N + 1, 0), ci;
        std::vector<double> vals;
        ci.reserve(trips.size()); vals.reserve(trips.size());
        {
            size_t k = 0;
            for (int r = 0; r < N; ++r) {
                while (k < trips.size() && std::get<0>(trips[k]) == r) {
                    const int c = std::get<1>(trips[k]);
                    double v = std::get<2>(trips[k]);
                    ++k;
                    while (k < trips.size() && std::get<0>(trips[k]) == r &&
                           std::get<1>(trips[k]) == c) {
                        v += std::get<2>(trips[k]); ++k;
                    }
                    ci.push_back(c); vals.push_back(v);
                }
                rp[(size_t)r + 1] = (int)ci.size();
            }
        }
        auto optsI32 = torch::TensorOptions().dtype(torch::kInt);
        auto rp_t = torch::from_blob(rp.data(), {(py::ssize_t)rp.size()}, optsI32).clone();
        auto ci_t = torch::from_blob(ci.data(), {(py::ssize_t)ci.size()}, optsI32).clone();
        auto v_t = torch::from_blob(vals.data(), {(py::ssize_t)vals.size()},
                                    torch::TensorOptions().dtype(torch::kFloat64))
                       .clone().reshape({1, (int64_t)vals.size()});
        auto _tk = [] {
            return std::chrono::duration<double>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
        };
        const double _t0 = _tk();
        CusolverLUSolver lu(1, N, rp_t.to(torch::kCUDA), ci_t.to(torch::kCUDA), AMD);
        const double _t1 = _tk();
        auto sing = lu.factor(v_t.to(torch::kCUDA));
        const double _t2 = _tk();
        if (!sing.empty() && sing[0] >= 0)
            throw std::runtime_error("augmented_solve_sparse: LU 奇异");
        auto rhs = torch::zeros({1, N}, torch::TensorOptions().dtype(torch::kFloat64));
        auto rg_t = torch::from_blob((void *)gv, {q},
                                     torch::TensorOptions().dtype(torch::kFloat64)).clone();
        rhs.slice(1, 0, q).copy_(rg_t);
        rhs = rhs.to(torch::kCUDA);
        lu.solve(rhs);
        const double _t3 = _tk();
        fprintf(stderr, "[aug/sparse] N=%lld A_nnz=%zu LU_nnz=%lld "
                        "t_setup=%.2fs t_factor=%.2fs t_solve=%.2fs\n",
                (long long)N, vals.size(), (long long)(lu.L_nnz + lu.U_nnz),
                _t1 - _t0, _t2 - _t1, _t3 - _t2);
        auto sol = rhs.to(torch::kCPU).contiguous();
        py::array_t<double> out({(py::ssize_t)q});
        std::memcpy(out.mutable_data(), sol.data_ptr<double>(), sizeof(double) * q);
        return out;
    }

    // 指定节点的每星 6×6 后验（selected inverse 的对角块，物理 QOE 空间）。
    py::array_t<double> P_node_host(int node) {
        if (node < 0 || node >= L_) throw std::runtime_error("P_node_host: node 越界");
        assemble();
        scaled_AtA_Atb();
        auto A_cpu = A_val_.to(torch::kCPU).contiguous();
        auto sc_cpu = mtm_scl_.to(torch::kCPU).contiguous();
        auto ci_cpu = A_ci_t_.to(torch::kCPU);
        auto rp_cpu = A_rp_t_.to(torch::kCPU);
        auto As = A_cpu * sc_cpu.index_select(0, ci_cpu);
        auto nd = symc_->createNumericDecomposition(1);
        nd.add_MtM(As.reshape({1, S_.nnz}), rp_cpu, ci_cpu);
        add_cross_to_nd_cpu_(nd);
        nd.factor();
        auto perm = nd.param_to_span().to(torch::kCPU);
        auto Pd = nd.compute_diag_blocks(6).index_select(0, perm).cpu().contiguous();
        const double *pp = Pd.data_ptr<double>();
        const double *sp = sc_cpu.data_ptr<double>();
        py::array_t<double> out({(py::ssize_t)n_, (py::ssize_t)6, (py::ssize_t)6});
        auto *op = out.mutable_data();
        for (int i = 0; i < n_; ++i) {
            const int64_t b = (int64_t)node * n_ + i;
            Eigen::Matrix<double, 6, 6> P;
            for (int a = 0; a < 6; ++a)
                for (int c = 0; c < 6; ++c)
                    P(a, c) = pp[(b * 6 + a) * 6 + c] * sp[b * 6 + a] * sp[b * 6 + c];
            P = 0.5 * (P + P.transpose());
            for (int a = 0; a < 6; ++a)
                for (int c = 0; c < 6; ++c) op[(i * 6 + a) * 6 + c] = P(a, c);
        }
        return out;
    }

    // 小 N 诊断：selected inverse / r=0 Schur vs 稠密 H⁻¹（ncol≲600）。
    std::vector<double> marg_probe_dense() {
        if (S_.ncol > 600) throw std::runtime_error("marg_probe_dense: ncol 太大，仅小 N 诊断");
        assemble();
        scaled_AtA_Atb();
        auto A_cpu = A_val_.to(torch::kCPU).contiguous();
        auto sc_cpu = mtm_scl_.to(torch::kCPU).contiguous();
        auto ci_t = A_ci_t_.to(torch::kCPU);
        auto rp_t = A_rp_t_.to(torch::kCPU);
        const double *av = A_cpu.data_ptr<double>();
        const double *sc = sc_cpu.data_ptr<double>();
        const int64_t *ci = ci_t.data_ptr<int64_t>();
        const int64_t *rp = rp_t.data_ptr<int64_t>();
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(S_.ncol, S_.ncol);
        for (int r = 0; r < S_.nrows; ++r)
            for (int64_t j = rp[r]; j < rp[r + 1]; ++j) {
                const int64_t cj = ci[j];
                const double vj = av[j] * sc[cj];
                for (int64_t k = rp[r]; k < rp[r + 1]; ++k)
                    H(cj, ci[k]) += vj * av[k] * sc[ci[k]];
            }
        Eigen::MatrixXd P = H.inverse();
        py::array_t<double> Psel = P_all_host();
        auto ps = Psel.request();
        const double *psel = (const double *)ps.ptr;
        double err = 0.0, scale = 0.0;
        for (int node = 0; node < L_; ++node)
            for (int i = 0; i < n_; ++i) {
                const int64_t b = (int64_t)node * n_ + i;
                for (int a = 0; a < 6; ++a)
                    for (int c = 0; c < 6; ++c) {
                        const double ref = P(b * 6 + a, b * 6 + c) * sc[b * 6 + a] * sc[b * 6 + c];
                        const double got = psel[(((size_t)node * n_ + i) * 36) + a * 6 + c];
                        err = std::max(err, std::abs(got - ref));
                        scale = std::max(scale, std::abs(ref));
                    }
            }
        marginalize_first();
        double err0 = 0.0, scale0 = 0.0;
        for (int i = 0; i < n_; ++i) {
            const int64_t b = (int64_t)n_ + i;
            for (int a = 0; a < 6; ++a)
                for (int c = 0; c < 6; ++c) {
                    const double ref = P(b * 6 + a, b * 6 + c) * sc[b * 6 + a] * sc[b * 6 + c];
                    const double got = Pm_host_[(size_t)i * 36 + a * 6 + c];
                    err0 = std::max(err0, std::abs(got - ref));
                    scale0 = std::max(scale0, std::abs(ref));
                }
        }
        double lmin = 1e300, lmax = 0.0;
        for (int q = 0; q < S_.ncol; ++q) lmin = std::min(lmin, P(q, q)), lmax = std::max(lmax, P(q, q));
        return {err / std::max(scale, 1e-300), err0 / std::max(scale0, 1e-300), lmin, lmax};
    }

    // 只取指定节点的精确 selected inverse 块（原始参数顺序 0..(node+1)*n-1）。
    // 用于关键帧交接：只算 node1，避免 P_all 对 9n 个块做全量 Takahashi。
    // FLS_SELINV_CUDA=1 走 GPU Baspacho 分解（实测 frame0 列前代本身是 CPU Eigen，
    // 且 CUDA 因子数据布局下输出 NaN——勿启用；加速应改 frame0 只算 node1 列）。
    py::array_t<double> P_node_host_fast(int node) {
        if (node < 0 || node >= L_) throw std::runtime_error("P_node_host_fast: node 越界");
        assemble();
        scaled_AtA_Atb();
        auto sc_cpu = mtm_scl_.to(torch::kCPU).contiguous();
        const int nparam = (node + 1) * n_;
        const char *ce = std::getenv("FLS_SELINV_CUDA");
        torch::Tensor Pd;
        if (ce && std::atoi(ce) != 0) {
            auto As = A_val_ * mtm_scl_.index_select(0, A_ci_t_);
            auto nd = sym_->createNumericDecomposition(1);
            nd.add_MtM(As.reshape({1, S_.nnz}), A_rp_t_, A_ci_t_);
            add_cross_to_nd_(nd);
            nd.factor();
            Pd = nd.compute_diag_blocks_frame0(6, nparam).cpu().contiguous();
        } else {
        auto A_cpu = A_val_.to(torch::kCPU).contiguous();
        auto ci_cpu = A_ci_t_.to(torch::kCPU);
        auto rp_cpu = A_rp_t_.to(torch::kCPU);
        auto As = A_cpu * sc_cpu.index_select(0, ci_cpu);
        auto nd = symc_->createNumericDecomposition(1);
        nd.add_MtM(As.reshape({1, S_.nnz}), rp_cpu, ci_cpu);
        add_cross_to_nd_cpu_(nd);
        nd.factor();
        Pd = nd.compute_diag_blocks_frame0(6, nparam).cpu().contiguous();
        }
        const double *pp = Pd.data_ptr<double>();
        const double *sp = sc_cpu.data_ptr<double>();
        py::array_t<double> out({(py::ssize_t)n_, (py::ssize_t)6, (py::ssize_t)6});
        auto *op = out.mutable_data();
        for (int i = 0; i < n_; ++i) {
            const int64_t b = (int64_t)node * n_ + i;          // 原始参数块号
            Eigen::Matrix<double, 6, 6> P;
            for (int a = 0; a < 6; ++a)
                for (int c = 0; c < 6; ++c)
                    P(a, c) = pp[((size_t)b * 6 + a) * 6 + c] * sp[b * 6 + a] * sp[b * 6 + c];
            P = 0.5 * (P + P.transpose());
            for (int a = 0; a < 6; ++a)
                for (int c = 0; c < 6; ++c) op[(i * 6 + a) * 6 + c] = P(a, c);
        }
        return out;
    }

    // gauge 感知的精确节点边缘：与 marginalize_first_local_gauge 同一 Woodbury，
    // 只取节点 node 的每星 6×6 对角块。两条路径都从同一 A_val_/mtm_scl_ 建
    // nd（含同一 kGaugeSolveEps 的 scaled 正则行），再调用同一个
    // build_gauge_woodbury_；Dinv_hash/Dinv_fro/ku 作为同一 D 的内容指纹返回。
    //   Hwin=(I-Pi)K(I-Pi)+G Pg_eff^-1 G^T（步骤4时再含 -P ΔΛ P），
    //   [Hwin^-1]_rr=[K^-1]_rr - Utot_r D^-1 Utot_r^T,
    //   Utot=K^-1 U=[X,G]（步骤4时并 [K^-1 Y,K^-1 Z]）。
    // K 的口径与 build_gauge_woodbury_ 一致（含 kGaugeSolveEps 的 scaled 正则行）。
    // 返回 dict：P(n,6,6) 为 floor 前的对称化块；gauge_applied；
    // min_eig 为各块最小特征值；Pm_rel=max_i ||corr_i||_F/||[K^-1]_ii||_F。
    // 无 gauge 时退化为当前精确 selected inverse（不改变 P_node_host_fast 行为）。
    py::dict P_node_host_fast_gauge(int node) {
        if (node < 0 || node >= L_)
            throw std::runtime_error("P_node_host_fast_gauge: node 越界");
        py::dict out;
        if (!has_gauge_ || gauge_k_ <= 0) {
            py::array_t<double> P = P_node_host_fast(node);
            auto pr = P.request();
            const double *px = (const double *)pr.ptr;
            double lmin = std::numeric_limits<double>::infinity();
            for (int i = 0; i < n_; ++i) {
                Eigen::Matrix<double, 6, 6> Pb;
                for (int a = 0; a < 6; ++a)
                    for (int c = 0; c < 6; ++c)
                        Pb(a, c) = px[((size_t)i * 6 + a) * 6 + c];
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> es(Pb);
                lmin = std::min(lmin, es.eigenvalues().minCoeff());
            }
            out["P"] = P;
            out["gauge_applied"] = false;
            out["min_eig"] = lmin;
            out["Pm_rel"] = 0.0;
            out["woodbury_ku"] = 0;
            out["Dinv_fro"] = 0.0;
            out["Dinv_hash"] = 0;
            out["D_cond"] = std::numeric_limits<double>::quiet_NaN();
            return out;
        }
        assemble();
        auto mtm_ex = scaled_AtA_Atb();
        const int k = gauge_k_;
        const int64_t nc = (int64_t)S_.ncol;
        const int nparam = (node + 1) * n_;
        auto sc_cpu = mtm_scl_.to(torch::kCPU).contiguous();
        // 与 marginalize_first_local_gauge 同源：CUDA 符号 + 数值 factor，
        // 不再回 CPU symc_ 做 6N 分解（那是 8–10 s/窗的瓶颈）。
        auto nd = sym_->createNumericDecomposition(1);
        nd.add_MtM(mtm_ex.first.reshape({1, S_.nnz}), A_rp_t_, A_ci_t_);
        add_cross_to_nd_(nd);
        {
            auto ev = torch::full({1, nc}, kGaugeSolveEps,
                                  torch::TensorOptions().dtype(torch::kFloat64)
                                      .device(torch::kCUDA));
            nd.add_M(ev, rp_eps_t_, ci_eps_t_);
        }
        nd.factor();
        GaugeWoodburyData gw;
        if (!build_gauge_woodbury_(&nd, mtm_scl_, false, gw))
            throw std::runtime_error(
                "P_node_host_fast_gauge: gauge Woodbury D 不可逆");
        const int ku = gw.ku;
        auto Pd = nd.compute_diag_blocks_frame0(6, nparam).cpu().contiguous();
        const double *pp = Pd.data_ptr<double>();
        const double *sp = sc_cpu.data_ptr<double>();
        py::array_t<double> Pout({(py::ssize_t)n_, (py::ssize_t)6, (py::ssize_t)6});
        auto *op = Pout.mutable_data();
        double lmin = std::numeric_limits<double>::infinity();
        double rel = 0.0;
        for (int i = 0; i < n_; ++i) {
            const int64_t b = (int64_t)node * n_ + i;
            Eigen::Matrix<double, 6, 6> PK;
            for (int a = 0; a < 6; ++a)
                for (int c = 0; c < 6; ++c)
                    PK(a, c) = pp[(b * 6 + a) * 6 + c] * sp[b * 6 + a] * sp[b * 6 + c];
            PK = 0.5 * (PK + PK.transpose());
            Eigen::Matrix<double, 6, Eigen::Dynamic> Ai(6, ku);
            for (int a = 0; a < 6; ++a)
                for (int j = 0; j < ku; ++j) Ai(a, j) = gw.Utot(b * 6 + a, j);
            Eigen::Matrix<double, 6, 6> corr = Ai * gw.Dinv * Ai.transpose();
            Eigen::Matrix<double, 6, 6> P = PK - corr;
            P = 0.5 * (P + P.transpose());
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> es(P);
            lmin = std::min(lmin, es.eigenvalues().minCoeff());
            const double kn = PK.norm();
            rel = std::max(rel, kn > 0.0 ? corr.norm() / kn : 0.0);
            for (int a = 0; a < 6; ++a)
                for (int c = 0; c < 6; ++c)
                    op[((size_t)i * 6 + a) * 6 + c] = P(a, c);
        }
        out["P"] = Pout;
        out["gauge_applied"] = true;
        out["min_eig"] = lmin;
        out["Pm_rel"] = rel;
        // D4：同一 build_gauge_woodbury_ 的 D 指纹。marginalize_first_local_gauge
        // 之后调用 gauge_woodbury_diag 得到同一组 Dinv_hash/Dinv_fro/ku 即可
        // 验证两条路径没有各自形成不同的 D。
        out["woodbury_ku"] = gauge_ku_host_;
        out["Dinv_fro"] = gauge_dinv_fro_host_;
        out["Dinv_hash"] = gauge_dinv_hash_host_;
        out["D_cond"] = gauge_dinv_cond_();
        return out;
    }

    // D4/D5：node1 精确 selected-inverse 对拍辅助。P 与
    // marginalize_first_local_gauge 的 Pm 是同一个 [Hwin^-1]_11：
    //   Hwin=(I-Pi)K(I-Pi)+G Pg_eff^-1 G^T（步骤4时再加 -PΔΛP），
    //   [Hwin^-1]_rr = [K^-1]_rr - Utot_r D^-1 Utot_r^T。
    // 本函数只调用 P_node_host_fast_gauge(1)（同一 K= A^T A+eps S_c^-2、同一
    // build_gauge_woodbury_），可选传入 Pm_ref (n,6,6) 做相对误差对拍。
    py::dict P_node1_gauge_check(py::object Pm_ref = py::none()) {
        py::dict out = P_node_host_fast_gauge(1);
        out["node"] = 1;
        out["exact_selected_inverse"] = true;
        const double nan = std::numeric_limits<double>::quiet_NaN();
        out["Pm_ref_rel_fro"] = nan;
        out["Pm_ref_rel_max"] = nan;
        out["Pm_ref_abs_max"] = nan;
        out["note"] =
            "P 是 [Hwin^-1]_11 的精确 selected inverse；Dinv_hash/ku 与 "
            "gauge_woodbury_diag 对拍可确认与 marginalize 同一 D。Pm_ref 可选，"
            "预期为 marginalize_first_local_gauge 返回的 Pm(n,6,6)。"
            "注意：marginalize_first_local_gauge 末尾会把 node0 先验改成下一窗的 "
            "Lp/z；若在其之后调用本函数，fresh P 与刚返回的 Pm 不是同一 Hwin。"
            "逐位对拍应在 marginalize 之前用同一 prior 状态调用，或在新窗口重建同 "
            "prior 后再比。";
        if (!Pm_ref.is_none()) {
            using Arr = py::array_t<double, py::array::c_style | py::array::forcecast>;
            Arr Pa = out["P"].cast<Arr>();
            Arr Ra = Pm_ref.cast<Arr>();
            auto pr = Pa.request();
            auto rr = Ra.request();
            if (rr.ndim != 3 || rr.shape[0] != n_ || rr.shape[1] != 6 ||
                rr.shape[2] != 6)
                throw std::runtime_error(
                    "P_node1_gauge_check: Pm_ref 需 (n,6,6)，与 P 同形");
            if (pr.size != rr.size)
                throw std::runtime_error("P_node1_gauge_check: P/Pm_ref size 不一致");
            const double *pp = (const double *)pr.ptr;
            const double *rp = (const double *)rr.ptr;
            double sum_d2 = 0.0, sum_r2 = 0.0, abs_max = 0.0;
            double max_block_rel = 0.0;
            for (int i = 0; i < n_; ++i) {
                double bd2 = 0.0, br2 = 0.0;
                for (int a = 0; a < 36; ++a) {
                    const size_t q = (size_t)i * 36 + (size_t)a;
                    const double dv = pp[q] - rp[q];
                    bd2 += dv * dv;
                    br2 += rp[q] * rp[q];
                    abs_max = std::max(abs_max, std::abs(dv));
                }
                sum_d2 += bd2;
                sum_r2 += br2;
                if (br2 > 0.0)
                    max_block_rel =
                        std::max(max_block_rel, std::sqrt(bd2 / br2));
            }
            out["Pm_ref_abs_max"] = abs_max;
            out["Pm_ref_rel_fro"] =
                std::sqrt(sum_d2) / std::max(std::sqrt(sum_r2), 1e-300);
            out["Pm_ref_rel_max"] = max_block_rel;
        }
        return out;
    }

    // 全窗口每节点每星 6×6 后验（一次 selected inverse，供 NEES/协方差诊断）。
    py::array_t<double> P_all_host() {
        assemble();
        scaled_AtA_Atb();
        auto sc_cpu = mtm_scl_.to(torch::kCPU).contiguous();
        const char *ce = std::getenv("FLS_SELINV_CUDA");
        const bool use_cuda = (ce && std::atoi(ce) != 0);
        torch::Tensor Pd;
        if (use_cuda) {
            auto As = A_val_ * mtm_scl_.index_select(0, A_ci_t_);
            auto nd = sym_->createNumericDecomposition(1);
            nd.add_MtM(As.reshape({1, S_.nnz}), A_rp_t_, A_ci_t_);
            add_cross_to_nd_(nd);
            nd.factor();
            auto perm = nd.param_to_span();
            Pd = nd.compute_diag_blocks(6).index_select(0, perm).cpu().contiguous();
        } else {
            auto A_cpu = A_val_.to(torch::kCPU).contiguous();
            auto ci_cpu = A_ci_t_.to(torch::kCPU);
            auto rp_cpu = A_rp_t_.to(torch::kCPU);
            auto As = A_cpu * sc_cpu.index_select(0, ci_cpu);
            auto nd = symc_->createNumericDecomposition(1);
            nd.add_MtM(As.reshape({1, S_.nnz}), rp_cpu, ci_cpu);
            add_cross_to_nd_cpu_(nd);
            nd.factor();
            auto perm = nd.param_to_span().to(torch::kCPU);
            Pd = nd.compute_diag_blocks(6).index_select(0, perm).cpu().contiguous();
        }
        const double *pp = Pd.data_ptr<double>();
        const double *sp = sc_cpu.data_ptr<double>();
        py::array_t<double> out({(py::ssize_t)L_, (py::ssize_t)n_, (py::ssize_t)6,
                                 (py::ssize_t)6});
        auto *op = out.mutable_data();
        for (int node = 0; node < L_; ++node)
            for (int i = 0; i < n_; ++i) {
                const int64_t b = (int64_t)node * n_ + i;
                Eigen::Matrix<double, 6, 6> P;
                for (int a = 0; a < 6; ++a)
                    for (int c = 0; c < 6; ++c)
                        P(a, c) = pp[(b * 6 + a) * 6 + c] * sp[b * 6 + a] * sp[b * 6 + c];
                P = 0.5 * (P + P.transpose());
                for (int a = 0; a < 6; ++a)
                    for (int c = 0; c < 6; ++c)
                        op[(((size_t)node * n_ + i) * 6 + a) * 6 + c] = P(a, c);
            }
        return out;
    }

    // 纯 C++ 递推一步（不依赖 py）：x_new_in 为空时由上一历元 x_{L-1} 经 GVE 前推 1 帧（dt）。
    // 队列左移 → 全窗 rv/Ap → 装配 + GN（列均衡/Baspacho/线搜索）→
    // Schur 边缘保留历元（节点 1）→ P_marg → Λ0 白化 Lp、先验中心 z_prior。

    // ---- L=1 r=1 生产快路径 ----
    // 邻接/球结构只依赖固定 ISL 图，首次构建后缓存；A_val_/b_ 每帧只向 host
    // 拷贝一次，同一 LLT 同时给出状态增量 dx 和目标星 6×6 后验，消除
    // ekf_local_dc 与 ekf_local_handoff 的重复装配与重复分解。
    void build_ekf_local_topology(int r) {
        if (ekf_ball_r_ == r && (int)ekf_ball_cache_.size() == n_) return;
        std::vector<int> deg((size_t)n_, 0);
        for (int e = 0; e < E_; ++e) {
            const int a = isl_i_[e], b = isl_j_[e];
            if (a == b) continue;
            ++deg[(size_t)a];
            ++deg[(size_t)b];
        }
        ekf_adj_ptr_.assign((size_t)n_ + 1, 0);
        for (int i = 0; i < n_; ++i)
            ekf_adj_ptr_[(size_t)i + 1] = ekf_adj_ptr_[(size_t)i] + deg[(size_t)i];
        ekf_adj_star_.assign((size_t)ekf_adj_ptr_[(size_t)n_], 0);
        ekf_adj_eid_.assign((size_t)ekf_adj_ptr_[(size_t)n_], 0);
        std::vector<int> fill(ekf_adj_ptr_.begin(), ekf_adj_ptr_.end() - 1);
        for (int e = 0; e < E_; ++e) {
            const int a = isl_i_[e], b = isl_j_[e];
            if (a == b) continue;
            ekf_adj_star_[(size_t)fill[(size_t)a]] = b;
            ekf_adj_eid_[(size_t)fill[(size_t)a]++] = e;
            ekf_adj_star_[(size_t)fill[(size_t)b]] = a;
            ekf_adj_eid_[(size_t)fill[(size_t)b]++] = e;
        }
        ekf_gts_by_star_.assign((size_t)n_, {});
        for (int g = 0; g < G_; ++g) {
            const int s = gts_[g];
            if (s >= 0 && s < n_) ekf_gts_by_star_[(size_t)s].push_back(g);
        }
        ekf_ball_cache_.assign((size_t)n_, {});
        int m_max = 6;
        for (int i = 0; i < n_; ++i) {
            std::vector<int> B{i}, frontier{i};
            for (int lev = 0; lev < r; ++lev) {
                std::vector<int> next;
                for (int s : frontier)
                    for (int q = ekf_adj_ptr_[(size_t)s]; q < ekf_adj_ptr_[(size_t)s + 1]; ++q) {
                        const int j = ekf_adj_star_[(size_t)q];
                        if (std::find(B.begin(), B.end(), j) == B.end()) {
                            B.push_back(j);
                            next.push_back(j);
                        }
                    }
                frontier.swap(next);
            }
            std::sort(B.begin() + 1, B.end());
            m_max = std::max(m_max, 6 * (int)B.size());
            ekf_ball_cache_[(size_t)i] = std::move(B);
        }
        ekf_ball_m_max_ = m_max;
        ekf_ball_r_ = r;
    }

    // r=1：固定上界存储（最大 48 维，度>7 自动回退动态路径），
    // 单次装配 + 单次 LLT 解 [b | e_target]；返回 scaled 坐标状态增量。
    std::vector<double> ekf_local_update_r1() {
        build_ekf_local_topology(1);
        if (ekf_ball_m_max_ > 48) return {};
        auto A_cpu = A_val_.to(torch::kCPU).contiguous();
        auto b_cpu = b_.to(torch::kCPU).contiguous();
        auto sc_cpu = mtm_scl_.to(torch::kCPU).contiguous();
        const double *av = A_cpu.data_ptr<double>();
        const double *bv = b_cpu.data_ptr<double>();
        const double *sc = sc_cpu.data_ptr<double>();
        const int n6 = 6 * n_, row_prior0 = S_.nISL + S_.nGTS + S_.nDYN;
        const int kp0 = (prior_node_ >= 0) ? prior_node_ : 0;
        const int kp1 = (prior_node_ >= 0) ? prior_node_ + 1 : L_;
        using M6 = Eigen::Matrix<double, 6, 6>;
        using MatM = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 48, 48>;
        using MatR = Eigen::Matrix<double, Eigen::Dynamic, 7, 0, 48, 7>;
        if ((int)ekf_Pplus6_cache_.size() != n_) {
            ekf_Pplus6_cache_.resize((size_t)n_);
            ekf_J6_cache_.resize((size_t)n_);
            ekf_dc_buf_.assign((size_t)n_ * 6, 0.0);
        }
        std::vector<char> ok((size_t)n_, 0);
        std::vector<int> clamp_count((size_t)n_, 0);
        std::vector<double> rho_st((size_t)n_, 0.0);
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < n_; ++i) {
            try {
                const auto &B = ekf_ball_cache_[(size_t)i];
                const int M = 6 * (int)B.size();
                MatM H = MatM::Zero(M, M);
                MatR Rhs = MatR::Zero(M, 7);
                auto pof = [&](int star) {
                    for (int p = 0; p < (int)B.size(); ++p)
                        if (B[(size_t)p] == star) return p;
                    return -1;
                };
                int cidx[48];
                double cval[48];
                auto add_row = [&](int row) {
                    int nn = 0;
                    for (int64_t j = S_.A_rp[row]; j < S_.A_rp[row + 1]; ++j) {
                        const int c = S_.A_ci[j];
                        const int star = (c % n6) / 6, comp = c % 6;
                        const int p = pof(star);
                        if (p < 0) continue;
                        const int idx = p * 6 + comp;
                        const double v = av[j] * sc[c];
                        bool hit = false;
                        for (int q = 0; q < nn; ++q)
                            if (cidx[q] == idx) { cval[q] += v; hit = true; break; }
                        if (!hit && nn < 48) { cidx[nn] = idx; cval[nn] = v; ++nn; }
                    }
                    for (int a = 0; a < nn; ++a) {
                        const int ia = cidx[a];
                        const double va = cval[a];
                        Rhs(ia, 0) += va * bv[row];
                        for (int b = 0; b < nn; ++b)
                            H(ia, cidx[b]) += va * cval[b];
                    }
                };
                for (int p = 0; p < (int)B.size(); ++p) {
                    const int s = B[(size_t)p];
                    for (int q = ekf_adj_ptr_[(size_t)s]; q < ekf_adj_ptr_[(size_t)s + 1]; ++q) {
                        const int t = ekf_adj_star_[(size_t)q];
                        if (t <= s || pof(t) < 0) continue;
                        add_row(ekf_adj_eid_[(size_t)q]);
                    }
                    for (int g : ekf_gts_by_star_[(size_t)s]) add_row(S_.nISL + g);
                }
                for (int s : B)
                    for (int comp = 0; comp < 6; ++comp)
                        for (int k = kp0; k < kp1; ++k)
                            add_row(row_prior0 + (k * n_ + s) * 6 + comp);
                // 目标星先验 6×6 块（Λ_B 对星块对角）
                const int tgt = B[0];
                M6 L6 = M6::Zero();
                for (int comp = 0; comp < 6; ++comp)
                    for (int k = kp0; k < kp1; ++k) {
                        const int row = row_prior0 + (k * n_ + tgt) * 6 + comp;
                        Eigen::Matrix<double, 6, 1> v = Eigen::Matrix<double, 6, 1>::Zero();
                        for (int64_t j = S_.A_rp[row]; j < S_.A_rp[row + 1]; ++j) {
                            const int c = S_.A_ci[j];
                            if ((c % n6) / 6 == tgt) v(c % 6) += av[j] * sc[c];
                        }
                        L6 += v * v.transpose();
                    }
                L6 = 0.5 * (L6 + L6.transpose());
                // RHS 附加目标星 6 个单位列 → 直接得到 Pplus6 = H⁻¹ 的目标块
                for (int comp = 0; comp < 6; ++comp) Rhs(comp, 1 + comp) = 1.0;
                bool solved = false;
                MatR X;
                const double tr = std::max(H.diagonal().sum(), 0.0);
                double jit = std::max(tr * 1e-14, 1e-300);
                for (int t = 0; t < 4; ++t) {
                    MatM Hj = H;
                    if (t > 0) Hj.diagonal().array() += jit;
                    Eigen::LLT<MatM> llt(Hj);
                    if (llt.info() != Eigen::Success) { jit *= 1e3; continue; }
                    X = llt.solve(Rhs);
                    if (X.allFinite()) { solved = true; break; }
                    jit *= 1e3;
                }
                if (!solved) continue;
                for (int c = 0; c < 6; ++c)
                    ekf_dc_buf_[(size_t)i * 6 + c] = X(c, 0) * sc[(size_t)i * 6 + c];
                M6 P6 = X.block<6, 6>(0, 1);
                P6 = 0.5 * (P6 + P6.transpose());
                M6 J6 = P6 * L6;
                int ncl = 0;
                double raw = 0.0, st = 0.0;
                if (!fls::j6_metric_within_unit(P6, L6)) {
                    J6 = fls::contraction_clamp_6x6(J6, L6, &ncl, &raw, &st);
                    if (st > 1.0 + 1e-8) continue;
                }
                Eigen::Matrix<double, 6, 1> sd;
                for (int a = 0; a < 6; ++a) sd(a) = sc[(size_t)i * 6 + a];
                const Eigen::Matrix<double, 6, 6> Sci = sd.asDiagonal();
                const Eigen::Matrix<double, 6, 6> Sci_inv = sd.cwiseInverse().asDiagonal();
                M6 Pplus_phys = Sci * P6 * Sci;
                Pplus_phys = 0.5 * (Pplus_phys + Pplus_phys.transpose());
                ekf_Pplus6_cache_[(size_t)i] = Pplus_phys;
                ekf_J6_cache_[(size_t)i] = Sci * J6 * Sci_inv;
                clamp_count[(size_t)i] = ncl;
                rho_st[(size_t)i] = st;
                ok[(size_t)i] = 1;
            } catch (...) {
                ok[(size_t)i] = 0;
            }
        }
        int nbad = 0;
        for (int i = 0; i < n_; ++i) if (!ok[(size_t)i]) ++nbad;
        if (nbad) {
            ekf_cache_valid_ = false;
            return {};
        }
        if (std::getenv("FLS_EKFDBG")) {
            int nc = 0;
            double rmax = 0.0;
            for (int i = 0; i < n_; ++i) {
                if (clamp_count[(size_t)i] > 0) ++nc;
                rmax = std::max(rmax, rho_st[(size_t)i]);
            }
            if (nc)
                fprintf(stderr, "[ekf-local] r=1 clamp: %d/%d stars, rho_max=%.6e\n",
                        nc, n_, rmax);
        }
        ekf_cache_valid_ = true;
        ekf_cache_step_ = ekf_step_ - 1;
        return ekf_dc_buf_;
    }

    // 统一入口：r=1 单次装配+单次 LLT；失败或 r>1 回退原两遍路径。
    std::vector<double> ekf_local_update(int r) {
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<double> out;
        bool cache_ok = false;
        if (r == 1) {
            try {
                out = ekf_local_update_r1();
                cache_ok = ekf_cache_valid_ && ekf_cache_step_ == ekf_step_ - 1;
            } catch (const std::exception &e) {
                fprintf(stderr, "[ekf-local] r=1 快路径失败(%s)，回退两遍路径\n", e.what());
                out.clear();
            }
        }
        if (out.empty()) {
            out = ekf_local_dc_dynamic(r);
            cache_ok = false;
        }
        ekf_cache_valid_ = cache_ok;
        const auto t1 = std::chrono::steady_clock::now();
        ekf_prof_dc_ += 1e3 * std::chrono::duration<double>(t1 - t0).count();
        return out;
    }

    // 交接分派：r=1 且 cache 有效时只用缓存的后验块做 Φ/Q/Lp/KH 后处理。
    void ekf_local_handoff(int r, const std::vector<double> &P_before,
                           const std::vector<double> &Phi_host) {
        const bool full_diag = ekf_store_block_T_ ||
                               (std::getenv("FLS_EKF_FULL_DIAG") != nullptr);
        if (r == 1 && !full_diag && ekf_cache_valid_ && ekf_cache_step_ == ekf_step_ - 1) {
            ekf_local_handoff_cached(P_before, Phi_host);
            return;
        }
        ekf_local_handoff_full(r, P_before, Phi_host);
    }

    // 缓存后验的 L=1 后处理：Q 只算一次；Pplus 直接来自 LLT（仅对称化）；
    // Lp 由 Ppred 的 Cholesky 因子求逆得到（LpᵀLp=Ppred⁻¹），不再特征分解。
    void ekf_local_handoff_cached(const std::vector<double> &P_before,
                                  const std::vector<double> &Phi_host) {
        using M6 = Eigen::Matrix<double, 6, 6>;
        const auto _t0 = std::chrono::steady_clock::now();
        if (!ekf_qphys_valid_ || (int)ekf_qphys_cache_.size() != n_) {
            std::vector<double> qh((size_t)n_ * 36);
            cudaMemcpy(qh.data(), d_qingt_, (size_t)n_ * 36 * sizeof(double),
                       cudaMemcpyDeviceToHost);
            ekf_qphys_cache_.resize((size_t)n_);
            for (int i = 0; i < n_; ++i) {
                M6 Li;
                for (int a = 0; a < 6; ++a)
                    for (int b = 0; b < 6; ++b)
                        Li(a, b) = qh[(size_t)i * 36 + a * 6 + b];
                const M6 Lq = Li.inverse();
                M6 Q = Lq * Lq.transpose();
                ekf_qphys_cache_[(size_t)i] = 0.5 * (Q + Q.transpose());
            }
            ekf_qphys_valid_ = true;
        }
        Lp_host_.assign((size_t)n_ * 36, 0.0);
        Pm_host_.assign((size_t)n_ * 36, 0.0);
        Pplus_host_.assign((size_t)n_ * 36, 0.0);
        KH_host_.assign((size_t)n_ * 36, 0.0);
        const bool dbg = std::getenv("FLS_EKFDBG") != nullptr;
        std::vector<double> jnormh(dbg ? (size_t)n_ : 0, 0.0);
        std::vector<double> jminh(dbg ? (size_t)n_ : 0, 0.0);
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < n_; ++i) {
            M6 Pplus = ekf_Pplus6_cache_[(size_t)i];
            Pplus = 0.5 * (Pplus + Pplus.transpose());
            if (!(Pplus.diagonal().minCoeff() > 0.0))
                Pplus = fls::psd_floor_6x6(Pplus);
            M6 Phi, Q = ekf_qphys_cache_[(size_t)i];
            for (int a = 0; a < 6; ++a)
                for (int b = 0; b < 6; ++b)
                    Phi(a, b) = Phi_host[(size_t)i * 36 + a * 6 + b];
            M6 Ppred = Phi * Pplus * Phi.transpose() + Q;
            Ppred = 0.5 * (Ppred + Ppred.transpose());
            M6 Lp;
            Eigen::LLT<M6> llt(Ppred);
            if (llt.info() != Eigen::Success) {
                const M6 Pf = fls::psd_floor_6x6(Ppred);
                llt.compute(Pf);
            }
            if (llt.info() == Eigen::Success) {
                Lp = llt.matrixL().solve(M6::Identity());
            } else {
                Eigen::SelfAdjointEigenSolver<M6> es(Ppred);
                Eigen::Matrix<double, 6, 1> w = es.eigenvalues();
                const double fl = std::max(w.maxCoeff() * 1e-14, 1e-300);
                for (int k = 0; k < 6; ++k) if (!(w(k) > 0.0)) w(k) = fl;
                const auto V = es.eigenvectors();
                for (int a = 0; a < 6; ++a) {
                    const double si = 1.0 / std::sqrt(w(a));
                    for (int b = 0; b < 6; ++b) Lp(a, b) = V(b, a) * si;
                }
            }
            const M6 &Jupd = ekf_J6_cache_[(size_t)i];
            if (dbg) {
                Eigen::JacobiSVD<M6> sj(Jupd);
                M6 T6 = Phi * Jupd;
                Eigen::JacobiSVD<M6> st(T6);
                jnormh[(size_t)i] = sj.singularValues()(0);
                jminh[(size_t)i] = st.singularValues().minCoeff();
            }
            for (int a = 0; a < 6; ++a)
                for (int b = 0; b < 6; ++b) {
                    const size_t o = (size_t)i * 36 + a * 6 + b;
                    Pm_host_[o] = Ppred(a, b);
                    Pplus_host_[o] = Pplus(a, b);
                    Lp_host_[o] = Lp(a, b);
                    double kh = (a == b ? 1.0 : 0.0) - Jupd(a, b);
                    if (!std::isfinite(kh)) kh = (a == b ? 1.0 : 0.0);
                    KH_host_[o] = std::min(std::max(kh, -1e12), 1e12);
                }
        }
        ekf_jnorm_host_.assign((size_t)n_, 0.0);
        ekf_jmin_host_.assign((size_t)n_, 0.0);
        if (dbg) {
            ekf_jnorm_host_ = jnormh;
            ekf_jmin_host_ = jminh;
        }
        cudaMemcpy(Lp_.data_ptr<double>(), Lp_host_.data(), (size_t)n_ * 36 * sizeof(double),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_PmMarg_, Pm_host_.data(), (size_t)n_ * 36 * sizeof(double),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_Pplus_, Pplus_host_.data(), (size_t)n_ * 36 * sizeof(double),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_KH_, KH_host_.data(), (size_t)n_ * 36 * sizeof(double),
                   cudaMemcpyHostToDevice);
        prior_gauge_projected_ = false;
        prior_proj_lowrank_ = false;
        prior_proj_A0_host_.clear();
        prior_proj_B0_host_.clear();
        prior_proj_S0_host_.clear();
        prior_proj_S0inv_host_.clear();
        if (std::getenv("FLS_EKF_PROF")) {
            const double post_ms = 1e3 * std::chrono::duration<double>(
                std::chrono::steady_clock::now() - _t0).count();
            static int _pc = 0;
            static double _pp = 0.0, _pdc = 0.0;
            _pp += post_ms;
            _pdc += ekf_prof_dc_;
            ekf_prof_dc_ = 0.0;
            if (++_pc % 50 == 0) {
                fprintf(stderr, "[ekf/prof] dc=%.2f post=%.2f ms/frame (avg)\n",
                        _pdc / 50.0, _pp / 50.0);
                _pp = 0.0;
                _pdc = 0.0;
            }
        }
    }

    // L=1 EKF 每帧协方差：对每个星 i 在 B_i={i}∪N_r(i) 内用
    // H_B = prior+ISL+GTS（当前帧）求局部后验 Pplus；再用
    // P⁻=ΦPplusΦᵀ+Q、Jupd=Pplus·pinv(P_before)、KH=I−Jupd 交接。
    // 与逐星条件核不同：邻星观测/先验进入同一个局部块，保留互协方差。
    void ekf_local_handoff_full(int r, const std::vector<double> &P_before,
                                const std::vector<double> &Phi_host) {
        auto A_cpu = A_val_.to(torch::kCPU).contiguous();
        auto sc_cpu = mtm_scl_.to(torch::kCPU).contiguous();
        const double *av = A_cpu.data_ptr<double>();
        const double *sc = sc_cpu.data_ptr<double>();
        const int n6 = 6 * n_, row_prior0 = S_.nISL + S_.nGTS + S_.nDYN;
        using _kclk = std::chrono::steady_clock;
        auto _kms = [](auto a, auto b) {
            return 1e3 * std::chrono::duration<double>(b - a).count();
        };
        const auto _k0 = _kclk::now();
        double _tk[4] = {0.0, 0.0, 0.0, 0.0};
        int _tki = 0;
        auto _mark = [&]() { if (_tki < 4) _tk[_tki++] = _kms(_k0, _kclk::now()); };

        std::vector<std::vector<int>> nbr((size_t)n_);
        for (int e = 0; e < E_; ++e) {
            const int a = isl_i_[e], b = isl_j_[e];
            if (a != b) { nbr[a].push_back(b); nbr[b].push_back(a); }
        }
        std::vector<std::vector<int>> ball((size_t)n_);
        int m_max = 6;
        for (int i = 0; i < n_; ++i) {
            std::vector<int> B{i};
            std::vector<int> frontier{i};
            for (int lev = 0; lev < r; ++lev) {
                std::vector<int> next;
                for (int s : frontier)
                    for (int j : nbr[s])
                        if (std::find(B.begin(), B.end(), j) == B.end()) {
                            B.push_back(j); next.push_back(j);
                        }
                frontier.swap(next);
            }
            std::sort(B.begin() + 1, B.end());
            m_max = std::max(m_max, 6 * (int)B.size());
            ball[i] = std::move(B);
        }
        if (m_max > 78)
            throw std::runtime_error("ekf_local_handoff: r=2 邻域超过 78 维，请用 r=1");
        const int mm = m_max;
        _mark();
        std::vector<double> H11((size_t)n_ * mm * mm, 0.0);
        std::vector<double> Hprior((size_t)n_ * mm * mm, 0.0);

#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < n_; ++i) {
            const auto &B = ball[i];
            const int M = 6 * (int)B.size();
            auto pof = [&](int star) {
                for (int p = 0; p < (int)B.size(); ++p)
                    if (B[p] == star) return p;
                return -1;
            };
            double *h11 = H11.data() + (size_t)i * mm * mm;
            double *hpr = Hprior.data() + (size_t)i * mm * mm;
            auto add_row = [&](int row, double *hacc) {
                std::vector<std::pair<int, double>> nz;
                for (int64_t j = S_.A_rp[row]; j < S_.A_rp[row + 1]; ++j) {
                    const int c = S_.A_ci[j];
                    const int star = (c % n6) / 6, comp = c % 6;
                    const int p = pof(star);
                    if (p < 0) continue;
                    const int idx = p * 6 + comp;
                    const double v = av[j] * sc[c];
                    bool hit = false;
                    for (auto &q : nz) if (q.first == idx) { q.second += v; hit = true; break; }
                    if (!hit) nz.emplace_back(idx, v);
                }
                for (const auto &a : nz)
                    for (const auto &b : nz) hacc[(a.first) * mm + b.first] +=
                        a.second * b.second;
            };
            for (int e = 0; e < E_; ++e)
                if (pof(isl_i_[e]) >= 0 && pof(isl_j_[e]) >= 0) add_row(e, h11);
            for (int g = 0; g < G_; ++g)
                if (pof(gts_[g]) >= 0) add_row(S_.nISL + g, h11);
            const int kp0 = (prior_node_ >= 0) ? prior_node_ : 0;
            const int kp1 = (prior_node_ >= 0) ? prior_node_ + 1 : L_;
            for (int s : B)
                for (int comp = 0; comp < 6; ++comp)
                    for (int k = kp0; k < kp1; ++k) {
                        const int row = row_prior0 + (k * n_ + s) * 6 + comp;
                        add_row(row, h11);
                        add_row(row, hpr);      // J_B = Pplus_B Λ_prior_B 需要先验块
                    }
            for (int a = M; a < mm; ++a) {
                h11[a * mm + a] = 1.0;          // padding：补齐维用单位阵
                hpr[a * mm + a] = 1.0;          // 先验也必须 PD（补齐维解耦）
            }
        }
        _mark();
        // 每星对完整局部 joint H_B 求 J_B=Pplus_B Λ_prior_B（保留交叉星项），
        // 取目标星 6×6 对角块进 T 链；同时记录 ||J6|| 与 T6 最小奇异值。
        // 计算在 scaled 空间做（QOE 混合量纲 cond~1e14），再按 S_c 折回物理：
        //   Pplus_phys=S_c Pplus_s S_c, J6_phys=S_c J6_s S_c^-1, T6_phys=Φ J6_phys.
        std::vector<Eigen::Matrix<double, 6, 6>> Pplus6(n_), J6h(n_);
        std::vector<double> jnormh(n_, 0.0), jminh(n_, 0.0), jrad6h(n_, 0.0),
            jraw6h(n_, 0.0), jprinc6h(n_, 0.0);
        std::vector<int> jclamp6h(n_, 0);
        std::vector<char> hok(n_, 0);
        if (ekf_store_block_T_) {
            ekf_ball_host_.resize((size_t)n_);
            ekf_TB_host_.resize((size_t)n_);
        }
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < n_; ++i) {
            try {
                const int M = 6 * (int)ball[i].size();
                Eigen::MatrixXd Hb(M, M), Lb(M, M);
                const double *h11 = H11.data() + (size_t)i * mm * mm;
                const double *hpr = Hprior.data() + (size_t)i * mm * mm;
                for (int a = 0; a < M; ++a)
                    for (int b = 0; b < M; ++b) {
                        Hb(a, b) = h11[(size_t)a * mm + b];
                        Lb(a, b) = hpr[(size_t)a * mm + b];
                    }
                Eigen::Matrix<double, 6, 6> Phi;
                for (int a = 0; a < 6; ++a)
                    for (int b = 0; b < 6; ++b)
                        Phi(a, b) = Phi_host[(size_t)i * 36 + a * 6 + b];
                // 生产快路径只求目标星 6×6 边缘；完整 J_B/T_B/Qeff 只在
                // ekf_store_block_T_ 或 FLS_EKF_FULL_DIAG=1 时计算。
                const bool full_diag = ekf_store_block_T_ ||
                                       (std::getenv("FLS_EKF_FULL_DIAG") != nullptr);
                Eigen::Matrix<double, 6, 6> P6s, J6s;
                double rho_princ = 0.0, rho_raw = 0.0, rho_st = 0.0;
                int clamp6 = 0;
                if (full_diag) {
                    const auto res = fls::local_handoff_blocks(Hb, Lb, Phi, 0);
                    if (!res.contraction_radius) {
                        fprintf(stderr,
                                "[ekf-local] J_B 谱半径>1: i=%d rho=%.6e "
                                "||J_B||_metric=%.6e ||J_B||_2=%.6e\n",
                                i, res.jradius, res.jnorm_metric, res.jnorm_euclid);
                        hok[i] = 0;
                        continue;
                    }
                    if (!res.contraction)
                        fprintf(stderr,
                                "[ekf-local] 警告: i=%d scaled 度量范数=%.6e>1"
                                "（谱半径 %.6e，仅坐标相似放大）\n",
                                i, res.jnorm_metric, res.jradius);
                    P6s = res.Pplus6; J6s = res.J6_stable;
                    rho_princ = res.j6_radius_principal;
                    rho_raw = res.j6_radius_raw; rho_st = res.j6_radius_stable;
                    clamp6 = res.j6_clamped;
                    if (ekf_store_block_T_) {
                        const int MM = 6 * (int)ball[i].size();
                        ekf_ball_host_[i] = ball[i];
                        ekf_TB_host_[i].resize((size_t)MM * MM);
                        for (int aa = 0; aa < MM; ++aa)
                            for (int bb = 0; bb < MM; ++bb)
                                ekf_TB_host_[i][(size_t)aa * MM + bb] = res.T_B(aa, bb);
                    }
                } else {
                    const auto r6 = fls::local_handoff_target6(Hb, Lb, Phi, 0);
                    P6s = r6.Pplus6; J6s = r6.J6_stable;
                    rho_princ = r6.j6_radius_principal;
                    rho_raw = r6.j6_radius_raw; rho_st = r6.j6_radius_stable;
                    clamp6 = r6.j6_clamped;
                }
                Eigen::Matrix<double, 6, 1> sd;
                for (int a = 0; a < 6; ++a) sd(a) = sc[i * 6 + a];
                const Eigen::Matrix<double, 6, 6> Sci = sd.asDiagonal();
                const Eigen::Matrix<double, 6, 6> Sci_inv = sd.cwiseInverse().asDiagonal();
                Eigen::Matrix<double, 6, 6> Pplus_phys = Sci * P6s * Sci;
                Eigen::Matrix<double, 6, 6> J6_phys = Sci * J6s * Sci_inv;
                Pplus_phys = 0.5 * (Pplus_phys + Pplus_phys.transpose());
                const Eigen::Matrix<double, 6, 6> T6_phys = Phi * J6_phys;
                Eigen::JacobiSVD<Eigen::Matrix<double, 6, 6>> sj(J6_phys);
                Eigen::JacobiSVD<Eigen::Matrix<double, 6, 6>> st(T6_phys);
                Eigen::EigenSolver<Eigen::Matrix<double, 6, 6>> ej6(J6_phys, false);
                Pplus6[i] = Pplus_phys;
                J6h[i] = J6_phys;
                jnormh[i] = sj.singularValues()(0);
                jminh[i] = st.singularValues().minCoeff();
                jrad6h[i] = rho_st;
                jraw6h[i] = rho_raw;
                jprinc6h[i] = rho_princ;
                jclamp6h[i] = clamp6;
                if (ej6.info() == Eigen::Success)
                    jrad6h[i] = std::max(jrad6h[i], ej6.eigenvalues().cwiseAbs().maxCoeff());
                hok[i] = 1;
            } catch (const std::exception &e) {
                fprintf(stderr, "[ekf-local] i=%d 失败: %s\n", i, e.what());
                hok[i] = 0;
            }
        }
        for (int i = 0; i < n_; ++i)
            if (!hok[i])
                throw std::runtime_error(
                    "ekf_local_handoff: 局部 joint 更新失败（J_B 非收缩或 H_B 非正定）");
        _mark();
        // 生产 T 链用 J6_stable（Schur-consistent + Λ6 度量截断）。截断只吸收
        // 数值越界；若截断后仍有 rho>1，说明 Λ6/J6 数据本身不满足理论前提，
        // 必须硬失败并报告，不得继续用非收缩 T。
        int nbad6 = 0, nclamped = 0;
        double rho6max = 0.0, rho6rawmax = 0.0, rho6princmax = 0.0;
        int bad_i = -1;
        for (int i = 0; i < n_; ++i) {
            rho6max = std::max(rho6max, jrad6h[i]);
            rho6rawmax = std::max(rho6rawmax, jraw6h[i]);
            rho6princmax = std::max(rho6princmax, jprinc6h[i]);
            if (jclamp6h[i] > 0) ++nclamped;
            if (jrad6h[i] > 1.0 + 1e-8) {
                if (bad_i < 0) bad_i = i;
                ++nbad6;
            }
        }
        if ((nclamped > 0 || rho6princmax > 1.0 + 1e-8) &&
            std::getenv("FLS_EKFDBG"))
            fprintf(stderr,
                    "[ekf-local] J6 per-star 稳定化: %d/%d 星被截断（principal rho "
                    "max=%.6e，Schur-marginal raw rho max=%.6e，stable rho max=%.6e）；"
                    "生产 T 使用 J6_stable（交叉星项经 Schur 边缘消去）\n",
                    nclamped, n_, rho6princmax, rho6rawmax, rho6max);
        if (nbad6 > 0)
            throw std::runtime_error(
                "ekf_local_handoff: Schur 稳定化后仍有 rho(J6_stable)>1"
                "（i=" + std::to_string(bad_i) + ", bad=" + std::to_string(nbad6) +
                "/" + std::to_string(n_) + ", rho_max=" + std::to_string(rho6max) +
                "）；Λ6 度量/先验块不一致，拒绝非收缩 T 链");
        // 截断只应吸收数值越界。若边缘因子 raw rho 已远超 1，说明局部先验块
        // 与 Λ6 不一致（或 Pplus6 非后验），截断会掩盖真实错误，必须硬失败。
        if (rho6rawmax > 1.5)
            throw std::runtime_error(
                "ekf_local_handoff: Schur-marginal J6 raw rho=" +
                std::to_string(rho6rawmax) +
                " > 1.5；Λ6/Pplus6 不一致，拒绝用谱截断掩盖");
        ekf_jnorm_host_ = jnormh;
        ekf_jmin_host_ = jminh;

        if (std::getenv("FLS_EKFDBG") && n_ > 1048) {
            for (int ii : {0, 1048}) {
            Eigen::Matrix<double, 6, 6> Hp = Eigen::Matrix<double, 6, 6>::Zero();
            const int kp = (prior_node_ >= 0) ? prior_node_ : 0;
            for (int comp = 0; comp < 6; ++comp) {
                const int row = row_prior0 + (kp * n_ + ii) * 6 + comp;
                Eigen::Matrix<double, 6, 1> v = Eigen::Matrix<double, 6, 1>::Zero();
                for (int64_t j = S_.A_rp[row]; j < S_.A_rp[row + 1]; ++j) {
                    const int c = S_.A_ci[j];
                    if (c / 6 == ii) v(c % 6) = av[j] * sc[c];
                }
                Hp += v * v.transpose();
            }
            Eigen::Matrix<double, 6, 6> Pb;
            for (int a = 0; a < 6; ++a)
                for (int b = 0; b < 6; ++b)
                    Pb(a, b) = P_before[(size_t)ii * 36 + a * 6 + b];
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> ehp(Hp), ebi(Pb.inverse());
            fprintf(stderr, "[localp] i=%d Hp w=[%.2e %.2e %.2e %.2e %.2e %.2e] "
                    "Pbinv w=[%.2e %.2e %.2e %.2e %.2e %.2e]\n", ii,
                    ehp.eigenvalues()(0), ehp.eigenvalues()(1), ehp.eigenvalues()(2),
                    ehp.eigenvalues()(3), ehp.eigenvalues()(4), ehp.eigenvalues()(5),
                    ebi.eigenvalues()(0), ebi.eigenvalues()(1), ebi.eigenvalues()(2),
                    ebi.eigenvalues()(3), ebi.eigenvalues()(4), ebi.eigenvalues()(5));
            }
        }

        std::vector<double> qh((size_t)n_ * 36);
        cudaMemcpy(qh.data(), d_qingt_, (size_t)n_ * 36 * sizeof(double),
                   cudaMemcpyDeviceToHost);
        Lp_host_.assign((size_t)n_ * 36, 0.0);
        Pm_host_.assign((size_t)n_ * 36, 0.0);
        Pplus_host_.assign((size_t)n_ * 36, 0.0);
        KH_host_.assign((size_t)n_ * 36, 0.0);
        for (int i = 0; i < n_; ++i) {
            Eigen::Matrix<double, 6, 6> Pplus = Pplus6[i], Pb, Phi, Li, Q;
            for (int a = 0; a < 6; ++a)
                for (int b = 0; b < 6; ++b) {
                    Pb(a, b) = P_before[(size_t)i * 36 + a * 6 + b];
                    Phi(a, b) = Phi_host[(size_t)i * 36 + a * 6 + b];
                    Li(a, b) = qh[(size_t)i * 36 + a * 6 + b];
                }
            Pplus = 0.5 * (Pplus + Pplus.transpose());
            {
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> es(Pplus);
                Eigen::Matrix<double, 6, 1> w = es.eigenvalues();
                const double fl = std::max(w.maxCoeff() * 1e-14, 1e-300);
                for (int k = 0; k < 6; ++k)
                    if (!(w(k) > 0.0)) w(k) = fl;
                Pplus = es.eigenvectors() * w.asDiagonal() * es.eigenvectors().transpose();
            }
            const Eigen::Matrix<double, 6, 6> Lq = Li.inverse();
            Q = Lq * Lq.transpose();
            Eigen::Matrix<double, 6, 6> Ppred = Phi * Pplus * Phi.transpose() + Q;
            Ppred = 0.5 * (Ppred + Ppred.transpose());
            Eigen::Matrix<double, 6, 6> Lp;
            {
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> es(Ppred);
                Eigen::Matrix<double, 6, 1> w = es.eigenvalues();
                const double fl = std::max(w.maxCoeff() * 1e-14, 1e-300);
                for (int k = 0; k < 6; ++k)
                    if (!(w(k) > 0.0)) w(k) = fl;
                const auto V = es.eigenvectors();
                for (int a = 0; a < 6; ++a) {
                    const double si = 1.0 / std::sqrt(w(a));
                    for (int b = 0; b < 6; ++b) Lp(a, b) = V(b, a) * si;
                }
            }
            // Jupd = J_B 的目标星对角块（Pplus_B Λ_prior_B），已在 scaled 空间
            // 用完整局部 joint 求得并折回物理；不是 Pplus_ii * P_before_i^-1。
            const Eigen::Matrix<double, 6, 6> &Jupd = J6h[i];
            if ((i == 0 || i == 1048) && std::getenv("FLS_EKFDBG")) {
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> ep(Pplus), eb(Pb);
                Eigen::EigenSolver<Eigen::Matrix<double, 6, 6>> ej(Jupd);
                fprintf(stderr, "[localJ] i=%d P+ w=[%.2e %.2e %.2e %.2e %.2e %.2e] "
                        "Pb w=[%.2e %.2e %.2e %.2e %.2e %.2e] J re=[%.2e %.2e %.2e %.2e %.2e %.2e] "
                        "||J6||=%.3e tmin=%.3e\n",
                        i, ep.eigenvalues()(0), ep.eigenvalues()(1), ep.eigenvalues()(2),
                        ep.eigenvalues()(3), ep.eigenvalues()(4), ep.eigenvalues()(5),
                        eb.eigenvalues()(0), eb.eigenvalues()(1), eb.eigenvalues()(2),
                        eb.eigenvalues()(3), eb.eigenvalues()(4), eb.eigenvalues()(5),
                        ej.eigenvalues()(0).real(), ej.eigenvalues()(1).real(),
                        ej.eigenvalues()(2).real(), ej.eigenvalues()(3).real(),
                        ej.eigenvalues()(4).real(), ej.eigenvalues()(5).real(),
                        jnormh[i], jminh[i]);
            }
            for (int a = 0; a < 6; ++a)
                for (int b = 0; b < 6; ++b) {
                    const size_t o = (size_t)i * 36 + a * 6 + b;
                    Pm_host_[o] = Ppred(a, b);
                    Pplus_host_[o] = Pplus(a, b);
                    Lp_host_[o] = Lp(a, b);
                    double kh = (a == b ? 1.0 : 0.0) - Jupd(a, b);
                    if (!std::isfinite(kh)) kh = (a == b ? 1.0 : 0.0);
                    KH_host_[o] = std::min(std::max(kh, -1e12), 1e12);
                }
        }
        cudaMemcpy(Lp_.data_ptr<double>(), Lp_host_.data(), (size_t)n_ * 36 * sizeof(double),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_PmMarg_, Pm_host_.data(), (size_t)n_ * 36 * sizeof(double),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_Pplus_, Pplus_host_.data(), (size_t)n_ * 36 * sizeof(double),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_KH_, KH_host_.data(), (size_t)n_ * 36 * sizeof(double),
                   cudaMemcpyHostToDevice);
        _mark();
        prior_gauge_projected_ = false;   // Lp_ 已由 Ppred 重写（未投影）
        prior_proj_lowrank_ = false;      // 旧 A0/B0 随 Lp_ 重写失效
        prior_proj_A0_host_.clear();
        prior_proj_B0_host_.clear();
        prior_proj_S0_host_.clear();
        prior_proj_S0inv_host_.clear();
        if (std::getenv("FLS_EKF_PROF")) {
            static int _pc = 0;
            static double _pa = 0, _pb = 0, _pc2 = 0, _pd = 0;
            _pa += _tk[0];
            _pb += (_tk[1] - _tk[0]);
            _pc2 += (_tk[2] - _tk[1]);
            _pd += (_tk[3] - _tk[2]);
            if (++_pc % 50 == 0) {
                fprintf(stderr,
                        "[ekf/prof] ball=%.2f asm=%.2f par=%.2f post=%.2f ms/frame (avg)\n",
                        _pa / 50.0, _pb / 50.0, _pc2 / 50.0, _pd / 50.0);
                _pa = _pb = _pc2 = _pd = 0.0;
            }
        }
    }


    // r=1/2 局部 EKF 状态更新：每星只解自己的邻域 H_B dx = A_B^T b，
    // 取目标星 6 维增量。用于替代 L=1 每帧的 6N 全局 GN 因子。
    std::vector<double> ekf_local_dc_dynamic(int r) {
        auto A_cpu = A_val_.to(torch::kCPU).contiguous();
        auto b_cpu = b_.to(torch::kCPU).contiguous();
        auto sc_cpu = mtm_scl_.to(torch::kCPU).contiguous();
        const double *av = A_cpu.data_ptr<double>();
        const double *bv = b_cpu.data_ptr<double>();
        const double *sc = sc_cpu.data_ptr<double>();
        const int n6 = 6 * n_, row_prior0 = S_.nISL + S_.nGTS + S_.nDYN;
        std::vector<std::vector<int>> nbr((size_t)n_);
        for (int e = 0; e < E_; ++e) {
            const int a = isl_i_[e], b = isl_j_[e];
            if (a != b) { nbr[a].push_back(b); nbr[b].push_back(a); }
        }
        std::vector<std::vector<int>> ball((size_t)n_);
        for (int i = 0; i < n_; ++i) {
            std::vector<int> B{i}, frontier{i};
            for (int lev = 0; lev < r; ++lev) {
                std::vector<int> next;
                for (int s : frontier)
                    for (int j : nbr[s])
                        if (std::find(B.begin(), B.end(), j) == B.end()) {
                            B.push_back(j); next.push_back(j);
                        }
                frontier.swap(next);
            }
            std::sort(B.begin() + 1, B.end());
            ball[i] = std::move(B);
        }
        std::vector<double> dc((size_t)n_ * 6, 0.0);
        std::vector<char> ok(n_, 0);
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < n_; ++i) {
            try {
                const auto &B = ball[i];
                const int M = 6 * (int)B.size();
                auto pof = [&](int star) {
                    for (int p = 0; p < (int)B.size(); ++p)
                        if (B[p] == star) return p;
                    return -1;
                };
                std::vector<double> H((size_t)M * M, 0.0), rhs((size_t)M, 0.0);
                auto add_row = [&](int row) {
                    std::vector<std::pair<int, double>> nz;
                    for (int64_t j = S_.A_rp[row]; j < S_.A_rp[row + 1]; ++j) {
                        const int c = S_.A_ci[j];
                        const int star = (c % n6) / 6, comp = c % 6;
                        const int p = pof(star);
                        if (p < 0) continue;
                        const int idx = p * 6 + comp;
                        const double v = av[j] * sc[c];
                        bool hit = false;
                        for (auto &q : nz)
                            if (q.first == idx) { q.second += v; hit = true; break; }
                        if (!hit) nz.emplace_back(idx, v);
                    }
                    for (const auto &a : nz) {
                        rhs[(size_t)a.first] += a.second * bv[row];
                        for (const auto &b : nz)
                            H[(size_t)a.first * M + b.first] += a.second * b.second;
                    }
                };
                for (int e = 0; e < E_; ++e)
                    if (pof(isl_i_[e]) >= 0 && pof(isl_j_[e]) >= 0) add_row(e);
                for (int g = 0; g < G_; ++g)
                    if (pof(gts_[g]) >= 0) add_row(S_.nISL + g);
                const int kp0 = (prior_node_ >= 0) ? prior_node_ : 0;
                const int kp1 = (prior_node_ >= 0) ? prior_node_ + 1 : L_;
                for (int s : B)
                    for (int comp = 0; comp < 6; ++comp)
                        for (int k = kp0; k < kp1; ++k)
                            add_row(row_prior0 + (k * n_ + s) * 6 + comp);
                Eigen::MatrixXd Hm(M, M);
                for (int a = 0; a < M; ++a)
                    for (int b = 0; b < M; ++b) Hm(a, b) = H[(size_t)a * M + b];
                const double tr = std::max(Hm.trace(), 0.0);
                Eigen::VectorXd rv(M);
                for (int a = 0; a < M; ++a) rv(a) = rhs[(size_t)a];
                Eigen::VectorXd dx;
                bool solved = false;
                double jit = std::max(tr * 1e-14, 1e-300);
                for (int t = 0; t < 4 && !solved; ++t) {
                    Eigen::MatrixXd Hj = Hm;
                    Hj.diagonal().array() += (t == 0 ? 0.0 : jit);
                    Eigen::LLT<Eigen::MatrixXd> llt(Hj);
                    if (llt.info() != Eigen::Success) { jit *= 1e3; continue; }
                    dx = llt.solve(rv);
                    solved = dx.allFinite();
                    if (!solved) { jit *= 1e3; continue; }
                }
                if (!solved) {
                    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Hm);
                    if (es.info() != Eigen::Success) continue;
                    Eigen::VectorXd w = es.eigenvalues();
                    const double fl = std::max(w.maxCoeff() * 1e-12, 1e-300);
                    for (int q = 0; q < w.size(); ++q) if (!(w(q) > fl)) w(q) = fl;
                    dx = es.eigenvectors() * w.cwiseInverse().asDiagonal() *
                         es.eigenvectors().transpose() * rv;
                    solved = dx.allFinite();
                }
                if (!solved) continue;
                // H 用 A*sc 装配，解在 scaled 坐标；折回物理 QOE 增量。
                for (int c = 0; c < 6; ++c)
                    dc[(size_t)i * 6 + c] = dx(c) * sc[i * 6 + c];
                ok[i] = 1;
            } catch (...) {
                ok[i] = 0;
            }
        }
        int nbad = 0;
        for (int i = 0; i < n_; ++i) if (!ok[i]) ++nbad;
        if (nbad)
            fprintf(stderr, "[ekf-local] local dc fallback: %d/%d 星 H_B 非正定，增量置零\n",
                    nbad, n_);
        return dc;
    }

    void step(const double *x_new_in, int steps) {
        if (has_gauge_)
            throw std::runtime_error(
                "FlsWindow::step: window gauge 仅支持 solve_batch"
                "（step 的 GN/Atb 未接 gauge Woodbury，继续会静默用未投影 Hessian）");
        using clk = std::chrono::steady_clock;
        auto _now = [] { return std::chrono::duration<double>(clk::now().time_since_epoch()).count(); };
        const int _ekf_this = ekf_step_++;
        ekf_cache_valid_ = false;
        double t0 = _now(), t1 = 0, t_pred = 0, t_geo = 0, t_asm = 0, t_fac = 0, t_solve = 0,
               t_ls = 0, t_schur = 0, t_gn = 0;
        dbg_mtm = dbg_fac = dbg_solve = 0;      // 每步归因（GN 内：列均衡/因子/求解）
        qoejopt::winResetTiming();              // 装配内归因（dyn STM / 装配核 / res2）
        std::vector<double> P_before = Pm_host_; // L=1：本步前的 P（host）
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
        if (L_ == 1 && !skip_update_) {
            std::copy(hX_.begin(), hX_.begin() + (size_t)n_ * 6, zp_host_.begin());
            cudaMemcpy(z_prior_.data_ptr<double>(), hX_.data(), (size_t)n_ * 6 * sizeof(double),
                       cudaMemcpyHostToDevice);
        }
        t1 = _now(); geo_all(); t_geo += _now() - t1;
        // 关键帧 skip（历元 b）：观测被 mask，不走 A/GN。区间量按关键帧历元锚定：
        //   T_host  = ekf_T_chain_ = T_{a→b}（δx⁺_a → δx⁻_b，含首因子 Φ_a、不含 Φ_b）
        //   P_prev  = P⁻_a、P_pred = P⁻_b = P_before（与 X_host 同历元）
        //   Qeff    = P⁻_b − T·P⁻_a·Tᵀ（固定缩放空间计算，PSD）
        // 随后才做 P⁻_{b+1} = Φ_bP⁻_bΦᵀ+Q 的预测（下一普通帧先验用），并把
        // chain 重置为 Φ_b 作为下一区间起点。Python 侧不得再调 reset_T。
        if (skip_update_ && L_ == 1) {
            qoejopt::gveStmDeviceGrp(wc_.d_X, n_, dt_, 1.0, d_end_, d_phi_);
            std::vector<double> Phi_host((size_t)n_ * 36);
            cudaMemcpy(Phi_host.data(), d_phi_, (size_t)n_ * 36 * sizeof(double),
                       cudaMemcpyDeviceToHost);
            phi_host_ = Phi_host;
            ekf_p_prev_ = ekf_p_prev_key_.empty() ? P_before : ekf_p_prev_key_;
            ekf_p_pred_ = P_before;                       // P⁻_b（历元 b）
            ekf_qeff_.assign((size_t)n_ * 36, 0.0);
            // 物理过程 Q 的 scaled 迹（Qeff 下限用；Q=Li^-1 Li^-T）
            std::vector<double> qh((size_t)n_ * 36, 0.0);
            cudaMemcpy(qh.data(), d_qingt_, (size_t)n_ * 36 * sizeof(double),
                       cudaMemcpyDeviceToHost);
            double qlmin = 1e300, qscale = 0.0;
            for (int i = 0; i < n_; ++i) {
                // Qeff = Ppred − T·Pprev·Tᵀ 在固定缩放空间计算（QOE 混合量纲下
                // 大数相减的消元误差可达 λmin ~ −1e12）；结果折回 QOE 存 ekf_qeff_。
                Eigen::Matrix<double, 6, 1> dd, di;
                for (int c = 0; c < 6; ++c) {
                    dd(c) = d_ekf_host_[(size_t)i * 6 + c];
                    di(c) = 1.0 / dd(c);
                }
                const Eigen::Matrix<double, 6, 6> D = dd.asDiagonal();
                const Eigen::Matrix<double, 6, 6> Di = di.asDiagonal();
                Eigen::Matrix<double, 6, 6> T, Pp, Pn, Q;
                for (int a = 0; a < 6; ++a)
                    for (int b = 0; b < 6; ++b) {
                        T(a, b) = ekf_T_chain_[(size_t)i * 36 + a * 6 + b];
                        Pp(a, b) = ekf_p_prev_[(size_t)i * 36 + a * 6 + b];
                        Pn(a, b) = P_before[(size_t)i * 36 + a * 6 + b];
                    }
                Eigen::Matrix<double, 6, 6> Ts = Di * T * D;
                Eigen::Matrix<double, 6, 6> Pps = Di * Pp * Di;
                Eigen::Matrix<double, 6, 6> Pns = Di * Pn * Di;
                Eigen::Matrix<double, 6, 6> TPT = Ts * Pps * Ts.transpose();
                if (i == 0)
                    ekf_qeff_trace_margin_ = TPT.trace() - Pns.trace();
                Q = Pns - TPT;
                Q = 0.5 * (Q + Q.transpose());
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> es(Q);
                Eigen::Matrix<double, 6, 1> w = es.eigenvalues();
                qlmin = std::min(qlmin, w.minCoeff());
                // Qeff 物理下限：不能只用相对 wmax·1e-14（若 Qeff 整块塌缩，
                // 相对下限也跟着塌缩到 1e-21）。这里同时挂到区间端点协方差
                // （1e-14·tr/6）和 scaled 过程 Q（1e-12·tr/6），保持 PSD。
                Eigen::Matrix<double, 6, 6> Lq, Qphys, Qs;
                for (int a = 0; a < 6; ++a)
                    for (int b = 0; b < 6; ++b)
                        Lq(a, b) = qh[(size_t)i * 36 + a * 6 + b];
                Qphys = Lq.inverse() * Lq.inverse().transpose();
                Qs = Di * Qphys * Di;
                const double trq = Qs.trace();
                const double tr_end =
                    0.5 * std::max(Pns.trace(), 0.0) + 0.5 * std::max(TPT.trace(), 0.0);
                qscale = std::max(qscale, std::max(tr_end, 0.0));
                const double fl_end =
                    (std::isfinite(tr_end) && tr_end > 0.0) ? tr_end * 1e-14 / 6.0 : 0.0;
                const double fl_q =
                    (std::isfinite(trq) && trq > 0.0) ? trq * 1e-12 / 6.0 : 0.0;
                const double wfl = std::max(
                    std::max(std::max(w.maxCoeff(), 0.0) * 1e-14, fl_end),
                    std::max(fl_q, 1e-300));
                for (int k = 0; k < 6; ++k) w(k) = std::max(w(k), wfl);
                Q = es.eigenvectors() * w.asDiagonal() * es.eigenvectors().transpose();
                Q = 0.5 * (Q + Q.transpose());
                Q = D * Q * D;                              // 折回 QOE 空间
                for (int a = 0; a < 6; ++a)
                    for (int b = 0; b < 6; ++b)
                        ekf_qeff_[(size_t)i * 36 + a * 6 + b] = Q(a, b);
            }
            ekf_qeff_lmin_ = qlmin;
            // floor 只应吸收数值舍入级负值；显著负 λmin 通常意味着 T 链非收缩
            // （J6 谱半径>1）或 Qeff 的 6×6 近似失效，必须报出来而不是静默抬正。
            if (qlmin < -1e-10 * std::max(qscale, 1.0))
                fprintf(stderr,
                        "[ekf/qeff] 警告: 区间 Qeff λmin=%.3e < 0（端点尺度 %.3e）；"
                        "floor 仅吸收数值负值，请检查 J6/T 链收缩性\n",
                        qlmin, qscale);
            T_host_ = ekf_T_chain_;                         // 区间快照（Python 读取）
            ekf_T_chain_ = Phi_host;                        // 新区间起点：chain := Φ_b
            if (std::getenv("FLS_EKFDBG")) {
                const int i0 = 0;
                double Tt = 0.0, Pt = 0.0, Tmax = 0.0;
                for (int a = 0; a < 6; ++a) {
                    Tt += T_host_[(size_t)i0 * 36 + a * 6 + a];
                    Pt += P_before[(size_t)i0 * 36 + a * 6 + a];
                    for (int b = 0; b < 6; ++b)
                        Tmax = std::max(Tmax, std::fabs(T_host_[(size_t)i0 * 36 + a * 6 + b]));
                }
                fprintf(stderr, "[ekfstep] f=%d KEY skip=1 Ttr=%.3e Tmax=%.3e Pkf=%.3e "
                        "lmin=%.3e\n", _ekf_this, Tt, Tmax, Pt, qlmin);
            }
            // 预测 P⁻_{b+1} = Φ_b P⁻_b Φ_bᵀ + Q（下一普通帧的先验），Pm_host_ 前滚
            cudaMemcpy(d_PmMarg_, Pm_host_.data(), (size_t)n_ * 36 * sizeof(double),
                       cudaMemcpyHostToDevice);
            qoejopt::flsEkfPredDevice(d_PmMarg_, d_phi_, d_qingt_, d_ekf_scale_, n_,
                                      Lp_.data_ptr<double>(), d_PmMarg_);
            cudaMemcpy(Lp_host_.data(), Lp_.data_ptr<double>(), (size_t)n_ * 36 * sizeof(double),
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(Pm_host_.data(), d_PmMarg_, (size_t)n_ * 36 * sizeof(double),
                       cudaMemcpyDeviceToHost);
            ekf_p_prev_key_ = P_before;                     // P⁻_b（下一区间的 Pprev）
            std::copy(hX_.begin(), hX_.begin() + (size_t)n_ * 6, zp_host_.begin());
            cudaMemcpy(z_prior_.data_ptr<double>(), hX_.data(), (size_t)n_ * 6 * sizeof(double),
                       cudaMemcpyHostToDevice);
            has_prior_ = true; multi_prior_ = false;
            prior_gauge_projected_ = false;
            prior_proj_lowrank_ = false;
            prior_proj_A0_host_.clear();
            prior_proj_B0_host_.clear();
            prior_proj_S0_host_.clear();
            prior_proj_S0inv_host_.clear();
            res2_ = 0.0;
            return;
        }
        // ① GN（线搜索用 assemble 返回的 ISL res2）
        t1 = _now(); double res2 = assemble(); t_asm += _now() - t1;
        const bool ekf_local = (L_ == 1 && !skip_update_ && ekf_cov_r_ > 0);
        const double cost0 = ekf_local ? 0.0 : total_cost();
        (void)cost0;
        t1 = _now();
        int gn_it = 0;   // GN 实际迭代数（诊断）
        for (int it = 0; it < std::max(1, steps); ++it) {
            ++gn_it;
            double tm = _now();
            std::pair<torch::Tensor, torch::Tensor> mtm;
            if (ekf_local) {
                scaled_cols_only();
            } else {
                mtm = scaled_AtA_Atb();
            }
            dbg_mtm += _now() - tm;
            torch::Tensor dcpu;
            if (ekf_local) {
                // r=1/2：局部 H_B dx=A_B^T b，替代每帧 6N 全局 GN 因子。
                const double _tld = _now();
                auto dcv = ekf_local_update(ekf_cov_r_);
                dcpu = torch::from_blob(dcv.data(), {(int64_t)S_.ncol},
                                        torch::TensorOptions().dtype(torch::kFloat64))
                           .clone();
                dbg_fac += _now() - _tld;
            } else if (sweep_) {   // 信息形式逐帧消去（l430-433 选项 2）：每帧 6N 因子 + 逐星 Schur 修正 + 回代
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
                if (cross_active_)
                    nd.add_M(cross_vals_t_.view({1, (int64_t)cross_vals_.size()}),
                             cross_rp_t_, cross_ci_t_);
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
            // 步长上限：弱方向上的 GN 步可到 1e4，4 次线搜索走不回来。
            // KF_DC_MAX>0 时按比例缩 dc（方向不变），只作为 solver 保护。
            {
                const char *ce = std::getenv("KF_DC_MAX");
                const double cap = ce ? std::atof(ce) : 0.0;
                if (cap > 0.0 && std::isfinite(dcmax) && dcmax > cap) {
                    const double s = cap / dcmax;
                    dcpu.mul_(s);
                    dcn2 *= s * s;
                    fprintf(stderr, "[fls/step] cap |dc|max=%.3e -> %.3e (s=%.3e)\n",
                            dcmax, cap, s);
                    dcmax = cap;
                }
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
            if (L_ == 1 && !skip_update_) {
                // QOE 逐星 EKF：状态和协方差必须来自同一个线性化系统。
                // 一次完整线性步，禁止线搜索截断；否则 T=I-KH 与真实状态
                // 更新不同源，关键帧处的 Q_eff 会负定。
                // 不在更新后重装 A/b：handoff 的 KH/P⁺ 必须用产生本次状态
                // 更新的同一套 x⁻ 线性化行，重装到 x⁺ 会引入 O(dx) 的不同源误差。
                for (size_t i = 0; i < hX_.size(); ++i) hX_[i] = X0[i] + dc[(int64_t)i];
                retract_qoe(hX_, L_, n_);
                // 只把 x⁺ 上传到 d_X（handoff 的 Φ=STM(x⁺) 需要），A_val/b 保持 x⁻ 线性化
                cudaMemcpy(wc_.d_X, hX_.data(), (size_t)n_ * 6 * sizeof(double),
                           cudaMemcpyHostToDevice);
                acc = true;
            } else for (int ls = 0; ls < 4; ++ls) {
                for (size_t i = 0; i < hX_.size(); ++i) hX_[i] = X0[i] + alpha * dc[(int64_t)i];
                retract_qoe(hX_, L_, n_);
                t1 = _now();
                // L>1 必须用全代价（ISL+GTS+dyn+prior+gauge）线搜索；
                // 只比 ISL 会把“ISL 减小但先验变坏”的步放进来。
                const double rn = (L_ > 1) ? total_cost() : res2_only();
                t_asm += _now() - t1;
                if (getenv("FLS_LSDBG") && gn_it <= 2)
                    fprintf(stderr, "[fls/ls] it=%d ls=%d alpha=%.4g res2=%.8e rn=%.8e |dc|max=%.3e\n",
                            gn_it, ls, alpha, res2, rn, dcmax);
                if (std::isfinite(rn) && rn < res2) {
                    res2 = rn; acc = true;
                    t1 = _now(); assemble(); t_asm += _now() - t1;
                    break;
                }
                alpha *= 0.5;
            }
            // 硬门禁：接受步后全代价必须为正且有限。KF_DEV_MAX_M 放大后
            // dev 门禁失效，这里是拦住“负 gauge 能量/不定 Pg_inv”的唯一检查。
            if (acc && L_ > 1) {
                const double ct = total_cost();
                if (!std::isfinite(ct) || ct < 0.0) {
                    fprintf(stderr,
                            "[fls/cost] reject it=%d total_cost=%.6e "
                            "（<0/nonfinite）；恢复本步并停止该窗链\n",
                            gn_it, ct);
                    hX_ = X0;
                    geo_all();
                    assemble();
                    throw std::runtime_error(
                        "solve_batch: total_cost<0/nonfinite（gauge/Pg 账本不一致）");
                }
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
            phi_host_.resize((size_t)n_ * 36);
            const std::vector<double> &Phi_host = phi_host_;
            cudaMemcpy(phi_host_.data(), d_phi_, (size_t)n_ * 36 * sizeof(double),
                       cudaMemcpyDeviceToHost);
            double t2;
            if (ekf_cov_r_ > 0) {
                ekf_local_handoff(ekf_cov_r_, P_before, Phi_host);
                t2 = _now();
            } else {
                qoejopt::flsEkfHandoffDevice(A_val_.data_ptr<double>(),
                                             d00_p_.data_ptr<int64_t>(), d00_s_.data_ptr<int64_t>(),
                                             d00_c_.data_ptr<int64_t>(), d_phi_, d_qingt_,
                                             d_PmMarg_, d_ekf_scale_, n_,
                                             Lp_.data_ptr<double>(), d_PmMarg_, d_Pplus_, d_KH_);
                cudaMemcpy(Lp_host_.data(), Lp_.data_ptr<double>(), (size_t)n_ * 36 * sizeof(double),
                           cudaMemcpyDeviceToHost);
                cudaMemcpy(Pm_host_.data(), d_PmMarg_, (size_t)n_ * 36 * sizeof(double),
                           cudaMemcpyDeviceToHost);
                Pplus_host_.assign((size_t)n_ * 36, 0.0);
                cudaMemcpy(Pplus_host_.data(), d_Pplus_, (size_t)n_ * 36 * sizeof(double),
                           cudaMemcpyDeviceToHost);
                KH_host_.assign((size_t)n_ * 36, 0.0);
                cudaMemcpy(KH_host_.data(), d_KH_, (size_t)n_ * 36 * sizeof(double),
                           cudaMemcpyDeviceToHost);
                t2 = _now();
            }
            // L=1 交接后 Lp_ 已由 Ppred/flsEkfHandoffDevice 重写，旧投影数据失效。
            prior_gauge_projected_ = false;
            prior_proj_lowrank_ = false;
            prior_proj_A0_host_.clear();
            prior_proj_B0_host_.clear();
            prior_proj_S0_host_.clear();
            prior_proj_S0inv_host_.clear();
            dbg_marg = t2 - tm;
            {
#pragma omp parallel for schedule(static)
                for (int i = 0; i < n_; ++i) {
                    Eigen::Matrix<double, 6, 6> Pb, Pa, Ph, J, T;
                    for (int a = 0; a < 6; ++a)
                        for (int b = 0; b < 6; ++b) {
                            Pb(a, b) = P_before[(size_t)i * 36 + a * 6 + b];
                            Pa(a, b) = Pplus_host_[(size_t)i * 36 + a * 6 + b];
                            Ph(a, b) = Phi_host[(size_t)i * 36 + a * 6 + b];
                            T(a, b) = ekf_T_chain_[(size_t)i * 36 + a * 6 + b];
                        }
                    Eigen::Matrix<double, 6, 6> Jupd = Eigen::Matrix<double, 6, 6>::Identity();
                    if (!skip_update_) {
                        for (int a = 0; a < 6; ++a)
                            for (int b = 0; b < 6; ++b)
                                Jupd(a, b) = -KH_host_[(size_t)i * 36 + a * 6 + b];
                        for (int a = 0; a < 6; ++a) Jupd(a, a) += 1.0;
                    }
                    Jupd = Jupd.unaryExpr([](double v) {
                        return std::isfinite(v) ? std::min(std::max(v, -1e12), 1e12) : 0.0;
                    });
                    // 不做欧氏 SVD σ≤1 钳位：I−KH 在 QOE 混合量纲下非正规，
                    // σ_max≈|K|·|v|≫1 属正常；它在 P⁻ 度量下才是收缩（Joseph 保证 P⁺⪯P⁻）。
                    // 钳位会把 T 改写成另一个矩阵，破坏与状态更新的同源性，Qeff 随之失真。
                    // 逐帧因子 = Φ_f·J_f：J_f 把 δx⁻_f 映到 δx⁺_f（本历元更新），
                    // Φ_f=STM(x⁺_f) 再映到 δx⁻_{f+1}；与协方差链
                    // P⁻_{f+1}=Φ_f P⁺_f Φ_fᵀ+Q、P⁺_f=J_f P⁻_f J_fᵀ+KKᵀ 同源。
                    // 顺序写成 J_f·Φ_f 会使 T 与 P 链差一对交换子，Qeff 出负特征值。
                    Eigen::Matrix<double, 6, 6> Tn = Ph * Jupd * T;
                    Tn = Tn.unaryExpr([](double v) {
                        return std::isfinite(v) ? std::min(std::max(v, -1e15), 1e15) : 0.0;
                    });
                    for (int a = 0; a < 6; ++a)
                        for (int b = 0; b < 6; ++b) {
                            ekf_T_chain_[(size_t)i * 36 + a * 6 + b] = Tn(a, b);
                            T_host_[(size_t)i * 36 + a * 6 + b] = Tn(a, b);
                        }
                }
            }
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
            if (std::getenv("FLS_EKFDBG")) {
                const int i0 = 0;
                double Tt = 0.0, Pt = 0.0, Kt = 0.0, Ppt = 0.0, P0t = 0.0;
                for (int a = 0; a < 6; ++a) {
                    Tt += T_host_[(size_t)i0 * 36 + a * 6 + a];
                    Pt += Pm_host_[(size_t)i0 * 36 + a * 6 + a];
                    Kt += KH_host_[(size_t)i0 * 36 + a * 6 + a];
                    Ppt += Pplus_host_[(size_t)i0 * 36 + a * 6 + a];
                    P0t += P_before[(size_t)i0 * 36 + a * 6 + a];
                }
                const int i0b = i0 * 36;
                fprintf(stderr, "[ekfstep] f=%d nonkey Ttr=%.3e Ptr=%.3e P0tr=%.3e "
                        "Pplustr=%.3e KHtr=%.3e P0[a,u]=%.3e P+[a,u]=%.3e P0[u,u]=%.3e "
                        "P+[u,u]=%.3e P-[a,u]=%.3e P-[u,u]=%.3e\n",
                        _ekf_this, Tt, Pt, P0t, Ppt, Kt,
                        P_before[i0b + 1], Pplus_host_[i0b + 1],
                        P_before[i0b + 7], Pplus_host_[i0b + 7],
                        Pm_host_[i0b + 1], Pm_host_[i0b + 7]);
            }
            dbg_rest = _now() - t3;
            t_fac += _now() - t1;
            t1 = _now();
            has_prior_ = true;
        } else {
            double tm = _now();
            if (global_marg_) {
                // 全局 selected inverse：对全窗 H=A_sᵀA_s 做一次稀疏因子，
                // 只取保留节点 x1 的 6×6 对角块（含全网 ISL；块间相关仍丢弃）。
                auto A_cpu = A_val_.to(torch::kCPU).contiguous();
                auto sc_cpu = mtm_scl_.to(torch::kCPU).contiguous();
                auto ci_cpu = A_ci_t_.to(torch::kCPU);
                auto rp_cpu = A_rp_t_.to(torch::kCPU);
                auto As = A_cpu * sc_cpu.index_select(0, ci_cpu);
                auto nd = symc_->createNumericDecomposition(1);
                nd.add_MtM(As.reshape({1, S_.nnz}), rp_cpu, ci_cpu);
                add_cross_to_nd_cpu_(nd);
                nd.factor();
                auto perm = nd.param_to_span().to(torch::kCPU);
                auto Pd = nd.compute_diag_blocks(6).index_select(0, perm).cpu().contiguous();
                const double *pp = Pd.data_ptr<double>();
                const double *sp = sc_cpu.data_ptr<double>();
                for (int i = 0; i < n_; ++i) {
                    Eigen::Matrix<double, 6, 6> P;
                    const int64_t b = (int64_t)n_ + i;          // 节点 1
                    for (int a = 0; a < 6; ++a)
                        for (int c = 0; c < 6; ++c)
                            P(a, c) = pp[(b * 6 + a) * 6 + c] * sp[b * 6 + a] * sp[b * 6 + c];
                    P = 0.5 * (P + P.transpose());
                    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> es(P);
                    Eigen::Matrix<double, 6, 1> w = es.eigenvalues();
                    for (int a = 0; a < 6; ++a) w[a] = std::max(w[a], 1e-3);
                    const auto V = es.eigenvectors();
                    for (int a = 0; a < 6; ++a) {
                        const double si = 1.0 / std::sqrt(w[a]);
                        for (int c = 0; c < 6; ++c) Lp_host_[(size_t)i * 36 + a * 6 + c] = V(c, a) * si;
                    }
                    for (int a = 0; a < 6; ++a)
                        for (int c = 0; c < 6; ++c) Pm_host_[(size_t)i * 36 + a * 6 + c] = P(a, c);
                }
                cudaMemcpy(Lp_.data_ptr<double>(), Lp_host_.data(), (size_t)n_ * 36 * sizeof(double),
                           cudaMemcpyHostToDevice);
                std::copy(hX_.begin() + (size_t)n_ * 6, hX_.begin() + (size_t)2 * n_ * 6,
                          zp_host_.begin());
                cudaMemcpy(z_prior_.data_ptr<double>(), hX_.data() + (size_t)n_ * 6,
                           (size_t)n_ * 6 * sizeof(double), cudaMemcpyHostToDevice);
                has_prior_ = true;
                prior_gauge_projected_ = false;   // Lp_ 重写，旧 A0/B0 失效
                prior_proj_lowrank_ = false;
                prior_proj_A0_host_.clear();
                prior_proj_B0_host_.clear();
                prior_proj_S0_host_.clear();
                prior_proj_S0inv_host_.clear();
            } else {
                qoejopt::flsMargDevice(A_val_.data_ptr<double>(), mtm_scl_.data_ptr<double>(),
                                       d00_p_.data_ptr<int64_t>(), d00_s_.data_ptr<int64_t>(),
                                       d00_c_.data_ptr<int64_t>(),
                                       d11_p_.data_ptr<int64_t>(), d11_s_.data_ptr<int64_t>(),
                                       d11_c_.data_ptr<int64_t>(),
                                       pr_s1_.data_ptr<int64_t>(), pr_s0_.data_ptr<int64_t>(),
                                       n_, Lp_.data_ptr<double>(), d_PmMarg_);
                cudaMemcpy(Lp_host_.data(), Lp_.data_ptr<double>(), (size_t)n_ * 36 * sizeof(double),
                           cudaMemcpyDeviceToHost);
                cudaMemcpy(Pm_host_.data(), d_PmMarg_, (size_t)n_ * 36 * sizeof(double),
                           cudaMemcpyDeviceToHost);
                std::copy(hX_.begin() + (size_t)n_ * 6, hX_.begin() + (size_t)2 * n_ * 6,
                          zp_host_.begin());
                cudaMemcpy(z_prior_.data_ptr<double>(), hX_.data() + (size_t)n_ * 6,
                           (size_t)n_ * 6 * sizeof(double), cudaMemcpyHostToDevice);
                has_prior_ = true;
                prior_gauge_projected_ = false;   // Lp_ 重写，旧 A0/B0 失效
                prior_proj_lowrank_ = false;
                prior_proj_A0_host_.clear();
                prior_proj_B0_host_.clear();
                prior_proj_S0_host_.clear();
                prior_proj_S0inv_host_.clear();
            }
            double t2 = _now(); dbg_marg = t2 - tm;
            double t3 = t2; dbg_d2h = t3 - t2;
            dbg_rest = _now() - t3;
            t_fac += _now() - t1;
            t1 = _now();
        }
        t_ls += _now() - t1;
        t_schur = _now() - t0 - t_pred - t_geo - t_asm - t_gn;
        dbg_pred = t_pred; dbg_geo = t_geo; dbg_asm = t_asm; dbg_gn = t_gn;
        dbg_total = _now() - t0;
        if (getenv("FLS_EKF_PROF")) {
            static int _pn = 0;
            static double _pp = 0, _pg = 0, _pa = 0, _pgn = 0, _pl = 0, _ps = 0, _pt = 0;
            _pp += t_pred; _pg += t_geo; _pa += t_asm; _pgn += t_gn;
            _pl += t_ls; _ps += t_schur; _pt += dbg_total;
            if (++_pn % 50 == 0) {
                fprintf(stderr,
                        "[ekf/step] pred=%.2f geo=%.2f asm=%.2f gn=%.2f ls=%.2f "
                        "rest=%.2f total=%.2f ms/frame\n",
                        1e3 * _pp / 50, 1e3 * _pg / 50, 1e3 * _pa / 50,
                        1e3 * _pgn / 50, 1e3 * _pl / 50, 1e3 * _ps / 50,
                        1e3 * _pt / 50);
                _pp = _pg = _pa = _pgn = _pl = _ps = _pt = 0.0;
            }
        }
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
    py::array_t<double> T_host() const {
        py::array_t<double> out({(py::ssize_t)n_, (py::ssize_t)6, (py::ssize_t)6});
        std::copy(T_host_.begin(), T_host_.end(), out.mutable_data());
        return out;
    }
    py::array_t<double> phi_host() const {
        py::array_t<double> out({(py::ssize_t)n_, (py::ssize_t)6, (py::ssize_t)6});
        std::copy(phi_host_.begin(), phi_host_.end(), out.mutable_data());
        return out;
    }
    py::array_t<double> ekf_scale_host() const {
        // 固定每星缩放 d=√diag(P0)（set_P_init 冻结）：窗口消费 T/Qeff 时
        // 在 D 缩放空间做数值处理，避免 QOE 混合量纲的信息 spread 超 double。
        py::array_t<double> out({(py::ssize_t)n_, (py::ssize_t)6});
        std::copy(d_ekf_host_.begin(), d_ekf_host_.end(), out.mutable_data());
        return out;
    }
    py::array_t<double> P_prev_host() const {
        py::array_t<double> out({(py::ssize_t)n_, (py::ssize_t)6, (py::ssize_t)6});
        std::copy(ekf_p_prev_.begin(), ekf_p_prev_.end(), out.mutable_data());
        return out;
    }
    py::array_t<double> P_pred_host() const {
        py::array_t<double> out({(py::ssize_t)n_, (py::ssize_t)6, (py::ssize_t)6});
        std::copy(ekf_p_pred_.begin(), ekf_p_pred_.end(), out.mutable_data());
        return out;
    }
    py::array_t<double> Qeff_host() const {
        py::array_t<double> out({(py::ssize_t)n_, (py::ssize_t)6, (py::ssize_t)6});
        std::copy(ekf_qeff_.begin(), ekf_qeff_.end(), out.mutable_data());
        return out;
    }
    double Qeff_lmin() const { return ekf_qeff_lmin_; }
    double Qeff_trace_margin() const { return ekf_qeff_trace_margin_; }
    void reset_T() {
        for (int i = 0; i < n_; ++i)
            for (int a = 0; a < 6; ++a)
                for (int b = 0; b < 6; ++b) {
                    T_host_[(size_t)i * 36 + a * 6 + b] = (a == b) ? 1.0 : 0.0;
                    ekf_T_chain_[(size_t)i * 36 + a * 6 + b] = (a == b) ? 1.0 : 0.0;
                }
    }
    void set_skip(int on) { skip_update_ = (on != 0); }
    void set_ekf_cov_r(int r) {
        if (r < 0 || r > 2) throw std::runtime_error("set_ekf_cov_r: r 仅支持 0/1/2");
        ekf_cov_r_ = r;
    }
    int ekf_cov_r() const { return ekf_cov_r_; }
    py::array_t<double> P_marg_host() const {
        py::array_t<double> out({(py::ssize_t)n_, (py::ssize_t)6, (py::ssize_t)6});
        std::copy(Pm_host_.begin(), Pm_host_.end(), out.mutable_data());
        return out;
    }
    py::array_t<double> P_plus_host() const {
        py::array_t<double> out({(py::ssize_t)n_, (py::ssize_t)6, (py::ssize_t)6});
        std::copy(Pplus_host_.begin(), Pplus_host_.end(), out.mutable_data());
        return out;
    }
    py::array_t<double> KH_host() const {
        py::array_t<double> out({(py::ssize_t)n_, (py::ssize_t)6, (py::ssize_t)6});
        std::copy(KH_host_.begin(), KH_host_.end(), out.mutable_data());
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
    // 当前窗口的有效 gauge 先验 Pg_eff=(Pg^-1+Λ_g)^-1（Λ_g 仅含
    // gts_pg_node_range 内节点的 GTS G 分量；范围外 GTS 行仍是 raw）。
    py::array_t<double> gauge_pg_host() const {
        if (!has_gauge_ || gauge_k_ <= 0)
            throw std::runtime_error("gauge_pg_host: 未设置 gauge prior");
        const int k = gauge_k_;
        const std::vector<double> &src = Pg_eff_host_.empty() ? Pg_host_ : Pg_eff_host_;
        py::array_t<double> out({(py::ssize_t)k, (py::ssize_t)k});
        std::copy(src.begin(), src.begin() + (size_t)k * k, out.mutable_data());
        return out;
    }
    // GTS 的 G 分量贡献 Λ_g（未与 Pg^-1 合并；仅 gts_pg_node_range 内节点）。
    py::array_t<double> gauge_pginfo_host() const {
        if (!has_gauge_ || gauge_k_ <= 0)
            throw std::runtime_error("gauge_pginfo_host: 未设置 gauge prior");
        const int k = gauge_k_;
        py::array_t<double> out({(py::ssize_t)k, (py::ssize_t)k});
        auto *op = out.mutable_data();
        for (int a = 0; a < k * k; ++a)
            op[a] = (int)Pg_info_host_.size() == k * k ? Pg_info_host_[(size_t)a] : 0.0;
        return out;
    }
    // GTS gauge Fisher 的系数空间伪逆：P_f=pinv(Λ_g)，Λ_g=Pg_info_host_。
    // 只在 C++ 内做 6×6 特征分解；rank<k 时明确返回 rank，调用方不得把
    // 未观测方向的零协方差当成有效 sigma。
    py::dict gauge_gts_fisher_q_host() const {
        if (!has_gauge_ || gauge_k_ <= 0)
            throw std::runtime_error("gauge_gts_fisher_q_host: 未设置 gauge prior");
        const int k = gauge_k_;
        Eigen::MatrixXd L = Eigen::MatrixXd::Zero(k, k);
        if ((int)Pg_info_host_.size() == k * k)
            for (int a = 0; a < k; ++a)
                for (int b = 0; b < k; ++b)
                    L(a, b) = Pg_info_host_[(size_t)a * k + b];
        L = 0.5 * (L + L.transpose());
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(L);
        const Eigen::VectorXd w = es.eigenvalues();
        const Eigen::MatrixXd V = es.eigenvectors();
        const double wmax = w.size() ? w.maxCoeff() : 0.0;
        const double cut = std::max(std::abs(wmax) * 1e-12, 1e-300);
        int rank = 0;
        double wmin = 0.0;
        Eigen::MatrixXd P = Eigen::MatrixXd::Zero(k, k);
        for (int i = 0; i < k; ++i) {
            if (w(i) > cut) {
                ++rank;
                P.noalias() += (V.col(i) * V.col(i).transpose()) / w(i);
                if (rank == 1) wmin = w(i);
                else wmin = std::min(wmin, w(i));
            }
        }
        P = 0.5 * (P + P.transpose());
        py::dict d;
        d["P"] = eigen_to_np_rowmajor(P);
        d["rank"] = rank;
        d["k"] = k;
        d["lambda_min"] = w.size() ? w.minCoeff() : 0.0;
        d["lambda_max"] = wmax;
        d["cond"] = (rank == k && wmin > 0.0)
                        ? wmax / wmin : std::numeric_limits<double>::infinity();
        d["valid"] = (rank == k);
        return d;
    }
    // 末次 GN 的 Xg=K^-1 G、Hg=K G（物理 QOE，ncol×k），供 Python 协方差修正。
    py::array_t<double> gauge_xg_host() const {
        const int k = std::max(gauge_k_, 1);
        py::array_t<double> out({(py::ssize_t)S_.ncol, (py::ssize_t)k});
        auto *op = out.mutable_data();
        const size_t want = (size_t)S_.ncol * k;
        for (size_t a = 0; a < want; ++a)
            op[a] = gauge_xg_host_.size() == want ? gauge_xg_host_[a] : 0.0;
        return out;
    }
    py::array_t<double> gauge_hg_host() const {
        const int k = std::max(gauge_k_, 1);
        py::array_t<double> out({(py::ssize_t)S_.ncol, (py::ssize_t)k});
        auto *op = out.mutable_data();
        const size_t want = (size_t)S_.ncol * k;
        for (size_t a = 0; a < want; ++a)
            op[a] = gauge_hg_host_.size() == want ? gauge_hg_host_[a] : 0.0;
        return out;
    }
    // D1：当前窗口 Λ_win=(I-Pi)K(I-Pi)+G Pg_eff^-1 G^T 的 gauge 块。
    //   S=G^T G+eps_g I, (I-Pi)G=eps_g G S^-1,
    //   G^T Λ_win G = T^T (G^T K G) T + (G^T G) Pg_eff^-1 (G^T G), T=eps_g S^-1.
    // 只做 k 列稀疏 matmul（KG=K G），不构造 6N×6N K，也不做 6N 逆。
    // 返回的 min/max eig 是 k 维 gauge 块的，不是 λ_min(Λ_win) 的 6N 全谱；
    // 全谱正性仍需另做逆迭代/Lanczos（本函数显式不声称）。
    // 时机：solve_batch 之后、marginalize_first_local_gauge 之前调用才是本窗
    // solve 的 Λ_win；marginalize 末尾会把 multi_prior 切到下一窗 node0 先验，
    // 之后调用得到的是“下一窗 prior + 当前 A”的组合，不是刚被消去的 Hwin。
    py::dict gauge_lamwin_host() {
        if (!has_gauge_ || gauge_k_ <= 0)
            throw std::runtime_error("gauge_lamwin_host: 未设置 gauge prior");
        if (!has_x_ || (int)hX_.size() != L_ * n_ * 6)
            throw std::runtime_error("gauge_lamwin_host: 需先 set_X/assemble");
        // 用当前 X 重装 A（与 solve_batch 末尾/marginalize 起点的 A 同源），再
        // 只重算列均衡与 KG；不触发任何 solve/因子分解。
        assemble();
        scaled_AtA_Atb();
        const int k = gauge_k_;
        const int64_t nc = S_.ncol;
        if ((int)Qg_host_.size() != (int)(nc * k))
            throw std::runtime_error("gauge_lamwin_host: G 尺寸不一致");
        if ((int)Pg_eff_inv_host_.size() != k * k)
            throw std::runtime_error("gauge_lamwin_host: Pg_eff^-1 未就绪");
        Eigen::MatrixXd Gmat(nc, k), PgInv(k, k), KG;
        for (int64_t a = 0; a < nc; ++a)
            for (int j = 0; j < k; ++j)
                Gmat(a, j) = Qg_host_[(size_t)a * k + j];
        for (int a = 0; a < k; ++a)
            for (int b = 0; b < k; ++b)
                PgInv(a, b) = Pg_eff_inv_host_[(size_t)a * k + b];
        PgInv = 0.5 * (PgInv + PgInv.transpose());
        if (!compute_kg_phys_(mtm_scl_, torch::Device(torch::kCPU), KG))
            throw std::runtime_error("gauge_lamwin_host: KG 计算失败/非有限");
        Eigen::MatrixXd Sg = Gmat.transpose() * Gmat;
        Sg = 0.5 * (Sg + Sg.transpose());
        const double epsg = gauge_eps_from_gram(Sg);
        const fls::GaugeLamWinBlock blk =
            fls::gauge_lamwin_block(KG, Gmat, PgInv, epsg);
        if (!blk.ok)
            throw std::runtime_error("gauge_lamwin_host: " + blk.reason);

        py::dict d;
        d["k"] = k;
        d["ncol"] = (int)nc;
        d["eps_g"] = epsg;
        d["GtKG"] = eigen_to_np_rowmajor(blk.GtKG);
        d["Sg"] = eigen_to_np_rowmajor(blk.Sg);
        d["GtLamG"] = eigen_to_np_rowmajor(blk.GtLamG);
        d["lamwin_gg_min_eig"] = blk.min_eig;
        d["lamwin_gg_max_eig"] = blk.max_eig;
        d["lamwin_gg_cond"] = blk.cond;
        d["gauge_block_positive"] = (blk.min_eig > 0.0);
        d["Q"] = eigen_to_np_rowmajor(blk.Q);
        d["R"] = eigen_to_np_rowmajor(blk.R);
        d["QtLamQ"] = eigen_to_np_rowmajor(blk.QtLamQ);
        d["Pg_eff_inv"] = eigen_to_np_rowmajor(PgInv);
        d["Pg_eff_inv_Q"] = eigen_to_np_rowmajor(blk.Pg_eff_inv_Q);
        d["identity_abs"] = blk.identity_abs;
        d["identity_rel"] = blk.identity_rel;
        d["QtLamQ_min_eig"] = blk.min_eig_Q;
        d["QtLamQ_max_eig"] = blk.max_eig_Q;
        d["QtLamQ_cond"] = blk.cond_Q;
        d["qr_rank_ok"] = blk.qr_rank_ok;
        d["R_diag_min"] = blk.R_diag_min;
        d["R_diag_max"] = blk.R_diag_max;
        d["source"] = "current_X_assemble";
        d["full_min_eig_computed"] = false;
        d["prior_proj_lowrank"] = prior_proj_lowrank_;
        d["prior_multi"] = multi_prior_;
        d["prior_node"] = prior_node_;
        d["note"] =
            "min/max eig 只针对 G 的 k 维 gauge 块；Lambda_win 的 6N 全谱最小"
            "特征值未计算（需单独逆迭代/Lanczos）。identity_rel=||Q^TΛQ-R Pg_eff^-1 R^T||/||...||。";
        d["timing_note"] =
            "solve_batch 之后、marginalize_first_local_gauge 之前调用才是本窗 "
            "solve 的 Λ_win；marginalize 末尾会切到下一窗 node0 先验。";
        if (prior_proj_lowrank_)
            d["note_prior_proj"] =
                "本块按 D1 定义的 base Λ_win 计算；实际 solve 的 Hwin 还含步骤4 "
                "-P ΔΛ P，未并入此 gauge 块。";
        // 与末次 Woodbury build 的 KG 对拍（若缓存有效）。用于确认当前 A 与
        // 最近一次 solve/marginalize 的 D 是否同源；不是另一个 eig 口径。
        if (gauge_woodbury_host_valid_ &&
            (int)gauge_hg_host_.size() == (int)(nc * k)) {
            Eigen::MatrixXd KGlast(nc, k);
            for (int64_t a = 0; a < nc; ++a)
                for (int j = 0; j < k; ++j)
                    KGlast(a, j) = gauge_hg_host_[(size_t)a * k + j];
            const double den = std::max(KGlast.norm(), 1e-300);
            d["last_build_available"] = true;
            d["last_build_ku"] = gauge_ku_host_;
            d["last_build_Dinv_fro"] = gauge_dinv_fro_host_;
            d["last_build_Dinv_hash"] = gauge_dinv_hash_host_;
            d["GtKG_vs_last_build_rel"] =
                (Gmat.transpose() * (KG - KGlast)).norm() / den;
        } else {
            d["last_build_available"] = false;
            d["GtKG_vs_last_build_rel"] =
                std::numeric_limits<double>::quiet_NaN();
        }
        return d;
    }
    // D3：窗口首次入窗状态 x_first（set_prior_multi 时刻）。(L,n,6)
    py::array_t<double> prior_first_x_host() const {
        if (!fej_first_x_valid_ || (int)fej_first_x_.size() != L_ * n_ * 6)
            throw std::runtime_error("prior_first_x_host: 尚无 set_prior_multi 的 x_first");
        py::array_t<double> out({(py::ssize_t)L_, (py::ssize_t)n_, (py::ssize_t)6});
        std::copy(fej_first_x_.begin(), fej_first_x_.end(), out.mutable_data());
        return out;
    }
    // D3：FEJ 冻结状态（先验中心/Lp、G_win、首次求解起点、当前漂移）。
    // prior_hessian_constant 恒为 true：winPriorAsmKernel 的 Jacobian 是固定 Lp，
    // 先验是精确二次型，H 不随 X 变化；变化的只有 b=Lp(z-x)。因此“在最终 X
    // 重装先验行”不会改变边缘化用的 H，只改变 RHS。该字段是这一事实的显式声明。
    py::dict fej_status() const {
        py::dict d;
        d["enabled"] = fej_first_x_valid_;
        d["prior_multi"] = multi_prior_;
        d["prior_node"] = prior_node_;
        d["has_prior"] = has_prior_;
        d["gauge_set"] = has_gauge_;
        d["gauge_k"] = gauge_k_;
        d["gauge_frozen"] = fej_g_frozen_;
        d["x_at_solve_valid"] = fej_x_at_solve_valid_;
        d["last_marg_check_ok"] = fej_last_marg_check_ok_;
        d["last_marg_check_reason"] = fej_last_marg_check_reason_;
        d["prior_hessian_constant"] = true;
        d["note"] =
            "C++ 验证不了 G 的构造来源（Python 必须在 x_first 构造）；这里只验证 "
            "G 从 set_gauge_prior/首次 solve 到 marginalize 未变，以及节点0 的 "
            "(Lp,z) 与 x_first 一致。x 本身的漂移允许非零。";
        if (!fej_first_x_valid_ || (int)fej_first_x_.size() != L_ * n_ * 6) {
            d["x_drift_max"] = std::numeric_limits<double>::quiet_NaN();
            d["prior_center_dev"] = std::numeric_limits<double>::quiet_NaN();
            d["prior_center_is_first_x"] = false;
        } else {
            double xdrift = 0.0;
            if ((int)hX_.size() == L_ * n_ * 6)
                for (int a = 0; a < L_ * n_ * 6; ++a)
                    xdrift = std::max(
                        xdrift, std::abs(hX_[(size_t)a] - fej_first_x_[(size_t)a]));
            d["x_drift_max"] = xdrift;
            double pcd = fej_prior_center_dev_;
            if (multi_prior_ && has_prior_) {
                std::vector<double> z0((size_t)n_ * 6);
                cudaMemcpy(z0.data(), z_prior_all_.data_ptr<double>(),
                           (size_t)n_ * 6 * sizeof(double), cudaMemcpyDeviceToHost);
                pcd = 0.0;
                for (int a = 0; a < n_ * 6; ++a)
                    pcd = std::max(
                        pcd, std::abs(z0[(size_t)a] - fej_first_x_[(size_t)a]));
            }
            double x0s = 1.0;
            for (int a = 0; a < n_ * 6; ++a)
                x0s = std::max(x0s, std::abs(fej_first_x_[(size_t)a]));
            d["prior_center_dev"] = pcd;
            d["prior_center_is_first_x"] = (pcd <= 1e-8 * x0s);
        }
        d["prior_Lp_dev"] = fej_prior_Lp_dev_;
        d["solve_start_dev"] = fej_solve_start_dev_;
        d["G_frozen_since_prior"] =
            has_gauge_ ? (fej_g_frozen_ && gauge_G_hash_() == fej_G_hash_frozen_)
                       : true;
        d["G_unchanged_since_solve_start"] =
            has_gauge_ ? (!fej_x_at_solve_valid_ || gauge_G_hash_() == fej_G_hash_at_solve_)
                       : true;
        d["G_hash"] = has_gauge_ ? gauge_G_hash_() : 0;
        d["G_hash_frozen"] = fej_G_hash_frozen_;
        d["G_hash_at_solve"] = fej_G_hash_at_solve_;
        d["x_hash"] = gauge_x_hash();
        d["x_hash_first"] =
            (fej_first_x_valid_ && (int)fej_first_x_.size() == L_ * n_ * 6)
                ? fnv1a64(fej_first_x_.data(), fej_first_x_.size() * sizeof(double))
                : 0;
        d["x_hash_at_solve"] =
            (fej_x_at_solve_valid_ && (int)fej_x_at_solve_.size() == L_ * n_ * 6)
                ? fnv1a64(fej_x_at_solve_.data(), fej_x_at_solve_.size() * sizeof(double))
                : 0;
        return d;
    }
    // P=Hwin^-1 的 gauge 子空间/补空间一致性诊断（末次 build_gauge_woodbury_）。
    // G 是物理 QOE、不做 QR，故一般 G^T G != I。正确分解是
    //   S = G^T G + eps_g I,
    //   P = G S^-1 Pg_eff S^-1 G^T + P_perp,
    //   G^T P_perp = 0,
    //   G^T P G = Pg_eff,
    // 其中 Hwin G = G (Pg^-1+Λ_g) S。doc 的 P = G Pg G^T + P_perp 以及
    // G^T Qeff G 的写法默认 G^T G=I；本实现必须保留 S^-1 夹心。
    // 用末次 Woodbury：P G = K^-1 G - Utot D^-1 Utot^T G，
    // 因此 G^T P G = G^T X - (G^T Utot) D^-1 (Utot^T G)。
    // G^T K G 仍由 gauge_hg_host()=K G 给出（Step-6 的 Qeff 诊断口径）。
    // 只在刚 build_gauge_woodbury_ 后有效（solve_batch/marginalize），否则抛错。
    py::dict gauge_cov_diagnostics() const {
        if (!has_gauge_ || gauge_k_ <= 0)
            throw std::runtime_error("gauge_cov_diagnostics: 未设置 gauge prior");
        if (!gauge_woodbury_host_valid_)
            throw std::runtime_error(
                "gauge_cov_diagnostics: 无有效 Woodbury（需在 solve_batch 或 "
                "marginalize_first_local_gauge 之后立即调用）");
        const int k = gauge_k_;
        const int ku = gauge_ku_host_;
        const int64_t nc = S_.ncol;
        if ((int)gauge_utot_host_.size() != (int)(nc * ku) ||
            (int)gauge_dinv_host_.size() != ku * ku ||
            (int)gauge_xg_host_.size() != (int)(nc * k) ||
            (int)gauge_hg_host_.size() != (int)(nc * k))
            throw std::runtime_error("gauge_cov_diagnostics: Woodbury 缓存尺寸不一致");
        Eigen::MatrixXd Gmat(nc, k);
        for (int64_t a = 0; a < nc; ++a)
            for (int j = 0; j < k; ++j) Gmat(a, j) = Qg_host_[(size_t)a * k + j];
        Eigen::Map<const Eigen::MatrixXd> Utot(gauge_utot_host_.data(), nc, ku);
        Eigen::Map<const Eigen::MatrixXd> Dinv(gauge_dinv_host_.data(), ku, ku);
        Eigen::MatrixXd X(nc, k), KG(nc, k);
        for (int64_t a = 0; a < nc; ++a)
            for (int j = 0; j < k; ++j) {
                X(a, j) = gauge_xg_host_[(size_t)a * k + j];
                KG(a, j) = gauge_hg_host_[(size_t)a * k + j];
            }
        const Eigen::MatrixXd M = Utot.transpose() * Gmat;
        const Eigen::MatrixXd PG = X - Utot * Dinv * M;
        const Eigen::MatrixXd GtPG_raw = Gmat.transpose() * PG;
        const Eigen::MatrixXd GtPG = 0.5 * (GtPG_raw + GtPG_raw.transpose());
        // 用末次 build 当时的 Pg_eff（不是当前值）：后续 assemble 可能已重算
        // Λ_g，而 Utot/Dinv 仍对应旧 build；诊断自洽性必须固定在 build 时刻。
        if ((int)gauge_pgeff_build_host_.size() != k * k)
            throw std::runtime_error("gauge_cov_diagnostics: Pg_eff 缓存缺失");
        Eigen::MatrixXd Pge(k, k);
        for (int a = 0; a < k; ++a)
            for (int b = 0; b < k; ++b)
                Pge(a, b) = gauge_pgeff_build_host_[(size_t)a * k + b];
        Pge = 0.5 * (Pge + Pge.transpose());
        Eigen::MatrixXd Sg = Gmat.transpose() * Gmat;
        Sg = 0.5 * (Sg + Sg.transpose());
        const double epsg = gauge_eps_from_gram(Sg);
        const Eigen::MatrixXd S = Sg + epsg * Eigen::MatrixXd::Identity(k, k);
        const Eigen::MatrixXd GtP = PG.transpose();
        Eigen::LLT<Eigen::MatrixXd> lltS(S);
        if (lltS.info() != Eigen::Success)
            throw std::runtime_error("gauge_cov_diagnostics: S=G^T G+eps I 非正定");
        const Eigen::MatrixXd SinvGt = lltS.solve(Gmat.transpose());
        const Eigen::MatrixXd GtPperp = GtP - Pge * SinvGt;
        const Eigen::MatrixXd GtKG_raw = Gmat.transpose() * KG;
        const Eigen::MatrixXd GtKG = 0.5 * (GtKG_raw + GtKG_raw.transpose());
        const double pgscale = std::max(Pge.norm(), 1e-300);
        const double gtscale = std::max(GtP.norm(), 1e-300);
        py::dict d;
        d["GtPG"] = eigen_to_np_rowmajor(GtPG);
        d["Pg_eff"] = eigen_to_np_rowmajor(Pge);
        d["GtPG_rel_err"] = (GtPG - Pge).norm() / pgscale;
        d["GtPperp"] = eigen_to_np_rowmajor(GtPperp);
        d["GtPperp_rel"] = GtPperp.norm() / gtscale;
        d["GtKG"] = eigen_to_np_rowmajor(GtKG);
        d["eps_g"] = epsg;
        d["k"] = k;
        d["ku"] = ku;
        return d;
    }
    // 末次 Woodbury 的条件诊断（不改 solve）：
    //   D_cond/D_cond_bal 是 build 时对 D 本身做 SVD 的原始/均衡条件数；
    //   D_rank/D_svd_trunc 是均衡 SVD 在 sv>max(sv)*1e-12 下的保留/截断数；
    //   D_blockdiag 表示最终 Dinv 是否按正交 Q 的 6×6 块对角装配；
    //   S_cond 用当前 G 重算 S=G^T G+eps_g I；
    //   Pg_eff_cond 用末次 build 当时缓存的 Pg_eff（cond(Pg_eff)=cond(Pg_eff^-1)）；
    //   B0_cond/B0_min_eig/B0_max_eig 来自最近一次 project_prior_gauge。
    // 与 gauge_cov_diagnostics 一样，只在刚 build_gauge_woodbury_ 后有效。
    py::dict gauge_woodbury_diag() const {
        if (!has_gauge_ || gauge_k_ <= 0)
            throw std::runtime_error("gauge_woodbury_diag: 未设置 gauge prior");
        if (!gauge_woodbury_host_valid_)
            throw std::runtime_error(
                "gauge_woodbury_diag: 无有效 Woodbury（需在 solve_batch 或 "
                "marginalize_first_local_gauge 之后立即调用）");
        const int k = gauge_k_;
        const int ku = gauge_ku_host_;
        if ((int)gauge_dinv_host_.size() != ku * ku)
            throw std::runtime_error("gauge_woodbury_diag: Woodbury 缓存尺寸不一致");
        const double inf = std::numeric_limits<double>::infinity();
        auto sym_cond = [&](const Eigen::MatrixXd &A) -> double {
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(A);
            if (es.info() != Eigen::Success) return inf;
            const double wmax = es.eigenvalues().maxCoeff();
            const double wmin = es.eigenvalues().minCoeff();
            return (wmax > 0.0 && wmin > 0.0) ? wmax / wmin : inf;
        };
        Eigen::Map<const Eigen::MatrixXd> Dinv(gauge_dinv_host_.data(), ku, ku);
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(Dinv);
        const auto &sv = svd.singularValues();
        const double smax = sv.size() > 0 ? sv(0) : 0.0;
        const double smin = sv.size() > 0 ? sv(sv.size() - 1) : 0.0;
        // 伪逆路径下 cond(Dinv) 不等于 cond(D)（截断方向被置零）；优先用
        // build 时对 D 本身做均衡 SVD 记录的值。
        const double dcond_fallback =
            (smax > 0.0 && smin > 0.0) ? smax / smin : inf;
        const double dcond =
            gauge_d_diag_valid_host_ ? gauge_d_cond_host_ : dcond_fallback;
        const int64_t nc = S_.ncol;
        if ((int)Qg_host_.size() != (int)(nc * k))
            throw std::runtime_error("gauge_woodbury_diag: G 缓存尺寸不一致");
        Eigen::MatrixXd Gmat(nc, k);
        for (int64_t a = 0; a < nc; ++a)
            for (int j = 0; j < k; ++j) Gmat(a, j) = Qg_host_[(size_t)a * k + j];
        Eigen::MatrixXd Sg = Gmat.transpose() * Gmat;
        Sg = 0.5 * (Sg + Sg.transpose());
        const double epsg = gauge_eps_from_gram(Sg);
        const Eigen::MatrixXd S = Sg + epsg * Eigen::MatrixXd::Identity(k, k);
        Eigen::MatrixXd PgE(k, k);
        // 用末次 build 当时的 Pg_eff（不是当前值）：后续 assemble 可能已重算
        // Λ_g，而 D_cond/pg_eff_cond 必须与缓存的 D^-1 同一时刻自洽。
        if ((int)gauge_pgeff_build_host_.size() != k * k)
            throw std::runtime_error("gauge_woodbury_diag: Pg_eff 缓存缺失");
        for (int a = 0; a < k; ++a)
            for (int b = 0; b < k; ++b)
                PgE(a, b) = gauge_pgeff_build_host_[(size_t)a * k + b];
        PgE = 0.5 * (PgE + PgE.transpose());
        py::dict d;
        d["D_cond"] = dcond;
        d["D_cond_bal"] = gauge_d_diag_valid_host_
                              ? gauge_d_cond_bal_host_
                              : inf;
        d["D_rank"] = gauge_d_diag_valid_host_ ? gauge_d_rank_host_ : -1;
        d["D11_cond"] = gauge_d_diag_valid_host_
                            ? gauge_d11_cond_host_
                            : inf;
        d["D11_cond_bal"] = gauge_d_diag_valid_host_
                                ? gauge_d11_cond_bal_host_
                                : inf;
        d["D22_cond"] = gauge_d_diag_valid_host_
                            ? gauge_d22_cond_host_
                            : inf;
        d["D22_cond_bal"] = gauge_d_diag_valid_host_
                                ? gauge_d22_cond_bal_host_
                                : inf;
        d["D11_rank"] = gauge_d_diag_valid_host_ ? gauge_d11_rank_host_ : -1;
        d["D22_rank"] = gauge_d_diag_valid_host_ ? gauge_d22_rank_host_ : -1;
        d["D_svd_trunc"] = gauge_d_diag_valid_host_
                               ? gauge_d_svd_trunc_host_
                               : -1;
        d["D_blockdiag"] = gauge_d_diag_valid_host_ &&
                           gauge_d_blockdiag_host_;
        d["D_orthonormal"] = gauge_d_diag_valid_host_ &&
                             gauge_d_orthonormal_host_;
        d["D_prior_proj"] = gauge_d_diag_valid_host_ &&
                            (gauge_ku_host_ == 4 * k);
        d["D_orth_dev"] = gauge_d_diag_valid_host_
                              ? gauge_d_orth_dev_host_
                              : std::numeric_limits<double>::quiet_NaN();
        d["Dinv_fro"] = gauge_dinv_fro_host_;
        d["Dinv_hash"] = gauge_dinv_hash_host_;
        d["S_cond"] = sym_cond(S);
        d["Pg_eff_cond"] = sym_cond(PgE);
        d["B0_cond"] = prior_proj_B0_cond_;
        d["B0_min_eig"] = prior_proj_B0_min_eig_;
        d["B0_max_eig"] = prior_proj_B0_max_eig_;
        d["B0_applied"] = prior_proj_lowrank_;
        d["eps_g"] = epsg;
        d["k"] = k;
        d["ku"] = ku;
        return d;
    }
    // 末次 D 的原始条件数 smax/smin；D_cond_bal 另给均衡后的条件数。
    double gauge_d_cond_host() const {
        if (!has_gauge_ || gauge_k_ <= 0 || !gauge_d_diag_valid_host_)
            return std::numeric_limits<double>::quiet_NaN();
        return gauge_d_cond_host_;
    }
    // FEJ 诊断同源的内容指纹：G 为当前 Qg_host_，x 为当前 hX_（两者都是
    // 物理 QOE 参数顺序）。Python 可用二者确认求解前后 G/x 是否真的未变。
    uint64_t gauge_G_hash() const { return gauge_G_hash_(); }
    uint64_t gauge_x_hash() const {
        if (hX_.empty()) return 0;
        return fnv1a64(hX_.data(), hX_.size() * sizeof(double));
    }
    py::array_t<double> Jnorm_host() const {
        py::array_t<double> out({(py::ssize_t)n_});
        std::copy(ekf_jnorm_host_.begin(), ekf_jnorm_host_.end(), out.mutable_data());
        return out;
    }
    py::array_t<double> JminSing_host() const {
        py::array_t<double> out({(py::ssize_t)n_});
        std::copy(ekf_jmin_host_.begin(), ekf_jmin_host_.end(), out.mutable_data());
        return out;
    }
    // 诊断：开/关每星球的完整 block-T 保存（T_B=Φ_B J_B，球内 m×m）。
    // 默认关闭；生产逐星链仍用 6×6 J6_stable/T6_stable，完整 block-T 只用于
    // 小窗验证交叉星耦合（跨星块不能直接塞进 per-star dyn 因子）。
    void set_ekf_store_block_T(bool on) {
        ekf_store_block_T_ = on;
        if (!on) { ekf_ball_host_.clear(); ekf_TB_host_.clear(); }
    }
    bool ekf_store_block_T() const { return ekf_store_block_T_; }
    py::tuple ekf_block_T_host(int star) const {
        if (star < 0 || star >= n_) throw std::runtime_error("ekf_block_T_host: star 越界");
        if (!ekf_store_block_T_ || (int)ekf_TB_host_.size() != n_ ||
            (int)ekf_ball_host_.size() != n_)
            throw std::runtime_error(
                "ekf_block_T_host: 需 set_ekf_store_block_T(true) 并在 ekf_local_handoff 后读取");
        const auto &idx = ekf_ball_host_[star];
        const int m = 6 * (int)idx.size();
        py::array_t<int32_t> ia((py::ssize_t)idx.size());
        auto *ip = ia.mutable_data();
        for (size_t a = 0; a < idx.size(); ++a) ip[a] = (int32_t)idx[a];
        py::array_t<double> ta({(py::ssize_t)m, (py::ssize_t)m});
        auto *tp = ta.mutable_data();
        const auto &tb = ekf_TB_host_[(size_t)star];
        if ((int)tb.size() != m * m)
            throw std::runtime_error("ekf_block_T_host: T_B 尺寸不一致");
        for (int a = 0; a < m; ++a)
            for (int b = 0; b < m; ++b) tp[(size_t)a * m + b] = tb[(size_t)a * m + b];
        return py::make_tuple(ia, ta);
    }
    bool prior_gauge_projected() const { return prior_gauge_projected_; }
    // 步骤4 B0=G0^T Λ0 G0 的 guard 诊断。cond<0 表示本次会话尚未调用
    // project_prior_gauge（或未推进到 B0 计算）。reason 为空表示 guard 通过。
    double prior_b0_cond() const { return prior_proj_B0_cond_; }
    double prior_b0_min_eig() const { return prior_proj_B0_min_eig_; }
    double prior_b0_max_eig() const { return prior_proj_B0_max_eig_; }
    py::dict prior_gauge_diagnostics() const {
        py::dict d;
        d["applied"] = prior_proj_lowrank_;
        d["projected"] = prior_gauge_projected_;
        d["reason"] = prior_proj_guard_reason_;
        d["B0_cond"] = prior_proj_B0_cond_;
        d["B0_min_eig"] = prior_proj_B0_min_eig_;
        d["B0_max_eig"] = prior_proj_B0_max_eig_;
        return d;
    }
    void set_gauge_gts_split(bool on) {
        gauge_gts_split_ = on;
        gauge_woodbury_host_valid_ = false;
        if (has_gauge_ && gauge_k_ > 0) apply_gts_gauge_split_();
        else update_pg_eff_();
    }
    bool gauge_gts_split() const { return gauge_gts_split_; }
    // 跨窗 GTS→Pg 节点范围（半开区间 [first,last)，节点索引 0..L-1），默认 [0,L)。
    //   Pg 的 G 分量只在范围内节点的 GTS 行上累加；范围外 GTS 行仍保留在
    //   A_val/b 中参与状态求解（raw/whole）。首窗用 [0,L)；滑窗从第二窗起用
    //   [L-1,L)（只让新进入的末节点贡献），因为此前节点的 GTS G 信息已经
    //   包含在上一窗 marginalize_first_local_gauge 返回的 Pg_next 中。该 last
    //   侧收窄是实验性用法：它会丢掉范围外旧节点的 GTS G 信息，只有确实持有
    //   上一窗 Pg_next 时才成立；默认仍为 [0,L)，不做此丢弃。
    //   调用后按当前 A_val 重算 Pg_info/Pg_eff；若尚未 assemble，A_val 为
    //   零则 Pg_info 暂为 0，下一次 assemble 会按真实 GTS 行重算。
    void set_gts_pg_node_range(int first, int last) {
        int f = std::max(0, std::min(first, L_));
        int l = std::max(0, std::min(last, L_));
        if (l < f) { const int t = f; f = l; l = t; }
        gts_pg_node_first_ = f;
        gts_pg_node_last_ = l;
        gauge_woodbury_host_valid_ = false;   // Pg_eff 将随范围改变
        if (has_gauge_ && gauge_k_ > 0) apply_gts_gauge_split_();
        if (std::getenv("FLS_GAUGEDBG"))
            fprintf(stderr, "[gts/pg] node range=[%d,%d) L=%d\n", f, l, L_);
    }
    py::tuple gts_pg_node_range() const { return py::make_tuple(gts_pg_node_first_, gts_pg_node_last_); }
    double res2() const { return res2_; }
    // 逐帧消去（sweep）开关：True 时 GN 因子/求解走信息形式逐帧消去（Baspacho 只解 6N 单帧）。
    void set_sweep(bool on) { sweep_ = on; }
    void set_marg_global(bool on) { global_marg_ = on; }
    bool marg_global() const { return global_marg_; }
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
    uint64_t gauge_G_hash_() const {
        if (Qg_host_.empty()) return 0;
        return fnv1a64(Qg_host_.data(), Qg_host_.size() * sizeof(double));
    }
    // D3/FEJ 检查：先验节点0 的 (Lp,z) 与 G_win 必须仍是首次入窗/首次求解时的
    // 冻结值；x 本身允许从 x_first 漂移到当前解（那是优化的正常结果）。
    // 返回 false 时 fej_last_marg_check_reason_ 给出原因；是否中止由调用方决定。
    bool fej_check_frozen_(const char *where) {
        (void)where;
        fej_last_marg_check_ok_ = true;
        fej_last_marg_check_reason_.clear();
        auto fail = [&](const std::string &r) {
            fej_last_marg_check_ok_ = false;
            fej_last_marg_check_reason_ = r;
            return false;
        };
        if (!has_prior_ || !multi_prior_ || prior_node_ > 0) {
            fej_last_marg_check_reason_ = "not_applicable";
            return true;
        }
        if (!fej_first_x_valid_ || (int)fej_first_x_.size() != L_ * n_ * 6)
            return fail("first_x_missing");
        const bool all_nodes = (prior_node_ < 0) &&
                               ((int)fej_prior_z_all_.size() == L_ * n_ * 6 &&
                                (int)fej_prior_Lp_all_.size() == L_ * n_ * 36);
        if (!all_nodes && ((int)fej_prior_z0_.size() != n_ * 6 ||
                           (int)fej_prior_Lp0_.size() != n_ * 36))
            return fail("prior_snapshot_missing");

        const int nz = all_nodes ? L_ * n_ * 6 : n_ * 6;
        const int nl = all_nodes ? L_ * n_ * 36 : n_ * 36;
        std::vector<double> z0((size_t)nz), lp0((size_t)nl);
        cudaMemcpy(z0.data(), z_prior_all_.data_ptr<double>(),
                   (size_t)nz * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(lp0.data(), Lp_all_.data_ptr<double>(),
                   (size_t)nl * sizeof(double), cudaMemcpyDeviceToHost);
        const std::vector<double> &zref = all_nodes ? fej_prior_z_all_ : fej_prior_z0_;
        const std::vector<double> &lref = all_nodes ? fej_prior_Lp_all_ : fej_prior_Lp0_;
        double dz = 0.0, dl = 0.0, zs = 1.0, ls = 1.0;
        for (int a = 0; a < nz; ++a) {
            dz = std::max(dz, std::abs(z0[(size_t)a] - zref[(size_t)a]));
            zs = std::max(zs, std::abs(zref[(size_t)a]));
        }
        for (int a = 0; a < nl; ++a) {
            dl = std::max(dl, std::abs(lp0[(size_t)a] - lref[(size_t)a]));
            ls = std::max(ls, std::abs(lref[(size_t)a]));
        }
        fej_prior_Lp_dev_ = dl;
        if (dz > 1e-10 * std::max(1.0, zs)) return fail("prior_center_changed");
        if (dl > 1e-10 * std::max(1.0, ls)) return fail("prior_Lp_changed");

        double d0 = 0.0, x0s = 1.0;
        const int nk_center = all_nodes ? L_ : 1;
        for (int kk = 0; kk < nk_center; ++kk)
            for (int a = 0; a < n_ * 6; ++a) {
                const size_t q = (size_t)kk * n_ * 6 + a;
                d0 = std::max(d0, std::abs(z0[q] - fej_first_x_[q]));
                x0s = std::max(x0s, std::abs(fej_first_x_[q]));
            }
        fej_prior_center_dev_ = d0;
        if (d0 > 1e-8 * std::max(1.0, x0s)) return fail("prior_center_not_first_x");

        if (fej_x_at_solve_valid_ &&
            (int)fej_x_at_solve_.size() == L_ * n_ * 6) {
            double dx = 0.0, xs = 1.0;
            for (int a = 0; a < L_ * n_ * 6; ++a) {
                dx = std::max(dx,
                              std::abs(fej_x_at_solve_[(size_t)a] - fej_first_x_[(size_t)a]));
                xs = std::max(xs, std::abs(fej_first_x_[(size_t)a]));
            }
            fej_solve_start_dev_ = dx;
            if (dx > 1e-8 * std::max(1.0, xs))
                return fail("solve_start_not_first_x");
        }
        fej_x_drift_ = 0.0;
        if ((int)hX_.size() == L_ * n_ * 6) {
            for (int a = 0; a < L_ * n_ * 6; ++a)
                fej_x_drift_ =
                    std::max(fej_x_drift_, std::abs(hX_[(size_t)a] - fej_first_x_[(size_t)a]));
            if (!std::isfinite(fej_x_drift_))
                fej_x_drift_ = std::numeric_limits<double>::infinity();
        }
        if (has_gauge_) {
            if (!fej_g_frozen_) return fail("gauge_not_frozen_at_prior");
            const uint64_t gh = gauge_G_hash_();
            if (gh != fej_G_hash_frozen_) return fail("G_changed_after_prior");
            if (fej_x_at_solve_valid_ && fej_G_hash_at_solve_ != gh)
                return fail("G_changed_after_solve_start");
        }
        if (cross_active_) {
            if (fej_cross_val_.size() != cross_val_.size())
                return fail("cross_prior_snapshot_missing");
            double dc = 0.0, cs = 1.0;
            for (size_t a = 0; a < cross_val_.size(); ++a) {
                dc = std::max(dc, std::abs(cross_val_[a] - fej_cross_val_[a]));
                cs = std::max(cs, std::abs(fej_cross_val_[a]));
            }
            if (dc > 1e-10 * cs) return fail("cross_prior_changed");
        }
        return true;
    }
    void enforce_fej_(const char *where) {
        const bool ok = fej_check_frozen_(where);
        const char *e = std::getenv("FLS_FEJ_STRICT");
        const bool strict = !(e && std::atoi(e) == 0);
        if (!ok && strict)
            throw std::runtime_error(
                std::string(where) + ": FEJ 检查失败: " + fej_last_marg_check_reason_ +
                "（先验/G 在窗内被改动；可用 FLS_FEJ_STRICT=0 降级为状态报告）");
    }
    // 物理 KG = A_phys^T A_phys G + eps S_c^-2 G。与 build_gauge_woodbury_ 同口径；
    // 只做 k 列稀疏 matmul，不显式构造/求逆 K。
    bool compute_kg_phys_(const torch::Tensor &sc_in, const torch::Device &dev,
                          Eigen::MatrixXd &KGh) {
        if (!has_gauge_ || gauge_k_ <= 0) return false;
        const int k = gauge_k_;
        const int64_t nc = S_.ncol;
        if ((int64_t)Qg_host_.size() != nc * k) return false;
        const auto optsD = torch::TensorOptions().dtype(torch::kFloat64).device(dev);
        auto Gt = torch::from_blob(Qg_host_.data(), {nc, (int64_t)k},
                                   torch::TensorOptions().dtype(torch::kFloat64))
                      .to(dev).contiguous();
        auto sc = sc_in.to(dev).contiguous();
        auto Av = A_val_.to(dev).contiguous();
        auto rnz = row_nz_.to(dev);
        auto aci = A_ci_t_.to(dev);
        auto AG = torch::zeros({(int64_t)S_.nrows, (int64_t)k}, optsD);
        AG.index_add_(0, rnz, Av.view({(int64_t)S_.nnz, 1}) * Gt.index_select(0, aci));
        auto HG = torch::zeros({nc, (int64_t)k}, optsD);
        HG.index_add_(0, aci, Av.view({(int64_t)S_.nnz, 1}) * AG.index_select(0, rnz));
        HG.add_(Gt / (sc * sc).view({nc, 1}) * kGaugeSolveEps);
        auto HGc = HG.cpu().contiguous();
        KGh.resize(nc, k);
        const double *p = HGc.data_ptr<double>();
        for (int64_t a = 0; a < nc; ++a)
            for (int j = 0; j < k; ++j) KGh(a, j) = p[a * k + j];
        return KGh.allFinite();
    }
    double gauge_dinv_cond_() const {
        if (gauge_d_diag_valid_host_) return gauge_d_cond_host_;
        const int ku = gauge_ku_host_;
        if (ku <= 0 || (int)gauge_dinv_host_.size() != ku * ku)
            return std::numeric_limits<double>::quiet_NaN();
        Eigen::Map<const Eigen::MatrixXd> Dinv(gauge_dinv_host_.data(), ku, ku);
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(Dinv);
        const auto &sv = svd.singularValues();
        if (sv.size() == 0) return std::numeric_limits<double>::quiet_NaN();
        const double smax = sv(0), smin = sv(sv.size() - 1);
        return (smax > 0.0 && smin > 0.0)
                   ? smax / smin
                   : std::numeric_limits<double>::infinity();
    }
    // 全窗几何（**全 CUDA**）：QOE → rv 与 ∂p/∂oe；避开宿主 DACE 的反复 init / 线程态开销。
    void geo_all() {
        cudaMemcpy(d_Xa_, hX_.data(), (size_t)L_ * n_ * 6 * sizeof(double), cudaMemcpyHostToDevice);
        qoejopt::oeRvJacDevice(d_Xa_, L_ * n_, d_rvGeo_, d_ApGeo_);
        cudaMemcpy(hRv_.data(), d_rvGeo_, (size_t)L_ * n_ * 6 * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(hAp_.data(), d_ApGeo_, (size_t)L_ * n_ * 18 * sizeof(double), cudaMemcpyDeviceToHost);
    }
    // 列均衡（1/√列能量，上限 1e6）+ Aᵀb
    // 仅列均衡（L=1 局部 dc 只需要 S_c；不必构造 A_s 与 A_s^T b）。
    void scaled_cols_only() {
        auto A2 = A_val_ * A_val_;
        auto cn = torch::zeros({S_.ncol}, A_val_.options());
        cn.index_add_(0, A_ci_t_, A2);
        mtm_scl_ = torch::clamp(1.0 / torch::sqrt(torch::clamp_min(cn, 1e-30)), 0.0, 1e6);
    }

    std::pair<torch::Tensor, torch::Tensor> scaled_AtA_Atb() {
        auto A2 = A_val_ * A_val_;
        auto cn = torch::zeros({S_.ncol}, A_val_.options());
        cn.index_add_(0, A_ci_t_, A2);
        mtm_scl_ = torch::clamp(1.0 / torch::sqrt(torch::clamp_min(cn, 1e-30)), 0.0, 1e6);
        auto Av_s = A_val_ * mtm_scl_.index_select(0, A_ci_t_);
        auto Atb = torch::zeros({S_.ncol}, A_val_.options());
        Atb.index_add_(0, A_ci_t_, Av_s * b_.index_select(0, row_nz_));
        // 局部联合边界先验的跨变量块：RHS = Λ_s (z-x)/sc，加进 scaled Atb。
        if (cross_active_ && !cross_val_.empty()) {
            auto sc_cpu = mtm_scl_.to(torch::kCPU).contiguous();
            std::vector<double> scv((size_t)S_.ncol);
            std::copy(sc_cpu.data_ptr<double>(), sc_cpu.data_ptr<double>() + S_.ncol,
                      scv.begin());
            std::vector<double> cr;
            add_cross_rhs_(scv, cr);
            auto crt = torch::from_blob(cr.data(), {(int64_t)S_.ncol},
                                        torch::TensorOptions().dtype(torch::kFloat64))
                           .to(A_val_.device());
            Atb.add_(crt);
        }
        // 步骤 4：Atb_prior 原为 Λ0 e，投影后应为 P_G0 Λ0 P_G0 e = Λ0 e - ΔΛ e。
        // 这里只改 RHS（Hessian 的 -P ΔΛ P 由 build_gauge_woodbury_ 的增广项处理）。
        if (prior_proj_lowrank_ && has_gauge_) {
            Eigen::VectorXd e, de;
            if (prior_projection_vectors_(e, de)) {
                auto corr = torch::from_blob(de.data(), {(int64_t)S_.ncol},
                                             torch::TensorOptions().dtype(torch::kFloat64))
                                .to(A_val_.device());
                Atb.add_(-mtm_scl_ * corr);
            }
        }
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

    // 联合 scaled 法方程 H_s Y = b_s 的相消感知相对残差（诊断；sweep/joint 共用同一
    // 口径，便于判断「两条路径是否同解」）。H_s 的条目跨 ~20 个数量级（dyn 行互相
    // 抵消），裸残差 max|H_sY−b_s| 只能作上界；这里的分母取双精度算 H_sY 的舍入
    // 上界 |b_q| + Σ_j |H_s[q,j]||Y_j|，故 rel_scaled ~1e-12 即等价于「同一线性方程
    // 组被两条不同消去序解到机器精度」。返回 rel_scaled，并打印一整套。
    double sweep_scaled_resid_(const torch::Tensor &As, const torch::Tensor &bs,
                               const torch::Tensor &Yflat, double t_fwd, double t_bwd,
                               int diag_flag) {
        auto Ay = torch::zeros({S_.nrows}, As.options());
        Ay.index_add_(0, row_nz_, As * Yflat.index_select(0, A_ci_t_));
        auto Hd = torch::zeros({S_.ncol}, As.options());
        Hd.index_add_(0, A_ci_t_, As * Ay.index_select(0, row_nz_));
        auto Yabs = Yflat.abs();
        auto Aya = torch::zeros({S_.nrows}, As.options());
        Aya.index_add_(0, row_nz_, As.abs() * Yabs.index_select(0, A_ci_t_));
        auto Habs = torch::zeros({S_.ncol}, As.options());
        Habs.index_add_(0, A_ci_t_, As.abs() * Aya.index_select(0, row_nz_));
        const double rn = ((Hd - bs).abs() / (Habs + bs.abs() + 1e-300)).max().item<double>();
        const double rmax = (Hd - bs).abs().max().item<double>();
        const double bmax = bs.abs().max().item<double>();
        fprintf(stderr,
                "[sweep/res] fwd=%.2f bwd=%.2f ms diag=%d res_max=%.3e b_max=%.3e rel=%.3e "
                "rel_scaled=%.3e\n",
                1e3 * t_fwd, 1e3 * t_bwd, diag_flag, rmax, bmax, rmax / (bmax + 1e-300), rn);
        return rn;
    }

    // 逐帧消去成本量化（FLS_SWEEP_BENCH=1，只在 solve_batch 入口跑一次）：
    // 对比单帧因子 vs 批量 L 帧因子的 create/add_MtM/factor/solve/selinv 成本，
    // 用于决定 sweep 的批量策略（批量能否摊薄 BaSpaCho 的逐超节点调用开销）。
    void sweep_bench_() {
        using clk = std::chrono::steady_clock;
        auto now = [] {
            return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
        };
        auto optsD = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA);
        const int64_t nc = (int64_t)n_ * 6;
        auto vals1 = torch::empty({1, (int64_t)F_.nnz}, optsD);
        auto valsB = torch::empty({(int64_t)L_, (int64_t)F_.nnz}, optsD);
        auto add1 = torch::zeros({1, nc * 6}, optsD);
        auto addB = torch::zeros({(int64_t)L_, nc * 6}, optsD);
        for (int m = 0; m < L_; ++m) {
            gather_frame_(m, A_val_);
            if (m == 0) vals1[0].copy_(Af_t_);
            valsB[m].copy_(Af_t_);
        }
        auto rhs1 = torch::ones({1, nc}, optsD);
        auto rhsB = torch::ones({(int64_t)L_, nc}, optsD);
        double t = now();
        auto nd1 = symF_->createNumericDecomposition(1);
        const double t_c1 = now() - t;
        t = now();
        nd1.add_MtM(vals1, ArpF_t_, AciF_t_);
        nd1.add_M(add1, rpC_t_, ciC_t_);
        const double t_a1 = now() - t;
        t = now(); nd1.factor(); const double t_f1 = now() - t;
        t = now(); nd1.solve(rhs1); const double t_s1 = now() - t;
        t = now(); auto blk1 = nd1.compute_diag_blocks(6); const double t_i1 = now() - t;
        t = now();
        auto ndB = symF_->createNumericDecomposition(L_);
        const double t_cB = now() - t;
        t = now();
        ndB.add_MtM(valsB, ArpF_t_, AciF_t_);
        ndB.add_M(addB, rpC_t_, ciC_t_);
        const double t_aB = now() - t;
        t = now(); ndB.factor(); const double t_fB = now() - t;
        t = now(); ndB.solve(rhsB); const double t_sB = now() - t;
        t = now(); auto blkB = ndB.compute_diag_blocks_batch(6); const double t_iB = now() - t;
        double t_cs = 0, t_as = 0, t_fs = 0, t_ss = 0, t_is = 0;
        for (int m = 0; m < L_; ++m) {
            t = now();
            auto nd = symF_->createNumericDecomposition(1);
            t_cs += now() - t;
            t = now();
            nd.add_MtM(valsB[m].view({1, (int64_t)F_.nnz}), ArpF_t_, AciF_t_);
            nd.add_M(add1, rpC_t_, ciC_t_);
            t_as += now() - t;
            t = now(); nd.factor(); t_fs += now() - t;
            auto r = rhs1.clone();
            t = now(); nd.solve(r); t_ss += now() - t;
            t = now(); auto bb = nd.compute_diag_blocks(6); t_is += now() - t;
        }
        fprintf(stderr,
                "[sweep/bench] F_nnz=%lld frame_data=%.3fM | 1x: create=%.2f add=%.2f "
                "factor=%.2f solve=%.2f selinv=%.2f | %dx: create=%.2f add=%.2f factor=%.2f "
                "solve=%.2f selinvB=%.2f | serial %dx: create=%.2f add=%.2f factor=%.2f "
                "solve=%.2f selinv=%.2f ms\n",
                (long long)F_.nnz, 1e-6 * (double)nd1.data_size(), 1e3 * t_c1, 1e3 * t_a1,
                1e3 * t_f1, 1e3 * t_s1, 1e3 * t_i1, L_, 1e3 * t_cB, 1e3 * t_aB, 1e3 * t_fB,
                1e3 * t_sB, 1e3 * t_iB, L_, 1e3 * t_cs, 1e3 * t_as, 1e3 * t_fs, 1e3 * t_ss,
                1e3 * t_is);
        // 备选 selected-inverse 与多 RHS solve：逐列 OMP 版 与 multi-RHS（链式前代用）。
        t = now();
        auto blkF0 = nd1.compute_diag_blocks_frame0(6, n_);
        const double t_iF0 = now() - t;
        t = now();
        auto blkF0B = ndB.compute_diag_blocks_frame0(6, n_);
        const double t_iF0B = now() - t;
        auto rhsM4 = torch::ones({1, (int64_t)4 * nc}, optsD);
        t = now(); nd1.solve_multi(rhsM4, 4); const double t_sm4 = now() - t;
        auto rhsM8 = torch::ones({1, (int64_t)8 * nc}, optsD);
        t = now(); ndB.solve_multi(rhsM8, 8); const double t_sm8b = now() - t;
        fprintf(stderr,
                "[sweep/bench] selinv_frame0 1x=%.2f batch=%.2f | solve_multi nrhs=4 (1 frame)=%.2f "
                "nrhs=8 (batch %d)=%.2f ms\n",
                1e3 * t_iF0, 1e3 * t_iF0B, 1e3 * t_sm4, L_, 1e3 * t_sm8b);
        (void)blk1; (void)blkB; (void)blkF0; (void)blkF0B;
    }

    // ================= 窗口级 gauge（QOE 物理基，非 QR）=================
    // 物理增量 dc = S_c y。因此实际物理算子
    //   K = A_phys^T A_phys + eps S_c^-2 = S_c^-1 (A_s^T A_s + eps I) S_c^-1，
    // 且 K^-1 b_phys 正是现有 solve 的 dc。以下所有 G/Pg 公式都在物理 QOE 坐标。
    // 记 S = G^T G + eps_g I，Pi = G S^-1 G^T，U=[G, K G]，则
    //   Hwin = (I-Pi)K(I-Pi)+G Q G^T = K + U C U^T,
    //   C = [[Q+S^-1(KG)^T G S^-1, -S^-1],[-S^-1, 0]], Q = Pg_eff^-1.
    // Woodbury: D = C^-1 + U^T K^-1 U = [[G^T X, -eps_g I],[-eps_g I, -S Q S]]
    //   （X=K^-1 G, 用到 U^T K^-1 U = [[XG,Sg],[Sg,M]]）
    //   Hwin^-1 = K^-1 - [X,G] D^-1 [X,G]^T.
    // 正交 Q（max|G^T G-I| 很小）时 S=I+eps_g I，交叉块 -eps_g I 只是数值
    // 噪声；build_gauge_woodbury_ 将其置零并按 D11=XG、D22=-S Q S 两个
    // 6×6 块分别做均衡 SVD 伪逆，12 个 gauge 方向一个不丢。非正交物理 G
    // 仍保留完整 D，并对整块做均衡 SVD 伪逆（rank loss 截断而非抛错）。
    // 该式对 marginalize 的 rr 块同样精确（无需显式 ncol×ncol 投影矩阵）。
    // 协方差侧（物理 G 不做 QR/SVD，G^T G != I）：
    //   Hwin G = G Q S  =>  G^T P G = Q^-1 = Pg_eff,
    //   P = G S^-1 Pg_eff S^-1 G^T + P_perp,  G^T P_perp = 0.
    // doc 的 P=G Pg G^T+P_perp 以及 G^T Qeff G 的逐块写法默认 G^T G=I；
    // 本实现必须保留 S^-1 夹心（S=G^T G+eps_g I）。gauge_cov_diagnostics()
    // 用末次 Woodbury 数值检查 G^T P G=Pg_eff 与 G^T P_perp=0；G^T K G 由
    // gauge_hg_host()=K G 继续作为 Step-6 的 G^T Qeff G 口径。

    // 汇总 Pg_eff^-1 = Pg^-1 + (split 时的 GTS G 分量信息)。
    void update_pg_eff_() {
        if (!has_gauge_ || gauge_k_ <= 0) {
            Pg_eff_host_.clear();
            Pg_eff_inv_host_.clear();
            return;
        }
        const int k = gauge_k_;
        if ((int)Pg_inv_host_.size() != k * k) {
            Pg_eff_host_.clear();
            Pg_eff_inv_host_.clear();
            return;
        }
        // Λ_g 是 GTS G 分量的 Gram 和；数值上可能有 ≤1e-17 相对量的负模。
        // 先按同一相对阈值做 PSD 截断，只让 λ>cut 的方向进入 Pg_eff，
        // 负/零方向保持输入 Pg，避免无效 Fisher 被 eigen-floor 反成大方差。
        Eigen::MatrixXd Lg = Eigen::MatrixXd::Zero(k, k);
        if (gauge_gts_split_ && (int)Pg_info_host_.size() == k * k) {
            for (int a = 0; a < k; ++a)
                for (int b = 0; b < k; ++b)
                    Lg(a, b) = Pg_info_host_[(size_t)a * k + b];
            Lg = 0.5 * (Lg + Lg.transpose());
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> esL(Lg);
            const Eigen::VectorXd wl = esL.eigenvalues();
            const double lcut = std::max(
                (wl.size() ? wl.cwiseAbs().maxCoeff() : 0.0) * 1e-12, 1e-300);
            Lg.setZero();
            for (int i = 0; i < k; ++i) {
                if (!(wl(i) > lcut)) continue;
                const Eigen::VectorXd ui = esL.eigenvectors().col(i);
                Lg.noalias() += wl(i) * ui * ui.transpose();
            }
        }
        Eigen::MatrixXd A(k, k);
        for (int a = 0; a < k; ++a)
            for (int b = 0; b < k; ++b)
                A(a, b) = Pg_inv_host_[(size_t)a * k + b] + Lg(a, b);
        A = 0.5 * (A + A.transpose());
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(A);
        Eigen::VectorXd w = es.eigenvalues();
        // 同上：只抬非正特征值，避免平移块把旋转块信息抬掉。
        const double fl = std::max(w.maxCoeff() * 1e-14, 1e-300);
        for (int i = 0; i < k; ++i)
            if (!(w(i) > 0.0)) w(i) = fl;
        const Eigen::MatrixXd V = es.eigenvectors();
        const Eigen::MatrixXd Afl = V * w.asDiagonal() * V.transpose();
        const Eigen::MatrixXd Ai = V * w.cwiseInverse().asDiagonal() * V.transpose();
        Pg_eff_inv_host_.assign((size_t)k * k, 0.0);
        Pg_eff_host_.assign((size_t)k * k, 0.0);
        for (int a = 0; a < k; ++a)
            for (int b = 0; b < k; ++b) {
                Pg_eff_inv_host_[(size_t)a * k + b] = Afl(a, b);
                Pg_eff_host_[(size_t)a * k + b] = Ai(a, b);
            }
    }

    // 步骤 4 的 RHS/代价修正：e=z_prior-x0（仅节点0），de=ΔΛ e，
    // ΔΛ=Λ0-P_G0Λ0P_G0。返回 false 表示低秩投影未启用/数据失效。
    bool prior_projection_vectors_(Eigen::VectorXd &e, Eigen::VectorXd &de) const {
        const int k = gauge_k_;
        const int64_t nc = (int64_t)n_ * 6 * L_;
        e = Eigen::VectorXd::Zero(nc);
        de = Eigen::VectorXd::Zero(nc);
        if (!prior_proj_lowrank_ || k <= 0) return false;
        if ((int)prior_proj_A0_host_.size() != n_ * 6 * k ||
            (int)prior_proj_G0_host_.size() != n_ * 6 * k ||
            (int)prior_proj_B0_host_.size() != k * k ||
            (int)prior_proj_S0inv_host_.size() != k * k)
            return false;
        const torch::Tensor &zp_src = multi_prior_ ? z_prior_all_ : z_prior_;
        std::vector<double> zp((size_t)n_ * 6);
        cudaMemcpy(zp.data(), zp_src.data_ptr<double>(), (size_t)n_ * 6 * sizeof(double),
                   cudaMemcpyDeviceToHost);
        Eigen::MatrixXd A0(n_ * 6, k), G0(n_ * 6, k), B0(k, k), S0i(k, k);
        for (int a = 0; a < n_ * 6; ++a)
            for (int j = 0; j < k; ++j) {
                A0(a, j) = prior_proj_A0_host_[(size_t)a * k + j];
                G0(a, j) = prior_proj_G0_host_[(size_t)a * k + j];
            }
        for (int a = 0; a < k; ++a)
            for (int b = 0; b < k; ++b) {
                B0(a, b) = prior_proj_B0_host_[(size_t)a * k + b];
                S0i(a, b) = prior_proj_S0inv_host_[(size_t)a * k + b];
            }
        for (int i = 0; i < n_ * 6; ++i) e(i) = zp[(size_t)i] - hX_[(size_t)i];
        const Eigen::VectorXd g0e = G0.transpose() * e;
        const Eigen::VectorXd a0e = A0.transpose() * e;
        const Eigen::VectorXd sg = S0i * g0e;
        const Eigen::VectorXd sa = S0i * a0e;
        const Eigen::VectorXd sbs = S0i * (B0 * sg);
        de.head(n_ * 6) = A0 * sg + G0 * sa - G0 * sbs;
        return true;
    }

    // 步骤 5：GTS 的 G 子空间信息只进 Pg，不改本地 A 行。
    // 对每条已白化 GTS 行 v（物理 QOE，单星 6 列）：
    //   Λ_g += S^-1 (G^T v)(G^T v)^T S^-1,  S=G^T G+eps_g I.
    // 本地块保持原始 v；窗口 Hwin=(I-Pi)K(I-Pi)+G(Pg^-1+Λ_g)G^T 中的 (I-Pi)
    // 自动给出 J_perp=H_gts(I-Pi)，与 doc 第 5 步的局部块一致。若在这里把 A 行
    // 改成星内投影 v-Gb S^-1(Gb^T v)，会与全局 Pi 不一致并残留 gauge 分量；
    // 保持 raw row 既精确又天然幂等（assemble 每次重写 raw 行）。
    // 残差 b 不需要拆分：A^T b 中 G^T v 的 gauge 分量正是 GTS 的 gauge RHS，
    // 由 Woodbury 的 gb=G^T Atb 自动进入 G(Pg^-1+Λ_g)G^T δ = Pg^-1 z + r_g。
    //
    // 跨窗节点范围（修复 GTS 的 G 分量重复计数）：
    //   Pg_next = U_r^T Hwin^-1 U_r 已经包含本窗全部节点（含被消掉的 node0）
    //   的 GTS、ISL、dyn 与 Pg 的联合后验。下一窗若再把 node0..L-2 的 GTS G
    //   信息加进 Pg，就与上一窗 Pg_next 重复。正确规则（按当前窗口节点索引）：
    //     首窗（无上一窗 Pg_next）：[0, L)，全部节点；
    //     滑窗（有上一窗 Pg_next）：[L-1, L)，只让新进入的末节点贡献。
    //   本函数只 mask Pg 的 G 信息累加；A_val/b 中的 GTS 行始终保持 raw/whole，
    //   状态求解的 J_perp 由窗口 (I-Pi) 给出，不受此 mask 影响。
    //   范围是半开区间 [first,last)；默认 [0,L)，由 Python 在每窗显式设置。
    void apply_gts_gauge_split_() {
        if (!has_gauge_ || gauge_k_ <= 0) return;
        const int k = gauge_k_;
        Pg_info_host_.assign((size_t)k * k, 0.0);
        if (!gauge_gts_split_ || G_ <= 0) {
            update_pg_eff_();
            return;
        }
        const int pf = std::max(0, std::min(gts_pg_node_first_, L_));
        const int pl = std::max(pf, std::min(gts_pg_node_last_, L_));
        const int64_t nc = S_.ncol;
        Eigen::MatrixXd Gmat(nc, k);
        for (int64_t a = 0; a < nc; ++a)
            for (int j = 0; j < k; ++j) Gmat(a, j) = Qg_host_[(size_t)a * k + j];
        Eigen::MatrixXd Sg = Gmat.transpose() * Gmat;
        Sg = 0.5 * (Sg + Sg.transpose());
        const double epsg = gauge_eps_from_gram(Sg);
        const Eigen::MatrixXd S =
            Sg + epsg * Eigen::MatrixXd::Identity(k, k);
        Eigen::LLT<Eigen::MatrixXd> llt_s(S);
        if (llt_s.info() != Eigen::Success)
            throw std::runtime_error("apply_gts_gauge_split_: S 非正定");
        const Eigen::MatrixXd Sinv =
            llt_s.solve(Eigen::MatrixXd::Identity(k, k));

        const int nGTS = L_ * G_;
        std::vector<double> av((size_t)nGTS * 6, 0.0);
        cudaDeviceSynchronize();   // prior 行可能刚由 default stream 写入
        cudaMemcpy(av.data(), A_val_.data_ptr<double>() + S_.off_gts,
                   (size_t)nGTS * 6 * sizeof(double), cudaMemcpyDeviceToHost);
        std::vector<double> pinfo((size_t)k * k, 0.0);
        for (int idx = 0; idx < nGTS; ++idx) {
            const int node = idx / G_;
            if (node < pf || node >= pl) continue;
            const int star = gts_[idx % G_];
            const int64_t base = (int64_t)(node * n_ + star) * 6;
            Eigen::Matrix<double, 6, 1> v;
            for (int c = 0; c < 6; ++c) v(c) = av[(size_t)idx * 6 + c];
            Eigen::MatrixXd Gb(6, k);
            for (int c = 0; c < 6; ++c)
                for (int j = 0; j < k; ++j) Gb(c, j) = Gmat(base + c, j);
            const Eigen::VectorXd g6 = Gb.transpose() * v;
            const Eigen::VectorXd sg = Sinv * g6;          // S^-1 G^T v
            for (int a = 0; a < k; ++a)
                for (int b = 0; b < k; ++b)
                    pinfo[(size_t)a * k + b] += sg(a) * sg(b);
        }
        Pg_info_host_ = pinfo;
        update_pg_eff_();
    }

    // rank-2k gauge Woodbury 数据（步骤 3；prior_proj 时扩到 4k）。
    //   base:  U=[G,KG], Utot=K^-1 U=[X,G], D=C^-1+U^T Utot.
    //   +步骤4: U=[G,KG,Y,Z], Utot=[X,G,KY,KZ], Y=P A0, Z=P G0,
    //           并追加 -P ΔΛ P 的 rank-2k 项（C2=-E）。
    struct GaugeWoodburyData {
        int k = 0;
        int ku = 0;
        bool prior_proj = false;
        bool orthonormal = false;  // G^T G ≈ I：D 用精确块对角逆
        double eps_g = 0.0;
        Eigen::MatrixXd U;      // ncol × ku
        Eigen::MatrixXd Utot;   // ncol × ku
        Eigen::MatrixXd Dinv;   // ku × ku
        Eigen::MatrixXd S;      // G^T G + eps_g I
        Eigen::MatrixXd Sinv;   // S^-1
        Eigen::MatrixXd Q;      // Pg_eff^-1（D 的 gauge 块与 RHS 中心项共用）
    };

    // 构造 gauge Woodbury 数据。cpu=true 时用 CPU 因子（边缘化）。
    // 返回 false 表示 D 不可逆/非有限（调用者应报错，不做静默降级）。
    bool build_gauge_woodbury_(NumericDecomposition *nd, const torch::Tensor &sc_in,
                               bool cpu, GaugeWoodburyData &out) {
        gauge_woodbury_host_valid_ = false;
        gauge_dinv_hash_host_ = 0;
        gauge_dinv_fro_host_ = 0.0;
        gauge_d_diag_valid_host_ = false;
        gauge_d_rank_host_ = 0;
        gauge_d_cond_host_ = std::numeric_limits<double>::infinity();
        gauge_d_cond_bal_host_ = std::numeric_limits<double>::infinity();
        gauge_d11_rank_host_ = 0;
        gauge_d11_cond_host_ = std::numeric_limits<double>::infinity();
        gauge_d11_cond_bal_host_ = std::numeric_limits<double>::infinity();
        gauge_d22_rank_host_ = 0;
        gauge_d22_cond_host_ = std::numeric_limits<double>::infinity();
        gauge_d22_cond_bal_host_ = std::numeric_limits<double>::infinity();
        gauge_d_svd_trunc_host_ = 0;
        gauge_d_blockdiag_host_ = false;
        gauge_d_orthonormal_host_ = false;
        gauge_d_orth_dev_host_ = 0.0;
        if (!has_gauge_ || gauge_k_ <= 0 || nd == nullptr) return false;
        const int k = gauge_k_;
        if ((int)Pg_eff_inv_host_.size() != k * k) update_pg_eff_();
        if ((int)Pg_eff_inv_host_.size() != k * k) return false;
        const int64_t nc = S_.ncol;
        const torch::Device dev = cpu ? torch::Device(torch::kCPU)
                                      : torch::Device(torch::kCUDA);
        auto Gt = torch::from_blob(Qg_host_.data(), {nc, (int64_t)k},
                                   torch::TensorOptions().dtype(torch::kFloat64))
                      .to(dev).contiguous();
        auto sc = sc_in.to(dev).contiguous();

        // X = K^-1 G（物理）：scaled RHS = S_c G，解后乘回 S_c。
        Eigen::MatrixXd Xh(nc, k);
        auto Bs = Gt * sc.view({nc, 1});
        for (int j = 0; j < k; ++j) {
            auto yj = Bs.select(1, j).reshape({1, nc}).clone();
            nd->solve(yj);
            auto xj = (yj.reshape({nc}) * sc).cpu().contiguous();
            const double *p = xj.data_ptr<double>();
            for (int64_t a = 0; a < nc; ++a) Xh(a, j) = p[a];
        }

        if (!Xh.allFinite()) {
            fprintf(stderr, "[gauge/D] K^-1 G 非有限（factor/solve 失败）\n");
            return false;
        }
        // K G = A_phys^T A_phys G + eps S_c^-2 G（稀疏 matmul，不显式 H）。
        // 与 gauge_lamwin_host 共用 compute_kg_phys_，保证 D1 的 gauge 块和
        // Woodbury 的 D 使用完全相同的 K 口径。
        Eigen::MatrixXd KGh;
        if (!compute_kg_phys_(sc_in, dev, KGh)) return false;

        Eigen::MatrixXd Gmat(nc, k);
        for (int64_t a = 0; a < nc; ++a)
            for (int j = 0; j < k; ++j) Gmat(a, j) = Qg_host_[(size_t)a * k + j];
        Eigen::MatrixXd Sg = Gmat.transpose() * Gmat;
        Sg = 0.5 * (Sg + Sg.transpose());
        const double epsg = gauge_eps_from_gram(Sg);
        const Eigen::MatrixXd S = Sg + epsg * Eigen::MatrixXd::Identity(k, k);
        Eigen::LLT<Eigen::MatrixXd> llt_s(S);
        if (llt_s.info() != Eigen::Success) return false;
        const Eigen::MatrixXd Sinv = llt_s.solve(Eigen::MatrixXd::Identity(k, k));
        Eigen::MatrixXd Q(k, k);
        for (int a = 0; a < k; ++a)
            for (int b = 0; b < k; ++b)
                Q(a, b) = Pg_eff_inv_host_[(size_t)a * k + b];
        Q = 0.5 * (Q + Q.transpose());
        Eigen::MatrixXd XG = Gmat.transpose() * Xh;
        XG = 0.5 * (XG + XG.transpose());
        Eigen::MatrixXd D22_blk = -S * Q * S;
        D22_blk = 0.5 * (D22_blk + D22_blk.transpose());

        // 正交 Q 检测：max|G^T G-I| 足够小 => S=I+eps_g I。此时
        // D=[[XG,-eps_g I],[-eps_g I,-S Q S]] 的交叉块只是 S=Sg+eps_g I
        // 带来的数值噪声（eps_g 相对两块主元小 10~20 个量级），取 eps_g→0
        // 后 D 精确块对角，可按 6×6 块独立求逆，12 个方向一个不丢。
        double orth_dev = 0.0;
        for (int a = 0; a < k; ++a)
            for (int b = 0; b < k; ++b)
                orth_dev = std::max(
                    orth_dev, std::abs(Sg(a, b) - ((a == b) ? 1.0 : 0.0)));
        const bool orthonormal = (orth_dev <= 1e-8);

        // 两块各自的均衡 SVD 伪逆：rank loss 只截断、不抛错。
        GaugePinvDiag d11_diag, d22_diag;
        Eigen::MatrixXd D11_inv, D22_inv;
        if (!XG.allFinite() || !D22_blk.allFinite()) {
            fprintf(stderr, "[gauge/D] nonfinite XG=%d D22=%d |XG|max=%.3e "
                            "|D22|max=%.3e |Xh|max=%.3e |K G|max=%.3e\n",
                    (int)XG.allFinite(), (int)D22_blk.allFinite(),
                    XG.cwiseAbs().maxCoeff(), D22_blk.cwiseAbs().maxCoeff(),
                    Xh.cwiseAbs().maxCoeff(), KGh.cwiseAbs().maxCoeff());
        }
        if (!svd_pinv_balanced_(XG, D11_inv, &d11_diag)) return false;
        if (!svd_pinv_balanced_(D22_blk, D22_inv, &d22_diag)) return false;

        Eigen::MatrixXd D = Eigen::MatrixXd::Zero(2 * k, 2 * k);
        D.block(0, 0, k, k) = XG;
        D.block(k, k, k, k) = D22_blk;
        if (!orthonormal) {
            D.block(0, k, k, k) = -epsg * Eigen::MatrixXd::Identity(k, k);
            D.block(k, 0, k, k) = -epsg * Eigen::MatrixXd::Identity(k, k);
        }
        D = 0.5 * (D + D.transpose());

        out = GaugeWoodburyData();
        out.k = k;
        out.ku = 2 * k;
        out.prior_proj = false;
        out.orthonormal = orthonormal;
        out.eps_g = epsg;
        out.S = S;
        out.Sinv = Sinv;
        out.Q = Q;
        out.U.resize(nc, 2 * k);
        out.U.leftCols(k) = Gmat;
        out.U.rightCols(k) = KGh;
        out.Utot.resize(nc, 2 * k);
        out.Utot.leftCols(k) = Xh;
        out.Utot.rightCols(k) = Gmat;

        GaugePinvDiag dmat_diag;
        bool dmat_blockdiag = false;
        if (orthonormal && !prior_proj_lowrank_) {
            // 精确块对角逆；D 的交叉块已按正交 Q 置零。
            out.Dinv = Eigen::MatrixXd::Zero(2 * k, 2 * k);
            out.Dinv.topLeftCorner(k, k) = D11_inv;
            out.Dinv.bottomRightCorner(k, k) = D22_inv;
            dmat_diag.rank = d11_diag.rank + d22_diag.rank;
            dmat_diag.trunc = d11_diag.trunc + d22_diag.trunc;
            dmat_diag.cond = std::max(d11_diag.cond, d22_diag.cond);
            dmat_diag.cond_bal =
                std::max(d11_diag.cond_bal, d22_diag.cond_bal);
            dmat_blockdiag = true;
        } else {
            // 非正交物理 G（或步骤4的 4k 增广）：整块均衡 SVD 伪逆，
            // 不再用 FullPivLU/isInvertible 硬失败。
            if (!svd_pinv_balanced_(D, out.Dinv, &dmat_diag)) return false;
        }

        if (prior_proj_lowrank_) {
            if ((int)prior_proj_A0_host_.size() != n_ * 6 * k ||
                (int)prior_proj_G0_host_.size() != n_ * 6 * k ||
                (int)prior_proj_B0_host_.size() != k * k ||
                (int)prior_proj_S0_host_.size() != k * k ||
                (int)prior_proj_S0inv_host_.size() != k * k)
                return false;
            // A0/G0 只有节点0非零，嵌入整窗 ncol 维。
            Eigen::MatrixXd A0 = Eigen::MatrixXd::Zero(nc, k);
            Eigen::MatrixXd G0 = Eigen::MatrixXd::Zero(nc, k);
            for (int a = 0; a < n_ * 6; ++a)
                for (int j = 0; j < k; ++j) {
                    A0(a, j) = prior_proj_A0_host_[(size_t)a * k + j];
                    G0(a, j) = prior_proj_G0_host_[(size_t)a * k + j];
                }
            Eigen::MatrixXd B0(k, k), S0(k, k);
            for (int a = 0; a < k; ++a)
                for (int b = 0; b < k; ++b) {
                    B0(a, b) = prior_proj_B0_host_[(size_t)a * k + b];
                    S0(a, b) = prior_proj_S0_host_[(size_t)a * k + b];
                }
            // Y = P A0, Z = P G0（P=I-G S^-1 G^T）。ΔΛ = [A0,G0] E [A0,G0]^T，
            // Hwin 额外项 -P ΔΛ P = -[Y,Z] E [Y,Z]^T，故 C2=-E。
            Eigen::MatrixXd Y = A0 - Gmat * (Sinv * (Gmat.transpose() * A0));
            Eigen::MatrixXd Z = G0 - Gmat * (Sinv * (Gmat.transpose() * G0));
            // KY=K^-1 Y, KZ=K^-1 Z（物理坐标）。
            auto solve_phys_cols = [&](const Eigen::MatrixXd &Bphys,
                                       Eigen::MatrixXd &Xphys) -> bool {
                const int cols = (int)Bphys.cols();
                Xphys.resize(nc, cols);
                std::vector<double> buf((size_t)nc * cols);
                for (int64_t a = 0; a < nc; ++a)
                    for (int j = 0; j < cols; ++j)
                        buf[(size_t)a * cols + j] = Bphys(a, j);
                auto Bt = torch::from_blob(buf.data(), {nc, (int64_t)cols},
                                           torch::TensorOptions().dtype(torch::kFloat64))
                              .to(dev).clone() * sc.view({nc, 1});
                for (int j = 0; j < cols; ++j) {
                    auto yj = Bt.select(1, j).reshape({1, nc}).clone();
                    nd->solve(yj);
                    auto xj = (yj.reshape({nc}) * sc).cpu().contiguous();
                    const double *p = xj.data_ptr<double>();
                    for (int64_t a = 0; a < nc; ++a) Xphys(a, j) = p[a];
                }
                return Xphys.allFinite();
            };
            Eigen::MatrixXd KY, KZ;
            if (!solve_phys_cols(Y, KY) || !solve_phys_cols(Z, KZ)) return false;

            out.ku = 4 * k;
            out.prior_proj = true;
            out.U.resize(nc, 4 * k);
            out.U.leftCols(k) = Gmat;
            out.U.middleCols(k, k) = KGh;
            out.U.middleCols(2 * k, k) = Y;
            out.U.rightCols(k) = Z;
            out.Utot.resize(nc, 4 * k);
            out.Utot.leftCols(k) = Xh;
            out.Utot.middleCols(k, k) = Gmat;
            out.Utot.middleCols(2 * k, k) = KY;
            out.Utot.rightCols(k) = KZ;

            Eigen::MatrixXd D12 = Eigen::MatrixXd::Zero(2 * k, 2 * k);
            D12.topLeftCorner(k, k) = Gmat.transpose() * KY;
            D12.topRightCorner(k, k) = Gmat.transpose() * KZ;
            D12.bottomLeftCorner(k, k) = Gmat.transpose() * Y;   // (KG)^T K^-1 Y
            D12.bottomRightCorner(k, k) = Gmat.transpose() * Z;  // (KG)^T K^-1 Z
            Eigen::MatrixXd D22 = Eigen::MatrixXd::Zero(2 * k, 2 * k);
            D22.topLeftCorner(k, k) = -B0 + Y.transpose() * KY;
            D22.topRightCorner(k, k) = -S0 + Y.transpose() * KZ;
            D22.bottomLeftCorner(k, k) = -S0 + Z.transpose() * KY;
            D22.bottomRightCorner(k, k) = Z.transpose() * KZ;
            Eigen::MatrixXd Dtot = Eigen::MatrixXd::Zero(4 * k, 4 * k);
            Dtot.topLeftCorner(2 * k, 2 * k) = D;
            Dtot.topRightCorner(2 * k, 2 * k) = D12;
            Dtot.bottomLeftCorner(2 * k, 2 * k) = D12.transpose();
            Dtot.bottomRightCorner(2 * k, 2 * k) = D22;
            Dtot = 0.5 * (Dtot + Dtot.transpose());
            GaugePinvDiag dtot_diag;
            if (!svd_pinv_balanced_(Dtot, out.Dinv, &dtot_diag)) return false;
            dmat_diag = dtot_diag;
            dmat_blockdiag = false;   // 4k 增广含 D12 耦合，最终 Dinv 非块对角
        }

        // 末次 D 的数值诊断（供 gauge_woodbury_diag / Python CSV 读取）。
        gauge_d_rank_host_ = dmat_diag.rank;
        gauge_d_cond_host_ = dmat_diag.cond;
        gauge_d_cond_bal_host_ = dmat_diag.cond_bal;
        gauge_d11_rank_host_ = d11_diag.rank;
        gauge_d11_cond_host_ = d11_diag.cond;
        gauge_d11_cond_bal_host_ = d11_diag.cond_bal;
        gauge_d22_rank_host_ = d22_diag.rank;
        gauge_d22_cond_host_ = d22_diag.cond;
        gauge_d22_cond_bal_host_ = d22_diag.cond_bal;
        gauge_d_svd_trunc_host_ = dmat_diag.trunc;
        gauge_d_blockdiag_host_ = dmat_blockdiag;
        gauge_d_orthonormal_host_ = orthonormal;
        gauge_d_orth_dev_host_ = orth_dev;
        gauge_d_diag_valid_host_ = true;
        if (std::getenv("FLS_GAUGEDBG")) {
            fprintf(stderr,
                    "[gauge/D] ku=%d orth=%.3e blockdiag=%d prior=%d "
                    "rank=%d/%d trunc=%d cond=%.3e cond_bal=%.3e D11_rank=%d "
                    "D11_cond=%.3e D22_rank=%d D22_cond=%.3e eps_g=%.3e\n",
                    out.ku, orth_dev, (int)dmat_blockdiag, (int)out.prior_proj,
                    dmat_diag.rank,
                    out.ku, dmat_diag.trunc, dmat_diag.cond,
                    dmat_diag.cond_bal, d11_diag.rank, d11_diag.cond,
                    d22_diag.rank, d22_diag.cond, epsg);
        }

        gauge_xg_host_.assign((size_t)nc * k, 0.0);
        gauge_hg_host_.assign((size_t)nc * k, 0.0);
        for (int64_t a = 0; a < nc; ++a)
            for (int j = 0; j < k; ++j) {
                gauge_xg_host_[(size_t)a * k + j] = Xh(a, j);
                gauge_hg_host_[(size_t)a * k + j] = KGh(a, j);
            }
        // 保存末次 Woodbury 的 Utot/Dinv，供 gauge_cov_diagnostics() 计算
        // P=Hwin^-1 在 G 子空间/补空间的一致性（与 gauge_xg/hg 同源）。
        // Eigen 列主序，std::vector 原样拷贝，Map 时用相同行列与列主序。
        gauge_ku_host_ = out.ku;
        gauge_utot_host_.assign(out.Utot.data(), out.Utot.data() + out.Utot.size());
        gauge_dinv_host_.assign(out.Dinv.data(), out.Dinv.data() + out.Dinv.size());
        gauge_dinv_fro_host_ = out.Dinv.norm();
        gauge_dinv_hash_host_ =
            fnv1a64(out.Dinv.data(), out.Dinv.size() * sizeof(double));
        gauge_pgeff_build_host_ = Pg_eff_host_;
        gauge_woodbury_host_valid_ = (int)gauge_pgeff_build_host_.size() == k * k;
        return true;
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
    int prior_node_ = -1;
    int dyn_sub_ = 1;                                  // >=0：结构只建该节点的 prior 行
    int r_cross_ = 1;                                  // >=2：块结构加 ISL r 跳邻域（局部联合先验）
    int dyn_sub_cfg_ = 1;                              // 构造时 KF_DYN_SUB，GVE 切换时恢复
    double qphys_sigma_ = 1e-7;                        // set_Q_phys 的 sigma_a
    std::vector<double> qphys_Q_host_;                 // set_Q_phys_arr 的物理 Q（诊断）
    FlsStruct S_;
    qoejopt::WinCtx wc_;
    int *d_isl_i_ = nullptr, *d_isl_jj_ = nullptr, *d_gts_a_ = nullptr;
    double *d_qingt_ = nullptr, *d_isl_D_ = nullptr, *d_isl_W_ = nullptr;
    double *d_gts_D_ = nullptr, *d_gts_W_ = nullptr, *d_gts_Q_ = nullptr, *d_ctr_ = nullptr;
    double *d_Xa_ = nullptr, *d_rv_ = nullptr, *d_Ap_ = nullptr;
    // D7：Px 上的临时几何缓冲（不覆盖 wc_.d_X/d_rv/d_Ap）
    double *d_Xcost_ = nullptr, *d_rvCost_ = nullptr, *d_ApCost_ = nullptr;
    double *d_end_ = nullptr, *d_phi_ = nullptr, *d_res_ = nullptr, *d_Pm_ = nullptr;
    double *d_ekf_scale_ = nullptr;                 // 固定每星缩放 D（L=1 EKF 缩放空间）
    std::vector<double> d_ekf_host_;
    double *d_state_sub_ = nullptr, *d_state_b_ = nullptr;
    double *d_phi_acc_ = nullptr, *d_phi_out_ = nullptr, *d_phi_step_ = nullptr;
    double *d_qingt_key_ = nullptr;
    double *d_ekf_xref_ = nullptr, *d_ekf_xpred_ = nullptr, *d_ekf_phi_ = nullptr;
    double *d_ekf_Pprev_ = nullptr, *d_ekf_Ppred_ = nullptr, *d_ekf_qeff_ = nullptr;
    // 全 N gauge 修正（固定 EKF dyn）：每区间 Ga/Gb (6N,k)、C(k,k)、corr(L−1,n,6)
    double *d_gauge_Ga_ = nullptr, *d_gauge_Gb_ = nullptr, *d_gauge_C_ = nullptr;
    double *d_gauge_corr_ = nullptr;
    std::vector<double> dyn_xref_host_, dyn_phi_host_;      // set_dyn_ekf_* 的 host 副本（诊断）
    std::vector<double> dyn_gauge_ga_host_, dyn_gauge_gb_host_, dyn_gauge_c_host_;
    std::vector<double> dyn_gauge_M_host_, dyn_gauge_Mp_host_, dyn_gauge_fro_host_;
    int dyn_gauge_k_ = 0;
    bool has_dyn_xref_ = false, has_dyn_phi_ = false, has_dyn_gauge_ = false;
    bool dyn_gauge_metric_ok_ = false;
    double *d_Pplus_ = nullptr; double *d_KH_ = nullptr;
    torch::Tensor A_rp_t_, A_ci_t_, A_val_, b_;
    torch::Tensor X_, rv_, Ap_;                             // x_k / 几何 / ∂p∂oe
    torch::Tensor Lambda0_, Lp_, z_prior_;                  // Λ0⁻ / 白化 / 先验中心（节点0）
    torch::Tensor Lp_all_, z_prior_all_;                    // 每节点 6×6 先验（set_prior_multi）
    bool multi_prior_ = false;                              // true：按节点使用 Lp_all_/z_prior_all_
    std::vector<double> hX_, hRv_, hAp_;
    // D7 cost 参考点 (L*n*6)：solve_batch 入口的装配/线性化点。
    // total_cost 的 gauge 路径用它作投影基点：xp=ref+P(x-ref)；线搜索内不更新。
    std::vector<double> cost_ref_;
    // FLS 求解/边缘化
    std::unique_ptr<SymbolicDecomposition> sym_;            // CUDA（GN 用）
    std::unique_ptr<SymbolicDecomposition> symc_;           // CPU（全局 selected inverse 用）
    // 星主序联合因子：每星 6L 变量一个块（块图 = ISL 星图，dyn 链在块内）。
    // 列置换 old(frame-major) ↔ new(star-major) 只用于非 gauge 的联合求解。
    std::unique_ptr<SymbolicDecomposition> symStar_;
    torch::Tensor A_ci_star_t_, col_perm_t_, inv_perm_t_;
    std::unique_ptr<NumericDecomposition> nd_;              // 当前 GN 因子（gauge Woodbury 复用）
    torch::Tensor row_nz_, mtm_scl_;                        // 非零行号 / 列均衡因子
    double *d_rvGeo_ = nullptr, *d_ApGeo_ = nullptr;        // 几何 scratch（全 CUDA）
    // P1'（逐星 Schur）：逐星行表（device int64；给 flsMargDevice）
    torch::Tensor d00_p_, d00_s_, d00_c_, d11_p_, d11_s_, d11_c_, pr_s1_, pr_s0_;
    double *d_PmMarg_ = nullptr;                            // Pmarg scratch (n*36)
    // 单步耗时归因（timings()）
    double dbg_pred = 0, dbg_geo = 0, dbg_asm = 0, dbg_gn = 0, dbg_marg = 0, dbg_d2h = 0,
           dbg_rest = 0, dbg_total = 0, dbg_mtm = 0, dbg_fac = 0, dbg_solve = 0;
    std::vector<double> Pm_host_, Lp_host_, zp_host_;       // P^marg / Lp / z_prior（host 副本）
    std::vector<double> T_host_;                            // L=1 复合转移 T'（QOE；关键帧处=区间快照）
    std::vector<double> ekf_T_chain_;                       // 区间内活跃复合链（关键帧 skip 时重置为 Φ_b）
    std::vector<double> phi_host_;                          // 最近一次 step 的单步 STM Φ（host 镜像）
    std::vector<double> ekf_p_prev_, ekf_p_pred_, ekf_qeff_, ekf_p_prev_key_;
    std::vector<double> Pplus_host_, KH_host_;
    double ekf_qeff_trace_margin_ = 0.0;
    double ekf_qeff_lmin_ = 0.0;
    bool skip_update_ = false;
    bool has_gauge_ = false;
    bool has_x_ = false;                    // set_X 后为 true；gauge_lamwin_host 的输入 guard
    int gauge_k_ = 0;
    // 物理坐标约定（详见 solve_batch 顶部注释块）：
    //   sc = 列均衡因子（diag(S_c)），A_s = A_phys S_c，实际求解算子
    //   K = A_phys^T A_phys + eps S_c^-2 = S_c^-1 (A_s^T A_s + eps I) S_c^-1。
    // G 为物理 QOE（不 QR）；S = G^T G + eps_g I；Pi = G S^-1 G^T。
    // Hwin = (I-Pi) K (I-Pi) + G Pg_eff^-1 G^T，用 rank-2k Woodbury 求解。
    std::vector<double> Qg_host_, Pg_host_, Pg_inv_host_, zg_host_; // G(ncol,k), Pg(k,k), Pg^-1(k,k), z(k)
    std::vector<double> Pg_info_host_;        // GTS 的 G 分量信息（k,k，assemble 时重算）
    std::vector<double> Pg_eff_host_;         // (Pg^-1 + Pg_info)^-1（k,k）
    std::vector<double> Pg_eff_inv_host_;     // Pg^-1 + Pg_info（k,k）
    std::vector<double> gauge_xg_host_;       // 末次 K^-1 G (ncol,k)
    std::vector<double> gauge_hg_host_;       // 末次 K G (ncol,k)
    // 步骤 5：GTS 的 G 分量是否并入 Pg（本地行始终保留原始 v，窗口投影 P 负责取 J_perp）。
    bool gauge_gts_split_ = true;
    bool prior_gauge_projected_ = false;      // 步骤 4 标志（外部投影或本类低秩修正）
    bool prior_proj_lowrank_ = false;         // true：用 A0/G0 在 solve/marginalize 中做精确 rank-2k 修正
    bool prior_proj_external_ = false;        // set_prior_projected 已投影 Lp；禁止再叠加低秩投影
    std::vector<double> prior_proj_G0_host_;  // G0 (n*6,k)
    std::vector<double> prior_proj_A0_host_;  // A0=Λ0 G0 (n*6,k)
    std::vector<double> prior_proj_B0_host_;  // B0=G0^T A0 (k,k)
    std::vector<double> prior_proj_S0_host_;  // S0=G0^T G0+eps I (k,k)
    std::vector<double> prior_proj_S0inv_host_; // S0^-1 (k,k)
    // 步骤4 B0 条件 guard 诊断（project_prior_gauge / householder 记录）
    double prior_proj_B0_cond_ = -1.0;        // <0：尚未计算
    double prior_proj_B0_min_eig_ = 0.0;
    double prior_proj_B0_max_eig_ = 0.0;
    std::string prior_proj_guard_reason_;     // 空：guard 通过；否则跳过原因
    std::vector<double> ekf_jnorm_host_;      // L=1 每星 ||J6_stable||_2（生产 T 用 6×6 块）
    std::vector<double> ekf_jmin_host_;       // L=1 每星 T6_stable 最小奇异值
    // L=1 r=1 快路径缓存：一次装配的局部后验块，供交接后处理直接复用。
    std::vector<Eigen::Matrix<double, 6, 6>> ekf_Pplus6_cache_;  // 物理 Pplus 目标块
    std::vector<Eigen::Matrix<double, 6, 6>> ekf_J6_cache_;      // 物理 J6_stable
    std::vector<Eigen::Matrix<double, 6, 6>> ekf_qphys_cache_;   // 物理过程 Q（常量，只算一次）
    std::vector<double> ekf_dc_buf_;                             // scaled 局部状态增量
    bool ekf_cache_valid_ = false;
    int ekf_cache_step_ = -1;
    bool ekf_qphys_valid_ = false;
    double ekf_prof_dc_ = 0.0;
    // 固定 ISL 图的邻接/球缓存（L=1 r=1 每帧复用，避免重建 nbr/ball）。
    std::vector<int> ekf_adj_ptr_, ekf_adj_star_, ekf_adj_eid_;
    std::vector<std::vector<int>> ekf_gts_by_star_;
    std::vector<std::vector<int>> ekf_ball_cache_;
    int ekf_ball_m_max_ = 0;
    int ekf_ball_r_ = -1;
    bool ekf_store_block_T_ = false;          // 诊断：保存每星球的完整 block-T=Φ_B J_B
    std::vector<std::vector<int>> ekf_ball_host_;   // 每星球的节点索引（目标星在首位）
    std::vector<std::vector<double>> ekf_TB_host_;  // 每星球的 T_B (m×m，行主序)
    // GTS→Pg 节点范围（半开 [first,last)，当前窗口索引；首窗 [0,L)，滑窗 [L-1,L)）
    int gts_pg_node_first_ = 0;
    int gts_pg_node_last_ = -1;
    // 末次 Woodbury 的 Utot/Dinv（gauge_cov_diagnostics 用；Eigen 列主序）
    int gauge_ku_host_ = 0;
    std::vector<double> gauge_utot_host_, gauge_dinv_host_;
    std::vector<double> gauge_pgeff_build_host_;   // 末次 build 实际使用的 Pg_eff(k,k)
    bool gauge_woodbury_host_valid_ = false;
    uint64_t gauge_dinv_hash_host_ = 0;            // D^-1 内容指纹（D4 同一 D 校验）
    double gauge_dinv_fro_host_ = 0.0;             // ||D^-1||_F
    // 末次 D 的秩/条件诊断（块对角正交路径与整块 SVD 伪逆共用）
    bool gauge_d_diag_valid_host_ = false;
    int gauge_d_rank_host_ = 0;
    double gauge_d_cond_host_ = std::numeric_limits<double>::infinity();
    double gauge_d_cond_bal_host_ = std::numeric_limits<double>::infinity();
    int gauge_d11_rank_host_ = 0;
    double gauge_d11_cond_host_ = std::numeric_limits<double>::infinity();
    double gauge_d11_cond_bal_host_ = std::numeric_limits<double>::infinity();
    int gauge_d22_rank_host_ = 0;
    double gauge_d22_cond_host_ = std::numeric_limits<double>::infinity();
    double gauge_d22_cond_bal_host_ = std::numeric_limits<double>::infinity();
    int gauge_d_svd_trunc_host_ = 0;
    bool gauge_d_blockdiag_host_ = false;
    bool gauge_d_orthonormal_host_ = false;
    double gauge_d_orth_dev_host_ = 0.0;
    // D3/FEJ：窗口首次入窗状态与冻结快照。set_prior_multi 建立 x_first；
    // solve_batch 记录首次求解起点；marginalize_* 只做检查，不用最终 X 重建先验/G。
    bool fej_first_x_valid_ = false;
    bool fej_x_at_solve_valid_ = false;
    bool fej_g_frozen_ = false;
    std::vector<double> fej_first_x_;              // (L,n,6)，set_prior_multi 时刻
    std::vector<double> fej_x_at_solve_;           // (L,n,6)，首次 solve_batch 起点
    std::vector<double> fej_prior_z0_;             // 节点0 先验中心 (n,6)
    std::vector<double> fej_prior_Lp0_;            // 节点0 先验白化因子 (n,36)
    std::vector<double> fej_prior_z_all_;          // 全节点先验中心 (L,n,6)
    std::vector<double> fej_prior_Lp_all_;         // 全节点先验白化因子 (L,n,36)
    // set_prior_multi 的宿主副本：total_cost 的 all-node prior 项与 FEJ/
    // 诊断共用同一份，避免每次线搜索都 D2H 拷贝。
    std::vector<double> Lp_all_host_;              // (L,n,36)
    std::vector<double> z_prior_all_host_;         // (L,n,6)
    // 局部联合边界先验的跨变量 6×6 块（scaled 信息，off-diagonal）。
    // 块 (bi,bj) 表示变量 bi、bj 的 6N 分量之间的信息；对角线仍走 Lp_all。
    std::vector<int> cross_bi_, cross_bj_;
    std::vector<double> cross_val_;                // nblk*36，scaled Λ_ij
    std::vector<int64_t> cross_rp_, cross_ci_;
    std::vector<int> cross_entry_blk_, cross_entry_ab_, cross_entry_row_;
    std::vector<double> cross_vals_;               // 保留（未用）
    torch::Tensor cross_rp_t_, cross_ci_t_, cross_vals_t_;
    bool cross_active_ = false;
    std::vector<double> fej_cross_val_;            // 冻结快照
    uint64_t fej_cross_hash_ = 0;
    uint64_t fej_G_hash_frozen_ = 0;
    uint64_t fej_G_hash_at_solve_ = 0;
    double fej_prior_center_dev_ = 0.0;            // max|z0 - x_first(node0)|
    double fej_prior_Lp_dev_ = 0.0;                // max|Lp0(now) - Lp0(frozen)|
    double fej_solve_start_dev_ = 0.0;             // max|x_solve_start - x_first|
    double fej_x_drift_ = 0.0;                     // max|x_now - x_first|（允许非零）
    bool fej_last_marg_check_ok_ = false;
    std::string fej_last_marg_check_reason_;
    torch::Tensor rp_eps_t_, ci_eps_t_;
    int ekf_cov_r_ = 0;                                  // >0：L=1 每帧用 r 邻域局部 Schur 协方差
    int ekf_step_ = 0;
    double res2_ = 0.0;
    // 末次 solve_batch 的逐轮诊断（last_solve_diag 读取；不参与 solve）。
    std::vector<SolveIterDiag> last_solve_iters_;
    bool last_solve_valid_ = false;
    // D7 cost 分解（最近一次 total_cost；cost_gauge_diag 读取）
    bool cost_last_projected_ = false;
    bool cost_last_pg_eff_used_ = false;
    bool cost_last_pg_inv_used_ = false;
    double cost_last_total_ = 0.0;
    double cost_last_geom_ = 0.0;
    double cost_last_gauge_ = 0.0;
    double cost_last_prior_proj_ = 0.0;
    bool cost_last_prior_proj_applied_ = false;
    double cost_last_px_dev_ = 0.0;
    double cost_last_cost_ref_norm_ = 0.0;
    double cost_last_gxz_ = 0.0;
    double cost_last_pxgxz_ = 0.0;
    // 逐帧消去（sweep）：单帧结构/符号 + 缓冲 + 逐星相对行表 + 修正图案
    FlsFrameStruct F_;
    std::unique_ptr<SymbolicDecomposition> symF_;
    torch::Tensor ArpF_t_, AciF_t_, rnzF_t_, rpC_t_, ciC_t_;
    torch::Tensor Af_t_, bf_t_, eta_t_, yF_t_, dlt_t_, gam_t_, tmp_t_, weak_Lp_, zw_t_;
    torch::Tensor fis_p_, fis_s_, fis_c_, fgt_p_, fgt_s_, fgt_c_;
    bool sweep_ = false;
    bool global_marg_ = true;
};

// 生产口径的窗口 gauge 基（逐节点 r/v/J；first_node=False）在 C++ 内构造：
// 每星 J_QOE←RV 的 6×6 逆、六列物理 QOE 基、整窗 QR 都由 C++ 完成，Python
// 不再做 1152×L 个 6×6 逆或 np.linalg.qr。first_node=True 与额外模仍由
// Python 诊断路径处理，本入口只接生产 first_node=False、extra_modes=None。
static py::tuple fls_gauge_basis_window_cpp(
        py::array_t<double, py::array::c_style | py::array::forcecast> X,
        int n, int L, bool orth) {
    auto r = X.request();
    if (r.ndim != 3 || r.shape[0] != L || r.shape[1] != n || r.shape[2] != 6)
        throw std::runtime_error("fls_gauge_basis_window: X 需 (L,n,6)");
    const double *px = static_cast<const double *>(r.ptr);
    std::vector<Vector6d> oes((size_t)L * n);
    for (int l = 0; l < L; ++l)
        for (int i = 0; i < n; ++i)
            for (int c = 0; c < 6; ++c)
                oes[(size_t)l * n + i](c) = px[((size_t)l * n + i) * 6 + c];
    std::vector<double> RV((size_t)L * n * 6), J6((size_t)L * n * 36);
    noeOsc2rvBatch(oes, 16, RV);
    noeOsc2rvJacBatch(oes, 16, J6);
    const int rows = L * n * 6;
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(rows, 6);
    for (int l = 0; l < L; ++l) {
        for (int i = 0; i < n; ++i) {
            const size_t o = (size_t)l * n + i;
            Eigen::Matrix<double, 6, 6> J;
            for (int a = 0; a < 6; ++a)
                for (int c = 0; c < 6; ++c) J(a, c) = J6[o * 36 + a * 6 + c];
            Eigen::Matrix<double, 6, 6> Ji = J.inverse();
            Eigen::Matrix<double, 6, 1> rv;
            for (int a = 0; a < 6; ++a) rv(a) = RV[o * 6 + a];
            const int64_t base = (int64_t)(l * n + i) * 6;
            for (int a = 0; a < 3; ++a) {
                Eigen::Matrix<double, 6, 1> drv = Eigen::Matrix<double, 6, 1>::Zero();
                drv(a) = 1.0;
                B.block<6, 1>(base, a) = Ji * drv;
            }
            for (int a = 0; a < 3; ++a) {
                Eigen::Vector3d e = Eigen::Vector3d::Zero(); e(a) = 1.0;
                Eigen::Matrix<double, 6, 1> drv;
                drv.head<3>() = e.cross(rv.head<3>());
                drv.tail<3>() = e.cross(rv.tail<3>());
                B.block<6, 1>(base, 3 + a) = Ji * drv;
            }
        }
    }
    Eigen::MatrixXd Q = B, R = Eigen::MatrixXd::Identity(6, 6);
    if (orth) {
        Eigen::HouseholderQR<Eigen::MatrixXd> qr(B);
        Q = qr.householderQ() * Eigen::MatrixXd::Identity(rows, 6);
        R = qr.matrixQR().topRows(6).triangularView<Eigen::Upper>();
    }
    return py::make_tuple(eigen_to_np_rowmajor(Q), eigen_to_np_rowmajor(R));
}

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


template <int N>
static double nees_quadform(const Eigen::Matrix<double, N, N> &P,
                            const Eigen::Matrix<double, N, 1> &e) {
    Eigen::Matrix<double, N, N> Ps = 0.5 * (P + P.transpose());
    double dmax = 0.0, dmin = std::numeric_limits<double>::infinity();
    for (int i = 0; i < N; ++i) {
        if (!std::isfinite(Ps(i, i))) continue;
        dmax = std::max(dmax, std::abs(Ps(i, i)));
        if (Ps(i, i) > 0.0) dmin = std::min(dmin, Ps(i, i));
    }
    if (!std::isfinite(dmin)) dmin = std::max(dmax, 1.0);
    const double jit = std::max(1e-30, 1e-12 * dmin);
    Eigen::Matrix<double, N, N> Pj = Ps;
    for (int i = 0; i < N; ++i) Pj(i, i) += jit;
    Eigen::LLT<Eigen::Matrix<double, N, N>> llt(Pj);
    if (llt.info() == Eigen::Success) {
        const Eigen::Matrix<double, N, 1> x = llt.solve(e);
        const double q = e.dot(x);
        if (std::isfinite(q)) return q;
    }
    double floor = std::max(1e-30, 1e-12 * dmax);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, N, N>> es(Ps);
    if (es.info() == Eigen::Success) {
        const auto &w = es.eigenvalues();
        const auto &V = es.eigenvectors();
        if (w.size() == N && w.allFinite()) {
            const double wmax = std::max(0.0, w.maxCoeff());
            floor = std::max(1e-30, 1e-12 * wmax);
            double q = 0.0;
            for (int i = 0; i < N; ++i) {
                const double p = V.col(i).dot(e);
                q += p * p / std::max(w(i), floor);
            }
            if (std::isfinite(q)) return q;
        }
    }
    return e.squaredNorm() / floor;
}

static py::dict fls_nees_directional(
        py::array_t<double, py::array::c_style | py::array::forcecast> e_rv,
        py::array_t<double, py::array::c_style | py::array::forcecast> Pq,
        py::array_t<double, py::array::c_style | py::array::forcecast> Xq,
        int nthreads = 16) {
    auto er = e_rv.request();
    auto pq = Pq.request();
    auto xq = Xq.request();
    if (er.ndim != 3 || er.shape[2] != 6)
        throw std::runtime_error("fls_nees_directional: e_rv 需 (m,n,6)");
    const py::ssize_t m = er.shape[0], n = er.shape[1];
    if (pq.ndim != 4 || pq.shape[0] != m || pq.shape[1] != n ||
        pq.shape[2] != 6 || pq.shape[3] != 6)
        throw std::runtime_error("fls_nees_directional: Pq 需 (m,n,6,6)");
    if (xq.ndim != 3 || xq.shape[0] != m || xq.shape[1] != n || xq.shape[2] != 6)
        throw std::runtime_error("fls_nees_directional: Xq 需 (m,n,6)");
    const size_t total = (size_t)m * (size_t)n;
    const double *per = static_cast<const double *>(er.ptr);
    const double *ppq = static_cast<const double *>(pq.ptr);
    const double *pxq = static_cast<const double *>(xq.ptr);
    std::vector<Vector6d> oes(total);
    for (size_t o = 0; o < total; ++o)
        for (int c = 0; c < 6; ++c) oes[o](c) = pxq[o * 6 + c];
    std::vector<double> RV(total * 6), J6(total * 36);
    noeOsc2rvBatch(oes, nthreads, RV);
    noeOsc2rvJacBatch(oes, nthreads, J6);

    py::array_t<double> o_nees({m, n}), o_nees_p({m, n}), o_nees_v({m, n}), o_nees_q({m, n});
    py::array_t<double> o_err_rtn({m, n, (py::ssize_t)6}), o_sig_rtn({m, n, (py::ssize_t)6}),
            o_z_rtn({m, n, (py::ssize_t)6});
    // QOE→RV→RTN 的完整协方差与 RV 位置/速度，供协方差诊断复用同一条 C++ 路径。
    py::array_t<double> o_P_rtn({m, n, (py::ssize_t)6, (py::ssize_t)6});
    py::array_t<double> o_rv_out({m, n, (py::ssize_t)6});
    py::array_t<double> o_err_q({m, n, (py::ssize_t)6}), o_sig_q({m, n, (py::ssize_t)6}),
            o_z_q({m, n, (py::ssize_t)6});
    double *pn = o_nees.mutable_data(), *pnp = o_nees_p.mutable_data();
    double *pnv = o_nees_v.mutable_data(), *pnq = o_nees_q.mutable_data();
    double *perr = o_err_rtn.mutable_data(), *psr = o_sig_rtn.mutable_data();
    double *pzr = o_z_rtn.mutable_data(), *peq = o_err_q.mutable_data();
    double *psq = o_sig_q.mutable_data(), *pzq = o_z_q.mutable_data();
    double *pprtn = o_P_rtn.mutable_data(), *prv_out = o_rv_out.mutable_data();

    const int nt = std::max(1, nthreads);
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(nt)
#endif
    for (long long oo = 0; oo < (long long)total; ++oo) {
        const size_t o = (size_t)oo;
        Eigen::Matrix<double, 6, 6> J, Pq_m, P_rv, P_rtn, B;
        Eigen::Matrix<double, 6, 1> evec, e_rtn, e_q;
        for (int a = 0; a < 6; ++a) {
            evec(a) = per[o * 6 + a];
            for (int c = 0; c < 6; ++c) {
                J(a, c) = J6[o * 36 + a * 6 + c];
                Pq_m(a, c) = ppq[o * 36 + a * 6 + c];
            }
        }
        P_rv = J * Pq_m * J.transpose();
        P_rv = 0.5 * (P_rv + P_rv.transpose());

        Eigen::Vector3d r, v;
        for (int a = 0; a < 3; ++a) {
            r(a) = RV[o * 6 + a];
            v(a) = RV[o * 6 + 3 + a];
        }
        Eigen::Matrix3d A;
        const double nr = r.norm(), nv = v.norm();
        const Eigen::Vector3d h = r.cross(v);
        const double nh = h.norm();
        if (std::isfinite(nr) && std::isfinite(nv) && std::isfinite(nh) &&
            nr > 1e-12 && nv > 1e-12 && nh > 1e-12 * nr * nv) {
            const Eigen::Vector3d er_ax = r / nr;
            const Eigen::Vector3d en_ax = h / nh;
            const Eigen::Vector3d et_ax = en_ax.cross(er_ax);
            A.col(0) = er_ax;
            A.col(1) = et_ax;
            A.col(2) = en_ax;
        } else {
            A.setIdentity();
        }
        B.setZero();
        B.topLeftCorner<3, 3>() = A;
        B.bottomRightCorner<3, 3>() = A;

        e_rtn = B.transpose() * evec;
        P_rtn = B.transpose() * P_rv * B;
        P_rtn = 0.5 * (P_rtn + P_rtn.transpose());
        double dmax = 0.0;
        for (int a = 0; a < 6; ++a)
            if (std::isfinite(P_rtn(a, a))) dmax = std::max(dmax, P_rtn(a, a));
        const double floor_rtn = std::max(1e-30, 1e-12 * std::max(0.0, dmax));
        Eigen::Matrix<double, 6, 1> sig_rtn, z_rtn;
        for (int a = 0; a < 6; ++a) {
            const double d = std::isfinite(P_rtn(a, a)) ? P_rtn(a, a) : floor_rtn;
            sig_rtn(a) = std::sqrt(std::max(d, floor_rtn));
            z_rtn(a) = e_rtn(a) / sig_rtn(a);
        }
        const double nees = nees_quadform<6>(P_rtn, e_rtn);
        const double nees_p = nees_quadform<3>(P_rtn.topLeftCorner<3, 3>(), e_rtn.head<3>());
        const double nees_v = nees_quadform<3>(P_rtn.bottomRightCorner<3, 3>(), e_rtn.tail<3>());

        e_q = J.fullPivLu().solve(evec);
        if (!e_q.allFinite()) e_q.setZero();
        Eigen::Matrix<double, 6, 6> Pq_s = 0.5 * (Pq_m + Pq_m.transpose());
        double dmax_q = 0.0;
        for (int a = 0; a < 6; ++a)
            if (std::isfinite(Pq_s(a, a))) dmax_q = std::max(dmax_q, Pq_s(a, a));
        const double floor_q = std::max(1e-30, 1e-12 * std::max(0.0, dmax_q));
        Eigen::Matrix<double, 6, 1> sig_q, z_q;
        for (int a = 0; a < 6; ++a) {
            const double d = std::isfinite(Pq_s(a, a)) ? Pq_s(a, a) : floor_q;
            sig_q(a) = std::sqrt(std::max(d, floor_q));
            z_q(a) = e_q(a) / sig_q(a);
        }
        const double nees_q = nees_quadform<6>(Pq_s, e_q);

        pn[o] = nees;
        pnp[o] = nees_p;
        pnv[o] = nees_v;
        pnq[o] = nees_q;
        for (int a = 0; a < 6; ++a) {
            perr[o * 6 + a] = e_rtn(a);
            psr[o * 6 + a] = sig_rtn(a);
            pzr[o * 6 + a] = z_rtn(a);
            peq[o * 6 + a] = e_q(a);
            psq[o * 6 + a] = sig_q(a);
            pzq[o * 6 + a] = z_q(a);
            prv_out[o * 6 + a] = RV[o * 6 + a];
            for (int b = 0; b < 6; ++b)
                pprtn[(o * 6 + a) * 6 + b] = P_rtn(a, b);
        }
    }
    py::dict out;
    out["nees"] = o_nees;
    out["nees_pos"] = o_nees_p;
    out["nees_vel"] = o_nees_v;
    out["nees_q"] = o_nees_q;
    out["err_rtn"] = o_err_rtn;
    out["sig_rtn"] = o_sig_rtn;
    out["z_rtn"] = o_z_rtn;
    out["err_q"] = o_err_q;
    out["sig_q"] = o_sig_q;
    out["z_q"] = o_z_q;
    out["P_rtn"] = o_P_rtn;
    out["rv"] = o_rv_out;
    return out;
}

static py::array_t<double> fls_local_schur_py(
        py::array_t<double, py::array::c_style | py::array::forcecast> H00,
        py::array_t<double, py::array::c_style | py::array::forcecast> H01,
        py::array_t<double, py::array::c_style | py::array::forcecast> H11) {
    auto h = H00.request(); auto o = H01.request(); auto e = H11.request();
    if (h.ndim != 3 || o.ndim != 3 || e.ndim != 3)
        throw std::runtime_error("fls_local_schur: 需 3D 数组");
    const int n = (int)h.shape[0], m = (int)h.shape[1];
    if (m < 6 || h.shape[2] != m || o.shape[0] != n || o.shape[1] != m || o.shape[2] != m ||
        e.shape[0] != n || e.shape[1] != m || e.shape[2] != m)
        throw std::runtime_error("fls_local_schur: H00/H01/H11 需同形 (n,m,m)，m>=6");
    py::array_t<double> out({(py::ssize_t)n, (py::ssize_t)6, (py::ssize_t)6});
    qoejopt::flsLocalSchurBatch((const double *)h.ptr, (const double *)o.ptr,
                                (const double *)e.ptr, n, m, (double *)out.mutable_data());
    return out;
}

static py::array_t<double> boundary_info_csr(
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> H_rp,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> H_ci,
        py::array_t<double, py::array::c_style | py::array::forcecast> H_val,
        int64_t ncol, int64_t off, int64_t q) {
    auto r = H_rp.request(); auto c = H_ci.request(); auto v = H_val.request();
    if (r.ndim != 1 || c.ndim != 1 || v.ndim != 1)
        throw std::runtime_error("boundary_info_csr: H_rp/H_ci/H_val 需 1D");
    if (r.size != ncol + 1) throw std::runtime_error("boundary_info_csr: H_rp size != ncol+1");
    if (c.size != v.size) throw std::runtime_error("boundary_info_csr: H_ci/H_val size 不一致");
    if (q <= 0 || off < 0 || off + q > ncol)
        throw std::runtime_error("boundary_info_csr: off/q 越界");
    if (!torch::cuda::is_available())
        throw std::runtime_error("boundary_info_csr: CUDA 不可用");
    auto optsD = torch::TensorOptions().dtype(torch::kFloat64);
    auto optsI64 = torch::TensorOptions().dtype(torch::kInt64);
    auto dev = torch::kCUDA;
    auto rp = torch::from_blob((void *)r.ptr, {ncol + 1}, optsI64)
                  .clone().to(torch::kInt32).to(dev);
    auto ci = torch::from_blob((void *)c.ptr, {(int64_t)c.size}, optsI64)
                  .clone().to(torch::kInt32).to(dev);
    auto va = torch::from_blob((void *)v.ptr, {(int64_t)v.size}, optsD).clone().to(dev);
    CusolverLUSolver lu(1, ncol, rp, ci, AMD);
    auto sing = lu.factor(va.reshape({1, va.numel()}));
    if (!sing.empty() && sing[0] >= 0)
        throw std::runtime_error("boundary_info_csr: LU 奇异 at " + std::to_string(sing[0]));
    auto B = torch::zeros({1, q * ncol}, optsD);
    {
        auto ba = B.accessor<double, 2>();
        for (int64_t k = 0; k < q; ++k) ba[0][k * ncol + off + k] = 1.0;
    }
    B = B.to(dev);
    lu.solve_multi(B, q);
    auto X = B[0].view({q, ncol}).narrow(1, off, q);
    auto Sig = 0.5 * (X + X.transpose(0, 1));
    auto Lam = torch::linalg_inv(Sig);
    Lam = 0.5 * (Lam + Lam.transpose(0, 1));
    auto Lc = Lam.to(torch::kCPU).contiguous();
    py::array_t<double> out({(py::ssize_t)q, (py::ssize_t)q});
    std::memcpy(out.mutable_data(), Lc.data_ptr<double>(),
                sizeof(double) * (size_t)q * (size_t)q);
    return out;
}

static py::tuple woodbury_solve_baspacho(
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> H_rp,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> H_ci,
        py::array_t<double, py::array::c_style | py::array::forcecast> H_val,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> ps,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> bp,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> bi,
        py::array_t<double, py::array::c_style | py::array::forcecast> U,
        py::array_t<double, py::array::c_style | py::array::forcecast> C,
        py::array_t<double, py::array::c_style | py::array::forcecast> g,
        int64_t off, int64_t q) {
    auto r = H_rp.request(); auto c = H_ci.request(); auto v = H_val.request();
    auto rp = ps.request(); auto rb = bp.request(); auto ri = bi.request();
    auto ru = U.request(); auto rc = C.request(); auto rg = g.request();
    const int64_t n = (int64_t)rg.shape[0];
    const int64_t nr = (int64_t)ru.shape[1];
    if (rg.ndim != 1 || ru.ndim != 2 || rc.ndim != 2 ||
        ru.shape[0] != q || rc.shape[0] != nr || rc.shape[1] != nr ||
        off < 0 || off + q > n)
        throw std::runtime_error("woodbury_solve_baspacho: 参数维数不符");
    if (r.size != n + 1 || c.size != v.size)
        throw std::runtime_error("woodbury_solve_baspacho: H CSR 形状不符");
    auto optsD = torch::TensorOptions().dtype(torch::kFloat64);
    auto optsI = torch::TensorOptions().dtype(torch::kInt64);
    auto dev = torch::kCUDA;
    auto ps_t = torch::from_blob((void *)rp.ptr, {(int64_t)rp.size}, optsI).clone();
    auto bp_t = torch::from_blob((void *)rb.ptr, {(int64_t)rb.size}, optsI).clone();
    auto bi_t = torch::from_blob((void *)ri.ptr, {(int64_t)ri.size}, optsI).clone();
    SymbolicDecomposition sym(ps_t, bp_t, bi_t, "cuda");
    auto nd = sym.createNumericDecomposition(1);
    auto hrp = torch::from_blob((void *)r.ptr, {n + 1}, optsI).clone().to(dev);
    auto hci = torch::from_blob((void *)c.ptr, {(int64_t)c.size}, optsI).clone().to(dev);
    auto hv = torch::from_blob((void *)v.ptr, {(int64_t)v.size}, optsD).clone().to(dev);
    nd.add_M(hv.reshape({1, hv.numel()}), hrp, hci);
    nd.factor();
    auto B = torch::zeros({1, (nr + 1) * n}, optsD);
    {
        auto ba = B.accessor<double, 2>();
        const double *gp = (const double *)rg.ptr;
        const double *up = (const double *)ru.ptr;
        for (int64_t i = 0; i < n; ++i) ba[0][i] = gp[i];
        for (int64_t j = 0; j < nr; ++j)
            for (int64_t i = 0; i < q; ++i)
                ba[0][(j + 1) * n + off + i] = up[i * nr + j];
    }
    B = B.to(dev);
    nd.solve_multi(B, nr + 1);                 // BaSpaCho CUDA parallel multi-RHS
    auto X = B[0].view({nr + 1, n}).to(torch::kCPU).contiguous();
    auto x0 = X[0].to(dev);
    auto Y = X.narrow(0, 1, nr).transpose(0, 1).contiguous().to(dev);
    auto Ut = torch::from_blob((void *)ru.ptr, {q, nr}, optsD).clone().to(dev);
    auto Ct = torch::from_blob((void *)rc.ptr, {nr, nr}, optsD).clone().to(dev);
    auto S = torch::linalg_inv(Ct) + Ut.transpose(0, 1).matmul(Y.narrow(0, off, q));
    S = 0.5 * (S + S.transpose(0, 1));
    auto jit = 1e-10 * torch::diagonal(S).abs().max().clamp_min(1e-30);
    auto Sinv = torch::linalg_inv(S + jit * torch::eye(nr, optsD.device(dev)));
    auto rhs = Ut.transpose(0, 1).matmul(x0.narrow(0, off, q));
    auto delta = x0 - Y.matmul(Sinv.matmul(rhs));
    auto dc = delta.to(torch::kCPU).contiguous();
    auto Yc = Y.to(torch::kCPU).contiguous();
    auto Sc = Sinv.to(torch::kCPU).contiguous();
    py::array_t<double> od({(py::ssize_t)n});
    py::array_t<double> oy({(py::ssize_t)n, (py::ssize_t)nr});
    py::array_t<double> os({(py::ssize_t)nr, (py::ssize_t)nr});
    std::memcpy(od.mutable_data(), dc.data_ptr<double>(), sizeof(double) * (size_t)n);
    std::memcpy(oy.mutable_data(), Yc.data_ptr<double>(), sizeof(double) * (size_t)n * (size_t)nr);
    std::memcpy(os.mutable_data(), Sc.data_ptr<double>(), sizeof(double) * (size_t)nr * (size_t)nr);
    return py::make_tuple(od, oy, os);
}

static py::tuple woodbury_solve_csr(
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> H_rp,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> H_ci,
        py::array_t<double, py::array::c_style | py::array::forcecast> H_val,
        py::array_t<double, py::array::c_style | py::array::forcecast> U,
        py::array_t<double, py::array::c_style | py::array::forcecast> C,
        py::array_t<double, py::array::c_style | py::array::forcecast> g,
        int64_t off, int64_t q) {
    auto r = H_rp.request(); auto c = H_ci.request(); auto v = H_val.request();
    auto ru = U.request(); auto rc = C.request(); auto rg = g.request();
    if (rg.ndim != 1 || ru.ndim != 2 || rc.ndim != 2)
        throw std::runtime_error("woodbury_solve_csr: U/C/g 维数不符");
    const int64_t n = (int64_t)rg.shape[0];
    const int64_t nr = (int64_t)ru.shape[1];
    if (ru.shape[0] != q || rc.shape[0] != nr || rc.shape[1] != nr || off + q > n)
        throw std::runtime_error("woodbury_solve_csr: U/C/off/q 越界");
    if (r.size != n + 1 || c.size != v.size)
        throw std::runtime_error("woodbury_solve_csr: H CSR 形状不符");
    auto optsD = torch::TensorOptions().dtype(torch::kFloat64);
    auto optsI64 = torch::TensorOptions().dtype(torch::kInt64);
    auto dev = torch::kCUDA;
    auto rp = torch::from_blob((void *)r.ptr, {n + 1}, optsI64)
                  .clone().to(torch::kInt32).to(dev);
    auto ci = torch::from_blob((void *)c.ptr, {(int64_t)c.size}, optsI64)
                  .clone().to(torch::kInt32).to(dev);
    auto hv = torch::from_blob((void *)v.ptr, {(int64_t)v.size}, optsD).clone().to(dev);
    CusolverLUSolver lu(1, n, rp, ci, AMD);
    auto sing = lu.factor(hv.reshape({1, hv.numel()}));
    if (!sing.empty() && sing[0] >= 0)
        throw std::runtime_error("woodbury_solve_csr: LU 奇异 at " + std::to_string(sing[0]));
    // [g, U] 一次 C++/CUDA 多右端
    auto B = torch::zeros({1, (nr + 1) * n}, optsD);
    {
        auto ba = B.accessor<double, 2>();
        const double *gp = (const double *)rg.ptr;
        const double *up = (const double *)ru.ptr;
        for (int64_t i = 0; i < n; ++i) ba[0][i] = gp[i];
        for (int64_t j = 0; j < nr; ++j)
            for (int64_t i = 0; i < q; ++i)
                ba[0][(j + 1) * n + off + i] = up[i * nr + j];
    }
    B = B.to(dev);
    lu.solve_multi(B, nr + 1);
    auto X = B[0].view({nr + 1, n}).to(torch::kCPU).contiguous();
    auto x0 = X[0].to(dev);
    auto Y = X.narrow(0, 1, nr).transpose(0, 1).contiguous().to(dev);   // (n,nr)
    auto Ut = torch::from_blob((void *)ru.ptr, {q, nr}, optsD).clone().to(dev);
    auto Ct = torch::from_blob((void *)rc.ptr, {nr, nr}, optsD).clone().to(dev);
    auto S = torch::linalg_inv(Ct) + Ut.transpose(0, 1).matmul(Y.narrow(0, off, q));
    S = 0.5 * (S + S.transpose(0, 1));
    auto jit = 1e-10 * torch::diagonal(S).abs().max().clamp_min(1e-30);
    auto Sinv = torch::linalg_inv(S + jit * torch::eye(nr, optsD.device(dev)));
    auto rhs = Ut.transpose(0, 1).matmul(x0.narrow(0, off, q));
    auto delta = x0 - Y.matmul(Sinv.matmul(rhs));
    auto dc = delta.to(torch::kCPU).contiguous();
    auto Yc = Y.to(torch::kCPU).contiguous();
    auto Sc = Sinv.to(torch::kCPU).contiguous();
    py::array_t<double> od({(py::ssize_t)n});
    py::array_t<double> oy({(py::ssize_t)n, (py::ssize_t)nr});
    py::array_t<double> os({(py::ssize_t)nr, (py::ssize_t)nr});
    std::memcpy(od.mutable_data(), dc.data_ptr<double>(), sizeof(double) * (size_t)n);
    std::memcpy(oy.mutable_data(), Yc.data_ptr<double>(), sizeof(double) * (size_t)n * (size_t)nr);
    std::memcpy(os.mutable_data(), Sc.data_ptr<double>(), sizeof(double) * (size_t)nr * (size_t)nr);
    return py::make_tuple(od, oy, os);
}

static py::array_t<double> solve_csr_csr(
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> H_rp,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> H_ci,
        py::array_t<double, py::array::c_style | py::array::forcecast> H_val,
        py::array_t<double, py::array::c_style | py::array::forcecast> B) {
    auto r = H_rp.request(); auto c = H_ci.request(); auto v = H_val.request();
    auto b = B.request();
    if (b.ndim != 2) throw std::runtime_error("solve_csr_csr: B 需 (n,r)");
    const int64_t n = (int64_t)b.shape[0];
    const int64_t nr = (int64_t)b.shape[1];
    if (r.size != n + 1 || c.size != v.size)
        throw std::runtime_error("solve_csr_csr: H CSR 形状不符");
    auto optsD = torch::TensorOptions().dtype(torch::kFloat64);
    auto optsI64 = torch::TensorOptions().dtype(torch::kInt64);
    auto dev = torch::kCUDA;
    auto rp = torch::from_blob((void *)r.ptr, {n + 1}, optsI64)
                  .clone().to(torch::kInt32).to(dev);
    auto ci = torch::from_blob((void *)c.ptr, {(int64_t)c.size}, optsI64)
                  .clone().to(torch::kInt32).to(dev);
    auto va = torch::from_blob((void *)v.ptr, {(int64_t)v.size}, optsD).clone().to(dev);
    CusolverLUSolver lu(1, n, rp, ci, AMD);
    auto sing = lu.factor(va.reshape({1, va.numel()}));
    if (!sing.empty() && sing[0] >= 0)
        throw std::runtime_error("solve_csr_csr: LU 奇异 at " + std::to_string(sing[0]));
    // B (n,r) row-major → 每个 RHS 连续的 (r,n) 缓冲，再转 CUDA
    auto Bt = torch::from_blob((void *)b.ptr, {n, nr}, optsD).clone().transpose(0, 1).contiguous();
    auto X = torch::zeros({1, nr * n}, optsD);
    std::memcpy(X.data_ptr<double>(), Bt.data_ptr<double>(),
                sizeof(double) * (size_t)n * (size_t)nr);
    X = X.to(dev);
    lu.solve_multi(X, nr);
    auto Xc = X[0].view({nr, n}).transpose(0, 1).contiguous().to(torch::kCPU).contiguous();
    py::array_t<double> out({(py::ssize_t)n, (py::ssize_t)nr});
    std::memcpy(out.mutable_data(), Xc.data_ptr<double>(),
                sizeof(double) * (size_t)n * (size_t)nr);
    return out;
}

static py::array_t<double> boundary_probe_csr(
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> H_rp,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> H_ci,
        py::array_t<double, py::array::c_style | py::array::forcecast> H_val,
        py::array_t<double, py::array::c_style | py::array::forcecast> Omega,
        int64_t ncol, int64_t off, int64_t q) {
    auto r = H_rp.request(); auto c = H_ci.request(); auto v = H_val.request();
    auto o = Omega.request();
    if (o.ndim != 2 || o.shape[0] != q)
        throw std::runtime_error("boundary_probe_csr: Omega 需 (q,r)");
    const int64_t nr = (int64_t)o.shape[1];
    if (nr <= 0 || q <= 0 || off < 0 || off + q > ncol)
        throw std::runtime_error("boundary_probe_csr: off/q/r 越界");
    if (r.size != ncol + 1 || c.size != v.size)
        throw std::runtime_error("boundary_probe_csr: H CSR 形状不符");
    auto optsD = torch::TensorOptions().dtype(torch::kFloat64);
    auto optsI64 = torch::TensorOptions().dtype(torch::kInt64);
    auto dev = torch::kCUDA;
    auto rp = torch::from_blob((void *)r.ptr, {ncol + 1}, optsI64)
                  .clone().to(torch::kInt32).to(dev);
    auto ci = torch::from_blob((void *)c.ptr, {(int64_t)c.size}, optsI64)
                  .clone().to(torch::kInt32).to(dev);
    auto va = torch::from_blob((void *)v.ptr, {(int64_t)v.size}, optsD).clone().to(dev);
    CusolverLUSolver lu(1, ncol, rp, ci, AMD);
    auto sing = lu.factor(va.reshape({1, va.numel()}));
    if (!sing.empty() && sing[0] >= 0)
        throw std::runtime_error("boundary_probe_csr: LU 奇异 at " + std::to_string(sing[0]));
    auto B = torch::zeros({1, nr * ncol}, optsD);
    const double *om = (const double *)o.ptr;
    {
        auto ba = B.accessor<double, 2>();
        for (int64_t j = 0; j < nr; ++j)
            for (int64_t i = 0; i < q; ++i)
                ba[0][j * ncol + off + i] = om[i * nr + j];
    }
    B = B.to(dev);
    lu.solve_multi(B, nr);
    auto X = B[0].view({nr, ncol}).narrow(1, off, q).transpose(0, 1).contiguous();
    auto Xc = X.to(torch::kCPU).contiguous();
    py::array_t<double> out({(py::ssize_t)q, (py::ssize_t)nr});
    std::memcpy(out.mutable_data(), Xc.data_ptr<double>(),
                sizeof(double) * (size_t)q * (size_t)nr);
    return out;
}

// P2 边界信息：一次 C++/CUDA 调用完成 Schur probe、gauge QR、低秩 C 与尺度门禁。
// 输入 H 为整窗信息 CSR，D 为边界逐星 6×6 协方差（由 Baspacho 精确输出），
// Omega 为未正交化探针；返回 (DLa, U, C, info, stats)。Python 只做窗循环。
static py::dict boundary_lowrank_csr(
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> H_rp,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> H_ci,
        py::array_t<double, py::array::c_style | py::array::forcecast> H_val,
        py::array_t<double, py::array::c_style | py::array::forcecast> D,
        py::array_t<double, py::array::c_style | py::array::forcecast> Omega,
        int64_t ncol, int64_t off, int64_t q, double cap_rel,
        int64_t rank_out) {
    auto rp_a = H_rp.request(); auto ci_a = H_ci.request(); auto val_a = H_val.request();
    auto d_a = D.request(); auto o_a = Omega.request();
    if (rp_a.size != ncol + 1 || ci_a.size != val_a.size)
        throw std::runtime_error("boundary_lowrank_csr: H CSR 形状不符");
    if (d_a.ndim != 3 || d_a.shape[1] != 6 || d_a.shape[2] != 6 ||
        d_a.shape[0] * 6 != q)
        throw std::runtime_error("boundary_lowrank_csr: D 需 (q/6,6,6)");
    if (o_a.ndim != 2 || o_a.shape[0] != q)
        throw std::runtime_error("boundary_lowrank_csr: Omega 需 (q,r)");
    if (off <= 0 || q <= 0 || off + q != ncol)
        throw std::runtime_error("boundary_lowrank_csr: 要求边界列在末尾");
    const int64_t r0 = (int64_t)o_a.shape[1];
    const int64_t r = (rank_out > 0) ? (int64_t)rank_out : r0;
    if (r < r0 || r > 32)
        throw std::runtime_error("boundary_lowrank_csr: rank_out 需在 [r0,32]");
    const int64_t N = q / 6;
    const int64_t *rp = (const int64_t *)rp_a.ptr;
    const int64_t *ci = (const int64_t *)ci_a.ptr;
    const double *hv = (const double *)val_a.ptr;
    const double *dp = (const double *)d_a.ptr;
    const double *op = (const double *)o_a.ptr;

    // Omega 正交化（小 r，Eigen QR）。numpy 的 Omega 是 C 行主序，
    // 必须用 RowMajor Map；默认列主序会把 q×r 缓冲错位成另一个子空间。
    using RowMajorXd = Eigen::Matrix<double, Eigen::Dynamic,
                                     Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<const RowMajorXd> Om0(op, q, r0);
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(Om0);
    Eigen::MatrixXd Q0 = qr.householderQ() *
        Eigen::MatrixXd::Identity(q, std::min(q, r0));
    Q0 = Q0.leftCols(r0);

    // Hss 的 CSR（行/列都 < off）。
    std::vector<int64_t> srp((size_t)off + 1, 0), sci;
    std::vector<double> sv;
    sci.reserve((size_t)ci_a.size / 4);
    sv.reserve((size_t)ci_a.size / 4);
    for (int64_t i = 0; i < off; ++i) {
        for (int64_t p = rp[i]; p < rp[i + 1]; ++p) {
            int64_t j = ci[p];
            if (j >= 0 && j < off) { sci.push_back(j); sv.push_back(hv[p]); }
        }
        srp[i + 1] = (int64_t)sci.size();
    }

    auto optsD = torch::TensorOptions().dtype(torch::kFloat64);
    auto optsI = torch::TensorOptions().dtype(torch::kInt64);
    auto dev = torch::kCUDA;
    auto srp_t = torch::from_blob(srp.data(), {off + 1}, optsI).clone()
                     .to(torch::kInt32).to(dev);
    auto sci_t = torch::from_blob(sci.data(), {(int64_t)sci.size()}, optsI).clone()
                     .to(torch::kInt32).to(dev);
    auto sv_t = torch::from_blob(sv.data(), {(int64_t)sv.size()}, optsD).clone().to(dev);
    CusolverLUSolver lu(1, off, srp_t, sci_t, AMD);
    auto sing = lu.factor(sv_t.reshape({1, sv_t.numel()}));
    if (!sing.empty() && sing[0] >= 0)
        throw std::runtime_error("boundary_lowrank_csr: Hss LU 奇异 at " +
                                 std::to_string(sing[0]));
    // 同一 Hss 分解上反复作用 Lambda_b，构造 gauge-Krylov 基：
    // Q0 = gauge, Q1 = Lambda_b Q0, Q2 = Lambda_b Q1, ...
    auto apply_lambda = [&](const Eigen::MatrixXd& Qb) -> Eigen::MatrixXd {
        const int64_t rb = Qb.cols();
        std::vector<double> W((size_t)off * rb, 0.0);
        for (int64_t i = 0; i < off; ++i)
            for (int64_t p = rp[i]; p < rp[i + 1]; ++p) {
                int64_t j = ci[p];
                if (j >= off)
                    for (int64_t b = 0; b < rb; ++b)
                        W[(size_t)i * rb + b] += hv[p] * Qb(j - off, b);
            }
        auto B = torch::zeros({1, rb * off}, optsD);
        {
            auto ba = B.accessor<double, 2>();
            for (int64_t b = 0; b < rb; ++b)
                for (int64_t i = 0; i < off; ++i)
                    ba[0][b * off + i] = W[(size_t)i * rb + b];
        }
        B = B.to(dev);
        lu.solve_multi(B, rb);
        auto Yt = B[0].view({rb, off}).transpose(0, 1).contiguous()
                      .to(torch::kCPU).contiguous();
        const double *Y = Yt.data_ptr<double>();
        Eigen::MatrixXd Z(q, rb);
        Z.setZero();
        for (int64_t i = 0; i < q; ++i)
            for (int64_t p = rp[off + i]; p < rp[off + i + 1]; ++p) {
                int64_t j = ci[p];
                if (j >= off)
                    for (int64_t b = 0; b < rb; ++b)
                        Z(i, b) += hv[p] * Qb(j - off, b);
                else
                    for (int64_t b = 0; b < rb; ++b)
                        Z(i, b) -= hv[p] * Y[(size_t)j * rb + b];
            }
        return Z;
    };

    Eigen::MatrixXd K(q, r0);
    K.leftCols(r0) = Q0;
    int64_t nblk = 1;
    Eigen::MatrixXd Qblk = Q0;
    while (K.cols() < r && nblk < 4) {
        Qblk = apply_lambda(Qblk);
        const int64_t old = K.cols();
        K.conservativeResize(q, old + Qblk.cols());
        K.rightCols(Qblk.cols()) = Qblk;
        ++nblk;
    }
    if (K.cols() < r) {
        const int64_t old = K.cols();
        K.conservativeResize(q, r);
        for (int64_t b = old; b < r; ++b)
            for (int64_t i = 0; i < q; ++i)
                K(i, b) = std::sin(1e-4 * (double)(i + 1) * (double)(b + 1));
    }
    Eigen::HouseholderQR<Eigen::MatrixXd> qk(K);
    Eigen::MatrixXd Q = qk.householderQ() *
        Eigen::MatrixXd::Identity(q, std::min(q, K.cols()));
    Q = Q.leftCols(r);
    Eigen::MatrixXd Z = apply_lambda(Q);

    // D^{-1} 与 G = Q^T D^{-1} Q
    Eigen::MatrixXd Di(N * 6, 6);
    for (int64_t s = 0; s < N; ++s) {
        Eigen::Matrix<double, 6, 6, Eigen::RowMajor> d6;
        std::memcpy(d6.data(), dp + (size_t)s * 36, sizeof(double) * 36);
        Di.block<6, 6>(s * 6, 0) = Eigen::MatrixXd(d6).inverse();
    }
    Eigen::MatrixXd G = Eigen::MatrixXd::Zero(r, r);
    for (int64_t s = 0; s < N; ++s) {
        Eigen::MatrixXd Qs = Q.block(s * 6, 0, 6, r);
        Eigen::MatrixXd Dis = Di.block<6, 6>(s * 6, 0);
        G.noalias() += Qs.transpose() * Dis * Qs;
    }
    Eigen::MatrixXd Ys = Q.transpose() * Z;              // Omega^T Lambda_b Omega
    Ys = 0.5 * (Ys + Ys.transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eg(G);
    Eigen::VectorXd gev = eg.eigenvalues().cwiseMax(1e-30);
    Eigen::MatrixXd Gh = eg.eigenvectors() *
        gev.cwiseInverse().cwiseSqrt().asDiagonal() *
        eg.eigenvectors().transpose();
    Eigen::MatrixXd M = Gh * Ys * Gh;
    M = 0.5 * (M + M.transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> em(M);
    double alpha = std::min(1.0, std::max(0.0, em.eigenvalues().minCoeff()));
    Eigen::MatrixXd C = Ys - alpha * G;
    C = 0.5 * (C + C.transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> ec(C);
    Eigen::VectorXd cw = ec.eigenvalues().cwiseMax(0.0);
    double raw_max = ec.eigenvalues().size() ? ec.eigenvalues().maxCoeff() : 0.0;
    double gmax = G.diagonal().maxCoeff();
    double cap = 0.0; int64_t n_capped = 0;
    if (cap_rel > 0.0) {
        cap = cap_rel * std::max(gmax, 1e-30);
        for (int64_t b = 0; b < r; ++b) {
            if (cw(b) > cap) { cw(b) = cap; ++n_capped; }
        }
    }
    C = ec.eigenvectors() * cw.asDiagonal() * ec.eigenvectors().transpose();
    double jit = std::max(1e-8 * raw_max, 1e-12);
    C += jit * Eigen::MatrixXd::Identity(r, r);

    py::array_t<double> DLa({(py::ssize_t)N, (py::ssize_t)6, (py::ssize_t)6});
    {
        auto *p = DLa.mutable_data();
        for (int64_t s = 0; s < N; ++s)
            for (int a = 0; a < 6; ++a)
                for (int b = 0; b < 6; ++b)
                    p[((size_t)s * 6 + a) * 6 + b] = alpha * Di(s * 6 + a, b);
    }
    py::dict out;
    out["DLa"] = DLa;
    out["U"] = eigen_to_np_rowmajor(Q);
    out["C"] = eigen_to_np_rowmajor(C);
    out["info"] = eigen_to_np_rowmajor(Ys);
    out["alpha"] = alpha;
    out["raw_max"] = raw_max;
    out["cap"] = cap;
    out["n_capped"] = n_capped;
    out["gmax"] = gmax;
    return out;
}

// ---- 精确边界信息 Λ_b（Baspacho 分解 H_mm，信息形式，不求逆）----
struct CsrSub {
    std::vector<int64_t> rp, ci;
    std::vector<double> v;
    int64_t rows = 0, cols = 0;
};

static CsrSub _extract_csr(
        const int64_t *rp, const int64_t *ci, const double *hv, int64_t ncol,
        const std::vector<int64_t> &rows, const std::vector<int64_t> &cols,
        const std::vector<double> &d);

static void _block_pattern_from_csr(const CsrSub &A, int64_t bs,
                                    std::vector<int64_t> &bp,
                                    std::vector<int64_t> &bi) {
    if (A.rows != A.cols || bs <= 0 || A.rows % bs != 0)
        throw std::runtime_error("boundary_info_baspacho: H_mm 需方阵且可被块大小整除");
    const int64_t nb = A.rows / bs;
    std::vector<std::vector<int64_t>> rows((size_t)nb);
    for (int64_t i = 0; i < A.rows; ++i) {
        const int64_t br = i / bs;
        for (int64_t p = A.rp[i]; p < A.rp[i + 1]; ++p) {
            const int64_t bc = A.ci[p] / bs;
            rows[(size_t)br].push_back(bc);
            rows[(size_t)bc].push_back(br);       // 对称块图
        }
    }
    bp.assign((size_t)nb + 1, 0);
    bi.clear();
    for (int64_t b = 0; b < nb; ++b) {
        auto &v = rows[(size_t)b];
        v.push_back(b);
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
        bp[(size_t)b + 1] = bp[(size_t)b] + (int64_t)v.size();
        for (int64_t c : v) bi.push_back(c);
    }
}

static py::tuple boundary_info_baspacho(
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> H_rp,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> H_ci,
        py::array_t<double, py::array::c_style | py::array::forcecast> H_val,
        int64_t ncol, int64_t off, int64_t q, py::object scale_obj,
        double jitter_rel) {
    auto r = H_rp.request(); auto c = H_ci.request(); auto v = H_val.request();
    if (r.ndim != 1 || c.ndim != 1 || v.ndim != 1 ||
        r.size != ncol + 1 || c.size != v.size)
        throw std::runtime_error("boundary_info_baspacho: H CSR 形状不符");
    if (q <= 0 || off <= 0 || off + q > ncol || off % 6 != 0)
        throw std::runtime_error("boundary_info_baspacho: off/q 越界或 off 非 6 的倍数");
    const int64_t *rp = (const int64_t *)r.ptr;
    const int64_t *ci = (const int64_t *)c.ptr;
    const double *hv = (const double *)v.ptr;

    std::vector<double> d((size_t)ncol, 1.0);
    bool scaled = false;
    if (!scale_obj.is_none()) {
        auto sc = scale_obj.cast<
            py::array_t<double, py::array::c_style | py::array::forcecast>>();
        auto sa = sc.request();
        if (sa.size != ncol)
            throw std::runtime_error("boundary_info_baspacho: scale 长度 != ncol");
        std::memcpy(d.data(), sa.ptr, sizeof(double) * (size_t)ncol);
        scaled = true;
    }
    std::vector<int64_t> M, B, I;
    M.reserve((size_t)off); B.reserve((size_t)q); I.reserve((size_t)off);
    for (int64_t i = 0; i < off; ++i) M.push_back(i);
    for (int64_t i = 0; i < q; ++i) B.push_back(off + i);
    auto t0 = std::chrono::steady_clock::now();
    CsrSub hMM = _extract_csr(rp, ci, hv, ncol, M, M, d);
    CsrSub hMB = _extract_csr(rp, ci, hv, ncol, M, B, d);
    CsrSub hBB = _extract_csr(rp, ci, hv, ncol, B, B, d);
    if (hMM.rows != off || hMM.cols != off || hMB.cols != q || hBB.rows != q)
        throw std::runtime_error("boundary_info_baspacho: 子块抽取失败");

    auto optsD = torch::TensorOptions().dtype(torch::kFloat64);
    auto optsI = torch::TensorOptions().dtype(torch::kInt64);
    auto dev = torch::kCUDA;
    if (jitter_rel > 0.0) {
        double dmax = 0.0;
        for (int64_t i = 0; i < hMM.rows; ++i)
            for (int64_t p = hMM.rp[i]; p < hMM.rp[i + 1]; ++p)
                if (hMM.ci[p] == i) dmax = std::max(dmax, std::abs(hMM.v[p]));
        const double add = jitter_rel * std::max(dmax, 1e-300);
        for (int64_t i = 0; i < hMM.rows; ++i)
            for (int64_t p = hMM.rp[i]; p < hMM.rp[i + 1]; ++p)
                if (hMM.ci[p] == i) hMM.v[p] += add;
    }
    auto rp_t = torch::from_blob(hMM.rp.data(), {(int64_t)hMM.rp.size()}, optsI)
                    .clone().to(dev);
    auto ci_t = torch::from_blob(hMM.ci.data(), {(int64_t)hMM.ci.size()}, optsI)
                    .clone().to(dev);
    auto v_t = torch::from_blob(hMM.v.data(), {(int64_t)hMM.v.size()}, optsD)
                   .clone().to(dev);
    std::vector<int64_t> bp, bi;
    _block_pattern_from_csr(hMM, 6, bp, bi);
    auto ps_t = torch::full({off / 6}, 6, optsI);
    auto bp_t = torch::from_blob(bp.data(), {(int64_t)bp.size()}, optsI).clone();
    auto bi_t = torch::from_blob(bi.data(), {(int64_t)bi.size()}, optsI).clone();
    auto sym = SymbolicDecomposition(ps_t, bp_t, bi_t, "cuda");
    auto nd = sym.createNumericDecomposition(1);
    nd.add_M(v_t.reshape({1, v_t.numel()}), rp_t, ci_t);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::steady_clock::now();
    nd.factor();
    cudaDeviceSynchronize();
    auto t2 = std::chrono::steady_clock::now();

    // 解 H_mm X = H_mb（q 个右端）；B 布局 (q, off)。
    auto Bcpu = torch::zeros({1, q * off}, optsD);
    {
        auto ba = Bcpu.accessor<double, 2>();
        for (int64_t i = 0; i < hMB.rows; ++i)
            for (int64_t p = hMB.rp[i]; p < hMB.rp[i + 1]; ++p)
                ba[0][hMB.ci[p] * off + i] = hMB.v[p];
    }
    auto Bg = Bcpu.to(dev);
    nd.solve_multi(Bg, q);
    cudaDeviceSynchronize();
    auto t3 = std::chrono::steady_clock::now();
    auto Xc = Bg[0].to(torch::kCPU).contiguous();
    const double *xp = Xc.data_ptr<double>();

    // Λ_b = H_bb - H_bm X；再按列平衡还原到原始 QOE 坐标。
    std::vector<double> out((size_t)q * q, 0.0);
    for (int64_t i = 0; i < hBB.rows; ++i)
        for (int64_t p = hBB.rp[i]; p < hBB.rp[i + 1]; ++p)
            out[(size_t)i * q + hBB.ci[p]] += hBB.v[p];
    double xnorm2 = 0.0, corrmax = 0.0, corrn2 = 0.0;
    for (int64_t b = 0; b < q; ++b)
        for (int64_t i = 0; i < off; ++i) {
            const double xv = xp[b * off + i];
            xnorm2 += xv * xv;
        }
    for (int64_t i = 0; i < hMB.rows; ++i)
        for (int64_t p = hMB.rp[i]; p < hMB.rp[i + 1]; ++p) {
            const int64_t a = hMB.ci[p];
            const double val = hMB.v[p];
            for (int64_t b = 0; b < q; ++b) {
                const double c = val * xp[b * off + i];
                out[(size_t)a * q + b] -= c;
                corrmax = std::max(corrmax, std::abs(c));
                corrn2 += c * c;
            }
        }
    double dg_scaled_min = std::numeric_limits<double>::infinity();
    double dg_scaled_max = -std::numeric_limits<double>::infinity();
    for (int64_t a = 0; a < q; ++a) {
        const double dv = out[(size_t)a * q + a];
        dg_scaled_min = std::min(dg_scaled_min, dv);
        dg_scaled_max = std::max(dg_scaled_max, dv);
    }
    for (int64_t a = 0; a < q; ++a) {
        const double da = scaled ? d[(size_t)(off + a)] : 1.0;
        for (int64_t b = a; b < q; ++b) {
            const double db = scaled ? d[(size_t)(off + b)] : 1.0;
            const double s = 0.5 * (out[(size_t)a * q + b] +
                                    out[(size_t)b * q + a]) / (da * db);
            out[(size_t)a * q + b] = s;
            out[(size_t)b * q + a] = s;
        }
    }
    bool finite = true;
    for (double x : out) finite = finite && std::isfinite(x);
    py::array_t<double> Lam({(py::ssize_t)q, (py::ssize_t)q});
    std::memcpy(Lam.mutable_data(), out.data(), sizeof(double) * (size_t)q * q);
    auto t4 = std::chrono::steady_clock::now();
    auto sec = [](auto a, auto b) {
        return std::chrono::duration<double>(b - a).count();
    };
    py::dict stats;
    stats["extract_s"] = sec(t0, t1);
    stats["factor_s"] = sec(t1, t2);
    stats["solve_s"] = sec(t2, t3);
    stats["update_s"] = sec(t3, t4);
    stats["n_mm"] = off;
    stats["nnz_mm"] = (int64_t)hMM.ci.size();
    stats["nnz_mb"] = (int64_t)hMB.ci.size();
    stats["finite"] = finite;
    stats["scaled"] = scaled;
    stats["x_norm"] = std::sqrt(xnorm2);
    stats["corr_max"] = corrmax;
    stats["corr_norm"] = std::sqrt(corrn2);
    stats["diag_scaled_min"] = dg_scaled_min;
    stats["diag_scaled_max"] = dg_scaled_max;
    return py::make_tuple(Lam, stats);
}

// ---- GPU 局部三角解 vs CPU 对拍 ----
static py::dict partial_tri_probe(
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> H_rp,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> H_ci,
        py::array_t<double, py::array::c_style | py::array::forcecast> H_val,
        int64_t ncol, int64_t span, int64_t bs) {
    auto r = H_rp.request(); auto c = H_ci.request(); auto v = H_val.request();
    if (r.size != ncol + 1 || c.size != v.size || bs <= 0 || ncol % bs != 0)
        throw std::runtime_error("partial_tri_probe: 参数不符");
    const int64_t *rp = (const int64_t *)r.ptr;
    const int64_t *ci = (const int64_t *)c.ptr;
    const double *hv = (const double *)v.ptr;
    std::vector<int64_t> all((size_t)ncol);
    for (int64_t i = 0; i < ncol; ++i) all[(size_t)i] = i;
    std::vector<double> ones((size_t)ncol, 1.0);
    CsrSub hA = _extract_csr(rp, ci, hv, ncol, all, all, ones);
    std::vector<int64_t> bp, bi;
    _block_pattern_from_csr(hA, bs, bp, bi);
    const int64_t nblk = hA.rows / bs;
    auto optsD = torch::TensorOptions().dtype(torch::kFloat64);
    auto optsI = torch::TensorOptions().dtype(torch::kInt64);
    auto ps_t = torch::full({nblk}, bs, optsI);
    auto bp_t = torch::from_blob(bp.data(), {(int64_t)bp.size()}, optsI).clone();
    auto bi_t = torch::from_blob(bi.data(), {(int64_t)bi.size()}, optsI).clone();
    auto rp_c = torch::from_blob(hA.rp.data(), {(int64_t)hA.rp.size()}, optsI).clone();
    auto ci_c = torch::from_blob(hA.ci.data(), {(int64_t)hA.ci.size()}, optsI).clone();
    auto v_c = torch::from_blob(hA.v.data(), {(int64_t)hA.v.size()}, optsD).clone();
    auto sym_cpu = SymbolicDecomposition(ps_t, bp_t, bi_t, "cpu");
    auto nd_cpu = sym_cpu.createNumericDecomposition(1);
    nd_cpu.add_M(v_c.reshape({1, v_c.numel()}), rp_c, ci_c);
    nd_cpu.factor();
    auto sym_gpu = SymbolicDecomposition(ps_t, bp_t, bi_t, "cuda");
    auto nd_gpu = sym_gpu.createNumericDecomposition(1);
    nd_gpu.add_M(v_c.to(torch::kCUDA).reshape({1, v_c.numel()}),
                 rp_c.to(torch::kCUDA), ci_c.to(torch::kCUDA));
    nd_gpu.factor();
    auto x0 = torch::randn({1, ncol}, optsD);
    auto xc = x0.clone();
    auto xg = x0.clone().to(torch::kCUDA);
    nd_cpu.solveL_upto(xc, span);
    nd_gpu.solveL_upto_gpu(xg, span);
    auto eL = (xc - xg.to(torch::kCPU)).norm().item<double>() /
              std::max(xc.norm().item<double>(), 1e-300);
    auto xc2 = x0.clone();
    auto xg2 = x0.clone().to(torch::kCUDA);
    nd_cpu.solveLt_upto(xc2, span);
    nd_gpu.solveLt_upto_gpu(xg2, span);
    auto eLt = (xc2 - xg2.to(torch::kCPU)).norm().item<double>() /
               std::max(xc2.norm().item<double>(), 1e-300);
    py::dict out;
    out["rel_L"] = eL; out["rel_Lt"] = eLt;
    return out;
}

// ---- 精确内部条件：factorUpTo + GPU 局部三角解，不做 factorFrom ----
static std::shared_ptr<SymbolicDecomposition> _partial_symbolic(
        const torch::Tensor &ps_t, const torch::Tensor &bp_t,
        const torch::Tensor &bi_t, const std::vector<int64_t> &last,
        const std::vector<int64_t> &bp, const std::vector<int64_t> &bi,
        const std::vector<int64_t> &ids, int64_t nblk, int64_t bs,
        bool &hit);

static py::array_t<double> partial_interior_solve(
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> H_rp,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> H_ci,
        py::array_t<double, py::array::c_style | py::array::forcecast> H_val,
        int64_t ncol,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> ids_arr,
        py::array_t<double, py::array::c_style | py::array::forcecast> rhs_arr,
        int64_t bs) {
    auto r = H_rp.request(); auto c = H_ci.request(); auto v = H_val.request();
    auto ia = ids_arr.request(); auto ra = rhs_arr.request();
    if (r.size != ncol + 1 || c.size != v.size || ia.ndim != 1 || ia.size <= 0 ||
        ra.size != ncol || bs <= 0 || ncol % bs != 0)
        throw std::runtime_error("partial_interior_solve: 参数不符");
    const int64_t *rp = (const int64_t *)r.ptr;
    const int64_t *ci = (const int64_t *)c.ptr;
    const double *hv = (const double *)v.ptr;
    std::vector<int64_t> all((size_t)ncol);
    for (int64_t i = 0; i < ncol; ++i) all[(size_t)i] = i;
    std::vector<double> ones((size_t)ncol, 1.0);
    CsrSub hA = _extract_csr(rp, ci, hv, ncol, all, all, ones);
    std::vector<int64_t> bp, bi;
    _block_pattern_from_csr(hA, bs, bp, bi);
    const int64_t nblk = hA.rows / bs;
    {
        std::vector<std::vector<int64_t>> rows((size_t)nblk);
        for (int64_t i = 0; i < nblk; ++i)
            for (int64_t p = bp[(size_t)i]; p < bp[(size_t)i + 1]; ++p)
                rows[(size_t)i].push_back(bi[(size_t)p]);
        for (int64_t i = 0; i < ia.size; ++i) {
            const int64_t bi0 = ((const int64_t *)ia.ptr)[i] / bs;
            for (int64_t j = 0; j < ia.size; ++j)
                rows[(size_t)bi0].push_back(((const int64_t *)ia.ptr)[j] / bs);
        }
        bp.assign((size_t)nblk + 1, 0); bi.clear();
        for (int64_t i = 0; i < nblk; ++i) {
            auto &vv = rows[(size_t)i]; vv.push_back(i);
            std::sort(vv.begin(), vv.end());
            vv.erase(std::unique(vv.begin(), vv.end()), vv.end());
            bp[(size_t)i + 1] = bp[(size_t)i] + (int64_t)vv.size();
            for (int64_t j : vv) bi.push_back(j);
        }
    }
    auto optsD = torch::TensorOptions().dtype(torch::kFloat64);
    auto optsI = torch::TensorOptions().dtype(torch::kInt64);
    auto dev = torch::kCUDA;
    auto ps_t = torch::full({nblk}, bs, optsI);
    auto bp_t = torch::from_blob(bp.data(), {(int64_t)bp.size()}, optsI).clone();
    auto bi_t = torch::from_blob(bi.data(), {(int64_t)bi.size()}, optsI).clone();
    std::vector<int64_t> last;
    for (int64_t i = 0; i < ia.size; ++i) {
        int64_t id = ((const int64_t *)ia.ptr)[i];
        if (id < 0 || id >= nblk)
            throw std::runtime_error("partial_interior_solve: id 越界");
        last.push_back(id);
    }
    bool cache_hit = false;
    auto sym = _partial_symbolic(ps_t, bp_t, bi_t, last, bp, bi, last,
                                 nblk, bs, cache_hit);
    auto nd = sym->createNumericDecomposition(1);
    auto rp_t = torch::from_blob(hA.rp.data(), {(int64_t)hA.rp.size()}, optsI)
                    .clone().to(dev);
    auto ci_t = torch::from_blob(hA.ci.data(), {(int64_t)hA.ci.size()}, optsI)
                    .clone().to(dev);
    auto v_t = torch::from_blob(hA.v.data(), {(int64_t)hA.v.size()}, optsD)
                   .clone().to(dev);
    nd.add_M(v_t.reshape({1, v_t.numel()}), rp_t, ci_t);
    auto ids_t = torch::from_blob(ia.ptr, {(int64_t)ia.size}, optsI).clone();
    int64_t span = nd.span_boundary(ids_t);
    nd.factor_upto(span);
    auto xt = torch::from_blob((void *)ra.ptr, {1, (int64_t)ncol}, optsD)
                  .clone().to(dev);
    const int64_t nids = (int64_t)ia.size;
    const int64_t nfirst = nids / 2;
    const int64_t q1 = nfirst * bs;
    const int64_t first_off = ((const int64_t *)ia.ptr)[0] * bs;
    const int64_t last_off = ((const int64_t *)ia.ptr)[nfirst] * bs;
    nd.solveL_upto_gpu(xt, span);
    xt[0].slice(0, first_off, first_off + q1).zero_();
    xt[0].slice(0, last_off, last_off + (nids - nfirst) * bs).zero_();
    nd.solveLt_upto_gpu(xt, span);
    xt[0].slice(0, first_off, first_off + q1).zero_();
    xt[0].slice(0, last_off, last_off + (nids - nfirst) * bs).zero_();
    auto xc = xt[0].to(torch::kCPU).contiguous();
    py::array_t<double> out({(py::ssize_t)ncol});
    std::memcpy(out.mutable_data(), xc.data_ptr<double>(),
                sizeof(double) * (size_t)ncol);
    return out;
}

// ---- BaSpaCho factorUpTo 探针：接口块留最后，提取 trailing Schur ----
static py::dict baspacho_partial_probe(
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> H_rp,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> H_ci,
        py::array_t<double, py::array::c_style | py::array::forcecast> H_val,
        int64_t ncol,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> ids_arr,
        int64_t bs, int64_t do_factor_from) {
    auto r = H_rp.request(); auto c = H_ci.request(); auto v = H_val.request();
    auto ia = ids_arr.request();
    if (r.size != ncol + 1 || c.size != v.size || ia.ndim != 1 || ia.size <= 0)
        throw std::runtime_error("baspacho_partial_probe: 参数维数不符");
    if (bs <= 0 || ncol % bs != 0)
        throw std::runtime_error("baspacho_partial_probe: ncol 需被 bs 整除");
    const int64_t *rp = (const int64_t *)r.ptr;
    const int64_t *ci = (const int64_t *)c.ptr;
    const double *hv = (const double *)v.ptr;
    std::vector<int64_t> all((size_t)ncol);
    for (int64_t i = 0; i < ncol; ++i) all[(size_t)i] = i;
    std::vector<double> ones((size_t)ncol, 1.0);
    auto t0 = std::chrono::steady_clock::now();
    CsrSub hA = _extract_csr(rp, ci, hv, ncol, all, all, ones);
    std::vector<int64_t> bp, bi;
    _block_pattern_from_csr(hA, bs, bp, bi);
    const int64_t nblk = hA.rows / bs;
    {
        std::vector<std::vector<int64_t>> rows((size_t)nblk);
        for (int64_t i = 0; i < nblk; ++i)
            for (int64_t p = bp[(size_t)i]; p < bp[(size_t)i + 1]; ++p)
                rows[(size_t)i].push_back(bi[(size_t)p]);
        for (int64_t i = 0; i < ia.size; ++i) {
            const int64_t bi0 = ((const int64_t *)ia.ptr)[i] / bs;
            for (int64_t j = 0; j < ia.size; ++j)
                rows[(size_t)bi0].push_back(((const int64_t *)ia.ptr)[j] / bs);
        }
        bp.assign((size_t)nblk + 1, 0); bi.clear();
        for (int64_t i = 0; i < nblk; ++i) {
            auto &vv = rows[(size_t)i]; vv.push_back(i);
            std::sort(vv.begin(), vv.end());
            vv.erase(std::unique(vv.begin(), vv.end()), vv.end());
            bp[(size_t)i + 1] = bp[(size_t)i] + (int64_t)vv.size();
            for (int64_t j : vv) bi.push_back(j);
        }
    }
    auto optsD = torch::TensorOptions().dtype(torch::kFloat64);
    auto optsI = torch::TensorOptions().dtype(torch::kInt64);
    auto dev = torch::kCUDA;
    auto ps_t = torch::full({nblk}, bs, optsI);
    auto bp_t = torch::from_blob(bp.data(), {(int64_t)bp.size()}, optsI).clone();
    auto bi_t = torch::from_blob(bi.data(), {(int64_t)bi.size()}, optsI).clone();
    std::vector<int64_t> last;
    for (int64_t i = 0; i < ia.size; ++i) {
        int64_t id = ((const int64_t *)ia.ptr)[i];
        if (id < 0 || id >= nblk)
            throw std::runtime_error("baspacho_partial_probe: id 越界");
        last.push_back(id);
    }
    auto sym = SymbolicDecomposition(ps_t, bp_t, bi_t, "cuda", {}, last,
                                     /*AddFillComplete=*/0, /*find=*/true);
    auto nd = sym.createNumericDecomposition(1);
    auto rp_t = torch::from_blob(hA.rp.data(), {(int64_t)hA.rp.size()}, optsI)
                    .clone().to(dev);
    auto ci_t = torch::from_blob(hA.ci.data(), {(int64_t)hA.ci.size()}, optsI)
                    .clone().to(dev);
    auto v_t = torch::from_blob(hA.v.data(), {(int64_t)hA.v.size()}, optsD)
                   .clone().to(dev);
    nd.add_M(v_t.reshape({1, v_t.numel()}), rp_t, ci_t);
    cudaDeviceSynchronize();
        auto t1 = std::chrono::steady_clock::now();
    auto ids_t = torch::from_blob(ia.ptr, {(int64_t)ia.size}, optsI).clone();
    int64_t span = nd.span_boundary(ids_t);
        fflush(stderr);
    nd.factor_upto(span);
    cudaDeviceSynchronize();
        auto t2 = std::chrono::steady_clock::now();
    auto S = nd.trailing_block_dev(ids_t, bs).cpu();
    cudaDeviceSynchronize();
        auto t3 = std::chrono::steady_clock::now();
    if (do_factor_from) nd.factor_from(span);
    cudaDeviceSynchronize();
        auto t4 = std::chrono::steady_clock::now();
    auto sec = [](auto a, auto b) {
        return std::chrono::duration<double>(b - a).count();
    };
        bool finite = bool(torch::isfinite(S).all().item<bool>());
        py::dict out;
    {
        auto Sc = S.contiguous();
        py::array_t<double> Sa({(py::ssize_t)Sc.size(0), (py::ssize_t)Sc.size(1)});
        std::memcpy(Sa.mutable_data(), Sc.data_ptr<double>(),
                    sizeof(double) * (size_t)Sc.numel());
        out["S"] = Sa;
    }
    out["span"] = span;
    out["nblk"] = nblk;
    out["nnz"] = (int64_t)hA.ci.size();
    out["setup_s"] = sec(t0, t1);
    out["factor_upto_s"] = sec(t1, t2);
    out["block_s"] = sec(t2, t3);
    out["factor_from_s"] = sec(t3, t4);
    out["S_shape"] = py::make_tuple(S.size(0), S.size(1));
    out["finite"] = finite;
        return out;
}

// ---- BaSpaCho 端口因子窗口求解：接口块最后，factorUpTo/factorFrom ----
static uint64_t _partial_hash(const std::vector<int64_t> &a, uint64_t h) {
    for (int64_t x : a) { h ^= (uint64_t)x; h *= 1099511628211ull; }
    return h;
}

// 故意泄漏：静态 map 若在 CUDA driver 关闭后再析构，会在解释器退出时
// 触发 "driver shutting down"。缓存只随进程结束，不参与静态析构。
static auto *g_partial_cache =
    new std::unordered_map<std::string, std::shared_ptr<SymbolicDecomposition>>();

static std::shared_ptr<SymbolicDecomposition> _partial_symbolic(
        const torch::Tensor &ps_t, const torch::Tensor &bp_t,
        const torch::Tensor &bi_t, const std::vector<int64_t> &last,
        const std::vector<int64_t> &bp, const std::vector<int64_t> &bi,
        const std::vector<int64_t> &ids, int64_t nblk, int64_t bs,
        bool &hit) {
    uint64_t h = 1469598103934665603ull;
    h = _partial_hash(bp, h);
    h = _partial_hash(bi, h);
    h = _partial_hash(ids, h);
    std::string key = std::to_string(nblk) + "_" + std::to_string(bs) + "_" +
                      std::to_string(h);
    auto it = g_partial_cache->find(key);
    if (it != g_partial_cache->end()) { hit = true; return it->second; }
    hit = false;
    auto sym = std::make_shared<SymbolicDecomposition>(
        ps_t, bp_t, bi_t, "cuda", std::vector<int64_t>{}, last,
        /*AddFillComplete=*/0, /*find=*/true);
    (*g_partial_cache)[key] = sym;
    return sym;
}

static py::dict baspacho_partial_solve(
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> H_rp,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> H_ci,
        py::array_t<double, py::array::c_style | py::array::forcecast> H_val,
        int64_t ncol,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> ids_arr,
        py::array_t<double, py::array::c_style | py::array::forcecast> g_arr,
        int64_t bs) {
    auto r = H_rp.request(); auto c = H_ci.request(); auto v = H_val.request();
    auto ia = ids_arr.request(); auto ga = g_arr.request();
    if (r.size != ncol + 1 || c.size != v.size || ia.ndim != 1 ||
        ia.size <= 0 || ia.size % 2 != 0 || ga.size != ncol)
        throw std::runtime_error("baspacho_partial_solve: 参数维数不符");
    const int64_t *rp = (const int64_t *)r.ptr;
    const int64_t *ci = (const int64_t *)c.ptr;
    const double *hv = (const double *)v.ptr;
    std::vector<int64_t> all((size_t)ncol);
    for (int64_t i = 0; i < ncol; ++i) all[(size_t)i] = i;
    std::vector<double> ones((size_t)ncol, 1.0);
    auto t0 = std::chrono::steady_clock::now();
    CsrSub hA = _extract_csr(rp, ci, hv, ncol, all, all, ones);
    std::vector<int64_t> bp, bi;
    _block_pattern_from_csr(hA, bs, bp, bi);
    const int64_t nblk = hA.rows / bs;
    auto optsD = torch::TensorOptions().dtype(torch::kFloat64);
    auto optsI = torch::TensorOptions().dtype(torch::kInt64);
    auto dev = torch::kCUDA;
    auto ps_t = torch::full({nblk}, bs, optsI);
    auto bp_t = torch::from_blob(bp.data(), {(int64_t)bp.size()}, optsI).clone();
    auto bi_t = torch::from_blob(bi.data(), {(int64_t)bi.size()}, optsI).clone();
    std::vector<int64_t> last;
    for (int64_t i = 0; i < ia.size; ++i) {
        int64_t id = ((const int64_t *)ia.ptr)[i];
        if (id < 0 || id >= nblk)
            throw std::runtime_error("baspacho_partial_solve: id 越界");
        last.push_back(id);
    }
    bool cache_hit = false;
    auto sym = _partial_symbolic(ps_t, bp_t, bi_t, last, bp, bi, last,
                                 nblk, bs, cache_hit);
    auto nd = sym->createNumericDecomposition(1);
    auto rp_t = torch::from_blob(hA.rp.data(), {(int64_t)hA.rp.size()}, optsI)
                    .clone().to(dev);
    auto ci_t = torch::from_blob(hA.ci.data(), {(int64_t)hA.ci.size()}, optsI)
                    .clone().to(dev);
    auto v_t = torch::from_blob(hA.v.data(), {(int64_t)hA.v.size()}, optsD)
                   .clone().to(dev);
    nd.add_M(v_t.reshape({1, v_t.numel()}), rp_t, ci_t);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::steady_clock::now();
    auto ids_t = torch::from_blob(ia.ptr, {(int64_t)ia.size}, optsI).clone();
    int64_t span = nd.span_boundary(ids_t);
    nd.factor_upto(span);
    cudaDeviceSynchronize();
    auto t2 = std::chrono::steady_clock::now();
    auto S = nd.trailing_block_dev(ids_t, bs);
    cudaDeviceSynchronize();
    auto t3 = std::chrono::steady_clock::now();
    const int64_t nids = (int64_t)ia.size;
    const int64_t nfirst = nids / 2;
    const int64_t q1 = nfirst * bs;
    const int64_t q2 = (nids - nfirst) * bs;
    auto Sg = S.to(dev);
    auto S11 = Sg.slice(0, 0, q1).slice(1, 0, q1);
    auto S12 = Sg.slice(0, 0, q1).slice(1, q1, q1 + q2);
    auto S22 = Sg.slice(0, q1, q1 + q2).slice(1, q1, q1 + q2);
    auto S11j = S11.clone();
    {
        auto dg = torch::diagonal(S11j);
        auto jit = 1e-12 * dg.abs().max().clamp_min(1e-30);
        S11j += jit * torch::eye(q1, optsD.device(dev));
    }
    auto Lc = torch::linalg_cholesky(S11j);
    auto Xc = torch::linalg_solve_triangular(Lc, S12, /*upper=*/false);
    auto Yc = torch::linalg_solve_triangular(Lc.transpose(0, 1), Xc, /*upper=*/true);
    auto Pnext = S22 - S12.transpose(0, 1).matmul(Yc);
    Pnext = 0.5 * (Pnext + Pnext.transpose(0, 1));
    auto S22j = S22.clone();
    {
        auto dg = torch::diagonal(S22j);
        auto jit = 1e-12 * dg.abs().max().clamp_min(1e-30);
        S22j += jit * torch::eye(q2, optsD.device(dev));
    }
    auto L2 = torch::linalg_cholesky(S22j);
    auto X2 = torch::linalg_solve_triangular(L2, S12.transpose(0, 1), false);
    auto Y2 = torch::linalg_solve_triangular(L2.transpose(0, 1), X2, true);
    auto Pfirst = S11 - S12.matmul(Y2);
    Pfirst = 0.5 * (Pfirst + Pfirst.transpose(0, 1));
    auto t4 = std::chrono::steady_clock::now();
    nd.factor_from(span);
    cudaDeviceSynchronize();
    auto t5 = std::chrono::steady_clock::now();
    auto gt = torch::from_blob((void *)ga.ptr, {1, (int64_t)ncol}, optsD)
                  .clone().to(dev);
    nd.solve(gt);
    cudaDeviceSynchronize();
    auto t6 = std::chrono::steady_clock::now();
    auto xc = gt[0].to(torch::kCPU).contiguous();
    auto Pc = Pnext.to(torch::kCPU).contiguous();
    auto center = xc.narrow(0, ncol - q2, q2).contiguous();
    py::array_t<double> xo({(py::ssize_t)ncol});
    py::array_t<double> Po({(py::ssize_t)q2, (py::ssize_t)q2});
    py::array_t<double> co({(py::ssize_t)q2});
    std::memcpy(xo.mutable_data(), xc.data_ptr<double>(), sizeof(double) * (size_t)ncol);
    std::memcpy(Po.mutable_data(), Pc.data_ptr<double>(), sizeof(double) * (size_t)q2 * q2);
    std::memcpy(co.mutable_data(), center.data_ptr<double>(), sizeof(double) * (size_t)q2);
    py::dict out;
    out["x"] = xo; out["Pnext"] = Po; out["center"] = co;
    auto sec = [](auto a, auto b) {
        return std::chrono::duration<double>(b - a).count();
    };
    out["span"] = span;
    out["cache_hit"] = cache_hit;
    out["setup_s"] = sec(t0, t1);
    out["factor_upto_s"] = sec(t1, t2);
    out["block_s"] = sec(t2, t3);
    out["marginal_s"] = sec(t3, t4);
    out["factor_from_s"] = sec(t4, t5);
    out["solve_s"] = sec(t5, t6);
    out["finite"] = bool(torch::isfinite(gt).all().item<bool>() &&
                         torch::isfinite(Pnext).all().item<bool>() &&
                         torch::isfinite(Pfirst).all().item<bool>());
    return out;
}

// ---- 端口因子 + 独立 prior 块：避免 Python 端 49M nnz CSR 加法 ----
static py::dict _partial_solve_impl(
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> H_rp,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> H_ci,
        py::array_t<double, py::array::c_style | py::array::forcecast> H_val,
        int64_t ncol,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> ids_arr,
        py::array_t<double, py::array::c_style | py::array::forcecast> g_arr,
        int64_t bs,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> P_rp,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> P_ci,
        py::array_t<double, py::array::c_style | py::array::forcecast> P_val,
        int64_t prior_off,
        py::object scale_obj,
        py::object P2_rp_obj, py::object P2_ci_obj, py::object P2_val_obj,
        int64_t prior2_off, int64_t do_cov, int64_t need_pfirst,
        int64_t port_ldl, int64_t do_gain, py::object cov_nodes_obj,
        py::object rts_phi_obj, py::object rts_q_obj,
        py::object rts_delta_obj, py::object rts_cf_obj,
        py::object rts_cl_obj, py::object rts_xs_obj,
        int64_t rts_gain) {
    auto r = H_rp.request(); auto c = H_ci.request(); auto v = H_val.request();
    auto ia = ids_arr.request(); auto ga = g_arr.request();
    auto pr = P_rp.request(); auto pc = P_ci.request(); auto pv = P_val.request();
    const int64_t qp = (int64_t)pr.size - 1;
    if (r.size != ncol + 1 || c.size != v.size || ia.ndim != 1 ||
        ia.size <= 0 || ia.size % 2 != 0 || ga.size != ncol ||
        pr.size <= 1 || pc.size != pv.size || prior_off < 0 ||
        prior_off + qp > ncol)
        throw std::runtime_error("baspacho_partial_solve_prior: 参数维数不符");
    const int64_t *rp = (const int64_t *)r.ptr;
    const int64_t *ci = (const int64_t *)c.ptr;
    const double *hv = (const double *)v.ptr;
    const int64_t *prp = (const int64_t *)pr.ptr;
    const int64_t *pci = (const int64_t *)pc.ptr;
    const double *pval = (const double *)pv.ptr;
    const bool has2 = !P2_rp_obj.is_none();
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> cov_nodes_a;
    const int64_t *cov_nodes = nullptr;
    int64_t n_cov_nodes = 0;
    if (!cov_nodes_obj.is_none()) {
        cov_nodes_a = cov_nodes_obj.cast<
            py::array_t<int64_t, py::array::c_style | py::array::forcecast>>();
        auto cn = cov_nodes_a.request();
        if (cn.ndim != 1)
            throw std::runtime_error("baspacho_partial_solve_prior: cov_nodes 需 1D");
        n_cov_nodes = (int64_t)cn.size;
        cov_nodes = (const int64_t *)cn.ptr;
    }
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> P2_rp_a, P2_ci_a;
    py::array_t<double, py::array::c_style | py::array::forcecast> P2_val_a;
    const int64_t *p2rp = nullptr; const int64_t *p2ci = nullptr;
    const double *p2val = nullptr; int64_t qp2 = 0;
    if (has2) {
        P2_rp_a = P2_rp_obj.cast<py::array_t<int64_t, py::array::c_style | py::array::forcecast>>();
        P2_ci_a = P2_ci_obj.cast<py::array_t<int64_t, py::array::c_style | py::array::forcecast>>();
        P2_val_a = P2_val_obj.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        auto r2 = P2_rp_a.request(); auto c2 = P2_ci_a.request(); auto v2 = P2_val_a.request();
        qp2 = (int64_t)r2.size - 1;
        if (r2.size <= 1 || c2.size != v2.size || prior2_off < 0 ||
            prior2_off + qp2 > ncol || qp2 != qp)
            throw std::runtime_error("baspacho_partial_solve_prior: 第二 prior 维数不符");
        p2rp = (const int64_t *)r2.ptr; p2ci = (const int64_t *)c2.ptr;
        p2val = (const double *)v2.ptr;
    }
    std::vector<double> d((size_t)ncol, 1.0);
    if (!scale_obj.is_none()) {
        auto sc = scale_obj.cast<
            py::array_t<double, py::array::c_style | py::array::forcecast>>();
        auto sa = sc.request();
        if (sa.size != ncol)
            throw std::runtime_error("baspacho_partial_solve_prior: scale 长度 != ncol");
        std::memcpy(d.data(), sa.ptr, sizeof(double) * (size_t)ncol);
    }
    std::vector<int64_t> all((size_t)ncol);
    for (int64_t i = 0; i < ncol; ++i) all[(size_t)i] = i;
    std::vector<double> ones((size_t)ncol, 1.0);
    auto t0 = std::chrono::steady_clock::now();
    CsrSub hA = _extract_csr(rp, ci, hv, ncol, all, all, ones);
    std::vector<int64_t> bp, bi;
    _block_pattern_from_csr(hA, bs, bp, bi);
    const int64_t nblk = hA.rows / bs;
    // 把 prior 块的实际 6×6 块对并入符号图案（prior 可能是块对角或稠密）。
    {
        std::vector<std::vector<int64_t>> rows((size_t)nblk);
        for (int64_t i = 0; i < nblk; ++i) {
            for (int64_t p = bp[(size_t)i]; p < bp[(size_t)i + 1]; ++p)
                rows[(size_t)i].push_back(bi[(size_t)p]);
        }
        // 固定把 prior 节点当作稠密块图：prior 值可能是块对角或稠密，
        // 但符号图案必须跨窗稳定，否则每次 Pnext 的零元素变化都会击穿缓存。
        auto _add_dense_node = [&](int64_t off, int64_t qq) {
            const int64_t pb0 = off / bs;
            const int64_t pnb = qq / bs;
            for (int64_t i = 0; i < pnb; ++i)
                for (int64_t j = 0; j < pnb; ++j)
                    rows[(size_t)(pb0 + i)].push_back(pb0 + j);
        };
        _add_dense_node(prior_off, qp);
        if (has2) _add_dense_node(prior2_off, qp2);
        // 两个端口之间的 fill 也必须预先存在；否则 factorUpTo 没有位置写
        // Schur 的 cross 块，trailing_block 会读回错误的 S。
        for (int64_t i = 0; i < ia.size; ++i) {
            const int64_t bi0 = ((const int64_t *)ia.ptr)[i] / bs;
            for (int64_t j = 0; j < ia.size; ++j) {
                const int64_t bj0 = ((const int64_t *)ia.ptr)[j] / bs;
                rows[(size_t)bi0].push_back(bj0);
            }
        }
        bp.assign((size_t)nblk + 1, 0);
        bi.clear();
        for (int64_t i = 0; i < nblk; ++i) {
            auto &vv = rows[(size_t)i];
            vv.push_back(i);
            std::sort(vv.begin(), vv.end());
            vv.erase(std::unique(vv.begin(), vv.end()), vv.end());
            bp[(size_t)i + 1] = bp[(size_t)i] + (int64_t)vv.size();
            for (int64_t j : vv) bi.push_back(j);
        }
    }
    auto optsD = torch::TensorOptions().dtype(torch::kFloat64);
    auto optsI = torch::TensorOptions().dtype(torch::kInt64);
    auto dev = torch::kCUDA;
    auto ps_t = torch::full({nblk}, bs, optsI);
    auto bp_t = torch::from_blob(bp.data(), {(int64_t)bp.size()}, optsI).clone();
    auto bi_t = torch::from_blob(bi.data(), {(int64_t)bi.size()}, optsI).clone();
    std::vector<int64_t> last;
    for (int64_t i = 0; i < ia.size; ++i) {
        int64_t id = ((const int64_t *)ia.ptr)[i];
        if (id < 0 || id >= nblk)
            throw std::runtime_error("baspacho_partial_solve_prior: id 越界");
        last.push_back(id);
    }
    bool cache_hit = false;
    auto sym = _partial_symbolic(ps_t, bp_t, bi_t, last, bp, bi, last,
                                 nblk, bs, cache_hit);
    auto nd = sym->createNumericDecomposition(1);
    auto rp_t = torch::from_blob(hA.rp.data(), {(int64_t)hA.rp.size()}, optsI)
                    .clone().to(dev);
    auto ci_t = torch::from_blob(hA.ci.data(), {(int64_t)hA.ci.size()}, optsI)
                    .clone().to(dev);
    auto v_t = torch::from_blob(hA.v.data(), {(int64_t)hA.v.size()}, optsD)
                   .clone().to(dev);
    nd.add_M(v_t.reshape({1, v_t.numel()}), rp_t, ci_t);
    // prior 嵌入 ncol CSR：局部 (r,c) -> (off+r, off+c)，并做 D_p M D_p。
    std::vector<int64_t> xrp((size_t)ncol + 1, 0);
    std::vector<int64_t> xci;
    std::vector<double> xval;
    xci.reserve((size_t)pc.size);
    xval.reserve((size_t)pc.size);
    for (int64_t i = 0; i < ncol; ++i) {
        int64_t cnt = 0;
        if (i >= prior_off && i < prior_off + qp) {
            const int64_t lr = i - prior_off;
            cnt = prp[lr + 1] - prp[lr];
            for (int64_t p = prp[lr]; p < prp[lr + 1]; ++p) {
                const int64_t lc = pci[p];
                xci.push_back(prior_off + lc);
                xval.push_back(pval[p] * d[(size_t)i] * d[(size_t)(prior_off + lc)]);
            }
        }
        xrp[(size_t)i + 1] = xrp[(size_t)i] + cnt;
    }
    auto xrp_t = torch::from_blob(xrp.data(), {(int64_t)xrp.size()}, optsI)
                     .clone().to(dev);
    auto xci_t = torch::from_blob(xci.data(), {(int64_t)xci.size()}, optsI)
                     .clone().to(dev);
    auto xv_t = torch::from_blob(xval.data(), {(int64_t)xval.size()}, optsD)
                    .clone().to(dev);
    if (xci.size()) nd.add_M(xv_t.reshape({1, xv_t.numel()}), xrp_t, xci_t);
    if (has2) {
        std::vector<int64_t> rp2((size_t)ncol + 1, 0);
        std::vector<int64_t> ci2;
        std::vector<double> v2;
        ci2.reserve((size_t)P2_ci_a.size());
        v2.reserve((size_t)P2_val_a.size());
        for (int64_t i = 0; i < ncol; ++i) {
            int64_t cnt = 0;
            if (i >= prior2_off && i < prior2_off + qp2) {
                const int64_t lr = i - prior2_off;
                cnt = p2rp[lr + 1] - p2rp[lr];
                for (int64_t p = p2rp[lr]; p < p2rp[lr + 1]; ++p) {
                    const int64_t lc = p2ci[p];
                    ci2.push_back(prior2_off + lc);
                    v2.push_back(p2val[p] * d[(size_t)i] *
                                 d[(size_t)(prior2_off + lc)]);
                }
            }
            rp2[(size_t)i + 1] = rp2[(size_t)i] + cnt;
        }
        auto rp2_t = torch::from_blob(rp2.data(), {(int64_t)rp2.size()}, optsI)
                         .clone().to(dev);
        auto ci2_t = torch::from_blob(ci2.data(), {(int64_t)ci2.size()}, optsI)
                         .clone().to(dev);
        auto v2_t = torch::from_blob(v2.data(), {(int64_t)v2.size()}, optsD)
                        .clone().to(dev);
        if (ci2.size()) nd.add_M(v2_t.reshape({1, v2_t.numel()}), rp2_t, ci2_t);
    }
    nd.damp(torch::zeros(1, optsD.device(dev)),
            torch::full({1}, 1e-9, optsD.device(dev)));
    cudaDeviceSynchronize();
    auto t1 = std::chrono::steady_clock::now();
    auto ids_t = torch::from_blob(ia.ptr, {(int64_t)ia.size}, optsI).clone();
    int64_t span = nd.span_boundary(ids_t);
    nd.factor_upto(span);
    cudaDeviceSynchronize();
    auto t2 = std::chrono::steady_clock::now();
    auto S = nd.trailing_block_dev(ids_t, bs);
    cudaDeviceSynchronize();
    auto t3 = std::chrono::steady_clock::now();
    const int64_t nids = (int64_t)ia.size;
    const int64_t nfirst = nids / 2;
    const int64_t q1 = nfirst * bs;
    const int64_t q2 = (nids - nfirst) * bs;
    auto Sg = S.to(dev);
    auto S11 = Sg.slice(0, 0, q1).slice(1, 0, q1);
    auto S12 = Sg.slice(0, 0, q1).slice(1, q1, q1 + q2);
    auto S22 = Sg.slice(0, q1, q1 + q2).slice(1, q1, q1 + q2);
    auto S11j = S11.clone();
    {
        auto dg = torch::diagonal(S11j);
        auto jit = 1e-12 * dg.abs().max().clamp_min(1e-30);
        S11j += jit * torch::eye(q1, optsD.device(dev));
    }
    auto Lc = torch::linalg_cholesky(S11j);
    auto Xc = torch::linalg_solve_triangular(Lc, S12, /*upper=*/false);
    auto Yc = torch::linalg_solve_triangular(Lc.transpose(0, 1), Xc, /*upper=*/true);
    auto Pnext = S22 - S12.transpose(0, 1).matmul(Yc);
    Pnext = 0.5 * (Pnext + Pnext.transpose(0, 1));
    torch::Tensor Pfirst;
    if (need_pfirst) {
        auto S22j = S22.clone();
        {
            auto dg = torch::diagonal(S22j);
            auto jit = 1e-12 * dg.abs().max().clamp_min(1e-30);
            S22j += jit * torch::eye(q2, optsD.device(dev));
        }
        auto L2 = torch::linalg_cholesky(S22j);
        auto X2 = torch::linalg_solve_triangular(L2, S12.transpose(0, 1), false);
        auto Y2 = torch::linalg_solve_triangular(L2.transpose(0, 1), X2, true);
        Pfirst = S11 - S12.matmul(Y2);
        Pfirst = 0.5 * (Pfirst + Pfirst.transpose(0, 1));
    } else {
        Pfirst = torch::zeros({1, 1}, optsD);
    }
    auto t4 = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point t5, t6;
    auto gt = torch::from_blob((void *)ga.ptr, {1, (int64_t)ncol}, optsD)
                  .clone().to(dev);
    torch::Tensor xc_t;
    bool full_factored = false;
    if (port_ldl) {
        // 端口参数 -> S 块位置（首端口 0..q1，末端口 q1..q1+q2）
        const int64_t *idp = (const int64_t *)ia.ptr;
        std::vector<int64_t> ppos((size_t)ncol, -1);
        for (int64_t j = 0; j < nfirst; ++j)
            for (int64_t a = 0; a < bs; ++a)
                ppos[(size_t)(idp[j] * bs + a)] = j * bs + a;
        for (int64_t j = 0; j < nids - nfirst; ++j)
            for (int64_t a = 0; a < bs; ++a)
                ppos[(size_t)(idp[nfirst + j] * bs + a)] = q1 + j * bs + a;
        // 1) 内部零端口解 x_I0 = A_II^{-1} g_I（复用 factorUpTo 的内部因子）
        auto xt = gt.clone();
        nd.solveL_upto_gpu(xt, span);
        for (int64_t p = 0; p < ncol; ++p)
            if (ppos[(size_t)p] >= 0) xt[0][p] = 0.0;
        nd.solveLt_upto_gpu(xt, span);
        for (int64_t p = 0; p < ncol; ++p)
            if (ppos[(size_t)p] >= 0) xt[0][p] = 0.0;
        auto xI0 = xt[0].to(torch::kCPU).contiguous();
        auto xI0p = xI0.data_ptr<double>();
        // 2) 端口右端 rhs_B = g_B - H_BI x_I0；H 的 prior 只落在端口，故 H_BI = AtA_BI
        const double *gvp = (const double *)ga.ptr;
        std::vector<double> rb((size_t)(q1 + q2), 0.0);
        for (int64_t p = 0; p < ncol; ++p)
            if (ppos[(size_t)p] >= 0) rb[(size_t)ppos[(size_t)p]] = gvp[p];
        for (int64_t i = 0; i < ncol; ++i) {
            if (ppos[(size_t)i] >= 0) continue;
            double xi = xI0p[i];
            if (xi == 0.0) continue;
            for (int64_t p = hA.rp[(size_t)i]; p < hA.rp[(size_t)i + 1]; ++p) {
                int64_t j = hA.ci[(size_t)p];
                int64_t pj = ppos[(size_t)j];
                if (pj >= 0) rb[(size_t)pj] -= hA.v[(size_t)p] * xi;
            }
        }
        // 3) 块 LDL：S = [[S11,S12],[S12ᵀ,S22]]，复用 Lc 与 Pnext
        auto rbt = torch::from_blob(rb.data(), {1, q1 + q2}, optsD).clone().to(dev);
        auto z1 = torch::linalg_solve_triangular(
            Lc, rbt[0].narrow(0, 0, q1).unsqueeze(1), /*upper=*/false);
        z1 = torch::linalg_solve_triangular(
            Lc.transpose(0, 1), z1, /*upper=*/true);
        auto r2p = rbt[0].narrow(0, q1, q2).unsqueeze(1) -
                   torch::matmul(S12.transpose(0, 1), z1);
        auto Pnj = Pnext.clone();
        {
            auto dg = torch::diagonal(Pnj);
            auto jit = 1e-12 * dg.abs().max().clamp_min(1e-30);
            Pnj += jit * torch::eye(q2, optsD.device(dev));
        }
        auto L2b = torch::linalg_cholesky(Pnj);
        auto y2 = torch::linalg_solve_triangular(L2b, r2p, false);
        y2 = torch::linalg_solve_triangular(L2b.transpose(0, 1), y2, true);
        auto y1 = z1 - torch::matmul(Yc, y2);
        auto yt = torch::cat({y1.squeeze(1), y2.squeeze(1)});
        auto yc = yt.to(torch::kCPU).contiguous();
        auto ycp = yc.data_ptr<double>();
        t5 = std::chrono::steady_clock::now();
        // 4) 内部条件 A_II x_I = g_I - H_IB y_B，复用同一内部因子
        std::vector<double> ri((size_t)ncol, 0.0);
        for (int64_t i = 0; i < ncol; ++i) {
            if (ppos[(size_t)i] >= 0) continue;
            double s = gvp[i];
            for (int64_t p = hA.rp[(size_t)i]; p < hA.rp[(size_t)i + 1]; ++p) {
                int64_t j = hA.ci[(size_t)p];
                int64_t pj = ppos[(size_t)j];
                if (pj >= 0) s -= hA.v[(size_t)p] * ycp[pj];
            }
            ri[(size_t)i] = s;
        }
        auto xt2 = torch::from_blob(ri.data(), {1, ncol}, optsD).clone().to(dev);
        nd.solveL_upto_gpu(xt2, span);
        for (int64_t p = 0; p < ncol; ++p)
            if (ppos[(size_t)p] >= 0) xt2[0][p] = 0.0;
        nd.solveLt_upto_gpu(xt2, span);
        for (int64_t p = 0; p < ncol; ++p)
            if (ppos[(size_t)p] >= 0) xt2[0][p] = 0.0;
        auto xIc = xt2[0].to(torch::kCPU).contiguous();
        auto xIcp = xIc.data_ptr<double>();
        std::vector<double> xv((size_t)ncol);
        for (int64_t p = 0; p < ncol; ++p)
            xv[(size_t)p] = (ppos[(size_t)p] >= 0) ? ycp[ppos[(size_t)p]]
                                                   : xIcp[p];
        xc_t = torch::from_blob(xv.data(), {ncol}, optsD).clone();
        cudaDeviceSynchronize();
        t6 = std::chrono::steady_clock::now();
    } else {
        nd.factor_from(span);
        cudaDeviceSynchronize();
        full_factored = true;
        t5 = std::chrono::steady_clock::now();
        nd.solve(gt);
        cudaDeviceSynchronize();
        t6 = std::chrono::steady_clock::now();
        xc_t = gt[0];
    }
    auto xc = xc_t.to(torch::kCPU).contiguous();
    auto Pc = Pnext.to(torch::kCPU).contiguous();
    const int64_t first_off = ((const int64_t *)ia.ptr)[0] * bs;
    const int64_t last_off = ((const int64_t *)ia.ptr)[nfirst] * bs;
    auto center = xc.narrow(0, last_off, q2).contiguous();
    auto center_first = xc.narrow(0, first_off, q1).contiguous();
    auto dt = torch::from_blob(d.data(), {(int64_t)ncol}, optsD).clone();
    auto xorig = xc * dt;
    auto dl = dt.narrow(0, last_off, q2);
    auto df = dt.narrow(0, first_off, q1);
    auto Porig = Pc / (dl.unsqueeze(1) * dl.unsqueeze(0));
    torch::Tensor Pf_orig;
    if (need_pfirst) {
        auto Pfc = Pfirst.to(torch::kCPU).contiguous();
        Pf_orig = Pfc / (df.unsqueeze(1) * df.unsqueeze(0));
    } else {
        Pf_orig = torch::zeros({q1, q1}, optsD);
    }
    auto corig = center * dl;
    auto cf_orig = center_first * df;
    // ---- T3 后向 RTS：全部在端口因子 S11/S12/Pnext 与内部因子上完成 ----
    // 生产接口的相邻窗并不共享自由节点：窗 w 的末自由节点 l 经 GVE+Q 预测
    // 到端点 e（即下一窗首节点）。给定过滤分布 (f,l) 与 e=Phi l + w，
    // 缩放坐标：u_e=Phi_s u_l+w_s，Phi_s=dl^{-1}Phi_Q dl, Q_s=dl^{-1}Q dl^{-1};
    //   K_l=Pnext_s^{-1}Phi_s^T(Phi_s Pnext_s^{-1}Phi_s^T+Q_s)^{-1},
    //   u_l=K_l(delta/dl), u_f=-Yc u_l,
    // 平滑均值 ys_f=cf+df*u_f, ys_l=cl+dl*u_l；内部节点用同一 factorUpTo
    // 因子解条件方程。
    torch::Tensor rts_kf_t, rts_kl_t;
    py::array_t<double> rts_xo, rts_dfo, rts_dlo;
    const bool rts_apply = (rts_gain >= 0 && !rts_phi_obj.is_none() &&
                            !rts_q_obj.is_none() && !rts_delta_obj.is_none() &&
                            !rts_cf_obj.is_none() && !rts_cl_obj.is_none() &&
                            !rts_xs_obj.is_none());
    if (rts_apply) {
        if (!port_ldl)
            throw std::runtime_error("rts: 需要 port_ldl=1 的内部因子");
        auto _arr = [](py::object o) {
            return o.cast<py::array_t<double,
                py::array::c_style | py::array::forcecast>>();
        };
        auto pa = _arr(rts_phi_obj); auto qa = _arr(rts_q_obj);
        auto da = _arr(rts_delta_obj); auto cfa = _arr(rts_cf_obj);
        auto cla = _arr(rts_cl_obj); auto xsa = _arr(rts_xs_obj);
        if (pa.size() != nfirst * bs * bs || qa.size() != pa.size() ||
            da.size() != q2 || cfa.size() != q1 || cla.size() != q2 ||
            xsa.size() != ncol)
            throw std::runtime_error("rts: 参数维数不符");
        auto opts_dev = torch::TensorOptions().dtype(torch::kFloat64)
                            .device(dev);
        auto phi_t = torch::from_blob(pa.mutable_data(),
                        {nfirst, bs, bs}, optsD).clone().to(dev);
        auto q_t = torch::from_blob(qa.mutable_data(),
                        {nfirst, bs, bs}, optsD).clone().to(dev);
        auto delta_t = torch::from_blob(da.mutable_data(), {q2}, optsD)
                           .clone().to(dev);
        // **在缩放坐标做 RTS**：Pnext_scaled 与 Yc 都是 well-conditioned，
        // 避免了原始 QOE 信息阵 ~1e15 动态范围下的 inv 与抵消。
        //   u_e = Phi_s u_l + w_s,  Phi_s = dl^{-1} Phi_Q dl,
        //   Q_s = dl^{-1} Q dl^{-1},  delta_s = delta/dl,
        //   K_l = Pnext_s^{-1} Phi_s^T (Phi_s Pnext_s^{-1} Phi_s^T + Q_s)^{-1},
        //   u_l = K_l delta_s,  u_f = -Yc u_l.
        auto dl_g = dt.narrow(0, last_off, q2).to(opts_dev);
        auto df_g = dt.narrow(0, first_off, q1).to(opts_dev);
        auto dl6 = dl_g.reshape({nfirst, bs});
        auto phi_s = phi_t * dl6.unsqueeze(1) / dl6.unsqueeze(2);
        auto q_s = q_t / (dl6.unsqueeze(2) * dl6.unsqueeze(1));
        auto Pn = Pnext.to(opts_dev).clone();
        {
            auto dg = torch::diagonal(Pn);
            auto jit = 1e-12 * dg.abs().max().clamp_min(1e-30);
            Pn += jit * torch::eye(q2, opts_dev);
        }
        auto Lp = torch::linalg_cholesky(Pn);
        auto RHS = torch::zeros({q2, q2}, opts_dev);
        auto phiTs = phi_s.transpose(1, 2).contiguous();
        for (int64_t i = 0; i < nfirst; ++i)
            RHS.slice(0, i * bs, (i + 1) * bs)
               .slice(1, i * bs, (i + 1) * bs).copy_(phiTs[i]);
        auto M = torch::linalg_solve_triangular(Lp, RHS, /*upper=*/false);
        M = torch::linalg_solve_triangular(Lp.transpose(0, 1), M, true);
        auto Mv = M.reshape({nfirst, bs, q2});
        auto Se = torch::matmul(phi_s, Mv).reshape({q2, q2}).clone();
        for (int64_t i = 0; i < nfirst; ++i)
            Se.slice(0, i * bs, (i + 1) * bs)
              .slice(1, i * bs, (i + 1) * bs) += q_s[i];
        Se = 0.5 * (Se + Se.transpose(0, 1));
        auto Lse = torch::linalg_cholesky(Se);
        auto ds = delta_t / dl_g;
        auto yv = torch::linalg_solve_triangular(
            Lse, ds.reshape({q2, 1}), /*upper=*/false);
        yv = torch::linalg_solve_triangular(
            Lse.transpose(0, 1), yv, /*upper=*/true);
        auto ul = M.matmul(yv).reshape({q2});
        auto uf = -(Yc.matmul(ul));
        auto Dl = dl_g * ul;
        auto Df = df_g * uf;
        if (rts_gain) {
            auto Klt = torch::linalg_solve_triangular(
                Lse, M.transpose(0, 1), /*upper=*/false);
            Klt = torch::linalg_solve_triangular(
                Lse.transpose(0, 1), Klt, /*upper=*/true);
            auto Kl = Klt.transpose(0, 1);
            auto Kf = -(df_g.unsqueeze(1) *
                        (Yc.matmul(Kl) / dl_g.unsqueeze(0)));
            // K_l 原始坐标 = dl K_l_s dl^{-1}；K_f 已按上式吸收同一尺度。
            auto Kl_orig = dl_g.unsqueeze(1) * Kl / dl_g.unsqueeze(0);
            rts_kl_t = Kl_orig.to(torch::kCPU).contiguous();
            rts_kf_t = Kf.to(torch::kCPU).contiguous();
        }
        auto Dfc = Df.to(torch::kCPU).contiguous();
        auto Dlc = Dl.to(torch::kCPU).contiguous();
        const double *dfp = Dfc.data_ptr<double>();
        const double *dlp = Dlc.data_ptr<double>();
        const double *cfp = (const double *)cfa.request().ptr;
        const double *clp = (const double *)cla.request().ptr;
        const double *xsp = (const double *)xsa.request().ptr;
        const int64_t *idp = (const int64_t *)ia.ptr;
        auto wrap_pi = [](double v) {
            v = std::fmod(v + M_PI, 2.0 * M_PI);
            if (v < 0.0) v += 2.0 * M_PI;
            return v - M_PI;
        };
        // 端口 scal ed 增量 u_B = (y_smooth - xs_lin) / d（角度先 wrap）。
        std::vector<double> ub((size_t)(q1 + q2), 0.0);
        for (int64_t i = 0; i < q1; ++i) {
            double v = cfp[i] + dfp[i] - xsp[first_off + i];
            if ((i % bs) == 1 || (i % bs) == 5) v = wrap_pi(v);
            ub[(size_t)i] = v / d[(size_t)(first_off + i)];
        }
        for (int64_t i = 0; i < q2; ++i) {
            double v = clp[i] + dlp[i] - xsp[last_off + i];
            if ((i % bs) == 1 || (i % bs) == 5) v = wrap_pi(v);
            ub[(size_t)(q1 + i)] = v / d[(size_t)(last_off + i)];
        }
        std::vector<int64_t> ppos((size_t)ncol, -1);
        for (int64_t j = 0; j < nfirst; ++j)
            for (int64_t a = 0; a < bs; ++a)
                ppos[(size_t)(idp[j] * bs + a)] = j * bs + a;
        for (int64_t j = 0; j < nids - nfirst; ++j)
            for (int64_t a = 0; a < bs; ++a)
                ppos[(size_t)(idp[nfirst + j] * bs + a)] = q1 + j * bs + a;
        const double *gvp = (const double *)ga.ptr;
        std::vector<double> rhs((size_t)ncol, 0.0);
        for (int64_t i = 0; i < ncol; ++i) {
            if (ppos[(size_t)i] >= 0) continue;
            double s = gvp[i];
            for (int64_t p = hA.rp[(size_t)i]; p < hA.rp[(size_t)i + 1]; ++p) {
                const int64_t j = hA.ci[(size_t)p];
                const int64_t pj = ppos[(size_t)j];
                if (pj >= 0) s -= hA.v[(size_t)p] * ub[(size_t)pj];
            }
            rhs[(size_t)i] = s;
        }
        auto xt = torch::from_blob(rhs.data(), {1, ncol}, optsD)
                      .clone().to(dev);
        nd.solveL_upto_gpu(xt, span);
        for (int64_t p = 0; p < ncol; ++p)
            if (ppos[(size_t)p] >= 0) xt[0][p] = 0.0;
        nd.solveLt_upto_gpu(xt, span);
        for (int64_t p = 0; p < ncol; ++p)
            if (ppos[(size_t)p] >= 0) xt[0][p] = 0.0;
        auto uic = xt[0].to(torch::kCPU).contiguous();
        const double *uip = uic.data_ptr<double>();
        std::vector<double> ufull((size_t)ncol, 0.0);
        for (int64_t p = 0; p < ncol; ++p)
            ufull[(size_t)p] = (ppos[(size_t)p] >= 0)
                                   ? ub[(size_t)ppos[(size_t)p]] : uip[p];
        std::vector<double> xsm((size_t)ncol, 0.0);
        for (int64_t p = 0; p < ncol; ++p) {
            double v = xsp[p] + d[(size_t)p] * ufull[(size_t)p];
            if ((p % bs) == 1 || (p % bs) == 5) {
                v = std::fmod(v, 2.0 * M_PI);
                if (v < 0.0) v += 2.0 * M_PI;
            }
            xsm[(size_t)p] = v;
        }
        rts_xo = py::array_t<double>({(py::ssize_t)ncol});
        std::memcpy(rts_xo.mutable_data(), xsm.data(),
                    sizeof(double) * (size_t)ncol);
        rts_dfo = py::array_t<double>({(py::ssize_t)q1});
        rts_dlo = py::array_t<double>({(py::ssize_t)q2});
        std::memcpy(rts_dfo.mutable_data(), dfp, sizeof(double) * (size_t)q1);
        std::memcpy(rts_dlo.mutable_data(), dlp, sizeof(double) * (size_t)q2);
    }
    py::array_t<double> xo({(py::ssize_t)ncol});
    py::array_t<double> Po({(py::ssize_t)q2, (py::ssize_t)q2});
    py::array_t<double> Pfo({(py::ssize_t)q1, (py::ssize_t)q1});
    py::array_t<double> co({(py::ssize_t)q2});
    py::array_t<double> cfo({(py::ssize_t)q1});
    std::memcpy(xo.mutable_data(), xorig.data_ptr<double>(), sizeof(double) * (size_t)ncol);
    std::memcpy(Po.mutable_data(), Porig.data_ptr<double>(), sizeof(double) * (size_t)q2 * q2);
    std::memcpy(Pfo.mutable_data(), Pf_orig.data_ptr<double>(), sizeof(double) * (size_t)q1 * q1);
    std::memcpy(co.mutable_data(), corig.data_ptr<double>(), sizeof(double) * (size_t)q2);
    std::memcpy(cfo.mutable_data(), cf_orig.data_ptr<double>(), sizeof(double) * (size_t)q1);
    py::dict out;
    out["x"] = xo; out["Pnext"] = Po; out["center"] = co;
    out["Pfirst"] = Pfo; out["center_first"] = cfo;
    if (rts_apply) {
        out["rts_x"] = rts_xo;
        out["rts_df"] = rts_dfo;
        out["rts_dl"] = rts_dlo;
        if (rts_gain) {
            auto kfc = rts_kf_t.contiguous();
            auto klc = rts_kl_t.contiguous();
            py::array_t<double> kfo({(py::ssize_t)q1, (py::ssize_t)q2});
            py::array_t<double> klo({(py::ssize_t)q1, (py::ssize_t)q2});
            std::memcpy(kfo.mutable_data(), kfc.data_ptr<double>(),
                        sizeof(double) * (size_t)q1 * q2);
            std::memcpy(klo.mutable_data(), klc.data_ptr<double>(),
                        sizeof(double) * (size_t)q1 * q2);
            out["rts_kf"] = kfo; out["rts_kl"] = klo;
        }
    }
    if (do_gain) {
        // RTS 增益 G = (A_data+P)^{-1} C_data = Yc（scaled）；
        // 前向已经算过，T3 直接复用，省掉每窗 probe + 稠密 solve。
        auto Gc = Yc.to(torch::kCPU).contiguous();
        py::array_t<double> gg({(py::ssize_t)q1, (py::ssize_t)q2});
        std::memcpy(gg.mutable_data(), Gc.data_ptr<double>(),
                    sizeof(double) * (size_t)q1 * q2);
        out["gain"] = gg;
    }
    auto sec = [](auto a, auto b) {
        return std::chrono::duration<double>(b - a).count();
    };
    out["span"] = span; out["cache_hit"] = cache_hit;
    out["setup_s"] = sec(t0, t1);
    out["factor_upto_s"] = sec(t1, t2);
    out["block_s"] = sec(t2, t3);
    out["marginal_s"] = sec(t3, t4);
    out["factor_from_s"] = sec(t4, t5);
    out["solve_s"] = sec(t5, t6);
    out["factor_mb"] = (double)nd.data_size() * 8.0 / 1e6;
    if (do_cov) {
        // 全因子必须完整；port_ldl 路径没有调用 factor_from，这里补一次。
        if (port_ldl) { nd.factor_from(span); cudaDeviceSynchronize(); }
        full_factored = true;
        // 全因子已由 factor_from 完成；Takahashi 取逐 6×6 逆对角块。
        // factor 是 H'=D H D，故 P=D P' D；raw 为 span 序，用 param_to_span 还原。
        auto perm = nd.param_to_span().to(torch::kCPU).contiguous();
        auto raw = nd.compute_diag_blocks(bs).to(torch::kCPU).contiguous();
        const int64_t nb = ncol / bs;
        auto pv = perm.data_ptr<int64_t>();
        auto rv = raw.data_ptr<double>();
        py::array_t<double> dp({(py::ssize_t)nb, (py::ssize_t)bs,
                                (py::ssize_t)bs});
        auto dv = dp.mutable_data();
        auto dtc = dt.to(torch::kCPU).contiguous();
        auto dvv = dtc.data_ptr<double>();
        for (int64_t i = 0; i < nb; ++i) {
            int64_t s = pv[i];
            for (int64_t a = 0; a < bs; ++a)
                for (int64_t b = 0; b < bs; ++b)
                    dv[(i * bs + a) * bs + b] =
                        rv[(s * bs + a) * bs + b] * dvv[i * bs + a] *
                        dvv[i * bs + b];
        }
        out["diagP"] = dp;
    }
    // 指定节点的每星 6×6 后验。内部节点用精确 Schur 公式
    //   P_II = H_II^{-1} + (H_II^{-1} H_IP) S^{-1} (H_II^{-1} H_IP)^T,
    // S 为端口 Schur。H_II 无稠密 prior，内部分解和多 RHS 解都只在
    // sparse interior 上做；稠密端口部分只做 S^{-1} 与小规模 GEMM。
    // 这避免了对稠密端口列做 O(N^3) 的全因子 block Takahashi。
    // 坐标：factor 是 H'=D H D，最后 P = D P' D。
    // 指定节点的每星 6×6 后验。H 在节点维上是块三对角；直接对 q×q
    // 节点块做前向 Schur + 后向 selected-inverse，避免 2q 个右端的
    // 稀疏回代（实测在大窗口上过慢）和稠密端口 block Takahashi。
    //   S_0=H_00, S_k=H_kk-H_{k,k-1} S_{k-1}^{-1} H_{k-1,k};
    //   P_{W-1}=S_{W-1}^{-1},
    //   P_k=S_k^{-1}+S_k^{-1} H_{k,k+1} P_{k+1} H_{k+1,k} S_k^{-1}.
    // 坐标：factor/H 是 D·H·D，最后 P = D P' D。
    if (n_cov_nodes > 0) {
        auto tc0 = std::chrono::steady_clock::now();
        const bool cov_dbg = std::getenv("FLS_COV_DBG") != nullptr;
        auto _el = [&](const char *tag, double t) {
            if (cov_dbg) fprintf(stderr, "[cov_nodes] %s %.3fs\n", tag, t);
        };
        const int64_t qn = nfirst * bs;                  // 每个节点的变量数 6N
        const int64_t nb = nfirst;                       // 每个节点的 6×6 块数 N
        const int64_t nnode = ncol / qn;                 // 窗内节点数 W
        TORCH_CHECK(qn * nnode == ncol, "cov_nodes: 节点块维数不符");
        auto dt_cpu = dt.to(torch::kCPU).contiguous();
        const double *dvv = dt_cpu.data_ptr<double>();
        auto out_t = torch::zeros({n_cov_nodes, nb, bs, bs}, optsD);
        double *outp = out_t.data_ptr<double>();
        // 先判断跨节点块是否逐星块对角（GVE dyn 耦合是逐星的）。
        bool bdiag = true;
        for (int64_t k = 0; k + 1 < nnode && bdiag; ++k) {
            const int64_t b0 = k * qn, b1 = (k + 1) * qn;
            for (int64_t i = 0; i < qn && bdiag; ++i) {
                for (int64_t pp = hA.rp[(size_t)(b0 + i)];
                     pp < hA.rp[(size_t)(b0 + i + 1)]; ++pp) {
                    const int64_t j = hA.ci[(size_t)pp];
                    if (j >= b1 && j < b1 + qn && (i / bs) != ((j - b1) / bs))
                        bdiag = false;
                }
            }
        }
        std::vector<torch::Tensor> Hd((size_t)nnode);
        std::vector<torch::Tensor> Hob, Ho;
        if (bdiag) {
            Hob.resize((size_t)std::max<int64_t>(0, nnode - 1));
            for (int64_t k = 0; k + 1 < nnode; ++k)
                Hob[(size_t)k] = torch::zeros({nb, bs, bs}, optsD);
        } else {
            Ho.resize((size_t)std::max<int64_t>(0, nnode - 1));
            for (int64_t k = 0; k + 1 < nnode; ++k)
                Ho[(size_t)k] = torch::zeros({qn, qn}, optsD);
        }
        for (int64_t k = 0; k < nnode; ++k)
            Hd[(size_t)k] = torch::zeros({qn, qn}, optsD);
        for (int64_t k = 0; k < nnode; ++k) {
            auto &Dk = Hd[(size_t)k];
            double *pd = Dk.data_ptr<double>();
            const int64_t base = k * qn;
            for (int64_t i = 0; i < qn; ++i) {
                for (int64_t pp = hA.rp[(size_t)(base + i)];
                     pp < hA.rp[(size_t)(base + i + 1)]; ++pp) {
                    const int64_t j = hA.ci[(size_t)pp] - base;
                    if (j >= 0 && j < qn) pd[i * qn + j] += hA.v[(size_t)pp];
                }
            }
            if (k + 1 < nnode) {
                const int64_t base1 = (k + 1) * qn;
                if (bdiag) {
                    auto &Uk = Hob[(size_t)k];
                    double *pu = Uk.data_ptr<double>();
                    const int64_t qb = nb * bs * bs;
                    for (int64_t i = 0; i < qn; ++i) {
                        const int64_t s = i / bs, a = i % bs;
                        for (int64_t pp = hA.rp[(size_t)(base + i)];
                             pp < hA.rp[(size_t)(base + i + 1)]; ++pp) {
                            const int64_t j = hA.ci[(size_t)pp] - base1;
                            if (j >= 0 && j < qn)
                                pu[s * bs * bs + a * bs + (j % bs)] +=
                                    hA.v[(size_t)pp];
                        }
                    }
                } else {
                    auto &Uk = Ho[(size_t)k];
                    double *pu = Uk.data_ptr<double>();
                    for (int64_t i = 0; i < qn; ++i) {
                        for (int64_t pp = hA.rp[(size_t)(base + i)];
                             pp < hA.rp[(size_t)(base + i + 1)]; ++pp) {
                            const int64_t j = hA.ci[(size_t)pp] - base1;
                            if (j >= 0 && j < qn) pu[i * qn + j] += hA.v[(size_t)pp];
                        }
                    }
                }
            }
        }
        {
            TORCH_CHECK(qp == qn, "cov_nodes: prior 块与节点块不同维");
            double *pd = Hd[0].data_ptr<double>();
            for (int64_t i = 0; i < qp; ++i)
                for (int64_t pp = prp[i]; pp < prp[i + 1]; ++pp)
                    pd[i * qn + pci[pp]] += pval[pp] * dvv[i] * dvv[pci[pp]];
            Hd[0] = 0.5 * (Hd[0] + Hd[0].transpose(0, 1));
        }
        std::vector<torch::Tensor> Hdg((size_t)nnode);
        for (int64_t k = 0; k < nnode; ++k) Hdg[(size_t)k] = Hd[(size_t)k].to(dev);
        Hd.clear();
        if (bdiag) {
            for (auto &t : Hob) t = t.to(dev);
        } else {
            for (auto &t : Ho) t = t.to(dev);
        }
        cudaDeviceSynchronize();
        _el("blocks", std::chrono::duration<double>(
            std::chrono::steady_clock::now() - tc0).count());
        auto _chol_inv = [&](const torch::Tensor &S) {
            auto Sj = S.clone();
            auto dg = torch::diagonal(Sj);
            auto jit = 1e-10 * dg.abs().max().clamp_min(1e-30);
            Sj += jit * torch::eye(qn, optsD.device(dev));
            auto L = torch::linalg_cholesky(Sj);
            auto Pi = torch::cholesky_inverse(L);
            return 0.5 * (Pi + Pi.transpose(0, 1));
        };
        std::vector<torch::Tensor> Sinv((size_t)nnode), Pm((size_t)nnode);
        Sinv[0] = _chol_inv(Hdg[0]);
        for (int64_t k = 1; k < nnode; ++k) {
            torch::Tensor S;
            if (bdiag) {
                auto A = torch::matmul(Hob[(size_t)(k - 1)].transpose(1, 2),
                                       Sinv[(size_t)(k - 1)].view({nb, bs, qn}));
                auto Av = A.reshape({qn, qn});
                auto B = torch::matmul(
                    Av.view({qn, nb, bs}).permute({1, 0, 2}),
                    Hob[(size_t)(k - 1)]).permute({1, 0, 2}).reshape({qn, qn});
                S = Hdg[(size_t)k] - B;
            } else {
                S = Hdg[(size_t)k]
                    - torch::matmul(torch::matmul(Ho[(size_t)(k - 1)].transpose(0, 1),
                                                  Sinv[(size_t)(k - 1)]),
                                    Ho[(size_t)(k - 1)]);
            }
            S = 0.5 * (S + S.transpose(0, 1));
            Sinv[(size_t)k] = _chol_inv(S);
        }
        cudaDeviceSynchronize();
        _el("forward", std::chrono::duration<double>(
            std::chrono::steady_clock::now() - tc0).count());
        Pm[(size_t)(nnode - 1)] = Sinv[(size_t)(nnode - 1)];
        const auto _idx = torch::arange(nb, optsI.device(dev));
        for (int64_t k = nnode - 2; k >= 0; --k) {
            torch::Tensor T;
            if (bdiag) {
                // C = H P Hᵀ 仍是稠密 q×q（P 稠密、跨星耦合）；用逐星
                // 块对角的 H 把 H P 与 (H P) Hᵀ 分别降到 O(q N b²)。
                auto A = torch::matmul(Hob[(size_t)k],
                                       Pm[(size_t)(k + 1)].view({nb, bs, qn}));
                auto Av = A.reshape({qn, qn});
                auto C = torch::matmul(
                    Av.view({qn, nb, bs}).permute({1, 0, 2}),
                    Hob[(size_t)k].transpose(1, 2))
                             .permute({1, 0, 2}).reshape({qn, qn});
                T = torch::matmul(torch::matmul(Sinv[(size_t)k], C),
                                  Sinv[(size_t)k]);
            } else {
                auto A = torch::matmul(torch::matmul(Sinv[(size_t)k],
                                                     Ho[(size_t)k]),
                                       Pm[(size_t)(k + 1)]);
                T = torch::matmul(torch::matmul(A, Ho[(size_t)k].transpose(0, 1)),
                                  Sinv[(size_t)k]);
            }
            auto Pk = Sinv[(size_t)k] + T;
            Pm[(size_t)k] = 0.5 * (Pk + Pk.transpose(0, 1));
        }
        cudaDeviceSynchronize();
        _el("backward", std::chrono::duration<double>(
            std::chrono::steady_clock::now() - tc0).count());
        for (int64_t ci = 0; ci < n_cov_nodes; ++ci) {
            const int64_t node = cov_nodes[ci];
            if (node < 0 || node >= nnode)
                throw std::runtime_error(
                    "baspacho_partial_solve_prior: cov_nodes 越界");
            const int64_t off = node * qn;
            auto Pmc = Pm[(size_t)node].to(torch::kCPU).contiguous();
            const double *pb = Pmc.data_ptr<double>();
            for (int64_t s = 0; s < nb; ++s) {
                Eigen::Matrix<double, 6, 6> P = Eigen::Matrix<double, 6, 6>::Zero();
                for (int a = 0; a < bs; ++a)
                    for (int b = 0; b < bs; ++b)
                        P(a, b) = pb[(s * bs + a) * qn + (s * bs + b)] *
                                  dvv[off + s * bs + a] *
                                  dvv[off + s * bs + b];
                P = 0.5 * (P + P.transpose());
                for (int a = 0; a < bs; ++a)
                    for (int b = 0; b < bs; ++b)
                        outp[((ci * nb + s) * bs + a) * bs + b] = P(a, b);
            }
        }
        py::array_t<double> ncv({(py::ssize_t)n_cov_nodes, (py::ssize_t)nb,
                                 (py::ssize_t)bs, (py::ssize_t)bs});
        std::memcpy(ncv.mutable_data(), out_t.data_ptr<double>(),
                    sizeof(double) * (size_t)out_t.numel());
        out["node_cov"] = ncv;
        auto tc1 = std::chrono::steady_clock::now();
        out["cov_nodes_s"] = std::chrono::duration<double>(tc1 - tc0).count();
    }
    out["finite"] = bool(torch::isfinite(gt).all().item<bool>() &&
                         torch::isfinite(Pnext).all().item<bool>() &&
                         torch::isfinite(Pfirst).all().item<bool>());
    return out;
}

// ----
// 前向端口求解（默认路径，不带 RTS）
static py::dict baspacho_partial_solve_prior(
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> H_rp,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> H_ci,
        py::array_t<double, py::array::c_style | py::array::forcecast> H_val,
        int64_t ncol,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> ids_arr,
        py::array_t<double, py::array::c_style | py::array::forcecast> g_arr,
        int64_t bs,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> P_rp,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> P_ci,
        py::array_t<double, py::array::c_style | py::array::forcecast> P_val,
        int64_t prior_off,
        py::object scale_obj,
        py::object P2_rp_obj, py::object P2_ci_obj, py::object P2_val_obj,
        int64_t prior2_off, int64_t do_cov, int64_t need_pfirst,
        int64_t port_ldl, int64_t do_gain, py::object cov_nodes_obj) {
    return _partial_solve_impl(H_rp, H_ci, H_val, ncol, ids_arr, g_arr, bs, P_rp, P_ci, P_val, prior_off, scale_obj, P2_rp_obj, P2_ci_obj, P2_val_obj, prior2_off, do_cov, need_pfirst, port_ldl, do_gain, cov_nodes_obj, py::none(), py::none(), py::none(),
                              py::none(), py::none(), py::none(), 0);
}

// 端口求解 + RTS 后向平滑（仅 Python 显式调用，--kf9-backward 1）
static py::dict baspacho_partial_solve_rts(
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> H_rp,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> H_ci,
        py::array_t<double, py::array::c_style | py::array::forcecast> H_val,
        int64_t ncol,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> ids_arr,
        py::array_t<double, py::array::c_style | py::array::forcecast> g_arr,
        int64_t bs,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> P_rp,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast> P_ci,
        py::array_t<double, py::array::c_style | py::array::forcecast> P_val,
        int64_t prior_off,
        py::object scale_obj,
        py::object P2_rp_obj, py::object P2_ci_obj, py::object P2_val_obj,
        int64_t prior2_off, int64_t do_cov, int64_t need_pfirst,
        int64_t port_ldl, int64_t do_gain, py::object cov_nodes_obj,
        py::object rts_phi_obj, py::object rts_q_obj,
        py::object rts_delta_obj, py::object rts_cf_obj,
        py::object rts_cl_obj, py::object rts_xs_obj,
        int64_t rts_gain) {
    return _partial_solve_impl(H_rp, H_ci, H_val, ncol, ids_arr, g_arr, bs, P_rp, P_ci, P_val, prior_off, scale_obj, P2_rp_obj, P2_ci_obj, P2_val_obj, prior2_off, do_cov, need_pfirst, port_ldl, do_gain, cov_nodes_obj, rts_phi_obj, rts_q_obj,
                              rts_delta_obj, rts_cf_obj, rts_cl_obj,
                              rts_xs_obj, rts_gain);
}
// 精确接口 Schur 算子（矩阵自由，不 materialize 2q×2q）----
static CsrSub _extract_csr(
        const int64_t *rp, const int64_t *ci, const double *hv, int64_t ncol,
        const std::vector<int64_t> &rows, const std::vector<int64_t> &cols,
        const std::vector<double> &d) {
    CsrSub out;
    out.rows = (int64_t)rows.size();
    out.cols = (int64_t)cols.size();
    std::vector<int64_t> cmap((size_t)ncol, -1);
    for (int64_t j = 0; j < out.cols; ++j) cmap[(size_t)cols[j]] = j;
    out.rp.assign((size_t)out.rows + 1, 0);
    for (int64_t i = 0; i < out.rows; ++i) {
        int64_t r = rows[i];
        for (int64_t p = rp[r]; p < rp[r + 1]; ++p) {
            int64_t cj = cmap[(size_t)ci[p]];
            if (cj >= 0) {
                out.ci.push_back(cj);
                out.v.push_back(hv[p] * d[(size_t)r] * d[(size_t)ci[p]]);
            }
        }
        out.rp[i + 1] = (int64_t)out.ci.size();
    }
    return out;
}

static void _csr_mm(const CsrSub &A, const double *X, int64_t xr,
                    int64_t r, std::vector<double> &Y, bool subtract=false) {
    for (int64_t i = 0; i < A.rows; ++i)
        for (int64_t p = A.rp[i]; p < A.rp[i + 1]; ++p) {
            int64_t c = A.ci[p];
            double a = A.v[p];
            for (int64_t b = 0; b < r; ++b) {
                double &dst = Y[(size_t)i * r + b];
                if (subtract) dst -= a * X[(size_t)c * r + b];
                else dst += a * X[(size_t)c * r + b];
            }
        }
}

class InterfaceSchur {
   public:
    InterfaceSchur(
            py::array_t<int64_t, py::array::c_style | py::array::forcecast> H_rp,
            py::array_t<int64_t, py::array::c_style | py::array::forcecast> H_ci,
            py::array_t<double, py::array::c_style | py::array::forcecast> H_val,
            int64_t ncol, int64_t off, int64_t q, py::object scale_obj) {
        auto rp_a = H_rp.request(); auto ci_a = H_ci.request();
        auto val_a = H_val.request();
        if (rp_a.size != ncol + 1 || ci_a.size != val_a.size)
            throw std::runtime_error("InterfaceSchur: H CSR 形状不符");
        if (q <= 0 || off <= q || off + q > ncol)
            throw std::runtime_error("InterfaceSchur: off/q 越界");
        ncol_ = ncol; off_ = off; q_ = q;
        const int64_t *rp = (const int64_t *)rp_a.ptr;
        const int64_t *ci = (const int64_t *)ci_a.ptr;
        const double *hv = (const double *)val_a.ptr;
        std::vector<double> d((size_t)ncol, 1.0);
        if (!scale_obj.is_none()) {
            auto sc = scale_obj.cast<
                py::array_t<double, py::array::c_style | py::array::forcecast>>();
            auto sa = sc.request();
            if (sa.size != ncol)
                throw std::runtime_error("InterfaceSchur: scale 长度 != ncol");
            std::memcpy(d.data(), sa.ptr, sizeof(double) * (size_t)ncol);
        }
        // E = 首节点 + 末节点，I = 中间节点。
        std::vector<int64_t> E, I, L, R;
        for (int64_t i = 0; i < q; ++i) { E.push_back(i); L.push_back(i); }
        for (int64_t i = q; i < off; ++i) I.push_back(i);
        for (int64_t i = off; i < off + q; ++i) { E.push_back(i); R.push_back(i); }
        nI_ = (int64_t)I.size();
        if (nI_ <= 0) throw std::runtime_error("InterfaceSchur: 内部维数为 0");
        hEE_ = _extract_csr(rp, ci, hv, ncol, E, E, d);
        hEI_ = _extract_csr(rp, ci, hv, ncol, E, I, d);
        hIE_ = _extract_csr(rp, ci, hv, ncol, I, E, d);
        CsrSub hII = _extract_csr(rp, ci, hv, ncol, I, I, d);
        auto optsD = torch::TensorOptions().dtype(torch::kFloat64);
        auto optsI = torch::TensorOptions().dtype(torch::kInt64);
        auto dev = torch::kCUDA;
        auto rp_t = torch::from_blob(hII.rp.data(), {(int64_t)hII.rp.size()}, optsI)
                        .clone().to(torch::kInt32).to(dev);
        auto ci_t = torch::from_blob(hII.ci.data(), {(int64_t)hII.ci.size()}, optsI)
                        .clone().to(torch::kInt32).to(dev);
        auto v_t = torch::from_blob(hII.v.data(), {(int64_t)hII.v.size()}, optsD)
                       .clone().to(dev);
        lu_ = std::make_unique<CusolverLUSolver>(1, nI_, rp_t, ci_t, AMD);
        auto sing = lu_->factor(v_t.reshape({1, v_t.numel()}));
        if (!sing.empty() && sing[0] >= 0)
            throw std::runtime_error("InterfaceSchur: H_II 奇异 at " +
                                     std::to_string(sing[0]));
    }

    py::array_t<double> apply(
            py::array_t<double, py::array::c_style | py::array::forcecast> X) {
        auto xa = X.request();
        if (xa.ndim != 2 || xa.shape[0] != 2 * q_)
            throw std::runtime_error("InterfaceSchur.apply: X 需 (2q,r)");
        const int64_t r = (int64_t)xa.shape[1];
        const double *xp = (const double *)xa.ptr;
        // W = H_IE X
        std::vector<double> W((size_t)nI_ * r, 0.0);
        _csr_mm(hIE_, xp, 2 * q_, r, W, false);
        auto optsD = torch::TensorOptions().dtype(torch::kFloat64);
        auto B = torch::zeros({1, r * nI_}, optsD);
        {
            auto ba = B.accessor<double, 2>();
            for (int64_t b = 0; b < r; ++b)
                for (int64_t i = 0; i < nI_; ++i)
                    ba[0][b * nI_ + i] = W[(size_t)i * r + b];
        }
        B = B.to(torch::kCUDA);
        lu_->solve_multi(B, r);
        auto Y = B[0].view({r, nI_}).transpose(0, 1).contiguous()
                     .to(torch::kCPU).contiguous();
        const double *yp = Y.data_ptr<double>();
        std::vector<double> Z((size_t)2 * q_ * r, 0.0);
        _csr_mm(hEE_, xp, 2 * q_, r, Z, false);
        _csr_mm(hEI_, yp, nI_, r, Z, true);
        py::array_t<double> out({(py::ssize_t)(2 * q_), (py::ssize_t)r});
        std::memcpy(out.mutable_data(), Z.data(),
                    sizeof(double) * (size_t)2 * q_ * r);
        return out;
    }

    py::tuple dims() const { return py::make_tuple(q_, nI_, 2 * q_); }

   private:
    int64_t ncol_ = 0, off_ = 0, q_ = 0, nI_ = 0;
    CsrSub hEE_, hEI_, hIE_;
    std::unique_ptr<CusolverLUSolver> lu_;
};

PYBIND11_MODULE(qoefls, m) {
    m.doc() = "FLS 固定滞后平滑 C++ 核心（factor/滑窗/Schur 边缘化）";
    m.def("set_pole", [](double x, double y, double z){ setPole(x, y, z); kep3::setPole(x, y, z); },
          py::arg("x"), py::arg("y"), py::arg("z"),
          "设置当日真极（ICRF 单位向量），覆盖本模块 CPU/CUDA 的 zonal 力模型");
    m.def("fls_probe", &fls_probe, py::arg("n"), "torch+BaSpaCho 链接探针");
    m.def("boundary_info_csr", &boundary_info_csr,
          py::arg("H_rp"), py::arg("H_ci"), py::arg("H_val"),
          py::arg("ncol"), py::arg("off"), py::arg("q"),
          "对 H CSR 的指定节点做多右端 solve_multi，返回边界信息 Λ_b (q,q)（CUDA）");
    m.def("boundary_probe_csr", &boundary_probe_csr,
          py::arg("H_rp"), py::arg("H_ci"), py::arg("H_val"), py::arg("Omega"),
          py::arg("ncol"), py::arg("off"), py::arg("q"),
          "对 H CSR 的指定节点做 r 个右端的 solve_multi，返回 P_b·Omega (q,r)（CUDA）");
    m.def("boundary_lowrank_csr", &boundary_lowrank_csr,
          py::arg("H_rp"), py::arg("H_ci"), py::arg("H_val"), py::arg("D"),
          py::arg("Omega"), py::arg("ncol"), py::arg("off"), py::arg("q"),
          py::arg("cap_rel") = 1e6, py::arg("rank_out") = 0,
          "P2 边界：CUDA Schur probe + 逐星 D^{-1} + 低秩 C（含尺度门禁），"
          "rank_out>0 时在同一 Hss 分解上构造 gauge-Krylov 基；"
          "返回 dict(DLa,U,C,info,alpha,raw_max,cap,n_capped,gmax)");
    m.def("baspacho_partial_solve", &baspacho_partial_solve,
          py::arg("H_rp"), py::arg("H_ci"), py::arg("H_val"), py::arg("ncol"),
          py::arg("ids"), py::arg("g"), py::arg("bs") = 6,
          "端口因子窗口求解：H 已含 prior，ids=首/末节点块；返回 x, Pnext, center");
    m.def("baspacho_partial_solve_prior", &baspacho_partial_solve_prior,
          py::arg("H_rp"), py::arg("H_ci"), py::arg("H_val"), py::arg("ncol"),
          py::arg("ids"), py::arg("g"), py::arg("bs"),
          py::arg("P_rp"), py::arg("P_ci"), py::arg("P_val"),
          py::arg("prior_off"), py::arg("scale") = py::none(),
          py::arg("P2_rp") = py::none(), py::arg("P2_ci") = py::none(),
          py::arg("P2_val") = py::none(), py::arg("prior2_off") = 0,
          py::arg("do_cov") = 0, py::arg("need_pfirst") = 0,
          py::arg("port_ldl") = 1, py::arg("do_gain") = 0,
          py::arg("cov_nodes") = py::none(),
          "端口因子求解：H 与 prior 块分开传入，C++ 内嵌 prior 并缩放；"
          "port_ldl=1 用端口块 LDL + 内部因子回代替代 factorFrom；"
          "need_pfirst=1 才计算首端口边际；do_cov 返回逐 6×6 后验对角块；"
          "cov_nodes=[node,...] 用块三对角 selected inverse 返回这些节点的逐星 6×6 后验");
        m.def("baspacho_partial_solve_rts", &baspacho_partial_solve_rts,
          py::arg("H_rp"), py::arg("H_ci"), py::arg("H_val"), py::arg("ncol"),
          py::arg("ids"), py::arg("g"), py::arg("bs"),
          py::arg("P_rp"), py::arg("P_ci"), py::arg("P_val"),
          py::arg("prior_off"), py::arg("scale") = py::none(),
          py::arg("P2_rp") = py::none(), py::arg("P2_ci") = py::none(),
          py::arg("P2_val") = py::none(), py::arg("prior2_off") = 0,
          py::arg("do_cov") = 0, py::arg("need_pfirst") = 0,
          py::arg("port_ldl") = 1, py::arg("do_gain") = 0,
          py::arg("cov_nodes") = py::none(),
          py::arg("rts_phi"), py::arg("rts_q"), py::arg("rts_delta"),
          py::arg("rts_cf"), py::arg("rts_cl"), py::arg("rts_xs"),
          py::arg("rts_gain") = 0,
          "端口求解 + RTS 后向平滑：仅显式调用时启用，默认路径用 prior 入口");
m.def("partial_interior_solve", &partial_interior_solve,
          py::arg("H_rp"), py::arg("H_ci"), py::arg("H_val"), py::arg("ncol"),
          py::arg("ids"), py::arg("rhs"), py::arg("bs") = 6,
          "内部条件解：factorUpTo + GPU solveL/Lt_upto，返回全向量");
    m.def("partial_tri_probe", &partial_tri_probe,
          py::arg("H_rp"), py::arg("H_ci"), py::arg("H_val"),
          py::arg("ncol"), py::arg("span"), py::arg("bs") = 6,
          "GPU solveL/Lt_upto vs CPU 对拍");
    m.def("baspacho_partial_probe", &baspacho_partial_probe,
          py::arg("H_rp"), py::arg("H_ci"), py::arg("H_val"), py::arg("ncol"),
          py::arg("ids"), py::arg("bs") = 6, py::arg("factor_from") = 1,
          "factorUpTo + trailing Schur 探针：ids 为留最后的原始参数块 id；"
          "factor_from=0 只取数据侧 S，不做端口分解");
    m.def("boundary_info_baspacho", &boundary_info_baspacho,
          py::arg("H_rp"), py::arg("H_ci"), py::arg("H_val"),
          py::arg("ncol"), py::arg("off"), py::arg("q"),
          py::arg("scale") = py::none(), py::arg("jitter_rel") = 0.0,
          "精确边界信息：Baspacho 分解 H_mm，解 H_mm X=H_mb，"
          "返回 (Λ_b=H_bb-H_bm X, stats)；scale=D 时在 DHD 坐标计算并还原");
    m.def("solve_csr_csr", &solve_csr_csr,
          py::arg("H_rp"), py::arg("H_ci"), py::arg("H_val"), py::arg("B"),
          "对稀疏 H 做多右端求解 H X=B，返回 X (n,r)（CUDA）");
    py::class_<InterfaceSchur>(m, "InterfaceSchur",
                               "精确接口 Schur 算子：Lambda_EE=H_EE-H_EI H_II^{-1} H_IE，"
                               "矩阵自由，不 materialize 2q×2q")
        .def(py::init<py::array_t<int64_t, py::array::c_style | py::array::forcecast>,
                      py::array_t<int64_t, py::array::c_style | py::array::forcecast>,
                      py::array_t<double, py::array::c_style | py::array::forcecast>,
                      int64_t, int64_t, int64_t, py::object>(),
             py::arg("H_rp"), py::arg("H_ci"), py::arg("H_val"),
             py::arg("ncol"), py::arg("off"), py::arg("q"),
             py::arg("scale") = py::none())
        .def("apply", &InterfaceSchur::apply, py::arg("X"),
             "X (2q,r) -> Lambda_EE X (2q,r)")
        .def("dims", &InterfaceSchur::dims, "(q, nI, 2q)");
    m.def("woodbury_solve_csr", &woodbury_solve_csr,
          py::arg("H_rp"), py::arg("H_ci"), py::arg("H_val"),
          py::arg("U"), py::arg("C"), py::arg("g"), py::arg("off"), py::arg("q"),
          "solve 级 Woodbury：H0+U C Uᵀ，返回 (delta, Y, Sinv)（cuSOLVER RF）");
    m.def("woodbury_solve_baspacho", &woodbury_solve_baspacho,
          py::arg("H_rp"), py::arg("H_ci"), py::arg("H_val"),
          py::arg("param_size"), py::arg("block_ptr"), py::arg("block_ind"),
          py::arg("U"), py::arg("C"), py::arg("g"), py::arg("off"), py::arg("q"),
          "solve 级 Woodbury：Baspacho CUDA 并行多右端，返回 (delta, Y, Sinv)");
    m.def("fls_baspacho_probe", &fls_baspacho_probe,
          "BaSpaCho C++ 全链探针（add_MtM/factor/solve/compute_diag_blocks vs 稠密）");
    m.def("fls_build_struct", &fls_build_struct, py::arg("n"), py::arg("L"),
          py::arg("edges_i"), py::arg("edges_j"), py::arg("gts"), py::arg("damp"),
          py::arg("prior_node") = -1, py::arg("r_cross") = 1,
          "FLS 滑窗装配结构（CSR + 块图案），等价 py _lift_struct(prior=True)/_lift_block_struct；"
          "r_cross>=2 时块图案加入同历元 ISL 的 r 跳邻域");
    m.def("fls_gauge_basis_window", &fls_gauge_basis_window_cpp,
          py::arg("X"), py::arg("n"), py::arg("L"), py::arg("orth") = true,
          "生产逐节点 gauge 基/Q-R（C++ 6×6 J 逆 + 整窗 QR）");
    m.def("fls_geom_probe", &fls_geom_probe, py::arg("X"),
          "探针：QOE(n,6) → (rv(n,6), ∂(r,v)/∂oe(n,6,6))，直调宿主 C++ 接口");
    m.def("fls_nees_directional", &fls_nees_directional,
          py::arg("e_rv"), py::arg("Pq"), py::arg("Xq"), py::arg("nthreads") = 16,
          "batched RTN/QOE NEES decomposition");
    m.def("fls_local_schur", &fls_local_schur_py, py::arg("H00"), py::arg("H01"),
          py::arg("H11"),
          "批量局部 Schur：H00/H01/H11 (n,m,m) → P_marg[0:6,0:6] (n,6,6)");
    py::class_<FlsWindow>(m, "FlsWindow")
        .def(py::init([](int n, int L, int E, int G, double dt, double s1, double s2,
                         py::array_t<int, py::array::c_style | py::array::forcecast> ei,
                         py::array_t<int, py::array::c_style | py::array::forcecast> ej,
                         py::array_t<int, py::array::c_style | py::array::forcecast> gts,
                         py::array_t<double, py::array::c_style | py::array::forcecast> Q,
                         double damp, bool prior, int prior_node, int dyn_sub, int r_cross) {
            auto ri = ei.request(); auto rj = ej.request();
            auto rg = gts.request(); auto rq = Q.request();
            return new FlsWindow(n, L, E, G, dt, s1, s2,
                                 std::vector<int>((int *)ri.ptr, (int *)ri.ptr + ri.size),
                                 std::vector<int>((int *)rj.ptr, (int *)rj.ptr + rj.size),
                                 std::vector<int>((int *)rg.ptr, (int *)rg.ptr + rg.size),
                                 std::vector<double>((double *)rq.ptr, (double *)rq.ptr + rq.size),
                                 damp, prior, prior_node, dyn_sub, r_cross);
        }), py::return_value_policy::take_ownership,
            py::arg("n"), py::arg("L"), py::arg("E"), py::arg("G"), py::arg("dt"),
            py::arg("sigma_isl"), py::arg("sigma_gts"), py::arg("isl_i"), py::arg("isl_j"),
            py::arg("gts"), py::arg("Q"), py::arg("damp") = 1e-6, py::arg("prior") = true,
            py::arg("prior_node") = -1, py::arg("dyn_sub") = 1, py::arg("r_cross") = 1)
        .def("set_obs", &FlsWindow::set_obs, py::arg("D_isl"), py::arg("W_isl"),
             py::arg("D_gts"), py::arg("W_gts"), py::arg("Q_gts"), py::arg("ctr"))
        .def("set_X", &FlsWindow::set_X, py::arg("X"))
        .def("set_prior", &FlsWindow::set_prior, py::arg("Lp"), py::arg("z_prior"))
        .def("set_P_init", &FlsWindow::set_P_init, py::arg("P"), "L=1 EKF 初始 P")
        .def("set_Q_phys", &FlsWindow::set_Q_phys, py::arg("sigma_a"),
             "物理 GVE dyn：完整切回 q_perstar=1/dyn_sub=KF_DYN_SUB/d_phi=d_phi_/x_ref=null")
        .def("set_Q_phys_arr", &FlsWindow::set_Q_phys_arr, py::arg("Q"), py::arg("sigma_a"))
        .def("get_Q_phys", &FlsWindow::get_Q_phys)
        .def("dyn_mode_diag", &FlsWindow::dyn_mode_diag,
             "物理过程噪声：Q_rv=diag(0.5σa dt², σa dt) → QOE")
        .def("set_prior_multi", &FlsWindow::set_prior_multi, py::arg("Lp"), py::arg("z_prior"),
             "每节点 6×6 先验：Lp (L,n,6,6)、z_prior (L,n,6)，用于关键帧窗口")
        .def("set_prior_cross", &FlsWindow::set_prior_cross,
             py::arg("val"), py::arg("bi"), py::arg("bj"),
             "局部联合边界先验跨块：val(B,6,6) scaled Λ_ij，bi/bj(B,) 变量号")
        .def("set_dyn_ekf", &FlsWindow::set_dyn_ekf, py::arg("x_ref"), py::arg("x_pred"),
             py::arg("phi"), py::arg("P_prev"), py::arg("P_pred"),
             "固定 EKF 区间 dyn：CUDA 内算 Q_eff 与白化，不再调用 GVE")
        .def("solve_batch", &FlsWindow::solve_batch, py::arg("steps") = 5,
             "固定 L 个关键帧 batch GN（不 push/不左移）")
        .def("last_solve_diag", &FlsWindow::last_solve_diag,
             "末次 solve_batch 逐轮位移/代价 dict：iters=[{iter,dx_norm,dx_max,"
             "dx_trial_norm,dx_trial_max,alpha,cost_before,cost_after,"
             "cost_after_refresh,accepted,ls_trials}]，另给 valid/n_iters 与 "
             "last_iter/last_dx_norm/last_dx_max/last_cost_before/last_cost_after/"
             "last_cost_after_refresh/last_accepted/last_ls_trials；未求解时 "
             "valid=false 且 last_* 为 NaN/false")
        .def("set_gauge_prior", &FlsWindow::set_gauge_prior, py::arg("G"), py::arg("Pg"),
             py::arg("z"),
             "窗口级 gauge 先验：G 为物理 QOE（不 QR），信息 G Pg^-1 G^T；"
             "Pg 跨窗传入、不重置；z 为 gauge 中心")
        .def("set_gauge_gts_split", &FlsWindow::set_gauge_gts_split, py::arg("on"),
             "GTS 的 G 子空间信息开关（默认开）：G 分量并入 Pg；本地 A 行始终保留"
             "原始 v，窗口投影 (I-Pi) 负责取 J_perp")
        .def("gauge_gts_split", &FlsWindow::gauge_gts_split)
        .def("set_gts_pg_node_range", &FlsWindow::set_gts_pg_node_range,
             py::arg("first"), py::arg("last"),
             "跨窗 GTS→Pg 节点范围（半开 [first,last)，当前窗口节点 0..L-1），"
             "默认 [0,L)。只有范围内节点的 GTS 行把 G 分量累加进 Pg；"
             "A_val/b 的 GTS 行始终 raw/whole。规则：首窗 [0,L)；有上一窗 "
             "Pg_next 的滑窗用 [L-1,L)，只让新进入的末节点贡献，避免旧节点的 "
             "GTS G 信息重复计数。注意 last 侧范围（如 [L-1,L)）是实验性用法："
             "它会丢掉范围内未覆盖的旧节点 GTS G 信息，仅在已持有上一窗 Pg_next "
             "时成立，不是默认行为。")
        .def("gts_pg_node_range", &FlsWindow::gts_pg_node_range,
             "返回当前 GTS→Pg 节点范围 (first,last)（半开）")
        .def("project_prior_gauge", &FlsWindow::project_prior_gauge, py::arg("G0"),
             "node0 先验精确投影 Λ0→(I-Pi_G0)Λ0(I-Pi_G0)：由 Lp 行构造 A0/B0，"
             "在 solve/marginalize 中加入 rank-2k 修正并修正 Atb/代价")
        .def("project_prior_gauge_householder", &FlsWindow::project_prior_gauge_householder,
             py::arg("G0"), py::arg("max_ncol") = 600, py::arg("gauge_block_mode") = 0,
             "小窗口 Householder 参考路径（n*6<=max_ncol）：G0=Q R、Z=Q^T、"
             "ZΛ0Z^T 的 gauge 块清零(mode 0)/替换为 R Pg^-1 R^T(mode 1)/"
             "(R Pg R^T)^-1(mode 2)，再变回。返回 Lambda0_proj/rank2k 与 doc 门禁；"
             "只做验证，不写回 solve 状态，也不是 N=1152 的默认路径。")
        .def("set_prior_projected", &FlsWindow::set_prior_projected, py::arg("Lp"),
             py::arg("z_prior"), "接受外部已投影好的逐星 node0 先验并置标志；"
             "与 project_prior_gauge 互斥（外部标志阻止二次投影）")
        .def("prior_gauge_projected", &FlsWindow::prior_gauge_projected)
        .def("prior_b0_cond", &FlsWindow::prior_b0_cond,
             "最近一次 project_prior_gauge/Householder 的 B0=G0^T Λ0 G0 条件数；"
             "<0 表示尚未计算")
        .def("prior_b0_min_eig", &FlsWindow::prior_b0_min_eig,
             "最近一次 B0 的最小特征值")
        .def("prior_b0_max_eig", &FlsWindow::prior_b0_max_eig,
             "最近一次 B0 的最大特征值")
        .def("prior_gauge_diagnostics", &FlsWindow::prior_gauge_diagnostics,
             "步骤4 B0 guard 诊断 dict：applied/projected/reason/"
             "B0_cond/B0_min_eig/B0_max_eig")
        .def("gauge_pg_host", &FlsWindow::gauge_pg_host,
             "当前窗口有效 Pg_eff=(Pg^-1+Λ_g)^-1 (k,k)；Λ_g 只含 "
             "gts_pg_node_range 内节点的 GTS G 分量")
        .def("gauge_pginfo_host", &FlsWindow::gauge_pginfo_host,
             "GTS 的 G 分量信息 Λ_g (k,k)，仅 gts_pg_node_range 内节点")
        .def("gauge_gts_fisher_q_host", &FlsWindow::gauge_gts_fisher_q_host,
             "GTS gauge Fisher 的系数空间伪逆 pinv(Λ_g)、rank、λmin/λmax、"
             "cond、valid；rank<k 时 valid=0")
        .def("gauge_xg_host", &FlsWindow::gauge_xg_host,
             "末次 K^-1 G（物理 QOE，ncol×k）")
        .def("gauge_hg_host", &FlsWindow::gauge_hg_host,
             "末次 K G（物理 QOE，ncol×k）")
        .def("gauge_cov_diagnostics", &FlsWindow::gauge_cov_diagnostics,
             "末次 Woodbury 的 P=Hwin^-1 一致性：G^T P G vs Pg_eff、"
             "G^T P_perp、G^T K G（G^T Qeff G 口径）。非正交 G 下用 "
             "P=G S^-1 Pg_eff S^-1 G^T+P_perp, G^T P_perp=0。")
        .def("gauge_woodbury_diag", &FlsWindow::gauge_woodbury_diag,
             "末次 Woodbury 条件诊断 dict：D_cond/D_cond_bal/D_rank/D11_cond/"
             "D11_rank/D22_cond/D22_rank/D_svd_trunc/D_blockdiag/D_orthonormal/"
             "D_prior_proj/D_orth_dev/Dinv_fro/Dinv_hash/S_cond/Pg_eff_cond/"
             "B0_cond/B0_min_eig/B0_max_eig/B0_applied；不改 solve")
        .def("gauge_G_hash", &FlsWindow::gauge_G_hash,
             "当前窗口 G(ncol,k) 的 FNV-1a 内容指纹（与 fej_status 的 G_hash 同源）")
        .def("gauge_x_hash", &FlsWindow::gauge_x_hash,
             "当前 X(L,n,6) 的 FNV-1a 内容指纹（fej_status 的 x_hash 同源）")
        .def("gauge_d_cond_host", &FlsWindow::gauge_d_cond_host,
             "末次 D 的原始条件数 smax/smin；gauge_woodbury_diag 另给 "
             "D_cond_bal（均衡后）")
        .def("gauge_lamwin_host", &FlsWindow::gauge_lamwin_host,
             "D1：当前窗口 Λ_win=(I-Pi)K(I-Pi)+G Pg_eff^-1 G^T 的 gauge 块。"
             "返回 GtLamG(k,k)、lamwin_gg_min/max_eig/cond、Q/R、QtLamQ、"
             "Pg_eff_inv_Q、identity_rel（Q 坐标）。只算 k 维 gauge 块，"
             "不声称 6N 全谱 lambda_min；无稠密 6N 逆。")
        .def("prior_first_x_host", &FlsWindow::prior_first_x_host,
             "D3：set_prior_multi 时刻的 x_first (L,n,6)（FEJ 冻结快照）")
        .def("fej_status", &FlsWindow::fej_status,
             "D3：FEJ 状态 dict：x_first/先验中心与 Lp 是否冻结、G_win 是否在"
             "首次 solve 后变化、最后 marginalize 检查结果与原因")
        .def("marginalize_first", &FlsWindow::marginalize_first,
             "消 node0（保留 node1）→ Lp_host/Pm_host/z_prior_host（下一窗 node0 先验）")
        .def("marginalize_first_local", &FlsWindow::marginalize_first_local, py::arg("r") = 1,
             "r=1/r=2 局部 Schur 消 node0 → 物理 QOE node1 6×6 Pm (n,6,6)")
        .def("marginalize_first_local_gauge", &FlsWindow::marginalize_first_local_gauge,
             py::arg("r") = 1, py::arg("exact_compare") = false,
             "gauge 版边缘化：精确投影+先验下消 node0，返回 (Pm(n,6,6), Pg_next(k,k))")
        .def("P_node_host", &FlsWindow::P_node_host, py::arg("node"),
             "指定节点的每星 6×6 selected-inverse 后验（物理 QOE）")
        .def("solve_rhs", &FlsWindow::solve_rhs, py::arg("U"),
             "解 H X = U（物理 QOE，U=(ncol,k)）→ X=(ncol,k)")
        .def("joint_node_info", &FlsWindow::joint_node_info, py::arg("node"),
             py::arg("eps") = 0.0,
             "精确节点联合信息 Λ=(Σ_node)^-1（Σ_node=H^-1 的节点块；稠密 6N×6N）。"
             "用 solve_multi 全右端，不用 selected inverse；eps>0 为物理对角正则。")
        .def("info_predict", &FlsWindow::info_predict,
             py::arg("Lam"), py::arg("Phi"), py::arg("Q"),
             "含过程噪声的精确信息预测 Λ⁺=(ΦΣΦᵀ+Q)^-1（Σ=Λ^-1；稠密 6N×6N）")
        .def("local_cross_info", &FlsWindow::local_cross_info,
             py::arg("node"), py::arg("center"), py::arg("r") = 1,
             py::arg("eps") = 0.0,
             "r 跳 ISL 邻域的局部联合信息 Λ_S=Σ_S^-1，返回 (center,j) 跨块 "
             "(val(B,6,6), bi(B,), bj(B,))，供 set_prior_cross")
        .def("local_cross_info_global", &FlsWindow::local_cross_info_global,
             py::arg("node"), py::arg("r") = 1, py::arg("eps") = 0.0,
             py::arg("global_shrink") = 0,
             "多中心组装：较小端点拥有每个无序对，一次分解逐中心求 Λ_S；"
             "返回 (val(B,6,6), bi(B,), bj(B,), s_g, diag(n,6,6))；diag 与跨块同源，"
             "global_shrink=1 时乘 s_g 保证对角占优。")
        .def("local_cross_info_partition", &FlsWindow::local_cross_info_partition,
             py::arg("node"), py::arg("r") = 1, py::arg("eps") = 0.0,
             "不相交 r 跳簇：一次多右端取整节点协方差，簇内求逆并 PSD 投影；"
             "返回 (val(B,6,6), bi(B,), bj(B,), diag(n,6,6), cluster_id(n,))，"
             "簇间跨块为零，组装矩阵天然块对角 PSD。")
        .def("augmented_solve", &FlsWindow::augmented_solve,
             py::arg("H_new"), py::arg("H_prev"), py::arg("Phi"), py::arg("Q"),
             py::arg("g"),
             "含 Q 的增广系统 [H_new+Q^-1,B;B^T,-M][δ;y]=[g;0] 的 LU 解（小 N 稠密模式）")
        .def("augmented_solve_sparse", &FlsWindow::augmented_solve_sparse,
             py::arg("Hn_rp"), py::arg("Hn_ci"), py::arg("Hn_val"),
             py::arg("Hp_rp"), py::arg("Hp_ci"), py::arg("Hp_val"),
             py::arg("Phi"), py::arg("Q"), py::arg("g"),
             "含 Q 的增广系统（H_new/H_prev 为 CSR）：支持 N=1152 的块稀疏组装 + LU")
        .def("T_host", &FlsWindow::T_host, "L=1 复合转移 T' (n,6,6)；关键帧处为区间快照 T_{a→b}")
        .def("phi_host", &FlsWindow::phi_host, "最近一次 step 的单步 STM Φ (n,6,6)")
        .def("P_prev_host", &FlsWindow::P_prev_host, "EKF 区间 P_prev (n,6,6)")
        .def("P_pred_host", &FlsWindow::P_pred_host, "EKF 区间 P_pred (n,6,6)")
        .def("Qeff_host", &FlsWindow::Qeff_host, "EKF 区间 Q_eff (n,6,6)")
        .def("Qeff_lmin", &FlsWindow::Qeff_lmin, "EKF 区间 Q_eff 截断前 lambda_min")
        .def("ekf_scale_host", &FlsWindow::ekf_scale_host, "L=1 固定每星缩放 d=√diag(P0) (n,6)")
        .def("Qeff_trace_margin", &FlsWindow::Qeff_trace_margin, "trace(TPT)-trace(Ppred)")
        .def("set_dyn_ekf_qeff", &FlsWindow::set_dyn_ekf_qeff, py::arg("x_ref"), py::arg("x_pred"),
             py::arg("phi"), py::arg("qeff"), "固定 EKF 区间 dyn：直接消费同源 Q_eff")
        .def("set_dyn_gauge_corr", &FlsWindow::set_dyn_gauge_corr,
             py::arg("Ga"), py::arg("Gb"), py::arg("C"),
             "全 N gauge 修正：每区间 Ga/Gb (6N,k)、C(k,k)；dyn 残差扣除 "
             "Gb·C·(Gaᵀ(X_a−x_ref))，等价 T'=T+Gb C Gaᵀ。仅固定 EKF dyn (q_perstar=2)")
        .def("clear_dyn_gauge_corr", &FlsWindow::clear_dyn_gauge_corr,
             "清除全 N gauge 修正，恢复逐星 T=Φ 路径")
        .def("has_dyn_gauge_corr", &FlsWindow::has_dyn_gauge_corr)
        .def("dyn_gauge_diag", &FlsWindow::dyn_gauge_diag,
             "诊断：Gbᵀ T Ga / Gbᵀ T' Ga (k,k)、||Gb C Gaᵀ||_F、当前 X 的 "
             "||corr|| 与 g=Gaᵀ(X_a−x_ref)；不改 solve 状态")
        .def("reset_T", &FlsWindow::reset_T, "T' 重置为单位阵")
        .def("set_skip", &FlsWindow::set_skip, py::arg("on"), "1=关键帧只预测不更新")
        .def("set_ekf_cov_r", &FlsWindow::set_ekf_cov_r, py::arg("r"),
             "L=1 EKF 每帧协方差：0=逐星条件核；1/2=r 邻域局部 Schur")
        .def("ekf_cov_r", &FlsWindow::ekf_cov_r)
        .def("P_node_host_fast", &FlsWindow::P_node_host_fast, py::arg("node"),
             "只算指定节点的精确 selected inverse 块（原始参数顺序）")
        .def("P_node_host_fast_gauge", &FlsWindow::P_node_host_fast_gauge, py::arg("node"),
             "gauge 感知的节点边缘 dict：P(n,6,6)/gauge_applied/min_eig/Pm_rel/"
             "woodbury_ku/Dinv_fro/Dinv_hash/D_cond；与 marginalize_first_local_gauge "
             "同一 Woodbury，无 gauge 时回退 selected inverse")
        .def("P_node1_gauge_check", &FlsWindow::P_node1_gauge_check,
             py::arg("Pm_ref") = py::none(),
             "D4/D5：node1 精确 selected-inverse 对拍。返回同 "
             "P_node_host_fast_gauge 的 P/gauge_applied/min_eig/woodbury 指纹；"
             "若给 Pm_ref(n,6,6) 额外返回 Pm_ref_rel_fro/Pm_ref_rel_max/Pm_ref_abs_max")
        .def("P_all_host", &FlsWindow::P_all_host,
             "全窗口每节点每星 6×6 selected-inverse 后验 (L,n,6,6)")
        .def("marg_probe_dense", &FlsWindow::marg_probe_dense,
             "小 N 诊断：selected inverse / r=0 Schur vs 稠密 H⁻¹")
        .def("set_sweep", &FlsWindow::set_sweep, py::arg("on"),
             "GN 因子/求解改走信息形式逐帧消去（Baspacho 只解 6N 单帧；边际/质量应与全窗一致）")
        .def("sweep", &FlsWindow::sweep, "逐帧消去是否开启")
        .def("set_marg_global", [](FlsWindow &w, bool on){ w.set_marg_global(on); }, py::arg("on"))
        .def("marg_global", &FlsWindow::marg_global)
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
        .def("P_plus_host", &FlsWindow::P_plus_host, "EKF 当帧每星后验 P⁺ (n,6,6)")
        .def("KH_host", &FlsWindow::KH_host, "EKF 当帧 KH=I−J (n,6,6)")
        .def("Jnorm_host", &FlsWindow::Jnorm_host,
             "L=1 每星局部 joint 更新对角块 ||J6||_2 (n)")
        .def("JminSing_host", &FlsWindow::JminSing_host,
             "L=1 每星 T6_stable=ΦJ6_stable 最小奇异值 (n)")
        .def("set_ekf_store_block_T", &FlsWindow::set_ekf_store_block_T, py::arg("on"),
             "诊断开关：保存每星局部球的完整 block-T=Φ_B J_B（m×m），供 "
             "ekf_block_T_host 读取；默认关，生产 T 链仍用 6×6 J6_stable")
        .def("ekf_store_block_T", &FlsWindow::ekf_store_block_T)
        .def("ekf_block_T_host", &FlsWindow::ekf_block_T_host, py::arg("star"),
             "返回 (ball_indices, T_B)（目标星在 ball_indices[0]；T_B 为球内 m×m "
             "block-T，坐标是局部球；需先 set_ekf_store_block_T(True)）。注："
             "T_B 的 Φ 用目标星 Phi 广播到球内每星（现有核只传一个 Phi6），"
             "严格 block-T 需每邻星各自的 Phi_i")
        .def("Lp_host", &FlsWindow::Lp_host, "Λ0 白化因子 Lp (n,6,6)")
        .def("z_prior_host", &FlsWindow::z_prior_host, "先验中心 z_prior (n,6)")
        .def("res2", &FlsWindow::res2,
             "上次装配的残差平方和：step 路径为 ISL res2；solve_batch 路径为全代价"
             "（ISL+GTS+dyn+prior+gauge prior，Σb² 口径）")
        .def("total_cost_host", &FlsWindow::total_cost_host,
             "在当前 X 上现算 total_cost()（ISL+GTS+dyn+prior+gauge prior，与线搜索"
             "同口径；gauge 开启时几何在 Px 上计、gauge 项用 Pg_eff^-1）；"
             "供有限差分/梯度一致性检查，不改 solve 状态")
        .def("cost_gauge_diag", &FlsWindow::cost_gauge_diag,
             "D7 代价分解：total/geom/gauge/prior_proj、||Px-x||、cost_ref_norm、"
             "||G^T x-z||、Pg_eff/Pg 使用标志与 d7_cost_projected；"
             "会按当前 X 刷新一次 total_cost()")
        .def("cost_ref_host", &FlsWindow::cost_ref_host,
             "D7 cost 参考点 (L,n,6)：最近一次 solve_batch 入口的装配/线性化点"
             "（线搜索内不更新）")
        .def("timings", &FlsWindow::timings,
             "单步耗时归因 [pred, geo, asm, gn, schur核, schur D2H, schur 其余, 全步] (s)")
        .def("sc_host", &FlsWindow::sc_host, "GN 末次列均衡因子 sc（调试/参考口径一致用）");
}
