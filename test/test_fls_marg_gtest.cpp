// FLS 逐星边缘核（P1'）的 C++ gtest：合成小例 → 数据落盘 → 与**同算法 host 参考**逐位对拍。
// 目的：把"行表构造/核内 gather"的接线 bug 与"6×6 逆/特征分解的数值差"分离——
// 参考与核使用**完全相同的算法**（Gauss-Jordan + 对称 Jacobi + 同截断口径），
// 因此两者之差只可能来自行表/索引/并行实现；稠密 Eigen 参考另在 py 侧（validate_fls_marg）。
//
// 运行：build/gtest_fls_marg
#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "extlib/baspacho_solver.h"   // 先拉 torch：c10 会覆盖 LOG 宏
#include "fls_lists.h"
#include "qoejopt.h"

#include <glog/logging.h>             // 必须放 torch 之后，glog 的 LOG 才是最终定义

namespace {

using V6 = Eigen::Matrix<double, 6, 1>;
using M6 = Eigen::Matrix<double, 6, 6>;

// ---- 与核相同的算法（host 版）----
void inv6_host(const double *A, double *Ainv) {
    double M[6][12];
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j) { M[i][j] = A[i * 6 + j]; M[i][6 + j] = (i == j) ? 1.0 : 0.0; }
    for (int col = 0; col < 6; ++col) {
        int piv = col; double bv = std::fabs(M[col][col]);
        for (int r = col + 1; r < 6; ++r)
            if (std::fabs(M[r][col]) > bv) { bv = std::fabs(M[r][col]); piv = r; }
        if (piv != col)
            for (int c = 0; c < 12; ++c) std::swap(M[col][c], M[piv][c]);
        double d = M[col][col];
        if (std::fabs(d) < 1e-300) d = (d >= 0 ? 1e-300 : -1e-300);
        for (int c = 0; c < 12; ++c) M[col][c] /= d;
        for (int r = 0; r < 6; ++r) {
            if (r == col) continue;
            double f = M[r][col];
            if (f == 0.0) continue;
            for (int c = 0; c < 12; ++c) M[r][c] -= f * M[col][c];
        }
    }
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j) Ainv[i * 6 + j] = M[i][6 + j];
}
void eigh6_host(const double *A, double *w, double *V) {
    double a[6][6], v[6][6];
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j) { a[i][j] = A[i * 6 + j]; v[i][j] = (i == j) ? 1.0 : 0.0; }
    for (int sweep = 0; sweep < 24; ++sweep) {
        double off = 0.0;
        for (int p = 0; p < 6; ++p)
            for (int q = p + 1; q < 6; ++q) off += a[p][q] * a[p][q];
        if (off < 1e-30) break;
        for (int p = 0; p < 6; ++p)
            for (int q = p + 1; q < 6; ++q) {
                if (std::fabs(a[p][q]) < 1e-300) continue;
                double theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                double t = (theta >= 0 ? 1.0 : -1.0) / (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
                for (int k = 0; k < 6; ++k) { double x = a[k][p], y = a[k][q]; a[k][p] = c * x - s * y; a[k][q] = s * x + c * y; }
                for (int k = 0; k < 6; ++k) { double x = a[p][k], y = a[q][k]; a[p][k] = c * x - s * y; a[q][k] = s * x + c * y; }
                for (int k = 0; k < 6; ++k) { double x = v[k][p], y = v[k][q]; v[k][p] = c * x - s * y; v[k][q] = s * x + c * y; }
            }
    }
    for (int i = 0; i < 6; ++i) {
        w[i] = a[i][i];
        for (int j = 0; j < 6; ++j) V[i * 6 + j] = v[i][j];
    }
}

void psd_clip6_host(double *P, double *w, const double *V) {
    double wmax = 0.0;
    for (int a = 0; a < 6; ++a) wmax = std::max(wmax, w[a]);
    (void)wmax;
    for (int a = 0; a < 6; ++a) w[a] = std::max(w[a], 0.0);
    for (int a = 0; a < 6; ++a)
        for (int b = 0; b < 6; ++b) {
            double s = 0.0;
            for (int k = 0; k < 6; ++k) s += V[a * 6 + k] * w[k] * V[b * 6 + k];
            P[a * 6 + b] = s;
        }
}
// 合成用例：n 星、L 帧、E 条 ISL（固定图）、G 个 GTS；按 FLS 布局给出 A_val/A CSR/AᵀA 块图案。
// with_chain=false 时去掉跨帧 dyn 行/链（结构消融：L 个互不相连的帧）。
struct Case {
    int n, L, E, G;
    int64_t off_isl = 0, off_gts = 0, off_dyn = 0, off_damp = 0, nnz = 0;
    std::vector<int> isl_i, isl_j, gts;
    std::vector<double> A_val, sc;
    std::vector<int64_t> A_rp, A_ci, bptrs, binds;
    FlsLists lists;
};

// 行序与 fls_struct_impl 相同：ISL(L*E×12) → GTS(L*G×6) → DYN((L-1)*n*6×12) → PRI(L*n*6×6)。
// layout=0：节点编号帧优先 k*n+i；layout=1：卫星优先 i*L+k（供"每星一段消去区间"使用）。
void build_csr(Case &c, const std::vector<int> &isl_i, const std::vector<int> &isl_j,
               bool with_chain, int layout = 0) {
    const int n = c.n, L = c.L;
    auto NID = [&](int k, int i) { return layout ? (int64_t)i * L + k : (int64_t)k * n + i; };
    c.A_rp.assign(1, 0);
    c.A_ci.clear();
    auto row = [&](std::initializer_list<int64_t> bs) {
        for (auto b : bs)
            for (int t = 0; t < 6; ++t) c.A_ci.push_back(b + t);
        c.A_rp.push_back((int64_t)c.A_ci.size());
    };
    for (int k = 0; k < L; ++k)
        for (size_t e = 0; e < isl_i.size(); ++e) {
            int64_t b1 = NID(k, isl_i[e]) * 6, b2 = NID(k, isl_j[e]) * 6;
            row({b1, b2});
        }
    for (int k = 0; k < L; ++k)
        for (int g = 0; g < c.G; ++g) row({NID(k, c.gts[g]) * 6});
    if (with_chain)
        for (int k = 1; k < L; ++k)
            for (int i = 0; i < n; ++i)
                for (int comp = 0; comp < 6; ++comp)
                    row({NID(k - 1, i) * 6, NID(k, i) * 6});
    for (int64_t r = 0; r < (int64_t)L * n * 6; ++r) {
        // 先验行 r 对应"物理变量"r（帧优先布局下的 k*n*6+...），其块首列 = 该变量所在块*6
        int64_t node = r / 6;                       // 帧优先下的线性节点号
        int64_t k = node / n, i = node % n;
        row({NID((int)k, (int)i) * 6});
    }
}

// AᵀA 块图案（下三角 + 对角，行内严格升序）：帧内 ISL 边 + （with_chain 时）跨帧链。
void build_block_pattern(Case &c, bool with_chain, int layout = 0) {
    const int64_t nv = (int64_t)c.L * c.n;
    auto NID = [&](int64_t k, int64_t i) { return layout ? i * c.L + k : k * c.n + i; };
    std::vector<std::vector<int64_t>> adj((size_t)nv);
    auto add = [&](int64_t a, int64_t b) {
        adj[(size_t)a].push_back(b);
        adj[(size_t)b].push_back(a);
    };
    for (int k = 0; k < c.L; ++k)
        for (size_t e = 0; e < c.isl_i.size(); ++e)
            add(NID(k, c.isl_i[e]), NID(k, c.isl_j[e]));
    if (with_chain)
        for (int k = 1; k < c.L; ++k)
            for (int i = 0; i < c.n; ++i) add(NID(k - 1, i), NID(k, i));
    c.bptrs.assign(1, 0);
    c.binds.clear();
    for (int64_t a = 0; a < nv; ++a) {
        auto &v = adj[(size_t)a];
        v.push_back(a);
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
        for (int64_t b : v)
            if (b <= a) c.binds.push_back(b);   // BaSpaCho 规范：下三角块（含对角，行内严格升序）
        c.bptrs.push_back((int64_t)c.binds.size());
    }
}

Case make_case(int n, int L, int E, int G, uint64_t seed, bool with_chain = true, int layout = 0) {
    Case c; c.n = n; c.L = L; c.E = E; c.G = G;
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> gaus(0.0, 1.0);
    // 固定 ISL 图：i ↔ (i*3+1) mod n，再接 i ↔ (i+7) mod n（总量可达 2n）
    for (int i = 0; i < n && (int)c.isl_i.size() < E; ++i) {
        int j = (i * 3 + 1) % n;
        if (i != j) { c.isl_i.push_back(i); c.isl_j.push_back(j); }
    }
    for (int i = 0; i < n && (int)c.isl_i.size() < E; ++i) {
        int j = (i + 7) % n;
        if (i != j) { c.isl_i.push_back(i); c.isl_j.push_back(j); }
    }
    for (int g = 0; g < G; ++g) c.gts.push_back((g * 5 + 2) % n);
    const int64_t nISL = (int64_t)L * c.isl_i.size(), nGTS = (int64_t)L * G;
    const int64_t nDYN = with_chain ? (int64_t)(L - 1) * n * 6 : 0, nPRI = (int64_t)L * n * 6;
    c.off_isl = 0;
    c.off_gts = nISL * 12;
    c.off_dyn = c.off_gts + nGTS * 6;
    c.off_damp = c.off_dyn + nDYN * 12;
    c.nnz = c.off_damp + nPRI * 6;
    c.A_val.assign((size_t)c.nnz, 0.0);
    build_csr(c, c.isl_i, c.isl_j, with_chain, layout);
    build_block_pattern(c, with_chain, layout);
    // 量级与真实一致：ISL/GTS ~1、dyn ~1e7（白化）、prior ~1e6（紧先验的 Lp 行）
    for (int64_t q = 0; q < nISL * 12; ++q) c.A_val[c.off_isl + q] = gaus(rng);
    for (int64_t q = 0; q < nGTS * 6; ++q) c.A_val[c.off_gts + q] = gaus(rng);
    for (int64_t q = 0; q < nDYN * 12; ++q) c.A_val[c.off_dyn + q] = 1e7 * gaus(rng);
    for (int64_t q = 0; q < nPRI * 6; ++q) c.A_val[c.off_damp + q] = 1e6 * gaus(rng);
    // 真实 FLS 只有帧 0 有先验行；结构为 L*n*6 行，帧 1..L-1 的先验行必须为零。
    for (int64_t r = (int64_t)c.n * 6; r < nPRI; ++r)
        for (int cc = 0; cc < 6; ++cc) c.A_val[c.off_damp + r * 6 + cc] = 0.0;
    if (with_chain && L >= 2)   // 行表描述 Λ00/Λ11/Λ10，需要窗口含下一帧；L=1 无此概念
        c.lists = fls_make_lists(n, L, (int)c.isl_i.size(), G, c.isl_i, c.isl_j, c.gts,
                                 c.off_isl, c.off_gts, c.off_dyn, c.off_damp);
    // 列能量（按行表覆盖的项）→ sc = 1/sqrt(能量)（同 FLS 口径）
    const int64_t ncol = (int64_t)L * n * 6;
    std::vector<double> e2((size_t)ncol, 0.0);
    auto acc = [&](const std::vector<int64_t> &s, const std::vector<int64_t> &cc) {
        for (size_t p = 0; p < s.size(); ++p)
            for (int k = 0; k < 6; ++k) {
                double v = c.A_val[s[p] + k];
                e2[cc[p] + k] += v * v;
            }
    };
    acc(c.lists.d00_s, c.lists.d00_c);
    acc(c.lists.d11_s, c.lists.d11_c);
    c.sc.assign((size_t)ncol, 0.0);
    for (int64_t i = 0; i < ncol; ++i) {
        double s = 1.0 / std::sqrt(std::max(e2[i], 1e-30));
        c.sc[i] = std::min(std::max(s, 0.0), 1e6);
    }
    return c;
}

// host 参考（与核同算法，按行表收集）
void ref_marg(const Case &c, std::vector<double> &Lp, std::vector<double> &Pm) {
    const int n = c.n;
    Lp.assign((size_t)n * 36, 0.0);
    Pm.assign((size_t)n * 36, 0.0);
    for (int i = 0; i < n; ++i) {
        double L00[36] = {0}, L11[36] = {0}, L10[36] = {0};
        auto outer_acc = [&](double *M, const std::vector<int64_t> &s,
                             const std::vector<int64_t> &cc, int64_t p0, int64_t p1) {
            for (int64_t p = p0; p < p1; ++p) {
                double v[6];
                for (int k = 0; k < 6; ++k) v[k] = c.A_val[s[p] + k] * c.sc[cc[p] + k];
                for (int a = 0; a < 6; ++a)
                    for (int b = 0; b < 6; ++b) M[a * 6 + b] += v[a] * v[b];
            }
        };
        outer_acc(L00, c.lists.d00_s, c.lists.d00_c, c.lists.d00_p[i], c.lists.d00_p[i + 1]);
        outer_acc(L11, c.lists.d11_s, c.lists.d11_c, c.lists.d11_p[i], c.lists.d11_p[i + 1]);
        for (int k = 0; k < 6; ++k) {
            double v1[6], v0[6];
            for (int cc = 0; cc < 6; ++cc) {
                v1[cc] = c.A_val[c.lists.pr_s1[i * 6 + k] + cc] * c.sc[(int64_t)(n + i) * 6 + cc];
                v0[cc] = c.A_val[c.lists.pr_s0[i * 6 + k] + cc] * c.sc[(int64_t)i * 6 + cc];
            }
            for (int a = 0; a < 6; ++a)
                for (int b = 0; b < 6; ++b) L10[a * 6 + b] += v1[a] * v0[b];
        }
        {
            double tr = 0.0;
            for (int a = 0; a < 6; ++a) tr += L00[a * 6 + a];
            const double eps = std::max(tr * 1e-12, 1e-30);
            for (int a = 0; a < 6; ++a) L00[a * 6 + a] += eps;
        }
        double A0inv[36]; inv6_host(L00, A0inv);
        double U[36];
        for (int a = 0; a < 6; ++a)
            for (int b = 0; b < 6; ++b) {
                double s = 0.0;
                for (int k = 0; k < 6; ++k) s += L10[a * 6 + k] * A0inv[k * 6 + b];
                U[a * 6 + b] = s;
            }
        double Lmv[36], Lm[36];
        for (int a = 0; a < 6; ++a)
            for (int b = 0; b < 6; ++b) {
                double s = 0.0;
                for (int k = 0; k < 6; ++k) s += U[a * 6 + k] * L10[b * 6 + k];
                Lmv[a * 6 + b] = L11[a * 6 + b] - s;
            }
        for (int a = 0; a < 6; ++a)
            for (int b = 0; b < 6; ++b) Lm[a * 6 + b] = 0.5 * (Lmv[a * 6 + b] + Lmv[b * 6 + a]);
        {
            double tr = 0.0;
            for (int a = 0; a < 6; ++a) tr += Lm[a * 6 + a];
            const double eps = std::max(tr * 1e-12, 1e-30);
            for (int a = 0; a < 6; ++a) Lm[a * 6 + a] += eps;
        }
        double Pmv[36]; inv6_host(Lm, Pmv);
        for (int a = 0; a < 6; ++a)
            for (int b = 0; b < 6; ++b)
                Pm[(size_t)i * 36 + a * 6 + b] = Pmv[a * 6 + b] * c.sc[(int64_t)(n + i) * 6 + a]
                                                 * c.sc[(int64_t)(n + i) * 6 + b];
        for (int a = 0; a < 6; ++a)
            for (int b = 0; b < 6; ++b)
                Pm[(size_t)i * 36 + a * 6 + b] = 0.5 * (Pm[(size_t)i * 36 + a * 6 + b] +
                                                        Pm[(size_t)i * 36 + b * 6 + a]);
        double w[6], V[36];
        eigh6_host(&Pm[(size_t)i * 36], w, V);
        psd_clip6_host(&Pm[(size_t)i * 36], w, V);
        double wmx = 0.0;                       // 相对下限 wmax·1e-14，与 flsMargKernel 同步：
        for (int a = 0; a < 6; ++a) wmx = std::max(wmx, w[a]);   // Pm 与 Lp 用同一组 floored w
        const double wfl = std::max(wmx * 1e-14, 1e-300);
        for (int a = 0; a < 6; ++a) w[a] = std::max(w[a], wfl);
        for (int a = 0; a < 6; ++a)
            for (int b = 0; b < 6; ++b) {
                double s = 0.0;
                for (int k = 0; k < 6; ++k) s += V[a * 6 + k] * w[k] * V[b * 6 + k];
                Pm[(size_t)i * 36 + a * 6 + b] = s;
            }
        for (int a = 0; a < 6; ++a) {
            double si = 1.0 / std::sqrt(w[a]);
            for (int b = 0; b < 6; ++b) Lp[(size_t)i * 36 + a * 6 + b] = V[b * 6 + a] * si;
        }
    }
}

// 数据落盘（便于 py/人工核对；纯二进制，带魔数）
void dump_case(const Case &c, const char *path) {
    std::ofstream f(path, std::ios::binary);
    int hdr[4] = {c.n, c.L, c.E, c.G};
    int64_t off[5] = {c.off_isl, c.off_gts, c.off_dyn, c.off_damp, c.nnz};
    int64_t nl[6] = {(int64_t)c.lists.d00_s.size(), (int64_t)c.lists.d11_s.size(),
                     (int64_t)c.lists.pr_s1.size(), 0, 0, 0};
    f.write("FLSMARG1", 8);
    f.write((char *)hdr, sizeof(hdr));
    f.write((char *)off, sizeof(off));
    f.write((char *)nl, sizeof(nl));
    f.write((char *)c.isl_i.data(), (std::streamsize)c.isl_i.size() * sizeof(int));
    f.write((char *)c.isl_j.data(), (std::streamsize)c.isl_j.size() * sizeof(int));
    f.write((char *)c.gts.data(), (std::streamsize)c.gts.size() * sizeof(int));
    f.write((char *)c.A_val.data(), (std::streamsize)c.A_val.size() * sizeof(double));
    f.write((char *)c.sc.data(), (std::streamsize)c.sc.size() * sizeof(double));
    f.write((char *)c.lists.d00_s.data(), (std::streamsize)c.lists.d00_s.size() * sizeof(int64_t));
    f.write((char *)c.lists.d11_s.data(), (std::streamsize)c.lists.d11_s.size() * sizeof(int64_t));
}

}  // namespace

TEST(FlsMarg, KernelVsHostRef) {
    for (uint64_t seed : {1ULL, 7ULL, 42ULL}) {
        Case c = make_case(8, 3, 10, 2, seed);
        dump_case(c, "test_fls_marg_case.bin");

        std::vector<double> Lp_ref, Pm_ref;
        ref_marg(c, Lp_ref, Pm_ref);

        // 上传核输入
        double *d_A = nullptr, *d_sc = nullptr, *d_Lp = nullptr, *d_Pm = nullptr;
        int64_t *d_p[8] = {nullptr};
        cudaMalloc(&d_A, c.A_val.size() * sizeof(double));
        cudaMalloc(&d_sc, c.sc.size() * sizeof(double));
        cudaMalloc(&d_Lp, (size_t)c.n * 36 * sizeof(double));
        cudaMalloc(&d_Pm, (size_t)c.n * 36 * sizeof(double));
        cudaMemcpy(d_A, c.A_val.data(), c.A_val.size() * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_sc, c.sc.data(), c.sc.size() * sizeof(double), cudaMemcpyHostToDevice);
        auto up = [&](const std::vector<int64_t> &v) {
            int64_t *p = nullptr;
            cudaMalloc(&p, v.size() * sizeof(int64_t));
            cudaMemcpy(p, v.data(), v.size() * sizeof(int64_t), cudaMemcpyHostToDevice);
            return p;
        };
        d_p[0] = up(c.lists.d00_p); d_p[1] = up(c.lists.d00_s); d_p[2] = up(c.lists.d00_c);
        d_p[3] = up(c.lists.d11_p); d_p[4] = up(c.lists.d11_s); d_p[5] = up(c.lists.d11_c);
        d_p[6] = up(c.lists.pr_s1); d_p[7] = up(c.lists.pr_s0);
        qoejopt::flsMargDevice(d_A, d_sc, d_p[0], d_p[1], d_p[2], d_p[3], d_p[4], d_p[5],
                               d_p[6], d_p[7], c.n, d_Lp, d_Pm);
        std::vector<double> Lp_k((size_t)c.n * 36), Pm_k((size_t)c.n * 36);
        cudaMemcpy(Lp_k.data(), d_Lp, Lp_k.size() * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(Pm_k.data(), d_Pm, Pm_k.size() * sizeof(double), cudaMemcpyDeviceToHost);

        auto rel = [](const std::vector<double> &a, const std::vector<double> &b) {
            double na = 0, nb = 0, nd = 0;
            for (size_t i = 0; i < a.size(); ++i) {
                na += a[i] * a[i]; nb += b[i] * b[i];
                nd += (a[i] - b[i]) * (a[i] - b[i]);
            }
            return std::sqrt(nd) / std::max(std::sqrt(nb), 1e-300);
        };
        double rLp = rel(Lp_k, Lp_ref), rPm = rel(Pm_k, Pm_ref);
        std::printf("[gtest/fls-marg] seed=%llu  rel Lp=%.3e  rel Pm=%.3e  |Pm|max=%.3e\n",
                    (unsigned long long)seed, rLp, rPm,
                    *std::max_element(Pm_ref.begin(), Pm_ref.end(),
                                      [](double a, double b) { return std::fabs(a) < std::fabs(b); }));
        EXPECT_LT(rLp, 1e-9) << "seed=" << seed;
        EXPECT_LT(rPm, 1e-12) << "seed=" << seed;
        cudaFree(d_A); cudaFree(d_sc); cudaFree(d_Lp); cudaFree(d_Pm);
        for (auto *p : d_p) cudaFree(p);
    }
}


namespace {
void run_kernel_case(const Case &c, std::vector<double> &Lp_k, std::vector<double> &Pm_k) {
    double *d_A = nullptr, *d_sc = nullptr, *d_Lp = nullptr, *d_Pm = nullptr;
    cudaMalloc(&d_A, c.A_val.size() * sizeof(double));
    cudaMalloc(&d_sc, c.sc.size() * sizeof(double));
    cudaMalloc(&d_Lp, (size_t)c.n * 36 * sizeof(double));
    cudaMalloc(&d_Pm, (size_t)c.n * 36 * sizeof(double));
    cudaMemcpy(d_A, c.A_val.data(), c.A_val.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sc, c.sc.data(), c.sc.size() * sizeof(double), cudaMemcpyHostToDevice);
    auto up = [&](const std::vector<int64_t> &v) {
        int64_t *p = nullptr;
        cudaMalloc(&p, v.size() * sizeof(int64_t));
        cudaMemcpy(p, v.data(), v.size() * sizeof(int64_t), cudaMemcpyHostToDevice);
        return p;
    };
    int64_t *dp[8] = {up(c.lists.d00_p), up(c.lists.d00_s), up(c.lists.d00_c),
                      up(c.lists.d11_p), up(c.lists.d11_s), up(c.lists.d11_c),
                      up(c.lists.pr_s1), up(c.lists.pr_s0)};
    qoejopt::flsMargDevice(d_A, d_sc, dp[0], dp[1], dp[2], dp[3], dp[4], dp[5], dp[6], dp[7],
                           c.n, d_Lp, d_Pm);
    Lp_k.assign((size_t)c.n * 36, 0.0); Pm_k.assign((size_t)c.n * 36, 0.0);
    cudaMemcpy(Lp_k.data(), d_Lp, Lp_k.size() * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(Pm_k.data(), d_Pm, Pm_k.size() * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_sc); cudaFree(d_Lp); cudaFree(d_Pm);
    for (auto *p : dp) cudaFree(p);
}
}  // namespace

TEST(FlsLists, PriorRowsSix) {
    Case c = make_case(6, 2, 0, 0, 11);          // E=G=0：d00 = 6 行先验 + 1 行 dyn
    for (int i = 0; i < c.n; ++i)
        EXPECT_EQ(c.lists.d00_p[i + 1] - c.lists.d00_p[i], 12) << "star=" << i;
}

TEST(FlsLists, PairOnlyNoFutureDyn) {
    Case c = make_case(6, 4, 0, 0, 12);          // L=4：d11 只应有 dyn(0→1) 一行
    for (int i = 0; i < c.n; ++i)
        EXPECT_EQ(c.lists.d11_p[i + 1] - c.lists.d11_p[i], 6) << "star=" << i;
}

TEST(FlsMarg, DensePairSchurL2) {
    const int n = 6, L = 2, E = 8, G = 2;
    Case c = make_case(n, L, E, G, 21);
    std::vector<double> Lp_k, Pm_k;
    run_kernel_case(c, Lp_k, Pm_k);
    std::vector<double> Lp_ref, Pm_ref;
    ref_marg(c, Lp_ref, Pm_ref);

    const int ncol = L * n * 6;
    const int nrows = (int)c.A_rp.size() - 1;
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(nrows, ncol);
    for (int r = 0; r < nrows; ++r)
        for (int64_t p = c.A_rp[r]; p < c.A_rp[r + 1]; ++p)
            A(r, c.A_ci[(size_t)p]) = c.A_val[(size_t)p];
    for (int j = 0; j < ncol; ++j) A.col(j) *= c.sc[(size_t)j];
    Eigen::MatrixXd H = A.transpose() * A;

    double max_rel = 0.0;
    for (int i = 0; i < n; ++i) {
        auto blk = [&](int fr, int fc) {
            return H.block<6, 6>(fr * n * 6 + i * 6, fc * n * 6 + i * 6);
        };
        M6 H00 = blk(0, 0);
        M6 H11 = blk(1, 1);
        M6 H10 = blk(1, 0);
        H00 += M6::Identity() * std::max(H00.trace() * 1e-12, 1e-30);
        M6 Lm = H11 - H10 * H00.inverse() * H10.transpose();
        Lm += M6::Identity() * std::max(Lm.trace() * 1e-12, 1e-30);
        M6 P = Lm.inverse();
        Eigen::Matrix<double, 6, 1> s1;
        for (int a = 0; a < 6; ++a) s1[a] = c.sc[(size_t)(n * 6 + i * 6 + a)];
        P = s1.asDiagonal() * P * s1.asDiagonal();
        Eigen::SelfAdjointEigenSolver<M6> es(P);
        Eigen::Matrix<double, 6, 1> wv = es.eigenvalues();
        (void)wv.maxCoeff();
        for (int a = 0; a < 6; ++a) wv[a] = std::max(wv[a], 0.0);
        P = es.eigenvectors() * wv.asDiagonal() * es.eigenvectors().transpose();
        M6 Pk;
        for (int a = 0; a < 6; ++a)
            for (int b = 0; b < 6; ++b) Pk(a, b) = Pm_k[(size_t)i * 36 + a * 6 + b];
        double nd = (P - Pk).norm(), nb = P.norm();
        if (i == 0) {
            std::printf("[dense] P diag: ");
            for (int a = 0; a < 6; ++a) std::printf("%.3e ", P(a, a));
            std::printf("\n[dense] Pk diag: ");
            for (int a = 0; a < 6; ++a) std::printf("%.3e ", Pk(a, a));
            std::printf("\n[dense] Pref diag: ");
            for (int a = 0; a < 6; ++a) std::printf("%.3e ", Pm_ref[(size_t)a]);
            std::printf("\n[dense] rel=%.3e |P|=%.3e |Pk|=%.3e\n", nd / std::max(nb, 1e-300),
                        nb, Pk.norm());
        }
        max_rel = std::max(max_rel, nd / std::max(nb, 1e-300));
    }
    EXPECT_LT(max_rel, 1e-5) << "密集成对舒尔参考与核不一致（随机病态例，数值容差 1e-5）";
}

TEST(FlsMarg, DenseTinyNoObs) {
    const int n = 1, L = 2;
    Case c = make_case(n, L, 0, 0, 41);
    std::vector<double> Lp_k, Pm_k;
    run_kernel_case(c, Lp_k, Pm_k);
    const int ncol = L * n * 6, nrows = (int)c.A_rp.size() - 1;
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(nrows, ncol);
    for (int r = 0; r < nrows; ++r)
        for (int64_t p = c.A_rp[r]; p < c.A_rp[r + 1]; ++p)
            A(r, c.A_ci[(size_t)p]) = c.A_val[(size_t)p];
    for (int j = 0; j < ncol; ++j) A.col(j) *= c.sc[(size_t)j];
    Eigen::MatrixXd H = A.transpose() * A;
    M6 H00 = H.block<6, 6>(0, 0), H11 = H.block<6, 6>(6, 6), H10 = H.block<6, 6>(6, 0);
    M6 L00 = M6::Zero(), L11 = M6::Zero(), L10 = M6::Zero();
    auto add_rows = [&](M6 &M, const std::vector<int64_t> &p, const std::vector<int64_t> &s,
                        const std::vector<int64_t> &cc, int i) {
        for (int64_t q = p[i]; q < p[i + 1]; ++q) {
            Eigen::Matrix<double, 6, 1> v;
            for (int kk = 0; kk < 6; ++kk)
                v[kk] = c.A_val[s[(size_t)q] + kk] * c.sc[cc[(size_t)q] + kk];
            M += v * v.transpose();
        }
    };
    add_rows(L00, c.lists.d00_p, c.lists.d00_s, c.lists.d00_c, 0);
    add_rows(L11, c.lists.d11_p, c.lists.d11_s, c.lists.d11_c, 0);
    for (int k = 0; k < 6; ++k) {
        Eigen::Matrix<double, 6, 1> v1, v0;
        for (int cc = 0; cc < 6; ++cc) {
            v1[cc] = c.A_val[c.lists.pr_s1[k] + cc] * c.sc[6 + cc];
            v0[cc] = c.A_val[c.lists.pr_s0[k] + cc] * c.sc[cc];
        }
        L10 += v1 * v0.transpose();
    }
    std::printf("[tiny] sc:");
    for (int a = 0; a < 12; ++a) std::printf(" %.3e", c.sc[(size_t)a]);
    std::printf("\n[tiny] |L00-H00|max=%.3e |L11-H11|max=%.3e |L10-H10|max=%.3e\n",
                (L00 - H00).cwiseAbs().maxCoeff(), (L11 - H11).cwiseAbs().maxCoeff(),
                (L10 - H10).cwiseAbs().maxCoeff());
    H00 += M6::Identity() * std::max(H00.trace() * 1e-12, 1e-30);
    M6 Lm = H11 - H10 * H00.inverse() * H10.transpose();
    Lm += M6::Identity() * std::max(Lm.trace() * 1e-12, 1e-30);
    M6 P = Lm.inverse();
    Eigen::Matrix<double, 6, 1> s1;
    for (int a = 0; a < 6; ++a) s1[a] = c.sc[(size_t)(6 + a)];
    P = s1.asDiagonal() * P * s1.asDiagonal();
    Eigen::SelfAdjointEigenSolver<M6> es(P);
    auto wv = es.eigenvalues(); (void)wv.maxCoeff();
    for (int a = 0; a < 6; ++a) wv[a] = std::max(wv[a], 0.0);
    P = es.eigenvectors() * wv.asDiagonal() * es.eigenvectors().transpose();
    M6 Pk;
    for (int a = 0; a < 6; ++a)
        for (int b = 0; b < 6; ++b) Pk(a, b) = Pm_k[(size_t)a * 6 + b];
    std::printf("[tiny] rel=%.3e |P|=%.3e |Pk|=%.3e\\n",
                (P - Pk).norm() / std::max(P.norm(), 1e-300), P.norm(), Pk.norm());
    EXPECT_LT((P - Pk).norm() / std::max(P.norm(), 1e-300), 1e-5);
}

TEST(FlsMarg, PmPSD) {
    Case c = make_case(32, 4, 64, 8, 31);
    std::vector<double> Lp_k, Pm_k;
    run_kernel_case(c, Lp_k, Pm_k);
    double worst = 0.0;
    for (int i = 0; i < c.n; ++i) {
        M6 P;
        for (int a = 0; a < 6; ++a)
            for (int b = 0; b < 6; ++b) P(a, b) = Pm_k[(size_t)i * 36 + a * 6 + b];
        M6 Ps = 0.5 * (P + P.transpose());
        Eigen::SelfAdjointEigenSolver<M6> es(Ps);
        double wmax = es.eigenvalues().cwiseAbs().maxCoeff();
        worst = std::min(worst, es.eigenvalues().minCoeff() / std::max(wmax, 1e-300));
    }
    EXPECT_GT(worst, -1e-10) << "Pm 出现负特征值";
}

// ---- 真实规模核的耗时（glog 输出；FlsWindow 全步归因待其头文件化后并入）----
TEST(FlsTiming, MargKernel) {
    const int n = 1152, L = 4, E = 2306, G = 432;   // 与真实规模一致（ISL≈2N、GTS 432）
    Case c = make_case(n, L, E, G, 3);
    LOG(INFO) << "[fls/timing] case n=" << c.n << " L=" << c.L << " E=" << c.isl_i.size()
              << " G=" << c.G << " nnz=" << c.nnz;
    double *d_A = nullptr, *d_sc = nullptr, *d_Lp = nullptr, *d_Pm = nullptr;
    cudaMalloc(&d_A, c.A_val.size() * sizeof(double));
    cudaMalloc(&d_sc, c.sc.size() * sizeof(double));
    cudaMalloc(&d_Lp, (size_t)c.n * 36 * sizeof(double));
    cudaMalloc(&d_Pm, (size_t)c.n * 36 * sizeof(double));
    cudaMemcpy(d_A, c.A_val.data(), c.A_val.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sc, c.sc.data(), c.sc.size() * sizeof(double), cudaMemcpyHostToDevice);
    auto up = [&](const std::vector<int64_t> &v) {
        int64_t *p = nullptr;
        cudaMalloc(&p, v.size() * sizeof(int64_t));
        cudaMemcpy(p, v.data(), v.size() * sizeof(int64_t), cudaMemcpyHostToDevice);
        return p;
    };
    int64_t *dp[8] = {up(c.lists.d00_p), up(c.lists.d00_s), up(c.lists.d00_c),
                      up(c.lists.d11_p), up(c.lists.d11_s), up(c.lists.d11_c),
                      up(c.lists.pr_s1), up(c.lists.pr_s0)};
    for (int r = 0; r < 5; ++r) {
        cudaDeviceSynchronize();
        auto t0 = std::chrono::steady_clock::now();
        qoejopt::flsMargDevice(d_A, d_sc, dp[0], dp[1], dp[2], dp[3], dp[4], dp[5], dp[6], dp[7],
                               c.n, d_Lp, d_Pm);
        auto t1 = std::chrono::steady_clock::now();
        std::vector<double> Lp_k((size_t)c.n * 36), Pm_k((size_t)c.n * 36);
        cudaMemcpy(Lp_k.data(), d_Lp, Lp_k.size() * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(Pm_k.data(), d_Pm, Pm_k.size() * sizeof(double), cudaMemcpyDeviceToHost);
        auto t2 = std::chrono::steady_clock::now();
        using ms = std::chrono::duration<double, std::milli>;
        LOG(INFO) << "[fls/timing] run " << r << ": 核=" << ms(t1 - t0).count()
                  << " ms  D2H=" << ms(t2 - t1).count() << " ms";
    }
    cudaFree(d_A); cudaFree(d_sc); cudaFree(d_Lp); cudaFree(d_Pm);
    for (auto *p : dp) cudaFree(p);
}

// ---- 求解链耗时归因：对照 run_filter_ekf.py --joint 的 py 链
//      （SymbolicDecomposition → createNumericDecomposition → add_MtM → factor → solve），
//      分离符号/每调用/数值三部分；双 factor 分离"每调用准备"与"真数值分解"。----
namespace {
// 符号分解可调项（消去区间/末尾保留/填充策略/自动稀疏消去探测）
struct SymVar {
    std::string name;
    std::vector<int64_t> ranges, elim_last;
    int64_t policy = 0;   // 0=Complete 1=ForAutoElims 2=ForGivenElims 3=None
    bool find = true;
};

void run_solver_chain(const Case &c, const std::string &backend, int reps,
                      const SymVar *v = nullptr, std::vector<double> *out_x = nullptr) {
    const bool cuda = (backend == "cuda");
    const bool chain = (c.off_damp > c.off_dyn);
    const std::string tag = v ? v->name : "base";
    auto dv = cuda ? torch::kCUDA : torch::kCPU;
    auto optD = torch::TensorOptions().dtype(torch::kFloat64).device(dv);
    auto optI = torch::TensorOptions().dtype(torch::kInt64).device(dv);
    auto optsFcpu = torch::TensorOptions().dtype(torch::kFloat64);
    auto optsIcpu = torch::TensorOptions().dtype(torch::kInt64);
    auto paramSize = torch::full({(int64_t)c.L * c.n}, 6, optsIcpu);
    auto bptr = torch::from_blob((void *)c.bptrs.data(), {(int64_t)c.bptrs.size()}, optsIcpu).clone();
    auto bind = torch::from_blob((void *)c.binds.data(), {(int64_t)c.binds.size()}, optsIcpu).clone();
    auto Av = torch::from_blob((void *)c.A_val.data(), {(int64_t)c.A_val.size()}, optsFcpu)
                  .clone().to(dv);
    auto Arp = torch::from_blob((void *)c.A_rp.data(), {(int64_t)c.A_rp.size()}, optsIcpu)
                   .clone().to(dv);
    auto Aci = torch::from_blob((void *)c.A_ci.data(), {(int64_t)c.A_ci.size()}, optsIcpu)
                   .clone().to(dv);
    const int64_t ncol = (int64_t)c.L * c.n * 6;
    // 每非零的行号（残差校验用）
    std::vector<int64_t> row_idx((size_t)c.A_ci.size());
    for (size_t r = 0; r + 1 < c.A_rp.size(); ++r)
        for (int64_t i = c.A_rp[r]; i < c.A_rp[r + 1]; ++i) row_idx[(size_t)i] = (int64_t)r;
    auto RowIdx = torch::from_blob(row_idx.data(), {(int64_t)row_idx.size()}, optsIcpu)
                      .clone().to(dv);

    using clk = std::chrono::steady_clock;
    auto now = [] {
        return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
    };
    auto devsync = [&] {
        if (cuda) cudaDeviceSynchronize();
    };
    devsync();
    double t0 = now();
    std::unique_ptr<SymbolicDecomposition> sym;
    if (v)
        sym = std::make_unique<SymbolicDecomposition>(paramSize, bptr, bind, backend, v->ranges,
                                                      v->elim_last, v->policy, v->find);
    else
        sym = std::make_unique<SymbolicDecomposition>(paramSize, bptr, bind, backend);
    devsync();
    double t_sym = now() - t0;
    for (int r = 0; r < reps; ++r) {
        devsync(); t0 = now();
        auto nd = sym->createNumericDecomposition(1);
        devsync(); double t_create = now() - t0;
        t0 = now(); nd.add_MtM(Av.reshape({1, (int64_t)c.nnz}), Arp, Aci); devsync();
        double t_add = now() - t0;
        t0 = now(); nd.factor(); devsync(); double t_fac = now() - t0;
        auto x = torch::ones({1, ncol}, optD);
        auto resid_of = [&](const torch::Tensor &xx) {
            auto Ax = torch::zeros({(int64_t)c.A_rp.size() - 1}, optD)
                          .index_add_(0, RowIdx, Av * xx.reshape({-1}).index_select(0, Aci));
            auto AtAx = torch::zeros({ncol}, optD)
                            .index_add_(0, Aci, Av * Ax.index_select(0, RowIdx));
            return ((AtAx - 1.0).norm() / std::sqrt((double)ncol)).item<double>();
        };
        double res0 = resid_of(x);
        t0 = now(); nd.solve(x); devsync(); double t_solve = now() - t0;
        double res_rel = resid_of(x);
        if (out_x && r == reps - 1) {
            auto xc = x.reshape({-1}).cpu();
            const double *px = xc.data_ptr<double>();
            out_x->assign(px, px + xc.numel());
        }
        if (r == 0) {
            auto ss_t = nd.span_start();
            auto ss = ss_t.accessor<int64_t, 1>();
            int64_t ns = ss_t.numel() - 1, maxw = 0, lastw = 0;
            for (int64_t i = 0; i < ns; ++i) {
                int64_t w = ss[i + 1] - ss[i];
                maxw = std::max(maxw, w);
                if (i + 1 == ns) lastw = w;
            }
            LOG(INFO) << "[fls/solver] dev=" << backend << " L=" << c.L << " chain=" << (chain ? 1 : 0)
                      << " var=" << tag << " nA=" << (c.A_rp.size() - 1) << " A_nnz=" << c.nnz
                      << " ncol=" << nd.order() << " dataSize=" << nd.data_size()
                      << " nSpan=" << ns << " maxSpan=" << maxw << " lastSpan=" << lastw
                      << " | sym=" << t_sym << " ms";
        }
        LOG(INFO) << "[fls/solver] dev=" << backend << " L=" << c.L << " chain=" << (chain ? 1 : 0)
                  << " var=" << tag << " rep=" << r << " | create=" << t_create
                  << " add=" << t_add << " fac=" << t_fac
                  << " solve=" << t_solve << " ms resid=" << res_rel << " res0=" << res0;
    }
}
}  // namespace

TEST(FlsTiming, SolverChain) {
    const int n = 1152, E = 2306, G = 432;   // 与真实规模一致
    for (int L : {1, 2, 3, 4}) {             // L 扫描（L=1 ≈ 基线 6N 单层）
        Case c = make_case(n, L, E, G, 3, true);
        run_solver_chain(c, "cuda", 3);
    }
    {                                        // 结构消融：4 帧但去跨帧链
        Case c = make_case(n, 4, E, G, 3, false);
        run_solver_chain(c, "cuda", 3);
    }
    {                                        // L=4 消去/填充策略对照（块单位：每帧 n 块）
        Case c = make_case(n, 4, E, G, 3, true);
        const int64_t nb = n;
        std::vector<int64_t> last_ids;
        for (int64_t b = 3 * nb; b < 4 * nb; ++b) last_ids.push_back(b);
        std::vector<double> x_ref;
        run_solver_chain(c, "cuda", 1, nullptr, &x_ref);     // 基准解（默认设置）
        const SymVar vars[] = {
            {"r-all", {0, nb, 2 * nb, 3 * nb, 4 * nb}, {}, 0, true},
            {"r-012", {0, nb, 2 * nb, 3 * nb}, {}, 0, true},
            {"r-012-last", {0, nb, 2 * nb, 3 * nb}, last_ids, 0, true},
            {"r-01", {0, nb, 2 * nb}, {}, 0, true},
            {"find0", {}, {}, 0, false},
        };
        for (const auto &vv : vars) {
            std::vector<double> x_v;
            run_solver_chain(c, "cuda", 2, &vv, &x_v);
            double nd = 0, nb2 = 0, nv2 = 0;
            for (size_t i = 0; i < x_ref.size(); ++i) {
                nd += (x_v[i] - x_ref[i]) * (x_v[i] - x_ref[i]);
                nb2 += x_ref[i] * x_ref[i];
                nv2 += x_v[i] * x_v[i];
            }
            LOG(INFO) << "[fls/solver] variance check: var=" << vv.name
                      << " rel(x vs base)=" << std::sqrt(nd) / std::max(std::sqrt(nb2), 1e-300)
                      << " |x_ref|=" << std::sqrt(nb2) << " |x_v|=" << std::sqrt(nv2);
        }
    }
    for (int L : {2, 3}) {                   // r-all 随 L 的缩放
        Case c = make_case(n, L, E, G, 3, true);
        std::vector<int64_t> rr;
        for (int k = 0; k <= L; ++k) rr.push_back((int64_t)k * n);
        SymVar vv{"r-all", rr, {}, 0, true};
        run_solver_chain(c, "cuda", 2, &vv);
    }
    for (int L : {1, 2}) {                   // find=0（不启用自动稀疏消去区间 → 允许 lump 合并）
        Case c = make_case(n, L, E, G, 3, true);
        SymVar vf{"find0", {}, {}, 0, false};
        run_solver_chain(c, "cuda", 3, &vf);
    }
    {   // 数值尺度消融：整体缩放 1e-6（条件数不变）
        Case c = make_case(n, 1, E, G, 3, true);
        for (double &v : c.A_val) v *= 1e-6;
        run_solver_chain(c, "cuda", 1);
    }
    {   // ranges 定位：无链 L=4 与单帧 L=1 下 r-all 是否正确
        SymVar vr4{"r-all", {0, n, 2 * n, 3 * n, 4 * n}, {}, 0, true};
        Case c4 = make_case(n, 4, E, G, 3, false);
        run_solver_chain(c4, "cuda", 1, &vr4);
        SymVar vr1{"r-all", {0, n}, {}, 0, true};
        Case c1 = make_case(n, 1, E, G, 3, true);
        run_solver_chain(c1, "cuda", 1, &vr1);
    }
    {   // 卫星优先编号（i*L+k）+ 每星一段消去区间 → 已验证失效（sym 13s、dataSize 66M、解不正
        // 确），不再保留：BaSpaCho 消去区间适用"区间内无内部耦合"（BA 点式），我们的帧/链不满足。
    }
    {   // CPU 后端、同数据、捕获解
        Case c = make_case(n, 1, E, G, 3, true);
        run_solver_chain(c, "cpu", 1);
    }
    {                                        // CPU 后端对照（add_MtM 的 CPU 主机回环慢，只跑 1 次）
        Case c = make_case(n, 4, E, G, 3, true);
        run_solver_chain(c, "cpu", 1);
    }
}

// ---- dyn 因子 STM 核 A/B：gveStmDevice（1 线程/星） vs gveStmDeviceGrp（6 线程/星）----
TEST(FlsTiming, StmKernel) {
    const int n = 1152;
    std::mt19937_64 rng(7);
    std::normal_distribution<double> gaus(0.0, 1.0);
    std::vector<double> oe((size_t)n * 6);
    for (int i = 0; i < n; ++i) {
        double *x = &oe[(size_t)i * 6];
        x[0] = 7.0e6; x[1] = 0.3 + 0.02 * i; x[2] = 1e-4; x[3] = -2e-4; x[4] = 1.0;
        x[5] = 6.2831853 * i / n;
        for (int c = 0; c < 6; ++c) x[c] += 1e-3 * gaus(rng);
    }
    double *d_oe = nullptr, *d_of1 = nullptr, *d_of2 = nullptr, *d_p1 = nullptr, *d_p2 = nullptr;
    cudaMalloc(&d_oe, oe.size() * sizeof(double));
    cudaMalloc(&d_of1, oe.size() * sizeof(double));
    cudaMalloc(&d_of2, oe.size() * sizeof(double));
    cudaMalloc(&d_p1, (size_t)n * 36 * sizeof(double));
    cudaMalloc(&d_p2, (size_t)n * 36 * sizeof(double));
    cudaMemcpy(d_oe, oe.data(), oe.size() * sizeof(double), cudaMemcpyHostToDevice);
    qoejopt::gveStmDevice(d_oe, n, 10.0, 1.0, d_of1, d_p1);
    qoejopt::gveStmDeviceGrp(d_oe, n, 10.0, 1.0, d_of2, d_p2);
    std::vector<double> f1(oe.size()), f2(oe.size());
    std::vector<double> p1((size_t)n * 36), p2((size_t)n * 36);
    cudaMemcpy(f1.data(), d_of1, f1.size() * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(f2.data(), d_of2, f2.size() * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(p1.data(), d_p1, p1.size() * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(p2.data(), d_p2, p2.size() * sizeof(double), cudaMemcpyDeviceToHost);
    auto rel = [](const std::vector<double> &a, const std::vector<double> &b) {
        double nd = 0, nb = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            nd += (a[i] - b[i]) * (a[i] - b[i]);
            nb += b[i] * b[i];
        }
        return std::sqrt(nd) / std::max(std::sqrt(nb), 1e-300);
    };
    LOG(INFO) << "[fls/stm] rel phi=" << rel(f2, f1) << " rel Phi=" << rel(p2, p1);
    EXPECT_LT(rel(f2, f1), 1e-12);
    EXPECT_LT(rel(p2, p1), 1e-12);
    for (int r = 0; r < 5; ++r) {
        cudaDeviceSynchronize();
        auto t0 = std::chrono::steady_clock::now();
        qoejopt::gveStmDevice(d_oe, n, 10.0, 1.0, d_of1, d_p1);
        auto t1 = std::chrono::steady_clock::now();
        qoejopt::gveStmDeviceGrp(d_oe, n, 10.0, 1.0, d_of2, d_p2);
        auto t2 = std::chrono::steady_clock::now();
        using ms = std::chrono::duration<double, std::milli>;
        LOG(INFO) << "[fls/stm] run " << r << ": base=" << ms(t1 - t0).count()
                  << " ms  grp=" << ms(t2 - t1).count() << " ms";
    }
    cudaFree(d_oe); cudaFree(d_of1); cudaFree(d_of2); cudaFree(d_p1); cudaFree(d_p2);
}

int main(int argc, char **argv) {
    google::InitGoogleLogging(argv[0]);
    google::SetStderrLogging(google::INFO);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
