#ifndef EKF_CENTRAL_H
#define EKF_CENTRAL_H

// 集中式精确协方差 EKF（doc/EKF_optimization.md §六）。
// 状态 x=[q_0;b_0;...;q_{N-1};b_{N-1};c_0..c_{M-1}]，q 为 QOE，b/c 为米制钟差。
// 协方差在全流程保持 DU 归一化 z 空间：z=S^{-1}x，S=diag(DU,1,1,1,1,1) 逐节点，
// 钟差/地面站 S=1（与 run_dams_global.py 同口径）。DU=7000 km。
//
// 本头文件只含 host 声明与小规模稠密参考代数（gtest 用）；GPU 内核在
// src/ekf_central.cu，pybind 仅导出 ekf_predict / ekf_update 两个入口。

#include <cmath>
#include <cstring>
#include <vector>

namespace ekf_central {

constexpr double DU = 7.0e6;  // QOE 半长轴归一化

struct Params {
    int n = 0, M = 0, E = 0, G = 0, nstate = 0, nrows = 0;
    // 方法二：显式 4 维规范状态 g=[t_x,t_y,t_z, θ·R_ref]，附在 nstate 末尾。
    // ng=0 时完全退化为原 7N+M 状态（默认）。gauge_rref 把 RAAN 折成米。
    int ng = 0;
    double gauge_rref = 7.0e6;
    double sigma_isl = 30.0, sigma_gts = 19.0;
    double tau_b = 600.0, sigma_b = 0.3, tau_c = 600.0, sigma_c = 0.3;
    double q_acc = 2.5e-11;  // Q_ACC²，Q_ACC=5e-6 m/s²（fls_marg.py）
    int joseph = 1, diag_every = 1;
    int fp32 = 1;  // 1=K 与 P 更新的大 GEMM 用 FP32（P/S/Cholesky 仍 FP64）
};

struct PredictResult {
    double pred_ms = 0.0, fpft_ms = 0.0, copy_ms = 0.0;
    bool diag = false;
    std::vector<double> Phi;  // (n,6,6) 物理 Φ_oe
    std::vector<double> Q;    // (n,7,7) 物理 Q（含钟差 qb）
    double phi_b = 0.0, phi_c = 0.0, qb = 0.0, qc = 0.0;
    int P_nonfinite = 0, state_nonfinite = 0;
    double min_diag = NAN;
};

struct UpdateResult {
    double h_ms = 0.0, PHt_ms = 0.0, S_ms = 0.0, potrf_ms = 0.0, K_ms = 0.0;
    double transpose_ms = 0.0, Pupd_ms = 0.0, total_ms = 0.0;
    double min_eig = NAN, sym_rel = NAN;
    double S_min_eig = NAN, S_max_eig = NAN, S_cond = NAN, S_cond_chol = NAN;
    int potrf_info = 0;
    bool diag = false;
    bool diag_done = false;  // true 才有 min_eig；false 时 min_eig=NAN 仅表示未计算
    int P_nonfinite = 0, state_nonfinite = 0;
    double min_diag = NAN;
    int nstate = 0, nrows = 0;
    // 对拍诊断（do_diag==2）：均为 CUDA 缓冲区的原始内存拷贝。
    // H_colT: nrows*nstate，列主序 H（即 C 序 (nstate,nrows) 的 H^T）。
    // K:      nstate*nrows，列主序 K（即 C 序 (nrows,nstate) 的 K^T）。
    std::vector<double> H_colT, K, nu, dx, rv;
};

// 预测/更新入口。q,b,c,P 为设备指针（Python torch CUDA 张量），批内常驻状态由实现持有。
// ext：可选的外部 ECI 加速度 (n,3) 主机数组（培训力场，180 s 内常值）；空指针=不加。
void predict(const Params& p, const int* edges, const int* gts, const double* ext,
             double* q, double* b, double* c, double* P, double* g,
             int init, int frame, int do_diag, PredictResult* out);
void update(const double* D_isl, const double* M_isl,
            const double* D_gts, const double* M_gts, const double* Q_gts,
            int frame, int do_diag, UpdateResult* out);

// 释放设备缓冲（gtest 在切换规模或退出前调用）。
void release();

// ---------------------------------------------------------------------------
// host 稠密参考代数（小规模，row-major；与 GPU 实现同口径）
// ---------------------------------------------------------------------------
namespace ref {

inline bool chol(double* A, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double s = A[(size_t)i * n + j];
            for (int k = 0; k < j; ++k) s -= A[(size_t)i * n + k] * A[(size_t)j * n + k];
            if (i == j) {
                if (!(s > 0.0)) return false;
                A[(size_t)i * n + j] = std::sqrt(s);
            } else {
                A[(size_t)i * n + j] = s / A[(size_t)j * n + j];
            }
        }
    }
    return true;
}

// 解 L L^T X = B；L/B/X 均 row-major，B 与 X 为 n x nrhs。
inline void chol_solve(const double* L, int n, const double* B, double* X, int nrhs) {
    std::vector<double> y((size_t)n * nrhs);
    for (int c = 0; c < nrhs; ++c) {
        for (int i = 0; i < n; ++i) {
            double s = B[(size_t)i * nrhs + c];
            for (int k = 0; k < i; ++k) s -= L[(size_t)i * n + k] * y[(size_t)k * nrhs + c];
            y[(size_t)i * nrhs + c] = s / L[(size_t)i * n + i];
        }
        for (int i = n - 1; i >= 0; --i) {
            double s = y[(size_t)i * nrhs + c];
            for (int k = i + 1; k < n; ++k) s -= L[(size_t)k * n + i] * X[(size_t)k * nrhs + c];
            X[(size_t)i * nrhs + c] = s / L[(size_t)i * n + i];
        }
    }
}

// Fz 逐节点块为 S^{-1}ΦS（a 分量按 DU 缩放）与 φ_b；站为 φ_c。
// P 与 Pout 为 z 空间 (ns,ns) row-major；Phi (n,36) 与 Q (n,49) 为物理量。
inline void dense_predict(const Params& p, const double* P, const double* Phi, const double* Q,
                          double phi_b, double phi_c, double qc, double* Pout) {
    const int ns = p.nstate;
    std::vector<double> Fz((size_t)ns * ns, 0.0), Qz((size_t)ns * ns, 0.0);
    auto s_of = [](int a) { return a == 0 ? DU : 1.0; };
    for (int i = 0; i < p.n; ++i) {
        const int o = 7 * i;
        for (int a = 0; a < 6; ++a)
            for (int b = 0; b < 6; ++b)
                Fz[(size_t)(o + a) * ns + o + b] = Phi[i * 36 + a * 6 + b] * s_of(b) / s_of(a);
        Fz[(size_t)(o + 6) * ns + o + 6] = phi_b;
        for (int a = 0; a < 7; ++a)
            for (int b = 0; b < 7; ++b)
                Qz[(size_t)(o + a) * ns + o + b] = Q[i * 49 + a * 7 + b] / (s_of(a) * s_of(b));
    }
    for (int s = 0; s < p.M; ++s) {
        const int o = 7 * p.n + s;
        Fz[(size_t)o * ns + o] = phi_c;
        Qz[(size_t)o * ns + o] = qc;
    }
    std::vector<double> tmp((size_t)ns * ns, 0.0);
    for (int i = 0; i < ns; ++i)
        for (int j = 0; j < ns; ++j) {
            double v = 0.0;
            for (int k = 0; k < ns; ++k) v += Fz[(size_t)i * ns + k] * P[(size_t)k * ns + j];
            tmp[(size_t)i * ns + j] = v;
        }
    for (int i = 0; i < ns; ++i)
        for (int j = 0; j < ns; ++j) {
            double v = 0.0;
            for (int k = 0; k < ns; ++k) v += tmp[(size_t)i * ns + k] * Fz[(size_t)j * ns + k];
            Pout[(size_t)i * ns + j] = v + Qz[(size_t)i * ns + j];
        }
}

// z 空间更新：S=H P H^T+diag(R)，K=P H^T S^{-1}；joseph=1 用 Joseph，否则 P-K S K^T。
// H (nrows,nstate)、rdiag (nrows)、nu (nrows) 均 row-major。
inline void dense_update(const Params& p, const double* P, const double* H, const double* rdiag,
                         const double* nu, int joseph, double* dx, double* Pout) {
    const int ns = p.nstate, nr = p.nrows;
    std::vector<double> PHt((size_t)ns * nr, 0.0), S((size_t)nr * nr, 0.0), B((size_t)nr * ns, 0.0);
    for (int s = 0; s < ns; ++s)
        for (int r = 0; r < nr; ++r) {
            double v = 0.0;
            for (int t = 0; t < ns; ++t) v += P[(size_t)s * ns + t] * H[(size_t)r * ns + t];
            PHt[(size_t)s * nr + r] = v;
        }
    for (int r = 0; r < nr; ++r)
        for (int c = 0; c < nr; ++c) {
            double v = (r == c) ? rdiag[r] : 0.0;
            for (int s = 0; s < ns; ++s) v += H[(size_t)r * ns + s] * PHt[(size_t)s * nr + c];
            S[(size_t)r * nr + c] = v;
        }
    if (!chol(S.data(), nr)) {
        std::memset(Pout, 0, (size_t)ns * ns * sizeof(double));
        std::memset(dx, 0, (size_t)ns * sizeof(double));
        return;
    }
    for (int r = 0; r < nr; ++r)
        for (int s = 0; s < ns; ++s) B[(size_t)r * ns + s] = PHt[(size_t)s * nr + r];
    std::vector<double> X((size_t)nr * ns, 0.0);
    chol_solve(S.data(), nr, B.data(), X.data(), ns);
    std::vector<double> K((size_t)ns * nr, 0.0);
    for (int s = 0; s < ns; ++s) {
        dx[s] = 0.0;
        for (int r = 0; r < nr; ++r) {
            K[(size_t)s * nr + r] = X[(size_t)r * ns + s];
            dx[s] += K[(size_t)s * nr + r] * nu[r];
        }
    }
    if (joseph) {
        std::vector<double> A((size_t)ns * ns, 0.0), AP((size_t)ns * ns, 0.0),
            APA((size_t)ns * ns, 0.0), KRK((size_t)ns * ns, 0.0);
        for (int i = 0; i < ns; ++i) {
            A[(size_t)i * ns + i] = 1.0;
            for (int r = 0; r < nr; ++r)
                for (int j = 0; j < ns; ++j) A[(size_t)i * ns + j] -= K[(size_t)i * nr + r] * H[(size_t)r * ns + j];
        }
        for (int i = 0; i < ns; ++i)
            for (int j = 0; j < ns; ++j) {
                double v = 0.0;
                for (int k = 0; k < ns; ++k) v += A[(size_t)i * ns + k] * P[(size_t)k * ns + j];
                AP[(size_t)i * ns + j] = v;
            }
        for (int i = 0; i < ns; ++i)
            for (int j = 0; j < ns; ++j) {
                double v = 0.0;
                for (int k = 0; k < ns; ++k) v += AP[(size_t)i * ns + k] * A[(size_t)j * ns + k];
                APA[(size_t)i * ns + j] = v;
            }
        for (int i = 0; i < ns; ++i)
            for (int j = 0; j < ns; ++j) {
                double v = 0.0;
                for (int r = 0; r < nr; ++r) v += K[(size_t)i * nr + r] * rdiag[r] * K[(size_t)j * nr + r];
                KRK[(size_t)i * ns + j] = v;
            }
        for (size_t k = 0; k < (size_t)ns * ns; ++k) Pout[k] = APA[k] + KRK[k];
    } else {
        for (int i = 0; i < ns; ++i)
            for (int j = 0; j < ns; ++j) {
                double v = P[(size_t)i * ns + j];
                for (int r = 0; r < nr; ++r) v -= PHt[(size_t)i * nr + r] * K[(size_t)j * nr + r];
                Pout[(size_t)i * ns + j] = v;
            }
    }
    for (int i = 0; i < ns; ++i)
        for (int j = i + 1; j < ns; ++j) {
            const double v = 0.5 * (Pout[(size_t)i * ns + j] + Pout[(size_t)j * ns + i]);
            Pout[(size_t)i * ns + j] = Pout[(size_t)j * ns + i] = v;
        }
}

// G^T P^{-1} G（G 为 (ns,ng) row-major）；out 为 (ng,ng) row-major。
inline bool gauge_information(const double* P, const double* G, int ns, int ng, double* out) {
    std::vector<double> L(P, P + (size_t)ns * ns);
    if (!chol(L.data(), ns)) return false;
    std::vector<double> X((size_t)ns * ng, 0.0);
    chol_solve(L.data(), ns, G, X.data(), ng);
    for (int a = 0; a < ng; ++a)
        for (int b = 0; b < ng; ++b) {
            double v = 0.0;
            for (int i = 0; i < ns; ++i) v += G[(size_t)i * ng + a] * X[(size_t)i * ng + b];
            out[(size_t)a * ng + b] = v;
        }
    return true;
}

}  // namespace ref
}  // namespace ekf_central

#endif  // EKF_CENTRAL_H
