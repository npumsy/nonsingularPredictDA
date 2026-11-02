// gtest: 集中式精确协方差 EKF（ekf_central.h / src/ekf_central.cu）。
// 覆盖：16 节点单帧稠密对拍、全 N 的 PSD 与对称性、仅 ISL 帧的规范方向不变、
// Joseph 与简化形式 P-K S K^T 的对照。
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <Eigen/Dense>
#include <array>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "ekf_central.h"
#include "qoejopt.h"

using ekf_central::DU;

namespace {

constexpr double PI = 3.14159265358979323846;

template <class T>
struct DevBuf {
    T *p = nullptr;
    size_t k = 0;
    DevBuf() = default;
    explicit DevBuf(const std::vector<T> &v) { from(v); }
    void alloc(size_t n) {
        if (p) cudaFree(p);
        k = n; cudaMalloc(&p, n * sizeof(T));
    }
    void from(const std::vector<T> &v) {
        if (p) cudaFree(p);
        k = v.size(); cudaMalloc(&p, std::max<size_t>(1, k) * sizeof(T));
        if (k) cudaMemcpy(p, v.data(), k * sizeof(T), cudaMemcpyHostToDevice);
    }
    void to(std::vector<T> &v) const {
        v.resize(k);
        if (k) cudaMemcpy(v.data(), p, k * sizeof(T), cudaMemcpyDeviceToHost);
    }
    ~DevBuf() { if (p) cudaFree(p); }
};

double rel(const std::vector<double> &a, const std::vector<double> &b) {
    double d = 0.0, n = 0.0;
    for (size_t i = 0; i < a.size(); ++i) { d += (a[i] - b[i]) * (a[i] - b[i]); n += b[i] * b[i]; }
    return std::sqrt(d) / std::max(std::sqrt(n), 1e-300);
}

void inv6(const double *A, double *Ai) {
    double aug[6][12];
    for (int a = 0; a < 6; ++a) {
        for (int b = 0; b < 6; ++b) aug[a][b] = A[a * 6 + b];
        for (int b = 0; b < 6; ++b) aug[a][6 + b] = (a == b) ? 1.0 : 0.0;
    }
    for (int c = 0; c < 6; ++c) {
        int piv = c;
        for (int r = c + 1; r < 6; ++r)
            if (std::fabs(aug[r][c]) > std::fabs(aug[piv][c])) piv = r;
        for (int k = 0; k < 12; ++k) std::swap(aug[c][k], aug[piv][k]);
        const double d = aug[c][c];
        for (int k = 0; k < 12; ++k) aug[c][k] /= d;
        for (int r = 0; r < 6; ++r) if (r != c) {
            const double f = aug[r][c];
            for (int k = 0; k < 12; ++k) aug[r][k] -= f * aug[c][k];
        }
    }
    for (int a = 0; a < 6; ++a) for (int b = 0; b < 6; ++b) Ai[a * 6 + b] = aug[a][6 + b];
}

struct Sim {
    ekf_central::Params p;
    int n = 0, M = 0, E = 0, G = 0, NS = 0, NR = 0;
    std::vector<int> edges, gts;          // edges: 2E 交错；gts: 2G [node, station]
    std::vector<double> q0, b0, c0, P0z;
    std::vector<double> D_isl, M_isl, D_gts, M_gts, Qgts;
};

Sim make_sim(int n, int M, bool gts_on, unsigned seed) {
    Sim s; s.n = n; s.M = M;
    std::mt19937 gen(seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    s.q0.resize((size_t)n * 6);
    // 取真实数据首帧的量级与要素范围（a≈6919 km, i≈1.2645, e≈1.6e-3）；
    // 让 u 沿轨道均布、Ω/i 略有差异，避免所有节点重合导致几何退化。
    for (int i = 0; i < n; ++i) {
        double *q = &s.q0[(size_t)i * 6];
        q[0] = 6.91940297e6 + 1e4 * nd(gen);
        q[1] = std::fmod(4.62949307 + 2 * PI * i / n + 0.01 * nd(gen), 2 * PI);
        if (q[1] < 0) q[1] += 2 * PI;
        q[2] = 3.28927122e-4 + 1e-5 * nd(gen);
        q[3] = 1.51676048e-3 + 1e-5 * nd(gen);
        q[4] = 1.26453803 + 1e-3 * nd(gen);
        q[5] = 1.50809716e-2 + 0.01 * nd(gen);
    }
    s.b0.assign(n, 0.0); s.c0.assign(M, 0.0);
    // 图：环 + skip-7 + 2 条长弦（N=1152 时 E=2306）
    for (int i = 0; i < n; ++i) { s.edges.push_back(i); s.edges.push_back((i + 1) % n); }
    for (int i = 0; i < n; ++i) { s.edges.push_back(i); s.edges.push_back((i + 7) % n); }
    s.edges.push_back(0); s.edges.push_back(n / 2);
    s.edges.push_back(1); s.edges.push_back(n / 2 + 1);
    s.E = (int)s.edges.size() / 2;
    // GTS：每星一颗地面站
    std::vector<std::array<double, 3>> sta(M);
    for (int k = 0; k < M; ++k) {
        const double th = 2 * PI * k / std::max(1, M);
        sta[k] = {6.371e6 * std::cos(th), 6.371e6 * std::sin(th), 1.0e6 * std::sin(2 * th)};
    }
    for (int i = 0; i < n; ++i) { s.gts.push_back(i); s.gts.push_back(i % M); }
    s.G = (int)s.gts.size() / 2;
    s.NS = 7 * n + M; s.NR = s.E + s.G;
    s.p.n = n; s.p.M = M; s.p.E = s.E; s.p.G = s.G;
    s.p.nstate = s.NS; s.p.nrows = s.NR;
    s.p.sigma_isl = 30.0; s.p.sigma_gts = 19.0;
    s.p.tau_b = 600.0; s.p.sigma_b = 0.3; s.p.tau_c = 600.0; s.p.sigma_c = 0.3;
    s.p.q_acc = 1e-6; s.p.joseph = 1; s.p.diag_every = 1; s.p.fp32 = 0;
    // z 空间先验（对角 SPD；量级已归一化）
    s.P0z.assign((size_t)s.NS * s.NS, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int a = 0; a < 6; ++a)
            s.P0z[(size_t)(7 * i + a) * s.NS + 7 * i + a] = (a == 0) ? 1e-4 : 1e-6;
        s.P0z[(size_t)(7 * i + 6) * s.NS + 7 * i + 6] = 1e-2;
    }
    for (int k = 0; k < M; ++k) s.P0z[(size_t)(7 * n + k) * s.NS + 7 * n + k] = 1e-2;
    // 观测（b=c=0）
    std::vector<double> rv((size_t)n * 6), J((size_t)n * 36);
    qoejopt::oe2rvJacHost(s.q0.data(), n, rv.data(), J.data());
    s.D_isl.resize(s.E); s.M_isl.assign(s.E, 1.0);
    for (int e = 0; e < s.E; ++e) {
        const int i = s.edges[2 * e], j = s.edges[2 * e + 1];
        double d2 = 0.0;
        for (int k = 0; k < 3; ++k) { const double d = rv[i * 6 + k] - rv[j * 6 + k]; d2 += d * d; }
        s.D_isl[e] = std::sqrt(d2) + 1e-2 * nd(gen);
    }
    s.D_gts.resize(s.G); s.M_gts.assign(s.G, gts_on ? 1.0 : 0.0);
    s.Qgts.resize((size_t)s.G * 3);
    for (int k = 0; k < s.G; ++k) {
        const int i = s.gts[2 * k], si = s.gts[2 * k + 1];
        for (int a = 0; a < 3; ++a) s.Qgts[(size_t)k * 3 + a] = sta[si][a];
        double d2 = 0.0;
        for (int a = 0; a < 3; ++a) { const double d = rv[i * 6 + a] - sta[si][a]; d2 += d * d; }
        s.D_gts[k] = std::sqrt(d2) + 1e-2 * nd(gen);
    }
    return s;
}

// 参考 H（z 空间）与 nu；J = ∂rv/∂q，rv/J 由 qoejopt::oe2rvJacHost 给出。
void build_H(const Sim &s, const std::vector<double> &rv, const std::vector<double> &J,
             std::vector<double> &H, std::vector<double> &nu) {
    H.assign((size_t)s.NR * s.NS, 0.0); nu.assign(s.NR, 0.0);
    auto sc = [](int a) { return a == 0 ? DU : 1.0; };
    for (int e = 0; e < s.E; ++e) {
        if (s.M_isl[e] <= 0.5) continue;
        const int i = s.edges[2 * e], j = s.edges[2 * e + 1];
        double d[3], dist = 0.0;
        for (int k = 0; k < 3; ++k) { d[k] = rv[i * 6 + k] - rv[j * 6 + k]; dist += d[k] * d[k]; }
        dist = std::sqrt(dist);
        double u[3]; for (int k = 0; k < 3; ++k) u[k] = d[k] / (dist + 1e-12);
        for (int a = 0; a < 6; ++a) {
            double gi = 0.0, gj = 0.0;
            for (int k = 0; k < 3; ++k) {
                gi += u[k] * J[(size_t)i * 36 + k * 6 + a];
                gj += u[k] * J[(size_t)j * 36 + k * 6 + a];
            }
            H[(size_t)e * s.NS + 7 * i + a] = sc(a) * gi;
            H[(size_t)e * s.NS + 7 * j + a] = -sc(a) * gj;
        }
        H[(size_t)e * s.NS + 7 * i + 6] = 1.0;
        H[(size_t)e * s.NS + 7 * j + 6] = -1.0;
        nu[e] = s.D_isl[e] - dist;
    }
    for (int k = 0; k < s.G; ++k) {
        const int row = s.E + k;
        if (s.M_gts[k] <= 0.5) continue;
        const int i = s.gts[2 * k];
        double d[3], dist = 0.0;
        for (int a = 0; a < 3; ++a) {
            d[a] = rv[(size_t)i * 6 + a] - s.Qgts[(size_t)k * 3 + a]; dist += d[a] * d[a];
        }
        dist = std::sqrt(dist);
        double u[3]; for (int a = 0; a < 3; ++a) u[a] = d[a] / (dist + 1e-12);
        for (int a = 0; a < 6; ++a) {
            double gi = 0.0;
            for (int t = 0; t < 3; ++t) gi += u[t] * J[(size_t)i * 36 + t * 6 + a];
            H[(size_t)row * s.NS + 7 * i + a] = sc(a) * gi;
        }
        H[(size_t)row * s.NS + 7 * i + 6] = 1.0;
        H[(size_t)row * s.NS + 7 * s.n + s.gts[2 * k + 1]] = -1.0;
        nu[row] = s.D_gts[k] - dist;
    }
}

// 规范方向 G（z 空间，NS x 4）：3 平移 + RAAN。ISL 对二者完全不变。
void build_gauge(const Sim &s, const std::vector<double> &J, std::vector<double> &G) {
    G.assign((size_t)s.NS * 4, 0.0);
    for (int i = 0; i < s.n; ++i) {
        double Ji[36]; inv6(&J[(size_t)i * 36], Ji);
        for (int a = 0; a < 6; ++a)
            for (int t = 0; t < 3; ++t)
                G[(size_t)(7 * i + a) * 4 + t] = Ji[a * 6 + t] / (a == 0 ? DU : 1.0);
        G[(size_t)(7 * i + 5) * 4 + 3] = 1.0;   // δΩ（QOE 第 5 分量）
    }
}

struct CaseRun {
    std::vector<double> Ppred, qpred, Ppost;
    ekf_central::PredictResult pr;
    ekf_central::UpdateResult up;
};

CaseRun run_case(const Sim &s, int joseph, int do_diag, int frame = 1) {
    CaseRun r;
    ekf_central::Params p = s.p; p.joseph = joseph;
    DevBuf<double> q(s.q0), b(s.b0), c(s.c0), P(s.P0z);
    ekf_central::predict(p, s.edges.data(), s.gts.data(), nullptr, q.p, b.p, c.p, P.p,
                         1, frame, do_diag, &r.pr);
    P.to(r.Ppred); q.to(r.qpred);
    ekf_central::update(s.D_isl.data(), s.M_isl.data(), s.D_gts.data(), s.M_gts.data(),
                        s.Qgts.data(), frame, do_diag, &r.up);
    P.to(r.Ppost);
    return r;
}

std::vector<double> ref_predict(const Sim &s, const CaseRun &r) {
    std::vector<double> out((size_t)s.NS * s.NS);
    ekf_central::ref::dense_predict(s.p, s.P0z.data(), r.pr.Phi.data(), r.pr.Q.data(),
                                    r.pr.phi_b, r.pr.phi_c, r.pr.qc, out.data());
    return out;
}

std::vector<double> ref_update(const Sim &s, const std::vector<double> &Ppred,
                               const std::vector<double> &H, const std::vector<double> &nu,
                               int joseph, std::vector<double> &dx) {
    std::vector<double> rd(s.NR, 1.0), out((size_t)s.NS * s.NS);
    for (int e = 0; e < s.E; ++e) rd[e] = (s.M_isl[e] > 0.5) ? s.p.sigma_isl * s.p.sigma_isl : 1.0;
    for (int k = 0; k < s.G; ++k)
        rd[s.E + k] = (s.M_gts[k] > 0.5) ? s.p.sigma_gts * s.p.sigma_gts : 1.0;
    dx.assign(s.NS, 0.0);
    ekf_central::ref::dense_update(s.p, Ppred.data(), H.data(), rd.data(), nu.data(), joseph,
                                   dx.data(), out.data());
    return out;
}

double min_eig_sym(const std::vector<double> &P, int ns) {
    Eigen::MatrixXd M(ns, ns);
    for (int i = 0; i < ns; ++i)
        for (int j = 0; j < ns; ++j) M(i, j) = 0.5 * (P[(size_t)i * ns + j] + P[(size_t)j * ns + i]);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(M, Eigen::EigenvaluesOnly);
    return es.eigenvalues().minCoeff();
}

// ---------------------------------------------------------------------------
TEST(EkfCentral, SingleFrame16MatchesDenseKF) {
    const Sim s = make_sim(16, 2, true, 1);
    const CaseRun r = run_case(s, 1, 2);

    // 预测（用 C++ 回传的 Φ/Q，验证块装配与 Q 归一化）
    const std::vector<double> Pref = ref_predict(s, r);
    EXPECT_LT(rel(r.Ppred, Pref), 1e-10);

    // H/nu（在预测态 qpred 独立装配）
    std::vector<double> rv((size_t)s.n * 6), J((size_t)s.n * 36);
    qoejopt::oe2rvJacHost(r.qpred.data(), s.n, rv.data(), J.data());
    std::vector<double> H, nu;
    build_H(s, rv, J, H, nu);
    double dh = 0.0, hn = 0.0;
    for (int row = 0; row < s.NR; ++row)
        for (int col = 0; col < s.NS; ++col) {
            const double hc = r.up.H_colT[(size_t)col * s.NR + row];
            dh = std::max(dh, std::fabs(hc - H[(size_t)row * s.NS + col]));
            hn += H[(size_t)row * s.NS + col] * H[(size_t)row * s.NS + col];
        }
    EXPECT_LT(dh / std::sqrt(hn), 1e-10);
    EXPECT_LT(rel(r.up.nu, nu), 1e-10);

    // dx / P+（K 的作用由 dx 与 P+ 覆盖；K 的弱方向分量对二者无贡献）
    std::vector<double> dxr;
    const std::vector<double> Ppost_ref = ref_update(s, r.Ppred, H, nu, 1, dxr);
    EXPECT_LT(rel(r.up.dx, dxr), 1e-8);
    EXPECT_LT(rel(r.Ppost, Ppost_ref), 1e-10);
    EXPECT_GT(r.up.S_min_eig, 0.0);
    EXPECT_EQ(r.up.potrf_info, 0);
}

TEST(EkfCentral, FullNPSDAndSymmetry) {
    const Sim s = make_sim(1152, 4, true, 7);
    const CaseRun r = run_case(s, 1, 1);
    EXPECT_EQ(r.up.potrf_info, 0);
    EXPECT_GT(r.up.min_eig, 0.0);
    EXPECT_LT(r.up.sym_rel, 1e-12);
    EXPECT_TRUE(std::isfinite(r.up.S_cond));
    std::cout << "[fullN] min_eig(P+)=" << r.up.min_eig << " sym=" << r.up.sym_rel
              << " S_cond=" << r.up.S_cond << " upd_ms=" << r.up.total_ms
              << " pred_fpft_ms=" << r.pr.fpft_ms << std::endl;
}

TEST(EkfCentral, IslOnlyGaugeInvariant) {
    const Sim s = make_sim(16, 2, false, 3);
    const CaseRun r = run_case(s, 1, 2);

    std::vector<double> rv((size_t)s.n * 6), J((size_t)s.n * 36);
    qoejopt::oe2rvJacHost(r.qpred.data(), s.n, rv.data(), J.data());
    std::vector<double> G;
    build_gauge(s, J, G);

    // H G 应在机器精度内为零（ISL 对共同平移/RAAN 不变）
    std::vector<double> H, nu;
    build_H(s, rv, J, H, nu);
    double hg = 0.0;
    for (int g = 0; g < 4; ++g) {
        double gmax = 0.0; int grow = -1;
        for (int row = 0; row < s.NR; ++row) {
            double v = 0.0;
            for (int col = 0; col < s.NS; ++col) v += H[(size_t)row * s.NS + col] * G[(size_t)col * 4 + g];
            if (std::fabs(v) > gmax) { gmax = std::fabs(v); grow = row; }
        }
        std::cout << "[dbg] H*G dir=" << g << " max=" << gmax << " row=" << grow
                  << ((grow >= 0 && grow < s.E) ? " ISL" : " GTS") << std::endl;
        hg = std::max(hg, gmax);
    }
    EXPECT_LT(hg, 1e-6);

    auto info = [&](const std::vector<double> &P) {
        std::vector<double> Pj = P;
        for (int i = 0; i < s.NS; ++i) Pj[(size_t)i * s.NS + i] += 1e-14;  // 近奇异方向防 Cholesky
        std::vector<double> out(16, 0.0);
        EXPECT_TRUE(ekf_central::ref::gauge_information(Pj.data(), G.data(), s.NS, 4, out.data()));
        return out;
    };
    const std::vector<double> Lm = info(r.Ppred), Lp = info(r.Ppost);
    EXPECT_LT(rel(Lp, Lm), 1e-8);
}

TEST(EkfCentral, Fp32PreconditionedMatchesFp64) {
    const Sim s64 = make_sim(16, 2, true, 13);
    const CaseRun r64 = run_case(s64, 0, 2);
    Sim s32 = s64; s32.p.fp32 = 1;
    const CaseRun r32 = run_case(s32, 0, 2);
    const double rk = rel(r32.up.K, r64.up.K);
    const double rp = rel(r32.Ppost, r64.Ppost);
    const double rdx = rel(r32.up.dx, r64.up.dx);
    const double me = min_eig_sym(r32.Ppost, s32.NS);
    double dmax = 0.0;
    for (int i = 0; i < s32.NS; ++i) dmax = std::max(dmax, r32.Ppost[(size_t)i * s32.NS + i]);
    std::cout << "[fp32] relK=" << rk << " relP=" << rp << " rel_dx=" << rdx
              << " min_eig=" << me << " max_diag=" << dmax << std::endl;
    // K 的弱方向受 FP32 S 求解限制；状态 dx 由 FP64 单 RHS 解给出，P 误差 <2e-5。
    EXPECT_LT(rk, 1.0);
    EXPECT_LT(rp, 2e-5);
    EXPECT_LT(rdx, 1e-8);
    EXPECT_GE(me, -1e-9 * dmax);
}

TEST(EkfCentral, FullNFp32PSD) {
    const Sim s = make_sim(1152, 4, true, 17);
    Sim s32 = s; s32.p.fp32 = 1;
    const CaseRun r = run_case(s32, 0, 1);
    std::cout << "[fullN/fp32] min_eig=" << r.up.min_eig << " sym=" << r.up.sym_rel
              << " upd_ms=" << r.up.total_ms << " potrf_info=" << r.up.potrf_info
              << " S_cond=" << r.up.S_cond << std::endl;
    EXPECT_GT(r.up.min_eig, 0.0);
    EXPECT_EQ(r.up.potrf_info, 0);
    EXPECT_LT(r.up.sym_rel, 1e-12);
}

TEST(EkfCentral, FullNFp32Timing) {
    Sim s = make_sim(1152, 4, true, 19); s.p.fp32 = 1;
    const CaseRun r = run_case(s, 0, 0);
    std::cout << "[fullN/fp32-timing] upd_ms=" << r.up.total_ms
              << " fpft_ms=" << r.pr.fpft_ms << std::endl;
    EXPECT_LT(r.up.total_ms, 200.0);
}

TEST(EkfCentral, JosephVsSimplified) {
    const Sim s = make_sim(16, 2, true, 5);
    const CaseRun rj = run_case(s, 1, 2);
    const CaseRun rs = run_case(s, 0, 2);
    EXPECT_LT(rel(rj.Ppred, rs.Ppred), 1e-12);      // 同一预测
    EXPECT_LT(rel(rj.Ppost, rs.Ppost), 1e-8);       // 两种更新形式应一致
    EXPECT_GT(min_eig_sym(rj.Ppost, s.NS), -1e-10);
    EXPECT_GT(min_eig_sym(rs.Ppost, s.NS), -1e-10);
    std::cout << "[joseph] rel=" << rel(rj.Ppost, rs.Ppost)
              << " t_joseph=" << rj.up.total_ms << "ms t_simple=" << rs.up.total_ms << "ms"
              << std::endl;
}

}  // namespace

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    const int rc = RUN_ALL_TESTS();
    ekf_central::release();
    return rc;
}
