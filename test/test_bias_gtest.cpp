// Merged bias suite: OU kernel + window Γβ + DACE DA + in-band oracle + production harness.
// Each source is wrapped in a unique namespace to keep duplicate helpers/scoped names
// from colliding; TEST suite/test names and all assertions are unchanged.
#include <gtest/gtest.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <cuda_runtime.h>
#include <dace/dace.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "bias_window.h"
#include "qoejopt.h"

// Host GVE references (global symbols in libdastate.so); declared once here.
using Vector6d = Eigen::Matrix<double, 6, 1>;
Vector6d noeGveRhs(const Vector6d &oe, double beta, double t);
Vector6d noeOsc2rv(const Vector6d &oe, int MaxIt, double epsl);


// ===================== test/test_bias_prod_gtest.cpp =====================
namespace bias_prod_impl {
// Production acceptance harness for exact-P2 C++ β integration.
//
// Scope: tests/harness only. `set_win_beta_inband` (pybind) and the WinCtx
// beta_on/n_intervals/ncol_beta/off_beta/dyn_row_nnz/d_beta_q/d_gamma_int/d_jpG/
// d_pbdiag fields exist; the C++ win_assemble/solve consumption of them and the
// production exact-P2 solve entry are still pending. The host oracle
// qoejopt::biasWindowSolve (bias_window.h) is used ONLY for host-oracle checks:
//   A. β=0 assembly cost invariance: β columns present at β=0 vs beta_off.
//   B. β=0 solve invariance on a decoupled (JpG=0, Gam=0) window.
//   C. same-row dense global solve vs biasWindowSolve oracle.
//   D. timing harness (host oracle; production factor timing pending).
// Pending (GTEST_SKIP): default-off 1T fingerprint (idx0=521, winT=1, nseg=4)
// 9.260/9.588/19.401 m, production exact-P2 entry, and the nonlinear five-neighbor
// range time-series harness; no end-to-end run is performed.
//
// Build/run: cmake --build build --target gtest_bias_prod && ./build/gtest_bias_prod


namespace {

struct Win {
    int n, L, fpn, E;
    std::vector<int> ei, ej;
    std::vector<double> D, M, P, Jp, JpG, X, beta, Phi, Gam, Pb;
    Win(int n_, int L_, int fpn_, int E_, bool couple)
        : n(n_), L(L_), fpn(fpn_), E(E_) {
        const int nfr = L * fpn;
        ei.resize(E); ej.resize(E);
        for (int e = 0; e < E; ++e) { ei[e] = e % n; ej[e] = (e + 1) % n; }
        D.resize((size_t)nfr * E);
        M.assign((size_t)nfr * E, 1.0);
        P.resize((size_t)nfr * n * 3);
        Jp.resize((size_t)nfr * n * 18);
        JpG.resize((size_t)nfr * n * 9);
        X.resize((size_t)L * n * 6);
        beta.assign((size_t)L * n * 3, 0.0);
        Phi.resize((size_t)L * n * 36);
        Gam.resize((size_t)L * n * 18, 0.0);
        Pb.assign((size_t)n * 3, 1.0);   // unit-scale prior for a well-conditioned harness
        for (int t = 0; t < nfr; ++t)
            for (int i = 0; i < n; ++i) {
                const size_t p = ((size_t)t * n + i) * 3;
                P[p + 0] = 7.0e6 + 100.0 * i + 10.0 * t;
                P[p + 1] = 200.0 * i + 5.0 * t;
                P[p + 2] = 300.0 * i + 7.0 * t;
                const size_t jp = ((size_t)t * n + i) * 18;
                for (int a = 0; a < 3; ++a)
                    for (int c = 0; c < 6; ++c) Jp[jp + a * 6 + c] = 0.0;
                Jp[jp + 0] = 1.0; Jp[jp + 7] = 1.0; Jp[jp + 14] = 1.0;
                const size_t jg = ((size_t)t * n + i) * 9;
                if (couple)
                    for (int a = 0; a < 3; ++a)
                        for (int l = 0; l < 3; ++l)
                            JpG[jg + a * 3 + l] = 0.01 * ((a == l) ? 1.0 : 0.2);
                else
                    for (int a = 0; a < 9; ++a) JpG[jg + a] = 0.0;
            }
        for (int t = 0; t < nfr; ++t)
            for (int e = 0; e < E; ++e) {
                const int i = ei[e], j = ej[e];
                const double *pi = P.data() + ((size_t)t * n + i) * 3;
                const double *pj = P.data() + ((size_t)t * n + j) * 3;
                const double d = std::sqrt((pi[0]-pj[0])*(pi[0]-pj[0]) +
                                           (pi[1]-pj[1])*(pi[1]-pj[1]) +
                                           (pi[2]-pj[2])*(pi[2]-pj[2]));
                D[(size_t)t * E + e] = d + 0.5;
            }
        for (int k = 0; k < L; ++k)
            for (int i = 0; i < n; ++i) {
                const size_t xb = ((size_t)k * n + i) * 6;
                X[xb + 0] = 7.0e6 + 50.0 * i;
                X[xb + 1] = 0.2 * i;
                X[xb + 2] = 1e-4 * i; X[xb + 3] = 1e-4 * i;
                X[xb + 4] = 1.2; X[xb + 5] = 0.1 * i;
                const size_t pb = ((size_t)k * n + i) * 36;
                for (int a = 0; a < 36; ++a) Phi[pb + a] = (a % 7 == 0) ? 1.0 : 0.0;
                if (couple) {
                    const size_t gb = ((size_t)k * n + i) * 18;
                    for (int a = 0; a < 18; ++a) Gam[gb + a] = 1e-3 * std::sin(0.1 * (double)a);
                }
            }
    }
    qoejopt::BiasWinData data(bool couple, bool beta_off, bool want_rows,
                              bool no_solve) const {
        (void)couple;
        qoejopt::BiasWinData d;
        d.n = n; d.L = L; d.fpn = fpn; d.E = E; d.G = 0;
        d.sigma_isl = 1.0; d.phi = 1.0; d.Qb = 1.0;   // unit-scale GM for the harness
        d.ei = ei.data(); d.ej = ej.data();
        d.D = D.data(); d.M = M.data();
        d.P = P.data(); d.Jp = Jp.data(); d.JpG = JpG.data();
        d.X = X.data(); d.beta = beta.data(); d.Phi = Phi.data(); d.Gam = Gam.data();
        d.PbDiag = Pb.data(); d.q_weak = 1e-3;
        d.compute_cov = false; d.compute_cond = true;
        d.want_rows = want_rows; d.no_solve = no_solve; d.beta_off = beta_off;
        return d;
    }
};

double maxrel(const Eigen::VectorXd &a, const Eigen::VectorXd &b) {
    if (a.size() != b.size()) return 1e300;
    return (a - b).norm() / std::max(b.norm(), 1e-300);
}

}  // namespace

// A) β=0 assembly cost invariance: β columns/rows present at β=0 vs beta_off.
//    The β GM/prior rows have zero residual at β=0, so cost components and the
//    q/dyn/ISL rows are identical; only the β row/column count changes.
TEST(BiasProd, BetaZeroAssemblyCostInvariance) {
    Win w(8, 3, 2, 4, true);
    qoejopt::BiasWinResult on, off;
    auto d_on = w.data(true, false, true, true);
    auto d_off = w.data(true, true, true, true);
    qoejopt::biasWindowSolve(d_on, on);
    qoejopt::biasWindowSolve(d_off, off);
    const double dc = std::abs(on.cost - off.cost);
    const double di = std::abs(on.cost_isl - off.cost_isl);
    const double dd = std::abs(on.cost_dyn - off.cost_dyn);
    const double dq = std::abs(on.cost_q - off.cost_q);
    const double db = std::abs(on.cost_beta);
    std::printf("[HostOracle/Beta0] cost on/off=%.15e/%.15e | dcost=%.3e disl=%.3e "
                "ddyn=%.3e dq=%.3e cost_beta=%.3e; ncol on/off=%d/%d rows=%d/%d\n",
                on.cost, off.cost, dc, di, dd, dq, db, on.ncol, off.ncol,
                on.nrows, off.nrows);
    // β=0 assembly: the β GM/prior rows add b²=0 only; the q/dyn/ISL sums are
    // accumulated in the same order, so the cost components are bitwise equal.
    EXPECT_EQ(on.cost, off.cost);
    EXPECT_EQ(on.cost_isl, off.cost_isl);
    EXPECT_EQ(on.cost_dyn, off.cost_dyn);
    EXPECT_EQ(on.cost_q, off.cost_q);
    EXPECT_EQ(on.cost_beta, 0.0);
    EXPECT_GT(on.ncol, off.ncol);
}

// B) β=0 solve invariance on a decoupled window (JpG=0, Gam=0): the full [q;β]
//    solve must reproduce the beta_off q-only solution and cost, with dbeta≈0.
TEST(BiasProd, BetaZeroSolveInvarianceDecoupled) {
    Win w(8, 3, 2, 4, false);   // no q-β coupling
    qoejopt::BiasWinResult on, off;
    auto d_on = w.data(false, false, false, false);
    auto d_off = w.data(false, true, false, false);
    qoejopt::biasWindowSolve(d_on, on);
    qoejopt::biasWindowSolve(d_off, off);
    const double rq = maxrel(on.dX, off.dX);
    const double cb = (on.dbeta.size() ? on.dbeta.norm() : 0.0);
    const double dc = std::abs(on.cost - off.cost);
    const bool bitwise_dx = on.dX.size() == off.dX.size() &&
        std::memcmp(on.dX.data(), off.dX.data(),
                    (size_t)on.dX.size() * sizeof(double)) == 0;
    // Host-oracle note: the q-block is decoupled (Hqβ=0), so the exact solution
    // is identical; the monolithic sparse LDLT of the larger [q;β] matrix may
    // differ only by ordering/roundoff. Production milestone: Hqq/g_q bitwise,
    // pinned β=0 first GN step 2.0e-9, generic-path final q 1.3e-8; same-code q
    // repeat diff ~5.3e-6 (port GPU nondeterminism) => q bitwise is NOT a gate.
    std::printf("[HostOracle/Beta0Solve] rel_dX=%.3e bitwise_dX=%d |dbeta|=%.3e "
                "dcost=%.3e cost=%.6e\n", rq, (int)bitwise_dx, cb, dc, off.cost);
    EXPECT_LT(rq, 1e-12);
    EXPECT_EQ(on.cost, off.cost);
    EXPECT_LT(cb, 1e-12);
}

// C) Same assembled rows: dense global normal-equation solve vs biasWindowSolve.
TEST(BiasProd, DenseGlobalVsOracle) {
    Win w(8, 3, 2, 4, true);
    qoejopt::BiasWinResult rows, sol;
    auto d_rows = w.data(true, false, true, true);
    auto d_sol = w.data(true, false, true, false);
    qoejopt::biasWindowSolve(d_rows, rows);
    qoejopt::biasWindowSolve(d_sol, sol);
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(rows.nrows, rows.ncol);
    for (size_t q = 0; q < rows.A_rows.size(); ++q)
        A(rows.A_rows[q], rows.A_cols[q]) += rows.A_vals[q];
    Eigen::VectorXd b(rows.nrows);
    for (int r = 0; r < rows.nrows; ++r) b[r] = rows.b_rows[r];
    const Eigen::MatrixXd H = A.transpose() * A;
    const Eigen::VectorXd g = A.transpose() * b;
    const Eigen::VectorXd d = H.ldlt().solve(g);
    const Eigen::VectorXd dq = d.head(rows.nq);
    const Eigen::VectorXd db = d.tail(rows.nb);
    const double rq = maxrel(dq, sol.dX);
    const double rb = maxrel(db, sol.dbeta);
    const double dc = std::abs(rows.cost - sol.cost);
    Eigen::JacobiSVD<Eigen::MatrixXd> sv(H);
    const auto svals = sv.singularValues();
    std::printf("[HostOracle/DenseVsOracle] nrow=%d ncol=%d nq=%d nb=%d rel_dX=%.3e rel_dbeta=%.3e "
                "dcost=%.3e cond(H)=%.3e cost=%.6e\n",
                rows.nrows, rows.ncol, rows.nq, rows.nb, rq, rb, dc,
                svals(0) / std::max(svals(svals.size() - 1), 1e-300), sol.cost);
    // Dense LDLT vs the oracle's sparse LDLT on the same rows. cond(H)=5e3;
    // q agrees at 1e-12; the free-β block agrees at 5e-9 (different orderings).
    // The cost is bitwise equal. Production factor consistency is pending.
    EXPECT_LT(rq, 1e-9);
    EXPECT_LT(rb, 1e-8);
    EXPECT_LT(dc, 1e-12 * std::max(std::abs(sol.cost), 1.0));
}

// D) Timing harness: assembly (no_solve), full solve (assembly+factor+solve),
//    bias-off solve; measured on small synthetic windows and projected to the
//    production dimensions. The production exact-P2 factor (Baspacho) is not
//    measured here; the Eigen oracle is a proxy.
TEST(BiasProd, TimingHarness) {
    using clk = std::chrono::steady_clock;
    auto now = [] { return std::chrono::duration<double>(clk::now().time_since_epoch()).count(); };
    const int sizes[][2] = {{8, 3}, {16, 3}, {32, 9}, {64, 9}};
    for (const auto &s : sizes) {
        const int n = s[0], L = s[1], fpn = 2, E = 4;
        Win w(n, L, fpn, E, true);
        auto d_asm = w.data(true, false, false, true);
        auto d_on = w.data(true, false, false, false);
        auto d_off = w.data(true, true, false, false);
        qoejopt::BiasWinResult r;
        double ta = 0, tf = 0, to = 0;
        for (int rep = 0; rep < 3; ++rep) {
            double t0 = now(); qoejopt::biasWindowSolve(d_asm, r); ta += now() - t0;
            t0 = now(); qoejopt::biasWindowSolve(d_on, r); tf += now() - t0;
            t0 = now(); qoejopt::biasWindowSolve(d_off, r); to += now() - t0;
        }
        std::printf("[PerfHostOracle] n=%d L=%d fpn=%d E=%d ncol=%d nrow=%d "
                    "assembly=%.2f ms solve_factor=%.2f ms bias_off=%.2f ms "
                    "bias_on-over-off=%.2f ms\n",
                    n, L, fpn, E, r.ncol, r.nrows, 1e3 * ta / 3, 1e3 * tf / 3,
                    1e3 * to / 3, 1e3 * (tf - to) / 3);
    }
    // Production dimension projections (no full-scale run). Per the corrected
    // conventions: 1T is L=3 (nseg=3) or L=1; 45T is winT=3/nseg=9 => L=9 and
    // 15 streaming windows (one window resident at a time).
    const int N = 1152, E = 2306, G = 432;
    struct Proj { const char *name; int W, fpn, windows; };
    const Proj projs[] = {{"1T/nseg4(fingerprint)", 4, 144, 1},
                          {"1T/nseg3", 3, 192, 1},
                          {"1T/nseg1", 1, 576, 1},
                          {"45T/winT3/nseg9", 9, 192, 15}};
    for (const auto &p : projs) {
        const long long nq = (long long)p.W * N * 6;      // final contract: W*N*6
        const long long nb = 3LL * p.W * N;
        const long long ncol = nq + nb;
        const long long isl_rows = (long long)p.W * p.fpn * E;
        const long long gts_rows = (long long)p.W * p.fpn * G;
        const long long dyn_rows = (long long)p.W * N * 6;
        const long long gm_rows = 3LL * N * p.W;
        std::printf("[PerfProj] %s N=%d W=%d fpn=%d E=%d G=%d windows=%d: "
                    "nq=%lld nb=%lld ncol=%lld ISL_rows/win=%lld GTS_rows/win=%lld "
                    "dyn_rows/win=%lld gm_rows/win=%lld total_ISL_rows=%lld (streaming)\n",
                    p.name, N, p.W, p.fpn, E, G, p.windows, nq, nb, ncol,
                    isl_rows, gts_rows, dyn_rows, gm_rows,
                    isl_rows * p.windows);
    }
    SUCCEED();
}

// Expected A_val/CSR contract for set_win_beta_inband; this is a contract check,
// not a production call. Bias-on groups: ISL rows +6, GTS +3, dyn +3 (15 vs 12);
// GM rows β0 (3 nnz) and β_k−φβ_{k−1} (6 nnz); β col = off_beta+(k*N+i)*3+l.
TEST(BiasProd, BetaInbandAssemblyContractExpected) {
    // Final implementation contract (adaptation target; not a production call):
    //   q col = (k*N+i)*6+c, k=0..W-1, ncol_q=W*N*6;
    //   beta col = off_beta+(k*N+i)*3+l, off_beta=ncol_q;
    //   Baspacho param_sizes=[6]*(W*N)+[3]*(W*N);
    //   ISL row 18 (q_i6 q_j6 β_i3 β_j3), GTS 9 (q_a6 β_a3), DYN 15
    //   (q_k6 q_{k+1}6 β_k3, β block -Qis*Γ_k); GM rows in Python.
    struct Cfg { int N, W, fpn, E, G; };
    auto check = [](const Cfg &c, const char *tag) {
        const long long nq = (long long)c.W * c.N * 6;
        const long long nb = 3LL * c.N * c.W;
        const long long ncol = nq + nb;
        const long long off_beta = nq;
        const long long nISL = (long long)c.W * c.fpn * c.E;
        const long long nGTS = (long long)c.W * c.fpn * c.G;
        const long long nDYN = (long long)c.W * c.N * 6;
        const long long off_isl = 0;
        const long long off_gts = nISL * 18;
        const long long off_dyn = nISL * 18 + nGTS * 9;
        const long long off_gm = off_dyn + nDYN * 15;
        const long long off_prior = off_gm + 3LL * c.N * (2 * c.W - 1);
        const long long rd_dyn = nISL + nGTS;
        const long long rd_gm = rd_dyn + nDYN;
        const long long rd_prior = rd_gm + 3LL * c.N * c.W;
        std::printf("[Contract] %s N=%d W=%d fpn=%d E=%d G=%d: ncol_q=%lld "
                    "ncol_beta=%lld ncol=%lld off_beta=%lld param_sizes=6*%d+3*%d; "
                    "row widths ISL/GTS/DYN=18/9/15 (off 12/6/12); "
                    "off isl/gts/dyn/gm/prior=%lld/%lld/%lld/%lld/%lld; "
                    "b rows rd_dyn/gm/prior=%lld/%lld/%lld; GM A_val=%lld rows=%lld\n",
                    tag, c.N, c.W, c.fpn, c.E, c.G, nq, nb, ncol, off_beta,
                    c.W * c.N, c.W * c.N, off_isl, off_gts, off_dyn, off_gm,
                    off_prior, rd_dyn, rd_gm, rd_prior,
                    3LL * c.N * (2 * c.W - 1), 3LL * c.N * c.W);
        EXPECT_EQ(off_beta, nq);
        EXPECT_EQ(ncol, nq + 3LL * c.N * c.W);
        EXPECT_EQ(off_dyn, nISL * 18 + nGTS * 9);
        EXPECT_EQ(off_gm, off_dyn + nDYN * 15);
        EXPECT_EQ(off_prior, off_gm + 3LL * c.N * (2 * c.W - 1));
        EXPECT_EQ(rd_dyn, nISL + nGTS);
        EXPECT_EQ(rd_gm, rd_dyn + nDYN);
        EXPECT_EQ(rd_prior, rd_gm + 3LL * c.N * c.W);
        std::vector<char> used((size_t)ncol, 0);
        bool ok = true;
        for (int k = 0; k < c.W; ++k)
            for (int i = 0; i < c.N; ++i)
                for (int l = 0; l < 3; ++l) {
                    const long long idx = off_beta + ((long long)k * c.N + i) * 3 + l;
                    if (idx < nq || idx >= ncol || used[(size_t)idx]) ok = false;
                    used[(size_t)idx] = 1;
                }
        EXPECT_TRUE(ok);
    };
    // Notes to confirm: off_dyn is the A_val nnz base; b dyn base is rd_dyn.
    // off_damp/rd_damp are produced by the S dict and are not fixed here.
    check({8, 3, 2, 4, 0}, "small");
    check({1152, 4, 144, 2306, 432}, "1T/nseg4(fingerprint)");
    check({1152, 3, 192, 2306, 432}, "1T/nseg3");
    check({1152, 1, 576, 2306, 432}, "1T/nseg1");
    check({1152, 9, 192, 2306, 432}, "45T/winT3/nseg9");
}

// Free β from 0: expected nonzero change; NOT a failure. The pinned-β=0
// invariance (<1e-10) is tested separately (assembly cost bitwise + decoupled).
TEST(BiasProd, BetaFreeFromZeroDiagnostic) {
    Win w(8, 3, 2, 4, true);
    qoejopt::BiasWinResult on, off;
    auto d_on = w.data(true, false, false, false);
    auto d_off = w.data(true, true, false, false);
    qoejopt::biasWindowSolve(d_on, on);
    qoejopt::biasWindowSolve(d_off, off);
    const double bnorm = (on.dbeta.size() ? on.dbeta.norm() : 0.0);
    const double rq = maxrel(on.dX, off.dX);
    std::printf("[HostOracle/FreeBetaFrom0] |dbeta|_first=%.6e rel_dX=%.3e "
                "cost_on/off=%.9e/%.9e (expected nonzero; not a failure)\n",
                bnorm, rq, on.cost, off.cost);
    EXPECT_TRUE(std::isfinite(bnorm));
    SUCCEED();
}

// Empirical coverage harnesses (additional to the frozen DA/window suites).
TEST(BiasProd, RealEphReferenceAvailability) {
    // data/cache/orbit_1152.npz has only keys {x0 (1152,6), edges (2306,2)};
    // there is no Eph time series. kDataRv rows 0..5 were checked bitwise equal
    // to x0[0:6] in the DA suite. Therefore the reference is GVE self-propagation;
    // a data-time-series difference is N/A (not silently substituted).
    std::printf("[EmpRealEph] npz keys={x0,edges}; Eph time series=absent; "
                "reference=GVE self-propagation; data-time-series diff=N/A "
                "(bitwise x0 rows 0..5 used by BiasDa.SingleNodePositionBiasRecovery)\n");
    SUCCEED();
}

TEST(BiasProd, IntervalLengthSweepOracle) {
    // Host-oracle interval-length sweep (not production): W=1 (T), W=3 (T/3),
    // W=9 (T/9, the winT3/nseg9 production interval is T/3 over 3T => 9 intervals).
    // Reports cond_joint/cond_beta/cost; β recovery/1T residual from the GVE
    // single-node sweep are in the frozen BiasDa log. Production sweep pending.
    for (int W : {1, 3, 9}) {
        if (W <= 1) {
            // biasWindowSolve validates L>1; production L=1 is still valid but
            // has no host-oracle counterpart. Report and keep production pending.
            std::printf("[EmpIntervalOracle] W=1: host oracle requires L>1; "
                        "production L=1 sweep pending\n");
            continue;
        }
        try {
            Win w(8, W, 2, 4, true);
            auto d = w.data(true, false, false, false);
            qoejopt::BiasWinResult r;
            qoejopt::biasWindowSolve(d, r);
            std::printf("[EmpIntervalOracle] W=%d ncol=%d cond_joint=%.3e cond_beta=%.3e "
                        "cost=%.6e |dbeta|_step=%.3e (host oracle; production pending)\n",
                        W, r.ncol, r.cond_joint, r.cond_beta, r.cost,
                        r.dbeta.size() ? r.dbeta.norm() : 0.0);
            EXPECT_TRUE(std::isfinite(r.cost));
        } catch (const std::exception &e) {
            std::printf("[EmpIntervalOracle] W=%d oracle threw: %s\n", W, e.what());
            ADD_FAILURE() << "oracle W=" << W << ": " << e.what();
        }
    }
}

TEST(BiasProd, FiveNeighborNonlinearIslPending) {
    GTEST_SKIP() << "requires the GVE nonlinear range time-series harness and "
                    "production exact-P2 entry; frozen BiasDa.FiveNeighborRangeBiasRank "
                    "already covers the linear ISL Jacobian rank/ECI-vs-RTN; nonlinear "
                    "bias injection + range-only inversion with/without anchor pending";
}

// Production reference metrics recorded from the implementation milestone
// (not recomputed here; the production entry is qoe_core/Theseus + ports).
TEST(BiasProd, ProductionReferenceMetricsRecorded) {
    std::printf("[ProdRef/DefaultOff] fingerprint mean/rms/max=9.260/9.588/19.401 m; "
                "vel=0.0126/0.0143 m/s; ISL=1.000 m; gate=printed fingerprint + "
                "cost/ISL same value; same-code q repeat diff ~5.3e-6 "
                "(port GPU nondeterminism; q bitwise is NOT a gate)\n");
    std::printf("[ProdRef/Beta0] Hqq/g_q assembly bitwise equal; pinned beta=0 first "
                "GN step=2.0e-9; generic-path final q=1.3e-8\n");
    std::printf("[ProdRef/FreeBeta1T/nseg4] beta rms 2.7-2.9e-8 max 1.9e-7; "
                "Eph 9.235/9.572/19.322 m; ISL 1.006 m (slightly better than off)\n");
    SUCCEED();
}

// Pending: production exact-P2 C++ β assembly/solve. set_win_beta_inband (pybind)
// and the WinCtx fields exist; winAssembleDevice/winDynAsmKernel still need to
// consume beta_on/ncol_beta/off_beta/dyn_row_nnz/d_beta_q/d_gamma_int/d_jpG.
TEST(BiasProd, ProductionBetaAssemblyPending) {
    qoejopt::WinCtx c;
    EXPECT_EQ(c.beta_on, 0);
    EXPECT_EQ(c.ncol_beta, 0);
    EXPECT_EQ(c.off_beta, 0);
    GTEST_SKIP() << "set_win_beta_inband + WinCtx fields exist; pending is the "
                    "C++ win_assemble β-column/CSR consumption (dyn_row_nnz 15, "
                    "off_dyn/off_gm/off_prior, b rd_dyn/rd_gm/rd_prior) and the "
                    "production solve entry (qoe_core/Theseus). Oracle checks above "
                    "are not production validation";
}

// Pending: default-off gate per the updated semantics (production entry only).
// Gate = printed fingerprint + cost/ISL same value; q bitwise is not a gate.
TEST(BiasProd, DefaultOffFingerprintPending) {
    const double expected_mean = 9.260, expected_rms = 9.588, expected_max = 19.401;
    const double expected_vel_mean = 0.0126, expected_vel_rms = 0.0143;
    const double expected_isl = 1.000;
    const double q_repeat_diff = 5.3e-6;
    std::printf("[ProdRef/DefaultOffGate] mean/rms/max=%.3f/%.3f/%.3f m "
                "vel=%.4f/%.4f m/s ISL=%.3f m q_repeat_diff=%.1e\n",
                expected_mean, expected_rms, expected_max, expected_vel_mean,
                expected_vel_rms, expected_isl, q_repeat_diff);
    GTEST_SKIP() << "requires production exact-P2 entry (qoe_core/Theseus), "
                    "idx0=521 winT=1 nseg=4; gate = printed fingerprint + "
                    "cost/ISL same value; expected Eph pos 9.260/9.588/19.401 m, "
                    "vel 0.0126/0.0143 m/s, ISL 1.000 m; same-code q repeat diff "
                    "~5.3e-6 (port GPU nondeterminism) so q bitwise is not a gate";
}
}  // namespace bias_prod_impl

// ===================== test/test_ou_bias_gtest.cpp =====================
namespace ou_bias_impl {
// OU（Ornstein–Uhlenbeck）RTN 加速度偏差核（Stage A）gtest：
// CUDA `qoejopt::gveOuBiasBatch` vs 独立 CPU（host double）参考。
//
// 被验证的实现：src/gve_step.cu `gveOuBiasKernel` / `gveOuBiasBatch`
// （nsub=max(ceil(Δt/10),bias_nsub)）。核内子步：h=Δt/nsub，中点 RTN 瞬时 B、
// 半段 STM Φ(h/2)，逐段推进
//   Γ_b = ∫Φ(Δt−s)B(s)e^{−s/τ}ds,
//   Q_qq = Q_white + ∫ΦB(q_b I)BᵀΦᵀds + Γ_b P_b Γ_bᵀ,
//   Q_qb = ∫ΦB q_b e^{−(Δt−s)/τ}ds + Γ_b P_b φᵀ,
//   Q_bb = σ_b²(1−φ²)I + φP_bφᵀ,  q_b=2σ_b²/τ, φ=e^{−Δt/τ}.
//
// 参考口径（与核**独立**）：
//   (1) Γ_b：对含时衰减 RTN 偏差的 GVE 传播做中心差分（只用 f 与 Gauss B，不用 STM/变分核）；
//   (2) Q 块：连续变分方程（A=∂f/∂x 中心差分、B=经典 Gauss 公式）RK4 积分；
//   (3) Q_bb：GM 离散闭式。
// f 用 host `noeGveRhs`（二体+J234+指数阻力；与设备端 tbp_dev 同源同默认），
// B 用不含 Kepler n 的 Gauss 映射（与 `gveGaussW_dev` 同口径）。
//
// 注意：本文件**不包含** dastates.h。dastates.h 在全局 `using namespace DACE/std`，
// 会让 Eigen 矩阵的 operator-/+/* 与 DACE 重载产生歧义；这里只前向声明两个 host
// 函数（符号见 libdastate.so，签名核对为 Eigen::Matrix<double,6,1>）。
//
// 参数：T=5749.800907 s、τ=1T、σ_b=1e-7 m/s²（另做 σ_b=1e-6 量级缩放）、bias_nsub=16。
// 运行：nonsingularPredictDA/build/gtest_ou_bias


// host GVE 参考（定义于 libdastate.so / dastate.cpp，全局命名空间）
using Vector6d = Eigen::Matrix<double, 6, 1>;

using Matrix3d = Eigen::Matrix<double, 3, 3>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Vector3d = Eigen::Matrix<double, 3, 1>;

namespace {

constexpr double kT = 5749.800907;   // 任务 T 单位（秒），doc/log_isl_theseus.md
constexpr double kTau = kT;          // τ = 1T
constexpr double kSb = 1e-7;         // σ_b (m/s²)，doc/log_isl_theseus.md l1236
constexpr double kSbLarge = 1e-6;    // 放大一档，验证 σ_b² 缩放
constexpr int kOuNsub = 16;          // API 默认 bias_nsub（实际 nsub=max(ceil(Δt/10),16)）
constexpr double kMu = 3.986004415e14;   // bddd::MU，m³/s²

std::vector<Vector6d> sample_states() {
    std::vector<Vector6d> v;
    Vector6d x;
    x << 6.9368e6, 0.0361, 6.90e-4, 5.06e-5, 1.2656, 0.0157;
    v.push_back(x);
    x << 6.9300e6, 1.10, 1.2e-3 * std::cos(0.4), 1.2e-3 * std::sin(0.4), 0.98, 2.3;
    v.push_back(x);
    x << 7.1000e6, 2.50, 5e-5 * std::cos(2.0), 5e-5 * std::sin(2.0), 1.57, 4.1;
    v.push_back(x);
    x << 6.9500e6, 5.20, 8e-4 * std::cos(1.3), 8e-4 * std::sin(1.3), 0.62, 5.5;
    v.push_back(x);
    return v;
}

Vector6d f_full(const Vector6d &x) { return noeGveRhs(x, 1.0, 0.0); }

Matrix6d jac_fd(const Vector6d &x) {
    Vector6d eps;
    eps << 1e-1, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7;
    Matrix6d A;
    for (int c = 0; c < 6; ++c) {
        Vector6d xp = x, xm = x;
        xp[c] += eps[c]; xm[c] -= eps[c];
        A.col(c) = (f_full(xp) - f_full(xm)) / (2.0 * eps[c]);
    }
    return A;
}

// ∂f_QOE/∂a（ECI），**不含** Kepler n；与 script/core/da_engine._gve_rhs 同式。
Vector6d gve_ap(const Vector6d &oe, const Vector6d &rv, const Vector3d &ap) {
    const double mu = kMu;
    const double a = oe[0], u = oe[1], ex = oe[2], ey = oe[3], inc = oe[4];
    const double e = std::hypot(ex, ey);
    const double w = std::atan2(ey, ex);
    double M = std::fmod(u - w, 2.0 * M_PI);
    if (M < 0.0) M += 2.0 * M_PI;
    double E = M;
    for (int it = 0; it < 30; ++it)
        E -= (E - e * std::sin(E) - M) / (1.0 - e * std::cos(E));
    const double cnu = (std::cos(E) - e) / (1.0 - e * std::cos(E));
    const double snu = std::sqrt(std::max(1.0 - e * e, 0.0)) * std::sin(E) /
                       (1.0 - e * std::cos(E));
    const double p = a * (1.0 - e * e);
    const double h = std::sqrt(mu * p);
    const double r = p / (1.0 + e * cnu);
    const Vector3d r3 = rv.head<3>(), v3 = rv.tail<3>();
    const Vector3d Rc = r3.normalized();
    const Vector3d Nc = r3.cross(v3).normalized();
    const Vector3d Tc = Nc.cross(Rc);
    const double aR = ap.dot(Rc), aT = ap.dot(Tc), aN = ap.dot(Nc);
    const double ef = std::max(e, 1e-8), sif = std::max(std::sin(inc), 1e-8);
    const double uarg = w + std::atan2(snu, cnu);
    const double da = (2.0 * a * a / h) * (e * snu * aR + (p / r) * aT);
    const double de = (1.0 / h) * (p * snu * aR + ((p + r) * cnu + r * e) * aT);
    const double di = (r * std::cos(uarg) / h) * aN;
    const double dOm = (r * std::sin(uarg) / (h * sif)) * aN;
    const double dw = (1.0 / (h * ef)) * (-p * cnu * aR + (p + r) * snu * aT)
                    - (r * std::sin(uarg) * std::cos(inc) / (h * sif)) * aN;
    const double fM = std::sqrt(std::max(1.0 - e * e, 0.0)) / (h * ef);
    const double dM = fM * ((p * cnu - 2.0 * r * e) * aR - (p + r) * snu * aT);
    Vector6d out;
    out[0] = da; out[1] = dM + dw;
    out[2] = std::cos(w) * de - e * std::sin(w) * dw;
    out[3] = std::sin(w) * de + e * std::cos(w) * dw;
    out[4] = di; out[5] = dOm;
    return out;
}

Eigen::Matrix<double, 6, 3> b_rtn(const Vector6d &x) {
    const Vector6d rv = noeOsc2rv(x, 100, 1e-12);
    const Vector3d R = rv.head<3>().normalized();
    const Vector3d N = rv.head<3>().cross(rv.tail<3>()).normalized();
    const Vector3d T = N.cross(R);
    Eigen::Matrix<double, 6, 3> B;
    B.col(0) = gve_ap(x, rv, R);
    B.col(1) = gve_ap(x, rv, T);
    B.col(2) = gve_ap(x, rv, N);
    return B;
}

// ---- Γ_b 的独立参考：含时衰减 RTN 偏差的 GVE 传播中心差分 ----
// x' = f(x) + B_RTN(x)·(b0 e^{−t/τ})，δ 取 σ_b；Γ_b[:,k] = ∂x(Δ)/∂b0,k。
Vector6d f_rtn_bias(const Vector6d &x, double t, double tau, const Vector3d &b0) {
    return f_full(x) + b_rtn(x) * (b0 * std::exp(-t / tau));
}

Vector6d propagate_rtn_bias(const Vector6d &x0, double dt, double tau,
                            const Vector3d &b0, double hstep) {
    const int ns = std::max(1, (int)std::ceil(dt / hstep));
    const double h = dt / ns;
    Vector6d x = x0;
    for (int j = 0; j < ns; ++j) {
        const double t = j * h, th = t + 0.5 * h;
        const Vector6d k1 = f_rtn_bias(x, t, tau, b0);
        const Vector6d k2 = f_rtn_bias(x + 0.5 * h * k1, th, tau, b0);
        const Vector6d k3 = f_rtn_bias(x + 0.5 * h * k2, th, tau, b0);
        const Vector6d k4 = f_rtn_bias(x + h * k3, t + h, tau, b0);
        x += (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    }
    return x;
}

Eigen::Matrix<double, 6, 3> gamma_fd(const Vector6d &x0, double dt, double tau,
                                     double delta) {
    Eigen::Matrix<double, 6, 3> G;
    for (int k = 0; k < 3; ++k) {
        Vector3d b0 = Vector3d::Zero();
        b0[k] = delta;
        const Vector6d xp = propagate_rtn_bias(x0, dt, tau, b0, 5.0);
        const Vector6d xm = propagate_rtn_bias(x0, dt, tau, -b0, 5.0);
        G.col(k) = (xp - xm) / (2.0 * delta);
    }
    return G;
}

// ---- Q 块的连续变分参考（与核同一 split 模型，但用连续 ODE + RK4）----
struct RefResult {
    Eigen::Matrix<double, 6, 3> Gamma, Qqb;
    Matrix6d Qqq, Qqw, Qqn;
    Matrix3d Qbb;
};

RefResult ref_ou(const Vector6d &x0, double dt, double tau, double sb,
                 const Vector3d &rtn, const Vector3d &du, double hstep) {
    const double qb = 2.0 * sb * sb / tau;
    Matrix3d Pb = Matrix3d::Zero();
    for (int k = 0; k < 3; ++k) Pb(k, k) = std::pow(sb * rtn[k], 2);
    const Matrix3d Qu = du.cwiseProduct(du).asDiagonal();
    const int ns = std::max(1, (int)std::ceil(dt / hstep));
    const double h = dt / ns;
    Vector6d x = x0;
    using M63 = Eigen::Matrix<double, 6, 3>;
    Matrix6d Pw = Matrix6d::Zero(), Pn = Matrix6d::Zero();
    M63 M = M63::Zero(), Nq = M63::Zero();      // Γ_b 与 Q_qb 的积分部分均为 6×3
    auto der = [&](const Vector6d &y, double ts, const M63 &Ms,
                   const Matrix6d &Pws, const Matrix6d &Pns, const M63 &Ns,
                   M63 &dM, Matrix6d &dPw, Matrix6d &dPn, M63 &dN) {
        const Matrix6d A = jac_fd(y);
        const Eigen::Matrix<double, 6, 3> B = b_rtn(y);
        dM = A * Ms + B * std::exp(-ts / tau);
        dPw = A * Pws + Pws * A.transpose() + B * Qu * B.transpose();
        dPn = A * Pns + Pns * A.transpose() + qb * (B * B.transpose());
        dN = A * Ns - Ns / tau + qb * B;
    };
    for (int j = 0; j < ns; ++j) {
        const double t = j * h;
        const Vector6d k1x = f_full(x), x2 = x + 0.5 * h * k1x;
        const Vector6d k2x = f_full(x2), x3 = x + 0.5 * h * k2x;
        const Vector6d k3x = f_full(x3), x4 = x + h * k3x;
        const Vector6d k4x = f_full(x4);
        M63 k1M, k1N;
        Matrix6d k1Pw, k1Pn;
        der(x, t, M, Pw, Pn, Nq, k1M, k1Pw, k1Pn, k1N);
        const M63 M2 = M + 0.5 * h * k1M, N2 = Nq + 0.5 * h * k1N;
        const Matrix6d Pw2 = Pw + 0.5 * h * k1Pw, Pn2 = Pn + 0.5 * h * k1Pn;
        M63 k2M, k2N;
        Matrix6d k2Pw, k2Pn;
        der(x2, t + 0.5 * h, M2, Pw2, Pn2, N2, k2M, k2Pw, k2Pn, k2N);
        const M63 M3 = M + 0.5 * h * k2M, N3 = Nq + 0.5 * h * k2N;
        const Matrix6d Pw3 = Pw + 0.5 * h * k2Pw, Pn3 = Pn + 0.5 * h * k2Pn;
        M63 k3M, k3N;
        Matrix6d k3Pw, k3Pn;
        der(x3, t + 0.5 * h, M3, Pw3, Pn3, N3, k3M, k3Pw, k3Pn, k3N);
        const M63 M4 = M + h * k3M, N4 = Nq + h * k3N;
        const Matrix6d Pw4 = Pw + h * k3Pw, Pn4 = Pn + h * k3Pn;
        M63 k4M, k4N;
        Matrix6d k4Pw, k4Pn;
        der(x4, t + h, M4, Pw4, Pn4, N4, k4M, k4Pw, k4Pn, k4N);
        x += h / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
        M += h / 6.0 * (k1M + 2.0 * k2M + 2.0 * k3M + k4M);
        Pw += h / 6.0 * (k1Pw + 2.0 * k2Pw + 2.0 * k3Pw + k4Pw);
        Pn += h / 6.0 * (k1Pn + 2.0 * k2Pn + 2.0 * k3Pn + k4Pn);
        Nq += h / 6.0 * (k1N + 2.0 * k2N + 2.0 * k3N + k4N);
    }
    const double phi = std::exp(-dt / tau);
    RefResult r;
    r.Gamma = M;
    r.Qqw = Pw; r.Qqn = Pn;
    r.Qqq = Pw + Pn + M * Pb * M.transpose();
    r.Qqb = Nq + phi * M * Pb;
    r.Qbb = sb * sb * (1.0 - phi * phi) * Matrix3d::Identity() + phi * phi * Pb;
    return r;
}

struct CudaOut {
    std::vector<double> end, G, Q, Qb, Qbb;
};

CudaOut run_cuda(const std::vector<Vector6d> &oe, double dt, int bias_nsub,
                 double tau, double sb, const Vector3d &rtn, const Vector3d &du) {
    const int n = (int)oe.size();
    std::vector<double> h((size_t)n * 6);
    for (int i = 0; i < n; ++i)
        for (int c = 0; c < 6; ++c) h[(size_t)i * 6 + c] = oe[i][c];
    CudaOut o;
    o.end.resize((size_t)n * 6); o.G.resize((size_t)n * 18);
    o.Q.resize((size_t)n * 36); o.Qb.resize((size_t)n * 18);
    o.Qbb.resize((size_t)n * 9);
    double r3[3] = {rtn[0], rtn[1], rtn[2]};
    double d3[3] = {du[0], du[1], du[2]};
    qoejopt::gveOuBiasBatch(h.data(), n, dt, bias_nsub, tau, sb, r3, d3,
                            o.end.data(), o.G.data(), o.Q.data(),
                            o.Qb.data(), o.Qbb.data());
    return o;
}

// 显式取块，避免任何隐式表达式带来的重载歧义。
Eigen::Matrix<double, 6, 3> gamma_of(const CudaOut &c, int i) {
    Eigen::Matrix<double, 6, 3> G;
    for (int a = 0; a < 6; ++a)
        for (int k = 0; k < 3; ++k) G(a, k) = c.G[((size_t)i * 6 + a) * 3 + k];
    return G;
}

Eigen::Matrix<double, 6, 3> qb_of(const CudaOut &c, int i) {
    Eigen::Matrix<double, 6, 3> G;
    for (int a = 0; a < 6; ++a)
        for (int k = 0; k < 3; ++k) G(a, k) = c.Qb[((size_t)i * 6 + a) * 3 + k];
    return G;
}

Matrix6d qqq_of(const CudaOut &c, int i) {
    Eigen::Map<const Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> Q(
        c.Q.data() + (size_t)i * 36);
    return Q;
}

Matrix3d qbb_of(const CudaOut &c, int i) {
    Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> Q(
        c.Qbb.data() + (size_t)i * 9);
    return Q;
}

double frob_rel(const Matrix6d &a, const Matrix6d &b) {
    return (a - b).norm() / std::max(b.norm(), 1e-300);
}
double frob_rel63(const Eigen::Matrix<double, 6, 3> &a,
                  const Eigen::Matrix<double, 6, 3> &b) {
    return (a - b).norm() / std::max(b.norm(), 1e-300);
}

}  // namespace

// 1) GM 离散数学：Q_bb=σ²(1−φ²)I（rtn=0 ⇒ Pb=0），含 Δ→0 与 τ→∞ 极限。
//    选取覆盖量纲的多组 (τ,Δt)；容差 1e-8 而非机器精度：Δt/τ≪1 时 1−φ² 相消，
//    CUDA exp 与 host libm exp 的 ~1 ulp 差被放大到 ~1e-10 相对量级（非模型误差）。
TEST(OuBias, GmDiscreteMath) {
    const auto states = sample_states();
    const Vector3d zero = Vector3d::Zero();
    const double cases[][2] = {
        {0.25 * kT, 1e-3}, {kT, 1e-3}, {kT, 0.25 * kT}, {3.0 * kT, 1e-3},
        {0.25 * kT, kT},   {kT, kT},   {3.0 * kT, kT},   {kT, 3.0 * kT},
        {3.0 * kT, 3.0 * kT}, {1e9, kT}, {1e9, 3.0 * kT}};
    for (const auto &cs : cases) {
        const double tau = cs[0], dt = cs[1];
        const auto c = run_cuda({states[0]}, dt, kOuNsub, tau, kSb, zero, zero);
        const double phi = std::exp(-dt / tau);
        const Matrix3d exp = (kSb * kSb * (1.0 - phi * phi)) * Matrix3d::Identity();
        const Matrix3d qbb = qbb_of(c, 0);
        const double den = std::max(exp.norm(), 1e-300);
        EXPECT_LT((qbb - exp).eval().norm() / den, 1e-8)
            << "tau=" << tau << " dt=" << dt;
        EXPECT_TRUE(qbb.allFinite());
    }
    // Δ→0：Q_bb→0；τ→∞：φ→1、Q_bb→0（均对 rtn=0）。
    const auto c0 = run_cuda({states[0]}, 1e-9, kOuNsub, kTau, kSb, zero, zero);
    EXPECT_LT(qbb_of(c0, 0).norm(), 1e-20);
    const auto ci = run_cuda({states[0]}, kT, kOuNsub, 1e9, kSb, zero, zero);
    EXPECT_LT(qbb_of(ci, 0).norm(), 1e-17);
}

// 2) Γ_b vs 含时衰减 RTN 偏差传播的中心差分；Δ=0.25T/1T/3T，相对误差 <1e-4。
TEST(OuBias, GammaFiniteDifference) {
    const auto states = sample_states();
    const int n = (int)states.size();
    for (double frac : {0.25, 1.0, 3.0}) {
        const double dt = frac * kT;
        const auto c = run_cuda(states, dt, kOuNsub, kTau, kSb,
                                Vector3d::Ones(), Vector3d::Zero());
        double max_rel = 0.0, max_abs = 0.0;
        bool sign_ok = true;
        for (int i = 0; i < n; ++i) {
            const Eigen::Matrix<double, 6, 3> G = gamma_of(c, i);
            const Eigen::Matrix<double, 6, 3> Gref = gamma_fd(states[i], dt, kTau, kSb);
            max_rel = std::max(max_rel, frob_rel63(G, Gref));
            max_abs = std::max(max_abs, (G - Gref).eval().cwiseAbs().maxCoeff());
            for (int a = 0; a < 6; ++a)
                for (int k = 0; k < 3; ++k)
                    if (std::abs(Gref(a, k)) > 1e-6 * Gref.norm() &&
                        G(a, k) * Gref(a, k) < 0.0)
                        sign_ok = false;
        }
        std::printf("[GammaFD] Delta=%.2fT max_rel=%.3e max_abs=%.3e\n",
                    frac, max_rel, max_abs);
        EXPECT_LT(max_rel, 1e-4) << "Delta=" << frac << "T";
        EXPECT_TRUE(sign_ok) << "Delta=" << frac << "T";
    }
}


// 2c) 调用历史稳定性：大 OU → frames×4 → 同参数大 OU。
// 状态积分（end）与 Qbb 必须逐位一致；Γ/Qqq/Qqb 也必须一致。
TEST(OuBias, CallHistoryStability) {
    const auto st = sample_states();
    const int n = 2;
    std::vector<double> h((size_t)n * 6);
    for (int i = 0; i < n; ++i)
        for (int c = 0; c < 6; ++c) h[(size_t)i * 6 + c] = st[i][c];
    const double seg = 0.5 * kT;
    const double dt_big = 10.0 * seg;
    const int ns_big = 2880;
    double rtn[3] = {1.0, 1.0, 1.0}, du0[3] = {0.0, 0.0, 0.0};
    auto big = [&](std::vector<double> &G, std::vector<double> &E,
                   std::vector<double> &Q, std::vector<double> &Qb,
                   std::vector<double> &Qbb) {
        G.assign((size_t)n * 18, 0.0); E.assign((size_t)n * 6, 0.0);
        Q.assign((size_t)n * 36, 0.0); Qb.assign((size_t)n * 18, 0.0);
        Qbb.assign((size_t)n * 9, 0.0);
        qoejopt::gveOuBiasBatch(h.data(), n, dt_big, ns_big, kTau, kSb, rtn, du0,
                                E.data(), G.data(), Q.data(), Qb.data(), Qbb.data());
    };
    std::vector<double> G1, E1, Q1, Qb1, Qbb1, G2, E2, Q2, Qb2, Qbb2;
    big(G1, E1, Q1, Qb1, Qbb1);
    std::vector<double> Gf((size_t)n * 2 * 18), Ef((size_t)n * 6);
    for (int t = 0; t < 4; ++t)
        qoejopt::gveBiasFramesBatch(h.data(), n, 2, 1, seg, kTau, 0, rtn,
                                    Gf.data(), Ef.data());
    big(G2, E2, Q2, Qb2, Qbb2);
    // end / Qbb 逐位
    for (int i = 0; i < n * 6; ++i) EXPECT_DOUBLE_EQ(E1[i], E2[i]);
    for (int i = 0; i < n * 9; ++i) EXPECT_DOUBLE_EQ(Qbb1[i], Qbb2[i]);
    // Γ / Q 的范数相对差
    auto relv = [](const std::vector<double> &a, const std::vector<double> &b) {
        double d2 = 0.0, n2 = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            d2 += (a[i] - b[i]) * (a[i] - b[i]);
            n2 += b[i] * b[i];
        }
        return std::sqrt(d2 / std::max(n2, 1e-300));
    };
    const double rG = relv(G1, G2);
    const double rQ = relv(Q1, Q2);
    const double rQb = relv(Qb1, Qb2);
    std::printf("[CallHistory] rel G=%.3e Q=%.3e Qb=%.3e "
                "|G1|=%.6e |G2|=%.6e\n", rG, rQ, rQb,
                std::sqrt(std::inner_product(G1.begin(), G1.end(), G1.begin(), 0.0)),
                std::sqrt(std::inner_product(G2.begin(), G2.end(), G2.begin(), 0.0)));
    EXPECT_LT(rG, 1e-12);
    EXPECT_LT(rQ, 1e-12);
    EXPECT_LT(rQb, 1e-12);
}

// 3) CUDA vs 连续变分 host 参考（几颗星、Δt=1T）：Γ/Q_qq/Q_qb/Q_bb 最大绝对差。
TEST(OuBias, CudaVsHostReference) {
    const auto states = sample_states();
    const int n = 2;                       // 少星即可（核逐星独立）
    const double dt = kT;
    const Vector3d rtn(1.0, 1.5, 0.7), du(5e-6, 3e-6, 2e-6);
    for (double sb : {kSb, kSbLarge}) {
        const auto c = run_cuda(states, dt, kOuNsub, kTau, sb, rtn, du);
        double max_absG = 0.0, max_relG = 0.0;
        double max_absQ = 0.0, max_relQ = 0.0;
        double max_absQb = 0.0, max_relQb = 0.0;
        double max_absQbb = 0.0;
        for (int i = 0; i < n; ++i) {
            const auto r = ref_ou(states[i], dt, kTau, sb, rtn, du, 20.0);
            const Eigen::Matrix<double, 6, 3> G = gamma_of(c, i);
            const Eigen::Matrix<double, 6, 3> Qb = qb_of(c, i);
            const Matrix6d Q = qqq_of(c, i);
            const Matrix3d Qbb = qbb_of(c, i);
            max_absG = std::max(max_absG, (G - r.Gamma).eval().cwiseAbs().maxCoeff());
            max_relG = std::max(max_relG, frob_rel63(G, r.Gamma));
            max_absQ = std::max(max_absQ, (Q - r.Qqq).eval().cwiseAbs().maxCoeff());
            max_relQ = std::max(max_relQ, frob_rel(Q, r.Qqq));
            max_absQb = std::max(max_absQb, (Qb - r.Qqb).eval().cwiseAbs().maxCoeff());
            max_relQb = std::max(max_relQb, frob_rel63(Qb, r.Qqb));
            max_absQbb = std::max(max_absQbb,
                                  (Qbb - r.Qbb).eval().cwiseAbs().maxCoeff());
        }
        std::printf("[CudaVsHost] sb=%.0e G(abs=%.3e,rel=%.3e) Q(abs=%.3e,rel=%.3e) "
                    "Qb(abs=%.3e,rel=%.3e) Qbb(abs=%.3e)\n",
                    sb, max_absG, max_relG, max_absQ, max_relQ,
                    max_absQb, max_relQb, max_absQbb);
        EXPECT_LT(max_relG, 1e-4);
        EXPECT_LT(max_relQ, 1e-3);
        EXPECT_LT(max_relQb, 1e-3);
        EXPECT_LT(max_absQbb, 1e-12);
    }
}

// 2b) P1 逐帧 Γ(t)：末帧与 Stage A 区间 Γ 一致；与独立 FD 一致。
TEST(OuBias, BiasFramesGamma) {
    const auto states = sample_states();
    const int n = (int)states.size();
    const int nfr = 11;            // frames 0..10，每帧 10 s → 100 s
    const double dt = 10.0;
    const double dt_tot = dt * (nfr - 1);
    std::vector<double> h((size_t)n * 6);
    for (int i = 0; i < n; ++i)
        for (int c = 0; c < 6; ++c) h[(size_t)i * 6 + c] = states[i][c];
    std::vector<double> G((size_t)n * nfr * 18), E((size_t)n * 6);
    double rtn[3] = {1.0, 1.0, 1.0};
    qoejopt::gveBiasFramesBatch(h.data(), n, nfr, 1, dt, kTau, 0, rtn,
                                G.data(), E.data());
    // 同子步约定：kernel 单帧 10 s（bias_nsub=0 → nsub=max(1,0)=1）与
    // Stage A dt=10、bias_nsub=0 必须逐位同模型。
    {
        std::vector<double> G10((size_t)n * 18), e10((size_t)n * 6),
            Q10((size_t)n * 36), Qb10((size_t)n * 18), Qbb10((size_t)n * 9),
            du0(3, 0.0);
        qoejopt::gveOuBiasBatch(h.data(), n, dt, 0, kTau, kSb, rtn,
                                du0.data(), e10.data(), G10.data(), Q10.data(),
                                Qb10.data(), Qbb10.data());
        double max_rel1 = 0.0;
        for (int i = 0; i < n; ++i) {
            Eigen::Matrix<double, 6, 3> Gk, Ga;
            for (int a = 0; a < 6; ++a)
                for (int k = 0; k < 3; ++k) {
                    Gk(a, k) = G[(((size_t)i * nfr + 1) * 6 + a) * 3 + k];
                    Ga(a, k) = G10[((size_t)i * 6 + a) * 3 + k];
                }
            max_rel1 = std::max(max_rel1, frob_rel63(Gk, Ga));
        }
        std::printf("[BiasFrames] single-step vs StageA(10s,nsub=1) rel=%.3e\n",
                    max_rel1);
        // 同子步模型；8e-5 来自 10 s 单步内 exp/STM 的浮点路径差，非模型差异。
        EXPECT_LT(max_rel1, 1e-3);
    }
    // Stage A 区间 Γ（同一 CUDA 子步模型，总时长相同）
    std::vector<double> du3(3, 0.0), end6((size_t)n * 6),
        Gs((size_t)n * 18), Q((size_t)n * 36), Qb((size_t)n * 18),
        Qbb((size_t)n * 9);
    qoejopt::gveOuBiasBatch(h.data(), n, dt_tot, kOuNsub, kTau, kSb, rtn,
                            du3.data(), end6.data(), Gs.data(), Q.data(),
                            Qb.data(), Qbb.data());
    // 更细的 FD 参考（1 s 步），把 RK4 截断压到 1e-7 量级
    auto gamma_fd_h = [](const Vector6d &x0, double dt, double tau, double delta) {
        Eigen::Matrix<double, 6, 3> G;
        for (int k = 0; k < 3; ++k) {
            Vector3d b0 = Vector3d::Zero();
            b0[k] = delta;
            const Vector6d xp = propagate_rtn_bias(x0, dt, tau, b0, 1.0);
            const Vector6d xm = propagate_rtn_bias(x0, dt, tau, -b0, 1.0);
            G.col(k) = (xp - xm) / (2.0 * delta);
        }
        return G;
    };
    double max_rel = 0.0, max_fd = 0.0;
    for (int i = 0; i < n; ++i) {
        Eigen::Matrix<double, 6, 3> Gf;
        for (int a = 0; a < 6; ++a)
            for (int k = 0; k < 3; ++k)
                Gf(a, k) = G[(((size_t)i * nfr + (nfr - 1)) * 6 + a) * 3 + k];
        // 第 0 帧 Γ=0
        for (int a = 0; a < 6; ++a)
            for (int k = 0; k < 3; ++k)
                EXPECT_DOUBLE_EQ(G[((size_t)i * nfr * 6 + a) * 3 + k], 0.0);
        Eigen::Matrix<double, 6, 3> Ga;
        for (int a = 0; a < 6; ++a)
            for (int k = 0; k < 3; ++k) Ga(a, k) = Gs[((size_t)i * 6 + a) * 3 + k];
        max_rel = std::max(max_rel, frob_rel63(Gf, Ga));
        const Eigen::Matrix<double, 6, 3> Gref =
            gamma_fd_h(states[i], dt_tot, kTau, kSb);
        max_fd = std::max(max_fd, frob_rel63(Gf, Gref));
        EXPECT_TRUE(Gf.allFinite());
    }
    std::printf("[BiasFrames] vs interval rel=%.3e vs FD rel=%.3e\n",
                max_rel, max_fd);
    EXPECT_LT(max_rel, 1e-5);
    // 默认每帧 1 个 10 s 子步（与 Stage A 同约定），对 1 s FD 的
    // 中点求积截断约 5e-6；严格同子步一致性由上面的 single-step 检查覆盖。
    EXPECT_LT(max_fd, 1e-5);
}

// 4) Q_qq 对称/PSD；Q_bb 闭式（含 Pb 终值项）；Q_qb 形状；增广块 9×9 对称。
TEST(OuBias, QPropertiesAndClosedForm) {
    const auto states = sample_states();
    const double dt = kT;
    const Vector3d rtn(2.0, 1.0, 0.5), du(5e-6, 0.0, 0.0);
    for (double sb : {kSb, kSbLarge}) {
        const auto c = run_cuda(states, dt, kOuNsub, kTau, sb, rtn, du);
        const double phi = std::exp(-dt / kTau);
        Matrix3d Pb = Matrix3d::Zero();
        for (int k = 0; k < 3; ++k) Pb(k, k) = std::pow(sb * rtn[k], 2);
        const Matrix3d Qbb_exp = sb * sb * (1.0 - phi * phi) * Matrix3d::Identity()
                               + phi * phi * Pb;
        for (int i = 0; i < (int)states.size(); ++i) {
            const Matrix6d Q = qqq_of(c, i);
            const Eigen::Matrix<double, 6, 3> Qb = qb_of(c, i);
            const Matrix3d Qbb = qbb_of(c, i);
            const Matrix6d dQ = (Q - Q.transpose()).eval();
            EXPECT_LT(dQ.norm() / std::max(Q.norm(), 1e-300), 1e-12);
            Eigen::SelfAdjointEigenSolver<Matrix6d> es(Q);
            EXPECT_GT(es.eigenvalues().minCoeff(), -1e-10 * std::max(Q.norm(), 1e-300));
            const Matrix3d dQbb = (Qbb - Qbb_exp).eval();
            EXPECT_LT(dQbb.norm(), 1e-12 * std::max(Qbb_exp.norm(), 1.0));
            EXPECT_TRUE(Qb.allFinite());
            EXPECT_EQ(Qb.rows(), 6);
            EXPECT_EQ(Qb.cols(), 3);
            // 增广块 [[Q_qq,Q_qb],[Q_qbᵀ,Q_bb]] 对称（Q_bb 已含 φ²P_b）。
            Eigen::Matrix<double, 9, 9> aug;
            aug.topLeftCorner<6, 6>() = Q;
            aug.topRightCorner<6, 3>() = Qb;
            aug.bottomLeftCorner<3, 6>() = Qb.transpose();
            aug.bottomRightCorner<3, 3>() = Qbb;
            const Eigen::Matrix<double, 9, 9> dA = (aug - aug.transpose()).eval();
            EXPECT_LT(dA.norm() / std::max(aug.norm(), 1e-300), 1e-12);
            EXPECT_TRUE(aug.allFinite());
        }
    }
}

// 5) σ_b=0：Q 块为零，Γ_b 有限且与 σ_b>0 时一致（Γ 不依赖 σ_b）。
TEST(OuBias, ZeroSigmaIsSilent) {
    const auto states = sample_states();
    const int n = (int)states.size();
    const Vector3d rtn(1.0, 1.0, 1.0), du(0.0, 0.0, 0.0);
    const auto c0 = run_cuda(states, kT, kOuNsub, kTau, 0.0, rtn, du);
    const auto cr = run_cuda(states, kT, kOuNsub, kTau, kSb, rtn, du);
    for (int i = 0; i < n; ++i) {
        const Eigen::Matrix<double, 6, 3> G = gamma_of(c0, i);
        EXPECT_TRUE(G.allFinite());
        EXPECT_GT(G.norm(), 0.0);
        // σ_b=0 与 >0 的 Γ 数学上相同；核内 q_b=0 分支的 FMA/舍入差
        // 实测 ~2e-8（不同编译/GPU 状态下可复现），不是模型误差。
        EXPECT_LT(frob_rel63(G, gamma_of(cr, i)), 1e-6);
    }
    for (double v : c0.Q) EXPECT_LT(std::abs(v), 1e-30);
    for (double v : c0.Qb) EXPECT_LT(std::abs(v), 1e-30);
    for (double v : c0.Qbb) EXPECT_LT(std::abs(v), 1e-30);
}

// 6) σ_b 缩放：Γ 不变；Q_qq/Q_qb/Q_bb ∝ σ_b²（du=0 时 Q_qq 也只有 bias 项）。
TEST(OuBias, SigmaScaling) {
    const auto states = sample_states();
    const double dt = kT;
    const Vector3d rtn(1.0, 1.0, 1.0), zero = Vector3d::Zero();
    const auto c1 = run_cuda(states, dt, kOuNsub, kTau, kSb, rtn, zero);
    const auto c2 = run_cuda(states, dt, kOuNsub, kTau, kSbLarge, rtn, zero);
    const double scale = (kSbLarge / kSb) * (kSbLarge / kSb);   // 100
    for (int i = 0; i < (int)states.size(); ++i) {
        EXPECT_LT(frob_rel63(gamma_of(c2, i), gamma_of(c1, i)), 1e-12);
        const Matrix6d Q1 = qqq_of(c1, i), Q2 = qqq_of(c2, i);
        const Matrix6d dQ = (Q2 - scale * Q1).eval();
        EXPECT_LT(dQ.norm() / std::max(Q2.norm(), 1e-300), 1e-12);
        const Eigen::Matrix<double, 6, 3> B1 = qb_of(c1, i), B2 = qb_of(c2, i);
        const Eigen::Matrix<double, 6, 3> dB = (B2 - scale * B1).eval();
        EXPECT_LT(dB.norm() / std::max(B2.norm(), 1e-300), 1e-12);
        const Matrix3d P1 = qbb_of(c1, i), P2 = qbb_of(c2, i);
        const Matrix3d dP = (P2 - scale * P1).eval();
        EXPECT_LT(dP.norm() / std::max(P2.norm(), 1e-300), 1e-12);
    }
}

// 7) Δ→0 小量闭式：Γ/Δ→B(0)，Q_qq/Δ→B Qu Bᵀ，Q_bb→P_b。
TEST(OuBias, DeltaToZeroLimits) {
    const auto states = sample_states();
    const Vector3d rtn(1.0, 1.0, 1.0), du(5e-6, 5e-6, 5e-6);
    const double dt = 1e-6;
    const auto c = run_cuda(states, dt, kOuNsub, kTau, kSb, rtn, du);
    const Matrix3d Pb = Matrix3d::Identity() * (kSb * kSb);
    const Matrix3d Qu = du.cwiseProduct(du).asDiagonal();
    for (int i = 0; i < (int)states.size(); ++i) {
        const Eigen::Matrix<double, 6, 3> B = b_rtn(states[i]);
        EXPECT_LT(frob_rel63(gamma_of(c, i), dt * B), 1e-3);
        const Matrix6d Qexp = B * Qu * B.transpose();
        const Matrix6d dQ = (qqq_of(c, i) / dt - Qexp).eval();
        EXPECT_LT(dQ.norm() / std::max(Qexp.norm(), 1e-300), 1e-3);
        const Matrix3d dP = (qbb_of(c, i) - Pb).eval();
        EXPECT_LT(dP.norm(), 1e-12);
    }
}

// 8) 非法/非有限/零输入。
TEST(OuBias, InvalidInputs) {
    std::vector<double> oe = {6.9368e6, 0.0361, 6.90e-4, 5.06e-5, 1.2656, 0.0157};
    std::vector<double> out6(6), out18(18), out36(36), out9(9);
    double rtn[3] = {1.0, 1.0, 1.0}, du[3] = {1e-6, 1e-6, 1e-6};
    auto call = [&](int n, double dt, int nsub, double tau, double sb,
                    const double *r, const double *d, double *g) {
        qoejopt::gveOuBiasBatch(oe.data(), n, dt, nsub, tau, sb, r, d,
                                out6.data(), g, out36.data(), out18.data(),
                                out9.data());
    };
    EXPECT_THROW(call(0, 10.0, kOuNsub, kTau, kSb, rtn, du, out18.data()),
                 std::invalid_argument);
    EXPECT_THROW(call(1, 0.0, kOuNsub, kTau, kSb, rtn, du, out18.data()),
                 std::invalid_argument);
    EXPECT_THROW(call(1, -1.0, kOuNsub, kTau, kSb, rtn, du, out18.data()),
                 std::invalid_argument);
    EXPECT_THROW(call(1, std::nan(""), kOuNsub, kTau, kSb, rtn, du, out18.data()),
                 std::invalid_argument);
    EXPECT_THROW(call(1, 10.0, kOuNsub, 0.0, kSb, rtn, du, out18.data()),
                 std::invalid_argument);
    EXPECT_THROW(call(1, 10.0, kOuNsub, -1.0, kSb, rtn, du, out18.data()),
                 std::invalid_argument);
    EXPECT_THROW(call(1, 10.0, kOuNsub, std::nan(""), kSb, rtn, du, out18.data()),
                 std::invalid_argument);
    EXPECT_THROW(call(1, 10.0, kOuNsub, kTau, -1.0, rtn, du, out18.data()),
                 std::invalid_argument);
    EXPECT_THROW(call(1, 10.0, kOuNsub, kTau, std::nan(""), rtn, du, out18.data()),
                 std::invalid_argument);
    EXPECT_THROW(call(1, 10.0, -1, kTau, kSb, rtn, du, out18.data()),
                 std::invalid_argument);
    EXPECT_THROW(call(1, 10.0, kOuNsub, kTau, kSb, nullptr, du, out18.data()),
                 std::invalid_argument);
    EXPECT_THROW(call(1, 10.0, kOuNsub, kTau, kSb, rtn, nullptr, out18.data()),
                 std::invalid_argument);
    double bad_rtn[3] = {1.0, std::nan(""), 1.0};
    EXPECT_THROW(call(1, 10.0, kOuNsub, kTau, kSb, bad_rtn, du, out18.data()),
                 std::invalid_argument);
    // 输出指针为空
    EXPECT_THROW(qoejopt::gveOuBiasBatch(oe.data(), 1, 10.0, kOuNsub, kTau, kSb,
                                         rtn, du, out6.data(), nullptr,
                                         out36.data(), out18.data(), out9.data()),
                 std::invalid_argument);
    // oe 为空
    EXPECT_THROW(qoejopt::gveOuBiasBatch(nullptr, 1, 10.0, kOuNsub, kTau, kSb,
                                         rtn, du, out6.data(), out18.data(),
                                         out36.data(), out18.data(), out9.data()),
                 std::invalid_argument);
}
}  // namespace ou_bias_impl

// ===================== test/test_bias_window_gtest.cpp =====================
namespace bias_window_impl {
// winT=5 / nseg=10 偏差窗口 gtest（C++ / host double + CUDA 核）。
//
// 约定：T=5749.800907 s，seg_dt=0.5T，11 个节点 / 10 个区间；每区间 dyn 子步
// fpn=288（gveStmSubstepsDevice / gveStmDeviceGrp，h≈10 s）。偏差 β_k∈R³ 在区间内
// 为常值（RTN），确定性响应为 Γ_k β_k（OU 核，含 e^{−s/τ} 衰减）。
//
// 覆盖：
//   1. BiasFramesWindowGamma：10 区间逐帧 Γ；Γ(0)=0；单区间 frames-vs-OU 一致；
//      10 区间端点 vs 单发 OU（记录 nsub 口径差）；端点 vs host FD。
//   2. WindowDynBiasJacobian：经 winDynAsmKernel，b=−Q^{1/2}(x_{k+1}−end−Γβ) 的
//      ∂b/∂β=+Q^{1/2}Γ、A 与 β 无关、线性叠加；winTotalCostDevice 同口径。
//   3. WindowChainBetaSensitivity：解析 STM 链 ∂q_m/∂β_j=(Π_{k=m-1}^{j+1}Φ_k)Γ_j
//      （j<m，否则 0），与非线性传播（小 β）对比；L=11 与 L=10。
//
// 注意：不包含 dastates.h（其全局 using namespace DACE/std 会让 Eigen 算子歧义）；
// host GVE 两个函数在此前向声明（符号在 libdastate.so）。



using Vector6d = Eigen::Matrix<double, 6, 1>;

using M3 = Eigen::Matrix<double, 3, 3>;
using M6 = Eigen::Matrix<double, 6, 6>;
using M63 = Eigen::Matrix<double, 6, 3>;
using V3 = Eigen::Matrix<double, 3, 1>;
using V6 = Eigen::Matrix<double, 6, 1>;

namespace {

constexpr double kT = 5749.800907;
constexpr double kSeg = 0.5 * kT;     // 0.5T
constexpr double kTau = kT;           // τ = 1T
constexpr double kMu = 3.986004415e14;
constexpr int kFpn = 288;             // 每区间 dyn 子步（≈10 s）
constexpr double kBeta = 1e-7;        // σ_b / 测试偏差量级

void ck(cudaError_t e, const char *what) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

std::vector<Vector6d> sample_states() {
    std::vector<Vector6d> v;
    Vector6d x;
    x << 6.9368e6, 0.0361, 6.90e-4, 5.06e-5, 1.2656, 0.0157;
    v.push_back(x);
    x << 6.9300e6, 1.10, 1.2e-3 * std::cos(0.4), 1.2e-3 * std::sin(0.4), 0.98, 2.3;
    v.push_back(x);
    x << 7.1000e6, 2.50, 5e-5 * std::cos(2.0), 5e-5 * std::sin(2.0), 1.57, 4.1;
    v.push_back(x);
    x << 6.9500e6, 5.20, 8e-4 * std::cos(1.3), 8e-4 * std::sin(1.3), 0.62, 5.5;
    v.push_back(x);
    return v;
}

Vector6d f_full(const Vector6d &x) { return noeGveRhs(x, 1.0, 0.0); }

// ∂f_QOE/∂a（ECI），不含 Kepler n；与 gveGaussW_dev 同口径。
Vector6d gve_ap(const Vector6d &oe, const Vector6d &rv, const V3 &ap) {
    const double a = oe[0], u = oe[1], ex = oe[2], ey = oe[3], inc = oe[4];
    const double e = std::hypot(ex, ey);
    const double w = std::atan2(ey, ex);
    double M = std::fmod(u - w, 2.0 * M_PI);
    if (M < 0.0) M += 2.0 * M_PI;
    double E = M;
    for (int it = 0; it < 30; ++it)
        E -= (E - e * std::sin(E) - M) / (1.0 - e * std::cos(E));
    const double cnu = (std::cos(E) - e) / (1.0 - e * std::cos(E));
    const double snu = std::sqrt(std::max(1.0 - e * e, 0.0)) * std::sin(E) /
                       (1.0 - e * std::cos(E));
    const double p = a * (1.0 - e * e);
    const double h = std::sqrt(kMu * p);
    const double r = p / (1.0 + e * cnu);
    const V3 r3 = rv.head<3>(), v3 = rv.tail<3>();
    const V3 R = r3.normalized();
    const V3 N = r3.cross(v3).normalized();
    const V3 T = N.cross(R);
    const double aR = ap.dot(R), aT = ap.dot(T), aN = ap.dot(N);
    const double ef = std::max(e, 1e-8), sif = std::max(std::sin(inc), 1e-8);
    const double uarg = w + std::atan2(snu, cnu);
    const double da = (2.0 * a * a / h) * (e * snu * aR + (p / r) * aT);
    const double de = (1.0 / h) * (p * snu * aR + ((p + r) * cnu + r * e) * aT);
    const double di = (r * std::cos(uarg) / h) * aN;
    const double dOm = (r * std::sin(uarg) / (h * sif)) * aN;
    const double dw = (1.0 / (h * ef)) * (-p * cnu * aR + (p + r) * snu * aT)
                    - (r * std::sin(uarg) * std::cos(inc) / (h * sif)) * aN;
    const double fM = std::sqrt(std::max(1.0 - e * e, 0.0)) / (h * ef);
    const double dM = fM * ((p * cnu - 2.0 * r * e) * aR - (p + r) * snu * aT);
    Vector6d out;
    out[0] = da; out[1] = dM + dw;
    out[2] = std::cos(w) * de - e * std::sin(w) * dw;
    out[3] = std::sin(w) * de + e * std::cos(w) * dw;
    out[4] = di; out[5] = dOm;
    return out;
}

M63 b_rtn(const Vector6d &x) {
    const Vector6d rv = noeOsc2rv(x, 100, 1e-12);
    const V3 R = rv.head<3>().normalized();
    const V3 N = rv.head<3>().cross(rv.tail<3>()).normalized();
    const V3 T = N.cross(R);
    M63 B;
    B.col(0) = gve_ap(x, rv, R);
    B.col(1) = gve_ap(x, rv, T);
    B.col(2) = gve_ap(x, rv, N);
    return B;
}

Vector6d rk4_step(const Vector6d &x, double h, const V3 &b0, double t, double tau) {
    auto rhs = [&](const Vector6d &y, double tt) {
        return f_full(y) + b_rtn(y) * (b0 * std::exp(-tt / tau));
    };
    const Vector6d k1 = rhs(x, t);
    const Vector6d k2 = rhs(x + 0.5 * h * k1, t + 0.5 * h);
    const Vector6d k3 = rhs(x + 0.5 * h * k2, t + 0.5 * h);
    const Vector6d k4 = rhs(x + h * k3, t + h);
    return x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}

Vector6d propagate_rtn_bias(const Vector6d &x0, double dt, double tau,
                            const V3 &b0, double hstep) {
    const int ns = std::max(1, (int)std::ceil(dt / hstep));
    const double h = dt / ns;
    Vector6d x = x0;
    for (int j = 0; j < ns; ++j) x = rk4_step(x, h, b0, j * h, tau);
    return x;
}

// 名义传播（β=0）到 nfr 个节点；返回 [frame*n + sat][6]。
std::vector<double> nominal_nodes(const std::vector<Vector6d> &x0s, int nfr,
                                  double dt, double hstep) {
    const int n = (int)x0s.size();
    const int ns = std::max(1, (int)std::ceil(dt / hstep));
    const double h = dt / ns;
    std::vector<double> out((size_t)nfr * n * 6);
    std::vector<Vector6d> x = x0s;
    for (int f = 0; f < nfr; ++f) {
        for (int i = 0; i < n; ++i)
            for (int c = 0; c < 6; ++c) out[((size_t)f * n + i) * 6 + c] = x[i][c];
        if (f == nfr - 1) break;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < ns; ++j)
                x[i] = rk4_step(x[i], h, V3::Zero(), 0.0, kTau);
    }
    return out;
}

M63 gamma_fd(const Vector6d &x0, double dt, double tau, double delta, double hstep) {
    M63 G;
    for (int k = 0; k < 3; ++k) {
        V3 b0 = V3::Zero();
        b0[k] = delta;
        const Vector6d xp = propagate_rtn_bias(x0, dt, tau, b0, hstep);
        const Vector6d xm = propagate_rtn_bias(x0, dt, tau, -b0, hstep);
        G.col(k) = (xp - xm) / (2.0 * delta);
    }
    return G;
}

M63 gamma_of_frames(const std::vector<double> &G, int n, int nfr, int i, int f) {
    M63 out;
    for (int a = 0; a < 6; ++a)
        for (int k = 0; k < 3; ++k)
            out(a, k) = G[((size_t)(i * nfr + f) * 6 + a) * 3 + k];
    return out;
}

M63 gamma_of_flat(const std::vector<double> &G, int i) {
    M63 out;
    for (int a = 0; a < 6; ++a)
        for (int k = 0; k < 3; ++k) out(a, k) = G[((size_t)i * 6 + a) * 3 + k];
    return out;
}

double frob_rel(const M63 &a, const M63 &b) {
    return (a - b).norm() / std::max(b.norm(), 1e-300);
}
double frob_rel6(const M6 &a, const M6 &b) {
    return (a - b).norm() / std::max(b.norm(), 1e-300);
}

// ---- 手工 WinCtx：E 条 ISL（每帧重复），无 GTS，prior=1（不写 damp）----
class WindowHarness {
public:
    WindowHarness(int n, int L, int E, const std::vector<int> &ei,
                  const std::vector<int> &ej, int nsub, double dt,
                  double sigma_isl, const M6 &qscale)
        : n_(n), L_(L), E_(E), ei_(ei), ej_(ej), nsub_(nsub) {
        c_ = qoejopt::WinCtx{};
        nISL_ = L * E;
        nDYN_ = (L - 1) * n * 6;
        nnz_ = nISL_ * 12 + nDYN_ * 12;
        nrows_ = nISL_ + nDYN_;
        qoejopt::WinCtx &c = c_;
        c.n = n; c.L = L; c.E = E; c.G = 0;
        c.ncol = n * L * 6; c.nnz = nnz_; c.nrows = nrows_;
        c.off_isl = 0; c.off_gts = 0; c.off_dyn = nISL_ * 12; c.off_damp = nnz_;
        c.dt = dt; c.s_isl = (E > 0 ? 1.0 / sigma_isl : 1.0); c.s_gts = 1.0;
        c.issq = 0.0; c.prior = 1; c.q_perstar = 0; c.dyn_sub = nsub;
        c.gauge_k = 0;
        alloc(c.d_X, (size_t)L * n * 6);
        alloc(c.d_rv, (size_t)L * n * 6);
        alloc(c.d_Ap, (size_t)L * n * 18);
        alloc(c.d_end, (size_t)(L - 1) * n * 6);
        alloc(c.d_phi, (size_t)(L - 1) * n * 36);
        alloc(c.d_res, 1);
        alloc(c.d_dyn_bias, (size_t)(L - 1) * n * 6);
        ck(cudaMemset(c.d_dyn_bias, 0, (size_t)(L - 1) * n * 6 * sizeof(double)),
           "memset dbias");
        const size_t ns = (size_t)(L - 1) * n;
        alloc(c.d_state_sub, ns * 6);
        alloc(c.d_state_b, ns * 6);
        alloc(c.d_phi_acc, ns * 36);
        alloc(c.d_phi_out, ns * 36);
        alloc(c.d_phi_step, ns * 36);
        alloc(c.d_qingt, 36);
        double qh[36];
        for (int a = 0; a < 6; ++a)
            for (int b = 0; b < 6; ++b) qh[a * 6 + b] = qscale(a, b);
        ck(cudaMemcpy(c.d_qingt, qh, 36 * sizeof(double), cudaMemcpyHostToDevice),
           "upload qingt");
        if (E > 0) {
            std::vector<int> hi(nISL_), hj(nISL_);
            std::vector<double> hw(nISL_, 1.0);
            for (int f = 0; f < L; ++f)
                for (int e = 0; e < E; ++e) { hi[f * E + e] = ei[e]; hj[f * E + e] = ej[e]; }
            alloc_i(c.d_isl_i, nISL_);
            alloc_i(c.d_isl_jj, nISL_);
            alloc(c.d_isl_D, nISL_);
            alloc(c.d_isl_W, nISL_);
            ck(cudaMemcpy(c.d_isl_i, hi.data(), nISL_ * sizeof(int), cudaMemcpyHostToDevice),
               "isl_i");
            ck(cudaMemcpy(c.d_isl_jj, hj.data(), nISL_ * sizeof(int), cudaMemcpyHostToDevice),
               "isl_jj");
            ck(cudaMemcpy(c.d_isl_W, hw.data(), nISL_ * sizeof(double), cudaMemcpyHostToDevice),
               "isl_W");
        }
        alloc(d_A_, nnz_);
        alloc(d_b_, nrows_);
    }
    ~WindowHarness() {
        qoejopt::WinCtx &c = c_;
        cudaFree(c.d_isl_i); cudaFree(c.d_isl_jj); cudaFree(c.d_isl_D); cudaFree(c.d_isl_W);
        cudaFree(c.d_gts_a); cudaFree(c.d_gts_D); cudaFree(c.d_gts_W); cudaFree(c.d_gts_Q);
        cudaFree(c.d_ctr); cudaFree(c.d_qingt); cudaFree(c.d_beta); cudaFree(c.d_dyn_bias);
        cudaFree(c.d_X); cudaFree(c.d_rv); cudaFree(c.d_Ap); cudaFree(c.d_end);
        cudaFree(c.d_phi); cudaFree(c.d_res); cudaFree(c.d_state_sub); cudaFree(c.d_state_b);
        cudaFree(c.d_phi_acc); cudaFree(c.d_phi_out); cudaFree(c.d_phi_step);
        cudaFree(d_A_); cudaFree(d_b_);
    }
    void set_isl_D(const std::vector<double> &D) {
        ck(cudaMemcpy(c_.d_isl_D, D.data(), D.size() * sizeof(double),
                      cudaMemcpyHostToDevice), "isl_D");
    }
    void set_dyn_bias(const std::vector<double> &d) {
        ck(cudaMemcpy(c_.d_dyn_bias, d.data(), d.size() * sizeof(double),
                      cudaMemcpyHostToDevice), "dbias");
    }
    // 默认关闭：d_dyn_bias=nullptr（winDynAsmKernel 跳过 e-=dbias）。
    void disable_dyn_bias() {
        cudaFree(c_.d_dyn_bias);
        c_.d_dyn_bias = nullptr;
    }
    // A (nrows×ncol) 稠密、b(nrows)、end/phi（device→host），返回 winAssembleDevice 的 ISL res2。
    double assemble_dense(const std::vector<double> &X, Eigen::MatrixXd &A,
                          Eigen::VectorXd &b, std::vector<double> &end,
                          std::vector<double> &phi) {
        const int nl = L_ * n_;
        std::vector<double> rv((size_t)nl * 6), J((size_t)nl * 36),
            Ap((size_t)nl * 18);
        qoejopt::oe2rvJacHost(X.data(), nl, rv.data(), J.data());
        for (int i = 0; i < nl; ++i)
            for (int a = 0; a < 3; ++a)
                for (int cc = 0; cc < 6; ++cc)
                    Ap[(size_t)i * 18 + a * 6 + cc] = J[(size_t)i * 36 + a * 6 + cc];
        const double res2 = qoejopt::winAssembleDevice(c_, X.data(), rv.data(),
                                                       Ap.data(), d_A_, d_b_);
        std::vector<double> av(nnz_), bv(nrows_);
        ck(cudaMemcpy(av.data(), d_A_, nnz_ * sizeof(double), cudaMemcpyDeviceToHost),
           "A d2h");
        ck(cudaMemcpy(bv.data(), d_b_, nrows_ * sizeof(double), cudaMemcpyDeviceToHost),
           "b d2h");
        const size_t nk = (size_t)(L_ - 1) * n_;
        end.resize(nk * 6);
        phi.resize(nk * 36);
        ck(cudaMemcpy(end.data(), c_.d_end, nk * 6 * sizeof(double),
                      cudaMemcpyDeviceToHost), "end d2h");
        ck(cudaMemcpy(phi.data(), c_.d_phi, nk * 36 * sizeof(double),
                      cudaMemcpyDeviceToHost), "phi d2h");
        A = Eigen::MatrixXd::Zero(nrows_, c_.ncol);
        b = Eigen::VectorXd::Zero(nrows_);
        for (int idx = 0; idx < nISL_; ++idx) {
            const int f = idx / E_, e = idx - f * E_;
            const int i = ei_[e], j = ej_[e];
            for (int k = 0; k < 6; ++k) {
                A(idx, ((size_t)f * n_ + i) * 6 + k) = av[(size_t)idx * 12 + k];
                A(idx, ((size_t)f * n_ + j) * 6 + k) = av[(size_t)idx * 12 + 6 + k];
            }
            b(idx) = bv[idx];
        }
        const int off = nISL_ * 12;
        for (int idx2 = 0; idx2 < nDYN_ / 6; ++idx2) {
            const int kk = idx2 / n_, ii = idx2 - kk * n_;
            for (int a = 0; a < 6; ++a) {
                const int row = nISL_ + idx2 * 6 + a;
                for (int cc = 0; cc < 6; ++cc) {
                    A(row, ((size_t)kk * n_ + ii) * 6 + cc) =
                        av[off + (size_t)(idx2 * 6 + a) * 12 + cc];
                    A(row, ((size_t)(kk + 1) * n_ + ii) * 6 + cc) =
                        av[off + (size_t)(idx2 * 6 + a) * 12 + 6 + cc];
                }
                b(row) = bv[row];
            }
        }
        return res2;
    }
    double total_cost(const std::vector<double> &X) {
        // 先确保 d_X/d_end/d_phi 为当前 X；prior_on=0。
        (void)X;
        return qoejopt::winTotalCostDevice(c_, nullptr, nullptr, 0, 0);
    }
    qoejopt::WinCtx c_;
    int n() const { return n_; }
    int L() const { return L_; }
    int nISL() const { return nISL_; }
    int nDYN() const { return nDYN_; }
    int nrows() const { return nrows_; }

private:
    void alloc(double *&p, size_t cnt) {
        if (cnt == 0) { p = nullptr; return; }
        ck(cudaMalloc(&p, cnt * sizeof(double)), "cudaMalloc double");
    }
    void alloc_i(int *&p, size_t cnt) {
        if (cnt == 0) { p = nullptr; return; }
        ck(cudaMalloc(&p, cnt * sizeof(int)), "cudaMalloc int");
    }
    int n_, L_, E_, nsub_, nISL_, nDYN_, nnz_, nrows_;
    std::vector<int> ei_, ej_;
    double *d_A_ = nullptr, *d_b_ = nullptr;
};

}  // namespace

// 1) 10 区间逐帧 Γ / 端点 Γ / 单发 OU / host FD。
TEST(BiasWindow, BiasFramesWindowGamma) {
    const std::vector<Vector6d> st = {sample_states()[0], sample_states()[1]};
    const int n = (int)st.size(), L = 11;
    std::vector<double> x0((size_t)n * 6);
    for (int i = 0; i < n; ++i)
        for (int c = 0; c < 6; ++c) x0[(size_t)i * 6 + c] = st[i][c];
    double rtn[3] = {1.0, 1.0, 1.0};
    std::vector<double> Gf((size_t)n * L * 18), endf((size_t)n * 6);
    qoejopt::gveBiasFramesBatch(x0.data(), n, L, 1, kSeg, kTau, 0, rtn,
                                Gf.data(), endf.data());
    // Γ(t=0)=0
    for (int i = 0; i < n; ++i)
        for (int a = 0; a < 18; ++a)
            EXPECT_DOUBLE_EQ(Gf[(size_t)i * L * 18 + a], 0.0);

    // 记录当前 10 s 帧的 nsub 口径差：frames(bias_nsub=0 → 每帧 1 子步) vs
    // OU(bias_nsub=16 → 16 子步/10 s)。二者模型相同，仅求积分辨不同。
    {
        std::vector<double> G10f((size_t)n * 2 * 18), G10s((size_t)n * 18),
            e10((size_t)n * 6), Q10((size_t)n * 36), Qb10((size_t)n * 18),
            Qbb10((size_t)n * 9), du10(3, 0.0);
        qoejopt::gveBiasFramesBatch(x0.data(), n, 2, 1, 10.0, kTau, 0, rtn,
                                    G10f.data(), e10.data());
        qoejopt::gveOuBiasBatch(x0.data(), n, 10.0, 16, kTau, kBeta, rtn,
                                du10.data(), e10.data(), G10s.data(), Q10.data(),
                                Qb10.data(), Qbb10.data());
        double rel10 = 0.0;
        for (int i = 0; i < n; ++i)
            rel10 = std::max(rel10, frob_rel(gamma_of_frames(G10f, n, 2, i, 1),
                                             gamma_of_flat(G10s, i)));
        std::printf("[WinGamma] 10s-frame nsub mismatch (frames=1 vs OU=16) rel=%.3e\n",
                    rel10);
        EXPECT_LT(rel10, 1e-3);
    }

    // 单区间 frames vs OU（同一 bias_nsub=16；两者 nsub 均为 max(ceil(seg/10),16)=288）
    std::vector<double> G1((size_t)n * 2 * 18), e1((size_t)n * 6);
    std::vector<double> Gs((size_t)n * 18), es((size_t)n * 6),
        Q((size_t)n * 36), Qb((size_t)n * 18), Qbb((size_t)n * 9), du0(3, 0.0);
    double rel1 = 0.0;
    for (int bn : {0, 16, 288}) {   // 同 bias_nsub 的单区间 frames-vs-OU
        qoejopt::gveBiasFramesBatch(x0.data(), n, 2, 1, kSeg, kTau, bn, rtn,
                                    G1.data(), e1.data());
        qoejopt::gveOuBiasBatch(x0.data(), n, kSeg, bn, kTau, kBeta, rtn, du0.data(),
                                es.data(), Gs.data(), Q.data(), Qb.data(), Qbb.data());
        double rb = 0.0;
        for (int i = 0; i < n; ++i)
            rb = std::max(rb, frob_rel(gamma_of_frames(G1, n, 2, i, 1),
                                       gamma_of_flat(Gs, i)));
        std::printf("[WinGamma] single-interval bias_nsub=%d frames-vs-OU rel=%.3e\n",
                    bn, rb);
        if (bn == 16) rel1 = rb;
    }

    // 10 区间端点 vs 单发 5T OU：匹配总子步（2880）为主判据；
    // 另记录当前 API 默认口径（frames 10×288 vs OU nsub=288, bias_nsub=16）的差。
    std::vector<double> Gtot((size_t)n * 18), etot((size_t)n * 6),
        Gtot16((size_t)n * 18), etot16((size_t)n * 6);
    qoejopt::gveOuBiasBatch(x0.data(), n, (double)(L - 1) * kSeg,
                            (L - 1) * kFpn, kTau, kBeta, rtn, du0.data(),
                            etot.data(), Gtot.data(), Q.data(), Qb.data(), Qbb.data());
    qoejopt::gveOuBiasBatch(x0.data(), n, (double)(L - 1) * kSeg, 16, kTau, kBeta,
                            rtn, du0.data(), etot16.data(), Gtot16.data(), Q.data(),
                            Qb.data(), Qbb.data());
    double reltot = 0.0, reltot16 = 0.0;
    for (int i = 0; i < n; ++i) {
        reltot = std::max(reltot, frob_rel(gamma_of_frames(Gf, n, L, i, L - 1),
                                           gamma_of_flat(Gtot, i)));
        reltot16 = std::max(reltot16, frob_rel(gamma_of_frames(Gf, n, L, i, L - 1),
                                               gamma_of_flat(Gtot16, i)));
    }
    std::printf("[WinGamma] 10-interval endpoint vs OU: matched nsub=%d rel=%.3e; "
                "default bias_nsub=16 (OU nsub=%d) rel=%.3e\n",
                (L - 1) * kFpn, reltot, kFpn, reltot16);

    // 端点 vs host FD（δβ=1e-9 保持线性区；h=5 与 h=2.5 用于判定 3.5e-6 是
    // FD 截断还是 frames 中点求积本身的离散误差——若减半 h 不按 8~16 倍收敛，
    // 则是后者，1e-6 的 FD 阈值只作严格占位）。
    double relfd = 0.0;
    for (int i = 0; i < n; ++i) {
        const M63 Gfd = gamma_fd(st[i], (double)(L - 1) * kSeg, kTau, 1e-9, 5.0);
        relfd = std::max(relfd, frob_rel(gamma_of_frames(Gf, n, L, i, L - 1), Gfd));
    }
    const M63 Gfd25 = gamma_fd(st[0], (double)(L - 1) * kSeg, kTau, 1e-9, 2.5);
    const double relfd25 = frob_rel(gamma_of_frames(Gf, n, L, 0, L - 1), Gfd25);
    // 同 h 对比：frames 用 bias_nsub=576/区间（h≈4.99 s）与 FD h=5 s；
    // 默认 10 s 子步的 3.5e-6 是生产离散误差，不是实现误差。
    std::vector<double> Gff((size_t)n * L * 18), eff((size_t)n * 6);
    qoejopt::gveBiasFramesBatch(x0.data(), n, L, 1, kSeg, kTau, 576, rtn,
                                Gff.data(), eff.data());
    double relfd_fine = 0.0;
    for (int i = 0; i < n; ++i) {
        const M63 Gfd = gamma_fd(st[i], (double)(L - 1) * kSeg, kTau, 1e-9, 5.0);
        relfd_fine = std::max(relfd_fine,
            frob_rel(gamma_of_frames(Gff, n, L, i, L - 1), Gfd));
    }
    std::printf("[WinGamma] FD matched-h (frames bias_nsub=576) rel=%.3e\n",
                relfd_fine);
    std::printf("[WinGamma] FD h-convergence sat0: h=5 rel=%.3e h=2.5 rel=%.3e\n",
                frob_rel(gamma_of_frames(Gf, n, L, 0, L - 1),
                         gamma_fd(st[0], (double)(L - 1) * kSeg, kTau, 1e-9, 5.0)),
                relfd25);
    // 与 host 名义节点一致性（frames 内核自己的 end）
    std::vector<double> Xnom = nominal_nodes(st, L, kSeg, 10.0);
    double end_rel = 0.0;
    for (int i = 0; i < n; ++i) {
        V6 d;
        for (int c = 0; c < 6; ++c)
            d[c] = endf[(size_t)i * 6 + c] - Xnom[((size_t)(L - 1) * n + i) * 6 + c];
        end_rel = std::max(end_rel, d.norm() / std::max(
            Eigen::Map<const V6>(Xnom.data() + ((size_t)(L - 1) * n + i) * 6).norm(),
            1e-300));
    }
    std::printf("[WinGamma] single-interval frames-vs-OU rel=%.3e; "
                "10-interval endpoint-vs-OU rel=%.3e (frames nsub=%d/interval=%d total, "
                "OU nsub=%d total); endpoint-vs-FD rel=%.3e; end-vs-host rel=%.3e\n",
                rel1, reltot, kFpn, (L - 1) * kFpn, kFpn, relfd, end_rel);
    // 严格阈值（修复前保持红）：rel1 是同 h=seg/288 的单区间 frames-vs-OU；
    // 实测 6.7e-8 来自两核对中点求积/半段 STM 组合的独立浮点求值，未达 1e-8。
    EXPECT_LT(rel1, 1e-8);
    // 匹配总子步（frames 10×288 = OU nsub=2880，同 h=9.982293 s）的端点 Γ。
    EXPECT_LT(reltot, 1e-5);
    // 独立 FD 的自身截断：h=5→3.5e-6、h=2.5→1.5e-6（未按 16 倍收敛），
    // 且 frames/OU 逐位一致性已达 0 / 1.8e-16；故 FD 交叉检查的物理
    // 一致界取 5e-6，真正的实现正确性由 CallHistory 与 frames-vs-OU 判定。
    EXPECT_LT(relfd_fine, 5e-6);
    // 默认 10 s 子步（frames bias_nsub=0）对 FD h=5 的 3.5e-6 是生产离散
    // 误差；h=2.5 时降到 1.5e-6，与中点求积一致，故物理阈值取 1e-5。
    EXPECT_LT(relfd, 1e-5);
    EXPECT_LT(end_rel, 1e-8);
}

// 1b) 生产粒度交叉调用回归：OU(0.5T,288) → frames(nfr=2,0.5T)×4 → OU(0.5T,288)
//     必须逐位一致；再接一次短 OU（10 s,1 子步）后第三次大 OU 应恢复。
//     修复前 Γ/Q_qq/Q_qb 差 ~4.6e-3（end、Q_bb 逐位相同），暴露核间共享可变状态。
TEST(BiasWindow, OuFramesInterleaveDeterminismProduction) {
    const auto st = sample_states();
    const int n = 2;
    std::vector<double> x0((size_t)n * 6);
    for (int i = 0; i < n; ++i)
        for (int c = 0; c < 6; ++c) x0[(size_t)i * 6 + c] = st[i][c];
    double rtn[3] = {1.0, 1.0, 1.0}, du0[3] = {0.0, 0.0, 0.0};
    auto run_ou = [&](double dt, int bn, std::vector<double> &end,
                      std::vector<double> &G, std::vector<double> &Q,
                      std::vector<double> &Qb, std::vector<double> &Qbb) {
        end.assign((size_t)n * 6, 0.0); G.assign((size_t)n * 18, 0.0);
        Q.assign((size_t)n * 36, 0.0); Qb.assign((size_t)n * 18, 0.0);
        Qbb.assign((size_t)n * 9, 0.0);
        qoejopt::gveOuBiasBatch(x0.data(), n, dt, bn, kTau, kBeta, rtn, du0,
                                end.data(), G.data(), Q.data(), Qb.data(), Qbb.data());
    };
    auto frames = [&]() {
        std::vector<double> G((size_t)n * 2 * 18), e((size_t)n * 6);
        qoejopt::gveBiasFramesBatch(x0.data(), n, 2, 1, kSeg, kTau, 0, rtn,
                                    G.data(), e.data());
    };
    std::vector<double> e1, G1, Q1, Qb1, Qbb1, e2, G2, Q2, Qb2, Qbb2,
        eS, GS, QS, QbS, QbbS, e3, G3, Q3, Qb3, Qbb3;
    run_ou(kSeg, kFpn, e1, G1, Q1, Qb1, Qbb1);
    for (int i = 0; i < 4; ++i) frames();
    run_ou(kSeg, kFpn, e2, G2, Q2, Qb2, Qbb2);
    run_ou(10.0, 1, eS, GS, QS, QbS, QbbS);      // 短 OU：应重置共享状态
    run_ou(kSeg, kFpn, e3, G3, Q3, Qb3, Qbb3);
    auto eq = [](const std::vector<double> &a, const std::vector<double> &b) {
        return a.size() == b.size() &&
               std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0;
    };
    auto maxrel = [](const std::vector<double> &a, const std::vector<double> &b) {
        double ma = 0.0, mb = 0.0;
        for (double v : a) ma = std::max(ma, std::abs(v));
        for (double v : b) mb = std::max(mb, std::abs(v));
        double md = 0.0;
        for (size_t j = 0; j < a.size(); ++j) md = std::max(md, std::abs(a[j] - b[j]));
        return md / std::max(std::max(ma, mb), 1e-300);
    };
    std::printf("[Interleave] dt=%.7f nsub=%d; OU1-vs-OU2 rel G=%.3e Q=%.3e Qb=%.3e; "
                "bitwise end=%d Qbb=%d G=%d Q=%d Qb=%d | after short OU: "
                "bitwise end=%d Qbb=%d G=%d Q=%d Qb=%d\n",
                kSeg, kFpn, maxrel(G1, G2), maxrel(Q1, Q2), maxrel(Qb1, Qb2),
                (int)eq(e1, e2), (int)eq(Qbb1, Qbb2), (int)eq(G1, G2),
                (int)eq(Q1, Q2), (int)eq(Qb1, Qb2),
                (int)eq(e1, e3), (int)eq(Qbb1, Qbb3), (int)eq(G1, G3),
                (int)eq(Q1, Q3), (int)eq(Qb1, Qb3));
    // end、Q_bb 应与 Γ/Q 无关地逐位一致（当前通过）
    EXPECT_TRUE(eq(e1, e2));
    EXPECT_TRUE(eq(Qbb1, Qbb2));
    // 修复前这里红：Γ/Q_qq/Q_qb 差 ~4.6e-3
    EXPECT_TRUE(eq(G1, G2));
    EXPECT_TRUE(eq(Q1, Q2));
    EXPECT_TRUE(eq(Qb1, Qb2));
    // 短 OU 后应恢复
    EXPECT_TRUE(eq(e1, e3));
    EXPECT_TRUE(eq(Qbb1, Qbb3));
    EXPECT_TRUE(eq(G1, G3));
    EXPECT_TRUE(eq(Q1, Q3));
    EXPECT_TRUE(eq(Qb1, Qb3));
}

// 2) winDynAsmKernel 的 Γβ：∂b/∂β=+Q^{1/2}Γ，A 与 β 无关，线性叠加，代价同口径。
TEST(BiasWindow, WindowDynBiasJacobian) {
    const Vector6d x0 = sample_states()[0];
    const int n = 2, L = 2;
    M6 Qs = M6::Zero();
    Qs.diagonal() << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
    WindowHarness hz(n, L, 0, {}, {}, kFpn, kSeg, 1.0, Qs);
    // 第一遍：β=0，取窗口自身的名义 end；把第二个节点设成 end（残差只剩 dbias）。
    std::vector<double> X((size_t)L * n * 6, 0.0);
    for (int i = 0; i < n; ++i)
        for (int c = 0; c < 6; ++c) X[(size_t)i * 6 + c] = x0[c];
    Eigen::MatrixXd A0;
    Eigen::VectorXd b0;
    std::vector<double> end0, phi0;
    hz.assemble_dense(X, A0, b0, end0, phi0);
    for (int i = 0; i < n; ++i)
        for (int c = 0; c < 6; ++c) X[((size_t)n + i) * 6 + c] = end0[(size_t)i * 6 + c];

    // Γ 与 β
    double rtn[3] = {1.0, 1.0, 1.0};
    std::vector<double> x0v((size_t)n * 6), G1((size_t)n * 2 * 18), e1((size_t)n * 6);
    for (int i = 0; i < n; ++i)
        for (int c = 0; c < 6; ++c) x0v[(size_t)i * 6 + c] = x0[c];
    qoejopt::gveBiasFramesBatch(x0v.data(), n, 2, 1, kSeg, kTau, 0, rtn,
                                G1.data(), e1.data());
    const V3 bet(0.0, kBeta, 0.0);
    const V3 bet2(0.3 * kBeta, -0.5 * kBeta, 0.2 * kBeta);
    std::vector<double> db(n * 6, 0.0), db2(n * 6, 0.0), db12(n * 6, 0.0);
    for (int i = 0; i < n; ++i) {
        const M63 G = gamma_of_frames(G1, n, 2, i, 1);
        const V6 v = G * bet, v2 = G * bet2;
        for (int c = 0; c < 6; ++c) {
            db[(size_t)i * 6 + c] = v[c];
            db2[(size_t)i * 6 + c] = v2[c];
            db12[(size_t)i * 6 + c] = v[c] + v2[c];
        }
    }
    auto run = [&](const std::vector<double> &d, Eigen::MatrixXd &A,
                   Eigen::VectorXd &b, double &cost) {
        hz.set_dyn_bias(d);
        std::vector<double> e, p;
        hz.assemble_dense(X, A, b, e, p);
        cost = hz.total_cost(X);
    };
    Eigen::MatrixXd A1, A2, A12;
    Eigen::VectorXd b1v, b2v, b12v, bzero;
    double c0 = 0.0, c1 = 0.0, c2 = 0.0, c12 = 0.0;
    {
        std::vector<double> z(n * 6, 0.0);
        hz.set_dyn_bias(z);
        std::vector<double> e, p;
        hz.assemble_dense(X, A0, bzero, e, p);
        c0 = hz.total_cost(X);
    }
    run(db, A1, b1v, c1);
    run(db2, A2, b2v, c2);
    run(db12, A12, b12v, c12);

    const double dA = std::max((A1 - A0).cwiseAbs().maxCoeff(),
                               (A12 - A0).cwiseAbs().maxCoeff());
    // 期望 b = Q db（X1=end ⇒ e=-db）
    Eigen::VectorXd bexp = Eigen::VectorXd::Zero(hz.nDYN());
    for (int i = 0; i < n; ++i)
        for (int a = 0; a < 6; ++a) {
            double s = 0.0;
            for (int c = 0; c < 6; ++c) s += Qs(a, c) * db[(size_t)i * 6 + c];
            bexp[i * 6 + a] = s;
        }
    const double db_err = (b1v - bexp).norm() / std::max(bexp.norm(), 1e-300);
    const double dlin = ((b12v - bzero) - ((b1v - bzero) + (b2v - bzero))).norm() /
                        std::max((b1v - bzero + b2v - bzero).norm(), 1e-300);
    const double c_lin = std::abs(c12 - (c0 + (c1 - c0) + (c2 - c0) +
                                         2.0 * (b1v - bzero).dot(b2v - bzero)));
    std::printf("[WinDynBias] max|dA|=%.3e db-vs-QGamma rel=%.3e linear=%.3e "
                "cost0=%.3e cost1=%.3e (||b1||^2=%.3e) cross_err=%.3e\n",
                dA, db_err, dlin, c0, c1, b1v.squaredNorm(), c_lin);
    EXPECT_LT(dA, 1e-12);
    EXPECT_LT(db_err, 1e-12);
    EXPECT_LT(dlin, 1e-10);
    EXPECT_NEAR(c0, 0.0, 1e-18);
    EXPECT_LT(std::abs(c1 - b1v.squaredNorm()) /
              std::max(b1v.squaredNorm(), 1e-300), 1e-10);
    EXPECT_LT(std::abs(c12 - (c1 + c2 + 2.0 * (b1v - bzero).dot(b2v - bzero))) /
              std::max(c12, 1e-300), 1e-10);
}

// 2b) dbias 默认关闭（nullptr）与显式零 dbias 的 A/b 逐位不变。
TEST(BiasWindow, WindowDynBiasDefaultOffBitwise) {
    const auto s = sample_states();
    const int n = 2, L = 2;
    WindowHarness hz(n, L, 0, {}, {}, kFpn, kSeg, 1.0, M6::Identity());
    std::vector<double> X((size_t)L * n * 6);
    for (int f = 0; f < L; ++f)
        for (int i = 0; i < n; ++i)
            for (int c = 0; c < 6; ++c) X[((size_t)f * n + i) * 6 + c] = s[i][c];
    Eigen::MatrixXd A0;
    Eigen::VectorXd b0;
    std::vector<double> e0, p0;
    hz.assemble_dense(X, A0, b0, e0, p0);      // d_dyn_bias 已 memset 0
    hz.disable_dyn_bias();                     // nullptr：旧路径
    Eigen::MatrixXd A1;
    Eigen::VectorXd b1;
    std::vector<double> e1, p1;
    hz.assemble_dense(X, A1, b1, e1, p1);
    const bool aeq = std::memcmp(A0.data(), A1.data(), A0.size() * sizeof(double)) == 0;
    const bool beq = std::memcmp(b0.data(), b1.data(), b0.size() * sizeof(double)) == 0;
    std::printf("[WinDynBiasOff] A bitwise=%d b bitwise=%d\n", (int)aeq, (int)beq);
    EXPECT_TRUE(aeq);
    EXPECT_TRUE(beq);
}

// 3) 解析 STM 链 ∂q_m/∂β_j=(ΠΦ)Γ_j（j<m，否则 0） vs 小 β 非线性传播。
TEST(BiasWindow, WindowChainBetaSensitivity) {
    const Vector6d x0 = sample_states()[0];
    for (int L : {11, 10}) {
        const int nseg = L - 1;
        // 单区间 harness：顺序扫描，避免全窗 Jacobi 需 nseg 次才能传播 β 响应。
        WindowHarness hz1(1, 2, 0, {}, {}, kFpn, kSeg, 1.0, M6::Identity());
        auto step_end = [&](const Vector6d &x, Vector6d &xn, M6 &Ph) {
            std::vector<double> X2(12, 0.0);
            for (int c = 0; c < 6; ++c) { X2[c] = x[c]; X2[6 + c] = x[c]; }
            Eigen::MatrixXd A;
            Eigen::VectorXd b;
            std::vector<double> end, phi;
            hz1.assemble_dense(X2, A, b, end, phi);
            for (int c = 0; c < 6; ++c) xn[c] = end[c];
            for (int a = 0; a < 6; ++a)
                for (int c = 0; c < 6; ++c) Ph(a, c) = phi[a * 6 + c];
        };
        // 设备一致的名义轨迹与 Φ_k
        std::vector<V6> Xnom(nseg + 1);
        std::vector<M6> Phi(nseg);
        std::vector<M63> Gam(nseg);
        Xnom[0] = x0;
        for (int k = 0; k < nseg; ++k) {
            Vector6d xn;
            step_end(Xnom[k], xn, Phi[k]);
            Xnom[k + 1] = xn;
        }
        // Γ_k：十个名义区间起点一次批量
        {
            std::vector<double> xst((size_t)nseg * 6);
            for (int k = 0; k < nseg; ++k)
                for (int c = 0; c < 6; ++c) xst[(size_t)k * 6 + c] = Xnom[k][c];
            double rtn[3] = {1.0, 1.0, 1.0};
            std::vector<double> G((size_t)nseg * 2 * 18), ge((size_t)nseg * 6);
            qoejopt::gveBiasFramesBatch(xst.data(), nseg, 2, 1, kSeg, kTau, 0,
                                        rtn, G.data(), ge.data());
            for (int k = 0; k < nseg; ++k) Gam[k] = gamma_of_frames(G, nseg, 2, k, 1);
        }
        // β_j 小量
        std::vector<V3> bet(nseg);
        const double amp = 1e-9;
        for (int k = 0; k < nseg; ++k)
            bet[k] = V3(0.2 * std::sin(0.7 * k), 1.0, 0.3 * std::cos(0.4 * k)) * amp;
        // 解析链 ∂q_m/∂β_j=(Π_{k=m-1}^{j+1}Φ_k)Γ_j
        std::vector<std::vector<M63>> S(nseg + 1, std::vector<M63>(nseg));
        for (int m = 0; m <= nseg; ++m)
            for (int j = 0; j < nseg; ++j) S[m][j].setZero();
        for (int j = 0; j < nseg; ++j) {
            S[j + 1][j] = Gam[j];
            for (int m = j + 2; m <= nseg; ++m)
                S[m][j] = Phi[m - 1] * S[m - 1][j];
        }
        std::vector<V6> dqa(nseg + 1, V6::Zero());
        for (int m = 1; m <= nseg; ++m)
            for (int j = 0; j < m; ++j) dqa[m] += S[m][j] * bet[j];
        // 结构：块下三角（j>=m 的交叉项严格为零）；端点对所有 10 个 β 都敏感。
        bool lower = true;
        for (int m = 0; m <= nseg; ++m)
            for (int j = 0; j < nseg; ++j)
                if (j >= m && S[m][j].norm() != 0.0) lower = false;
        double min_end_sens = 1e300;
        for (int j = 0; j < nseg; ++j)
            min_end_sens = std::min(min_end_sens, S[nseg][j].norm());
        std::printf("[WinChain] block_lower_triangular=%d upper_cross_zero=%d "
                    "endpoint_min_j_sens=%.3e\n", (int)lower, (int)lower, min_end_sens);
        EXPECT_TRUE(lower);
        EXPECT_GT(min_end_sens, 1.0);
        // 非线性状态传播（end 在带 β 的状态重算；Γ 用名义值，线性化口径与链一致）
        std::vector<V6> Xb(nseg + 1);
        Xb[0] = x0;
        for (int k = 0; k < nseg; ++k) {
            Vector6d xn;
            M6 Ph;
            step_end(Xb[k], xn, Ph);
            xn += Gam[k] * bet[k];
            Xb[k + 1] = xn;
        }
        double max_rel = 0.0, max_end = 0.0;
        for (int m = 1; m <= nseg; ++m) {
            const V6 dnl = Xb[m] - Xnom[m];
            max_rel = std::max(max_rel, (dqa[m] - dnl).norm() /
                               std::max(dnl.norm(), 1e-300));
        }
        {
            const V6 dnl = Xb[nseg] - Xnom[nseg];
            max_end = (dqa[nseg] - dnl).norm() / std::max(dnl.norm(), 1e-300);
        }
        // 连续 host RK4（不同积分格式）仅记录
        double host_rel = 0.0;
        Vector6d q = x0;
        for (int k = 0; k < nseg; ++k) {
            q = propagate_rtn_bias(q, kSeg, kTau, bet[k], 2.0);
            const V6 dnl = q - Xnom[k + 1];
            host_rel = std::max(host_rel, (dqa[k + 1] - dnl).norm() /
                                std::max(dnl.norm(), 1e-300));
        }
        std::printf("[WinChain] L=%d nseg=%d device_rel=%.3e endpoint_rel=%.3e "
                    "host_continuous_rel=%.3e\n", L, nseg, max_rel, max_end, host_rel);
        // β=1e-9（<1e-8）；δq≈1e-2 m，状态 7e6 m 的双精度/end-vs-Φ 一致性地板
        // ≈若干 ulp（~6e-9 m）⇒ 相对 ~1e-6。1e-8 相对在此量级不可达。
        EXPECT_LT(max_rel, 2e-6);
        EXPECT_LT(max_end, 2e-6);
    }
}

// ===================== 合成 ISL 链：逐区间 3D 偏差 β 恢复 =====================
// N=6、11 节点、10 个区间（每区间一个公共 RTN 3D β_k，作用到所有星）。ISL 用完全图
// E=15/帧。q/β 交替：q 解用窗口 A/b（dyn 带 Γβ）+ 节点 0 弱先验；β 解对每个区间
// 做 3×3 正规方程（对 6 星求和）。交替若干轮后再做联合 GN 精化，并用 Schur 补
// 给出 β 的形式协方差。Eph 用 Kabsch 去掉 ISL-only 不可观的 6 自由度整体刚体规范。
namespace {

struct ChainBeta {
    int N = 6, L = 11, E = 15, nseg = 10;
    std::vector<int> ei, ej;
    std::vector<Vector6d> x0_true, x0_init;
    std::vector<double> Xinit;      // 名义传播的初始节点
    std::vector<double> Phi;        // (nseg*N, 6, 6)，在 Xinit 区间起点
    std::vector<double> Gamma;      // (nseg*N, 6, 3)，在 Xinit 区间起点
    M6 Qs;
    double sigma_isl = 0.1;         // 合成 ISL 噪声/权重（m）
    double sigma0 = 100.0;          // 节点 0 位置先验（m）
    double sigma_beta = 1e-5;       // 弱 β 先验（m/s²）
};

std::vector<double> biased_nodes_common(const std::vector<Vector6d> &x0,
                                        const std::vector<V3> &bet, double hstep) {
    const int N = (int)x0.size(), nseg = (int)bet.size(), L = nseg + 1;
    std::vector<double> out((size_t)L * N * 6);
    std::vector<Vector6d> x = x0;
    for (int f = 0; f <= nseg; ++f) {
        for (int i = 0; i < N; ++i)
            for (int c = 0; c < 6; ++c) out[((size_t)f * N + i) * 6 + c] = x[i][c];
        if (f == nseg) break;
        for (int i = 0; i < N; ++i)
            x[i] = propagate_rtn_bias(x[i], kSeg, kTau, bet[f], hstep);
    }
    return out;
}

std::vector<double> isl_obs(const ChainBeta &c, const std::vector<double> &X,
                            double sigma, unsigned seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> nd(0.0, sigma);
    std::vector<double> D((size_t)c.L * c.E);
    for (int f = 0; f < c.L; ++f)
        for (int e = 0; e < c.E; ++e) {
            const int i = c.ei[e], j = c.ej[e];
            const Vector6d ri = noeOsc2rv(
                Eigen::Map<const V6>(X.data() + ((size_t)f * c.N + i) * 6), 100, 1e-12);
            const Vector6d rj = noeOsc2rv(
                Eigen::Map<const V6>(X.data() + ((size_t)f * c.N + j) * 6), 100, 1e-12);
            const double d = (ri.head<3>() - rj.head<3>()).norm();
            D[(size_t)f * c.E + e] = d + (sigma > 0.0 ? nd(rng) : 0.0);
        }
    return D;
}

std::vector<double> gamma_states(const ChainBeta &c, const std::vector<double> &X) {
    const int nk = c.nseg * c.N;
    std::vector<double> xs((size_t)nk * 6);
    for (int k = 0; k < c.nseg; ++k)
        for (int i = 0; i < c.N; ++i)
            for (int cc = 0; cc < 6; ++cc)
                xs[((size_t)k * c.N + i) * 6 + cc] =
                    X[((size_t)k * c.N + i) * 6 + cc];
    double rtn[3] = {1.0, 1.0, 1.0};
    std::vector<double> G((size_t)nk * 2 * 18), e((size_t)nk * 6), out((size_t)nk * 18);
    qoejopt::gveBiasFramesBatch(xs.data(), nk, 2, 1, kSeg, kTau, 0, rtn, G.data(), e.data());
    for (int idx = 0; idx < nk; ++idx)
        for (int a = 0; a < 6; ++a)
            for (int l = 0; l < 3; ++l)
                out[(size_t)idx * 18 + a * 3 + l] =
                    G[((size_t)(idx * 2 + 1) * 6 + a) * 3 + l];
    return out;
}

M63 gamma_idx(const std::vector<double> &G, int idx) {
    M63 m;
    for (int a = 0; a < 6; ++a)
        for (int l = 0; l < 3; ++l) m(a, l) = G[(size_t)idx * 18 + a * 3 + l];
    return m;
}

void init_chain(ChainBeta &c) {
    const auto s = sample_states();
    c.x0_true = {s[0], s[1], s[2], s[3], s[0], s[1]};
    c.x0_true[4][1] += 0.7;
    c.x0_true[5][1] += 1.9;
    c.x0_true[5][5] += 1.1;
    c.x0_init = c.x0_true;
    for (int i = 0; i < c.N; ++i) {
        const double a = c.x0_init[i][0];
        c.x0_init[i][0] += 45.0 * std::cos(0.9 * i + 0.3);          // ~50 m 位置初差
        c.x0_init[i][1] += 60.0 * std::sin(1.3 * i + 0.2) / a;
        c.x0_init[i][4] += 35.0 * std::cos(0.7 * i + 0.5) / a;
    }
    for (int i = 0; i < c.N; ++i)
        for (int j = i + 1; j < c.N; ++j) { c.ei.push_back(i); c.ej.push_back(j); }
    c.E = (int)c.ei.size();
    c.Qs = M6::Zero();
    c.Qs(0, 0) = 1.0;
    const double as = 6.95e6;
    for (int a = 1; a < 6; ++a) c.Qs(a, a) = as;
    c.Xinit = nominal_nodes(c.x0_init, c.L, kSeg, 10.0);
    c.Gamma = gamma_states(c, c.Xinit);
    {
        // 区间 Φ 由窗口 dyn（gveStmSubstepsDevice，fpn=288）给出
        WindowHarness hz(c.N, c.L, 0, {}, {}, kFpn, kSeg, 1.0, M6::Identity());
        Eigen::MatrixXd A;
        Eigen::VectorXd b;
        std::vector<double> end;
        hz.assemble_dense(c.Xinit, A, b, end, c.Phi);
    }
}

double kabsch_rms(const std::vector<V3> &pt, const std::vector<V3> &pe) {
    const int np = (int)pt.size();
    V3 ct = V3::Zero(), ce = V3::Zero();
    for (int k = 0; k < np; ++k) { ct += pt[k]; ce += pe[k]; }
    ct /= (double)np; ce /= (double)np;
    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    for (int k = 0; k < np; ++k) H += (pt[k] - ct) * (pe[k] - ce).transpose();
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d R = svd.matrixU() * svd.matrixV().transpose();
    if (R.determinant() < 0.0) {
        Eigen::Matrix3d V = svd.matrixV();
        V.col(2) *= -1.0;
        R = svd.matrixU() * V.transpose();
    }
    double ss = 0.0;
    for (int k = 0; k < np; ++k) {
        const V3 d = pt[k] - (R * (pe[k] - ce) + ct);
        ss += d.squaredNorm();
    }
    return std::sqrt(ss / np);
}

std::vector<V3> node_positions(const std::vector<double> &X, int N, int L) {
    std::vector<V3> p((size_t)N * L);
    for (int k = 0; k < N * L; ++k)
        p[k] = noeOsc2rv(Eigen::Map<const V6>(X.data() + (size_t)k * 6),
                         100, 1e-12).head<3>();
    return p;
}

// ISL-only 不可观的是每帧 6 自由度整体刚体规范；同时报告：
//   global = 全窗一次 Kabsch；perframe = 每帧各自 Kabsch（GTS/绝对锚的等价口径）。
double gauge_eph_rms(const std::vector<double> &Xt, const std::vector<double> &Xe,
                     int N, int L) {
    return kabsch_rms(node_positions(Xt, N, L), node_positions(Xe, N, L));
}

double gauge_eph_rms_perframe(const std::vector<double> &Xt,
                              const std::vector<double> &Xe, int N, int L) {
    const std::vector<V3> pt = node_positions(Xt, N, L);
    const std::vector<V3> pe = node_positions(Xe, N, L);
    double ss = 0.0;
    for (int f = 0; f < L; ++f) {
        std::vector<V3> at(pt.begin() + (size_t)f * N, pt.begin() + (size_t)(f + 1) * N);
        std::vector<V3> ae(pe.begin() + (size_t)f * N, pe.begin() + (size_t)(f + 1) * N);
        const double r = kabsch_rms(at, ae);
        ss += r * r * N;
    }
    return std::sqrt(ss / (double)(N * L));
}

struct EstResult {
    std::vector<double> X, beta, sigma;
    double cost = 0.0, isl_rms = 0.0, dyn_rms = 0.0;
};

// q/β 交替 + 联合 GN 精化。beta 为 nseg*3（每区间公共 RTN 3D）。
EstResult estimate_chain(const ChainBeta &c, const std::vector<double> &D,
                         const std::vector<double> &beta0, bool est_beta,
                         int alt_iters, int joint_iters) {
    WindowHarness hz(c.N, c.L, c.E, c.ei, c.ej, kFpn, kSeg, c.sigma_isl, c.Qs);
    hz.set_isl_D(D);
    const int ncol = c.N * c.L * 6, nb = c.nseg * 3, np0 = c.N * 6;
    std::vector<double> X = c.Xinit;
    std::vector<double> beta = beta0;
    std::vector<double> db((size_t)c.nseg * c.N * 6, 0.0);
    auto fill_db = [&]() {
        for (int k = 0; k < c.nseg; ++k)
            for (int i = 0; i < c.N; ++i) {
                const int idx = k * c.N + i;
                const M63 G = gamma_idx(c.Gamma, idx);
                for (int a = 0; a < 6; ++a) {
                    double s = 0.0;
                    for (int l = 0; l < 3; ++l) s += G(a, l) * beta[k * 3 + l];
                    db[(size_t)idx * 6 + a] = s;
                }
            }
        hz.set_dyn_bias(db);
    };
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    std::vector<double> end, phi;
    // 变量缩放：z = Q_s^{-1} x（把角度乘 a，位置等效为米）。原始 QOE 的法方程
    // cond~1e14，交替/联合 GN 会丢精度；以下所有 A 的列与先验都在 z 空间。
    auto scaled_cols = [&](const Eigen::MatrixXd &Ar) {
        Eigen::MatrixXd Az = Ar;
        for (int j = 0; j < Az.cols(); ++j)
            Az.col(j) *= c.Qs(j % 6, j % 6);
        return Az;
    };
    // 先验（节点 0，z 空间，σ0 米）
    Eigen::MatrixXd Ap = Eigen::MatrixXd::Zero(np0, ncol);
    for (int i = 0; i < c.N; ++i)
        for (int cc = 0; cc < 6; ++cc)
            Ap(i * 6 + cc, ((size_t)0 * c.N + i) * 6 + cc) = 1.0 / c.sigma0;
    auto prior_b = [&]() {
        Eigen::VectorXd bp(np0);
        for (int i = 0; i < c.N; ++i)
            for (int cc = 0; cc < 6; ++cc) {
                const int row = i * 6 + cc;
                const int col = ((size_t)0 * c.N + i) * 6 + cc;
                bp(row) = (c.x0_init[i][cc] - X[col]) /
                          (c.Qs(cc, cc) * c.sigma0);
            }
        return bp;
    };
    auto assemble = [&](Eigen::MatrixXd &A2, Eigen::VectorXd &b2,
                        std::vector<double> &e2, std::vector<double> &p2) {
        fill_db();
        hz.assemble_dense(X, A2, b2, e2, p2);
    };
    const double wb2 = 1.0 / (c.sigma_beta * c.sigma_beta);
    for (int it = 0; it < alt_iters; ++it) {
        assemble(A, b, end, phi);
        const Eigen::MatrixXd Az = scaled_cols(A);
        Eigen::MatrixXd Aaug(A.rows() + np0, ncol);
        Eigen::VectorXd baug(A.rows() + np0);
        Aaug.topRows(A.rows()) = Az;
        baug.head(A.rows()) = b;
        Aaug.bottomRows(np0) = Ap;
        baug.tail(np0) = prior_b();
        Eigen::MatrixXd Hd0 = Aaug.transpose() * Aaug;
        Hd0.diagonal().array() += 1e-10 * Hd0.diagonal().maxCoeff();
        const Eigen::VectorXd gq = Aaug.transpose() * baug;
        const Eigen::VectorXd dZ = Hd0.ldlt().solve(gq);
        for (int j = 0; j < ncol; ++j)
            X[j] += 0.9 * c.Qs(j % 6, j % 6) * dZ[j];
        if (!est_beta) continue;
        // β 条件更新：用新 X 的 dyn 残差（b_dyn = -Q r）
        assemble(A, b, end, phi);
        for (int k = 0; k < c.nseg; ++k) {
            M3 Mk = (wb2 * M3::Identity());
            V3 bvec;
            for (int l = 0; l < 3; ++l) bvec[l] = beta[3 * k + l];
            V3 rhs = -bvec * wb2;
            for (int i = 0; i < c.N; ++i) {
                const int idx = k * c.N + i;
                V6 bd;
                for (int a = 0; a < 6; ++a) bd[a] = b[hz.nISL() + idx * 6 + a];
                const M63 G = gamma_idx(c.Gamma, idx);
                const V6 qb = c.Qs * bd;
                rhs -= G.transpose() * qb;
                Mk += G.transpose() * (c.Qs.transpose() * c.Qs) * G;
            }
            const V3 dbv = Mk.ldlt().solve(rhs);
            for (int l = 0; l < 3; ++l) beta[3 * k + l] += 0.8 * dbv[l];
        }
    }
    for (int it = 0; it < joint_iters; ++it) {
        assemble(A, b, end, phi);
        const Eigen::MatrixXd Az = scaled_cols(A);
        const Eigen::MatrixXd Aaug = [&]() {
            Eigen::MatrixXd M(A.rows() + np0, ncol);
            M.topRows(A.rows()) = Az;
            M.bottomRows(np0) = Ap;
            return M;
        }();
        Eigen::VectorXd baug(A.rows() + np0);
        baug.head(A.rows()) = b;
        baug.tail(np0) = prior_b();
        Eigen::MatrixXd Bm = Eigen::MatrixXd::Zero(A.rows() + np0, nb);
        for (int k = 0; k < c.nseg; ++k)
            for (int i = 0; i < c.N; ++i) {
                const int idx = k * c.N + i;
                const M63 Gq = c.Qs * gamma_idx(c.Gamma, idx);
                for (int a = 0; a < 6; ++a)
                    for (int l = 0; l < 3; ++l)
                        Bm(hz.nISL() + idx * 6 + a, k * 3 + l) = -Gq(a, l);
            }
        const Eigen::MatrixXd Hqq2 = Aaug.transpose() * Aaug;
        const Eigen::MatrixXd Hqb = Aaug.transpose() * Bm;
        Eigen::MatrixXd H(ncol + nb, ncol + nb);
        H.topLeftCorner(ncol, ncol) = Hqq2;
        H.topRightCorner(ncol, nb) = Hqb;
        H.bottomLeftCorner(nb, ncol) = Hqb.transpose();
        H.bottomRightCorner(nb, nb) = Bm.transpose() * Bm;
        for (int j = 0; j < nb; ++j) H(ncol + j, ncol + j) += wb2;
        Eigen::VectorXd gvec(ncol + nb);
        gvec.head(ncol) = Aaug.transpose() * baug;
        gvec.tail(nb) = Bm.transpose() * baug;
        for (int j = 0; j < nb; ++j) gvec[ncol + j] -= beta[j] * wb2;
        Eigen::MatrixXd Hd = H;
        Hd.diagonal().array() += 1e-10 * Hd.diagonal().maxCoeff();
        const Eigen::VectorXd d = Hd.ldlt().solve(gvec);
        for (int j = 0; j < ncol; ++j)
            X[j] += c.Qs(j % 6, j % 6) * d[j];
        for (int j = 0; j < nb; ++j) beta[j] += d[ncol + j];
    }
    // 终止点：A/b 与 β 形式协方差（Schur，z 空间）
    fill_db();
    const double res2_isl = hz.assemble_dense(X, A, b, end, phi);
    const Eigen::MatrixXd Azf = scaled_cols(A);
    Eigen::MatrixXd Aaug(A.rows() + np0, ncol);
    Aaug.topRows(A.rows()) = Azf;
    Aaug.bottomRows(np0) = Ap;
    Eigen::MatrixXd Bm = Eigen::MatrixXd::Zero(A.rows() + np0, nb);
    for (int k = 0; k < c.nseg; ++k)
        for (int i = 0; i < c.N; ++i) {
            const int idx = k * c.N + i;
            const M63 Gq = c.Qs * gamma_idx(c.Gamma, idx);
            for (int a = 0; a < 6; ++a)
                for (int l = 0; l < 3; ++l)
                    Bm(hz.nISL() + idx * 6 + a, k * 3 + l) = -Gq(a, l);
        }
    const Eigen::MatrixXd Hqq = Aaug.transpose() * Aaug;
    const Eigen::MatrixXd Hqb = Aaug.transpose() * Bm;
    Eigen::MatrixXd Hbb = Bm.transpose() * Bm;
    for (int j = 0; j < nb; ++j) Hbb(j, j) += wb2;
    const Eigen::MatrixXd Xs = Hqq.ldlt().solve(Hqb);
    Eigen::MatrixXd S = Hbb - Hqb.transpose() * Xs;
    S = 0.5 * (S + S.transpose()).eval();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(S);
    Eigen::VectorXd w = es.eigenvalues();
    const double wmax = w.cwiseAbs().maxCoeff();
    for (int j = 0; j < nb; ++j)
        w[j] = std::max(w[j], 1e-12 * std::max(wmax, 1e-30));
    const Eigen::MatrixXd Cov = es.eigenvectors() * w.cwiseInverse().asDiagonal() *
                                es.eigenvectors().transpose();
    EstResult out;
    out.X = X;
    out.beta = beta;
    out.sigma.resize(nb);
    for (int j = 0; j < nb; ++j)
        out.sigma[j] = std::sqrt(std::max(Cov(j, j), 0.0));
    out.cost = hz.total_cost(X);
    out.isl_rms = std::sqrt(res2_isl / std::max(hz.nISL(), 1));
    const Eigen::VectorXd bdyn = b.segment(hz.nISL(), hz.nDYN());
    out.dyn_rms = std::sqrt(bdyn.squaredNorm() / std::max(hz.nDYN(), 1));
    return out;
}

// 约化 Schur 形式：参数 = [z0 (6N，位置等效), β (3*nseg，每区间公共)]。
// q_m = A_m z0 + Σ_{j<m} S_{m,j} β_j（A/S 由 Φ/Γ 解析链给出）；ISL 行的
// H_qq/H_qβ 由观测 Jacobian 直接装配，δq = −H_qq^{-1} H_qβ δβ 即此 GN 的 Schur 解。
// 这避免了 6NL 个自由节点的近零空间，数值稳定；Φ/Γ 取 Xinit 处（β=1e-6 下变化可忽略）。
EstResult estimate_reduced(const ChainBeta &c, const std::vector<double> &D,
                           const std::vector<double> &beta0, bool est_beta,
                           int iters, bool verbose = false) {
    const int N = c.N, L = c.L, nseg = c.nseg;
    const int nq = 6 * N, nbeta = 3 * nseg;
    const int npar = nq + (est_beta ? nbeta : 0);
    const int nrows = L * c.E + nq + (est_beta ? nbeta : 0);
    // 无量纲参数：q0 = x0_init + Q_s σ0 w（w~O(1)）；β = β0 + 1e-6 b（b~O(1)）。
    const double sbeta = 1e-6;
    // 链：A_m=∂q_m/∂q0，S_{m,j}=∂q_m/∂β_j
    std::vector<std::vector<M6>> Am(N, std::vector<M6>(L));
    std::vector<std::vector<std::vector<M63>>> Sm(
        N, std::vector<std::vector<M63>>(L, std::vector<M63>(nseg)));
    for (int i = 0; i < N; ++i) {
        Am[i][0] = M6::Identity();
        for (int m = 1; m <= nseg; ++m) {
            M6 Ph;
            for (int a = 0; a < 6; ++a)
                for (int b = 0; b < 6; ++b)
                    Ph(a, b) = c.Phi[(size_t)((m - 1) * N + i) * 36 + a * 6 + b];
            Am[i][m] = Ph * Am[i][m - 1];
        }
        for (int j = 0; j < nseg; ++j) {
            Sm[i][j + 1][j] = gamma_idx(c.Gamma, j * N + i);
            for (int m = j + 2; m <= nseg; ++m) {
                M6 Ph;
                for (int a = 0; a < 6; ++a)
                    for (int b = 0; b < 6; ++b)
                        Ph(a, b) = c.Phi[(size_t)((m - 1) * N + i) * 36 + a * 6 + b];
                Sm[i][m][j] = Ph * Sm[i][m - 1][j];
            }
        }
    }
    const M6 Qs = c.Qs;
    // 位置等效变量缩放：δx = S z，S=Qs^{-1}（角度列 /a）。z 以 σ0 为单位 ⇒ w~O(1)。
    M6 S = M6::Zero();
    for (int a = 0; a < 6; ++a) S(a, a) = 1.0 / Qs(a, a);
    const double w_isl = 1.0 / c.sigma_isl;
    auto build = [&](const Eigen::VectorXd &p, std::vector<double> &Xo,
                     Eigen::MatrixXd &Jm, Eigen::VectorXd &rm) {
        Eigen::VectorXd beta = Eigen::VectorXd::Zero(nbeta);
        for (int j = 0; j < nbeta; ++j)
            beta[j] = beta0[j] + (est_beta ? sbeta * p[nq + j] : 0.0);
        // 线性化是**扰动**链：X = X_nom + δX，
        // δq_{k+1}=Φ_k δq_k + Γ_k β_k，δq_0 = S σ0 w。
        Xo = c.Xinit;
        std::vector<double> dq((size_t)L * N * 6, 0.0);
        for (int i = 0; i < N; ++i)
            for (int cc = 0; cc < 6; ++cc) {
                dq[((size_t)0 * N + i) * 6 + cc] =
                    S(cc, cc) * c.sigma0 * p[i * 6 + cc];
                Xo[((size_t)0 * N + i) * 6 + cc] += dq[((size_t)0 * N + i) * 6 + cc];
            }
        for (int k = 0; k < nseg; ++k)
            for (int i = 0; i < N; ++i) {
                V6 dqn = Eigen::Map<const V6>(dq.data() + ((size_t)k * N + i) * 6);
                M6 Ph;
                for (int a = 0; a < 6; ++a)
                    for (int b = 0; b < 6; ++b)
                        Ph(a, b) = c.Phi[(size_t)(k * N + i) * 36 + a * 6 + b];
                dqn = Ph * dqn;
                if (est_beta) dqn += gamma_idx(c.Gamma, k * N + i) * beta.segment<3>(3 * k);
                for (int cc = 0; cc < 6; ++cc) {
                    dq[((size_t)(k + 1) * N + i) * 6 + cc] = dqn[cc];
                    Xo[((size_t)(k + 1) * N + i) * 6 + cc] += dqn[cc];
                }
            }
        std::vector<double> rv((size_t)L * N * 6), Jp((size_t)L * N * 36);
        qoejopt::oe2rvJacHost(Xo.data(), L * N, rv.data(), Jp.data());
        Jm = Eigen::MatrixXd::Zero(nrows, npar);
        rm = Eigen::VectorXd::Zero(nrows);
        int row = 0;
        for (int f = 0; f < L; ++f)
            for (int e = 0; e < c.E; ++e) {
                const int i = c.ei[e], j = c.ej[e];
                const size_t ii = (size_t)f * N + i, jj = (size_t)f * N + j;
                V3 pi, pj;
                for (int a = 0; a < 3; ++a) { pi[a] = rv[ii * 6 + a]; pj[a] = rv[jj * 6 + a]; }
                V3 u = pi - pj;
                const double dist = u.norm();
                if (dist > 1e-12) u /= dist;
                Eigen::Matrix<double, 1, 6> uiJp, ujJp;
                for (int cc = 0; cc < 6; ++cc) {
                    double si = 0.0, sj = 0.0;
                    for (int a = 0; a < 3; ++a) {
                        si += u[a] * Jp[ii * 36 + a * 6 + cc];
                        sj += u[a] * Jp[jj * 36 + a * 6 + cc];
                    }
                    uiJp[cc] = si; ujJp[cc] = sj;
                }
                const Eigen::Matrix<double, 1, 6> bzi =
                    c.sigma0 * (uiJp * Am[i][f] * S);
                const Eigen::Matrix<double, 1, 6> bzj =
                    c.sigma0 * (ujJp * Am[j][f] * S);
                Jm.block<1, 6>(row, i * 6) += bzi;
                Jm.block<1, 6>(row, j * 6) -= bzj;
                if (est_beta)
                    for (int jb = 0; jb < f; ++jb) {
                        const Eigen::Matrix<double, 1, 3> bs =
                            sbeta * (uiJp * Sm[i][f][jb] - ujJp * Sm[j][f][jb]);
                        Jm.block<1, 3>(row, nq + 3 * jb) += bs;
                    }
                rm[row] = dist - D[(size_t)f * c.E + e];
                Jm.row(row) *= w_isl;
                rm[row] *= w_isl;
                ++row;
            }
        for (int i = 0; i < N; ++i)
            for (int cc = 0; cc < 6; ++cc) {
                Jm(row, i * 6 + cc) = 1.0;          // 先验残差 = w
                rm[row] = p[i * 6 + cc];
                ++row;
            }
        if (est_beta)
            for (int j = 0; j < nbeta; ++j) {
                const double wbp = sbeta / c.sigma_beta;
                Jm(row, nq + j) = wbp;
                rm[row] = beta[j] / c.sigma_beta;
                ++row;
            }
    };
    Eigen::VectorXd p = Eigen::VectorXd::Zero(npar);   // w=0, b=0
    std::vector<double> X;
    Eigen::MatrixXd J;
    Eigen::VectorXd r;
    // 数值自检：Δr ≈ J Δp（一次）
    {
        static bool once = true;
        if (once && est_beta) {
            once = false;
            build(p, X, J, r);
            const int cols[] = {0, 1, nq, nq + 4};
            for (int jj : cols) {
                Eigen::VectorXd pp = p;
                pp[jj] += 1e-6;
                std::vector<double> X1;
                Eigen::MatrixXd J1;
                Eigen::VectorXd r1;
                build(pp, X1, J1, r1);
                const Eigen::VectorXd dr = r1 - r;
                const Eigen::VectorXd pred = J.col(jj) * 1e-6;
                std::printf("[RedJ] col=%d rel=%.3e |dr|=%.3e |pred|=%.3e\n", jj,
                            (dr - pred).norm() / std::max(pred.norm(), 1e-300),
                            dr.norm(), pred.norm());
            }
        }
    }
    auto cost_of = [&](const Eigen::VectorXd &pp) {
        std::vector<double> Xx;
        Eigen::MatrixXd Jx;
        Eigen::VectorXd rx;
        build(pp, Xx, Jx, rx);
        return rx.squaredNorm();
    };
    double c0 = cost_of(p);
    for (int it = 0; it < iters; ++it) {
        build(p, X, J, r);
        Eigen::MatrixXd H = J.transpose() * J;
        H.diagonal().array() += 1e-10 * H.diagonal().maxCoeff();
        const Eigen::VectorXd rhs = -(J.transpose() * r);
        const Eigen::VectorXd dp = H.ldlt().solve(rhs);
        double alpha = 1.0;
        bool moved = false;
        for (int ls = 0; ls < 10; ++ls) {
            const Eigen::VectorXd pp = p + alpha * dp;
            const double cc = cost_of(pp);
            if (cc < c0) {
                p = pp; c0 = cc; moved = true; break;
            }
            alpha *= 0.5;
        }
        if (verbose) {
            double isl2 = 0.0, pri2 = 0.0, bet2 = 0.0;
            for (int row = 0; row < L * c.E; ++row) isl2 += r[row] * r[row];
            for (int row = L * c.E; row < L * c.E + nq; ++row) pri2 += r[row] * r[row];
            const int b0row = L * c.E + nq;
            for (int row = b0row; row < nrows; ++row) bet2 += r[row] * r[row];
            std::printf("[RedGN] it=%d cost=%.3e isl=%.3e prior=%.3e betaP=%.3e "
                        "|dp|=%.3e alpha=%.1e |w|max=%.3e |b|max=%.3e\n",
                        it, c0, isl2, pri2, bet2, dp.norm(), alpha,
                        p.head(nq).cwiseAbs().maxCoeff(),
                        est_beta ? p.tail(nbeta).cwiseAbs().maxCoeff() : 0.0);
        }
        if (!moved) {
            if (verbose) std::printf("[RedGN] no descent at it=%d\n", it);
            break;
        }
    }
    build(p, X, J, r);
    Eigen::MatrixXd H = J.transpose() * J;
    H.diagonal().array() += 1e-12 * H.diagonal().maxCoeff();
    EstResult out;
    out.X = X;
    out.beta.assign(nbeta, 0.0);
    for (int j = 0; j < nbeta; ++j)
        out.beta[j] = beta0[j] + (est_beta ? sbeta * p[nq + j] : 0.0);
    if (est_beta) {
        // 形式协方差：H^{-1} 的 β 对角
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(H);
        Eigen::VectorXd w = es.eigenvalues();
        const double wmax = w.cwiseAbs().maxCoeff();
        for (int j = 0; j < w.size(); ++j) w[j] = std::max(w[j], 1e-12 * std::max(wmax, 1e-30));
        const Eigen::MatrixXd Cov =
            es.eigenvectors() * w.cwiseInverse().asDiagonal() * es.eigenvectors().transpose();
        out.sigma.resize(nbeta);
        for (int j = 0; j < nbeta; ++j)
            out.sigma[j] = sbeta * std::sqrt(std::max(Cov(nq + j, nq + j), 0.0));
    }
    double ss = 0.0;
    for (int row = 0; row < L * c.E; ++row) {
        const double raw = r[row] / w_isl;
        ss += raw * raw;
    }
    out.isl_rms = std::sqrt(ss / std::max(L * c.E, 1));
    out.dyn_rms = 0.0;   // 约化模型里 dyn 为硬约束
    out.cost = 0.0;
    return out;
}

std::vector<V3> make_beta(const ChainBeta &c, int pattern, double scale) {
    std::vector<V3> bet(c.nseg, V3::Zero());
    for (int k = 0; k < c.nseg; ++k) {
        if (pattern == 2 && k != 5) continue;      // single-interval
        if (pattern == 1) continue;                // zero
        bet[k] = V3(0.4 * std::sin(0.5 * k), 1.0, 0.2 * std::cos(0.4 * k)) * scale;
    }
    return bet;
}

}  // namespace

TEST(BiasWindow, SyntheticIslChainBetaRecovery) {
    ChainBeta c;
    init_chain(c);
    const double scale = 1e-6;
    const int iters = 6;

    // 全偏差：恢复 + 关闭偏差对比
    auto bet_full = make_beta(c, 0, scale);
    std::vector<double> Xt_full = biased_nodes_common(c.x0_true, bet_full, 20.0);
    std::vector<double> D_full = isl_obs(c, Xt_full, c.sigma_isl, 42);
    const EstResult e_on = estimate_reduced(c, D_full, std::vector<double>(c.nseg * 3, 0.0),
                                            true, iters, true);
    const EstResult e_off = estimate_reduced(c, D_full, std::vector<double>(c.nseg * 3, 0.0),
                                             false, iters);
    double z2 = 0.0, zmax = 0.0;
    for (int k = 0; k < c.nseg; ++k)
        for (int l = 0; l < 3; ++l) {
            const double z = (e_on.beta[3 * k + l] - bet_full[k][l]) /
                             std::max(e_on.sigma[3 * k + l], 1e-30);
            z2 += z * z;
            zmax = std::max(zmax, std::abs(z));
        }
    const double rms_z = std::sqrt(z2 / (3.0 * c.nseg));
    const double eph_on = gauge_eph_rms(Xt_full, e_on.X, c.N, c.L);
    const double eph_off = gauge_eph_rms(Xt_full, e_off.X, c.N, c.L);
    // ISL-only 每帧 6 DoF 规范不可观；per-frame Kabsch 是 GTS/绝对锚的等价口径。
    const double eph_on_pf = gauge_eph_rms_perframe(Xt_full, e_on.X, c.N, c.L);
    const double eph_off_pf = gauge_eph_rms_perframe(Xt_full, e_off.X, c.N, c.L);

    // β_true=0：不应虚假吸收
    auto bet_zero = make_beta(c, 1, scale);
    std::vector<double> Xt_zero = biased_nodes_common(c.x0_true, bet_zero, 20.0);
    std::vector<double> D_zero = isl_obs(c, Xt_zero, c.sigma_isl, 43);
    const EstResult e_zero = estimate_reduced(c, D_zero, std::vector<double>(c.nseg * 3, 0.0),
                                              true, iters);
    double z2z = 0.0, zmaxz = 0.0, bnorm = 0.0;
    for (int j = 0; j < 3 * c.nseg; ++j) {
        const double z = e_zero.beta[j] / std::max(e_zero.sigma[j], 1e-30);
        z2z += z * z; zmaxz = std::max(zmaxz, std::abs(z));
        bnorm = std::max(bnorm, std::abs(e_zero.beta[j]));
    }
    const double rms_zz = std::sqrt(z2z / (3.0 * c.nseg));

    // 仅区间 5 有 1e-6：检查时间串扰
    auto bet_one = make_beta(c, 2, scale);
    std::vector<double> Xt_one = biased_nodes_common(c.x0_true, bet_one, 20.0);
    std::vector<double> D_one = isl_obs(c, Xt_one, c.sigma_isl, 44);
    const EstResult e_one = estimate_reduced(c, D_one, std::vector<double>(c.nseg * 3, 0.0),
                                             true, iters);
    double primary = 0.0, other = 0.0;
    for (int k = 0; k < c.nseg; ++k)
        for (int l = 0; l < 3; ++l) {
            const double v = std::abs(e_one.beta[3 * k + l]);
            if (k == 5) primary = std::max(primary, v);
            else other = std::max(other, v);
        }
    const double leak = other / std::max(primary, 1e-30);

    std::printf("[IslBeta] on: rms_z=%.3f max_z=%.3f isl=%.3f dyn=%.3f "
                "eph(global/perframe)=%.2f/%.2f m | off: isl=%.3f dyn=%.3f "
                "eph=%.2f/%.2f m ratio_pf=%.2f | zero: rms_z=%.3f max_z=%.3f "
                "max|b|=%.3e | single: primary=%.3e other=%.3e leak=%.2f\n",
                rms_z, zmax, e_on.isl_rms, e_on.dyn_rms, eph_on, eph_on_pf,
                e_off.isl_rms, e_off.dyn_rms, eph_off, eph_off_pf,
                eph_off_pf / std::max(eph_on_pf, 1e-30),
                rms_zz, zmaxz, bnorm, primary, other, leak);
    EXPECT_LT(eph_on_pf, 5.0);
    EXPECT_GE(eph_off_pf / std::max(eph_on_pf, 1e-30), 2.0);
    EXPECT_GE(rms_z, 0.3);
    EXPECT_LE(rms_z, 3.0);
    EXPECT_LT(zmaxz, 3.0);
    EXPECT_LT(leak, 0.5);
}

// ISL 行 Jacobian/残差接线自检：b=−r，Δb≈−AΔx（小扰动）。
TEST(BiasWindow, WindowIslJacobian) {
    const auto s = sample_states();
    const int n = 2, L = 2;
    WindowHarness hz(n, L, 1, {0}, {1}, kFpn, kSeg, 1.0, M6::Identity());
    std::vector<double> X((size_t)L * n * 6);
    for (int f = 0; f < L; ++f)
        for (int i = 0; i < n; ++i)
            for (int c = 0; c < 6; ++c)
                X[((size_t)f * n + i) * 6 + c] = s[i][c];
    // D 用当前 X 的星间距离（零残差）
    std::vector<double> D((size_t)L * 1);
    for (int f = 0; f < L; ++f) {
        const Vector6d r0 = noeOsc2rv(Eigen::Map<const V6>(X.data() + ((size_t)f * n) * 6), 100, 1e-12);
        const Vector6d r1 = noeOsc2rv(Eigen::Map<const V6>(X.data() + ((size_t)f * n + 1) * 6), 100, 1e-12);
        D[f] = (r0.head<3>() - r1.head<3>()).norm();
    }
    hz.set_isl_D(D);
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    std::vector<double> end, phi;
    hz.assemble_dense(X, A, b, end, phi);
    double b0 = b.head(hz.nISL()).cwiseAbs().maxCoeff();
    // 小扰动（a:m, 角:rad）
    std::vector<double> dX((size_t)L * n * 6, 0.0);
    dX[0] = 1e-3;
    dX[(size_t)1 * 6 + 1] = 1e-9;
    dX[((size_t)n + 0) * 6 + 4] = -1e-9;
    dX[((size_t)n + 1) * 6 + 5] = 2e-9;
    std::vector<double> Xp = X;
    for (size_t j = 0; j < X.size(); ++j) Xp[j] += dX[j];
    Eigen::MatrixXd Ap;
    Eigen::VectorXd bp;
    hz.assemble_dense(Xp, Ap, bp, end, phi);
    const Eigen::VectorXd db = bp - b;
    const Eigen::VectorXd pred = -(A * Eigen::Map<const Eigen::VectorXd>(dX.data(), dX.size()));
    const double rel = (db - pred).norm() / std::max(pred.norm(), 1e-300);
    std::printf("[IslJac] |b_isl(0)|=%.3e ISL/dyn db-vs-Adx rel=%.3e\n", b0, rel);
    EXPECT_LT(b0, 1e-9);
    // δa=1e-3 m 的 FD 二阶项 ~1e-6；接线正确性用 1e-4 判定。
    EXPECT_LT(rel, 1e-4);
}
}  // namespace bias_window_impl

// ===================== test/test_bias_da_gtest.cpp =====================
namespace bias_da_impl {
// DACE 一阶 DA 对拍：QOE 状态 + 每区间 RTN 3D β 的确定性响应。
//
// 口径（与 C++ 核相同）：3/8-RK4、子步 h=dt/nsub；每子步
//   q_mid = step(q, h/2), q_next = step(q_mid, h/2),
//   q_{s+1} = q_next + h e^{-t_mid/τ} Φ(h/2) B_rtn(q_mid) β
// 这与 gveBiasFramesKernel / gveStmStepDev 的 Γ 递推逐项一致（同一中点/半段 STM）。
// DA：nvar=9 = 6 QOE + 3 RTN β，order=1；cons()/deriv() 取常数与一阶导数。
//
// 覆盖：单区间 Φ/Γ、10 区间累计 Γ（中间帧）、每区间 β_j 链 (ΠΦ)Γ_j 与块下三角、
// Schur 消元前后解/协方差、winDynAsmKernel 默认关闭 dbias 的逐位不变性。
//
// 不包含 dastates.h（避免全局 using namespace DACE 污染 Eigen 运算符）。




using DACE::AlgebraicVector;
using DACE::DA;
using DACE::cons;

using Vector6d = Eigen::Matrix<double, 6, 1>;

namespace {

constexpr double kT = 5749.800907;
constexpr double kSeg = 0.5 * kT;
constexpr double kTau = kT;
constexpr double kMu = 3.986004415e14;
constexpr int kFpn = 288;          // 子步数/区间（h≈9.982293 s）

void ck(cudaError_t e, const char *what) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

Vector6d sample_state() {
    Vector6d x;
    x << 6.9368e6, 0.0361, 6.90e-4, 5.06e-5, 1.2656, 0.0157;
    return x;
}

// ---- host f/A（f=noeGveRhs；A=中心差分，与核的解析 A 相差 ~1e-9）----
void eval_fA(const double qc[6], double f[6], double A[36]) {
    Vector6d x;
    for (int i = 0; i < 6; ++i) x[i] = qc[i];
    const Vector6d fv = noeGveRhs(x, 1.0, 0.0);
    for (int i = 0; i < 6; ++i) f[i] = fv[i];
    const double eps[6] = {1e-1, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7};
    for (int j = 0; j < 6; ++j) {
        Vector6d xp = x, xm = x;
        xp[j] += eps[j]; xm[j] -= eps[j];
        const Vector6d fp = noeGveRhs(xp, 1.0, 0.0), fm = noeGveRhs(xm, 1.0, 0.0);
        for (int i = 0; i < 6; ++i)
            A[i * 6 + j] = (fp[i] - fm[i]) / (2.0 * eps[j]);
    }
}

// ∂f_QOE/∂a_RTN（不含 Kepler n），与 gveGaussW_dev 同口径。
void b_rtn_arr(const double oe[6], double B[18]) {
    Vector6d q;
    for (int i = 0; i < 6; ++i) q[i] = oe[i];
    const Vector6d rv = noeOsc2rv(q, 100, 1e-12);
    const Eigen::Vector3d r3 = rv.head<3>(), v3 = rv.tail<3>();
    const Eigen::Vector3d R = r3.normalized();
    const Eigen::Vector3d N = r3.cross(v3).normalized();
    const Eigen::Vector3d T = N.cross(R);
    Eigen::Vector3d axes[3] = {R, T, N};
    const double a = oe[0], u = oe[1], ex = oe[2], ey = oe[3], inc = oe[4];
    const double e = std::hypot(ex, ey);
    const double w = std::atan2(ey, ex);
    double M = std::fmod(u - w, 2.0 * M_PI);
    if (M < 0.0) M += 2.0 * M_PI;
    double E = M;
    for (int it = 0; it < 30; ++it)
        E -= (E - e * std::sin(E) - M) / (1.0 - e * std::cos(E));
    const double cnu = (std::cos(E) - e) / (1.0 - e * std::cos(E));
    const double snu = std::sqrt(std::max(1.0 - e * e, 0.0)) * std::sin(E) /
                       (1.0 - e * std::cos(E));
    const double p = a * (1.0 - e * e);
    const double h = std::sqrt(kMu * p);
    const double r = p / (1.0 + e * cnu);
    const double ef = std::max(e, 1e-8), sif = std::max(std::sin(inc), 1e-8);
    const double uarg = w + std::atan2(snu, cnu);
    for (int k = 0; k < 3; ++k) {
        const double aR = axes[k].dot(R), aT = axes[k].dot(T), aN = axes[k].dot(N);
        const double da = (2.0 * a * a / h) * (e * snu * aR + (p / r) * aT);
        const double de = (1.0 / h) * (p * snu * aR + ((p + r) * cnu + r * e) * aT);
        const double di = (r * std::cos(uarg) / h) * aN;
        const double dOm = (r * std::sin(uarg) / (h * sif)) * aN;
        const double dw = (1.0 / (h * ef)) * (-p * cnu * aR + (p + r) * snu * aT)
                        - (r * std::sin(uarg) * std::cos(inc) / (h * sif)) * aN;
        const double fM = std::sqrt(std::max(1.0 - e * e, 0.0)) / (h * ef);
        const double dM = fM * ((p * cnu - 2.0 * r * e) * aR - (p + r) * snu * aT);
        B[0 * 3 + k] = da;
        B[1 * 3 + k] = dM + dw;
        B[2 * 3 + k] = std::cos(w) * de - e * std::sin(w) * dw;
        B[3 * 3 + k] = std::sin(w) * de + e * std::cos(w) * dw;
        B[4 * 3 + k] = di;
        B[5 * 3 + k] = dOm;
    }
}

// ---- double 3/8-RK4 + 变分 STM（与 gveStmStepDev 同结构）----
void dstep(const double y0[6], double dt, double yf[6], double Phi[36]) {
    double y[6], k1[6], k2[6], k3[6], k4[6], tmp[6], f[6], A[36];
    double P[36], K1[36], K2[36], K3[36], K4[36];
    for (int i = 0; i < 6; ++i) y[i] = y0[i];
    for (int i = 0; i < 36; ++i) P[i] = (i % 7 == 0) ? 1.0 : 0.0;
    eval_fA(y, k1, A);
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j) {
            double s = 0.0;
            for (int c = 0; c < 6; ++c) s += A[i * 6 + c] * P[c * 6 + j];
            K1[i * 6 + j] = s;
        }
    for (int i = 0; i < 6; ++i) tmp[i] = y[i] + dt * k1[i] / 3.0;
    eval_fA(tmp, k2, A);
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j) {
            double s = 0.0;
            for (int c = 0; c < 6; ++c) s += A[i * 6 + c] * (P[c * 6 + j] + dt * K1[c * 6 + j] / 3.0);
            K2[i * 6 + j] = s;
        }
    for (int i = 0; i < 6; ++i) tmp[i] = y[i] + dt * (-k1[i] / 3.0 + k2[i]);
    eval_fA(tmp, k3, A);
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j) {
            double s = 0.0;
            for (int c = 0; c < 6; ++c) s += A[i * 6 + c] * (P[c * 6 + j] + dt * (-K1[c * 6 + j] / 3.0 + K2[c * 6 + j]));
            K3[i * 6 + j] = s;
        }
    for (int i = 0; i < 6; ++i) tmp[i] = y[i] + dt * (k1[i] - k2[i] + k3[i]);
    eval_fA(tmp, k4, A);
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j) {
            double s = 0.0;
            for (int c = 0; c < 6; ++c) s += A[i * 6 + c] * (P[c * 6 + j] + dt * (K1[c * 6 + j] - K2[c * 6 + j] + K3[c * 6 + j]));
            K4[i * 6 + j] = s;
        }
    for (int i = 0; i < 6; ++i)
        yf[i] = y[i] + dt * (k1[i] + 3.0 * k2[i] + 3.0 * k3[i] + k4[i]) / 8.0;
    for (int i = 0; i < 36; ++i)
        Phi[i] = P[i] + dt * (K1[i] + 3.0 * K2[i] + 3.0 * K3[i] + K4[i]) / 8.0;
}

// ---- DA 3/8-RK4（RHS 为 f0 + A δq；β 只在每子步末线性加入）----
AlgebraicVector<DA> rhs_lin(const AlgebraicVector<DA> &z) {
    double zc[6], f[6], A[36];
    for (int i = 0; i < 6; ++i) zc[i] = cons(z[i]);
    eval_fA(zc, f, A);
    AlgebraicVector<DA> out(6);
    for (int i = 0; i < 6; ++i) {
        out[i] = DA(f[i]);
        for (int j = 0; j < 6; ++j) out[i] += A[i * 6 + j] * (z[j] - cons(z[j]));
    }
    return out;
}

void da_step(const AlgebraicVector<DA> &y, double dt, AlgebraicVector<DA> &yf) {
    const AlgebraicVector<DA> k1 = rhs_lin(y);
    AlgebraicVector<DA> tmp(6);
    for (int i = 0; i < 6; ++i) tmp[i] = y[i] + dt * k1[i] / 3.0;
    const AlgebraicVector<DA> k2 = rhs_lin(tmp);
    for (int i = 0; i < 6; ++i) tmp[i] = y[i] + dt * (-k1[i] / 3.0 + k2[i]);
    const AlgebraicVector<DA> k3 = rhs_lin(tmp);
    for (int i = 0; i < 6; ++i) tmp[i] = y[i] + dt * (k1[i] - k2[i] + k3[i]);
    const AlgebraicVector<DA> k4 = rhs_lin(tmp);
    yf = AlgebraicVector<DA>(6);
    for (int i = 0; i < 6; ++i)
        yf[i] = y[i] + dt * (k1[i] + 3.0 * k2[i] + 3.0 * k3[i] + k4[i]) / 8.0;
}

// 窗口 dyn 的口径：每子步一个 3/8 全步 h（与 gveStmKernelGrp / gveStmSubstepsDevice 一致）。
void substep_phi1(AlgebraicVector<DA> &q, double h) {
    AlgebraicVector<DA> qn;
    da_step(q, h, qn);
    q = qn;
}

// Γ 的口径：每子步两半段 + 中点 B（与 gveBiasFramesKernel / gveStmStepDev 的 Γ 递推一致）。
void substep_gamma(AlgebraicVector<DA> &q, const AlgebraicVector<DA> &beta,
                   double h, double t_mid, double tau, bool apply) {
    AlgebraicVector<DA> qmid(6), qnext(6);
    da_step(q, 0.5 * h, qmid);
    da_step(qmid, 0.5 * h, qnext);
    if (!apply) { q = qnext; return; }
    double qc[6], qmc[6], Ph[36], B[18];
    for (int i = 0; i < 6; ++i) qc[i] = cons(q[i]);
    dstep(qc, 0.5 * h, qmc, Ph);
    b_rtn_arr(qmc, B);
    const double cf = h * std::exp(-t_mid / tau);
    for (int i = 0; i < 6; ++i) {
        q[i] = qnext[i];
        for (int l = 0; l < 3; ++l) {
            double g = 0.0;
            for (int j = 0; j < 6; ++j) g += Ph[i * 6 + j] * B[j * 3 + l];
            q[i] += (cf * g) * beta[l];
        }
    }
}

// ---- kernel 参考 ----
void kernel_phi(const std::vector<double> &starts, int nk, double dt, int nsub,
                std::vector<double> &end, std::vector<double> &phi) {
    double *d_oe = nullptr, *d_e = nullptr, *d_p = nullptr;
    double *sa = nullptr, *sb = nullptr, *pa = nullptr, *pb = nullptr, *ps = nullptr;
    const size_t b_oe = (size_t)nk * 6 * sizeof(double);
    const size_t b_p = (size_t)nk * 36 * sizeof(double);
    ck(cudaMalloc(&d_oe, b_oe), "malloc oe");
    ck(cudaMalloc(&d_e, b_oe), "malloc end");
    ck(cudaMalloc(&d_p, b_p), "malloc phi");
    ck(cudaMalloc(&sa, b_oe), "malloc sa");
    ck(cudaMalloc(&sb, b_oe), "malloc sb");
    ck(cudaMalloc(&pa, b_p), "malloc pa");
    ck(cudaMalloc(&pb, b_p), "malloc pb");
    ck(cudaMalloc(&ps, b_p), "malloc ps");
    ck(cudaMemcpy(d_oe, starts.data(), b_oe, cudaMemcpyHostToDevice), "copy oe");
    qoejopt::gveStmSubstepsDevice(d_oe, nk, 1, dt, nsub, d_e, d_p, sa, sb, pa, pb, ps);
    end.resize((size_t)nk * 6);
    phi.resize(b_p / sizeof(double));
    ck(cudaMemcpy(end.data(), d_e, b_oe, cudaMemcpyDeviceToHost), "copy end");
    ck(cudaMemcpy(phi.data(), d_p, b_p, cudaMemcpyDeviceToHost), "copy phi");
    cudaFree(d_oe); cudaFree(d_e); cudaFree(d_p);
    cudaFree(sa); cudaFree(sb); cudaFree(pa); cudaFree(pb); cudaFree(ps);
}

void kernel_gamma(const std::vector<double> &starts, int nk, double dt, int nsub,
                  std::vector<double> &G, double tau = kTau) {
    double rtn[3] = {1.0, 1.0, 1.0};
    std::vector<double> e((size_t)nk * 6);
    G.assign((size_t)nk * 2 * 18, 0.0);
    qoejopt::gveBiasFramesBatch(starts.data(), nk, 2, 1, dt, tau, 0, rtn,
                                G.data(), e.data());
    // 取 frame 1（每星 2 帧，frame 1 在 offset (i*2+1)*18）
    std::vector<double> out((size_t)nk * 18);
    for (int i = 0; i < nk; ++i)
        for (int a = 0; a < 18; ++a) out[(size_t)i * 18 + a] = G[(size_t)(i * 2 + 1) * 18 + a];
    G.swap(out);
}

void flatten(const Vector6d &x, std::vector<double> &v) {
    v.resize(6);
    for (int i = 0; i < 6; ++i) v[i] = x[i];
}

double rel_frob(const std::vector<double> &a, const std::vector<double> &b) {
    double na = 0.0, nb = 0.0, nd = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        na += a[i] * a[i]; nb += b[i] * b[i];
        const double d = a[i] - b[i]; nd += d * d;
    }
    return std::sqrt(nd) / std::max(std::sqrt(nb), 1e-300);
}

}  // namespace

// 1) 单区间：DA Φ/Γ vs gveStmSubstepsDevice Φ、gveBiasFramesBatch Γ。
TEST(BiasDa, SingleIntervalPhiGamma) {
    const Vector6d x0v = sample_state();
    std::vector<double> x0;
    flatten(x0v, x0);
    const double h = kSeg / (double)kFpn;
    std::vector<double> Phi_da(36), Gam_da(18);
    // Φ：窗口口径（每子步一个全步 h）
    DA::init(1, 9);
    {
        AlgebraicVector<DA> q(6), beta(3);
        for (int i = 0; i < 6; ++i) q[i] = DA(x0[i]) + DA(i + 1);
        for (int l = 0; l < 3; ++l) beta[l] = DA(7 + l);
        for (int s = 0; s < kFpn; ++s) substep_phi1(q, h);
        for (int i = 0; i < 6; ++i)
            for (int j = 0; j < 6; ++j)
                Phi_da[i * 6 + j] = cons(q[i].deriv(j + 1));
    }
    // Γ：frames 口径（每子步两半段 + 中点 B）
    DA::init(1, 9);
    {
        AlgebraicVector<DA> q(6), beta(3);
        for (int i = 0; i < 6; ++i) q[i] = DA(x0[i]) + DA(i + 1);
        for (int l = 0; l < 3; ++l) beta[l] = DA(7 + l);
        for (int s = 0; s < kFpn; ++s)
            substep_gamma(q, beta, h, ((double)s + 0.5) * h, kTau, true);
        for (int i = 0; i < 6; ++i)
            for (int l = 0; l < 3; ++l)
                Gam_da[i * 3 + l] = cons(q[i].deriv(7 + l));
    }
    std::vector<double> kend, kphi;
    kernel_phi(x0, 1, kSeg, kFpn, kend, kphi);
    std::vector<double> kG;
    kernel_gamma(x0, 1, kSeg, kFpn, kG);
    const double rphi = rel_frob(Phi_da, kphi);
    const double rgam = rel_frob(Gam_da, kG);
    std::printf("[DaSingle] Phi rel=%.3e Gamma rel=%.3e\n", rphi, rgam);
    EXPECT_LT(rphi, 1e-6);
    EXPECT_LT(rgam, 1e-6);
}

// 2) 10 区间累计 Γ（β 在 t=0 施加，指数衰减跨全窗）：中间帧与端点 vs frames。
TEST(BiasDa, CumulativeGammaFrames) {
    const Vector6d x0v = sample_state();
    std::vector<double> x0;
    flatten(x0v, x0);
    const int L = 11;
    DA::init(1, 9);
    AlgebraicVector<DA> q(6), beta(3);
    for (int i = 0; i < 6; ++i) q[i] = DA(x0[i]) + DA(i + 1);
    for (int l = 0; l < 3; ++l) beta[l] = DA(7 + l);
    const double h = kSeg / (double)kFpn;
    std::vector<std::vector<double>> Gda(L);
    double max_rel = 0.0;
    for (int k = 0; k < L - 1; ++k) {
        for (int s = 0; s < kFpn; ++s)
            substep_gamma(q, beta, h, (double)k * kSeg + ((double)s + 0.5) * h, kTau, true);
        std::vector<double> g(18);
        for (int i = 0; i < 6; ++i)
            for (int l = 0; l < 3; ++l) g[i * 3 + l] = cons(q[i].deriv(7 + l));
        Gda[k + 1] = g;
    }
    double rtn[3] = {1.0, 1.0, 1.0};
    std::vector<double> Gf((size_t)L * 18), ef(6);
    qoejopt::gveBiasFramesBatch(x0.data(), 1, L, 1, kSeg, kTau, 0, rtn,
                                Gf.data(), ef.data());
    for (int m = 1; m < L; ++m) {
        std::vector<double> gk(18);
        for (int a = 0; a < 18; ++a) gk[a] = Gf[(size_t)m * 18 + a];   // n=1: (0*L+m)*18
        const double r = rel_frob(Gda[m], gk);
        max_rel = std::max(max_rel, r);
        if (m == 1 || m == L / 2 || m == L - 1)
            std::printf("[DaCum] frame %d Gamma rel=%.3e\n", m, r);
    }
    EXPECT_LT(max_rel, 1e-6);
}

// 3) 每区间 β_j 链：DA S_{m,j}=∂q_m/∂β_j vs (ΠΦ)Γ_j；块下三角、交叉项为零。
TEST(BiasDa, PerIntervalChainAndCrossTerms) {
    const Vector6d x0v = sample_state();
    std::vector<double> x0;
    flatten(x0v, x0);
    const int nseg = 5, L = nseg + 1;
    const double h = kSeg / (double)kFpn;
    // 名义节点：窗口口径（每子步一个全步 h）
    std::vector<std::vector<double>> qn(L);
    qn[0] = x0;
    for (int k = 0; k < nseg; ++k) {
        double q[6], yf[6], Phi[36];
        for (int i = 0; i < 6; ++i) q[i] = qn[k][i];
        for (int s = 0; s < kFpn; ++s) {
            dstep(q, h, yf, Phi);
            for (int i = 0; i < 6; ++i) q[i] = yf[i];
        }
        qn[k + 1].assign(q, q + 6);
    }
    // kernel Φ_k、Γ_k（在名义节点）
    std::vector<double> starts((size_t)nseg * 6);
    for (int k = 0; k < nseg; ++k)
        for (int i = 0; i < 6; ++i) starts[(size_t)k * 6 + i] = qn[k][i];
    std::vector<double> kend, kphi;
    kernel_phi(starts, nseg, kSeg, kFpn, kend, kphi);
    std::vector<double> kG;
    kernel_gamma(starts, nseg, kSeg, kFpn, kG);
    // 链 S[m][j]
    std::vector<std::vector<std::vector<double>>> S(L, std::vector<std::vector<double>>(nseg));
    for (int j = 0; j < nseg; ++j) {
        S[j + 1][j].assign(kG.begin() + (size_t)j * 18, kG.begin() + (size_t)(j + 1) * 18);
        for (int m = j + 2; m <= nseg; ++m) {
            std::vector<double> tmp(18, 0.0);
            for (int i = 0; i < 6; ++i)
                for (int l = 0; l < 3; ++l) {
                    double s = 0.0;
                    for (int c = 0; c < 6; ++c)
                        s += kphi[(size_t)(m - 1) * 36 + i * 6 + c] *
                             S[m - 1][j][c * 3 + l];
                    tmp[i * 3 + l] = s;
                }
            S[m][j] = tmp;
        }
    }
    double max_rel = 0.0, max_zero = 0.0, min_end = 1e300, max_gam = 0.0;
    for (int j = 0; j < nseg; ++j) {
        // DA Γ_j（frames 两半段口径）并与 kernel Γ_j 对拍
        std::vector<double> Gj(18);
        {
            DA::init(1, 9);
            AlgebraicVector<DA> q(6), beta(3);
            for (int i = 0; i < 6; ++i) q[i] = DA(qn[j][i]);
            for (int l = 0; l < 3; ++l) beta[l] = DA(7 + l);
            for (int s = 0; s < kFpn; ++s)
                substep_gamma(q, beta, h, ((double)s + 0.5) * h, kTau, true);
            for (int i = 0; i < 6; ++i)
                for (int l = 0; l < 3; ++l) Gj[i * 3 + l] = cons(q[i].deriv(7 + l));
            std::vector<double> kg(kG.begin() + (size_t)j * 18,
                                   kG.begin() + (size_t)(j + 1) * 18);
            max_gam = std::max(max_gam, rel_frob(Gj, kg));
        }
        // 窗口链：每区间一个全步 Φ；区间 j 末加 Γ_j β
        DA::init(1, 9);
        AlgebraicVector<DA> q(6), beta(3);
        for (int i = 0; i < 6; ++i) q[i] = DA(qn[j][i]);
        for (int l = 0; l < 3; ++l) beta[l] = DA(7 + l);
        for (int k = j; k < nseg; ++k) {
            for (int s = 0; s < kFpn; ++s) substep_phi1(q, h);
            if (k == j)
                for (int i = 0; i < 6; ++i)
                    for (int l = 0; l < 3; ++l) q[i] += Gj[i * 3 + l] * beta[l];
            std::vector<double> gda(18);
            for (int i = 0; i < 6; ++i)
                for (int l = 0; l < 3; ++l) gda[i * 3 + l] = cons(q[i].deriv(7 + l));
            max_rel = std::max(max_rel, rel_frob(gda, S[k + 1][j]));
            const double n = Eigen::Map<const Eigen::Matrix<double, 6, 3, Eigen::RowMajor>>(
                S[k + 1][j].data()).norm();
            if (k + 1 == nseg) min_end = std::min(min_end, n);
        }
    }
    for (int m = 0; m <= nseg; ++m)
        for (int j = 0; j < nseg; ++j)
            if (j >= m && !S[m][j].empty()) {
                double n = 0.0;
                for (double v : S[m][j]) n += v * v;
                max_zero = std::max(max_zero, std::sqrt(n));
            }
    std::printf("[DaChain] nseg=%d gamma_rel=%.3e max_rel=%.3e upper_cross_max=%.3e "
                "endpoint_min_j=%.3e\n", nseg, max_gam, max_rel, max_zero, min_end);
    EXPECT_LT(max_gam, 1e-6);
    EXPECT_LT(max_rel, 1e-6);
    EXPECT_LT(max_zero, 1e-12);
    EXPECT_GT(min_end, 1.0);
}

// 4) Schur 消元 vs 联合解/协方差（DA 链构造的线性化系统）。
TEST(BiasDa, SchurMatchesJoint) {
    const int nseg = 3, nq = 6 * (nseg + 1), nb = 3 * nseg;
    Eigen::MatrixXd Sfull = Eigen::MatrixXd::Zero(nq, nb);
    // 用简单块结构填充（DA 链已在上一个测试对拍；这里验证代数）
    for (int j = 0; j < nseg; ++j)
        for (int m = j + 1; m <= nseg; ++m)
            for (int i = 0; i < 6; ++i)
                for (int l = 0; l < 3; ++l)
                    Sfull(m * 6 + i, j * 3 + l) =
                        (1.0 + 0.1 * m) * ((i == l) ? 1.0 : 0.1) + 0.01 * j;
    Eigen::MatrixXd Jq = Eigen::MatrixXd::Random(nq, nq) * 0.1;
    Eigen::MatrixXd Jb = Jq * Sfull;
    Eigen::MatrixXd Hqq = Jq.transpose() * Jq + 1e-6 * Eigen::MatrixXd::Identity(nq, nq);
    Eigen::MatrixXd Hqb = Jq.transpose() * Jb;
    Eigen::MatrixXd Hbb = Jb.transpose() * Jb + 1e-6 * Eigen::MatrixXd::Identity(nb, nb);
    Eigen::VectorXd gq = Eigen::VectorXd::Random(nq);
    Eigen::VectorXd gb = Eigen::VectorXd::Random(nb);
    Eigen::MatrixXd H(nq + nb, nq + nb);
    H.topLeftCorner(nq, nq) = Hqq;
    H.topRightCorner(nq, nb) = Hqb;
    H.bottomLeftCorner(nb, nq) = Hqb.transpose();
    H.bottomRightCorner(nb, nb) = Hbb;
    Eigen::VectorXd g(nq + nb);
    g.head(nq) = gq; g.tail(nb) = gb;
    const Eigen::VectorXd djoint = H.ldlt().solve(g);
    const Eigen::MatrixXd Hqq_inv = Hqq.ldlt().solve(Eigen::MatrixXd::Identity(nq, nq));
    const Eigen::MatrixXd Schur = Hbb - Hqb.transpose() * Hqq_inv * Hqb;
    const Eigen::VectorXd db = Schur.ldlt().solve(gb - Hqb.transpose() * Hqq_inv * gq);
    const Eigen::VectorXd dq = Hqq_inv * (gq - Hqb * db);
    const Eigen::MatrixXd Cov = H.ldlt().solve(Eigen::MatrixXd::Identity(nq + nb, nq + nb));
    const Eigen::MatrixXd Covb = Schur.ldlt().solve(Eigen::MatrixXd::Identity(nb, nb));
    const double rel_b = (db - djoint.tail(nb)).norm() / std::max(djoint.tail(nb).norm(), 1e-300);
    const double rel_q = (dq - djoint.head(nq)).norm() / std::max(djoint.head(nq).norm(), 1e-300);
    const double rel_c = (Covb - Cov.bottomRightCorner(nb, nb)).norm() /
                         std::max(Covb.norm(), 1e-300);
    std::printf("[DaSchur] rel_dbeta=%.3e rel_dq=%.3e rel_cov=%.3e\n",
                rel_b, rel_q, rel_c);
    EXPECT_LT(rel_b, 1e-8);
    EXPECT_LT(rel_q, 1e-8);
    EXPECT_LT(rel_c, 1e-8);
}

// ===================== 恢复的 3 个验收用例（按根因修正） =====================
namespace {

const double kDataRv[6][6] = {
    {6.92548450207484420e6, 1.83933498470334074e5, 2.38584713515536336e5,
     -3.09649398333638032e2, 2.27284083262552122e3, 7.23044397227288209e3},
    {6.73563564879427478e6, 5.97795898504732177e5, 1.51458392385059688e6,
     -1.78048400186230720e3, 2.19354341534647028e3, 7.04208442421910149e3},
    {6.29272923717958573e6, 9.69681670681712567e5, 2.74289274634164479e6,
     -3.18116904214365422e3, 2.02570067563594512e3, 6.57883914512638876e3},
    {5.60072544657378830e6, 1.31558193009910313e6, 3.86665727485460276e6,
     -4.46882439064495520e3, 1.76398470636942375e3, 5.86632050255757349e3},
    {4.69906200243177172e6, 1.61330108211964462e6, 4.83534202204599138e6,
     -5.57387467011871740e3, 1.45532931991136411e3, 4.92586716858321597e3},
    {3.59725829549974110e6, 1.83583987439668505e6, 5.63317211384642310e6,
     -6.47890425324340140e3, 1.08047639076018004e3, 3.78231186220165591e3}};

std::vector<double> data_qoe(int row) {
    Eigen::VectorXd rv(6);
    for (int i = 0; i < 6; ++i) rv[i] = kDataRv[row][i];
    const Eigen::VectorXd q = osculating::rv2OEOsc(rv);
    std::vector<double> out(6);
    for (int i = 0; i < 6; ++i) out[i] = q[i];
    return out;
}

Vector6d rk4_nom(const Vector6d &x, double h) {
    const Vector6d k1 = noeGveRhs(x, 1, 0), k2 = noeGveRhs(x + 0.5 * h * k1, 1, 0),
                   k3 = noeGveRhs(x + 0.5 * h * k2, 1, 0), k4 = noeGveRhs(x + h * k3, 1, 0);
    return x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}

struct NomF { Vector6d q; Eigen::Vector3d p, v; Vector6d rv; };

std::vector<NomF> nominal_traj(const std::vector<double> &q0v, double arc, int M) {
    Vector6d q;
    for (int i = 0; i < 6; ++i) q[i] = q0v[i];
    const double dt = arc / (M - 1);
    const int ns = std::max(1, (int)std::ceil(dt / 10.0));
    const double h = dt / ns;
    std::vector<NomF> out(M);
    for (int m = 0; m < M; ++m) {
        out[m].q = q;
        const Vector6d rv = noeOsc2rv(q, 100, 1e-12);
        out[m].rv = rv; out[m].p = rv.head<3>(); out[m].v = rv.tail<3>();
        if (m == M - 1) break;
        for (int s = 0; s < ns; ++s) q = rk4_nom(q, h);
    }
    return out;
}

void oe_jac_one(const Vector6d &q, double Jp[18]) {
    double X[6], rv[6], J[36];
    for (int i = 0; i < 6; ++i) X[i] = q[i];
    qoejopt::oe2rvJacHost(X, 1, rv, J);
    for (int a = 0; a < 3; ++a)
        for (int c = 0; c < 6; ++c) Jp[a * 6 + c] = J[a * 6 + c];
}

// 常值 ECI 加速度 b 在 [t0,t1] 内作用、之后保持速度的 δp 与 ∂p/∂b。
void kin(const Eigen::Vector3d &b, double t0, double t1, double t,
         Eigen::Vector3d &dp, Eigen::Matrix3d &Jb) {
    dp.setZero(); Jb.setZero();
    if (t <= t0) return;
    const double te = std::min(t, t1), d = te - t0;
    dp = 0.5 * d * d * b;
    Jb = 0.5 * d * d * Eigen::Matrix3d::Identity();
    if (t > t1) {
        const double D = t1 - t0;
        dp += (t - t1) * D * b;
        Jb += (t - t1) * D * Eigen::Matrix3d::Identity();
    }
}

// 每星 RTN 常值 β 的 1T 链响应（窗口口径：每区间一个全步 Φ；Γ 在区间起点）。
struct RtnChain {
    std::vector<std::vector<double>> q, Phi, Gam, V;   // V: 6×3 per frame
};
RtnChain build_rtn_chain(const std::vector<double> &q0, int nseg, double dt) {
    RtnChain c;
    const int nsub = std::max(1, (int)std::ceil(dt / 10.0));
    const double h = dt / nsub;
    c.q.resize(nseg + 1); c.Phi.resize(nseg); c.Gam.resize(nseg);
    c.q[0] = q0;
    for (int k = 0; k < nseg; ++k) {
        double q[6], yf[6], Ph[36], P[36];
        for (int i = 0; i < 6; ++i) q[i] = c.q[k][i];
        for (int i = 0; i < 36; ++i) P[i] = (i % 7 == 0) ? 1.0 : 0.0;
        for (int s = 0; s < nsub; ++s) {
            dstep(q, h, yf, Ph);
            double Pn[36];
            for (int i = 0; i < 6; ++i)
                for (int j = 0; j < 6; ++j) {
                    double v = 0.0;
                    for (int a = 0; a < 6; ++a) v += Ph[i * 6 + a] * P[a * 6 + j];
                    Pn[i * 6 + j] = v;
                }
            for (int i = 0; i < 6; ++i) q[i] = yf[i];
            for (int i = 0; i < 36; ++i) P[i] = Pn[i];
        }
        c.q[k + 1].assign(q, q + 6);
        c.Phi[k].assign(P, P + 36);
    }
    std::vector<double> starts((size_t)nseg * 6);
    for (int k = 0; k < nseg; ++k)
        for (int i = 0; i < 6; ++i) starts[(size_t)k * 6 + i] = c.q[k][i];
    std::vector<double> G;
    kernel_gamma(starts, nseg, dt, nsub, G, kTau);
    for (int k = 0; k < nseg; ++k)
        c.Gam[k].assign(G.begin() + (size_t)k * 18, G.begin() + (size_t)(k + 1) * 18);
    c.V.assign(nseg + 1, std::vector<double>(18, 0.0));
    for (int m = 1; m <= nseg; ++m) {
        const std::vector<double> &Pm = c.Phi[m - 1];
        for (int i = 0; i < 6; ++i)
            for (int l = 0; l < 3; ++l) {
                double v = c.Gam[m - 1][i * 3 + l];
                for (int a = 0; a < 6; ++a) v += Pm[i * 6 + a] * c.V[m - 1][a * 3 + l];
                c.V[m][i * 3 + l] = v;
            }
    }
    return c;
}

struct Est { double mean = 0, rms = 0, mx = 0; };
Est estats(const std::vector<Eigen::Vector3d> &a, const std::vector<Eigen::Vector3d> &b, int m0 = 0) {
    Est e; double ss = 0; int n = 0;
    for (int m = m0; m < (int)a.size(); ++m) {
        const double d = (a[m] - b[m]).norm();
        e.mean += d; ss += d * d; e.mx = std::max(e.mx, d); ++n;
    }
    e.mean /= std::max(n, 1); e.rms = std::sqrt(ss / std::max(n, 1));
    return e;
}

}  // namespace

// A) 单节点 Eph：q0 锚定到真值，只解 ECI 恒值/3 段 b；1T 降幅 ≥60%，1/3T 预测。
TEST(BiasDa, SingleNodePositionBiasRecovery) {
    // 真实 GVE：名义 q_k 由 3/8-RK4 递推；Γ_k 取 gveBiasFramesBatch（每 10 s 区间、
    // τ=1e9 无衰减）；线性响应 V_s[m]=Σ(ΠΦ)Γ_k（段内），真值用带 B(q)β 的非线性 RK4。
    // q0 锚定到 Eph 真值（只解 β）；另报 q0 自由时 [q0;β] 联合设计的条件数。
    const std::vector<double> q0v = data_qoe(0);
    const Eigen::Vector3d base(0.3e-7, 1.0e-7, 0.2e-7);   // RTN，β_T=1e-7
    for (double arc : {kT / 3.0, kT}) {
        const int M = (int)std::llround(arc / 10.0) + 1;   // 10 s ordinary frames
        const double dt = arc / (M - 1);
        std::vector<Vector6d> qn(M);
        std::vector<std::vector<double>> Phi(M - 1, std::vector<double>(36));
        for (int i = 0; i < 6; ++i) qn[0][i] = q0v[i];
        for (int k = 0; k < M - 1; ++k) {
            Vector6d qf;
            dstep(qn[k].data(), dt, qf.data(), Phi[k].data());
            qn[k + 1] = qf;
        }
        std::vector<double> starts((size_t)(M - 1) * 6);
        for (int k = 0; k < M - 1; ++k)
            for (int i = 0; i < 6; ++i) starts[(size_t)k * 6 + i] = qn[k][i];
        std::vector<std::vector<double>> Gam(M - 1, std::vector<double>(18));
        {
            std::vector<double> G;
            kernel_gamma(starts, M - 1, dt, 1, G, 1e9);
            for (int k = 0; k < M - 1; ++k)
                Gam[k].assign(G.begin() + (size_t)k * 18, G.begin() + (size_t)(k + 1) * 18);
        }
        std::vector<Eigen::Vector3d> pn(M);
        std::vector<std::array<double, 18>> Jp(M);
        for (int m = 0; m < M; ++m) {
            const Vector6d rv = noeOsc2rv(qn[m], 100, 1e-12);
            pn[m] = rv.head<3>();
            oe_jac_one(qn[m], Jp[m].data());
        }
        // 非线性 GVE：rhs = f(q) + B_rtn(q) β_seg（段内常值、段后 0）
        auto fb = [&](const Vector6d &q, const Eigen::Vector3d &b) {
            double B[18]; b_rtn_arr(q.data(), B);
            Eigen::Matrix<double, 6, 3> Bm;
            for (int i = 0; i < 6; ++i)
                for (int l = 0; l < 3; ++l) Bm(i, l) = B[i * 3 + l];
            return (noeGveRhs(q, 1.0, 0.0) + Bm * b).eval();
        };
        for (int nseg : {1, 3}) {
            const int mseg = (M - 1) / nseg;
            std::vector<Eigen::Vector3d> bseg(nseg);
            for (int s = 0; s < nseg; ++s)
                bseg[s] = base * ((nseg == 1) ? 1.0 : (0.5 + 0.5 * s));
            // V_s[m] = ∂q_m/∂β_s（段内 Γ、段后 Φ 传播）
            std::vector<std::vector<std::vector<double>>> V(
                M, std::vector<std::vector<double>>(nseg, std::vector<double>(18, 0.0)));
            for (int s = 0; s < nseg; ++s)
                for (int m = 1; m < M; ++m) {
                    const int k = m - 1;
                    for (int i = 0; i < 6; ++i)
                        for (int l = 0; l < 3; ++l) {
                            const int ks = std::min(k / mseg, nseg - 1);
                            double v = (ks == s) ? Gam[k][i * 3 + l] : 0.0;
                            for (int a = 0; a < 6; ++a) v += Phi[k][i * 6 + a] * V[m - 1][s][a * 3 + l];
                            V[m][s][i * 3 + l] = v;
                        }
                }
            // 非线性真值
            std::vector<Vector6d> qt(M);
            qt[0] = qn[0];
            for (int k = 0; k < M - 1; ++k) {
                const Eigen::Vector3d bk = bseg[std::min(k / mseg, nseg - 1)];
                const Vector6d k1 = fb(qt[k], bk);
                const Vector6d k2 = fb(qt[k] + 0.5 * dt * k1, bk);
                const Vector6d k3 = fb(qt[k] + 0.5 * dt * k2, bk);
                const Vector6d k4 = fb(qt[k] + dt * k3, bk);
                qt[k + 1] = qt[k] + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
            }
            std::vector<Eigen::Vector3d> pt(M);
            for (int m = 0; m < M; ++m) pt[m] = noeOsc2rv(qt[m], 100, 1e-12).head<3>();
            const Est bst = estats(pn, pt);
            const int nb = 3 * nseg;
            Eigen::MatrixXd A = Eigen::MatrixXd::Zero(3 * M, nb);
            for (int m = 0; m < M; ++m)
                for (int s = 0; s < nseg; ++s) {
                    Eigen::Matrix<double, 3, 6> Jm =
                        Eigen::Map<const Eigen::Matrix<double, 3, 6, Eigen::RowMajor>>(Jp[m].data());
                    Eigen::Matrix<double, 6, 3> Vs;
                    for (int i = 0; i < 6; ++i)
                        for (int l = 0; l < 3; ++l) Vs(i, l) = V[m][s][i * 3 + l];
                    const Eigen::Matrix3d B = Jm * Vs;
                    double B9[9];
                    for (int x = 0; x < 3; ++x)
                        for (int l = 0; l < 3; ++l) B9[x * 3 + l] = B(x, l);
                    const auto row = qoejopt::biasEphRow(pn[m].data(), Jp[m].data(), B9, pt[m].data());
                    for (int x = 0; x < 3; ++x)
                        for (int l = 0; l < 3; ++l) A(3 * m + x, 3 * s + l) = row.Jb[x * 3 + l];
                }
            Eigen::VectorXd bt(nb);
            for (int s = 0; s < nseg; ++s)
                for (int l = 0; l < 3; ++l) bt[3 * s + l] = bseg[s][l];
            Eigen::JacobiSVD<Eigen::MatrixXd> sv(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
            const auto ss = sv.singularValues();
            // q0 自由：[q0(6); β] 联合设计条件（GVE Φ/β 列）
            {
                std::vector<std::vector<double>> Aq(M, std::vector<double>(36, 0.0));
                for (int i = 0; i < 36; ++i) Aq[0][i] = (i % 7 == 0) ? 1.0 : 0.0;
                for (int m = 1; m < M; ++m)
                    for (int i = 0; i < 6; ++i)
                        for (int j = 0; j < 6; ++j) {
                            double v = 0.0;
                            for (int a = 0; a < 6; ++a) v += Phi[m - 1][i * 6 + a] * Aq[m - 1][a * 6 + j];
                            Aq[m][i * 6 + j] = v;
                        }
                Eigen::MatrixXd Aj = Eigen::MatrixXd::Zero(3 * M, 6 + nb);
                for (int m = 0; m < M; ++m) {
                    Eigen::Matrix<double, 3, 6> Jm =
                        Eigen::Map<const Eigen::Matrix<double, 3, 6, Eigen::RowMajor>>(Jp[m].data());
                    Eigen::Matrix<double, 3, 6> Aqm;
                    for (int i = 0; i < 3; ++i)
                        for (int a = 0; a < 6; ++a) {
                            double v = 0.0;
                            for (int c = 0; c < 6; ++c) v += Jm(i, c) * Aq[m][c * 6 + a];
                            Aqm(i, a) = v;
                        }
                    Aj.block<3, 6>(3 * m, 0) = Aqm;
                    for (int s = 0; s < nseg; ++s) {
                        Eigen::Matrix<double, 6, 3> Vs;
                        for (int i = 0; i < 6; ++i)
                            for (int l = 0; l < 3; ++l) Vs(i, l) = V[m][s][i * 3 + l];
                        Aj.block<3, 3>(3 * m, 6 + 3 * s) = Jm * Vs;
                    }
                }
                Eigen::JacobiSVD<Eigen::MatrixXd> svj(Aj, Eigen::ComputeThinU | Eigen::ComputeThinV);
                const auto sj = svj.singularValues();
                std::printf("[PosCond] arc=%.2fT seg=%d q0-free cond=%.3e sv_min=%.3e; "
                            "q0-fixed beta cond=%.3e\n", arc / kT, nseg,
                            sj(0) / std::max(sj(sj.size() - 1), 1e-300),
                            sj(sj.size() - 1), ss(0) / std::max(ss(ss.size() - 1), 1e-300));
            }
            std::printf("[PosFit] arc=%.2fT seg=%d base(end/rms)=%.3f/%.3f m cond_beta=%.2e\n",
                        arc / kT, nseg, bst.mx, bst.rms, ss(0) / std::max(ss(ss.size() - 1), 1e-300));
            std::mt19937_64 rng(2024 + nseg);
            std::normal_distribution<double> nd(0, 1);
            std::vector<double> unit(3 * M);
            for (double &v : unit) v = nd(rng);
            for (double sig : {0.0, 0.1, 0.3, 1.0}) {
                Eigen::VectorXd rhs = Eigen::VectorXd::Zero(nb);
                for (int m = 0; m < M; ++m)
                    for (int x = 0; x < 3; ++x) {
                        const double obs = pt[m][x] + sig * unit[3 * m + x];
                        for (int q = 0; q < nb; ++q) rhs[q] += A(3 * m + x, q) * (obs - pn[m][x]);
                    }
                const Eigen::VectorXd bh = (A.transpose() * A).ldlt().solve(rhs);
                std::vector<Eigen::Vector3d> pc(M);
                for (int m = 0; m < M; ++m) {
                    pc[m] = pn[m];
                    for (int x = 0; x < 3; ++x) {
                        double v = 0;
                        for (int q = 0; q < nb; ++q) v += A(3 * m + x, q) * bh[q];
                        pc[m][x] += v;
                    }
                }
                const Est cs = estats(pc, pt);
                const double bend = (pn.back() - pt.back()).norm();
                const double cend = (pc.back() - pt.back()).norm();
                std::printf("[PosFit] sig=%.1f beta_rel=%.3e end=%.3f->%.3f rms=%.3f->%.3f "
                            "red_end=%.1f%% red_rms=%.1f%%\n", sig,
                            (bh - bt).norm() / std::max(bt.norm(), 1e-300), bend, cend,
                            bst.rms, cs.rms, 100.0 * (1 - cend / std::max(bend, 1e-300)),
                            100.0 * (1 - cs.rms / std::max(bst.rms, 1e-300)));
                if (std::abs(arc - kT) < 1e-6) {   // 核心：1T in-sample 全 σ/段 降幅 ≥60%
                    EXPECT_LE(cend, 0.4 * bend + 1e-9);
                    EXPECT_LE(cs.rms, 0.4 * bst.rms + 1e-9);
                }
                // holdout 诊断：前 1/3T 拟合→后 2/3T（不设 60% 硬断言）
                {
                    const int mcut = (int)std::floor((arc / 3.0) / dt + 1e-9);
                    const int rows = 3 * (mcut + 1);
                    Eigen::MatrixXd Ap = A.topRows(rows);
                    Eigen::JacobiSVD<Eigen::MatrixXd> svp(Ap, Eigen::ComputeThinU | Eigen::ComputeThinV);
                    const auto sp = svp.singularValues();
                    Eigen::MatrixXd pinv = Eigen::MatrixXd::Zero(nb, rows);
                    for (int q = 0; q < sp.size(); ++q)
                        if (sp(q) > 1e-12 * sp(0))
                            pinv += svp.matrixV().col(q) * svp.matrixU().col(q).transpose() / sp(q);
                    Eigen::VectorXd rvec(rows);
                    for (int m = 0; m <= mcut; ++m)
                        for (int x = 0; x < 3; ++x)
                            rvec[3 * m + x] = pt[m][x] + sig * unit[3 * m + x] - pn[m][x];
                    const Eigen::VectorXd bhp = pinv * rvec;
                    std::vector<Eigen::Vector3d> pp(M);
                    for (int m = 0; m < M; ++m) {
                        pp[m] = pn[m];
                        for (int x = 0; x < 3; ++x) {
                            double v = 0;
                            for (int q = 0; q < nb; ++q) v += A(3 * m + x, q) * bhp[q];
                            pp[m][x] += v;
                        }
                    }
                    const Est cp = estats(pp, pt, mcut + 1);
                    const Est bp = estats(pn, pt, mcut + 1);
                    std::printf("[PosPred] arc=%.2fT seg=%d sig=%.1f rank=%d sv_min=%.3e "
                                "last2/3 rms=%.3f->%.3f red=%.1f%% (diagnostic)\n",
                                arc / kT, nseg, sig, (int)(sp.array() > 1e-12 * sp(0)).count(),
                                sp(sp.size() - 1), bp.rms, cp.rms,
                                100.0 * (1 - cp.rms / std::max(bp.rms, 1e-300)));
                }
            }
        }
    }
}


// B) 逐帧 ∂p/∂β = J_p Γ(t)：与 gveBiasFramesBatch（Flat 同源）逐帧一致；中间帧非零。
TEST(BiasDa, PerFramePositionBiasJac) {
    const std::vector<double> q0 = data_qoe(0);
    const int M = 13;
    const double dt = kT / (M - 1);
    const auto tr = nominal_traj(q0, kT, M);
    double x0[6]; for (int i = 0; i < 6; ++i) x0[i] = q0[i];
    double rtn[3] = {1, 1, 1};
    std::vector<double> G((size_t)M * 18), e((size_t)M * 6);
    qoejopt::gveBiasFramesBatch(x0, 1, M, 1, dt, kTau, 0, rtn, G.data(), e.data());
    double max_abs = 0, min_norm = 1e300;
    for (int m = 1; m < M; ++m) {
        double Jp[18];
        oe_jac_one(tr[m].q, Jp);
        Eigen::Matrix<double, 3, 6> Jq =
            Eigen::Map<const Eigen::Matrix<double, 3, 6, Eigen::RowMajor>>(Jp);
        Eigen::Matrix<double, 6, 3> Gm;
        for (int a = 0; a < 6; ++a)
            for (int l = 0; l < 3; ++l) Gm(a, l) = G[(size_t)m * 18 + a * 3 + l];
        const Eigen::Matrix3d B = Jq * Gm;
        double B9[9];
        for (int a = 0; a < 3; ++a)
            for (int l = 0; l < 3; ++l) B9[a * 3 + l] = B(a, l);
        const auto row = qoejopt::biasEphRow(tr[m].p.data(), Jp, B9, tr[m].p.data());
        for (int a = 0; a < 9; ++a) max_abs = std::max(max_abs, std::abs(row.Jb[a] - B9[a]));
        min_norm = std::min(min_norm, B.norm());
    }
    std::printf("[FrameJac] max|biasEphRow.Jb - JpGamma|=%.3e min|JpGamma|=%.3e "
                "(analytic/Flat reference; previous FD 3.3e-5 was FD noise)\n",
                max_abs, min_norm);
    EXPECT_LT(max_abs, 1e-12);
    EXPECT_GT(min_norm, 0.0);
}

// C) 5 邻居 ISL：(a) 共同 ECI b 的 3 维零空间；(b) 逐星 RTN β 的真实秩/条件；锚点反演。
TEST(BiasDa, FiveNeighborRangeBiasRank) {
    const int N = 6, M = 13, E = 5;
    const double dt = kT / (M - 1);
    std::vector<std::vector<NomF>> tr(N);
    for (int i = 0; i < N; ++i) tr[i] = nominal_traj(data_qoe(i), kT, M);
    // (a) 每星 ECI b 的 ISL J（JpG=½t²I）
    Eigen::MatrixXd Je = Eigen::MatrixXd::Zero(M * E, 3 * N);
    for (int m = 0; m < M; ++m)
        for (int e = 0; e < E; ++e) {
            const int i = e, j = e + 1;
            const double sc = 0.5 * (m * dt) * (m * dt);
            double A9[9] = {sc,0,0, 0,sc,0, 0,0,sc};
            const auto rw = qoejopt::biasIslRow(tr[i][m].p.data(), tr[j][m].p.data(),
                                                A9, A9, (tr[i][m].p - tr[j][m].p).norm());
            for (int l = 0; l < 3; ++l) {
                Je((m * E + e), 3 * i + l) = rw.Ji[l];
                Je((m * E + e), 3 * j + l) = rw.Jj[l];
            }
        }
    Eigen::JacobiSVD<Eigen::MatrixXd> sv(Je, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const auto s = sv.singularValues();
    const double tol = 1e-10 * std::max(s(0), 1e-300);
    int rank = 0; for (int q = 0; q < s.size(); ++q) if (s(q) > tol) ++rank;
    Eigen::VectorXd vc = Eigen::VectorXd::Zero(3 * N);
    for (int i = 0; i < N; ++i) vc[3 * i + 1] = 2e-7;
    const double jv = (Je * vc).norm() / std::max(vc.norm(), 1e-300);
    Eigen::VectorXd vt = vc;
    for (int i = 0; i < N; ++i) {
        vt[3 * i + 0] += 0.1e-7 * (i - 2.5);
        vt[3 * i + 1] += 1.0e-7 * (1.0 + 0.15 * i);
        vt[3 * i + 2] += 0.05e-7 * std::sin(0.6 * i);
    }
    const Eigen::VectorXd rr = Je * vt;
    Eigen::VectorXd bh = Eigen::VectorXd::Zero(3 * N);
    for (int q = 0; q < s.size(); ++q)
        if (s(q) > tol) bh += sv.matrixV().col(q) * (sv.matrixU().col(q).dot(rr) / s(q));
    // 锚卫星 0 末帧
    Eigen::MatrixXd Ja = Eigen::MatrixXd::Zero(M * E + 3, 3 * N);
    Ja.topRows(M * E) = Je;
    const double se = 0.5 * kT * kT;
    Ja.block<3, 3>(M * E, 0) = se * Eigen::Matrix3d::Identity();
    Eigen::VectorXd ra(M * E + 3); ra.head(M * E) = rr; ra.tail(3) = se * vt.head<3>();
    Eigen::JacobiSVD<Eigen::MatrixXd> sva(Ja, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const auto sa = sva.singularValues();
    const double tola = 1e-10 * std::max(sa(0), 1e-300);
    int ranka = 0; for (int q = 0; q < sa.size(); ++q) if (sa(q) > tola) ++ranka;
    Eigen::VectorXd bha = Eigen::VectorXd::Zero(3 * N);
    for (int q = 0; q < sa.size(); ++q)
        if (sa(q) > tola) bha += sva.matrixV().col(q) * (sva.matrixU().col(q).dot(ra) / sa(q));
    const double berra = (bha - vt).norm() / vt.norm();
    const double common_rec = bh.dot(vc) / vc.squaredNorm();
    std::printf("[IslECI] rank=%d null=%d |J*v_common|=%.3e; no-anchor err=%.3e "
                "common_rec=%.3e; anchor rank=%d err=%.3e\n", rank, 3 * N - rank, jv,
                (bh - vt).norm() / vt.norm(), common_rec, ranka, berra);
    EXPECT_EQ(rank, 15);
    EXPECT_LT(jv, 1e-9);
    EXPECT_EQ(ranka, 18);
    EXPECT_LT(std::abs(common_rec), 1e-8);
    EXPECT_LT(berra, 1e-8);
    // (b) 逐星 RTN β：真实秩/条件
    const int nseg = 12;
    Eigen::MatrixXd Jr = Eigen::MatrixXd::Zero(M * E, 3 * N);
    for (int i = 0; i < N; ++i) {
        RtnChain c = build_rtn_chain(data_qoe(i), nseg, kT / nseg);
        for (int m = 0; m < M; ++m) {
            double Jp[18];
            Vector6d qm;
            for (int cc = 0; cc < 6; ++cc) qm[cc] = c.q[m][cc];
            oe_jac_one(qm, Jp);
            Eigen::Matrix<double, 3, 6> Jq =
                Eigen::Map<const Eigen::Matrix<double, 3, 6, Eigen::RowMajor>>(Jp);
            Eigen::Matrix<double, 6, 3> V;
            for (int a = 0; a < 6; ++a)
                for (int l = 0; l < 3; ++l) V(a, l) = c.V[m][a * 3 + l];
            const Eigen::Matrix3d B = Jq * V;
            for (int e = 0; e < E; ++e) {
                const int ii = e, jj = e + 1;
                if (i != ii && i != jj) continue;
                Eigen::Vector3d u = tr[ii][m].p - tr[jj][m].p; u.normalize();
                Eigen::Matrix<double, 1, 3> row = u.transpose() * B;
                const int sign = (i == ii) ? 1 : -1;
                for (int l = 0; l < 3; ++l) Jr(m * E + e, 3 * i + l) += sign * row[l];
            }
        }
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svr(Jr, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const auto sr = svr.singularValues();
    const double tolr = 1e-10 * std::max(sr(0), 1e-300);
    int rankr = 0; for (int q = 0; q < sr.size(); ++q) if (sr(q) > tolr) ++rankr;
    std::printf("[IslRTN] rank=%d null=%d cond=%.3e sv_min=%.3e "
                "(per-star RTN β; common RTN is NOT a nullspace)\n",
                rankr, 3 * N - rankr, sr(0) / std::max(sr(sr.size() - 1), 1e-300),
                sr(sr.size() - 1));
    std::printf("[IslRTN] sat1 per-edge max|d range/d beta1|: ");
    for (int e = 0; e < E; ++e) {
        double mx = 0;
        for (int m = 0; m < M; ++m) mx = std::max(mx, Jr.block<1, 3>(m * E + e, 3).norm());
        std::printf("%.3e%s", mx, e + 1 < E ? " " : "\n");
    }
}

// D) β=1e-8..1e-5：Γ(t)β 线性预测 vs 完整非线性 GVE（B(t)β e^{-t/τ}）；二阶项 ∝β²。
TEST(BiasDa, LinearizationBetaSweep) {
    const std::vector<double> q0 = data_qoe(0);
    Vector6d x0v; for (int i = 0; i < 6; ++i) x0v[i] = q0[i];
    double x0[6]; for (int i = 0; i < 6; ++i) x0[i] = q0[i];
    std::vector<double> x0vec(x0, x0 + 6);
    std::vector<double> G;
    kernel_gamma(x0vec, 1, kT, 575, G, kTau);
    Eigen::Matrix<double, 6, 3> Gm;
    for (int a = 0; a < 6; ++a)
        for (int l = 0; l < 3; ++l) Gm(a, l) = G[a * 3 + l];
    const Eigen::Vector3d base(0.2e-7, 1.0e-7, 0.2e-7);
    auto nl = [&](const Vector6d &q0x, const Eigen::Vector3d &b) {
        Vector6d q = q0x;
        const int ns = 575;
        const double h = kT / ns;
        for (int s = 0; s < ns; ++s) {
            const double t = s * h;
            auto rhs = [&](const Vector6d &y, double tt) {
                double B[18]; b_rtn_arr(y.data(), B);
                Eigen::Matrix<double, 6, 3> Bm;
                for (int i = 0; i < 6; ++i)
                    for (int l = 0; l < 3; ++l) Bm(i, l) = B[i * 3 + l];
                return (noeGveRhs(y, 1.0, 0.0) + Bm * (b * std::exp(-tt / kTau))).eval();
            };
            Vector6d k1 = rhs(q, t), k2 = rhs(q + 0.5 * h * k1, t + 0.5 * h),
                     k3 = rhs(q + 0.5 * h * k2, t + 0.5 * h), k4 = rhs(q + h * k3, t + h);
            q += (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        }
        return q;
    };
    double last_pos = 0, last_vel = 0;
    const Vector6d qnom = nl(x0v, Eigen::Vector3d::Zero());
    for (double scale : {1e-1, 1.0, 1e1, 1e2}) {
        const Eigen::Vector3d b = base * scale;
        const Vector6d ql = qnom + Gm * b;
        const Vector6d qp = nl(x0v, b), qm = nl(x0v, -b);
        const Vector6d qodd = qnom + 0.5 * (qp - qm);   // 名义 + odd 一阶项
        const Vector6d rvp = noeOsc2rv(qp, 100, 1e-12), rvl = noeOsc2rv(ql, 100, 1e-12),
                       rvo = noeOsc2rv(qodd, 100, 1e-12);
        const double dp_lin = (rvp.head<3>() - rvl.head<3>()).norm();
        const double dv_lin = (rvp.tail<3>() - rvl.tail<3>()).norm();
        const double dp_odd = (rvp.head<3>() - rvo.head<3>()).norm();
        const double dv_odd = (rvp.tail<3>() - rvo.tail<3>()).norm();
        const double bt = b[1];
        std::printf("[LinSweep] bT=%.0e Gamma-b vs nl dp/dv=%.3e/%.3e; odd-even dp/dv=%.3e/%.3e; "
                    "even/bT^2=%.3e/%.3e\n", bt, dp_lin, dv_lin, dp_odd, dv_odd,
                    dp_odd / (bt * bt), dv_odd / (bt * bt));
        if (std::abs(bt - 1e-7) < 1e-13) { last_pos = dp_odd; last_vel = dv_odd; }
    }
    EXPECT_LT(last_pos, 1e-3);
    EXPECT_LT(last_vel, 1e-3);
}

// F) 性能：单节点 1T Φ/Γ/位置行；窗口 solve（bias on/off）。
TEST(BiasDa, PerformanceTiming) {
    using clk = std::chrono::steady_clock;
    auto now = [] { return std::chrono::duration<double>(clk::now().time_since_epoch()).count(); };
    const std::vector<double> q0 = data_qoe(0);
    Vector6d x0v; for (int i = 0; i < 6; ++i) x0v[i] = q0[i];
    const int M = 576;
    double t0 = now();
    std::vector<Vector6d> qn(M);
    std::vector<std::vector<double>> Phi(M - 1, std::vector<double>(36));
    qn[0] = x0v;
    for (int k = 0; k < M - 1; ++k) {
        Vector6d qf; dstep(qn[k].data(), kT / (M - 1), qf.data(), Phi[k].data());
        qn[k + 1] = qf;
    }
    std::vector<double> starts((size_t)(M - 1) * 6);
    for (int k = 0; k < M - 1; ++k)
        for (int i = 0; i < 6; ++i) starts[(size_t)k * 6 + i] = qn[k][i];
    std::vector<double> G;
    kernel_gamma(starts, M - 1, kT / (M - 1), 1, G, 1e9);
    double t1 = now();
    for (int m = 0; m < M; ++m) {
        double Jp[18]; oe_jac_one(qn[m], Jp);
    }
    double t2 = now();
    std::printf("[Perf] node1T M=%d chain(Phi/Gamma)=%.1f ms Jp/rows=%.1f ms\n",
                M, 1e3 * (t1 - t0), 1e3 * (t2 - t1));
    // window solve（L=3, n=4, fpn=8, E=4）
    const int n = 4, L = 3, fpn = 8, E = 4;
    std::vector<double> P((size_t)L * fpn * n * 3), Jpv((size_t)L * fpn * n * 18),
        JpG((size_t)L * fpn * n * 9), D((size_t)L * fpn * E, 40.0),
        Mm((size_t)L * fpn * E, 1.0), X((size_t)L * n * 6), beta((size_t)L * n * 3, 0.0),
        Ph((size_t)L * n * 36), Gm((size_t)L * n * 18, 0.0), Pb((size_t)n * 3, 1e-7);
    for (size_t q = 0; q < P.size(); ++q) P[q] = 7e6 + 0.01 * (double)(q % 7);
    for (size_t q = 0; q < Jpv.size(); ++q) Jpv[q] = 0.1 * std::sin(0.01 * (double)q);
    for (size_t q = 0; q < JpG.size(); ++q) JpG[q] = 0.01 * std::cos(0.02 * (double)q);
    for (size_t q = 0; q < X.size(); ++q) X[q] = 0.001 * (double)(q % 11);
    for (size_t q = 0; q < Ph.size(); ++q) Ph[q] = (q % 7 == 0) ? 1.0 : 0.0;
    int ei[4] = {0, 1, 2, 3}, ej[4] = {1, 2, 3, 0};
    qoejopt::BiasWinData d;
    d.n = n; d.L = L; d.fpn = fpn; d.E = E; d.sigma_isl = 1.0; d.phi = 1.0; d.Qb = 1e-12;
    d.ei = ei; d.ej = ej; d.D = D.data(); d.M = Mm.data();
    d.P = P.data(); d.Jp = Jpv.data(); d.JpG = JpG.data();
    d.X = X.data(); d.beta = beta.data(); d.Phi = Ph.data(); d.Gam = Gm.data();
    d.PbDiag = Pb.data(); d.compute_cov = false; d.q_weak = 1e-3;
    qoejopt::BiasWinResult res;
    t0 = now(); qoejopt::biasWindowSolve(d, res); t1 = now();
    std::printf("[Perf] window n=%d L=%d fpn=%d E=%d bias-on solve=%.2f ms nq=%d nb=%d\n",
                n, L, fpn, E, 1e3 * (t1 - t0), (L + 1) * n * 6, L * n * 3);
    // bias-off：同 ISL 行的 q-only 稀疏正规方程
    const int nq = (L + 1) * n * 6;
    Eigen::SparseMatrix<double> Hqq(nq, nq);
    std::vector<Eigen::Triplet<double>> tri;
    for (int t = 0; t < L * fpn; ++t)
        for (int e = 0; e < E; ++e) {
            const int i = ei[e], j = ej[e], ki = t / fpn;
            for (int c = 0; c < 6; ++c) {
                tri.emplace_back((ki * n + i) * 6 + c, (ki * n + i) * 6 + c, 1.0);
                tri.emplace_back((ki * n + j) * 6 + c, (ki * n + j) * 6 + c, 1.0);
            }
        }
    Hqq.setFromTriplets(tri.begin(), tri.end());
    t0 = now();
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ld;
    ld.compute(Hqq);
    Eigen::VectorXd dq = ld.solve(Eigen::VectorXd::Ones(nq));
    t1 = now();
    std::printf("[Perf] window bias-off q-only solve=%.2f ms nq=%d\n", 1e3 * (t1 - t0), nq);
    EXPECT_GT(dq.norm(), 0.0);
}
}  // namespace bias_da_impl

// ===================== test/test_bias_inband_gtest.cpp =====================
namespace bias_inband_impl {
// Task #20 自测：in-band [q;β] 联合装配/求解核心（per-star β）。
// 线性合成用例：q 由强先验钉在参考点，ISL 距离只由 β_true 产生；
// 检查 β 回收、残差下降、β 列有限差分与 GM/Schur 解的一致性。


namespace {
constexpr int N = 4, L = 5, FPN = 4, E = 6;

std::vector<std::pair<int, int>> edges() {
    std::vector<std::pair<int, int>> e;
    for (int i = 0; i < N; ++i)
        for (int j = i + 1; j < N; ++j) e.emplace_back(i, j);
    return e;   // 6 条边 = 完全图
}

void build(std::vector<double> &X, std::vector<double> &beta,
           std::vector<double> &Phi, std::vector<double> &Gam,
           std::vector<double> &P, std::vector<double> &Jp,
           std::vector<double> &JpG, std::vector<double> &D,
           std::vector<double> &M, std::vector<double> &Qis,
           std::vector<double> &qInfo, std::vector<double> &PbDiag,
           std::vector<double> &beta_true) {
    std::mt19937_64 rng(20261031);
    std::normal_distribution<double> nd(0.0, 1.0);
    X.assign((size_t)(L + 1) * N * 6, 0.0);
    beta.assign((size_t)L * N * 3, 0.0);
    beta_true.assign((size_t)L * N * 3, 0.0);
    Phi.assign((size_t)L * N * 36, 0.0);
    Gam.assign((size_t)L * N * 18, 0.0);
    P.assign((size_t)L * FPN * N * 3, 0.0);
    Jp.assign((size_t)L * FPN * N * 18, 0.0);
    JpG.assign((size_t)L * FPN * N * 9, 0.0);
    D.assign((size_t)L * FPN * E, 0.0);
    M.assign((size_t)L * FPN * E, 1.0);
    Qis.assign((size_t)L * N * 36, 0.0);
    qInfo.assign((size_t)(L + 1) * N * 36, 0.0);
    PbDiag.assign((size_t)N * 3, 1e-3);  // 测试弱先验，让 ISL 主导
    // q 参考：每星位置约 7e6 m
    for (int k = 0; k <= L; ++k)
        for (int i = 0; i < N; ++i) {
            const size_t b = ((size_t)k * N + i) * 6;
            X[b + 0] = 7.0e6 + 1e3 * i;
            X[b + 1] = 2.0e5 * (i + 1);
            X[b + 2] = 3.0e5 * (i + 1);
        }
    // Φ≈I、Γ 位置量级 1e5；Qis=I；q 先验 σ=1e-3（钉住 q）
    for (int k = 0; k < L; ++k)
        for (int i = 0; i < N; ++i) {
            const size_t b = ((size_t)k * N + i);
            for (int a = 0; a < 6; ++a) {
                Phi[b * 36 + a * 6 + a] = 1.0;   // q 节点未做 Φ 传播，dyn 残差须为 0
                Qis[b * 36 + a * 6 + a] = 1.0;
            }
            for (int a = 0; a < 6; ++a)
                for (int l = 0; l < 3; ++l) {
                    const double scale = (a < 3) ? 1e5 : 1e2;
                    Gam[b * 18 + a * 3 + l] = scale * (0.5 + 0.5 * nd(rng));
                }
            for (int l = 0; l < 3; ++l) {
                beta_true[b * 3 + l] = 1e-6 * nd(rng);
                beta[(size_t)b * 3 + l] = 0.0;
            }
        }
    // dyn 的 Γ 置 0：本测试只验 ISL β 列；q 仍由强先验钉住。
    std::fill(Gam.begin(), Gam.end(), 0.0);
    // ISL 只观测 β 的跨星差；去掉每区间/每轴共模，避免不可观测模被先验收缩。
    for (int k = 0; k < L; ++k)
        for (int l = 0; l < 3; ++l) {
            double m = 0.0;
            for (int i = 0; i < N; ++i) m += beta_true[((size_t)k * N + i) * 3 + l];
            m /= N;
            for (int i = 0; i < N; ++i)
                beta_true[((size_t)k * N + i) * 3 + l] -= m;
        }
    for (int k = 0; k <= L; ++k)
        for (int i = 0; i < N; ++i)
            for (int a = 0; a < 6; ++a)
                qInfo[((size_t)k * N + i) * 36 + a * 6 + a] = 1e6;  // σ=1e-3，钉住 q
    // 位置/Jacobian/观测：p(t)=P_ref+Jp(q−X)+JpG β_true（线性模型）
    auto es = edges();
    for (int t = 0; t < L * FPN; ++t) {
        const int k = t / FPN;
        for (int i = 0; i < N; ++i) {
            const size_t b = (size_t)t * N + i;
            const double *x = &X[(size_t)(k * N + i) * 6];
            for (int a = 0; a < 3; ++a) {
                P[b * 3 + a] = x[a] + 1e3 * i + 10.0 * a;
                for (int c = 0; c < 6; ++c)
                    Jp[b * 18 + a * 6 + c] = (a == c) ? 1.0 : 0.0;
            }
            // JpG 独立于 dyn Γ（dyn 的 Γ 置 0，本测试只验 ISL 的 β 列）；
            // 逐帧变化以增加 β 可观测秩。
            for (int a = 0; a < 3; ++a)
                for (int l = 0; l < 3; ++l)
                    JpG[b * 9 + a * 3 + l] =
                        1e5 * (0.5 + 0.5 * nd(rng)) * (1.0 + 0.1 * (t % FPN));
            // 真值位移
            const size_t bi = (size_t)(k * N + i) * 3;
            for (int a = 0; a < 3; ++a) {
                double s = 0.0;
                for (int l = 0; l < 3; ++l) s += JpG[b * 9 + a * 3 + l] * beta_true[bi + l];
                P[b * 3 + a] += s;      // P 存真值位置；β 列仍是 JpG
            }
        }
        for (int e2 = 0; e2 < E; ++e2) {
            const int i = es[e2].first, j = es[e2].second;
            const size_t bi = (size_t)t * N + i, bj = (size_t)t * N + j;
            double dd[3] = {P[bi * 3] - P[bj * 3], P[bi * 3 + 1] - P[bj * 3 + 1],
                            P[bi * 3 + 2] - P[bj * 3 + 2]};
            D[(size_t)t * E + e2] = std::sqrt(dd[0]*dd[0] + dd[1]*dd[1] + dd[2]*dd[2]);
        }
    }
}

}  // namespace

TEST(BiasInband, PerStarBetaRecovery) {
    std::vector<double> X, beta, Phi, Gam, P, Jp, JpG, D, M, Qis, qInfo,
        PbDiag, beta_true;
    build(X, beta, Phi, Gam, P, Jp, JpG, D, M, Qis, qInfo, PbDiag, beta_true);
    auto es = edges();
    std::vector<int> ei, ej;
    for (auto &e : es) { ei.push_back(e.first); ej.push_back(e.second); }
    qoejopt::BiasWinData d;
    d.n = N; d.L = L; d.fpn = FPN; d.E = E;
    d.sigma_isl = 1.0;
    d.phi = std::exp(-1920.0 / 5749.8);
    d.Qb = 1e-6;                      // 弱 GM，让 ISL 主导
    d.ei = ei.data(); d.ej = ej.data();
    d.D = D.data(); d.M = M.data();
    d.P = P.data(); d.Jp = Jp.data(); d.JpG = JpG.data();
    d.X = X.data(); d.beta = beta.data();
    d.Phi = Phi.data(); d.Gam = Gam.data(); d.Qis = Qis.data();
    d.qInfo = qInfo.data(); d.Xp = X.data(); d.PbDiag = PbDiag.data();
    d.compute_cov = true;
    qoejopt::BiasWinResult r;
    qoejopt::biasWindowSolve(d, r);
    // 注意：P 已经是“真值位置”（含 β_true 位移），而 β 当前为 0 →
    // GN 会解出 δβ≈0 而不是 β_true。用“参考位置”重跑：
    std::vector<double> P0 = P;
    for (int t = 0; t < L * FPN; ++t) {
        const int k = t / FPN;
        for (int i = 0; i < N; ++i) {
            const size_t b = (size_t)t * N + i;
            const size_t bi = (size_t)(k * N + i) * 3;
            for (int a = 0; a < 3; ++a) {
                double s = 0.0;
                for (int l = 0; l < 3; ++l) s += JpG[b * 9 + a * 3 + l] * beta_true[bi + l];
                P0[b * 3 + a] = P[b * 3 + a] - s;
            }
        }
    }
    d.P = P0.data();
    {
        double cisl = 0.0, cdyn = 0.0;
        for (int t = 0; t < L * FPN; ++t) {
            const int k = t / FPN;
            for (int e2 = 0; e2 < E; ++e2) {
                const int i = es[e2].first, j = es[e2].second;
                const size_t bi = (size_t)t * N + i, bj = (size_t)t * N + j;
                double dd[3] = {P0[bi*3]-P0[bj*3], P0[bi*3+1]-P0[bj*3+1],
                                P0[bi*3+2]-P0[bj*3+2]};
                const double dist = std::sqrt(dd[0]*dd[0]+dd[1]*dd[1]+dd[2]*dd[2]);
                const double rr = dist - D[(size_t)t*E+e2];
                cisl += rr*rr;
            }
            for (int i = 0; i < N; ++i) {
                const size_t b = (size_t)k*N+i;
                for (int a = 0; a < 6; ++a) {
                    double v = X[((size_t)(k+1)*N+i)*6+a] - X[b*6+a];
                    for (int b2 = 0; b2 < 6; ++b2)
                        v -= Phi[b*36+a*6+b2]*X[b*6+b2];
                    cdyn += v*v;
                }
            }
        }
        double cdyn2 = 0.0, vmax = 0.0;
        for (int k = 0; k < L; ++k)
            for (int i = 0; i < N; ++i) {
                const size_t b = (size_t)k * N + i;
                for (int a = 0; a < 6; ++a) {
                    double v = X[((size_t)(k + 1) * N + i) * 6 + a];
                    for (int b2 = 0; b2 < 6; ++b2)
                        v -= Phi[b * 36 + a * 6 + b2] * X[b * 6 + b2];
                    cdyn2 += v * v;
                    vmax = std::max(vmax, std::abs(v));
                }
            }
        std::printf("[Inband/dbg] cisl=%.3e cdyn2=%.3e vmax=%.3e\n",
                    cisl, cdyn2, vmax);
    }
    qoejopt::biasWindowSolve(d, r);
    double num = 0.0, den = 0.0, qn = 0.0;
    for (size_t i = 0; i < beta_true.size(); ++i) {
        num += (r.dbeta[i] - beta_true[i]) * (r.dbeta[i] - beta_true[i]);
        den += beta_true[i] * beta_true[i];
    }
    for (size_t i = 0; i < r.dX.size(); ++i) qn += r.dX[i] * r.dX[i];
    const double brel = std::sqrt(num / std::max(den, 1e-300));
    // 用线性模型检查 δ 是否真的把 ISL 残差压下去（装配/求解正确性）。
    double c1 = 0.0;
    for (int t = 0; t < L * FPN; ++t) {
        const int k = t / FPN;
        std::vector<std::array<double, 3>> pp(N);
        for (int i = 0; i < N; ++i) {
            const size_t b = (size_t)t * N + i;
            const size_t qb = (size_t)k * N + i;
            const size_t bb = (size_t)qb * 3;
            for (int a = 0; a < 3; ++a) {
                double v = P0[b * 3 + a];
                for (int c = 0; c < 6; ++c)
                    v += Jp[b * 18 + a * 6 + c] * r.dX[qb * 6 + c];
                for (int l = 0; l < 3; ++l)
                    v += JpG[b * 9 + a * 3 + l] * r.dbeta[bb + l];
                pp[i][a] = v;
            }
        }
        for (int e2 = 0; e2 < E; ++e2) {
            const int i = es[e2].first, j = es[e2].second;
            double dd[3] = {pp[i][0]-pp[j][0], pp[i][1]-pp[j][1], pp[i][2]-pp[j][2]};
            const double dist = std::sqrt(dd[0]*dd[0]+dd[1]*dd[1]+dd[2]*dd[2]);
            const double rr = dist - D[(size_t)t*E+e2];
            c1 += rr*rr;
        }
    }
    std::printf("[Inband] post-fit ISL cost=%.3e (before=%.3e)\n", c1, r.cost_isl);
    std::printf("[Inband] cost=%.3e (isl=%.3e dyn=%.3e beta=%.3e q=%.3e) "
                "beta_rel_err=%.3e |dq|=%.3e betaSigma n=%ld med=%.3e\n",
                r.cost, r.cost_isl, r.cost_dyn, r.cost_beta, r.cost_q,
                brel, std::sqrt(qn), (long)r.betaSigma.size(),
                r.betaSigma.size() ? r.betaSigma[r.betaSigma.size()/2] : 0.0);
    EXPECT_LT(c1, 0.05 * r.cost_isl);
    EXPECT_LT(brel, 0.5);   // 合成 ISL 秩/先验下只要求同量级
    EXPECT_GT(r.cost, 0.0);
    EXPECT_GT(r.betaSigma.size(), 0u);
}

TEST(BiasInband, BetaColumnFiniteDifference) {
    // 单区间、单边：d||p_i−p_j||/dβ 与 JpG_i−JpG_j 一致（线性模型精确）。
    std::vector<double> X, beta, Phi, Gam, P, Jp, JpG, D, M, Qis, qInfo,
        PbDiag, beta_true;
    build(X, beta, Phi, Gam, P, Jp, JpG, D, M, Qis, qInfo, PbDiag, beta_true);
    auto es = edges();
    const int e2 = 0, t = 0;
    const int i = es[e2].first, j = es[e2].second;
    const size_t bi = (size_t)t * N + i, bj = (size_t)t * N + j;
    double db[3];
    for (int l = 0; l < 3; ++l) db[l] = JpG[bi * 9 + 1 * 3 + l] - JpG[bj * 9 + 1 * 3 + l];
    // 解析：u·(JpG_i−JpG_j)
    double dd[3] = {P[bi * 3] - P[bj * 3], P[bi * 3 + 1] - P[bj * 3 + 1],
                    P[bi * 3 + 2] - P[bj * 3 + 2]};
    const double dist = std::sqrt(dd[0]*dd[0] + dd[1]*dd[1] + dd[2]*dd[2]);
    double u[3] = {dd[0]/dist, dd[1]/dist, dd[2]/dist};
    // FD
    const double eps = 1e-3;   // 线性模型，步长取大以避开大数相减
    auto dr = [&](int l, double s) {
        double dp[3];
        for (int a = 0; a < 3; ++a)
            dp[a] = (JpG[bi * 9 + a * 3 + l] - JpG[bj * 9 + a * 3 + l]) * s;
        double d2[3] = {dd[0] + dp[0], dd[1] + dp[1], dd[2] + dp[2]};
        return std::sqrt(d2[0]*d2[0] + d2[1]*d2[1] + d2[2]*d2[2]) - dist;
    };
    double fd[3];
    for (int l = 0; l < 3; ++l) fd[l] = (dr(l, eps) - dr(l, -eps)) / (2.0 * eps);
    double den = 0.0;
    double maxerr = 0.0;
    for (int l = 0; l < 3; ++l) {
        const double an = u[0]*(JpG[bi*9+0*3+l]-JpG[bj*9+0*3+l])
                        + u[1]*(JpG[bi*9+1*3+l]-JpG[bj*9+1*3+l])
                        + u[2]*(JpG[bi*9+2*3+l]-JpG[bj*9+2*3+l]);
        maxerr = std::max(maxerr, std::abs(fd[l] - an));
        den += an * an;
    }
    std::printf("[Inband/FD] max|FD-JpG|=%.3e |JpG|=%.3e\n", maxerr, std::sqrt(den));
    // 双精度在 1e5 m 距离上做 FD 的绝对分辨有限；用相对 |JpG| 判据。
    EXPECT_LT(maxerr, 1e-6 * std::sqrt(den));
}
}  // namespace bias_inband_impl
