// gauge / local-handoff header-only 核心代数测试（合并自原四个独立 gtest 源）：
//   1) Householder 显式投影 vs 生产 rank-2k 公式、mode 1 gauge 块、6×6 Schur 谱截断、
//      local_handoff_blocks 的 J6_stable/J6 一致性；
//   2) 离线 gauge 投影/边缘化代数参考（3 星，18 维）：显式投影 vs rank-12 Woodbury、
//      消 node0 后的 Pm/Pg_next、GTS 行分解恒等式、Pg 换基协方差/信息恒等式；
//   3) D1/D4 gauge 块闭式公式、Q 坐标 identity、Woodbury 节点边际；
//   4) L=1 EKF 局部 joint 交接（3 星，m=18）的数学门禁。
// 不依赖 torch/CUDA；只编译，不参与生产路径。运行：build/gtest_gauge_core
#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <random>

#include "fls_gauge_householder.h"
#include "fls_gauge_lamwin.h"
#include "fls_local_handoff.h"

namespace lamwin_test {

struct Problem {
    Eigen::MatrixXd K, G, Pg, PgInv, KG, Sg, S, SInv, Pi, Lam;
    int ncol = 0, k = 0;
    double eps = 0.0;
};

void set_eps(Problem &p, double eps) {
    p.eps = eps;
    const int ncol = p.ncol, k = p.k;
    p.Sg = p.G.transpose() * p.G;
    p.Sg = 0.5 * (p.Sg + p.Sg.transpose());
    p.S = p.Sg + eps * Eigen::MatrixXd::Identity(k, k);
    p.SInv = p.S.inverse();
    p.Pi = p.G * p.SInv * p.G.transpose();
    const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(ncol, ncol);
    p.Lam = (I - p.Pi) * p.K * (I - p.Pi) + p.G * p.PgInv * p.G.transpose();
}

Problem make_problem(int ncol, int k, unsigned seed = 7) {
    Eigen::MatrixXd A(ncol + 5, ncol);
    for (int i = 0; i < A.rows(); ++i)
        for (int j = 0; j < A.cols(); ++j)
            A(i, j) = std::sin(0.37 * (i + 1) * (j + 2) + 0.11 * seed);
    Problem p;
    p.ncol = ncol;
    p.k = k;
    p.K = A.transpose() * A + 0.7 * Eigen::MatrixXd::Identity(ncol, ncol);
    p.G.resize(ncol, k);
    for (int i = 0; i < ncol; ++i)
        for (int j = 0; j < k; ++j)
            p.G(i, j) = std::cos(0.19 * (i + 3) * (j + 1) + 0.05 * seed);
    Eigen::MatrixXd B(k, k);
    for (int i = 0; i < k; ++i)
        for (int j = 0; j < k; ++j) B(i, j) = std::sin(0.31 * (i + 1) * (j + 2));
    p.Pg = B * B.transpose() + 0.2 * Eigen::MatrixXd::Identity(k, k);
    p.PgInv = p.Pg.inverse();
    p.eps = 1e-9;
    p.KG = p.K * p.G;
    set_eps(p, p.eps);
    return p;
}

}  // namespace lamwin_test

namespace gaugefix_test {

using Mat = Eigen::MatrixXd;
using M6 = Eigen::Matrix<double, 6, 6>;

Mat sym(const Mat &a) { return 0.5 * (a + a.transpose()); }

Mat random_spd(int n, std::mt19937_64 &rng) {
    std::normal_distribution<double> g(0.0, 1.0);
    Mat a(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) a(i, j) = g(rng);
    return sym(a.transpose() * a) + 0.5 * Mat::Identity(n, n);
}

Mat random_tall(int n, int k, std::mt19937_64 &rng) {
    std::normal_distribution<double> g(0.0, 1.0);
    Mat a(n, k);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < k; ++j) a(i, j) = g(rng);
    return a;
}

}  // namespace gaugefix_test

namespace proj_test {

using Mat = Eigen::MatrixXd;
using Vec = Eigen::VectorXd;

Mat sym(const Mat &a) { return 0.5 * (a + a.transpose()); }

Mat build_gauge_rv(const Mat &rv) {
    const int n = (int)rv.rows();
    Mat g = Mat::Zero(6 * n, 6);
    for (int i = 0; i < n; ++i) {
        for (int a = 0; a < 3; ++a) g(6 * i + a, a) = 1.0;
        for (int a = 0; a < 3; ++a) {
            Eigen::Vector3d e = Eigen::Vector3d::Zero();
            e(a) = 1.0;
            const Eigen::Vector3d r = rv.block<1, 3>(i, 0).transpose();
            const Eigen::Vector3d v = rv.block<1, 3>(i, 3).transpose();
            g.block<3, 1>(6 * i, 3 + a) = e.cross(r);
            g.block<3, 1>(6 * i + 3, 3 + a) = e.cross(v);
        }
    }
    return g;
}

struct Problem {
    int n = 0;
    Mat G;
    Mat K;
    Mat Pg;
};

Problem make_problem() {
    const int n = 3;
    Mat rv(n, 6);
    rv << 1.0, 2.0, 3.0, 0.1, -0.2, 0.3,
          -2.0, 0.5, 1.0, 0.2, 0.1, -0.4,
          0.3, -1.0, 2.0, -0.3, 0.4, 0.2;
    Problem p;
    p.n = n;
    p.G = build_gauge_rv(rv);
    const int ncol = 6 * n;
    Mat a = Mat::Zero(3 * ncol, ncol);
    std::mt19937_64 rng(20261023ULL);
    std::normal_distribution<double> g(0.0, 1.0);
    for (int i = 0; i < a.rows(); ++i)
        for (int j = 0; j < a.cols(); ++j) a(i, j) = g(rng);
    p.K = sym(a.transpose() * a + 0.5 * Mat::Identity(ncol, ncol));
    p.Pg = Mat::Zero(6, 6);
    p.Pg(0, 0) = 2.0;
    p.Pg(1, 1) = 3.0;
    p.Pg(2, 2) = 4.0;
    p.Pg(3, 3) = 0.5;
    p.Pg(4, 4) = 0.6;
    p.Pg(5, 5) = 0.7;
    return p;
}

Problem make_window_problem() {
    Problem p = make_problem();
    const int nodes = 3;
    const Mat g_node = p.G;
    p.G = Mat::Zero(nodes * g_node.rows(), 6);
    for (int k = 0; k < nodes; ++k)
        p.G.block(k * g_node.rows(), 0, g_node.rows(), 6) = g_node;
    const int ncol = (int)p.G.rows();
    Mat a = Mat::Zero(3 * ncol, ncol);
    std::mt19937_64 rng(20261024ULL);
    std::normal_distribution<double> g(0.0, 1.0);
    for (int i = 0; i < a.rows(); ++i)
        for (int j = 0; j < a.cols(); ++j) a(i, j) = g(rng);
    p.K = sym(a.transpose() * a + 0.5 * Mat::Identity(ncol, ncol));
    return p;
}

Mat projector(const Mat &g, double eps, Mat *s_out = nullptr) {
    const int k = (int)g.cols();
    Mat s = sym(g.transpose() * g + eps * Mat::Identity(k, k));
    if (s_out) *s_out = s;
    return sym(g * s.llt().solve(g.transpose()));
}

Mat projected_hessian(const Mat &kmat, const Mat &g, const Mat &pg,
                      double eps = 0.0) {
    const Mat pi = projector(g, eps);
    const Mat eye = Mat::Identity(kmat.rows(), kmat.cols());
    const Mat a = eye - pi;
    return sym(a * kmat * a + g * pg.llt().solve(g.transpose()));
}

struct Terms {
    Mat X;
    Mat KG;
    Mat S;
    Mat C;
    Mat U;
    Mat Ucorr;
    Mat D;
};

Terms lowrank_terms(const Mat &kmat, const Mat &g, const Mat &pg,
                    double eps = 0.0) {
    Terms t;
    const int k = (int)g.cols();
    t.X = kmat.llt().solve(g);
    t.KG = kmat * g;
    t.S = sym(g.transpose() * g + eps * Mat::Identity(k, k));
    const Mat sinv = t.S.inverse();
    const Mat a = sym(g.transpose() * t.KG);
    const Mat q = pg.inverse();
    t.C = Mat::Zero(2 * k, 2 * k);
    t.C.topLeftCorner(k, k) = sinv * a * sinv + q;
    t.C.topRightCorner(k, k) = -sinv;
    t.C.bottomLeftCorner(k, k) = -sinv;
    t.C = sym(t.C);
    t.U = Mat::Zero(g.rows(), 2 * k);
    t.U.leftCols(k) = g;
    t.U.rightCols(k) = t.KG;
    t.Ucorr = Mat::Zero(g.rows(), 2 * k);
    t.Ucorr.leftCols(k) = t.X;
    t.Ucorr.rightCols(k) = g;
    t.D = Mat::Zero(2 * k, 2 * k);
    t.D.topLeftCorner(k, k) = g.transpose() * t.X;
    t.D.topRightCorner(k, k) = -eps * Mat::Identity(k, k);
    t.D.bottomLeftCorner(k, k) = -eps * Mat::Identity(k, k);
    t.D.bottomRightCorner(k, k) = -t.S * q * t.S;
    t.D = sym(t.D);
    return t;
}

Vec woodbury_solve(const Mat &kmat, const Mat &g, const Mat &pg, const Vec &b,
                   double eps = 0.0) {
    const Terms t = lowrank_terms(kmat, g, pg, eps);
    const int k = (int)g.cols();
    Vec rhs(2 * k);
    rhs.head(k) = t.X.transpose() * b;
    rhs.tail(k) = g.transpose() * b;
    return kmat.llt().solve(b) -
           t.Ucorr * t.D.fullPivLu().solve(rhs);
}

}  // namespace proj_test

namespace handoff_test {

using M6 = Eigen::Matrix<double, 6, 6>;

M6 random_spd6(std::mt19937_64 &rng, double scale) {
    // 良态 SPD：2I + 小对称扰动（cond ~1.1），保证本测试的欧氏范数检查有意义。
    std::normal_distribution<double> g(0.0, 1.0);
    Eigen::Matrix<double, 6, 6> A;
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j) A(i, j) = g(rng);
    const M6 B = 0.5 * (A + A.transpose());
    return scale * (2.0 * M6::Identity() + 0.05 * B);
}

}  // namespace handoff_test

TEST(GaugeLamWin, GaugeBlockAndQCoordinateIdentity) {
    using namespace lamwin_test;
    const Problem p = make_problem(18, 3);
    const fls::GaugeLamWinBlock r =
        fls::gauge_lamwin_block(p.KG, p.G, p.PgInv, p.eps);
    ASSERT_TRUE(r.ok) << r.reason;
    ASSERT_TRUE(r.qr_rank_ok) << r.reason;

    const Eigen::MatrixXd ref = p.G.transpose() * p.Lam * p.G;
    EXPECT_LT((r.GtLamG - ref).norm() / std::max(ref.norm(), 1e-300), 1e-10);
    EXPECT_LT((r.Q.transpose() * r.Q -
               Eigen::MatrixXd::Identity(p.k, p.k)).norm(), 1e-10);
    EXPECT_LT((p.G - r.Q * r.R).norm() / std::max(p.G.norm(), 1e-300), 1e-10);

    const Eigen::MatrixXd refQ = r.Q.transpose() * p.Lam * r.Q;
    EXPECT_LT((r.QtLamQ - refQ).norm() / std::max(refQ.norm(), 1e-300), 1e-10);
    EXPECT_LT(r.identity_rel, 1e-9);
    EXPECT_GT(r.min_eig, 0.0);
}

TEST(GaugeLamWin, WoodburyNode1MatchesDenseInverse) {
    using namespace lamwin_test;
    Problem p = make_problem(12, 2, 3);
    set_eps(p, 1e-4);                       // 让 D 的 1/eps 块条件数适合稠密对拍
    const int n0 = 6;                       // node0 / node1 各 6 维
    const int ku = 2 * p.k;
    const Eigen::MatrixXd Kinv = p.K.inverse();
    const Eigen::MatrixXd X = Kinv * p.G;   // K^-1 G
    Eigen::MatrixXd U(p.ncol, ku), Utot(p.ncol, ku);
    U.leftCols(p.k) = p.G;
    U.rightCols(p.k) = p.KG;
    Utot.leftCols(p.k) = X;
    Utot.rightCols(p.k) = p.G;

    Eigen::MatrixXd D = Eigen::MatrixXd::Zero(ku, ku);
    D.topLeftCorner(p.k, p.k) = p.G.transpose() * X;
    D.topRightCorner(p.k, p.k) = -p.eps * Eigen::MatrixXd::Identity(p.k, p.k);
    D.bottomLeftCorner(p.k, p.k) = -p.eps * Eigen::MatrixXd::Identity(p.k, p.k);
    D.bottomRightCorner(p.k, p.k) = -p.S * p.PgInv * p.S;
    D = 0.5 * (D + D.transpose());

    const Eigen::MatrixXd Hinv_wb = Kinv - Utot * D.inverse() * Utot.transpose();
    const Eigen::MatrixXd Hinv = p.Lam.inverse();
    EXPECT_LT((Hinv_wb - Hinv).norm() / std::max(Hinv.norm(), 1e-300), 1e-9);

    // node1 的 selected-inverse 块 == 稠密 Schur 消 node0。
    const Eigen::MatrixXd H00 = p.Lam.topLeftCorner(n0, n0);
    const Eigen::MatrixXd H01 = p.Lam.topLeftCorner(n0, n0 + 6).rightCols(6);
    const Eigen::MatrixXd H10 = p.Lam.topLeftCorner(n0 + 6, n0).bottomRows(6);
    const Eigen::MatrixXd H11 = p.Lam.bottomRightCorner(6, 6);
    const Eigen::MatrixXd P1 =
        (H11 - H10 * H00.inverse() * H01).inverse();
    const Eigen::MatrixXd P1_sel = Hinv.bottomRightCorner(6, 6);
    EXPECT_LT((P1 - P1_sel).norm() / std::max(P1.norm(), 1e-300), 1e-9);
    EXPECT_LT((P1 - Hinv_wb.bottomRightCorner(6, 6)).norm() /
                  std::max(P1.norm(), 1e-300), 1e-9);
}

TEST(GaugeFixes, HouseholderMatchesRank2k) {
    using namespace gaugefix_test;
    const int n = 18, k = 6;
    std::mt19937_64 rng(20261023ULL);
    const Mat lam = random_spd(n, rng);
    Mat g = random_tall(n, k, rng);
    // 让平移/旋转列量级接近生产（不改变代数）。
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < k; ++j)
            g(i, j) *= (j < 3 ? 1.0 : 1.0e-2);
    Mat pg = Mat::Zero(k, k);
    for (int j = 0; j < 3; ++j) pg(j, j) = 1e6;
    for (int j = 3; j < 6; ++j) pg(j, j) = 1e-8;

    const fls::HouseholderProjection h0 = fls::householder_project_prior(lam, g, &pg, 0);
    EXPECT_LT(h0.qr_err, 1e-12);
    EXPECT_LT(h0.gate_gauge, 1e-8);
    EXPECT_LT(h0.gate_perp, 1e-8);
    EXPECT_LT(h0.err_rel, 1e-6);
    EXPECT_GT(h0.lmin_total, 0.0);

    // mode 1 与 mode 0 的差恰为 G Pg^-1 G^T（低秩 gauge 项）。
    const fls::HouseholderProjection h1 = fls::householder_project_prior(lam, g, &pg, 1);
    const Mat Pginv = fls::spd_inverse_md(pg);
    const Mat gauge = g * Pginv * g.transpose();
    EXPECT_LT((h1.Lambda_proj - h0.Lambda_proj - gauge).norm() / gauge.norm(), 1e-8);
    EXPECT_LT(h1.err_rel, 1e-6);
    EXPECT_LT(h1.gate_gauge, 1e-8);
    EXPECT_LT(h1.gate_perp, 1e-8);
    EXPECT_GT(h1.lmin_total, 0.0);

    // mode 2 是 doc 的字面 h 坐标块，只检查与 expected 自洽。
    const fls::HouseholderProjection h2 = fls::householder_project_prior(lam, g, &pg, 2);
    EXPECT_LT(h2.err_rel, 1e-6);
    // mode 2 是 doc 字面 h 坐标块（R Pg R^T）^{-1}，不是生产路径；Pg 旋转
    // 1e-8 时该块条件数很大，允许 lmin_total 为小负（仅诊断自洽）。
    EXPECT_GE(h2.lmin_total, -1e-5);
}

TEST(GaugeFixes, ContractionClampBlocksSpectralRadius) {
    using namespace gaugefix_test;
    std::mt19937_64 rng(7ULL);
    std::normal_distribution<double> g(0.0, 1.0);
    Mat a(6, 6);
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j) a(i, j) = g(rng);
    const M6 lam = sym(a.transpose() * a) + 0.5 * M6::Identity();
    // 构造 Lambda 度量下特征值 0.3,0.4,2.0,0.5,0.6,0.7 的 J（rho=2>1）。
    Eigen::SelfAdjointEigenSolver<M6> es(lam);
    const M6 sq = es.eigenvectors() * es.eigenvalues().cwiseSqrt().asDiagonal() *
                  es.eigenvectors().transpose();
    const M6 sqi = es.eigenvectors() * es.eigenvalues().cwiseInverse().cwiseSqrt().asDiagonal() *
                   es.eigenvectors().transpose();
    Eigen::Matrix<double, 6, 1> wv;
    wv << 0.3, 0.4, 2.0, 0.5, 0.6, 0.7;
    const M6 J = sqi * wv.asDiagonal() * sq;
    int ncl = 0;
    double rraw = 0.0, rst = 0.0;
    const M6 Js = fls::contraction_clamp_6x6(J, lam, &ncl, &rraw, &rst);
    EXPECT_GE(ncl, 1);
    EXPECT_NEAR(rraw, 2.0, 1e-9);
    EXPECT_LE(rst, 1.0 + 1e-12);

    // 已收缩的 J 必须原样返回（只做相似变换，不做欧氏 SVD 钳位）。
    Eigen::Matrix<double, 6, 1> wv2 = 0.1 * Eigen::Matrix<double, 6, 1>::Ones();
    for (int i = 0; i < 6; ++i) wv2(i) = 0.1 + 0.1 * i;
    const M6 J2 = sqi * wv2.asDiagonal() * sq;
    const M6 J2s = fls::contraction_clamp_6x6(J2, lam, &ncl, &rraw, &rst);
    EXPECT_EQ(ncl, 0);
    EXPECT_LT((J2s - J2).norm(), 1e-10);
}

TEST(GaugeFixes, HandoffStableFactorContractive) {
    using namespace gaugefix_test;
    constexpr int ns = 3, m = 6 * ns;
    std::mt19937_64 rng(99ULL);
    std::normal_distribution<double> g(0.0, 1.0);
    Mat lam = Mat::Zero(m, m);
    for (int b = 0; b < ns; ++b) {
        Mat a(6, 6);
        for (int i = 0; i < 6; ++i)
            for (int j = 0; j < 6; ++j) a(i, j) = g(rng);
        lam.block<6, 6>(6 * b, 6 * b) = sym(a.transpose() * a) + 0.5 * M6::Identity();
    }
    Mat rr(12, m);
    for (int i = 0; i < 12; ++i)
        for (int j = 0; j < m; ++j) rr(i, j) = g(rng);
    const Mat H = sym(lam + 0.05 * (rr.transpose() * rr));
    M6 phi = M6::Identity();
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j) phi(i, j) += 0.05 * g(rng);
    const auto res = fls::local_handoff_blocks(H, lam, phi, 0);
    EXPECT_LE(res.j6_radius_stable, 1.0 + 1e-10);
    EXPECT_LT((res.J6_stable - res.J6).norm(), 1e-9);
    EXPECT_LT((res.T6_stable - phi * res.J6_stable).norm(), 1e-12);
}

TEST(GaugeProjection, Rank12WoodburyMatchesExplicit) {
    using namespace proj_test;
    const Problem p = make_problem();
    const Mat pi = projector(p.G, 0.0);
    EXPECT_LT((pi * pi - pi).norm(), 1e-10);
    EXPECT_LT((pi * p.G - p.G).norm(), 1e-10);
    EXPECT_LT((p.G.transpose() * pi - p.G.transpose()).norm(), 1e-10);

    const Mat h = projected_hessian(p.K, p.G, p.Pg, 0.0);
    const Terms t = lowrank_terms(p.K, p.G, p.Pg, 0.0);
    const Mat h_lr = sym(p.K + t.U * t.C * t.U.transpose());
    EXPECT_LT((h_lr - h).norm() / h.norm(), 1e-9);

    Eigen::JacobiSVD<Mat> svd(t.U);
    int rank = 0;
    const double smax = svd.singularValues().maxCoeff();
    for (int i = 0; i < svd.singularValues().size(); ++i)
        if (svd.singularValues()(i) > 1e-9 * smax) ++rank;
    EXPECT_EQ(rank, 12);

    Vec b(p.G.rows());
    std::mt19937_64 rng(11ULL);
    std::normal_distribution<double> g(0.0, 1.0);
    for (int i = 0; i < b.size(); ++i) b(i) = g(rng);
    const Vec xw = woodbury_solve(p.K, p.G, p.Pg, b, 0.0);
    const Vec xd = h.ldlt().solve(b);
    EXPECT_LT((xw - xd).norm() / xd.norm(), 1e-8);

    const double eps = 1e-9;
    const Mat h_eps = projected_hessian(p.K, p.G, p.Pg, eps);
    const Terms te = lowrank_terms(p.K, p.G, p.Pg, eps);
    const Mat h_lr_eps = sym(p.K + te.U * te.C * te.U.transpose());
    EXPECT_LT((h_lr_eps - h_eps).norm() / h_eps.norm(), 1e-9);
}

TEST(GaugeMarginalization, DenseMatchesWoodbury) {
    using namespace proj_test;
    const Problem p = make_window_problem();
    const int ncol = (int)p.G.rows();
    const int n0 = 6 * p.n;
    const int nr = ncol - n0;
    const Mat h = projected_hessian(p.K, p.G, p.Pg, 0.0);
    const Mat h00 = h.topLeftCorner(n0, n0);
    const Mat h01 = h.topRightCorner(n0, nr);
    const Mat h11 = h.bottomRightCorner(nr, nr);
    const Mat schur = sym(h11 - h01.transpose() * h00.llt().solve(h01));
    const Mat pm = sym(schur.inverse());
    const Mat h_inv = sym(h.inverse());
    EXPECT_LT((pm - h_inv.bottomRightCorner(nr, nr)).norm() / pm.norm(), 1e-8);

    const Mat g_rest = p.G.bottomRows(nr);
    const Mat pg_next = sym(g_rest.transpose() * pm * g_rest);
    Mat ur = Mat::Zero(ncol, 6);
    ur.bottomRows(nr) = g_rest;
    const Mat pg_direct = sym(ur.transpose() * h_inv * ur);
    EXPECT_LT((pg_next - pg_direct).norm() / pg_next.norm(), 1e-8);

    const Terms t = lowrank_terms(p.K, p.G, p.Pg, 0.0);
    const Mat kinv = p.K.llt().solve(Mat::Identity(ncol, ncol));
    const Mat h_inv_lr =
        sym(kinv - t.Ucorr * t.D.fullPivLu().solve(t.Ucorr.transpose()));
    const Mat pm_lr = sym(h_inv_lr.bottomRightCorner(nr, nr));
    const Mat pg_lr = sym(ur.transpose() * h_inv_lr * ur);
    EXPECT_LT((pm_lr - pm).norm() / pm.norm(), 1e-8);
    EXPECT_LT((pg_lr - pg_next).norm() / pg_next.norm(), 1e-8);
}

TEST(GaugeGtsSplit, GAndPerpIdentities) {
    using namespace proj_test;
    const Problem p = make_problem();
    const int ncol = (int)p.G.rows();
    const Mat s = p.G.transpose() * p.G;
    const Mat sinv = s.inverse();
    const Mat pi = p.G * sinv * p.G.transpose();

    Mat h(4, ncol);
    std::mt19937_64 rng(21ULL);
    std::normal_distribution<double> g(0.0, 1.0);
    for (int i = 0; i < h.rows(); ++i)
        for (int j = 0; j < h.cols(); ++j) h(i, j) = g(rng);

    const Mat hg = h * pi;
    const Mat hp = h - hg;
    EXPECT_LT((hg * p.G - h * p.G).norm() / (h * p.G).norm(), 1e-10);
    EXPECT_LT((hp * p.G).norm(), 1e-10);
    EXPECT_LT((hg * hp.transpose()).norm(), 1e-10);

    const double fro2 = h.squaredNorm();
    EXPECT_NEAR(fro2, hg.squaredNorm() + hp.squaredNorm(),
                1e-8 * std::max(1.0, fro2));

    const Mat lam = sinv * (p.G.transpose() * h.transpose() * h * p.G) * sinv;
    EXPECT_LT((hg.transpose() * hg - p.G * lam * p.G.transpose()).norm(), 1e-8);

    const Mat cross = hg.transpose() * hp + hp.transpose() * hg;
    const Mat lhs = h.transpose() * h;
    const Mat rhs = hp.transpose() * hp + p.G * lam * p.G.transpose() + cross;
    EXPECT_LT((lhs - rhs).norm() / lhs.norm(), 1e-8);
    EXPECT_GT(cross.norm(), 1e-8);
}

TEST(GaugePg, CovarianceAndInformationTransforms) {
    using namespace proj_test;
    const Problem p = make_problem();
    Mat t = Mat::Identity(6, 6);
    t(0, 1) = 0.10;
    t(1, 2) = -0.05;
    t(2, 2) = 1.20;
    t(3, 4) = 0.07;
    const Mat ti = t.inverse();
    const Mat g_new = p.G * t;

    const Mat pg_cov = sym(ti * p.Pg * ti.transpose());
    EXPECT_LT((g_new * pg_cov * g_new.transpose() -
               p.G * p.Pg * p.G.transpose()).norm() /
                  (p.G * p.Pg * p.G.transpose()).norm(),
              1e-9);

    const Mat pg_info = sym(t.transpose() * p.Pg * t);
    const Mat info_new = g_new * pg_info.llt().solve(g_new.transpose());
    const Mat info_old = p.G * p.Pg.llt().solve(p.G.transpose());
    EXPECT_LT((info_new - info_old).norm() / info_old.norm(), 1e-8);
}

TEST(FlsLocalHandoff, ThreeStarContractionPSD) {
    using namespace handoff_test;
    constexpr int ns = 3;
    constexpr int m = 6 * ns;
    std::mt19937_64 rng(20261023ULL);

    // 先验：块对角 SPD（跨星耦合通过观测 O 进入 joint 逆）。
    Eigen::MatrixXd Lam = Eigen::MatrixXd::Zero(m, m);
    for (int b = 0; b < ns; ++b) {
        const M6 Lb = random_spd6(rng, 0.5 + 0.25 * b);
        Lam.block<6, 6>(6 * b, 6 * b) = Lb;
    }

    // 观测信息：12 行随机白化 Jacobian → O = R^T R（rank 12, PSD）。
    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(12, m);
    std::normal_distribution<double> g(0.0, 1.0);
    for (int i = 0; i < 12; ++i)
        for (int j = 0; j < m; ++j) R(i, j) = g(rng);
    Eigen::MatrixXd O = 0.05 * (R.transpose() * R);
    Eigen::MatrixXd H = Lam + O;
    H = 0.5 * (H + H.transpose());

    // 良态 Φ（奇异值 ~1），保证 T6 不因坐标尺度塌缩。
    M6 Phi = M6::Identity();
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j) Phi(i, j) += 0.05 * g(rng);

    const auto res = fls::local_handoff_blocks(H, Lam, Phi, 0);

    // 1) 完整 joint 更新在 Λ_prior 度量下是收缩。
    EXPECT_LE(res.jnorm_metric, 1.0 + 1e-7);
    // 度量范数/谱半径才是收缩判据；物理欧氏范数可因相似变换 >1，不作门禁。
    EXPECT_GE(res.jnorm_euclid, 0.0);
    EXPECT_TRUE(res.contraction);
    EXPECT_LE(res.jradius, 1.0 + 1e-6);
    EXPECT_TRUE(res.contraction_radius);

    // 2) P+ ⪯ P-：Pminus-Pplus 半正定，最小特征值应 >=0（允许数值负）。
    EXPECT_GE(res.pplus_minus_pminus_min_eig, -1e-9);

    // 3) T6 / 完整 T_B 不塌缩。
    EXPECT_GT(res.tmin_sing, 1e-6);
    Eigen::JacobiSVD<Eigen::MatrixXd> svd_tb(res.T_B);
    EXPECT_GT(svd_tb.singularValues().minCoeff(), 1e-6);

    // 4) Qeff = K R K^T PSD。
    Eigen::SelfAdjointEigenSolver<M6> eq(res.Qeff6);
    EXPECT_GE(eq.eigenvalues().minCoeff(), -1e-9);

    // 与显式定义 Pplus_B Λ_prior_B 对拍（防 helper 内部错位）。
    Eigen::MatrixXd Pplus = H.inverse();
    Pplus = 0.5 * (Pplus + Pplus.transpose());
    const Eigen::MatrixXd Jref = Pplus * Lam;
    EXPECT_LT((res.J_B - Jref).norm(), 1e-10);
    EXPECT_LT((res.J6 - Jref.block<6, 6>(0, 0)).norm(), 1e-10);
    EXPECT_LT((res.Pplus6 - Pplus.block<6, 6>(0, 0)).norm(), 1e-10);

    // 完整 joint 的区间 Qeff_B=Ppred_B-T_B Pminus_B T_B^T
    // （=Q+Φ K R K^T Φ^T）PSD；Ppred_B 必须含过程 Q，不能用 Pplus 代替。
    Eigen::MatrixXd Pminus = Lam.inverse();
    Pminus = 0.5 * (Pminus + Pminus.transpose());
    Eigen::MatrixXd PhiB = Eigen::MatrixXd::Zero(m, m);
    for (int b = 0; b < ns; ++b) PhiB.block<6, 6>(6 * b, 6 * b) = Phi;
    const Eigen::MatrixXd TB = PhiB * Jref;
    const Eigen::MatrixXd Qproc = 0.01 * Eigen::MatrixXd::Identity(m, m);
    Eigen::MatrixXd PpredB = PhiB * Pplus * PhiB.transpose() + Qproc;
    PpredB = 0.5 * (PpredB + PpredB.transpose());
    Eigen::MatrixXd QeffB = PpredB - TB * Pminus * TB.transpose();
    QeffB = 0.5 * (QeffB + QeffB.transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eqb(QeffB);
    EXPECT_GE(eqb.eigenvalues().minCoeff(), -1e-9);
}
