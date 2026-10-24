// Householder 显式 gauge 投影 vs 生产 rank-2k 代数参考（header-only，小 n 验证用）。
//
// 生产口径（doc step 4）：
//   Pi_G0 = G0 S0^-1 G0^T,  S0 = G0^T G0 + eps_g I,
//   Lambda0_proj = (I-Pi_G0) Lambda0 (I-Pi_G0) = Lambda0 - dLam,
//   dLam = A0 S0^-1 G0^T + G0 S0^-1 A0^T - G0 S0^-1 B0 S0^-1 G0^T,
//   A0 = Lambda0 G0,  B0 = G0^T A0.
// solve/marginalize 不显式形成 dLam，而用 P=I-G S^-1 G^T 的低秩增广：
//   -P dLam P = -[Y,Z] E [Y,Z]^T,  Y=P A0, Z=P G0,
//   E = [[0, S0^-1], [S0^-1, -S0^-1 B0 S0^-1]],  C2=-E,
//   C2^-1 = [[-B0, -S0], [-S0, 0]]（用于 Woodbury 的 D22，见 fls_win.cpp）。
// 该式在实数运算下与上式恒等；唯一近似是 S0/S 的 eps 正则与浮点舍入。
//
// 本文件实现 doc 的 Householder 参考路径（小 n）：
//   1) G0 = Q R（Eigen HouseholderQR，Q 为 n×n 正交）；
//   2) Z = Q^T，使 Z Q0 = [I_k; 0]，Q0 = Q.leftCols(k)；
//   3) Lambda_tilde = Z Lambda0 Z^T；把 gauge 块（前 k 行/列）清零或替换；
//   4) Lambda_proj = Z^T Lambda_tilde Z。
// 与低秩减法不同，全程只有正交变换和分块赋值，不产生大数相消的负特征值。
//
// 注意 G 的坐标不是正交归一的。生产与 doc 的 P = G Pg G^T + P_perp
// 只在 G^T G = I 时成立；一般 G 下正确形式是
//   P = G S^-1 Pg_eff S^-1 G^T + P_perp,  G^T P_perp = 0,
//   G^T P G = Pg_eff,  S = G^T G + eps_g I.
// 本文件里的 gauge 块用 Q 坐标（h = R g）表示：
//   mode 0：gauge 块清零（纯投影，和 rank-2k 对拍）；
//   mode 1：gauge 块 = R Pg^-1 R^T，即 Q0 (R Pg^-1 R^T) Q0^T = G0 Pg^-1 G0^T；
//   mode 2：gauge 块 = (R Pg R^T)^-1，即 doc 第 3 节的字面写法（h 坐标先验）；
// mode 1 才是与生产 `G Pg^-1 G^T` 同一 g 坐标的口径，mode 2 不得再叠加一次窗口
// gauge 项。默认 mode 0 仅用于与 rank-2k 路径对拍。
#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace fls {

inline double gauge_gram_eps(const Eigen::MatrixXd &S) {
    const double tr = std::max(S.trace(), 0.0);
    double mind = 0.0;
    if (S.rows() > 0) {
        mind = S.diagonal().minCoeff();
        if (!(mind > 0.0)) mind = 0.0;
    }
    return std::max(std::max(mind * 1e-12, tr * 1e-18), 1e-300);
}

inline Eigen::MatrixXd symmetrize_md(const Eigen::MatrixXd &A) {
    return 0.5 * (A + A.transpose());
}

// 对称正定逆：只抬非正特征值，避免平移块 (~1e6) 的相对下限把旋转块 (~1e-8) 抬大。
inline Eigen::MatrixXd spd_inverse_md(const Eigen::MatrixXd &A) {
    const Eigen::MatrixXd S = symmetrize_md(A);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(S);
    if (es.info() != Eigen::Success)
        throw std::runtime_error("spd_inverse_md: 特征分解失败");
    Eigen::VectorXd w = es.eigenvalues();
    const double fl = std::max(w.maxCoeff() * 1e-14, 1e-300);
    for (int i = 0; i < w.size(); ++i)
        if (!(w(i) > 0.0)) w(i) = fl;
    const Eigen::MatrixXd V = es.eigenvectors();
    return V * w.cwiseInverse().asDiagonal() * V.transpose();
}

struct HouseholderProjection {
    int n = 0, k = 0;
    int gauge_block_mode = 0;
    Eigen::MatrixXd Q;              // n×n 正交（Q.leftCols(k)=Q0）
    Eigen::MatrixXd Q0;             // n×k
    Eigen::MatrixXd Qperp;          // n×(n-k)
    Eigen::MatrixXd R0;             // k×k，G0 = Q0 R0
    Eigen::MatrixXd Lambda0;        // n×n（节点 0 先验信息，块对角）
    Eigen::MatrixXd Lambda_proj;    // n×n，Z^T blkdiag(C_block, Lambda_perp) Z（不含 eps I）
    Eigen::MatrixXd Lambda_rank2k;  // n×n，Lambda0 - dLam（不含 eps I）
    Eigen::MatrixXd Lambda_expected;// rank2k + Q0 C_block Q0^T
    Eigen::MatrixXd gauge_term;     // Q0 C_block Q0^T
    double eps = 0.0;               // doc 的 eps = 1e-9 ||Lambda0||
    double eps_gram = 0.0;          // S0 = G0^T G0 + eps_gram I
    double err_rel = 0.0;           // ||proj-expected|| / max(||expected||, tiny)
    double gate_gauge = 0.0;        // ||Q0^T proj Q0 - C_block|| / ...
    double gate_perp = 0.0;         // 补空间信息不变（Qperp 坐标）
    double lmin_total = 0.0;        // lambda_min(proj + eps I + 生产 gauge 项)
    double qr_err = 0.0;            // ||Q0 R0 - G0|| / ||G0||
    // B0=G0^T Lambda0 G0 的条件诊断（同一 guard 口径：min_eig<=max(1e-12*max_eig,
    // 1e-300) 或 cond>1e12 视为近奇异）。status: 0=ok，1=近奇异，2=非有限/特征分解失败。
    int B0_status = 0;
    double B0_cond = -1.0;
    double B0_min_eig = 0.0;
    double B0_max_eig = 0.0;
};

// Lambda0: n×n 节点 0 先验信息；G0: n×k（列满秩）；Pg: k×k gauge 协方差（mode 1/2 必填）。
// mode 0 纯投影；mode 1 gauge 块 = R Pg^-1 R^T；mode 2 = (R Pg R^T)^-1。
inline HouseholderProjection householder_project_prior(
        const Eigen::MatrixXd &Lambda0_in, const Eigen::MatrixXd &G0_in,
        const Eigen::MatrixXd *Pg, int gauge_block_mode = 0,
        bool compute_rank2k = true) {
    if (Lambda0_in.rows() != Lambda0_in.cols() || G0_in.rows() != Lambda0_in.rows())
        throw std::runtime_error("householder_project_prior: Lambda0/G0 尺寸不匹配");
    if (gauge_block_mode < 0 || gauge_block_mode > 2)
        throw std::runtime_error("householder_project_prior: gauge_block_mode 仅支持 0/1/2");
    if (gauge_block_mode != 0 && Pg == nullptr)
        throw std::runtime_error("householder_project_prior: mode 1/2 需要 Pg");
    const int n = (int)Lambda0_in.rows();
    const int k = (int)G0_in.cols();
    if (k <= 0 || k > n)
        throw std::runtime_error("householder_project_prior: 需 0<k<=n");
    if (Pg != nullptr && (Pg->rows() != k || Pg->cols() != k))
        throw std::runtime_error("householder_project_prior: Pg 需 k×k");

    HouseholderProjection out;
    out.n = n; out.k = k; out.gauge_block_mode = gauge_block_mode;
    out.Lambda0 = symmetrize_md(Lambda0_in);
    const Eigen::MatrixXd G0 = G0_in;

    // 1) G0 = Q R。Eigen 的 HouseholderQR 对 n×k 给出 Q (n×n) 与 R (k×k)。
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(G0);
    out.Q = qr.householderQ() * Eigen::MatrixXd::Identity(n, n);
    out.R0 = Eigen::MatrixXd::Zero(k, k);
    for (int i = 0; i < k; ++i)
        for (int j = i; j < k; ++j) out.R0(i, j) = qr.matrixQR()(i, j);
    out.Q0 = out.Q.leftCols(k);
    out.Qperp = out.Q.rightCols(n - k);
    const double gnorm = std::max(G0.norm(), 1e-300);
    out.qr_err = (out.Q0 * out.R0 - G0).norm() / gnorm;
    if (out.qr_err > 1e-8)
        throw std::runtime_error("householder_project_prior: QR 重构误差过大");
    {
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(G0);
        const auto &sv = svd.singularValues();
        if (sv.size() == 0 || !(sv.minCoeff() > 1e-12 * sv.maxCoeff()))
            throw std::runtime_error("householder_project_prior: G0 列不满秩");
    }

    // 2) Z = Q^T 使 Z Q0 = [I_k;0]。Z 只在这里显式构造（小 n；生产不建 dense Z）。
    const Eigen::MatrixXd Z = out.Q.transpose();
    const Eigen::MatrixXd Lt = symmetrize_md(Z * out.Lambda0 * Z.transpose());

    // 3) gauge 块 C_block。
    Eigen::MatrixXd Cblock = Eigen::MatrixXd::Zero(k, k);
    Eigen::MatrixXd Cadded = Eigen::MatrixXd::Zero(k, k);   // 仅 mode 0 的 PD 门禁项
    if (gauge_block_mode != 0) {
        const Eigen::MatrixXd PgS = symmetrize_md(*Pg);
        if (gauge_block_mode == 1)
            Cblock = out.R0 * spd_inverse_md(PgS) * out.R0.transpose();
        else
            Cblock = spd_inverse_md(out.R0 * PgS * out.R0.transpose());
        Cblock = symmetrize_md(Cblock);
    } else if (Pg != nullptr) {
        // 生产模型在纯投影后另加 G Pg^-1 G^T；Q 坐标下其 gauge 块正是 R Pg^-1 R^T。
        Cadded = symmetrize_md(out.R0 * spd_inverse_md(symmetrize_md(*Pg)) *
                               out.R0.transpose());
    }

    Eigen::MatrixXd T = Lt;
    T.topRows(k).setZero();
    T.leftCols(k).setZero();
    if (gauge_block_mode != 0) T.topLeftCorner(k, k) = Cblock;
    out.Lambda_proj = symmetrize_md(Z.transpose() * T * Z);
    out.gauge_term = symmetrize_md(out.Q0 * Cblock * out.Q0.transpose());

    // 4) 生产 rank-2k 参考：Lambda0 - dLam（同一 eps_g 口径）。
    if (compute_rank2k) {
        const Eigen::MatrixXd A0 = out.Lambda0 * G0;
        const Eigen::MatrixXd B0 = symmetrize_md(G0.transpose() * A0);
        if (!B0.allFinite()) {
            out.B0_status = 2;
        } else {
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> esB(B0);
            if (esB.info() != Eigen::Success) {
                out.B0_status = 2;
            } else {
                const double bmax = esB.eigenvalues().maxCoeff();
                const double bmin = esB.eigenvalues().minCoeff();
                out.B0_max_eig = bmax;
                out.B0_min_eig = bmin;
                out.B0_cond = (bmax > 0.0 && bmin > 0.0)
                                  ? bmax / bmin
                                  : std::numeric_limits<double>::infinity();
                const double guard = std::max(1e-12 * std::max(bmax, 0.0), 1e-300);
                if (!(bmin > guard) || out.B0_cond > 1e12) out.B0_status = 1;
            }
        }
        Eigen::MatrixXd S0 = symmetrize_md(G0.transpose() * G0);
        out.eps_gram = gauge_gram_eps(S0);
        S0.diagonal().array() += out.eps_gram;
        const Eigen::MatrixXd Sinv = spd_inverse_md(S0);
        const Eigen::MatrixXd dLam =
            symmetrize_md(A0 * Sinv * G0.transpose() + G0 * Sinv * A0.transpose() -
                          G0 * Sinv * B0 * Sinv * G0.transpose());
        out.Lambda_rank2k = symmetrize_md(out.Lambda0 - dLam);
        out.Lambda_expected = symmetrize_md(
            out.Lambda_rank2k + out.Q0 * Cblock * out.Q0.transpose());
    } else {
        out.Lambda_rank2k = out.Lambda0;
        out.Lambda_expected = out.Lambda_proj;
    }

    // 5) doc 门禁：eps=1e-9||Lambda0||；gauge 块/补空间/正定。
    out.eps = 1e-9 * std::max(out.Lambda0.norm(), 1e-300);
    out.err_rel = (out.Lambda_proj - out.Lambda_expected).norm() /
                  std::max(out.Lambda_expected.norm(), 1e-300);
    const Eigen::MatrixXd Bq0 = symmetrize_md(out.Q0.transpose() * out.Lambda0 * out.Q0);
    const Eigen::MatrixXd Bqp = symmetrize_md(out.Q0.transpose() * out.Lambda_proj * out.Q0);
    out.gate_gauge = (Bqp - Cblock).norm() /
                     std::max(std::max(Bq0.norm(), Cblock.norm()), 1e-300);
    if (n > k) {
        const Eigen::MatrixXd Dperp =
            symmetrize_md(out.Qperp.transpose() * (out.Lambda_proj - out.Lambda0) * out.Qperp);
        const Eigen::MatrixXd Rperp =
            symmetrize_md(out.Qperp.transpose() * out.Lambda0 * out.Qperp);
        out.gate_perp = Dperp.norm() / std::max(Rperp.norm(), 1e-300);
    } else {
        out.gate_perp = 0.0;
    }
    const Eigen::MatrixXd total =
        out.Lambda_proj + out.eps * Eigen::MatrixXd::Identity(n, n) +
        out.Q0 * Cadded * out.Q0.transpose();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> est(total);
    out.lmin_total = est.info() == Eigen::Success ? est.eigenvalues().minCoeff()
                                                  : -std::numeric_limits<double>::infinity();
    return out;
}

}  // namespace fls
