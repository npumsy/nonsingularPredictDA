// 窗口 gauge 块诊断（D1）的 header-only 数学核。
//
// 生产算子（物理 QOE 坐标）：
//   S   = G^T G + eps_g I,
//   Pi  = G S^-1 G^T,
//   Lam = (I-Pi) K (I-Pi) + G Pg_eff^-1 G^T,
//   K   = A^T A + eps_s S_c^-2.
// 本文件只算 Lam 在 range(G) 上的 k×k 块，不构造 ncol×ncol 矩阵、不做 6N 稠密求逆：
//   记 T = eps_g S^-1，则 (I-Pi)G = G T（精确到 eps_g 正则），于是
//   G^T Lam G = T^T (G^T K G) T + (G^T G) Pg_eff^-1 (G^T G).
// 其中 G^T K G 只由调用方传入的 KG=K G 得到（O(ncol k^2)）。
//
// Q 坐标：G=Q R（thin Householder QR），Q^T Q=I_k。
//   Q^T Lam Q = R^-T (G^T Lam G) R^-1,
//   参考信息 = R Pg_eff^-1 R^T（Q 系数坐标下的先验信息）。
// 这两者的相对差就是 D1 的 identity/condition 判据。
//
// 注意：这里只给 k 维 gauge 块的最小/最大特征值，不是整窗 6N 维 Lam 的
// lambda_min。补空间 (I-Pi)K(I-Pi) 的理论正定性不在本文件数值验证范围内；
// 严格的 lambda_min(Lam)>0 需要单独做逆迭代/Lanczos。
#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace fls {

struct GaugeLamWinBlock {
    int k = 0;
    int ncol = 0;
    double eps_g = 0.0;
    bool ok = false;
    bool qr_rank_ok = false;
    std::string reason;

    Eigen::MatrixXd GtKG;          // G^T K G (k,k)
    Eigen::MatrixXd Sg;            // G^T G (k,k)
    Eigen::MatrixXd GtLamG;        // T^T GtKG T + Sg Pg_eff^-1 Sg
    Eigen::MatrixXd Q;             // G=Q R 的 thin Q (ncol,k)
    Eigen::MatrixXd R;             // k×k upper
    Eigen::MatrixXd QtLamQ;        // R^-T GtLamG R^-1
    Eigen::MatrixXd Pg_eff_inv_Q;  // R Pg_eff^-1 R^T

    double min_eig = 0.0;
    double max_eig = 0.0;
    double cond = std::numeric_limits<double>::infinity();
    double min_eig_Q = std::numeric_limits<double>::quiet_NaN();
    double max_eig_Q = std::numeric_limits<double>::quiet_NaN();
    double cond_Q = std::numeric_limits<double>::quiet_NaN();
    double identity_abs = std::numeric_limits<double>::quiet_NaN();
    double identity_rel = std::numeric_limits<double>::quiet_NaN();
    double R_diag_min = 0.0;
    double R_diag_max = 0.0;
};

inline Eigen::MatrixXd gauge_lamwin_sym(const Eigen::MatrixXd &A) {
    return 0.5 * (A + A.transpose());
}

inline double gauge_lamwin_cond(const Eigen::MatrixXd &A, double *lmin, double *lmax) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(gauge_lamwin_sym(A));
    if (es.info() != Eigen::Success) {
        if (lmin) *lmin = std::numeric_limits<double>::quiet_NaN();
        if (lmax) *lmax = std::numeric_limits<double>::quiet_NaN();
        return std::numeric_limits<double>::infinity();
    }
    const double wmin = es.eigenvalues().minCoeff();
    const double wmax = es.eigenvalues().maxCoeff();
    if (lmin) *lmin = wmin;
    if (lmax) *lmax = wmax;
    return (wmin > 0.0) ? wmax / wmin : std::numeric_limits<double>::infinity();
}

// KG = K G（物理 QOE，ncol×k）；G 为物理列（不要求正交）；eps_g 与生产
// S=G^T G+eps_g I 的 eps 完全一致（由调用方用同一 gauge_eps_from_gram 计算）。
inline GaugeLamWinBlock gauge_lamwin_block(const Eigen::MatrixXd &KG,
                                           const Eigen::MatrixXd &G,
                                           const Eigen::MatrixXd &Pg_eff_inv,
                                           double eps_g) {
    GaugeLamWinBlock out;
    const int64_t nc = G.rows();
    const int k = (int)G.cols();
    out.k = k;
    out.ncol = (int)nc;
    out.eps_g = eps_g;
    if (nc <= 0 || k <= 0) {
        out.reason = "empty_G";
        return out;
    }
    if (KG.rows() != nc || KG.cols() != k) {
        out.reason = "KG_shape_mismatch";
        return out;
    }
    if (Pg_eff_inv.rows() != k || Pg_eff_inv.cols() != k) {
        out.reason = "Pg_shape_mismatch";
        return out;
    }
    if (!G.allFinite() || !KG.allFinite() || !Pg_eff_inv.allFinite()) {
        out.reason = "non_finite_input";
        return out;
    }

    out.Sg = gauge_lamwin_sym(G.transpose() * G);
    const Eigen::MatrixXd S =
        out.Sg + eps_g * Eigen::MatrixXd::Identity(k, k);
    Eigen::LLT<Eigen::MatrixXd> llt(S);
    if (llt.info() != Eigen::Success) {
        out.reason = "S_not_pd";
        return out;
    }
    const Eigen::MatrixXd Sinv = llt.solve(Eigen::MatrixXd::Identity(k, k));
    // (I-Pi)G = G - G S^-1 Sg = G S^-1 (S-Sg) = eps_g G S^-1.
    const Eigen::MatrixXd T = eps_g * Sinv;
    out.GtKG = gauge_lamwin_sym(G.transpose() * KG);
    const Eigen::MatrixXd Pginv = gauge_lamwin_sym(Pg_eff_inv);
    out.GtLamG = gauge_lamwin_sym(T.transpose() * out.GtKG * T + out.Sg * Pginv * out.Sg);

    out.cond = gauge_lamwin_cond(out.GtLamG, &out.min_eig, &out.max_eig);

    Eigen::HouseholderQR<Eigen::MatrixXd> qr(G);
    out.R = qr.matrixQR().topLeftCorner(k, k).triangularView<Eigen::Upper>();
    out.Q = qr.householderQ() * Eigen::MatrixXd::Identity(nc, k);
    out.R_diag_min = out.R.diagonal().cwiseAbs().minCoeff();
    out.R_diag_max = out.R.diagonal().cwiseAbs().maxCoeff();
    const double rfloor =
        std::max(out.R_diag_max * 1e-14, std::numeric_limits<double>::min());
    out.qr_rank_ok = (out.R_diag_min > rfloor) && out.Q.allFinite();
    if (out.qr_rank_ok) {
        const Eigen::MatrixXd Rinv =
            out.R.triangularView<Eigen::Upper>()
                .solve(Eigen::MatrixXd::Identity(k, k));
        out.QtLamQ = gauge_lamwin_sym(Rinv.transpose() * out.GtLamG * Rinv);
        out.Pg_eff_inv_Q = gauge_lamwin_sym(out.R * Pginv * out.R.transpose());
        out.cond_Q = gauge_lamwin_cond(out.QtLamQ, &out.min_eig_Q, &out.max_eig_Q);
        out.identity_abs = (out.QtLamQ - out.Pg_eff_inv_Q).norm();
        out.identity_rel =
            out.identity_abs /
            std::max(out.Pg_eff_inv_Q.norm(), std::numeric_limits<double>::min());
    } else {
        out.reason = "G_rank_deficient_for_QR";
    }
    out.ok = true;
    return out;
}

}  // namespace fls
