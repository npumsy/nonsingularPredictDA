// L=1 EKF 局部 joint 交接的稠密小规模数学核（header-only，供 fls_win.cpp 与 gtest 共用）。
//
// 记局部球 B 内的先验信息 Lambda_B（m×m，m=6|B|）、观测信息 O_B=H_B^T R^-1 H_B。
// 后验信息 A_B = Lambda_B + O_B，后验协方差 Pplus_B = A_B^-1。
// 误差更新（相对先验误差）为
//     J_B = I - K_B H_B = Pplus_B * Lambda_B,
// 其中 K_B = Pplus_B H_B^T R^-1。J_B 在 Lambda_B 度量下是收缩：
//     Lambda_B^{1/2} J_B Lambda_B^{-1/2}
// 对称且特征值在 (0,1]；本文件返回完整 J_B 供收缩门禁与诊断。
//
// 生产 T 链不能用 J_B 的 6×6 主子块 J6 直接当 per-star 因子：全 J_B 的
// 谱半径门禁不保证 J6 在 Λ6 度量下收缩（先验块非严格对角、缩放相似或
// 舍入时 rho(J6) 可 >1），而 T 链要求目标星因子稳定。
// 这里改用 Schur-consistent per-star 因子
//     J6_marg = Pplus6 * Lambda6,  Lambda6 = Lambda_B[0:6,0:6],
// 并在 Lambda6 度量下做对称谱截断：
//     A = Lambda6^{1/2} J6_marg Lambda6^{-1/2}（理论上对称 PSD 且 <=I）,
//     A_clamped = V diag(clamp(lambda,0,1)) V^T,
//     J6_stable = Lambda6^{-1/2} A_clamped Lambda6^{1/2},
//     T6_stable = Phi6 J6_stable.
// 该因子只依赖目标星的边缘先验/后验（交叉星项被 Schur 消去），故与
// `Pplus6 Lambda6` 的 Kalman 边缘口径一致；谱截断只吸收数值越界，不是
// 用 2-范数钳位改写 T。完整 block-T = Phi_B J_B 仍由 T_B 返回，供小窗验证；
// 注：T_B 把传入的目标星 Phi6 广播到球内每颗星（调用方只给目标星 Phi）；
// 它用于验证 J_B 的 block 结构，不是生产逐星链。严格 block-T 需把每颗邻星
// 的 Phi_i 一并传入，当前 API 未提供（见 ekf_block_T_host 的限制说明）。
//
// 生产逐星链使用 J6_stable/T6_stable。若 j6_clamped>0，Qeff6 仍按完整 J_B
// 的 KRK^T 给出，6×6 恒等式 Pplus6 = J6_stable Pminus6 J6_stable^T + Qeff6
// 只是近似；P 链仍用精确局部后验 Pplus6，生产 Qeff 走 EKF 区间因子，
// 不依赖该恒等式。T/P 链的最终门禁仍是 Qeff 正定与全代价线搜索。
#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fls {

struct LocalHandoffBlocks {
    Eigen::MatrixXd J_B;                       // m×m 完整 joint 更新
    Eigen::MatrixXd T_B;                       // Phi_B * J_B（块对角 Phi）
    Eigen::Matrix<double, 6, 6> J6;            // 目标星对角块（诊断；可能 rho>1）
    Eigen::Matrix<double, 6, 6> T6;            // Phi6 * J6（诊断）
    Eigen::Matrix<double, 6, 6> J6_stable;     // Schur-consistent per-star（生产 T 链）
    Eigen::Matrix<double, 6, 6> T6_stable;     // Phi6 * J6_stable
    int j6_clamped = 0;                        // 被谱截断的特征值个数（0=原样）
    double j6_radius_principal = 0.0;          // max|eig(J_B[0:6,0:6])|（诊断）
    double j6_radius_raw = 0.0;                // max|eig(J6_marg)|（截断前）
    double j6_radius_stable = 0.0;             // max|eig(J6_stable)| <= 1
    Eigen::Matrix<double, 6, 6> Pplus6;        // 后验协方差目标块
    Eigen::Matrix<double, 6, 6> Pminus6;       // 先验协方差目标块
    Eigen::Matrix<double, 6, 6> Qeff6;         // 完整 joint 的 K R K^T 目标块（⪰0）
    double jnorm_metric = 0.0;                 // ||Lambda_B^{1/2} J_B Lambda_B^{-1/2}||_2
    double jnorm_euclid = 0.0;                 // ||J_B||_2（物理欧氏，非保证 ≤1）
    double jradius = 0.0;                      // max |eig(J_B)|（相似变换不变）
    double tmin_sing = 0.0;                    // T6 最小奇异值
    double pplus_minus_pminus_min_eig = 0.0;   // lambda_min(Pminus - Pplus)
    bool contraction = false;                  // 度量范数门禁（严格）
    bool contraction_radius = false;           // 谱半径门禁（生产用，避免坐标相似放大）
};

// 6×6 的 Lambda 度量谱截断：A=Lambda^{1/2} J Lambda^{-1/2} 理论上对称 PSD 且 <=I；
// 只把数值越界的特征值截到 [0,1]，再变回 J。若 J 已收缩（常见情形），返回原 J。
inline Eigen::Matrix<double, 6, 6> contraction_clamp_6x6(
        const Eigen::Matrix<double, 6, 6> &J,
        const Eigen::Matrix<double, 6, 6> &Lambda,
        int *n_clamped = nullptr, double *radius_raw = nullptr,
        double *radius_stable = nullptr) {
    using M6 = Eigen::Matrix<double, 6, 6>;
    const M6 L0 = 0.5 * (Lambda + Lambda.transpose());
    Eigen::SelfAdjointEigenSolver<M6> esL;
    double jit = std::max(std::max(L0.trace(), 0.0) * 1e-14, 1e-300);
    for (int t = 0; t < 4; ++t) {
        esL.compute(L0 + (t == 0 ? 0.0 : jit) * M6::Identity());
        if (esL.info() == Eigen::Success && esL.eigenvalues().minCoeff() > 0.0) break;
        jit *= 1e3;
    }
    if (esL.info() != Eigen::Success || !(esL.eigenvalues().minCoeff() > 0.0))
        throw std::runtime_error("contraction_clamp_6x6: Lambda6 非正定");
    const Eigen::Matrix<double, 6, 1> wl = esL.eigenvalues();
    const M6 Vl = esL.eigenvectors();
    const M6 sq = Vl * wl.cwiseSqrt().asDiagonal() * Vl.transpose();
    const M6 sqi = Vl * wl.cwiseInverse().cwiseSqrt().asDiagonal() * Vl.transpose();
    M6 A = sq * J * sqi;
    A = 0.5 * (A + A.transpose());
    Eigen::SelfAdjointEigenSolver<M6> esA(A);
    if (esA.info() != Eigen::Success)
        throw std::runtime_error("contraction_clamp_6x6: A 特征分解失败");
    Eigen::Matrix<double, 6, 1> wa = esA.eigenvalues();
    int nc = 0;
    const double rraw = wa.cwiseAbs().maxCoeff();
    for (int i = 0; i < 6; ++i) {
        if (wa(i) > 1.0) { wa(i) = 1.0; ++nc; }
        else if (wa(i) < 0.0) { wa(i) = 0.0; ++nc; }
    }
    const M6 Ac = esA.eigenvectors() * wa.asDiagonal() * esA.eigenvectors().transpose();
    const M6 Js = sqi * Ac * sq;
    if (n_clamped) *n_clamped = nc;
    if (radius_raw) *radius_raw = rraw;
    if (radius_stable) *radius_stable = wa.cwiseAbs().maxCoeff();
    return Js;
}

// 快速收缩门禁：A = R Pplus R^T（Lam=R^T R，R 为 Cholesky 上三角），
// 理论上 0<=eig(A)<=1。用 LDLT 的 D 对角检查惯性，正常情形无需特征分解；
// 只有检查失败（数值越界）时调用方再走 contraction_clamp_6x6 的谱截断。
inline bool j6_metric_within_unit(const Eigen::Matrix<double, 6, 6> &Pplus,
                                  const Eigen::Matrix<double, 6, 6> &Lam,
                                  double tol = 1e-7) {
    using M6 = Eigen::Matrix<double, 6, 6>;
    const M6 L0 = 0.5 * (Lam + Lam.transpose());
    Eigen::LLT<M6> llt(L0);
    if (llt.info() != Eigen::Success) return false;
    const M6 R = llt.matrixL().transpose();
    M6 A = R * Pplus * R.transpose();
    A = 0.5 * (A + A.transpose());
    Eigen::LDLT<M6> ld(A);
    if (ld.info() != Eigen::Success) return false;
    const Eigen::Matrix<double, 6, 1> D = ld.vectorD();
    return (D.minCoeff() >= -tol) && (D.maxCoeff() <= 1.0 + tol);
}

// 生产快路径：只求目标星 6×6 边缘，不做 m×m 逆/特征分解/SVD。
// Λ_B 是块对角先验，故 J_B[0:6,0:6]=Pplus6 Λ6；Pplus6 由 LLT 解 6 个 RHS 得到。
struct LocalHandoff6 {
    Eigen::Matrix<double, 6, 6> Pplus6;
    Eigen::Matrix<double, 6, 6> J6_stable;
    Eigen::Matrix<double, 6, 6> T6_stable;
    int j6_clamped = 0;
    double j6_radius_principal = 0.0;
    double j6_radius_raw = 0.0;
    double j6_radius_stable = 0.0;
    double jnorm = 0.0;
    double tmin_sing = 0.0;
};

inline LocalHandoff6 local_handoff_target6(const Eigen::MatrixXd &H_B,
                                           const Eigen::MatrixXd &Lam_B,
                                           const Eigen::Matrix<double, 6, 6> &Phi6,
                                           int target_block = 0) {
    using M6 = Eigen::Matrix<double, 6, 6>;
    const int m = (int)H_B.rows();
    if (m <= 0 || H_B.cols() != m || Lam_B.rows() != m || Lam_B.cols() != m)
        throw std::runtime_error("local_handoff_target6: H_B/Lam_B 需同阶方阵");
    if (m % 6 != 0)
        throw std::runtime_error("local_handoff_target6: m 必须为 6 的倍数");
    const int b0 = 6 * target_block;
    if (target_block < 0 || b0 + 6 > m)
        throw std::runtime_error("local_handoff_target6: target_block 越界");
    LocalHandoff6 out;
    const Eigen::MatrixXd Hb = 0.5 * (H_B + H_B.transpose());
    const Eigen::MatrixXd Lb = 0.5 * (Lam_B + Lam_B.transpose());
    const double trh = std::max(Hb.trace(), 0.0);
    double jit_h = std::max(trh * 1e-14, 1e-300);
    Eigen::LLT<Eigen::MatrixXd> llt_h;
    for (int try_i = 0; try_i < 4; ++try_i) {
        llt_h.compute(Hb + jit_h * Eigen::MatrixXd::Identity(m, m));
        if (llt_h.info() == Eigen::Success) break;
        jit_h *= 1e3;
    }
    if (llt_h.info() != Eigen::Success)
        throw std::runtime_error("local_handoff_target6: H_B 非正定");
    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(m, 6);
    R.block(b0, 0, 6, 6).setIdentity();
    Eigen::MatrixXd X = llt_h.solve(R);
    M6 Pplus6 = X.block(b0, 0, 6, 6);
    Pplus6 = 0.5 * (Pplus6 + Pplus6.transpose());
    const M6 Lam6 = 0.5 * (Lb.block(b0, b0, 6, 6) + Lb.block(b0, b0, 6, 6).transpose());
    const M6 J6_raw = Pplus6 * Lam6;
    Eigen::EigenSolver<M6> ej6(J6_raw, false);
    if (ej6.info() == Eigen::Success)
        out.j6_radius_principal = ej6.eigenvalues().cwiseAbs().maxCoeff();
    out.J6_stable = contraction_clamp_6x6(J6_raw, Lam6, &out.j6_clamped,
                                          &out.j6_radius_raw, &out.j6_radius_stable);
    out.Pplus6 = Pplus6;
    out.T6_stable = Phi6 * out.J6_stable;
    Eigen::JacobiSVD<M6> sj(out.J6_stable);
    Eigen::JacobiSVD<M6> st(out.T6_stable);
    out.jnorm = sj.singularValues()(0);
    out.tmin_sing = st.singularValues().minCoeff();
    return out;
}

inline LocalHandoffBlocks local_handoff_blocks(const Eigen::MatrixXd &H_B,
                                               const Eigen::MatrixXd &Lam_B,
                                               const Eigen::Matrix<double, 6, 6> &Phi6,
                                               int target_block = 0) {
    const int m = (int)H_B.rows();
    if (m <= 0 || H_B.cols() != m || Lam_B.rows() != m || Lam_B.cols() != m)
        throw std::runtime_error("local_handoff_blocks: H_B/Lam_B 需同阶方阵");
    if (m % 6 != 0)
        throw std::runtime_error("local_handoff_blocks: m 必须为 6 的倍数");
    if (target_block < 0 || 6 * target_block + 6 > m)
        throw std::runtime_error("local_handoff_blocks: target_block 越界");

    LocalHandoffBlocks out;
    Eigen::MatrixXd Hb = 0.5 * (H_B + H_B.transpose());
    Eigen::MatrixXd Lb = 0.5 * (Lam_B + Lam_B.transpose());

    const double trh = std::max(Hb.trace(), 0.0);
    const double trl = std::max(Lb.trace(), 0.0);
    double jit_h = std::max(trh * 1e-14, 1e-300);
    double jit_l = std::max(trl * 1e-14, 1e-300);
    Eigen::LLT<Eigen::MatrixXd> llt_h;
    Eigen::LLT<Eigen::MatrixXd> llt_l;
    for (int try_i = 0; try_i < 4; ++try_i) {
        llt_h.compute(Hb + jit_h * Eigen::MatrixXd::Identity(m, m));
        llt_l.compute(Lb + jit_l * Eigen::MatrixXd::Identity(m, m));
        if (llt_h.info() == Eigen::Success && llt_l.info() == Eigen::Success) break;
        jit_h *= 1e3;
        jit_l *= 1e3;
    }
    if (llt_h.info() != Eigen::Success || llt_l.info() != Eigen::Success)
        throw std::runtime_error("local_handoff_blocks: H_B 或 Lambda_B 非正定");

    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(m, m);
    Eigen::MatrixXd Pplus = llt_h.solve(I);
    Pplus = 0.5 * (Pplus + Pplus.transpose());
    Eigen::MatrixXd Pminus = llt_l.solve(I);
    Pminus = 0.5 * (Pminus + Pminus.transpose());

    out.J_B = Pplus * Lb;   // 保留非对称性：J_B 只在 Lambda_B 度量下收缩

    // 收缩度量：Lambda_B^{1/2} J_B Lambda_B^{-1/2}（对称平方根）。
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Lb);
    if (es.info() != Eigen::Success)
        throw std::runtime_error("local_handoff_blocks: Lambda_B 特征分解失败");
    Eigen::VectorXd wl = es.eigenvalues();
    if (!(wl.minCoeff() > 0.0))
        throw std::runtime_error("local_handoff_blocks: Lambda_B 非正定");
    const Eigen::MatrixXd Vl = es.eigenvectors();
    Eigen::MatrixXd sq = Vl * wl.cwiseSqrt().asDiagonal() * Vl.transpose();
    Eigen::MatrixXd sqi = Vl * wl.cwiseInverse().cwiseSqrt().asDiagonal() * Vl.transpose();
    Eigen::MatrixXd Jm = sq * out.J_B * sqi;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd_jm(Jm);
    Eigen::JacobiSVD<Eigen::MatrixXd> svd_j(out.J_B);
    out.jnorm_metric = svd_jm.singularValues()(0);
    out.jnorm_euclid = svd_j.singularValues()(0);
    out.contraction = (out.jnorm_metric <= 1.0 + 1e-8);
    Eigen::EigenSolver<Eigen::MatrixXd> ej(out.J_B, false);
    if (ej.info() == Eigen::Success)
        out.jradius = ej.eigenvalues().cwiseAbs().maxCoeff();
    // J_B 与对称收缩矩阵相似，谱半径必然 <=1；生产用谱半径门禁，
    // 避免 scaled/physical 相似变换把度量范数放大造成误报。
    out.contraction_radius = (out.jradius <= 1.0 + 1e-6);

    const int b0 = 6 * target_block;
    out.J6 = out.J_B.block(b0, b0, 6, 6);
    out.Pplus6 = Pplus.block(b0, b0, 6, 6);
    out.Pminus6 = Pminus.block(b0, b0, 6, 6);
    out.Pplus6 = 0.5 * (out.Pplus6 + out.Pplus6.transpose());
    out.Pminus6 = 0.5 * (out.Pminus6 + out.Pminus6.transpose());
    out.T6 = Phi6 * out.J6;
    // Schur-consistent per-star 因子：只用目标星边缘块，且在 Lambda6 度量下
    // 强制收缩。J6 与 J6_stable 在精确块对角先验下相同；差异仅为数值截断。
    using M6 = Eigen::Matrix<double, 6, 6>;
    const M6 Lam6 = 0.5 * (Lb.block<6, 6>(b0, b0) + Lb.block<6, 6>(b0, b0).transpose());
    {
        Eigen::EigenSolver<M6> ej6(out.J6, false);
        if (ej6.info() == Eigen::Success)
            out.j6_radius_principal = ej6.eigenvalues().cwiseAbs().maxCoeff();
    }
    out.J6_stable = contraction_clamp_6x6(out.Pplus6 * Lam6, Lam6,
                                         &out.j6_clamped, &out.j6_radius_raw,
                                         &out.j6_radius_stable);
    out.T6_stable = Phi6 * out.J6_stable;

    Eigen::MatrixXd PhiB = Eigen::MatrixXd::Zero(m, m);
    for (int k = 0; k < m / 6; ++k)
        PhiB.block(6 * k, 6 * k, 6, 6) = Phi6;
    out.T_B = PhiB * out.J_B;

    // 局部更新噪声 = Pplus_B - J_B Pminus_B J_B^T = K R K^T（完整 joint，PSD）。
    // 取目标星 6×6 对角块（PSD 矩阵的主子块仍 PSD），不是只用 6×6 先验块相减。
    Eigen::MatrixXd QB = Pplus - out.J_B * Pminus * out.J_B.transpose();
    QB = 0.5 * (QB + QB.transpose());
    out.Qeff6 = QB.block(b0, b0, 6, 6);
    out.Qeff6 = 0.5 * (out.Qeff6 + out.Qeff6.transpose());

    Eigen::JacobiSVD<Eigen::Matrix<double, 6, 6>> svd_t6(out.T6);
    out.tmin_sing = svd_t6.singularValues().minCoeff();

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> es_diff(
        out.Pminus6 - out.Pplus6);
    out.pplus_minus_pminus_min_eig = es_diff.eigenvalues().minCoeff();
    return out;
}

// 对称化 + 非正特征值翻转（仅用于协方差的数值 PSD 保护）。
// 负特征值按其自身量级翻正（而不是用全局 max 的相对下限），避免 QOE
// 混合量纲下把速度/旋转小块抬到位置块量级；正特征值一律不动。
inline Eigen::Matrix<double, 6, 6> psd_floor_6x6(const Eigen::Matrix<double, 6, 6> &A,
                                                 double abs_floor = 1e-300) {
    Eigen::Matrix<double, 6, 6> S = 0.5 * (A + A.transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> es(S);
    Eigen::Matrix<double, 6, 1> w = es.eigenvalues();
    for (int i = 0; i < 6; ++i)
        if (!(w(i) > 0.0))
            w(i) = std::max(std::abs(w(i)), std::max(abs_floor, 1e-300));
    return es.eigenvectors() * w.asDiagonal() * es.eigenvectors().transpose();
}

}  // namespace fls
