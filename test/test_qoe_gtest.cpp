// qoe 的 gtest：坐标/要素变换等价性 + 长期平均要素传播 + 三体摄动。
// 覆盖：rv<->非奇异要素、非奇异要素<->平均要素（Eckstein-Ustinov）、长期传播(J2/J3/J4)、
// 与笛卡尔 DA 的一致性、三体开关。单位一律 SI（m, m/s；要素 a 为 m，角为 rad）。
#include <gtest/gtest.h>
#include <Eigen/Core>
#include <vector>
#include <cmath>
#include <random>
#include "kepler.h"
#include "dastates.h"

using Eigen::VectorXd;

namespace {

// 由非奇异要素 (a, u=M+w, ex, ey, i, Om) 造 rv（m）。
VectorXd rv_from_oe(const VectorXd &oe, double a, double u, double ex, double ey,
                    double inc, double Om) {
    VectorXd o(6); o << a, u, ex, ey, inc, Om;
    return osculating::OEOsc2rv(o, 100, 1e-9);
}

std::vector<VectorXd> sample_oe() {
    std::vector<VectorXd> v;
    VectorXd a(6);
    a << 6.93e6, 0.5, 0.0, 0.0, 1.0, 0.3;                 // 近圆、倾斜
    v.push_back(a);
    a << 7.10e6, 1.2, 0.10 * std::cos(0.7), 0.10 * std::sin(0.7), 0.9, 2.1;  // 偏心
    v.push_back(a);
    a << 6.90e6, 2.5, 0.0, 0.0, 1e-6, 0.0;                // 近赤道、近圆（奇异规避）
    v.push_back(a);
    a << 7.00e6, 0.9, 0.02 * std::cos(2.3), 0.02 * std::sin(2.3), 1.57079632679, 5.5; // 近极轨
    v.push_back(a);
    return v;
}

}  // namespace

// 快速 qoe→ECI 位置 `oe2eciPos` 与 `OEOsc2rv` 的位置**逐位一致**（测距在 ECI，必须严格一致）
TEST(Transform, QOE2ECI_Position_Exact) {
    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> ua(6.8e6, 7.5e6), ue(0.0, 0.3),
        uang(-M_PI, M_PI), ui(1e-4, M_PI - 1e-4);
    for (int t = 0; t < 2000; ++t) {
        VectorXd oe(6);
        double ex = ue(rng) * std::cos(uang(rng)), ey = ue(rng) * std::sin(uang(rng));
        oe << ua(rng), uang(rng), ex, ey, ui(rng), uang(rng);
        VectorXd rv = osculating::OEOsc2rv(oe, 100, 1e-12);
        Eigen::Vector3d p_fast = osculating::oe2eciPos(oe, 100, 1e-12);
        EXPECT_LT((p_fast - rv.head<3>()).norm() / rv.head<3>().norm(), 1e-13);
    }
    // 边界：圆轨道、赤道、极轨
    for (double e : {0.0, 0.2}) for (double inc : {0.0, M_PI / 2, M_PI - 1e-6}) {
        VectorXd oe(6); oe << 7.0e6, 1.0, e, 0.0, inc, 0.5;
        VectorXd rv = osculating::OEOsc2rv(oe, 200, 1e-13);
        Eigen::Vector3d p = osculating::oe2eciPos(oe, 200, 1e-13);
        EXPECT_TRUE(p.allFinite());
        EXPECT_LT((p - rv.head<3>()).norm() / rv.head<3>().norm(), 1e-13);
    }
}

// qoe→ECI 位置与**独立**经典公式 R3(Ω)R1(i)R3(ω)·[a(cosE−e), b sinE, 0] 一致（证明用的是既有公式）
TEST(Transform, QOE2ECI_Position_Matches_Classical) {
    for (const auto &oe : sample_oe()) {
        double a = oe(0), u = oe(1), ex = oe(2), ey = oe(3), inc = oe(4), Om = oe(5);
        double e = std::hypot(ex, ey), w = std::atan2(ey, ex), M = u - w;
        double E = M; for (int k = 0; k < 60; ++k) E = E - (E - e * std::sin(E) - M) / (1 - e * std::cos(E));
        double b = a * std::sqrt(1 - e * e);
        Eigen::Vector3d rp(a * (std::cos(E) - e), b * std::sin(E), 0.0);
        Eigen::Matrix3d RzOm, Rx, Rzw;
        RzOm << std::cos(Om), -std::sin(Om), 0, std::sin(Om), std::cos(Om), 0, 0, 0, 1;
        Rx << 1, 0, 0, 0, std::cos(inc), -std::sin(inc), 0, std::sin(inc), std::cos(inc);
        Rzw << std::cos(w), -std::sin(w), 0, std::sin(w), std::cos(w), 0, 0, 0, 1;
        Eigen::Vector3d p = RzOm * Rx * Rzw * rp;
        VectorXd rv = osculating::OEOsc2rv(oe, 100, 1e-12);
        EXPECT_LT((p - rv.head<3>()).norm() / rv.head<3>().norm(), 1e-9);
    }
}

// 随机 ECI 态往返
TEST(Transform, Random_RoundTrip) {
    std::mt19937_64 rng(999);
    std::uniform_real_distribution<double> ua(6.8e6, 7.5e6), ue(0.0, 0.5), uang(-M_PI, M_PI), ui(1e-3, M_PI - 1e-3);
    for (int t = 0; t < 1000; ++t) {
        VectorXd oe(6);
        oe << ua(rng), uang(rng), ue(rng) * std::cos(uang(rng)), ue(rng) * std::sin(uang(rng)), ui(rng), uang(rng);
        VectorXd rv = osculating::OEOsc2rv(oe, 200, 1e-12);
        VectorXd back = osculating::rv2OEOsc(rv);
        VectorXd rv2 = osculating::OEOsc2rv(back, 200, 1e-12);
        EXPECT_LT((rv2.head<3>() - rv.head<3>()).norm() / rv.head<3>().norm(), 1e-6);
        EXPECT_LT((rv2.tail<3>() - rv.tail<3>()).norm() / rv.tail<3>().norm(), 1e-6);
    }
}

// rv -> 非奇异要素 -> rv 往返一致（相对误差）
TEST(Transform, RV_Osc_RoundTrip) {
    for (const auto &oe : sample_oe()) {
        VectorXd rv = osculating::OEOsc2rv(oe, 100, 1e-12);
        VectorXd back = osculating::rv2OEOsc(rv);
        VectorXd rv2 = osculating::OEOsc2rv(back, 100, 1e-12);
        double rp = std::max(rv.head<3>().norm(), 1.0);
        double rv_ = std::max(rv.tail<3>().norm(), 1.0);
        EXPECT_LT((rv2.head<3>() - rv.head<3>()).norm() / rp, 1e-7);
        EXPECT_LT((rv2.tail<3>() - rv.tail<3>()).norm() / rv_, 1e-8);
        // rv2OEOsc 应恢复输入半长轴 a（注意非圆轨道 r≠a）
        EXPECT_LT(std::fabs(back(0) - oe(0)) / oe(0), 1e-6);
    }
}

// 瞬时<->平均 变换的**位置误差**（判断它是否是 GVE/长弧 14 m 的来源）
TEST(Transform, Osc_Mean_RV_Error) {
    std::mt19937_64 rng(7);
    std::uniform_real_distribution<double> ua(6.8e6, 7.5e6), ue(0.0, 0.3), uang(-M_PI, M_PI), ui(1e-3, M_PI - 1e-3);
    double mx = 0.0;
    for (int t = 0; t < 500; ++t) {
        VectorXd oe(6);
        oe << ua(rng), uang(rng), ue(rng) * std::cos(uang(rng)), ue(rng) * std::sin(uang(rng)), ui(rng), uang(rng);
        VectorXd rv0 = osculating::OEOsc2rv(oe, 200, 1e-12);
        VectorXd om = osculating::OEOsc2OEMeanEU(oe, 200, 1e-3, 1e-4);   // 默认容差
        VectorXd back = osculating::OEMeanEU2OEOsc(om);
        VectorXd rv1 = osculating::OEOsc2rv(back, 200, 1e-12);
        mx = std::max(mx, (rv1.head<3>() - rv0.head<3>()).norm());
    }
    printf("[osc-mean] 500 随机 max 位置误差 = %.4e m\n", mx);
    // Eckstein-Ustinov 一阶短周期修正；位置量级应在 m 以下（若显著大于 1 m，即为 14 m 来源之一）
    EXPECT_LT(mx, 5.0);
}

// 非奇异瞬时要素 -> 平均要素 -> 瞬时要素 往返（Eckstein-Ustinov）
TEST(Transform, Osc_Mean_RoundTrip) {
    for (const auto &oe : sample_oe()) {
        VectorXd om = osculating::OEOsc2OEMeanEU(oe, 200, 1e-9, 1e-11);
        VectorXd back = osculating::OEMeanEU2OEOsc(om);
        EXPECT_TRUE(back.allFinite());
        // a：相对；角度量：绝对（rad）
        EXPECT_LT(std::fabs(back(0) - oe(0)) / oe(0), 1e-5);
        for (int k = 1; k < 6; ++k) EXPECT_LT(std::fabs(back(k) - oe(k)), 1e-4);
    }
}

// 长期传播 tf=0 应为恒等
TEST(LongProp, ZeroIsIdentity) {
    for (const auto &oe : sample_oe()) {
        VectorXd om = osculating::OEOsc2OEMeanEU(oe, 200, 1e-9, 1e-11);
        VectorXd out = osculating::OscElemsLongpropagate(
            0.0, om, bddd::RE, bddd::MU, 1e-9, bddd::J2, bddd::J3, bddd::J4);
        EXPECT_LT((out - om).norm(), 1e-12);
    }
}

// 短期(<1/4T) 与笛卡尔 DA(J234，关阻力) 一致：位置 < 阈值
TEST(LongProp, J234_vs_Cartesian_NoDrag) {
    for (const auto &oe : sample_oe()) {
        VectorXd rv0 = osculating::OEOsc2rv(oe, 100, 1e-12);
        VectorXd om = osculating::OEOsc2OEMeanEU(oe, 200, 1e-9, 1e-11);  // 平均要素
        double tf = 600.0;
        VectorXd op = osculating::OscElemsLongpropagate(
            tf, om, bddd::RE, bddd::MU, 1e-9, bddd::J2, bddd::J3, bddd::J4);
        VectorXd od = osculating::OEMeanEU2OEOsc(op);
        VectorXd rv_oe = osculating::OEOsc2rv(od, 100, 1e-12);
        Eigen::Matrix<double, 6, 6> Phi;
        Vector6d rv0v = rv0;
        Vector6d rv_cart = daJ234DragRV_RK4Step(rv0v, Phi, tf, 0.0, false, 1.0, 2);  // 阻力=0
        // 长期平均要素模型只含 J2/J3/J4 长期+一阶短周期，与笛卡尔 DA 在 600s 差 ~50–450 m（短期项），
        // 仅作量级 sanity 界（不是等价）。精确等价由 Transform.* 保证。
        EXPECT_LT((rv_oe - rv_cart).norm(), 1000.0);
        EXPECT_TRUE(rv_oe.allFinite());
    }
}

// 平均要素长期传播 tf 单调、有限（长弧稳定性 sanity）
TEST(LongProp, LongArcFinite) {
    for (const auto &oe : sample_oe()) {
        VectorXd om = osculating::OEOsc2OEMeanEU(oe, 200, 1e-9, 1e-11);
        for (double tf : {5769.0, 57693.0}) {
            VectorXd op = osculating::OscElemsLongpropagate(
                tf, om, bddd::RE, bddd::MU, 1e-9, bddd::J2, bddd::J3, bddd::J4);
            EXPECT_TRUE(op.allFinite());
            EXPECT_LT(std::fabs(op(0) - oe(0)) / oe(0), 0.05);   // 半长轴长期变化应很小
        }
    }
}

// 阻力长期传播：有限、非零效应
TEST(LongProp, DragFinite) {
    VectorXd om = osculating::OEOsc2OEMeanEU(sample_oe()[0], 200, 1e-9, 1e-11);
    VectorXd op = osculating::OscElemsLongDrag_BCrho0(om, 1440.0, 535e3, 65.35644970323516e3, bddd::MU);
    EXPECT_TRUE(op.allFinite());
}

// 三体开关 + 历元：应产生有限、量级约 1 m / 1/4T 的确定性偏差
TEST(ThirdBody, FlagMakesFiniteSmallDelta) {
    VectorXd rv0 = osculating::OEOsc2rv(sample_oe()[0], 100, 1e-12);
    Vector6d rv0v = rv0;
    double mjd0 = 23673600.0 / 86400.0;
    Eigen::Matrix<double, 6, 6> Phi;
    setThirdBody(false);
    Vector6d a = daJ234DragRV_RK4Step(rv0v, Phi, 1440.0, 1.0, false, 1.0, 2);   // 关
    setThirdBody(true);
    setPropEpoch(mjd0);
    Vector6d b = daJ234DragRV_RK4Step(rv0v, Phi, 1440.0, 1.0, false, 1.0, 2);   // 开
    setThirdBody(false);
    EXPECT_TRUE(b.allFinite());
    double dpos = (b.head<3>() - a.head<3>()).norm();
    EXPECT_GT(dpos, 1e-3);      // 非零
    EXPECT_LT(dpos, 50.0);      // 1/4T 内应为米级
}

// 解析 ∂x/∂κ 的两条独立解析路径互证（均为 DACE 解析，**无 FD**）：
//   (a) 变分增广 `daVarMultiEpochBatchPSKC`（dK/dt=A·K+Fκ）
//   (b) κ 进 DA 第 7 变量 `daJ234DragMultiEpochBatch`（解析线性系数）
TEST(Var, KappaSensitivity_AnalyticVsVariational) {
    Vector6d rv0;
    rv0 << osculating::OEOsc2rv(sample_oe()[0], 100, 1e-12).head<6>();
    std::vector<Vector6d> rv0s = {rv0};
    std::vector<double> tfs = {-720.0, 600.0, 1440.0};   // 含**负**（反向）历元
    std::vector<double> xf, Jt, Jx, Jk, xf2, sens;
    daVarMultiEpochBatchPSKC(rv0s, {}, 0, tfs, 10.0, 1, xf, Jt, Jx, Jk);
    daJ234DragMultiEpochBatch(rv0s, {1.0}, tfs, 10.0, 0.0, 1, xf2, sens);   // 解析
    for (int k = 0; k < 3; ++k) {
        Vector6d a = Eigen::Map<Vector6d>(&Jk[k * 6]);     // 变分解析 ∂x/∂κ (m)
        Vector6d b = Eigen::Map<Vector6d>(&sens[k * 6]);   // DA-N7 解析 ∂x/∂κ (m)
        EXPECT_LT((a - b).norm(), 1e-6 * std::max(1.0, b.norm()));
        Vector6d xa = Eigen::Map<Vector6d>(&xf[k * 6]);
        Vector6d xb = Eigen::Map<Vector6d>(&xf2[k * 6]);
        EXPECT_LT((xa - xb).norm(), 1e-6);
    }
}

// ---------------------------------------------------------------------------
// 正向/反向 RK4：简单模型（谐振子 + 时间显含）验证。论点：ODE 正/反向无区别，
// 反向 == rk4 取负时间步；现成 `rk4b` 与 `rk4(t1<0)` 等价（自治时），且其时间标签须正确。
// ---------------------------------------------------------------------------
namespace {
// 谐振子：d p/dt = v, d v/dt = -w2 p（自治，解析解已知）
Vector6d f_harmonic(Vector6d x, double /*t*/, double w2) {
    Vector6d dx;
    dx.head<3>() = x.tail<3>();
    dx.tail<3>() = -w2 * x.head<3>();
    return dx;
}
// 时间显含：dx/dt = a·t（解析 x(t)=x0 + a t^2/2），用于暴露反向时间标签错误
Vector6d f_ramp(Vector6d /*x*/, double t, double a) { return Vector6d::Constant(a * t); }
}  // namespace

TEST(Backward, Rk4_SignedTime_Harmonic_ForwardBackwardRoundtrip) {
    Vector6d x0; x0 << 1000.0, 0.0, 0.0, 0.5, 0.0, 0.0;
    const double w2 = 1e-6, w = 1e-3, T = 1440.0, h = 10.0;
    Vector6d xf = rk4<Vector6d>(x0, 0.0, T, f_harmonic, w2, h);    // 正向
    Vector6d xb = rk4<Vector6d>(x0, 0.0, -T, f_harmonic, w2, h);   // 反向（负时间步）
    auto pos = [&](double t) { return x0(0) * std::cos(w * t) + x0(3) / w * std::sin(w * t); };
    EXPECT_LT(std::fabs(xf(0) - pos(T)), 1e-4);
    EXPECT_LT(std::fabs(xb(0) - pos(-T)), 1e-4);
    Vector6d back = rk4<Vector6d>(xf, 0.0, -T, f_harmonic, w2, h);  // 往返
    EXPECT_LT((back - x0).norm(), 1e-6);
}

TEST(Backward, Rk4b_Equivalent_To_SignedRk4_Autonomous) {
    Vector6d x0; x0 << 1000.0, 0.0, 0.0, 0.0, 0.5, 0.0;
    const double w2 = 1e-6, T = 1440.0, h = 10.0;
    Vector6d a = rk4b<Vector6d>(x0, 0.0, T, f_harmonic, w2, h);     // 现成 backwardProp
    Vector6d b = rk4<Vector6d>(x0, 0.0, -T, f_harmonic, w2, h);     // 反向 signed rk4
    EXPECT_LT((a - b).norm(), 1e-9);
}

TEST(Backward, Rk4b_TimeLabel_NonAutonomous) {
    Vector6d x0 = Vector6d::Zero();
    const double a = 2.0, T = 1440.0, h = 10.0;
    Vector6d f = rk4<Vector6d>(x0, 0.0, T, f_ramp, a, h);           // 正向
    Vector6d b = rk4<Vector6d>(x0, 0.0, -T, f_ramp, a, h);          // 反向
    Vector6d c = rk4b<Vector6d>(x0, 0.0, T, f_ramp, a, h);          // 现成 backwardProp（时间标签）
    const double exact = a * T * T / 2.0;                           // x(±T) = x0 + a T^2/2
    EXPECT_LT(std::fabs(f(0) - exact), 1e-6);
    EXPECT_LT(std::fabs(b(0) - exact), 1e-6);
    EXPECT_LT(std::fabs(c(0) - exact), 1e-6);
}

// ---------------------------------------------------------------------------
// 非奇异要素（QOE）GVE 误差传播 `NominalErrorPropNOE`：与**笛卡尔 DA 引擎**等价。
// 论点：同物理（二体+J234+阻力）下，两条独立路径的终端状态在 RK4 截断级一致。
// ---------------------------------------------------------------------------
namespace {
// 非奇异要素 → rv 的**模板版** `noeOsc2rv` 与既有 `OEOsc2rv` 必须逐位同（同一公式，rel ~1e-16）。
void noeOsc2rv_matches_reference(const VectorXd &oe) {
    Vector6d oe6; oe6 << oe(0), oe(1), oe(2), oe(3), oe(4), oe(5);
    Vector6d a = noeOsc2rv(oe6, 200, 1e-13);
    Vector6d b = osculating::OEOsc2rv(oe, 200, 1e-13).head<6>();
    EXPECT_LT((a - b).norm() / std::max(b.norm(), 1.0), 1e-14);
}
}  // namespace

TEST(NOE, Osc2rv_MatchesReference) {
    for (const auto &oe : sample_oe()) noeOsc2rv_matches_reference(oe);
}

// 主断言：NOE（GVE 递推）终端状态 vs 笛卡尔 DA 引擎 `daJ234DragBatchD`（同物理）。
TEST(NOE, GvePropagation_MatchesCartesianDA) {
    const double step = 10.0;
    for (const auto &oe : sample_oe()) {
        if (std::hypot(oe(2), oe(3)) == 0.0) continue;   // e=0 时 ω 不定（GVE 口径退化），另测
        Vector6d rv0 = osculating::OEOsc2rv(oe, 200, 1e-13).head<6>();
        Vector6d oe0;   // 用 rv→要素的**引擎口径**（而非直接给 oe），消除两套要素约定的差
        oe0 << osculating::rv2OEOsc(rv0).head<6>();

        std::vector<Vector6d> sens;
        std::vector<Vector6d> ref = daJ234DragBatchD({rv0}, {1.0}, 1440.0, step, 0.0, sens, 1);

        NominalErrorPropNOE da(oe0, 1);
        Eigen::Matrix<double, 6, 6> Phi = Eigen::Matrix<double, 6, 6>::Zero();
        Vector6d oef = da.propNomJ234Drag(Phi, 1440.0, true, step);
        Vector6d rvf = noeOsc2rv(oef, 200, 1e-13);
        EXPECT_LT((rvf.head<3>() - ref[0].head<3>()).norm(), 0.02);      // 1/4T：实测 7 mm
        EXPECT_LT((rvf.tail<3>() - ref[0].tail<3>()).norm(), 1e-4);      // 速度
        EXPECT_TRUE(Phi.allFinite());   // Φ = ∂oe_f/∂oe_0；与笛卡尔 STM 的共轭关系 Φ ≈ D_f·Φ_cart·D_0^{-1}
    }
    // e≈0（圆轨）：ω 口径退化，位置仍应有限且与笛卡尔同物理（容差放宽）
    Vector6d oe_c; oe_c << 6.93e6, 0.5, 2e-3 * std::cos(2.0), 2e-3 * std::sin(2.0), 1.1, 0.4;
    Vector6d rv0 = osculating::OEOsc2rv(oe_c, 200, 1e-13).head<6>();
    Vector6d oe0; oe0 << osculating::rv2OEOsc(rv0).head<6>();
    std::vector<Vector6d> sens, ref = daJ234DragBatchD({rv0}, {1.0}, 1440.0, step, 0.0, sens, 1);
    NominalErrorPropNOE da(oe0, 1);
    Eigen::Matrix<double, 6, 6> Phi = Eigen::Matrix<double, 6, 6>::Zero();
    Vector6d rvf = noeOsc2rv(da.propNomJ234Drag(Phi, 1440.0, true, step), 200, 1e-13);
    EXPECT_TRUE(rvf.allFinite());
    EXPECT_LT((rvf.head<3>() - ref[0].head<3>()).norm(), 0.02);
}

// 反向递推：由终端要素倒推回初始要素（同一动力学，往返应闭合在 RK4 截断级）。
TEST(NOE, BackwardRoundtrip) {
    Vector6d rv0 = osculating::OEOsc2rv(sample_oe()[1], 200, 1e-13).head<6>();
    Vector6d oe0; oe0 << osculating::rv2OEOsc(rv0).head<6>();
    NominalErrorPropNOE fwd(oe0, 1);
    Eigen::Matrix<double, 6, 6> Pf = Eigen::Matrix<double, 6, 6>::Zero();
    Vector6d oef = fwd.propNomJ234Drag(Pf, 1440.0, true, 2.0);
    NominalErrorPropNOE bwd(oef, 1);
    Eigen::Matrix<double, 6, 6> Pb = Eigen::Matrix<double, 6, 6>::Zero();
    Vector6d back = bwd.bkpropNomJ234Drag(Pb, 1440.0, true, 2.0);
    EXPECT_LT(std::fabs(back(0) - oe0(0)), 1e-3);       // 半长轴闭环（m）
    EXPECT_LT((back - oe0).norm(), 1e-6);               // 全要素闭环
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
