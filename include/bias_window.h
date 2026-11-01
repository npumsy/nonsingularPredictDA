// Task #20: 关键帧区间加速度 bias 的 in-band [q;β] 联合装配/求解核心。
//
// 变量：q_k∈R^{6N}（k=0..L，共 L+1 个节点）、β_k∈R^{3N}（k=0..L-1，区间起点 RTN 常量）。
// 因子（全部在给定线性化点一次性装配，固定稀疏；不在核内重新积分）：
//   ISL/GTS 行：p(t)=p̄(t)+Jp(t)δq_k+JpΓ(t)δβ_k，只连本区间 (q_k,β_k)
//   dyn 行    ：q_{k+1}−Φ_k q_k−Γ_k β_k，∂q_k=−Φ、∂q_{k+1}=I、∂β_k=−Γ
//   GM 行     ：β_k−φβ_{k-1}~N(0,Q_b I)，β_0~N(0,diag(σ_b r_RTN)²)
// 求解：把 A^T A 的 [q;β] 联合系统做一次稀疏 LDLT（API 层，供小/中规模
// 自测与后续 Baspacho 装配参考）。Γ(t)/Φ/Jp 由调用方用解析核提供。
#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <cmath>
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace qoejopt {

// ---- 单节点行级接口（纯代数；Φ/Γ/Jp 由现有解析核提供）----
// Eph 位置行：r=p(q(t))−obs；Jq=∂p/∂q0=Jp(t)Φ(t)，Jb=∂p/∂β=Jp(t)Γ(t)。
struct BiasEphRow {
    double r[3];
    double Jq[18];   // 3×6 行主序
    double Jb[9];    // 3×3 行主序
};
inline BiasEphRow biasEphRow(const double *p, const double *Jp,
                             const double *JpG, const double *obs) {
    BiasEphRow o;
    for (int a = 0; a < 3; ++a) {
        o.r[a] = p[a] - obs[a];
        for (int c = 0; c < 6; ++c) o.Jq[a * 6 + c] = Jp[a * 6 + c];
        for (int l = 0; l < 3; ++l) o.Jb[a * 3 + l] = JpG[a * 3 + l];
    }
    return o;
}

// 两星 ISL 行：r=||p_i−p_j||−d；
// Ji=u·JpG_i（∂/∂β_i），Jj=−u·JpG_j（∂/∂β_j）。
struct BiasIslRow {
    double r;
    double Ji[3];
    double Jj[3];
};
inline BiasIslRow biasIslRow(const double *pi, const double *pj,
                             const double *JpGi, const double *JpGj,
                             double d) {
    double dd[3] = {pi[0] - pj[0], pi[1] - pj[1], pi[2] - pj[2]};
    const double dist = std::sqrt(dd[0]*dd[0] + dd[1]*dd[1] + dd[2]*dd[2]);
    const double u[3] = {dd[0]/std::max(dist, 1e-300),
                         dd[1]/std::max(dist, 1e-300),
                         dd[2]/std::max(dist, 1e-300)};
    BiasIslRow o;
    o.r = dist - d;
    for (int l = 0; l < 3; ++l) {
        o.Ji[l] = u[0]*JpGi[0*3+l] + u[1]*JpGi[1*3+l] + u[2]*JpGi[2*3+l];
        o.Jj[l] = -(u[0]*JpGj[0*3+l] + u[1]*JpGj[1*3+l] + u[2]*JpGj[2*3+l]);
    }
    return o;
}

// GTS 行：r=||p_i−q_sta||−d；Jq=u·Jp_i（3×6），Jb=u·JpG_i（3×3）。
struct BiasGtsRow {
    double r;
    double Jq[18];
    double Jb[9];
};
inline BiasGtsRow biasGtsRow(const double *pi, const double *Jp,
                             const double *JpG, const double *qsta, double d) {
    double dd[3] = {pi[0] - qsta[0], pi[1] - qsta[1], pi[2] - qsta[2]};
    const double dist = std::sqrt(dd[0]*dd[0] + dd[1]*dd[1] + dd[2]*dd[2]);
    const double u[3] = {dd[0]/std::max(dist, 1e-300),
                         dd[1]/std::max(dist, 1e-300),
                         dd[2]/std::max(dist, 1e-300)};
    BiasGtsRow o;
    o.r = dist - d;
    for (int a = 0; a < 3; ++a) {
        for (int c = 0; c < 6; ++c)
            o.Jq[a * 6 + c] = u[0]*Jp[0*6+c] + u[1]*Jp[1*6+c] + u[2]*Jp[2*6+c];
        for (int l = 0; l < 3; ++l)
            o.Jb[a * 3 + l] = u[0]*JpG[0*3+l] + u[1]*JpG[1*3+l] + u[2]*JpG[2*3+l];
    }
    return o;
}

// β 链 GM 行：r = w(β_k−φβ_{k−1})，w=1/sqrt(Qb)；
// Jk=wI，Jp=−φwI。k=0 时用 PbDiag：r=(β_0)/σ_b（各轴），J=1/σ_b I。
struct BiasGmRow {
    double r[3];
    double Jk[9];
    double Jp[9];
};
inline BiasGmRow biasGmRow(const double *bk, const double *bp, double phi,
                           double Qb) {
    BiasGmRow o;
    const double w = 1.0 / std::sqrt(std::max(Qb, 1e-300));
    for (int l = 0; l < 3; ++l) {
        o.r[l] = w * (bk[l] - phi * bp[l]);
        for (int m = 0; m < 3; ++m) {
            o.Jk[l * 3 + m] = (l == m) ? w : 0.0;
            o.Jp[l * 3 + m] = (l == m) ? -phi * w : 0.0;
        }
    }
    return o;
}
inline BiasGmRow biasGmRow0(const double *bk, const double *sigma_rtn) {
    BiasGmRow o;
    for (int l = 0; l < 3; ++l) {
        const double w = 1.0 / std::max(sigma_rtn[l], 1e-30);
        o.r[l] = w * bk[l];
        for (int m = 0; m < 3; ++m) {
            o.Jk[l * 3 + m] = (l == m) ? w : 0.0;
            o.Jp[l * 3 + m] = 0.0;
        }
    }
    return o;
}

// ---- RTN/ECI 投影（供 gtest 构造共同 ECI 零空间）----
// Γ/JpG 的 3 列按“区间起点 RTN 轴”解释；核内 B(t) 用瞬时 RTN（Stage A 口径）。
// 共同 ECI 加速度 b 的逐星 RTN 表示为 β_i = E_i^T b（E_i 列为 R/T/N）。
inline void biasRtnBasis(const double *rv, double *E) {
    const double r[3] = {rv[0], rv[1], rv[2]};
    const double v[3] = {rv[3], rv[4], rv[5]};
    const double rn = std::sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
    const double R[3] = {r[0]/std::max(rn, 1e-300), r[1]/std::max(rn, 1e-300),
                         r[2]/std::max(rn, 1e-300)};
    double h[3] = {r[1]*v[2]-r[2]*v[1], r[2]*v[0]-r[0]*v[2], r[0]*v[1]-r[1]*v[0]};
    const double hn = std::sqrt(h[0]*h[0] + h[1]*h[1] + h[2]*h[2]);
    const double N[3] = {h[0]/std::max(hn, 1e-300), h[1]/std::max(hn, 1e-300),
                         h[2]/std::max(hn, 1e-300)};
    const double T[3] = {N[1]*R[2]-N[2]*R[1], N[2]*R[0]-N[0]*R[2],
                         N[0]*R[1]-N[1]*R[0]};
    for (int a = 0; a < 3; ++a) { E[a*3+0] = R[a]; E[a*3+1] = T[a]; E[a*3+2] = N[a]; }
}
inline void biasEciToRtn(const double *b_eci, const double *E, double *b_rtn) {
    for (int l = 0; l < 3; ++l)
        b_rtn[l] = E[0*3+l]*b_eci[0] + E[1*3+l]*b_eci[1] + E[2*3+l]*b_eci[2];
}
inline void biasRtnToEci(const double *b_rtn, const double *E, double *b_eci) {
    for (int a = 0; a < 3; ++a)
        b_eci[a] = E[a*3+0]*b_rtn[0] + E[a*3+1]*b_rtn[1] + E[a*3+2]*b_rtn[2];
}
// JpG_eci = JpG_rtn · Eᵀ（3×3），使 β 的列可直接表示共同 ECI 加速度。
inline void biasJpGRtnToEci(const double *JpG_rtn, const double *E, double *JpG_eci) {
    for (int a = 0; a < 3; ++a)
        for (int k = 0; k < 3; ++k) {
            double s = 0.0;
            for (int l = 0; l < 3; ++l) s += JpG_rtn[a*3+l] * E[k*3+l];
            JpG_eci[a*3+k] = s;
        }
}

struct BiasWinData {
    int n = 0;            // 星数
    int L = 0;            // 关键帧区间数（节点数 L+1）
    int fpn = 0;          // 每区间普通帧数
    int E = 0;            // ISL 边数
    int G = 0;            // GTS (卫星,站) 对数
    double sigma_isl = 1.0;
    double sigma_gts = 1.0;
    double phi = 1.0;     // GM φ=e^{-Δt/τ}
    double Qb = 1.0;      // GM 过程方差（各向同性）
    const int *ei = nullptr;      // (E)
    const int *ej = nullptr;      // (E)
    const double *D = nullptr;    // (L*fpn*E) 已去偏距离
    const double *M = nullptr;    // (L*fpn*E) >0.5 有效
    const double *P = nullptr;    // (L*fpn*n*3) 当前线性化点位置
    const double *Jp = nullptr;   // (L*fpn*n*18) ∂p/∂q（3×6 行主序）
    const double *JpG = nullptr;  // (L*fpn*n*9) ∂p/∂β（3×3 行主序）
    const double *X = nullptr;    // (L*n*6) 当前 q
    const double *beta = nullptr; // ((L-1)*n*3) 当前 β（RTN）
    const double *Phi = nullptr;  // (L*n*36) Φ_k
    const double *dyn_end = nullptr;  // (L*n*6) 非线性 GVE(X_k) 端点；空=ΦX
    const double *Gam = nullptr;  // ((L-1)*n*18) Γ_k（RTN 列）
    const double *Qis = nullptr;  // ((L-1)*n*36) Q_cond^{-1/2}；空=I
    const double *qInfo = nullptr;  // (L*n*36) q 先验信息；空=弱先验
    const double *Xp = nullptr;     // (L*n*6) q 先验中心；空=当前 X
    const double *PbDiag = nullptr; // (n*3) σ_b·r_RTN（R/T/N）
    const int *gsat = nullptr;      // (G)
    const double *gD = nullptr;     // (L*fpn*G) 已去偏 GTS 距离
    const double *gM = nullptr;     // (L*fpn*G)
    const double *gQ = nullptr;     // (L*fpn*G*3) 站 ECI 位置
    double q_weak = 1e-6;           // qInfo 为空时的弱先验方差
    bool compute_cov = true;
    bool beta_only = false;   // true=固定/强先验 q，只解 β
    bool compute_cond = true;
    bool want_rows = false;   // true=输出增广 A 的行三元组（供 Baspacho）
    bool no_solve = false;    // true=只装配不求解
    bool beta_off = false;    // true=不创建 β 变量/列/行（q-only 对照）
    int cov_max_nb = 512;           // β 维数超过则跳过协方差
};

struct BiasWinResult {
    Eigen::VectorXd dX;       // (L*n*6)
    Eigen::VectorXd dbeta;    // ((L-1)*n*3)
    Eigen::VectorXd betaSigma;  // ((L-1)*n*3)，跳过时为空
    double cost = 0.0;          // 装配行的 b^T b
    double cost_isl = 0.0, cost_dyn = 0.0, cost_beta = 0.0, cost_q = 0.0;
    double cost_gts = 0.0;
    double cond_joint = NAN, cond_beta = NAN;  // 对角 max/min 代理
    // want_rows=true 时的行三元组：(row, col, val)、b(row)、nrows/ncol/nb/nq。
    std::vector<int> A_rows, A_cols; std::vector<double> A_vals, b_rows;
    int nrows = 0, ncol = 0, nq = 0, nb = 0;
};

// 装配一个线性化点的 [q;β] 法方程并解一次 GN 步：
//   H δ = A^T b,  δ=[dX;dbeta].
inline int biasWindowSolve(const BiasWinData &d, BiasWinResult &out) {
    const int n = d.n, L = d.L, fpn = d.fpn, E = d.E;
    if (n <= 0 || L <= 1 || fpn <= 0)
        throw std::invalid_argument("biasWindowSolve: 维度非法");
    const int nq = (L + 1) * n * 6;
    const int nb = d.beta_off ? 0 : L * n * 3;
    const int ncol = nq + nb;
    const int nfr_tot = L * fpn;
    const double s_isl = 1.0 / std::max(d.sigma_isl, 1e-30);

    Eigen::SparseMatrix<double> H(ncol, ncol);
    std::vector<Eigen::Triplet<double>> tri;
    tri.reserve((size_t)nfr_tot * E * 15 +
                (size_t)L * n * (6 * 18 + 3 * 6));
    Eigen::VectorXd g = Eigen::VectorXd::Zero(ncol);
    double cost = 0.0;

    auto qcol = [&](int k, int i, int c) { return ((size_t)k * n + i) * 6 + c; };
    auto bcol = [&](int k, int i, int l) {
        return nq + ((size_t)k * n + i) * 3 + l;
    };
    // A 行（列,值）与 b（GN 右端 b=-r）：g += A^T b, H += A^T A。
    int _row = 0;
    auto add_row = [&](const std::vector<std::pair<int, double>> &cols, double b,
                       double &cb) {
        if (d.want_rows) {
            for (auto &p : cols) {
                out.A_rows.push_back(_row); out.A_cols.push_back(p.first);
                out.A_vals.push_back(p.second);
            }
            out.b_rows.push_back(b);
        }
        ++_row;
        for (size_t a = 0; a < cols.size(); ++a) {
            const int ca = cols[a].first;
            const double va = cols[a].second;
            g[ca] += va * b;
            for (size_t c2 = 0; c2 < cols.size(); ++c2)
                tri.emplace_back(ca, cols[c2].first, va * cols[c2].second);
        }
        cb += b * b;
        cost += b * b;
    };
    double c_isl = 0.0, c_dyn = 0.0, c_beta = 0.0, c_q = 0.0, c_gts = 0.0;

    // ---- ISL ----
    if (E > 0 && d.D && d.M && d.P && d.Jp && d.JpG) {
        for (int t = 0; t < nfr_tot; ++t) {
            const int k = t / fpn;
            for (int e = 0; e < E; ++e) {
                if (!(d.M[(size_t)t * E + e] > 0.5)) continue;
                const int i = d.ei[e], j = d.ej[e];
                const size_t ii = (size_t)t * n + i, jj = (size_t)t * n + j;
                const double *pi = d.P + ii * 3, *pj = d.P + jj * 3;
                double dd[3] = {pi[0] - pj[0], pi[1] - pj[1], pi[2] - pj[2]};
                const double dist = std::sqrt(dd[0]*dd[0] + dd[1]*dd[1] + dd[2]*dd[2]);
                if (!(dist > 0.0)) continue;
                const double u[3] = {dd[0]/dist, dd[1]/dist, dd[2]/dist};
                const double r = dist - d.D[(size_t)t * E + e];
                // q 列：u·Jp_i / −u·Jp_j
                double ai[6], aj[6], bi[3], bj[3];
                for (int c = 0; c < 6; ++c) {
                    ai[c] = u[0] * d.Jp[ii * 18 + 0 * 6 + c]
                          + u[1] * d.Jp[ii * 18 + 1 * 6 + c]
                          + u[2] * d.Jp[ii * 18 + 2 * 6 + c];
                    aj[c] = u[0] * d.Jp[jj * 18 + 0 * 6 + c]
                          + u[1] * d.Jp[jj * 18 + 1 * 6 + c]
                          + u[2] * d.Jp[jj * 18 + 2 * 6 + c];
                }
                for (int l = 0; l < 3; ++l) {
                    bi[l] = u[0] * d.JpG[ii * 9 + 0 * 3 + l]
                          + u[1] * d.JpG[ii * 9 + 1 * 3 + l]
                          + u[2] * d.JpG[ii * 9 + 2 * 3 + l];
                    bj[l] = u[0] * d.JpG[jj * 9 + 0 * 3 + l]
                          + u[1] * d.JpG[jj * 9 + 1 * 3 + l]
                          + u[2] * d.JpG[jj * 9 + 2 * 3 + l];
                }
                std::vector<std::pair<int, double>> cols;
                cols.reserve(18);
                for (int c = 0; c < 6; ++c) cols.emplace_back(qcol(k, i, c), ai[c] * s_isl);
                for (int c = 0; c < 6; ++c) cols.emplace_back(qcol(k, j, c), -aj[c] * s_isl);
                if (!d.beta_off) {
                    for (int l = 0; l < 3; ++l) cols.emplace_back(bcol(k, i, l), bi[l] * s_isl);
                    for (int l = 0; l < 3; ++l) cols.emplace_back(bcol(k, j, l), -bj[l] * s_isl);
                }
                add_row(cols, -r * s_isl, c_isl);
            }
        }
    }

    // ---- GTS ----
    if (d.G > 0 && d.gD && d.gM && d.gQ && d.gsat && d.P && d.Jp && d.JpG) {
        const double s_gts = 1.0 / std::max(d.sigma_gts, 1e-30);
        for (int t = 0; t < nfr_tot; ++t) {
            const int k = t / fpn;
            for (int g = 0; g < d.G; ++g) {
                if (!(d.gM[(size_t)t * d.G + g] > 0.5)) continue;
                const int i = d.gsat[g];
                const size_t ii = (size_t)t * n + i;
                const double *pi = d.P + ii * 3;
                const double *qg = d.gQ + ((size_t)t * d.G + g) * 3;
                double dd[3] = {pi[0] - qg[0], pi[1] - qg[1], pi[2] - qg[2]};
                const double dist = std::sqrt(dd[0]*dd[0] + dd[1]*dd[1] + dd[2]*dd[2]);
                if (!(dist > 0.0)) continue;
                const double u[3] = {dd[0]/dist, dd[1]/dist, dd[2]/dist};
                const double r = dist - d.gD[(size_t)t * d.G + g];
                double ai[6], bi[3];
                for (int c = 0; c < 6; ++c)
                    ai[c] = u[0]*d.Jp[ii*18+0*6+c] + u[1]*d.Jp[ii*18+1*6+c]
                          + u[2]*d.Jp[ii*18+2*6+c];
                for (int l = 0; l < 3; ++l)
                    bi[l] = u[0]*d.JpG[ii*9+0*3+l] + u[1]*d.JpG[ii*9+1*3+l]
                          + u[2]*d.JpG[ii*9+2*3+l];
                std::vector<std::pair<int, double>> cols;
                cols.reserve(9);
                for (int c = 0; c < 6; ++c) cols.emplace_back(qcol(k, i, c), ai[c] * s_gts);
                if (!d.beta_off)
                    for (int l = 0; l < 3; ++l)
                        cols.emplace_back(bcol(k, i, l), bi[l] * s_gts);
                add_row(cols, -r * s_gts, c_gts);
            }
        }
    }

    // ---- dyn ----
    for (int k = 0; k < L; ++k) {
        for (int i = 0; i < n; ++i) {
            const size_t base = ((size_t)k * n + i);
            const double *Ph = d.Phi ? d.Phi + base * 36 : nullptr;
            const double *Gm = d.Gam ? d.Gam + base * 18 : nullptr;
            double rr[6];
            for (int a = 0; a < 6; ++a) {
                double v = d.X[((size_t)(k + 1) * n + i) * 6 + a];
                if (d.dyn_end) {
                    v -= d.dyn_end[base * 6 + a];
                } else if (Ph) {
                    double s = 0.0;
                    for (int b2 = 0; b2 < 6; ++b2)
                        s += Ph[a * 6 + b2] * d.X[base * 6 + b2];
                    v -= s;
                }
                if (Gm && d.beta && !d.beta_off) {
                    double s = 0.0;
                    for (int l = 0; l < 3; ++l) s += Gm[a * 3 + l] * d.beta[base * 3 + l];
                    v -= s;
                }
                rr[a] = v;
            }
            // bvec = −Qis r；Qis 行块
            for (int a = 0; a < 6; ++a) {
                double bvec = -rr[a];
                std::vector<std::pair<int, double>> cols;
                cols.reserve(15);
                // q_{k+1}: Qis
                for (int c = 0; c < 6; ++c) {
                    const double v = d.Qis ? d.Qis[base * 36 + a * 6 + c]
                                           : (a == c ? 1.0 : 0.0);
                    cols.emplace_back(qcol(k + 1, i, c), v);
                    bvec = d.Qis ? bvec : bvec;  // no-op
                }
                // q_k: −Qis Φ
                if (Ph) {
                    for (int c = 0; c < 6; ++c) {
                        double v = 0.0;
                        for (int b2 = 0; b2 < 6; ++b2)
                            v -= (d.Qis ? d.Qis[base * 36 + a * 6 + b2]
                                        : (a == b2 ? 1.0 : 0.0)) * Ph[b2 * 6 + c];
                        cols.emplace_back(qcol(k, i, c), v);
                    }
                }
                // β_k: −Qis Γ
                if (Gm && d.beta && !d.beta_off) {
                    for (int l = 0; l < 3; ++l) {
                        double v = 0.0;
                        for (int b2 = 0; b2 < 6; ++b2)
                            v -= (d.Qis ? d.Qis[base * 36 + a * 6 + b2]
                                        : (a == b2 ? 1.0 : 0.0)) * Gm[b2 * 3 + l];
                        cols.emplace_back(bcol(k, i, l), v);
                    }
                }
                add_row(cols, bvec, c_dyn);
            }
        }
    }

    // ---- β GM 先验 ----
    if (d.beta && !d.beta_off) {
        const double w0 = 1.0 / std::sqrt(std::max(d.Qb, 1e-300));
        for (int i = 0; i < n; ++i) {
            for (int l = 0; l < 3; ++l) {
                const double pb = d.PbDiag ? std::max(d.PbDiag[i * 3 + l], 1e-30)
                                           : 1e-7;
                const double w = 1.0 / pb;
                const int c0 = bcol(0, i, l);
                add_row({{c0, w}}, -w * d.beta[(size_t)i * 3 + l], c_beta);
            }
        }
        for (int k = 1; k < L; ++k) {
            for (int i = 0; i < n; ++i) {
                for (int l = 0; l < 3; ++l) {
                    const int ck = bcol(k, i, l), cp = bcol(k - 1, i, l);
                    const double bk = d.beta[((size_t)k * n + i) * 3 + l];
                    const double bp = d.beta[((size_t)(k - 1) * n + i) * 3 + l];
                    add_row({{ck, w0}, {cp, -d.phi * w0}},
                            -w0 * (bk - d.phi * bp), c_beta);
                }
            }
        }
    }

    // ---- q 先验 ----
    for (int k = 0; k <= L; ++k) {
        for (int i = 0; i < n; ++i) {
            const size_t base = ((size_t)k * n + i);
            if (d.qInfo) {
                Eigen::Matrix<double, 6, 6> I;
                for (int a = 0; a < 6; ++a)
                    for (int b2 = 0; b2 < 6; ++b2)
                        I(a, b2) = d.qInfo[base * 36 + a * 6 + b2];
                Eigen::LLT<Eigen::Matrix<double, 6, 6>> llt(I);
                if (llt.info() != Eigen::Success) continue;
                const Eigen::Matrix<double, 6, 6> S = llt.matrixL();
                Eigen::Matrix<double, 6, 1> dx;
                for (int a = 0; a < 6; ++a)
                    dx[a] = d.X[base * 6 + a]
                          - (d.Xp ? d.Xp[base * 6 + a] : d.X[base * 6 + a]);
                const Eigen::Matrix<double, 6, 1> rhs = -(S.transpose() * dx);
                for (int a = 0; a < 6; ++a) {
                    std::vector<std::pair<int, double>> cols;
                    for (int b2 = 0; b2 < 6; ++b2)
                        cols.emplace_back(qcol(k, i, b2), S(a, b2));
                    add_row(cols, rhs[a], c_q);
                }
            } else {
                const double w = 1.0 / std::sqrt(std::max(d.q_weak, 1e-300));
                for (int a = 0; a < 6; ++a) {
                    const double dx = d.X[base * 6 + a]
                                    - (d.Xp ? d.Xp[base * 6 + a] : d.X[base * 6 + a]);
                    add_row({{qcol(k, i, a), w}}, -w * dx, c_q);
                }
            }
        }
    }

    // ---- 求解 ----
    H.setFromTriplets(tri.begin(), tri.end());
    H.makeCompressed();
    double dmax = 0.0;
    for (int c = 0; c < ncol; ++c) dmax = std::max(dmax, H.coeff(c, c));
    if (!(dmax > 0.0)) throw std::runtime_error("biasWindowSolve: 空法方程");
    H.diagonal().array() += 1e-12 * dmax;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> llt(H);
    if (llt.info() != Eigen::Success)
        throw std::runtime_error("biasWindowSolve: LDLT 分解失败");
    auto _diag_ratio = [](const Eigen::SparseMatrix<double> &M) {
        double mx = 0.0, mn = 1e300;
        for (int c = 0; c < M.cols(); ++c) {
            const double v = M.coeff(c, c);
            if (v > 0.0) { mx = std::max(mx, v); mn = std::min(mn, v); }
        }
        return (mn < 1e299 && mx > 0.0) ? mx / mn : NAN;
    };
    out.nrows = _row; out.ncol = ncol; out.nq = nq; out.nb = nb;
    if (d.no_solve) { out.cost = cost; out.cost_isl = c_isl; out.cost_dyn = c_dyn;
        out.cost_beta = c_beta; out.cost_q = c_q; out.cost_gts = c_gts; return 0; }
    if (d.beta_only) {
        Eigen::SparseMatrix<double> Hbb = H.block(nq, nq, nb, nb);
        Hbb.makeCompressed();
        Eigen::VectorXd gb = g.tail(nb);
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> lb(Hbb);
        if (lb.info() != Eigen::Success)
            throw std::runtime_error("biasWindowSolve: β-only LDLT 失败");
        const Eigen::VectorXd db = lb.solve(gb);
        if (!db.allFinite())
            throw std::runtime_error("biasWindowSolve: β-only 解非有限");
        out.dX = Eigen::VectorXd::Zero(nq);
        out.dbeta = db;
        if (d.compute_cond) {
            out.cond_joint = _diag_ratio(H);
            out.cond_beta = _diag_ratio(Hbb);
        }
    } else {
        const Eigen::VectorXd delta = llt.solve(g);
        if (delta.allFinite() == false)
            throw std::runtime_error("biasWindowSolve: 解非有限");
        out.dX = delta.head(nq);
        out.dbeta = delta.tail(nb);
        if (d.compute_cond) {
            out.cond_joint = _diag_ratio(H);
            Eigen::SparseMatrix<double> Hbb = H.block(nq, nq, nb, nb);
            Hbb.makeCompressed();
            out.cond_beta = _diag_ratio(Hbb);
        }
    }
    out.cost = cost;
    out.cost_isl = c_isl; out.cost_dyn = c_dyn;
    out.cost_beta = c_beta; out.cost_q = c_q; out.cost_gts = c_gts;
    out.betaSigma.resize(0);
    if (d.compute_cov && nb > 0 && nb <= d.cov_max_nb) {
        Eigen::MatrixXd E = Eigen::MatrixXd::Zero(ncol, nb);
        for (int j = 0; j < nb; ++j) E(nq + j, j) = 1.0;
        const Eigen::MatrixXd Y = llt.solve(E);
        out.betaSigma.resize(nb);
        for (int j = 0; j < nb; ++j)
            out.betaSigma[j] = std::sqrt(std::max(Y(nq + j, j), 0.0));
    }
    return 0;
}

}  // namespace qoejopt
