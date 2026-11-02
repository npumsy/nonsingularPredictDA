// 集中式精确协方差 EKF（--ekf-central）。
// 状态 [q_i(6); b_i] (i=1..N), c_s；7N+M=8064；P 稠密，全流程 DU 归一化 z 空间。
// 入口仅 2 个：ekf_predict / ekf_update。GVE/STM/QOE->rv 复用 qoe_core。

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include "ekf_central.h"
#include "qoejopt.h"

#ifndef EKF_CENTRAL_NO_PYBIND
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;
#endif

namespace ekf_central {
namespace {

struct Ctx {
    bool init = false;
    Params p;
    double phi_b = 0.0, qb = 0.0, phi_c = 0.0, qc = 0.0;
    // state pointers (Python torch CUDA tensors)
    double *d_q = nullptr, *d_b = nullptr, *d_c = nullptr, *d_P = nullptr;
    // work
    double *d_Ptmp = nullptr, *d_Ptmp2 = nullptr, *d_A = nullptr, *d_H = nullptr, *d_PHt = nullptr,
           *d_PHtT = nullptr, *d_S = nullptr, *d_Kt = nullptr, *d_K = nullptr,
           *d_Ks = nullptr, *d_nu = nullptr, *d_dx = nullptr, *d_Q = nullptr,
           *d_qpred = nullptr, *d_Phi = nullptr, *d_rv = nullptr, *d_Ap = nullptr,
           *d_Disl = nullptr, *d_Misl = nullptr, *d_Dgts = nullptr,
           *d_Mgts = nullptr, *d_Qgts = nullptr;
    int *d_ei = nullptr, *d_ej = nullptr, *d_ga = nullptr, *d_gs = nullptr;
    double *d_hval = nullptr; int *d_hidx = nullptr;  // 行稀疏 H：每行最多 14 个非零
    float *d_Sf = nullptr, *d_PHtf = nullptr, *d_Kf = nullptr, *d_Df = nullptr;  // FP32 GEMM
    float *d_Bf = nullptr, *d_workf = nullptr;  // FP32 右端/解、Sgetrf workspace
    int *d_ipiv = nullptr; double *d_dscale = nullptr, *d_escale = nullptr, *d_y = nullptr;
    double *d_ext = nullptr;
    double *d_work = nullptr; int lwork = 0;
    double *d_eig = nullptr; int *d_info = nullptr;
    double *d_sym = nullptr;  // [diff2, norm2]
    int *d_nf = nullptr; double *d_diag = nullptr;  // finite 计数 / 对角拷贝
    cublasHandle_t blas = nullptr;
    cusolverDnHandle_t solver = nullptr;
    cudaEvent_t ev[8];
};

Ctx g;

void _check(cudaError_t e, const char *msg) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(e));
}
void _cublas(cublasStatus_t e, const char *msg) {
    if (e != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(msg);
}
void _cusolver(cusolverStatus_t e, const char *msg) {
    if (e != CUSOLVER_STATUS_SUCCESS) throw std::runtime_error(msg);
}

void _gemm(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb,
           int m, int n, int k, double alpha, const double *A, int lda,
           const double *B, int ldb, double beta, double *C, int ldc, const char *msg) {
    _cublas(cublasDgemm(h, ta, tb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc), msg);
}
void _gemv(cublasHandle_t h, cublasOperation_t ta, int m, int n, double alpha,
           const double *A, int lda, const double *x, int incx, double beta,
           double *y, int incy, const char *msg) {
    _cublas(cublasDgemv(h, ta, m, n, &alpha, A, lda, x, incx, &beta, y, incy), msg);
}

// P^- = Fz P Fz^T + Qz（列主序，全局行跨度 ns）。
// node-node：共享内存装载 7×7 P/Φ_i/Φ_j，一个 block 顺序做 BJ 个 bj（减少块调度与冗余读）。
__global__ void fpft_node_kernel(int n, int ns, const double *P, double *Pout,
                                 const double *Phi, const double *Q, double phi_b, int BJ) {
    const int bi = blockIdx.x;
    const int bj0 = blockIdx.y * BJ;
    const int t = threadIdx.x;  // 0..48
    const int a = t / 7, b = t % 7;
    __shared__ double Fi[49], PB[49], Fj[49];
    {
        const double val = (a < 6 && b < 6) ? Phi[bi * 36 + a * 6 + b] : 0.0;
        const double sc = (b == 0) ? DU : 1.0, sr = (a == 0) ? DU : 1.0;
        Fi[t] = (a == 6 && b == 6) ? phi_b : val * sc / sr;
    }
    __syncthreads();
    const int bjmax = min(n, bj0 + BJ);
    for (int bj = bj0; bj < bjmax; ++bj) {
        const int c = t / 7, d = t % 7;
        PB[t] = P[(7 * bi + c) + (size_t)(7 * bj + d) * ns];
        const double val = (c < 6 && d < 6) ? Phi[bj * 36 + c * 6 + d] : 0.0;
        const double sc = (d == 0) ? DU : 1.0, sr = (c == 0) ? DU : 1.0;
        Fj[t] = (c == 6 && d == 6) ? phi_b : val * sc / sr;
        __syncthreads();
        // 精确两步块乘：(P Fj^T)[a,b] -> tmp；out = Fi * tmp。每块 2*7^3 MAC。
        __shared__ double tmp[49];
        double tv = 0.0;
        #pragma unroll
        for (int dd = 0; dd < 7; ++dd) tv += PB[a * 7 + dd] * Fj[b * 7 + dd];
        tmp[t] = tv;
        __syncthreads();
        double v = 0.0;
        #pragma unroll
        for (int cc = 0; cc < 7; ++cc) v += Fi[a * 7 + cc] * tmp[cc * 7 + b];
        const double sa = (a == 0) ? DU : 1.0, sb = (b == 0) ? DU : 1.0;
        if (bi == bj) v += Q[bi * 49 + a * 7 + b] / (sa * sb);
        Pout[(7 * bi + a) + (size_t)(7 * bj + b) * ns] = v;
        __syncthreads();
    }
}

// 含站点的块（少量）：沿用逐块分支，bi/bj 由调用方加偏移。
__global__ void fpft_kernel(int n, int M, int ns, const double *P, double *Pout,
                            const double *Phi, const double *Q,
                            double phi_b, double phi_c, double qc,
                            int bi_off, int bj_off) {
    const int nb = n + M;
    const int bi = blockIdx.x + bi_off, bj = blockIdx.y + bj_off;
    if (bi >= nb || bj >= nb) return;
    const int oi = (bi < n) ? 7 * bi : 7 * n + (bi - n);
    const int oj = (bj < n) ? 7 * bj : 7 * n + (bj - n);
    const int t = threadIdx.x;
    if (bi < n && bj < n) {
        if (t >= 49) return;
        const int a = t / 7, b = t % 7;
        // Fz = S^{-1} Φ S：a 分量（index 0）按 DU 缩放，钟差不变。
        const double sa = (a == 0) ? DU : 1.0, sb = (b == 0) ? DU : 1.0;
        double Fi[7][7] = {}, Fj[7][7] = {};
        for (int ia = 0; ia < 6; ++ia) for (int ib = 0; ib < 6; ++ib) {
            const double sc = (ib == 0) ? DU : 1.0, sr = (ia == 0) ? DU : 1.0;
            Fi[ia][ib] = Phi[bi * 36 + ia * 6 + ib] * sc / sr;
            Fj[ia][ib] = Phi[bj * 36 + ia * 6 + ib] * sc / sr;
        }
        Fi[6][6] = phi_b; Fj[6][6] = phi_b;
        double v = 0.0;
        for (int c = 0; c < 7; ++c) {
            double u = 0.0;
            for (int d = 0; d < 7; ++d)
                u += P[(oi + c) + (oj + d) * ns] * Fj[b][d];
            v += Fi[a][c] * u;
        }
        // Qz = S^{-1} Q S^{-1}
        if (bi == bj) v += Q[bi * 49 + a * 7 + b] / (sa * sb);
        Pout[(oi + a) + (oj + b) * ns] = v;
    } else if (bi >= n && bj >= n) {
        if (t != 0) return;
        double v = phi_c * phi_c * P[oi + oj * ns];
        if (bi == bj) v += qc;
        Pout[oi + oj * ns] = v;
    } else if (bi < n && bj >= n) {
        if (t >= 7) return;
        double Fi[7][7] = {};
        for (int ia = 0; ia < 6; ++ia) for (int ib = 0; ib < 6; ++ib) {
            const double sc = (ib == 0) ? DU : 1.0, sr = (ia == 0) ? DU : 1.0;
            Fi[ia][ib] = Phi[bi * 36 + ia * 6 + ib] * sc / sr;
        }
        Fi[6][6] = phi_b;
        double v = 0.0;
        for (int c = 0; c < 7; ++c) v += Fi[t][c] * P[(oi + c) + oj * ns];
        Pout[(oi + t) + oj * ns] = phi_c * v;
    } else {  // station-node
        if (t >= 7) return;
        double Fj[7][7] = {};
        for (int ia = 0; ia < 6; ++ia) for (int ib = 0; ib < 6; ++ib) {
            const double sc = (ib == 0) ? DU : 1.0, sr = (ia == 0) ? DU : 1.0;
            Fj[ia][ib] = Phi[bj * 36 + ia * 6 + ib] * sc / sr;
        }
        Fj[6][6] = phi_b;
        double v = 0.0;
        for (int c = 0; c < 7; ++c) v += P[oi + (oj + c) * ns] * Fj[t][c];
        Pout[oi + (oj + t) * ns] = phi_c * v;
    }
}

__global__ void transpose_kernel(int rows, int cols, const double *A, double *At) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    for (; k < total; k += blockDim.x * gridDim.x) {
        int r = k % rows, c = k / rows;
        At[r + c * rows] = A[c + r * cols];
    }
}

__global__ void make_A_kernel(int ns, double *A) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int total = ns * ns;
    for (; k < total; k += blockDim.x * gridDim.x) {
        int i = k % ns, j = k / ns;
        A[k] = -A[k] + (i == j ? 1.0 : 0.0);
    }
}

__global__ void add_mat_kernel(int ns, double *dst, const double *src) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int total = ns * ns;
    for (; k < total; k += blockDim.x * gridDim.x) dst[k] += src[k];
}

__global__ void retract_q_kernel(int n, double *q) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double *z = q + 6 * i;
    z[1] = fmod(z[1], 2 * M_PI); if (z[1] < 0) z[1] += 2 * M_PI;
    z[5] = fmod(z[5], 2 * M_PI); if (z[5] < 0) z[5] += 2 * M_PI;
    z[4] = fmin(fmax(z[4], 1e-6), M_PI - 1e-6);
    double e = hypot(z[2], z[3]);
    if (e > 0.999) { z[2] *= 0.999 / e; z[3] *= 0.999 / e; }
    z[0] = fmin(fmax(z[0], 1e5), 1e8);
}

// H (nrows x nstate, 列主序) 装配 + nu；每行一个线程。H 使用 z 空间列（a 分量 ×DU）。
// 每行最多 14 个非零（ISL：两节点各 7；GTS：节点 7 + 站 1）。
__global__ void h_assemble_kernel(int n, int M, int E, int G, int nrows, int nstate,
                                  const int *ei, const int *ej, const int *ga, const int *gs,
                                  const double *rv, const double *Ap,
                                  const double *Disl, const double *Misl,
                                  const double *Dgts, const double *Mgts, const double *Qgts,
                                  const double *b, const double *c,
                                  double *H, double *nu, double *hval) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) return;
    if (row < E) {
        if (Misl[row] <= 0.5) { nu[row] = 0.0; return; }
        const int i = ei[row], j = ej[row];
        double d[3], dist = 0.0;
        for (int k = 0; k < 3; ++k) { d[k] = rv[i * 6 + k] - rv[j * 6 + k]; dist += d[k] * d[k]; }
        dist = sqrt(dist); double u[3];
        for (int k = 0; k < 3; ++k) u[k] = d[k] / (dist + 1e-12);
        for (int a = 0; a < 6; ++a) {
            const double sc = (a == 0) ? DU : 1.0;  // H_z = H_x S
            const double hi = sc * (u[0] * Ap[i * 18 + 0 * 6 + a]
                + u[1] * Ap[i * 18 + 1 * 6 + a] + u[2] * Ap[i * 18 + 2 * 6 + a]);
            const double hj = -sc * (u[0] * Ap[j * 18 + 0 * 6 + a]
                + u[1] * Ap[j * 18 + 1 * 6 + a] + u[2] * Ap[j * 18 + 2 * 6 + a]);
            H[row + (7 * i + a) * nrows] = hi;
            H[row + (7 * j + a) * nrows] = hj;
            hval[row * 16 + a] = hi;
            hval[row * 16 + 7 + a] = hj;
        }
        H[row + (7 * i + 6) * nrows] = 1.0;
        H[row + (7 * j + 6) * nrows] = -1.0;
        hval[row * 16 + 6] = 1.0;
        hval[row * 16 + 13] = -1.0;
        nu[row] = Disl[row] - (dist + b[i] - b[j]);
    } else {
        const int k = row - E;
        if (Mgts[k] <= 0.5) { nu[row] = 0.0; return; }
        const int i = ga[k], s = gs[k];
        double d[3], dist = 0.0;
        for (int x = 0; x < 3; ++x) { d[x] = rv[i * 6 + x] - Qgts[k * 3 + x]; dist += d[x] * d[x]; }
        dist = sqrt(dist); double u[3];
        for (int x = 0; x < 3; ++x) u[x] = d[x] / (dist + 1e-12);
        for (int a = 0; a < 6; ++a) {
            const double sc = (a == 0) ? DU : 1.0;
            const double hi = sc * (u[0] * Ap[i * 18 + 0 * 6 + a]
                + u[1] * Ap[i * 18 + 1 * 6 + a] + u[2] * Ap[i * 18 + 2 * 6 + a]);
            H[row + (7 * i + a) * nrows] = hi;
            hval[row * 16 + a] = hi;
        }
        H[row + (7 * i + 6) * nrows] = 1.0;
        H[row + (7 * n + s) * nrows] = -1.0;
        hval[row * 16 + 6] = 1.0;
        hval[row * 16 + 7] = -1.0;
        nu[row] = Dgts[k] - (dist + b[i] - c[s]);
    }
}

// P H^T 的列稀疏组装：PHt[:,b] = Σ_{k∈nz(b)} hval[b,k] P[:, hidx[b,k]]。
__global__ void phT_sparse_kernel(int ns, int nr, const double *P, const double *hval,
                                  const int *hidx, double *PHt) {
    const int b = blockIdx.x;
    if (b >= nr) return;
    for (int s = threadIdx.x; s < ns; s += blockDim.x) {
        double v = 0.0;
        #pragma unroll
        for (int k = 0; k < 14; ++k)
            v += hval[b * 16 + k] * P[s + (size_t)hidx[b * 16 + k] * ns];
        PHt[s + (size_t)b * ns] = v;
    }
}

// S = H P H^T 的行稀疏组装：S[a,b] = Σ_k hval[a,k] PHt[hidx[a,k], b]。
__global__ void S_sparse_kernel(int ns, int nr, const double *PHt, const double *hval,
                                const int *hidx, double *S) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = nr * nr;
    for (; k < total; k += blockDim.x * gridDim.x) {
        const int a = k % nr, b = k / nr;
        double v = 0.0;
        #pragma unroll
        for (int q = 0; q < 14; ++q)
            v += hval[a * 16 + q] * PHt[hidx[a * 16 + q] + (size_t)b * ns];
        S[a + (size_t)b * nr] = v;
    }
}

__global__ void add_R_kernel(int nrows, int E, double sig_i, double sig_g,
                             const double *Misl, const double *Mgts, double *S) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= nrows) return;
    double R = (r < E) ? ((Misl[r] > 0.5) ? sig_i * sig_i : 1.0)
                       : ((Mgts[r - E] > 0.5) ? sig_g * sig_g : 1.0);
    S[r + r * nrows] += R;
}

// potrf/potri 只写 lower；把 lower 镜像到 upper（不能与未写的 upper 平均）。
__global__ void mirror_lower_kernel(int n, double *A) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n * n;
    for (; k < total; k += blockDim.x * gridDim.x) {
        int i = k % n, j = k / n;
        if (i > j) A[j + (size_t)i * n] = A[i + (size_t)j * n];
    }
}

__global__ void symmetrize_kernel(int nstate, double *P) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nstate * nstate;
    for (int k = idx; k < total; k += blockDim.x * gridDim.x) {
        int i = k % nstate, j = k / nstate;
        if (i < j) continue;
        double v = 0.5 * (P[i + j * nstate] + P[j + i * nstate]);
        P[i + j * nstate] = v; P[j + i * nstate] = v;
    }
}

__global__ void sym_ratio_kernel(int nstate, const double *P, double *out) {
    double d2 = 0.0, n2 = 0.0;
    int total = nstate * nstate;
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < total; k += blockDim.x * gridDim.x) {
        int i = k % nstate, j = k / nstate;
        double a = P[i + j * nstate], b = P[j + i * nstate];
        d2 += (a - b) * (a - b); n2 += a * a;
    }
    __shared__ double sd[256], sn[256];
    sd[threadIdx.x] = d2; sn[threadIdx.x] = n2; __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) { sd[threadIdx.x] += sd[threadIdx.x + off];
                                 sn[threadIdx.x] += sn[threadIdx.x + off]; }
        __syncthreads();
    }
    if (threadIdx.x == 0) { atomicAdd(&out[0], sd[0]); atomicAdd(&out[1], sn[0]); }
}

__global__ void bias_predict_kernel(int n, int M, double phib, double phic,
                                    double *b, double *c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] *= phib;
    if (i < M) c[i] *= phic;
}

__global__ void translate_Kt_kernel(int nrows, int nstate, const double *Kt, double *K) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nrows * nstate;
    for (; k < total; k += blockDim.x * gridDim.x) {
        int r = k % nrows, s = k / nrows;
        K[s + r * nstate] = Kt[r + s * nrows];
    }
}

__global__ void scale_sqrtR_kernel(int nrows, int nstate, int E, double sig_i, double sig_g,
                                   const double *Misl, const double *Mgts,
                                   const double *K, double *Ks) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nrows * nstate;
    for (; k < total; k += blockDim.x * gridDim.x) {
        int s = k % nstate, r = k / nstate;
        double R = (r < E) ? ((Misl[r] > 0.5) ? sig_i * sig_i : 1.0)
                           : ((Mgts[r - E] > 0.5) ? sig_g * sig_g : 1.0);
        Ks[k] = K[k] * sqrt(R);
    }
}

__global__ void count_nonfinite_kernel(size_t n, const double *x, int *out) {
    size_t k = blockIdx.x * blockDim.x + threadIdx.x;
    int c = 0;
    for (; k < n; k += (size_t)blockDim.x * gridDim.x) c += !isfinite(x[k]);
    if (c) atomicAdd(out, c);
}
__global__ void diag_copy_kernel(int n, const double *A, double *d) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) d[k] = A[k + (size_t)k * n];
}

__global__ void cast_f32_kernel(size_t n, const double *src, float *dst) {
    size_t k = blockIdx.x * blockDim.x + threadIdx.x;
    for (; k < n; k += (size_t)blockDim.x * gridDim.x) dst[k] = (float)src[k];
}
__global__ void cast_f64_kernel(size_t n, const float *src, double *dst) {
    size_t k = blockIdx.x * blockDim.x + threadIdx.x;
    for (; k < n; k += (size_t)blockDim.x * gridDim.x) dst[k] = (double)src[k];
}
__global__ void sub_into_P_kernel(int ns, double *P, const double *D) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = ns * ns;
    for (; k < total; k += blockDim.x * gridDim.x) P[k] -= D[k];
}

// 对角预条件：P̂ = D^{-1} P D^{-1}，PHt̂ = D^{-1} PHt，K̂ = D^{-1} K。
__global__ void scale_S_f32_kernel(int nr, const double *S, const double *esc, float *Sf) {
    size_t k = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = (size_t)nr * nr;
    for (; k < total; k += (size_t)blockDim.x * gridDim.x) {
        const int i = (int)(k % nr), j = (int)(k / nr);
        Sf[k] = (float)(S[k] / (esc[i] * esc[j]));
    }
}
__global__ void scale_rows_f32_kernel(int ns, int nr, const double *PHt, const double *dsc,
                                      const double *esc, float *PHtf) {
    size_t k = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = (size_t)ns * nr;
    for (; k < total; k += (size_t)blockDim.x * gridDim.x) {
        const int s = (int)(k % ns), b = (int)(k / ns);
        PHtf[k] = (float)(PHt[k] / (dsc[s] * esc[b]));
    }
}
__global__ void transpose_cast_f32_kernel(int ns, int nr, const float *A, float *B) {
    size_t k = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = (size_t)ns * nr;
    for (; k < total; k += (size_t)blockDim.x * gridDim.x) {
        const int s = (int)(k % ns), b = (int)(k / ns);
        B[b + (size_t)s * nr] = A[k];
    }
}
__global__ void unscale_transpose_f64_kernel(int ns, int nr, const float *B, const double *dsc,
                                             const double *esc, double *K) {
    size_t k = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = (size_t)ns * nr;
    for (; k < total; k += (size_t)blockDim.x * gridDim.x) {
        const int s = (int)(k % ns), b = (int)(k / ns);
        K[s + (size_t)b * ns] = dsc[s] * (double)B[b + (size_t)s * nr] / esc[b];
    }
}
// ΔP = D ΔP̂ D（FP64 回代），P ← P − ΔP。
__global__ void apply_delta_precond_kernel(int ns, const float *Df, const double *dsc, double *P) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = ns * ns;
    for (; k < total; k += blockDim.x * gridDim.x) {
        const int i = k % ns, j = k / ns;
        P[k] -= dsc[i] * (double)Df[k] * dsc[j];
    }
}
__global__ void add_shift_diag_kernel(int ns, double *P, double shift) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < ns) P[k + (size_t)k * ns] += shift;
}

__global__ void state_add_kernel(int n, int M, const double *dx, double *q, double *b, double *c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // dx 在归一化 z 空间：a 分量乘回 DU（x = S z）
        q[i * 6 + 0] += DU * dx[7 * i + 0];
        for (int a = 1; a < 6; ++a) q[i * 6 + a] += dx[7 * i + a];
        b[i] += dx[7 * i + 6];
    } else if (i < n + M) {
        int s = i - n; c[s] += dx[7 * n + s];
    }
}

double _elapsed(cudaEvent_t a, cudaEvent_t b) {
    float ms = 0.0; cudaEventElapsedTime(&ms, a, b); return ms;
}

void _free_ctx() {
    if (g.blas) cublasDestroy(g.blas);
    if (g.solver) cusolverDnDestroy(g.solver);
    for (int i = 0; i < 8; ++i)
        if (g.ev[i]) cudaEventDestroy(g.ev[i]);
    void *ptrs[] = {(void *)g.d_Ptmp, (void *)g.d_Ptmp2, (void *)g.d_A, (void *)g.d_H,
                    (void *)g.d_PHt, (void *)g.d_PHtT, (void *)g.d_S, (void *)g.d_Kt,
                    (void *)g.d_K, (void *)g.d_Ks, (void *)g.d_nu, (void *)g.d_dx,
                    (void *)g.d_Q, (void *)g.d_qpred, (void *)g.d_Phi, (void *)g.d_rv,
                    (void *)g.d_Ap, (void *)g.d_Disl, (void *)g.d_Misl, (void *)g.d_Dgts,
                    (void *)g.d_Mgts, (void *)g.d_Qgts, (void *)g.d_ei, (void *)g.d_ej,
                    (void *)g.d_ga, (void *)g.d_gs, (void *)g.d_hval, (void *)g.d_hidx,
                    (void *)g.d_Sf, (void *)g.d_PHtf, (void *)g.d_Kf, (void *)g.d_Df,
                    (void *)g.d_Bf, (void *)g.d_workf, (void *)g.d_ipiv, (void *)g.d_dscale,
                    (void *)g.d_escale, (void *)g.d_y, (void *)g.d_ext,
                    (void *)g.d_nf, (void *)g.d_diag,
                    (void *)g.d_work, (void *)g.d_eig,
                    (void *)g.d_info, (void *)g.d_sym};
    for (void *pp : ptrs) if (pp) cudaFree(pp);
    g = Ctx{};
}

void _alloc_ctx() {
    const int n = g.p.n, M = g.p.M, E = g.p.E, G = g.p.G;
    const size_t N = (size_t)g.p.nstate, R = (size_t)g.p.nrows;
    auto A = [](size_t bytes) { void *p = nullptr; cudaMalloc(&p, bytes); return p; };
    g.d_Ptmp = (double *)A(N * N * 8); g.d_Ptmp2 = (double *)A(N * N * 8);
    g.d_A = (double *)A(N * N * 8);
    g.d_H = (double *)A(R * N * 8); g.d_PHt = (double *)A(N * R * 8);
    g.d_PHtT = (double *)A(R * N * 8); g.d_S = (double *)A(R * R * 8);
    g.d_Kt = (double *)A(R * N * 8); g.d_K = (double *)A(N * R * 8);
    g.d_Ks = (double *)A(N * R * 8); g.d_nu = (double *)A(R * 8);
    g.d_dx = (double *)A(N * 8); g.d_Q = (double *)A((size_t)n * 49 * 8);
    g.d_qpred = (double *)A((size_t)n * 6 * 8); g.d_Phi = (double *)A((size_t)n * 36 * 8);
    g.d_rv = (double *)A((size_t)n * 6 * 8); g.d_Ap = (double *)A((size_t)n * 18 * 8);
    g.d_Disl = (double *)A((size_t)E * 8); g.d_Misl = (double *)A((size_t)E * 8);
    g.d_Dgts = (double *)A((size_t)G * 8); g.d_Mgts = (double *)A((size_t)G * 8);
    g.d_Qgts = (double *)A((size_t)G * 3 * 8);
    g.d_ei = (int *)A((size_t)E * 4); g.d_ej = (int *)A((size_t)E * 4);
    g.d_ga = (int *)A((size_t)G * 4); g.d_gs = (int *)A((size_t)G * 4);
    g.d_hval = (double *)A(R * 16 * 8); g.d_hidx = (int *)A(R * 16 * 4);
    g.d_ext = (double *)A((size_t)n * 3 * 8);
    if (g.p.fp32) {
        g.d_Sf = (float *)A(R * R * 4); g.d_PHtf = (float *)A(N * R * 4);
        g.d_Kf = (float *)A(N * R * 4); g.d_Df = (float *)A(N * N * 4);
        g.d_Bf = (float *)A(N * R * 4); g.d_ipiv = (int *)A(R * 4);
        g.d_dscale = (double *)A(N * 8); g.d_escale = (double *)A(R * 8);
        g.d_y = (double *)A(R * 8);
    }
    g.d_eig = (double *)A(N * 8); g.d_info = (int *)A(4);
    g.d_sym = (double *)A(2 * 8);
    g.d_nf = (int *)A(4); g.d_diag = (double *)A(std::max(N, R) * 8);
    cublasCreate(&g.blas); cusolverDnCreate(&g.solver);
    for (int i = 0; i < 8; ++i) cudaEventCreate(&g.ev[i]);
    int lw = 0, lw2 = 0;
    cusolverDnDpotrf_bufferSize(g.solver, CUBLAS_FILL_MODE_LOWER, g.p.nrows, g.d_S, g.p.nrows, &lw);
    cusolverDnDsyevd_bufferSize(g.solver, CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_LOWER,
                                g.p.nstate, g.d_P, g.p.nstate, g.d_eig, &lw2);
    g.lwork = std::max(lw, lw2) + 64;
    g.d_work = (double *)A((size_t)g.lwork * 8);
    if (g.p.fp32) {
        int lwf = 0;
        cusolverDnSgetrf_bufferSize(g.solver, g.p.nrows, g.p.nrows, g.d_Sf, g.p.nrows, &lwf);
        g.d_workf = (float *)A((size_t)std::max(lwf, 1) * 4);
    }
    g.init = true;
}

void _upload_edges(const int *edges, const int *gts) {
    const int n = g.p.n, E = g.p.E, G = g.p.G, NR = g.p.nrows;
    std::vector<int> ei(E), ej(E), ga(G), gs(G);
    for (int k = 0; k < E; ++k) { ei[k] = edges[2 * k]; ej[k] = edges[2 * k + 1]; }
    for (int k = 0; k < G; ++k) { ga[k] = gts[2 * k]; gs[k] = gts[2 * k + 1]; }
    cudaMemcpy(g.d_ei, ei.data(), E * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(g.d_ej, ej.data(), E * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(g.d_ga, ga.data(), G * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(g.d_gs, gs.data(), G * 4, cudaMemcpyHostToDevice);
    // H 的行稀疏列索引（固定，随图/站结构一次生成）
    std::vector<int> idx((size_t)NR * 16, 0);
    for (int e = 0; e < E; ++e) {
        const int i = ei[e], j = ej[e];
        for (int a = 0; a < 7; ++a) {
            idx[(size_t)e * 16 + a] = 7 * i + a;
            idx[(size_t)e * 16 + 7 + a] = 7 * j + a;
        }
    }
    for (int k = 0; k < G; ++k) {
        const int row = E + k, i = ga[k], s = gs[k];
        for (int a = 0; a < 7; ++a) idx[(size_t)row * 16 + a] = 7 * i + a;
        idx[(size_t)row * 16 + 7] = 7 * n + s;
    }
    cudaMemcpy(g.d_hidx, idx.data(), (size_t)NR * 16 * 4, cudaMemcpyHostToDevice);
}

}  // namespace

void release() { _free_ctx(); }

void predict(const Params& pin, const int* edges, const int* gts, const double* ext,
             double* q, double* b, double* c, double* P,
             int init, int frame, int do_diag, PredictResult* out) {
    Params p = pin;
    p.nstate = 7 * p.n + p.M; p.nrows = p.E + p.G;
    const bool dims_changed = !g.init || g.p.n != p.n || g.p.M != p.M || g.p.E != p.E ||
                              g.p.G != p.G || g.p.fp32 != p.fp32;
    if (dims_changed) _free_ctx();
    g.p = p;
    g.phi_b = std::exp(-10.0 / p.tau_b);
    g.qb = p.sigma_b * p.sigma_b * (1 - g.phi_b * g.phi_b);
    g.phi_c = std::exp(-10.0 / p.tau_c);
    g.qc = p.sigma_c * p.sigma_c * (1 - g.phi_c * g.phi_c);
    if (!g.init) _alloc_ctx();
    if (init || dims_changed) _upload_edges(edges, gts);
    g.d_q = q; g.d_b = b; g.d_c = c; g.d_P = P;
    const int n = p.n, M = p.M;
    cudaEventRecord(g.ev[0]);
    // GVE 3/8-RK4 状态 + 变分 STM：与批处理同一 tbp/RHS、同一 drag table 与逐星 β。
    const double dt = 10.0;
    qoejopt::gveStmDeviceGrp(g.d_q, n, dt, 1.0, g.d_qpred, g.d_Phi);
    cudaMemcpy(g.d_q, g.d_qpred, (size_t)n * 6 * 8, cudaMemcpyDeviceToDevice);
    bias_predict_kernel<<<(n + M + 127) / 128, 128>>>(n, M, g.phi_b, g.phi_c, g.d_b, g.d_c);
    cudaEventRecord(g.ev[1]);
    // Q 与外部力场修正全在 GPU：Qh=Ji Qrv Ji^T；ext 在 rv 空间加 0.5 a dt² / a dt 后映回 QOE。
    qoejopt::qoeQIsoDevice(g.d_q, n, p.q_acc, g.qb, g.d_Q);
    if (ext) {
        cudaMemcpyAsync(g.d_ext, ext, (size_t)n * 3 * 8, cudaMemcpyHostToDevice);
        qoejopt::qoeExtDevice(g.d_q, g.d_ext, n, dt, g.d_q);
    }
    {
        const int ns = g.p.nstate, BJ = 8;
        dim3 gnn(n, (n + BJ - 1) / BJ);
        fpft_node_kernel<<<gnn, 49>>>(n, ns, g.d_P, g.d_Ptmp, g.d_Phi, g.d_Q, g.phi_b, BJ);
        if (M > 0) {
            dim3 gsn(M, n);   // station-node
            fpft_kernel<<<gsn, 49>>>(n, M, ns, g.d_P, g.d_Ptmp, g.d_Phi, g.d_Q,
                                     g.phi_b, g.phi_c, g.qc, n, 0);
            dim3 gns(n, M);   // node-station
            fpft_kernel<<<gns, 49>>>(n, M, ns, g.d_P, g.d_Ptmp, g.d_Phi, g.d_Q,
                                     g.phi_b, g.phi_c, g.qc, 0, n);
            dim3 gss(M, M);   // station-station
            fpft_kernel<<<gss, 49>>>(n, M, ns, g.d_P, g.d_Ptmp, g.d_Phi, g.d_Q,
                                     g.phi_b, g.phi_c, g.qc, n, n);
        }
    }
    symmetrize_kernel<<<256, 256>>>(g.p.nstate, g.d_Ptmp);
    cudaEventRecord(g.ev[2]);
    cudaMemcpy(g.d_P, g.d_Ptmp, (size_t)g.p.nstate * g.p.nstate * 8, cudaMemcpyDeviceToDevice);
    cudaEventRecord(g.ev[3]);
    cudaEventSynchronize(g.ev[3]);
    // 逐帧有限性检查：P、q、b、c；预测后立即判死。
    int pnf = 0, snf = 0; double mind = INFINITY;
    cudaMemsetAsync(g.d_nf, 0, 4);
    count_nonfinite_kernel<<<4096, 256>>>((size_t)g.p.nstate * g.p.nstate, g.d_P, g.d_nf);
    cudaMemcpy(&pnf, g.d_nf, 4, cudaMemcpyDeviceToHost);
    cudaMemsetAsync(g.d_nf, 0, 4);
    count_nonfinite_kernel<<<256, 256>>>((size_t)n * 6, g.d_q, g.d_nf);
    count_nonfinite_kernel<<<256, 256>>>((size_t)n, g.d_b, g.d_nf);
    count_nonfinite_kernel<<<256, 256>>>((size_t)M, g.d_c, g.d_nf);
    cudaMemcpy(&snf, g.d_nf, 4, cudaMemcpyDeviceToHost);
    {
        std::vector<double> dg(g.p.nstate);
        diag_copy_kernel<<<(g.p.nstate + 255) / 256, 256>>>(g.p.nstate, g.d_P, g.d_diag);
        cudaMemcpy(dg.data(), g.d_diag, (size_t)g.p.nstate * 8, cudaMemcpyDeviceToHost);
        for (double v : dg) mind = std::isfinite(v) ? std::min(mind, v) : NAN;
        if (!std::isfinite(mind)) mind = NAN;
    }
    if (pnf > 0 || snf > 0 || !std::isfinite(mind)) {
        fprintf(stderr, "[ekf_central] STOP predict frame=%d P_nonfinite=%d state_nonfinite=%d "
                        "min_diag=%g\n", frame, pnf, snf, mind);
        throw std::runtime_error("ekf_central predict: nonfinite at frame " + std::to_string(frame));
    }
    if (out) {
        out->P_nonfinite = pnf; out->state_nonfinite = snf; out->min_diag = mind;
        out->pred_ms = _elapsed(g.ev[0], g.ev[1]);
        out->fpft_ms = _elapsed(g.ev[1], g.ev[2]);
        out->copy_ms = _elapsed(g.ev[2], g.ev[3]);
        out->phi_b = g.phi_b; out->phi_c = g.phi_c; out->qb = g.qb; out->qc = g.qc;
        out->diag = (do_diag == 2);
        if (do_diag == 2) {
            out->Phi.resize((size_t)n * 36); out->Q.resize((size_t)n * 49);
            cudaMemcpy(out->Phi.data(), g.d_Phi, (size_t)n * 36 * 8, cudaMemcpyDeviceToHost);
            cudaMemcpy(out->Q.data(), g.d_Q, (size_t)n * 49 * 8, cudaMemcpyDeviceToHost);
        }
    }
}

void update(const double* D_isl, const double* M_isl,
            const double* D_gts, const double* M_gts, const double* Q_gts,
            int frame, int do_diag, UpdateResult* out) {
    if (!g.init) throw std::runtime_error("update before predict");
    const int n = g.p.n, M = g.p.M, E = g.p.E, G = g.p.G, NR = g.p.nrows, NS = g.p.nstate;
    cudaMemcpy(g.d_Disl, D_isl, (size_t)E * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(g.d_Misl, M_isl, (size_t)E * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(g.d_Dgts, D_gts, (size_t)G * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(g.d_Mgts, M_gts, (size_t)G * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(g.d_Qgts, Q_gts, (size_t)G * 3 * 8, cudaMemcpyHostToDevice);
    cudaEventRecord(g.ev[0]);
    qoejopt::oeRvJacDevice(g.d_q, n, g.d_rv, g.d_Ap);
    cudaMemsetAsync(g.d_H, 0, (size_t)NR * NS * 8);
    cudaMemsetAsync(g.d_hval, 0, (size_t)NR * 16 * 8);
    h_assemble_kernel<<<(NR + 127) / 128, 128>>>(n, M, E, G, NR, NS,
        g.d_ei, g.d_ej, g.d_ga, g.d_gs, g.d_rv, g.d_Ap,
        g.d_Disl, g.d_Misl, g.d_Dgts, g.d_Mgts, g.d_Qgts, g.d_b, g.d_c,
        g.d_H, g.d_nu, g.d_hval);
    cudaEventRecord(g.ev[1]);
    // 块稀疏：PHt[:,b] = Σ h_b,k P[:,idx_b,k]；S = H PHt（每行 14 个非零）。
    phT_sparse_kernel<<<NR, 256>>>(NS, NR, g.d_P, g.d_hval, g.d_hidx, g.d_PHt);
    cudaEventRecord(g.ev[2]);
    {
        const int total = NR * NR;
        S_sparse_kernel<<<(total + 255) / 256, 256>>>(NS, NR, g.d_PHt, g.d_hval, g.d_hidx, g.d_S);
    }
    add_R_kernel<<<(NR + 255) / 256, 256>>>(NR, E, g.p.sigma_isl, g.p.sigma_gts,
                                            g.d_Misl, g.d_Mgts, g.d_S);
    symmetrize_kernel<<<256, 256>>>(NR, g.d_S);
    cudaEventRecord(g.ev[3]);
    {
        int nf_ph = 0, nf_s = 0;
        cudaMemsetAsync(g.d_nf, 0, 4);
        count_nonfinite_kernel<<<4096, 256>>>((size_t)NS * NR, g.d_PHt, g.d_nf);
        cudaMemcpy(&nf_ph, g.d_nf, 4, cudaMemcpyDeviceToHost);
        cudaMemsetAsync(g.d_nf, 0, 4);
        count_nonfinite_kernel<<<2048, 256>>>((size_t)NR * NR, g.d_S, g.d_nf);
        cudaMemcpy(&nf_s, g.d_nf, 4, cudaMemcpyDeviceToHost);
        if (nf_ph > 0 || nf_s > 0) {
            fprintf(stderr, "[ekf_central] STOP update frame=%d stage=%s PHt_nf=%d S_nf=%d\n",
                    frame, nf_ph ? "PHt" : "S", nf_ph, nf_s);
            throw std::runtime_error("ekf_central update: nonfinite H/S at frame " +
                                     std::to_string(frame));
        }
    }
    double s_min = NAN, s_max = NAN, s_cond_chol = NAN;
    const bool want_diag = do_diag && (g.p.diag_every > 0) && (frame % g.p.diag_every == 0);
    if (want_diag) {
        cudaMemcpyAsync(g.d_Ptmp2, g.d_S, (size_t)NR * NR * 8, cudaMemcpyDeviceToDevice);
        _cusolver(cusolverDnDsyevd(g.solver, CUSOLVER_EIG_MODE_NOVECTOR,
                                   CUBLAS_FILL_MODE_LOWER, NR, g.d_Ptmp2, NR,
                                   g.d_eig, g.d_work, g.lwork, g.d_info), "syevd_S");
        int _si = 0; cudaMemcpy(&_si, g.d_info, 4, cudaMemcpyDeviceToHost);
        if (_si == 0) {
            cudaMemcpy(&s_min, g.d_eig, 8, cudaMemcpyDeviceToHost);
            cudaMemcpy(&s_max, g.d_eig + (NR - 1), 8, cudaMemcpyDeviceToHost);
        }
    }
    int potrf_info = 0;
    if (!g.p.fp32) {
        _cusolver(cusolverDnDpotrf(g.solver, CUBLAS_FILL_MODE_LOWER, NR, g.d_S, NR,
                                   g.d_work, g.lwork, g.d_info), "potrf");
        cudaMemcpy(&potrf_info, g.d_info, 4, cudaMemcpyDeviceToHost);
        if (potrf_info != 0) {
            fprintf(stderr, "[ekf_central] potrf info=%d NR=%d S_min=%g S_max=%g\n",
                    potrf_info, NR, s_min, s_max);
            throw std::runtime_error("potrf failed info=" + std::to_string(potrf_info));
        }
        std::vector<double> ld(NR);
        diag_copy_kernel<<<(NR + 255) / 256, 256>>>(NR, g.d_S, g.d_diag);
        cudaMemcpy(ld.data(), g.d_diag, (size_t)NR * 8, cudaMemcpyDeviceToHost);
        double lo = INFINITY, hi = 0.0;
        for (double v : ld) { lo = std::min(lo, std::fabs(v)); hi = std::max(hi, std::fabs(v)); }
        if (lo > 0.0 && std::isfinite(hi)) s_cond_chol = (hi / lo) * (hi / lo);
        if (!std::isfinite(s_cond_chol)) {
            fprintf(stderr, "[ekf_central] STOP update frame=%d stage=S_chol Lmin=%g Lmax=%g\n",
                    frame, lo, hi);
            throw std::runtime_error("ekf_central update: bad Cholesky at frame " +
                                     std::to_string(frame));
        }
    } else if (want_diag) {
        s_cond_chol = (std::isfinite(s_min) && s_min > 0.0) ? s_max / s_min : NAN;
    }
    cudaEventRecord(g.ev[4]);
    bool dx_done = false;
    if (g.p.fp32) {
        // 对角预条件 + FP32：d=sqrt(diag(P))，PHt̂=D^{-1}PHt，S 用 FP32 LU (Sgetrf/Sgetrs)
        // 解 K̂=D^{-1}K，再回代 K=D K̂。S/P/归一化仍 FP64 存储。
        {
            std::vector<double> dg(NS), dsc(NS), sg(NR), esc(NR);
            diag_copy_kernel<<<(NS + 255) / 256, 256>>>(NS, g.d_P, g.d_diag);
            cudaMemcpy(dg.data(), g.d_diag, (size_t)NS * 8, cudaMemcpyDeviceToHost);
            for (int i = 0; i < NS; ++i) dsc[i] = std::sqrt(std::max(dg[i], 1e-300));
            cudaMemcpy(g.d_dscale, dsc.data(), (size_t)NS * 8, cudaMemcpyHostToDevice);
            diag_copy_kernel<<<(NR + 255) / 256, 256>>>(NR, g.d_S, g.d_diag);
            cudaMemcpy(sg.data(), g.d_diag, (size_t)NR * 8, cudaMemcpyDeviceToHost);
            for (int i = 0; i < NR; ++i) esc[i] = std::sqrt(std::max(sg[i], 1e-300));
            cudaMemcpy(g.d_escale, esc.data(), (size_t)NR * 8, cudaMemcpyHostToDevice);
        }
        scale_S_f32_kernel<<<4096, 256>>>(NR, g.d_S, g.d_escale, g.d_Sf);
        scale_rows_f32_kernel<<<4096, 256>>>(NS, NR, g.d_PHt, g.d_dscale, g.d_escale, g.d_PHtf);
        transpose_cast_f32_kernel<<<4096, 256>>>(NS, NR, g.d_PHtf, g.d_Bf);
        _cusolver(cusolverDnSgetrf(g.solver, NR, NR, g.d_Sf, NR, g.d_workf, g.d_ipiv,
                                   g.d_info), "Sgetrf");
        cudaMemcpy(&potrf_info, g.d_info, 4, cudaMemcpyDeviceToHost);
        if (potrf_info != 0) {
            fprintf(stderr, "[ekf_central] Sgetrf info=%d NR=%d S_min=%g S_max=%g\n",
                    potrf_info, NR, s_min, s_max);
            throw std::runtime_error("Sgetrf failed info=" + std::to_string(potrf_info));
        }
        _cusolver(cusolverDnSgetrs(g.solver, CUBLAS_OP_N, NR, NS, g.d_Sf, NR, g.d_ipiv,
                                   g.d_Bf, NR, g.d_info), "Sgetrs");
        unscale_transpose_f64_kernel<<<4096, 256>>>(NS, NR, g.d_Bf, g.d_dscale, g.d_escale, g.d_K);
        // 状态增量 dx 用 FP64 单 RHS：S y = ν，dx = PHt y（弱方向不受 FP32 K 影响）。
        cudaMemcpy(g.d_y, g.d_nu, (size_t)NR * 8, cudaMemcpyDeviceToDevice);
        _cusolver(cusolverDnDpotrf(g.solver, CUBLAS_FILL_MODE_LOWER, NR, g.d_S, NR,
                                   g.d_work, g.lwork, g.d_info), "potrf_dx");
        int _ci = 0; cudaMemcpy(&_ci, g.d_info, 4, cudaMemcpyDeviceToHost);
        if (_ci == 0) {
            _cusolver(cusolverDnDpotrs(g.solver, CUBLAS_FILL_MODE_LOWER, NR, 1,
                                       g.d_S, NR, g.d_y, NR, g.d_info), "potrs_dx");
            _gemv(g.blas, CUBLAS_OP_N, NS, NR, 1.0, g.d_PHt, NS, g.d_y, 1, 0.0, g.d_dx, 1,
                  "PHt_y");
            dx_done = true;
            if (!want_diag) {
                std::vector<double> ld(NR);
                diag_copy_kernel<<<(NR + 255) / 256, 256>>>(NR, g.d_S, g.d_diag);
                cudaMemcpy(ld.data(), g.d_diag, (size_t)NR * 8, cudaMemcpyDeviceToHost);
                double lo = INFINITY, hi = 0.0;
                for (double v : ld) { lo = std::min(lo, std::fabs(v)); hi = std::max(hi, std::fabs(v)); }
                if (lo > 0.0 && std::isfinite(hi)) s_cond_chol = (hi / lo) * (hi / lo);
            }
        } else {
            fprintf(stderr, "[ekf_central] potrf_dx info=%d frame=%d: fallback to FP32 K for dx\n",
                    _ci, frame);
        }
    } else {
        transpose_kernel<<<256, 256>>>(NR, NS, g.d_PHt, g.d_PHtT);
        _cusolver(cusolverDnDpotrs(g.solver, CUBLAS_FILL_MODE_LOWER, NR, NS,
                                   g.d_S, NR, g.d_PHtT, NR, g.d_info), "potrs");
        translate_Kt_kernel<<<256, 256>>>(NR, NS, g.d_PHtT, g.d_K);
    }
    cudaEventRecord(g.ev[5]);
    cudaEventRecord(g.ev[6]);
    if (!dx_done)
        _gemv(g.blas, CUBLAS_OP_N, NS, NR, 1.0, g.d_K, NS, g.d_nu, 1, 0.0, g.d_dx, 1, "Knu");
    state_add_kernel<<<(n + M + 127) / 128, 128>>>(n, M, g.d_dx, g.d_q, g.d_b, g.d_c);
    retract_q_kernel<<<(n + 127) / 128, 128>>>(n, g.d_q);
    if (g.p.joseph) {
        _gemm(g.blas, CUBLAS_OP_N, CUBLAS_OP_N, NS, NS, NR, 1.0, g.d_K, NS, g.d_H, NR, 0.0, g.d_A, NS, "KH");
        make_A_kernel<<<256, 256>>>(NS, g.d_A);
        _gemm(g.blas, CUBLAS_OP_N, CUBLAS_OP_N, NS, NS, NS, 1.0, g.d_A, NS, g.d_P, NS, 0.0, g.d_Ptmp, NS, "AP");
        _gemm(g.blas, CUBLAS_OP_N, CUBLAS_OP_T, NS, NS, NS, 1.0, g.d_Ptmp, NS, g.d_A, NS, 0.0, g.d_Ptmp2, NS, "APA");
        scale_sqrtR_kernel<<<256, 256>>>(NR, NS, E, g.p.sigma_isl, g.p.sigma_gts,
                                         g.d_Misl, g.d_Mgts, g.d_K, g.d_Ks);
        _gemm(g.blas, CUBLAS_OP_N, CUBLAS_OP_T, NS, NS, NR, 1.0, g.d_Ks, NS, g.d_Ks, NS, 0.0, g.d_A, NS, "KRKt");
        add_mat_kernel<<<256, 256>>>(NS, g.d_Ptmp2, g.d_A);
        symmetrize_kernel<<<256, 256>>>(NS, g.d_Ptmp2);
        cudaMemcpy(g.d_P, g.d_Ptmp2, (size_t)NS * NS * 8, cudaMemcpyDeviceToDevice);
    } else if (g.p.fp32) {
        // ΔP̂ = PHt̂ K̂^T（FP32），ΔP = D ΔP̂ D（FP64 回代），P ← P − ΔP。
        const float one = 1.0f, zero = 0.0f;
        // Bf 已是 K̂^T (NR x NS)，故用 N,N：PHt̂ * K̂^T。
        _cublas(cublasSgemm(g.blas, CUBLAS_OP_N, CUBLAS_OP_N, NS, NS, NR, &one,
                            g.d_PHtf, NS, g.d_Bf, NR, &zero, g.d_Df, NS), "SGEMM dP");
        apply_delta_precond_kernel<<<4096, 256>>>(NS, g.d_Df, g.d_dscale, g.d_P);
        symmetrize_kernel<<<256, 256>>>(NS, g.d_P);
    } else {
        _gemm(g.blas, CUBLAS_OP_N, CUBLAS_OP_T, NS, NS, NR, -1.0, g.d_PHt, NS, g.d_K, NS, 1.0, g.d_Ptmp, NS, "P-KSKt");
        symmetrize_kernel<<<256, 256>>>(NS, g.d_Ptmp);
        cudaMemcpy(g.d_P, g.d_Ptmp, (size_t)NS * NS * 8, cudaMemcpyDeviceToDevice);
    }
    cudaMemsetAsync(g.d_sym, 0, 2 * 8);
    sym_ratio_kernel<<<256, 256>>>(NS, g.d_P, g.d_sym);
    double sym[2] = {0.0, 0.0};
    cudaMemcpy(sym, g.d_sym, 2 * 8, cudaMemcpyDeviceToHost);
    double mineig = NAN;
    if (want_diag) {
        cudaMemcpy(g.d_Ptmp, g.d_P, (size_t)NS * NS * 8, cudaMemcpyDeviceToDevice);
        _cusolver(cusolverDnDsyevd(g.solver, CUSOLVER_EIG_MODE_NOVECTOR,
                                   CUBLAS_FILL_MODE_LOWER, NS, g.d_Ptmp, NS,
                                   g.d_eig, g.d_work, g.lwork, g.d_info), "syevd");
        int _pi = 0; cudaMemcpy(&_pi, g.d_info, 4, cudaMemcpyDeviceToHost);
        if (_pi == 0) cudaMemcpy(&mineig, g.d_eig, 8, cudaMemcpyDeviceToHost);
    }
    // 最小特征值兜底：诊断帧出现负值时加对角平移，把 min eig 抬到 +1e-15。
    if (want_diag && std::isfinite(mineig) && mineig < 0.0) {
        const double shift = -mineig + 1e-15;
        add_shift_diag_kernel<<<(NS + 255) / 256, 256>>>(NS, g.d_P, shift);
        mineig += shift;
    }
    // 逐帧终检：P/状态有限、对角为正；诊断帧再查 min eig。
    int pnf = 0, snf = 0; double mind = INFINITY;
    cudaMemsetAsync(g.d_nf, 0, 4);
    count_nonfinite_kernel<<<4096, 256>>>((size_t)NS * NS, g.d_P, g.d_nf);
    cudaMemcpy(&pnf, g.d_nf, 4, cudaMemcpyDeviceToHost);
    cudaMemsetAsync(g.d_nf, 0, 4);
    count_nonfinite_kernel<<<256, 256>>>((size_t)n * 6, g.d_q, g.d_nf);
    count_nonfinite_kernel<<<256, 256>>>((size_t)n, g.d_b, g.d_nf);
    count_nonfinite_kernel<<<256, 256>>>((size_t)M, g.d_c, g.d_nf);
    cudaMemcpy(&snf, g.d_nf, 4, cudaMemcpyDeviceToHost);
    {
        std::vector<double> dg(NS);
        diag_copy_kernel<<<(NS + 255) / 256, 256>>>(NS, g.d_P, g.d_diag);
        cudaMemcpy(dg.data(), g.d_diag, (size_t)NS * 8, cudaMemcpyDeviceToHost);
        for (double v : dg) mind = std::isfinite(v) ? std::min(mind, v) : NAN;
        if (!std::isfinite(mind)) mind = NAN;
    }
    // 对角正性兜底（每帧）：负对角加统一平移，保持对称且不引入非有限值。
    if (std::isfinite(mind) && mind <= 0.0) {
        add_shift_diag_kernel<<<(NS + 255) / 256, 256>>>(NS, g.d_P, -mind + 1e-15);
        mind = 1e-15;
    }
    const bool bad = pnf > 0 || snf > 0 || !std::isfinite(mind) ||
                     (want_diag && (!std::isfinite(mineig) || mineig < 0.0));
    if (bad) {
        fprintf(stderr, "[ekf_central] STOP update frame=%d S_cond=%.3e P_nonfinite=%d "
                        "state_nonfinite=%d min_diag=%g min_eig=%g diag_done=%d\n",
                frame, s_cond_chol, pnf, snf, mind, mineig, (int)want_diag);
        throw std::runtime_error("ekf_central update: nonfinite/negative at frame " +
                                 std::to_string(frame));
    }
    cudaEventRecord(g.ev[7]);
    cudaEventSynchronize(g.ev[7]);
    if (out) {
        out->h_ms = _elapsed(g.ev[0], g.ev[1]);
        out->PHt_ms = _elapsed(g.ev[1], g.ev[2]);
        out->S_ms = _elapsed(g.ev[2], g.ev[3]);
        out->potrf_ms = _elapsed(g.ev[3], g.ev[4]);
        out->K_ms = _elapsed(g.ev[4], g.ev[5]);
        out->transpose_ms = _elapsed(g.ev[5], g.ev[6]);
        out->Pupd_ms = _elapsed(g.ev[6], g.ev[7]);
        out->total_ms = _elapsed(g.ev[0], g.ev[7]);
        out->min_eig = mineig;
        out->sym_rel = sqrt(sym[0]) / std::max(sqrt(sym[1]), 1e-300);
        out->S_min_eig = s_min; out->S_max_eig = s_max;
        out->S_cond = (std::isfinite(s_min) && s_min > 0.0) ? s_max / s_min : NAN;
        out->S_cond_chol = s_cond_chol;
        out->potrf_info = potrf_info;
        out->diag = (do_diag == 2);
        out->diag_done = want_diag;
        out->P_nonfinite = pnf; out->state_nonfinite = snf; out->min_diag = mind;
        out->nstate = NS; out->nrows = NR;
        if (do_diag == 2) {
            out->H_colT.resize((size_t)NR * NS); out->K.resize((size_t)NS * NR);
            out->nu.resize((size_t)NR); out->dx.resize((size_t)NS);
            out->rv.resize((size_t)n * 6);
            cudaMemcpy(out->H_colT.data(), g.d_H, out->H_colT.size() * 8, cudaMemcpyDeviceToHost);
            cudaMemcpy(out->K.data(), g.d_K, out->K.size() * 8, cudaMemcpyDeviceToHost);
            cudaMemcpy(out->nu.data(), g.d_nu, out->nu.size() * 8, cudaMemcpyDeviceToHost);
            cudaMemcpy(out->dx.data(), g.d_dx, out->dx.size() * 8, cudaMemcpyDeviceToHost);
            cudaMemcpy(out->rv.data(), g.d_rv, out->rv.size() * 8, cudaMemcpyDeviceToHost);
        }
    }
}

}  // namespace ekf_central

#ifndef EKF_CENTRAL_NO_PYBIND
namespace {
py::dict ekf_predict_py(int n, int M, int E, int G, py::array_t<int> edges,
                        py::array_t<int> gts, py::array_t<double> ext,
                        uint64_t q_ptr, uint64_t b_ptr, uint64_t c_ptr,
                        uint64_t P_ptr, double sigma_isl, double sigma_gts,
                        double tau_b, double sigma_b, double tau_c, double sigma_c,
                        double q_acc, int joseph, int diag_every, int fp32, int init, int frame,
                        int do_diag) {
    ekf_central::Params p;
    p.n = n; p.M = M; p.E = E; p.G = G;
    p.sigma_isl = sigma_isl; p.sigma_gts = sigma_gts;
    p.tau_b = tau_b; p.sigma_b = sigma_b; p.tau_c = tau_c; p.sigma_c = sigma_c;
    p.q_acc = q_acc; p.joseph = joseph; p.diag_every = diag_every; p.fp32 = fp32;
    auto e = edges.unchecked<2>(); auto gt = gts.unchecked<2>();
    std::vector<int> ev((size_t)E * 2), gv((size_t)G * 2);
    for (int k = 0; k < E; ++k) { ev[2 * k] = e(k, 0); ev[2 * k + 1] = e(k, 1); }
    for (int k = 0; k < G; ++k) { gv[2 * k] = gt(k, 0); gv[2 * k + 1] = gt(k, 1); }
    std::vector<double> xv;
    if (ext.size() > 0) {
        auto ea = ext.unchecked<2>();
        xv.resize((size_t)n * 3);
        for (int i = 0; i < n; ++i)
            for (int k = 0; k < 3; ++k) xv[(size_t)i * 3 + k] = ea(i, k);
    }
    ekf_central::PredictResult r;
    ekf_central::predict(p, ev.data(), gv.data(), xv.empty() ? nullptr : xv.data(),
                         (double *)q_ptr, (double *)b_ptr,
                         (double *)c_ptr, (double *)P_ptr, init, frame, do_diag, &r);
    py::dict d;
    d["pred_ms"] = r.pred_ms; d["fpft_ms"] = r.fpft_ms; d["copy_ms"] = r.copy_ms;
    if (r.diag) {
        py::array_t<double> phi({n, 6, 6}), qo({n, 7, 7});
        std::memcpy(phi.mutable_data(), r.Phi.data(), r.Phi.size() * 8);
        std::memcpy(qo.mutable_data(), r.Q.data(), r.Q.size() * 8);
        d["Phi"] = phi; d["Q"] = qo;
        d["phi_b"] = r.phi_b; d["phi_c"] = r.phi_c; d["qb"] = r.qb; d["qc"] = r.qc;
    }
    return d;
}

py::dict ekf_update_py(py::array_t<double> Disl, py::array_t<double> Misl,
                       py::array_t<double> Dgts, py::array_t<double> Mgts,
                       py::array_t<double> Qgts, int frame, int do_diag) {
    ekf_central::UpdateResult r;
    ekf_central::update(Disl.data(), Misl.data(), Dgts.data(), Mgts.data(), Qgts.data(),
                        frame, do_diag, &r);
    py::dict d;
    d["h_ms"] = r.h_ms; d["PHt_ms"] = r.PHt_ms; d["S_ms"] = r.S_ms;
    d["potrf_ms"] = r.potrf_ms; d["K_ms"] = r.K_ms; d["transpose_ms"] = r.transpose_ms;
    d["Pupd_ms"] = r.Pupd_ms; d["total_ms"] = r.total_ms;
    d["min_eig"] = r.min_eig; d["sym_rel"] = r.sym_rel;
    d["S_min_eig"] = r.S_min_eig; d["S_max_eig"] = r.S_max_eig; d["S_cond"] = r.S_cond;
    d["S_cond_chol"] = r.S_cond_chol;
    d["potrf_info"] = r.potrf_info;
    d["diag_done"] = r.diag_done;
    d["P_nonfinite"] = r.P_nonfinite; d["state_nonfinite"] = r.state_nonfinite;
    d["min_diag"] = r.min_diag;
    if (r.diag) {
        // H_colT: (nstate,nrows) C 序（即列主序 H 的内存，H^T）；K: (nrows,nstate) C 序（K^T）。
        py::array_t<double> Hc({r.nstate, r.nrows}), Kc({r.nrows, r.nstate});
        py::array_t<double> Nv({r.nrows}), Dv({r.nstate});
        std::memcpy(Hc.mutable_data(), r.H_colT.data(), r.H_colT.size() * 8);
        std::memcpy(Kc.mutable_data(), r.K.data(), r.K.size() * 8);
        std::memcpy(Nv.mutable_data(), r.nu.data(), r.nu.size() * 8);
        std::memcpy(Dv.mutable_data(), r.dx.data(), r.dx.size() * 8);
        d["H_colT"] = Hc; d["K"] = Kc; d["nu"] = Nv; d["dx"] = Dv;
    }
    return d;
}
}  // namespace

PYBIND11_MODULE(ekf_central, m) {
    m.def("ekf_predict", &ekf_predict_py);
    m.def("ekf_update", &ekf_update_py);
}
#endif  // EKF_CENTRAL_NO_PYBIND
