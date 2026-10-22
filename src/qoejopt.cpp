// qoejopt：joint 单发全局联合内环的 C++/CUDA 模块（P4.1）。
//
// 只暴露"结构/观测一次 setup + 每次 assemble 写 A_val/b（Python 提供 torch cuda 缓冲 data_ptr）"，
// 中间量 P/A 只驻显存；重活（GVE/STM 折叠/装配/res2）全部在本模块 CUDA 核内完成，见 src/gve_step.cu。
// Python 侧保留 thin LM 循环与 Theseus Baspacho（外环 θ 学习沿用 BaspachoSolveFunction 原语链式）。
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <stdexcept>
#include <string>
#include <array>
#include <vector>
#include "qoejopt.h"
#include "kepler.h"
#include <Eigen/Dense>
#include <cusolverDn.h>
#include <cublas_v2.h>

namespace py = pybind11;
using qoejopt::Ctx;

static Ctx g_ctx;

static void _free_ctx() {
    Ctx &c = g_ctx;
    cudaFree(c.d_isl_i); cudaFree(c.d_isl_jj); cudaFree(c.d_gts_a);
    cudaFree(c.d_scale); c.d_scale = nullptr;
    cudaFree(c.d_isl_D); cudaFree(c.d_isl_W);
    cudaFree(c.d_gts_D); cudaFree(c.d_gts_W); cudaFree(c.d_gts_Q);
    cudaFree(c.d_ctr);
    cudaFree(c.d_beta); c.d_beta = nullptr;
    cudaFree(c.d_P0); cudaFree(c.d_prior_fac); cudaFree(c.d_zprior);
    cudaFree(c.d_cov); cudaFree(c.d_phi);
    cudaFree(c.d_P0rv); cudaFree(c.d_Pm); cudaFree(c.d_Pp); cudaFree(c.d_PpPrev);
    cudaFree(c.d_adj_j); cudaFree(c.d_adj_w);
    cudaFree(c.d_X); cudaFree(c.d_rv0); cudaFree(c.d_A0);
    cudaFree(c.d_P); cudaFree(c.d_A); cudaFree(c.d_phi_all); cudaFree(c.d_Pw); cudaFree(c.d_res);
    g_ctx = Ctx{};
}

template <typename T>
static T *_upload(const py::array_t<T, py::array::c_style | py::array::forcecast> &a) {
    auto buf = a.request();
    const std::size_t bytes = (std::size_t)buf.size * sizeof(T);
    T *d = nullptr;
    if (bytes) {
        if (cudaMalloc(&d, bytes) != cudaSuccess) throw std::runtime_error("qoejopt: cudaMalloc 失败");
        if (cudaMemcpy(d, buf.ptr, bytes, cudaMemcpyHostToDevice) != cudaSuccess)
            throw std::runtime_error("qoejopt: cudaMemcpy H2D 失败");
    }
    return d;
}

template <typename T>
static void _copy_to_dev(T *d, const py::array_t<T, py::array::c_style | py::array::forcecast> &a,
                         std::size_t expect) {
    auto buf = a.request();
    if ((std::size_t)buf.size != expect)
        throw std::runtime_error("qoejopt: 输入长度不符");
    cudaMemcpy(d, buf.ptr, expect * sizeof(T), cudaMemcpyHostToDevice);
}

static void setup(int n, int nfr, int E, int G, int ncol, int nnz, int nrows,
                  int off_isl, int off_gts, int off_damp,
                  double dt, double s_isl, double s_gts, double issq,
                  py::array_t<int, py::array::c_style | py::array::forcecast> isl_i,
                  py::array_t<int, py::array::c_style | py::array::forcecast> isl_jj,
                  py::array_t<double, py::array::c_style | py::array::forcecast> isl_D,
                  py::array_t<double, py::array::c_style | py::array::forcecast> isl_W,
                  py::array_t<int, py::array::c_style | py::array::forcecast> gts_a,
                  py::array_t<double, py::array::c_style | py::array::forcecast> gts_D,
                  py::array_t<double, py::array::c_style | py::array::forcecast> gts_W,
                  py::array_t<double, py::array::c_style | py::array::forcecast> gts_Q,
                  py::array_t<double, py::array::c_style | py::array::forcecast> ctr) {
    _free_ctx();
    Ctx &c = g_ctx;
    c.n = n; c.nfr = nfr; c.E = E; c.G = G; c.ncol = ncol; c.nnz = nnz; c.nrows = nrows;
    c.off_isl = off_isl; c.off_gts = off_gts; c.off_damp = off_damp;
    c.dt = dt; c.s_isl = s_isl; c.s_gts = s_gts; c.issq = issq;

    const std::size_t nISL = (std::size_t)nfr * E, nGTS = (std::size_t)nfr * G;
    c.d_isl_i  = _upload(isl_i);
    c.d_isl_jj = _upload(isl_jj);
    c.d_isl_D  = _upload(isl_D);
    c.d_isl_W  = _upload(isl_W);
    c.d_gts_a  = _upload(gts_a);
    c.d_gts_D  = _upload(gts_D);
    c.d_gts_W  = _upload(gts_W);
    c.d_gts_Q  = _upload(gts_Q);
    c.d_ctr    = _upload(ctr);

    if (c.d_isl_i && isl_i.size() != (py::ssize_t)nISL)
        throw std::runtime_error("qoejopt: isl_i 长度 != nfr*E");
    if (nGTS > 0 && (!c.d_gts_a || gts_a.size() != (py::ssize_t)nGTS))
        throw std::runtime_error("qoejopt: gts_a 长度 != nfr*G");

    const std::size_t bX = (std::size_t)n * 6 * sizeof(double);
    const std::size_t bA0 = (std::size_t)n * 36 * sizeof(double);
    const std::size_t bP = (std::size_t)nfr * n * 6 * sizeof(double);
    const std::size_t bA = (std::size_t)nfr * n * 18 * sizeof(double);
    cudaMalloc(&c.d_X, bX);
    cudaMalloc(&c.d_rv0, bX);
    cudaMalloc(&c.d_A0, bA0);
    cudaMalloc(&c.d_P, bP);
    cudaMalloc(&c.d_A, bA);
    cudaMalloc(&c.d_phi_all, (std::size_t)nfr * n * 36 * sizeof(double));
    cudaMalloc(&c.d_Pw, (std::size_t)n * 36 * sizeof(double));
    cudaMalloc(&c.d_res, sizeof(double));
}

static double assemble(py::array_t<double, py::array::c_style | py::array::forcecast> x,
                       py::array_t<double, py::array::c_style | py::array::forcecast> rv0,
                       py::array_t<double, py::array::c_style | py::array::forcecast> A0,
                       uint64_t Aval_ptr, uint64_t b_ptr) {
    Ctx &c = g_ctx;
    if (!c.d_X) throw std::runtime_error("qoejopt.assemble: 未先 setup");
    _copy_to_dev(c.d_X, x, (std::size_t)c.n * 6);
    _copy_to_dev(c.d_rv0, rv0, (std::size_t)c.n * 6);
    _copy_to_dev(c.d_A0, A0, (std::size_t)c.n * 36);
    py::gil_scoped_release rel;
    return qoejopt::jointAssembleInto(c, reinterpret_cast<double *>(Aval_ptr),
                                      reinterpret_cast<double *>(b_ptr));
}

static double expand_res2(py::array_t<double, py::array::c_style | py::array::forcecast> x,
                          py::array_t<double, py::array::c_style | py::array::forcecast> rv0,
                          py::array_t<double, py::array::c_style | py::array::forcecast> A0) {
    Ctx &c = g_ctx;
    if (!c.d_X) throw std::runtime_error("qoejopt.expand: 未先 setup");
    _copy_to_dev(c.d_X, x, (std::size_t)c.n * 6);
    _copy_to_dev(c.d_rv0, rv0, (std::size_t)c.n * 6);
    _copy_to_dev(c.d_A0, A0, (std::size_t)c.n * 36);
    py::gil_scoped_release rel;
    return qoejopt::jointIslRes2(c);
}

static double assemble_cached(uint64_t Aval_ptr, uint64_t b_ptr) {
    Ctx &c = g_ctx;
    if (!c.d_X) throw std::runtime_error("qoejopt.assemble_cached: 未先 setup");
    py::gil_scoped_release rel;
    return qoejopt::jointAssembleCachedInto(c, reinterpret_cast<double *>(Aval_ptr),
                                            reinterpret_cast<double *>(b_ptr));
}



static inline void _retract_qoe6(double *z) {
    const double TWOPI = 6.283185307179586476925286766559;
    z[1] = std::fmod(z[1], TWOPI); if (z[1] < 0) z[1] += TWOPI;
    z[5] = std::fmod(z[5], TWOPI); if (z[5] < 0) z[5] += TWOPI;
    double e = std::sqrt(z[2]*z[2] + z[3]*z[3]);
    if (e > 0.9 && e > 1e-30) { z[2] *= 0.9/e; z[3] *= 0.9/e; }
    if (z[0] < 6.0e6) z[0] = 6.0e6;
    if (z[0] > 8.5e6) z[0] = 8.5e6;
    if (z[4] < 1e-4) z[4] = 1e-4;
    if (z[4] > 3.1415) z[4] = 3.1415;
}


static py::array_t<double> gts_delta_chain(
        py::array_t<double, py::array::c_style | py::array::forcecast> Zin,
        py::array_t<double, py::array::c_style | py::array::forcecast> D,
        py::array_t<double, py::array::c_style | py::array::forcecast> M,
        py::array_t<double, py::array::c_style | py::array::forcecast> Qg,
        py::array_t<int, py::array::c_style | py::array::forcecast> gsat,
        int K, double sigma_gts, int iters) {
    auto zb = Zin.request(); auto db = D.request(); auto mb = M.request();
    auto qb = Qg.request(); auto gb = gsat.request();
    if (zb.ndim != 3 || zb.shape[2] != 6) throw std::runtime_error("gts_delta_chain: Z 需 (nwin,n,6)");
    const int nwin = (int)zb.shape[0], n = (int)zb.shape[1], G = (int)gb.shape[0];
    const int nfr = 577;
    const double *Dp = static_cast<const double*>(db.ptr);
    const double *Mp = static_cast<const double*>(mb.ptr);
    const double *Qp = static_cast<const double*>(qb.ptr);
    const int *gp = static_cast<const int*>(gb.ptr);
    std::vector<double> Z((std::size_t)nwin*n*6), Zb((std::size_t)nwin*n*6);
    std::memcpy(Z.data(), zb.ptr, Z.size()*sizeof(double));
    Zb = Z;
    std::vector<Eigen::Matrix<double,6,6>> H(n, Eigen::Matrix<double,6,6>::Zero());
    std::vector<Eigen::Matrix<double,6,1>> b(n, Eigen::Matrix<double,6,1>::Zero());
    for (int it = 0; it < iters; ++it) {
        for (int s = 0; s < n; ++s) { H[s].setZero(); b[s].setZero(); }
        for (int w = 0; w < nwin; ++w) {
            std::vector<kep3::Vector6d> oe(n);
            for (int i = 0; i < n; ++i)
                for (int c = 0; c < 6; ++c) oe[i][c] = Z[((std::size_t)w*n+i)*6+c];
            std::vector<double> rv;
            kep3::gvePropagateNoeBatch(oe, nfr, 10.0, 16, 1.0, rv);
            double *d_oe=nullptr,*d_A=nullptr;
            cudaMalloc(&d_oe,(std::size_t)n*6*sizeof(double));
            cudaMalloc(&d_A,(std::size_t)nfr*n*18*sizeof(double));
            cudaMemcpy(d_oe, oe.data(), (std::size_t)n*6*sizeof(double), cudaMemcpyHostToDevice);
            qoejopt::stmFoldGveDevice(d_oe,n,nfr,10.0,1.0,d_A);
            std::vector<double> A((std::size_t)nfr*n*18);
            cudaMemcpy(A.data(),d_A,A.size()*sizeof(double),cudaMemcpyDeviceToHost);
            cudaFree(d_oe); cudaFree(d_A);
            for (int f=0; f<nfr-1; ++f) {
                const int t=w*576+f; if (t>=K) break;
                for (int g=0; g<G; ++g) {
                    if (Mp[(std::size_t)t*G+g] <= 0.5) continue;
                    const int s=gp[g];
                    const double *p=&rv[((std::size_t)f*n+s)*6];
                    const double *q=&Qp[((std::size_t)t*G+g)*3];
                    double dv[3]={p[0]-q[0],p[1]-q[1],p[2]-q[2]};
                    double dist=std::sqrt(dv[0]*dv[0]+dv[1]*dv[1]+dv[2]*dv[2]);
                    if (dist<1e-9) continue;
                    double u[3]={dv[0]/dist,dv[1]/dist,dv[2]/dist};
                    const double *Af=&A[((std::size_t)f*n+s)*18];
                    double J[6];
                    for (int c=0;c<6;++c) J[c]=(u[0]*Af[c]+u[1]*Af[6+c]+u[2]*Af[12+c])/sigma_gts;
                    const double r=(dist-Dp[(std::size_t)t*G+g])/sigma_gts;
                    for (int a=0;a<6;++a) {
                        for (int c=0;c<6;++c) H[s](a,c)+=J[a]*J[c];
                        b[s](a)+=J[a]*r;
                    }
                }
            }
        }
        for (int s=0;s<n;++s) {
            Eigen::Matrix<double,6,6> Hs=H[s];
            Eigen::Matrix<double,6,1> bs=b[s];
            for (int a=0;a<6;++a) Hs(a,a)+=1e-8;
            Eigen::LDLT<Eigen::Matrix<double,6,6>> ldlt(Hs);
            if (ldlt.info()!=Eigen::Success) continue;
            Eigen::Matrix<double,6,1> d=ldlt.solve(-bs);
            if (!d.allFinite()) continue;
            double cap[6]={1e3,1e-4,2e-5,2e-5,1e-5,1e-5};
            for (int a=0;a<6;++a) d(a)=std::max(-cap[a],std::min(cap[a],d(a)));
            for (int w=0;w<nwin;++w) {
                double *zz=&Z[((std::size_t)w*n+s)*6];
                for (int a=0;a<6;++a) zz[a]+=d(a);
                _retract_qoe6(zz);
            }
        }
    }
    py::array_t<double> out({(py::ssize_t)nwin,(py::ssize_t)n,(py::ssize_t)6});
    std::memcpy(out.mutable_data(), Z.data(), Z.size()*sizeof(double));
    return out;
}

static py::array_t<double> gts_z_chain(
        py::array_t<double, py::array::c_style | py::array::forcecast> Zin,
        py::array_t<double, py::array::c_style | py::array::forcecast> D,
        py::array_t<double, py::array::c_style | py::array::forcecast> M,
        py::array_t<double, py::array::c_style | py::array::forcecast> Qg,
        py::array_t<int, py::array::c_style | py::array::forcecast> gsat,
        int K, double sigma_gts, double sigma_dyn, int iters, double sigma_p, int mode) {
    auto zb = Zin.request();
    auto db = D.request(); auto mb = M.request(); auto qb = Qg.request();
    auto gb = gsat.request();
    if (zb.ndim != 3 || zb.shape[2] != 6) throw std::runtime_error("gts_z_chain: Z 需 (nwin,n,6)");
    const int nwin = (int)zb.shape[0], n = (int)zb.shape[1];
    const int G = (int)gb.shape[0];
    const int nfr = 577;
    std::vector<double> Z((std::size_t)nwin * n * 6);
    std::memcpy(Z.data(), zb.ptr, Z.size() * sizeof(double));
    const std::vector<double> Z0 = Z;
    const double *Dp = static_cast<const double *>(db.ptr);
    const double *Mp = static_cast<const double *>(mb.ptr);
    const double *Qp = static_cast<const double *>(qb.ptr);
    const int *gp = static_cast<const int *>(gb.ptr);

    using M96 = Eigen::Matrix<double, 96, 96>;
    using V96 = Eigen::Matrix<double, 96, 1>;
    const double inv_dyn2 = 1.0 / (sigma_dyn * sigma_dyn);
    const double inv_p2 = 1.0 / (sigma_p * sigma_p);

    for (int it = 0; it < iters; ++it) {
        std::vector<M96> H(n, M96::Zero());
        std::vector<V96> b(n, V96::Zero());
        // window GTS rows
        if (mode != 0) for (int w = 0; w < nwin; ++w) {
            std::vector<kep3::Vector6d> oe(n);
            for (int i = 0; i < n; ++i)
                for (int c = 0; c < 6; ++c) oe[i][c] = Z[((std::size_t)w * n + i) * 6 + c];
            std::vector<double> rv;
            kep3::gvePropagateNoeBatch(oe, nfr, 10.0, 16, 1.0, rv);
            double *d_oe = nullptr, *d_A = nullptr;
            cudaMalloc(&d_oe, (std::size_t)n * 6 * sizeof(double));
            cudaMalloc(&d_A, (std::size_t)nfr * n * 18 * sizeof(double));
            cudaMemcpy(d_oe, oe.data(), (std::size_t)n * 6 * sizeof(double), cudaMemcpyHostToDevice);
            qoejopt::stmFoldGveDevice(d_oe, n, nfr, 10.0, 1.0, d_A);
            std::vector<double> A((std::size_t)nfr * n * 18);
            cudaMemcpy(A.data(), d_A, A.size() * sizeof(double), cudaMemcpyDeviceToHost);
            cudaFree(d_oe); cudaFree(d_A);
            for (int f = 0; f < nfr - 1; ++f) {
                const int t = w * 576 + f;
                if (t >= K) break;
                for (int g = 0; g < G; ++g) {
                    if (Mp[(std::size_t)t * G + g] <= 0.5) continue;
                    const int s = gp[g];
                    const double *p = &rv[((std::size_t)f * n + s) * 6];
                    const double *q = &Qp[((std::size_t)t * G + g) * 3];
                    double dv[3] = {p[0]-q[0], p[1]-q[1], p[2]-q[2]};
                    double dist = std::sqrt(dv[0]*dv[0]+dv[1]*dv[1]+dv[2]*dv[2]);
                    if (dist < 1e-9) continue;
                    double u[3] = {dv[0]/dist, dv[1]/dist, dv[2]/dist};
                    const double *Af = &A[((std::size_t)f * n + s) * 18];
                    double J[6];
                    for (int c = 0; c < 6; ++c)
                        J[c] = (u[0]*Af[c] + u[1]*Af[6+c] + u[2]*Af[12+c]) / sigma_gts;
                    const double r = (dist - Dp[(std::size_t)t * G + g]) / sigma_gts;
                    const int o = w * 6;
                    for (int a = 0; a < 6; ++a) {
                        for (int c = 0; c < 6; ++c) H[s](o+a, o+c) += J[a] * J[c];
                        b[s](o+a) += J[a] * r;
                    }
                }
            }
        }
        // window dynamics
        if (mode != 1) for (int w = 0; w + 1 < nwin; ++w) {
            std::vector<kep3::Vector6d> oe(n);
            for (int i = 0; i < n; ++i)
                for (int c = 0; c < 6; ++c) oe[i][c] = Z[((std::size_t)w * n + i) * 6 + c];
            double *d_xlo = nullptr, *d_xhi = nullptr, *d_res = nullptr, *d_F = nullptr;
            cudaMalloc(&d_xlo, (std::size_t)n * 6 * sizeof(double));
            cudaMalloc(&d_xhi, (std::size_t)n * 6 * sizeof(double));
            cudaMalloc(&d_res, (std::size_t)n * 6 * sizeof(double));
            cudaMalloc(&d_F, (std::size_t)n * 36 * sizeof(double));
            cudaMemcpy(d_xlo, &Z[((std::size_t)w * n) * 6], (std::size_t)n * 6 * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(d_xhi, &Z[((std::size_t)(w+1) * n) * 6], (std::size_t)n * 6 * sizeof(double), cudaMemcpyHostToDevice);
            qoejopt::gveDynFactorDev(d_xlo, d_xhi, 1, n, 5760.0, 1.0, d_res, d_F);
            std::vector<double> F((std::size_t)n * 36), rr((std::size_t)n * 6);
            cudaMemcpy(F.data(), d_F, F.size() * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(rr.data(), d_res, rr.size() * sizeof(double), cudaMemcpyDeviceToHost);
            cudaFree(d_xlo); cudaFree(d_xhi); cudaFree(d_res); cudaFree(d_F);
            for (int s = 0; s < n; ++s) {
                const int a0 = w * 6, b0 = a0 + 6;
                for (int a = 0; a < 6; ++a)
                    for (int c = 0; c < 6; ++c) {
                        double fa = 0.0;
                        for (int k = 0; k < 6; ++k) fa += F[s*36 + a*6 + k] * F[s*36 + c*6 + k];
                        H[s](a0+a, a0+c) += fa * inv_dyn2;
                        H[s](b0+a, b0+c) += (a == c ? inv_dyn2 : 0.0);
                        H[s](a0+a, b0+c) -= F[s*36 + a*6 + c] * inv_dyn2;
                        H[s](b0+a, a0+c) -= F[s*36 + c*6 + a] * inv_dyn2;
                    }
                for (int a = 0; a < 6; ++a) {
                    double rv = rr[(std::size_t)s * 6 + a];
                    double fr = 0.0;
                    for (int k = 0; k < 6; ++k) fr += F[s*36 + k*6 + a] * rv;
                    b[s](a0+a) -= fr * inv_dyn2;
                    b[s](b0+a) += rv * inv_dyn2;
                }
            }
        }
        // prior first window; mode=1 only: weak prior every window to fix null modes
        for (int s = 0; s < n; ++s) {
            for (int a = 0; a < 6; ++a) {
                H[s](a, a) += inv_p2;
                b[s](a) += inv_p2 * (Z[(std::size_t)s * 6 + a] - Z0[(std::size_t)s * 6 + a]);
            }
            if (mode == 1) {
                for (int w = 1; w < nwin; ++w) {
                    for (int a = 0; a < 6; ++a) {
                        H[s](w*6+a, w*6+a) += 1e-8;
                        b[s](w*6+a) += 1e-8 * (Z[((std::size_t)w*n+s)*6+a] - Z0[((std::size_t)w*n+s)*6+a]);
                    }
                }
            }
        }
        // solve per satellite
        for (int s = 0; s < n; ++s) {
            H[s] = 0.5 * (H[s] + H[s].transpose());
            V96 dsc;
            for (int a = 0; a < nwin * 6; ++a)
                dsc(a) = std::sqrt(std::max(H[s](a, a), 1e-12));
            M96 Hs = H[s];
            V96 bs = b[s];
            for (int a = 0; a < nwin * 6; ++a) {
                bs(a) *= 1.0 / dsc(a);
                for (int c = 0; c < nwin * 6; ++c)
                    Hs(a, c) *= 1.0 / (dsc(a) * dsc(c));
            }
            Hs += M96::Identity() * 1e-6;
            Eigen::LDLT<M96> ldlt(Hs);
            if (ldlt.info() != Eigen::Success) continue;
            V96 y = ldlt.solve(-bs);
            if (!y.allFinite()) continue;
            V96 dx;
            const double cap6[6] = {1e3, 1e-4, 2e-5, 2e-5, 1e-5, 1e-5};
            for (int a = 0; a < nwin * 6; ++a) {
                const double cc = cap6[a % 6];
                dx(a) = std::max(-cc, std::min(cc, y(a) / dsc(a)));
            }
            double alpha = 1.0;
            for (int ls = 0; ls < 8; ++ls) {
                V96 d = alpha * dx;
                double lin = 0.5 * d.dot(H[s] * d) + b[s].dot(d);
                if (lin < 0.0) break;
                alpha *= 0.5;
            }
            for (int w = 0; w < nwin; ++w) {
                double *zz = &Z[((std::size_t)w * n + s) * 6];
                for (int a = 0; a < 6; ++a) zz[a] += alpha * dx(w*6+a);
                _retract_qoe6(zz);
            }
        }
    }
    py::array_t<double> out({(py::ssize_t)nwin, (py::ssize_t)n, (py::ssize_t)6});
    std::memcpy(out.mutable_data(), Z.data(), Z.size() * sizeof(double));
    return out;
}

static py::tuple gve_stm(py::array_t<double, py::array::c_style | py::array::forcecast> x,
                         double dt, double beta) {
    auto buf = x.request();
    if (buf.ndim != 2 || buf.shape[1] != 6)
        throw std::runtime_error("qoejopt.gve_stm: x 需 (n,6)");
    const std::size_t n = (std::size_t)buf.shape[0];
    double *dx = nullptr, *doe = nullptr, *dPhi = nullptr;
    cudaMalloc(&dx, n * 6 * sizeof(double));
    cudaMalloc(&doe, n * 6 * sizeof(double));
    cudaMalloc(&dPhi, n * 36 * sizeof(double));
    cudaMemcpy(dx, buf.ptr, n * 6 * sizeof(double), cudaMemcpyHostToDevice);
    {
        py::gil_scoped_release rel;
        qoejopt::gveStmDevice(dx, (int)n, dt, beta, doe, dPhi);
    }
    py::array_t<double> OE({(py::ssize_t)n, (py::ssize_t)6});
    py::array_t<double> PH({(py::ssize_t)n, (py::ssize_t)6, (py::ssize_t)6});
    cudaMemcpy(OE.mutable_data(), doe, n * 6 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(PH.mutable_data(), dPhi, n * 36 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(dx); cudaFree(doe); cudaFree(dPhi);
    return py::make_tuple(OE, PH);
}

static py::tuple gve_stm_ia(py::array_t<double, py::array::c_style | py::array::forcecast> x,
                            double dt, double beta) {
    auto buf = x.request();
    if (buf.ndim != 2 || buf.shape[1] != 6)
        throw std::runtime_error("qoejopt.gve_stm_ia: x 需 (n,6)");
    const std::size_t n = (std::size_t)buf.shape[0];
    double *dx = nullptr, *doe = nullptr, *dPhi = nullptr;
    cudaMalloc(&dx, n * 6 * sizeof(double));
    cudaMalloc(&doe, n * 6 * sizeof(double));
    cudaMalloc(&dPhi, n * 36 * sizeof(double));
    cudaMemcpy(dx, buf.ptr, n * 6 * sizeof(double), cudaMemcpyHostToDevice);
    {
        py::gil_scoped_release rel;
        qoejopt::gveStmIADev(dx, (int)n, dt, beta, doe, dPhi);
    }
    py::array_t<double> OE({(py::ssize_t)n, (py::ssize_t)6});
    py::array_t<double> PH({(py::ssize_t)n, (py::ssize_t)6, (py::ssize_t)6});
    cudaMemcpy(OE.mutable_data(), doe, n * 6 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(PH.mutable_data(), dPhi, n * 36 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(dx); cudaFree(doe); cudaFree(dPhi);
    return py::make_tuple(OE, PH);
}

static py::array_t<double> gve_ctrl(py::array_t<double, py::array::c_style | py::array::forcecast> x) {
    auto buf = x.request();
    if (buf.ndim != 2 || buf.shape[1] != 6)
        throw std::runtime_error("qoejopt.gve_ctrl: x 需 (n,6)");
    const std::size_t n = (std::size_t)buf.shape[0];
    double *dx = nullptr, *dB = nullptr;
    cudaMalloc(&dx, n * 6 * sizeof(double));
    cudaMalloc(&dB, n * 18 * sizeof(double));
    cudaMemcpy(dx, buf.ptr, n * 6 * sizeof(double), cudaMemcpyHostToDevice);
    {
        py::gil_scoped_release rel;
        qoejopt::gveCtrlDev(dx, (int)n, dB);
    }
    py::array_t<double> B({(py::ssize_t)n, (py::ssize_t)6, (py::ssize_t)3});
    cudaMemcpy(B.mutable_data(), dB, n * 18 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(dx); cudaFree(dB);
    return B;
}

static py::tuple gve_dyn_factor(py::array_t<double, py::array::c_style | py::array::forcecast> xlo,
                                py::array_t<double, py::array::c_style | py::array::forcecast> xhi,
                                double dt, double beta) {
    auto bl = xlo.request(); auto bh = xhi.request();
    if (bl.ndim != 3 || bl.shape[2] != 6)
        throw std::runtime_error("qoejopt.gve_dyn_factor: x_lo 需 (nseg,n,6)");
    const int nseg = (int)bl.shape[0], n = (int)bl.shape[1];
    const std::size_t tot = (std::size_t)nseg * n;
    double *dl = nullptr, *dh = nullptr, *dres = nullptr, *dphi = nullptr;
    cudaMalloc(&dl, tot * 6 * sizeof(double)); cudaMalloc(&dh, tot * 6 * sizeof(double));
    cudaMalloc(&dres, tot * 6 * sizeof(double)); cudaMalloc(&dphi, tot * 36 * sizeof(double));
    cudaMemcpy(dl, bl.ptr, tot * 6 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dh, bh.ptr, tot * 6 * sizeof(double), cudaMemcpyHostToDevice);
    {
        py::gil_scoped_release rel;
        qoejopt::gveDynFactorDev(dl, dh, nseg, n, dt, beta, dres, dphi);
    }
    py::array_t<double> RES({(py::ssize_t)nseg, (py::ssize_t)n, (py::ssize_t)6});
    py::array_t<double> PHI({(py::ssize_t)nseg, (py::ssize_t)n, (py::ssize_t)6, (py::ssize_t)6});
    cudaMemcpy(RES.mutable_data(), dres, tot * 6 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(PHI.mutable_data(), dphi, tot * 36 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(dl); cudaFree(dh); cudaFree(dres); cudaFree(dphi);
    return py::make_tuple(RES, PHI);
}

static void win_prior_asm(py::array_t<double, py::array::c_style | py::array::forcecast> X,
                          py::array_t<double, py::array::c_style | py::array::forcecast> pri_fac,
                          py::array_t<double, py::array::c_style | py::array::forcecast> zp,
                          int pnode, int off_pri, int base_b, uint64_t Aval_ptr, uint64_t b_ptr) {
    auto bx = X.request(); auto bf = pri_fac.request(); auto bz = zp.request();
    if (bx.ndim != 3 || bx.shape[2] != 6) throw std::runtime_error("win_prior_asm: X 需 (L,n,6)");
    const int L = (int)bx.shape[0], n = (int)bx.shape[1];
    if (pnode < 0 || pnode >= L) throw std::runtime_error("win_prior_asm: pnode 越界");
    if ((std::size_t)bf.size != (std::size_t)n * 36) throw std::runtime_error("win_prior_asm: pri_fac 需 (n,36)");
    if ((std::size_t)bz.size != (std::size_t)n * 6) throw std::runtime_error("win_prior_asm: zp 需 (n,6)");
    double *dX = nullptr, *dF = nullptr, *dZ = nullptr;
    cudaMalloc(&dX, (std::size_t)L * n * 6 * sizeof(double));
    cudaMalloc(&dF, (std::size_t)n * 36 * sizeof(double));
    cudaMalloc(&dZ, (std::size_t)n * 6 * sizeof(double));
    cudaMemcpy(dX, bx.ptr, (std::size_t)L * n * 6 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dF, bf.ptr, (std::size_t)n * 36 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dZ, bz.ptr, (std::size_t)n * 6 * sizeof(double), cudaMemcpyHostToDevice);
    {
        py::gil_scoped_release rel;
        qoejopt::winPriorAsmDevice(dX, dF, dZ, n, pnode, off_pri, base_b,
                                   reinterpret_cast<double *>(Aval_ptr), reinterpret_cast<double *>(b_ptr));
    }
    cudaFree(dX); cudaFree(dF); cudaFree(dZ);
}

static py::tuple state_sens(py::array_t<double, py::array::c_style | py::array::forcecast> x, double beta) {
    Ctx &c = g_ctx;
    if (!c.d_X) throw std::runtime_error("qoejopt.state_sens: 未先 setup");
    auto buf = x.request();
    if ((std::size_t)buf.size != (std::size_t)c.n * 6)
        throw std::runtime_error("qoejopt.state_sens: x 需 (n,6)");
    const std::size_t sz = (std::size_t)c.nfr * c.n * 6;
    std::vector<double> oe(sz), S(sz);
    {
        py::gil_scoped_release rel;
        qoejopt::jointStateSensBatch(c, static_cast<const double *>(buf.ptr), beta, oe.data(), S.data());
    }
    py::array_t<double> OE({(py::ssize_t)c.nfr, (py::ssize_t)c.n, (py::ssize_t)6});
    py::array_t<double> SS({(py::ssize_t)c.nfr, (py::ssize_t)c.n, (py::ssize_t)6});
    std::memcpy(OE.mutable_data(), oe.data(), sz * sizeof(double));
    std::memcpy(SS.mutable_data(), S.data(), sz * sizeof(double));
    return py::make_tuple(OE, SS);
}

static py::array_t<double> expand_rv(py::array_t<double, py::array::c_style | py::array::forcecast> x) {
    Ctx &c = g_ctx;
    if (!c.d_X) throw std::runtime_error("qoejopt.expand_rv: 未先 setup");
    _copy_to_dev(c.d_X, x, (std::size_t)c.n * 6);
    std::vector<double> h((std::size_t)c.nfr * c.n * 6);
    {
        py::gil_scoped_release rel;
        qoejopt::jointExpandDevice(c);
        qoejopt::jointCopyOut(c, h.data());
    }
    py::array_t<double> out({(py::ssize_t)c.nfr, (py::ssize_t)c.n, (py::ssize_t)6});
    std::memcpy(out.mutable_data(), h.data(), h.size() * sizeof(double));
    return out;
}

// P4.2/CUDA：整弧 RBF θ 前向敏度（centers/s 需先 qoe.setRbfCuda 预设）。
static py::tuple gve_rbf_sens(py::array_t<double, py::array::c_style | py::array::forcecast> oe,
                              int nfr, double dt, double beta, int m) {
    auto buf = oe.request();
    if (buf.ndim != 2 || buf.shape[1] != 6) throw std::runtime_error("gve_rbf_sens: oe 需 (n,6)");
    const std::size_t n = (std::size_t)buf.shape[0];
    std::vector<kep3::Vector6d> v(n);
    const double *p = static_cast<const double *>(buf.ptr);
    for (std::size_t i = 0; i < n; ++i) for (int c = 0; c < 6; ++c) v[i](c) = p[i*6 + c];
    std::vector<double> oe_all, S_all;
    { py::gil_scoped_release rel; qoejopt::gveRbfSensBatch(v, nfr, dt, beta, m, oe_all, S_all); }
    py::array_t<double> OE({(py::ssize_t)nfr, (py::ssize_t)n, (py::ssize_t)6});
    py::array_t<double> SS({(py::ssize_t)nfr, (py::ssize_t)n, (py::ssize_t)6, (py::ssize_t)m});
    std::memcpy(OE.mutable_data(), oe_all.data(), oe_all.size()*sizeof(double));
    std::memcpy(SS.mutable_data(), S_all.data(), S_all.size()*sizeof(double));
    return py::make_tuple(OE, SS);
}

// ===================== 单发窗口（win_assemble_ss）：单发 L 帧 + 每星 6×6 先验 + 协方差白化 =====================
static void setup_win_ss(int n, int L, int E, int G, int ncol, int nnz, int nrows,
                         int off_isl, int off_gts, int off_damp,
                         double dt, double s_isl, double s_gts, double sigma_a,
                         py::array_t<int, py::array::c_style | py::array::forcecast> isl_i,
                         py::array_t<int, py::array::c_style | py::array::forcecast> isl_jj,
                         py::array_t<double, py::array::c_style | py::array::forcecast> isl_D,
                         py::array_t<double, py::array::c_style | py::array::forcecast> isl_W,
                         py::array_t<int, py::array::c_style | py::array::forcecast> gts_a,
                         py::array_t<double, py::array::c_style | py::array::forcecast> gts_D,
                         py::array_t<double, py::array::c_style | py::array::forcecast> gts_W,
                         py::array_t<double, py::array::c_style | py::array::forcecast> gts_Q) {
    py::array_t<double> ctr0((py::ssize_t)((std::size_t)n * 6));
    std::memset(ctr0.mutable_data(), 0, (std::size_t)n * 6 * sizeof(double));
    setup(n, L, E, G, ncol, nnz, nrows, off_isl, off_gts, off_damp, dt, s_isl, s_gts, 1.0,
          isl_i, isl_jj, isl_D, isl_W, gts_a, gts_D, gts_W, gts_Q, ctr0);
    Ctx &c = g_ctx;
    const char *wc = std::getenv("WINCOV");          // 默认常数权重（标准 MAP）；WINCOV=1 启用 P0 白化
    c.prior_mode = 1; c.cov_mode = (wc && std::atoi(wc) == 1) ? 1 : 0; c.sigma_a = sigma_a;
    cudaMalloc(&c.d_P0, (std::size_t)n * 36 * sizeof(double));
    cudaMalloc(&c.d_prior_fac, (std::size_t)n * 36 * sizeof(double));
    cudaMalloc(&c.d_zprior, (std::size_t)n * 6 * sizeof(double));
    cudaMalloc(&c.d_P0rv, (std::size_t)n * 36 * sizeof(double));
    // d_cov/d_phi/d_Pm/d_Pp/d_PpPrev 与 d_phi_all 在长窗口（L≥8T）各占 L*n*36*8，
    // 全部预分配会让 16T 窗口超出显存；只在用到协方差/iEKF 时按需分配。
    cudaFree(c.d_phi_all); c.d_phi_all = nullptr;
}

// 逐帧协方差滤波（iEKF）：设置 rv 空间边界 P0；mode=1 串行（迭代 1）、mode=2 Jacobi（迭代 ≥2）。
static void set_cov0_ss(py::array_t<double, py::array::c_style | py::array::forcecast> P0rv) {
    Ctx &c = g_ctx;
    if (!c.d_P0rv) throw std::runtime_error("qoejopt.set_cov0_ss: 未先 setup_win_ss");
    qoejopt::setCov0Device(c, static_cast<const double *>(P0rv.request().ptr));
}

static void set_adj_ss(py::array_t<int, py::array::c_style | py::array::forcecast> adj_j,
                       py::array_t<double, py::array::c_style | py::array::forcecast> adj_w) {
    Ctx &c = g_ctx;
    if (!c.d_Pm) throw std::runtime_error("qoejopt.set_adj_ss: 未先 setup_win_ss");
    auto bj = adj_j.request();
    if (bj.ndim != 2) throw std::runtime_error("qoejopt.set_adj_ss: adj_j 需 (nfr*n, D)");
    qoejopt::setAdjDevice(c, static_cast<const int *>(bj.ptr),
                          static_cast<const double *>(adj_w.request().ptr),
                          (int)bj.shape[1], (int)bj.shape[0]);
}

static void win_cov_rv_ss(int mode) {
    Ctx &c = g_ctx;
    if (!c.d_P0) throw std::runtime_error("qoejopt.win_cov_rv_ss: 未先 setup_win_ss");
    const std::size_t bP = (std::size_t)c.nfr * c.n * 36 * sizeof(double);
    if (!c.d_Pm) cudaMalloc(&c.d_Pm, bP);
    if (!c.d_Pp) cudaMalloc(&c.d_Pp, bP);
    if (!c.d_PpPrev) cudaMalloc(&c.d_PpPrev, bP);
    qoejopt::ssFrameCovDevice(c, mode);
    c.cov_mode = 0;                 // 口径 A：ISL/GTS 用常数权重 1/σ（P_k 不进白化）
}

static py::array_t<double> win_Pp_ss() {
    Ctx &c = g_ctx;
    if (!c.d_Pp) throw std::runtime_error("qoejopt.win_Pp_ss: 未先 setup_win_ss / 未先 win_cov_rv_ss");
    const std::size_t sz = (std::size_t)c.nfr * c.n * 36;
    std::vector<double> h(sz);
    cudaMemcpy(h.data(), c.d_Pp, sz * sizeof(double), cudaMemcpyDeviceToHost);
    py::array_t<double> P({(py::ssize_t)c.nfr, (py::ssize_t)c.n, (py::ssize_t)6, (py::ssize_t)6});
    std::memcpy(P.mutable_data(), h.data(), sz * sizeof(double));
    return P;
}

static double win_asm_cov_ss(uint64_t Aval_ptr, uint64_t b_ptr) {
    Ctx &c = g_ctx;
    if (!c.d_X) throw std::runtime_error("qoejopt.win_asm_cov_ss: 未先 setup_win_ss");
    py::gil_scoped_release rel;
    return qoejopt::jointAssembleCachedInto(c, reinterpret_cast<double *>(Aval_ptr),
                                            reinterpret_cast<double *>(b_ptr));
}

static void set_obs_ss(py::array_t<double, py::array::c_style | py::array::forcecast> isl_D,
                       py::array_t<double, py::array::c_style | py::array::forcecast> isl_W,
                       py::array_t<double, py::array::c_style | py::array::forcecast> gts_D,
                       py::array_t<double, py::array::c_style | py::array::forcecast> gts_W,
                       py::array_t<double, py::array::c_style | py::array::forcecast> gts_Q) {
    Ctx &c = g_ctx;
    if (!c.d_X) throw std::runtime_error("qoejopt.set_obs_ss: 未先 setup_win_ss");
    qoejopt::setObsDevice(c, static_cast<const double *>(isl_D.request().ptr),
                          static_cast<const double *>(isl_W.request().ptr),
                          static_cast<const double *>(gts_D.request().ptr),
                          static_cast<const double *>(gts_W.request().ptr),
                          static_cast<const double *>(gts_Q.request().ptr));
}

static void set_prior_ss(py::array_t<double, py::array::c_style | py::array::forcecast> fac,
                         py::array_t<double, py::array::c_style | py::array::forcecast> zprior,
                         py::array_t<double, py::array::c_style | py::array::forcecast> P0) {
    Ctx &c = g_ctx;
    if (!c.d_X) throw std::runtime_error("qoejopt.set_prior_ss: 未先 setup_win_ss");
    qoejopt::setPriorDevice(c, static_cast<const double *>(fac.request().ptr),
                            static_cast<const double *>(zprior.request().ptr),
                            static_cast<const double *>(P0.request().ptr));
}

static void set_beta_ss(py::array_t<double, py::array::c_style | py::array::forcecast> beta) {
    Ctx &c = g_ctx;
    if (!c.d_X) throw std::runtime_error("qoejopt.set_beta_ss: 未先 setup_win_ss");
    auto b = beta.request();
    if ((std::size_t)b.size != (std::size_t)c.n)
        throw std::runtime_error("qoejopt.set_beta_ss: beta 需 (n,)");
    if (c.d_beta) cudaFree(c.d_beta);
    c.d_beta = _upload(beta);
}

static double win_assemble_ss(py::array_t<double, py::array::c_style | py::array::forcecast> X,
                              py::array_t<double, py::array::c_style | py::array::forcecast> rv0,
                              py::array_t<double, py::array::c_style | py::array::forcecast> A0,
                              uint64_t Aval_ptr, uint64_t b_ptr) {
    Ctx &c = g_ctx;
    if (!c.d_X) throw std::runtime_error("qoejopt.win_assemble_ss: 未先 setup_win_ss");
    _copy_to_dev(c.d_X, X, (std::size_t)c.n * 6);
    _copy_to_dev(c.d_rv0, rv0, (std::size_t)c.n * 6);
    _copy_to_dev(c.d_A0, A0, (std::size_t)c.n * 36);
    py::gil_scoped_release rel;
    return qoejopt::jointAssembleInto(c, reinterpret_cast<double *>(Aval_ptr),
                                      reinterpret_cast<double *>(b_ptr));
}

static double win_expand_ss(py::array_t<double, py::array::c_style | py::array::forcecast> X,
                            py::array_t<double, py::array::c_style | py::array::forcecast> rv0,
                            py::array_t<double, py::array::c_style | py::array::forcecast> A0) {
    Ctx &c = g_ctx;
    if (!c.d_X) throw std::runtime_error("qoejopt.win_expand_ss: 未先 setup_win_ss");
    _copy_to_dev(c.d_X, X, (std::size_t)c.n * 6);
    _copy_to_dev(c.d_rv0, rv0, (std::size_t)c.n * 6);
    _copy_to_dev(c.d_A0, A0, (std::size_t)c.n * 36);
    py::gil_scoped_release rel;
    return qoejopt::jointIslRes2(c);
}

static py::array_t<double> win_rv_ss() {
    Ctx &c = g_ctx;
    if (!c.d_X) throw std::runtime_error("qoejopt.win_rv_ss: 未先 setup_win_ss");
    std::vector<double> h((std::size_t)c.nfr * c.n * 6);
    {
        py::gil_scoped_release rel;
        qoejopt::jointCopyOut(c, h.data());
    }
    py::array_t<double> R({(py::ssize_t)c.nfr, (py::ssize_t)c.n, (py::ssize_t)6});
    std::memcpy(R.mutable_data(), h.data(), h.size() * sizeof(double));
    return R;
}

static double win_cached_ss(uint64_t Aval_ptr, uint64_t b_ptr) {
    Ctx &c = g_ctx;
    if (!c.d_X) throw std::runtime_error("qoejopt.win_cached_ss: 未先 setup_win_ss");
    py::gil_scoped_release rel;
    return qoejopt::jointAssembleCachedInto(c, reinterpret_cast<double *>(Aval_ptr),
                                            reinterpret_cast<double *>(b_ptr));
}

static py::tuple win_cov_ss(py::array_t<double, py::array::c_style | py::array::forcecast> X) {
    Ctx &c = g_ctx;
    if (!c.d_X) throw std::runtime_error("qoejopt.win_cov_ss: 未先 setup_win_ss");
    auto bx = X.request();
    if ((std::size_t)bx.size != (std::size_t)c.n * 6)
        throw std::runtime_error("qoejopt.win_cov_ss: X 需 (n,6)");
    const std::size_t sz = (std::size_t)c.nfr * c.n * 36;
    if (!c.d_cov) cudaMalloc(&c.d_cov, sz * sizeof(double));
    std::vector<double> cov(sz);
    std::vector<double> rv((std::size_t)c.nfr * c.n * 6);
    {
        py::gil_scoped_release rel;
        qoejopt::ssCovDevice(c, static_cast<const double *>(bx.ptr), cov.data(), rv.data());
    }
    py::array_t<double> C({(py::ssize_t)c.nfr, (py::ssize_t)c.n, (py::ssize_t)6, (py::ssize_t)6});
    py::array_t<double> R({(py::ssize_t)c.nfr, (py::ssize_t)c.n, (py::ssize_t)6});
    std::memcpy(C.mutable_data(), cov.data(), sz * sizeof(double));
    std::memcpy(R.mutable_data(), rv.data(), rv.size() * sizeof(double));
    return py::make_tuple(C, R);
}

// ===================== L 节点批量窗口（每帧一节点，ncol=6NL）device 常驻装配 =====================
static qoejopt::WinCtx g_wctx;

static void _free_wctx() {
    qoejopt::WinCtx &c = g_wctx;
    cudaFree(c.d_isl_i); cudaFree(c.d_isl_jj); cudaFree(c.d_gts_a);
    cudaFree(c.d_isl_D); cudaFree(c.d_isl_W);
    cudaFree(c.d_gts_D); cudaFree(c.d_gts_W); cudaFree(c.d_gts_Q);
    cudaFree(c.d_ctr); cudaFree(c.d_qingt); cudaFree(c.d_beta);
    cudaFree(c.d_X); cudaFree(c.d_rv); cudaFree(c.d_Ap);
    cudaFree(c.d_end); cudaFree(c.d_phi); cudaFree(c.d_res);
    g_wctx = qoejopt::WinCtx{};
}

static void setup_win(int n, int L, int E, int G, int ncol, int nnz, int nrows,
                      int off_isl, int off_gts, int off_dyn, int off_damp,
                      double dt, double s_isl, double s_gts, double issq, int prior, int q_perstar,
                      py::array_t<int, py::array::c_style | py::array::forcecast> isl_i,
                      py::array_t<int, py::array::c_style | py::array::forcecast> isl_jj,
                      py::array_t<double, py::array::c_style | py::array::forcecast> isl_D,
                      py::array_t<double, py::array::c_style | py::array::forcecast> isl_W,
                      py::array_t<int, py::array::c_style | py::array::forcecast> gts_a,
                      py::array_t<double, py::array::c_style | py::array::forcecast> gts_D,
                      py::array_t<double, py::array::c_style | py::array::forcecast> gts_W,
                      py::array_t<double, py::array::c_style | py::array::forcecast> gts_Q,
                      py::array_t<double, py::array::c_style | py::array::forcecast> ctr,
                      py::array_t<double, py::array::c_style | py::array::forcecast> qingt) {
    _free_wctx();
    qoejopt::WinCtx &c = g_wctx;
    c.n = n; c.L = L; c.E = E; c.G = G; c.ncol = ncol; c.nnz = nnz; c.nrows = nrows;
    c.off_isl = off_isl; c.off_gts = off_gts; c.off_dyn = off_dyn; c.off_damp = off_damp;
    c.dt = dt; c.s_isl = s_isl; c.s_gts = s_gts; c.issq = issq; c.prior = prior ? 1 : 0;
    c.q_perstar = q_perstar ? 1 : 0;
    c.d_isl_i = _upload(isl_i); c.d_isl_jj = _upload(isl_jj);
    c.d_isl_D = _upload(isl_D); c.d_isl_W = _upload(isl_W);
    c.d_gts_a = _upload(gts_a); c.d_gts_D = _upload(gts_D);
    c.d_gts_W = _upload(gts_W); c.d_gts_Q = _upload(gts_Q);
    c.d_ctr = _upload(ctr); c.d_qingt = _upload(qingt);
    const std::size_t bRv = (std::size_t)L * n * 6 * sizeof(double);
    const std::size_t bAp = (std::size_t)L * n * 18 * sizeof(double);
    const std::size_t bEnd = (std::size_t)(L > 0 ? (L - 1) : 0) * n * 6 * sizeof(double);
    cudaMalloc(&c.d_X, bRv); cudaMalloc(&c.d_rv, bRv); cudaMalloc(&c.d_Ap, bAp);
    cudaMalloc(&c.d_end, bEnd); cudaMalloc(&c.d_phi, bEnd * 6);
    cudaMalloc(&c.d_res, sizeof(double));
}

static void set_win_beta(py::array_t<double, py::array::c_style | py::array::forcecast> beta) {
    qoejopt::WinCtx &c = g_wctx;
    if (!c.d_X) throw std::runtime_error("qoejopt.set_win_beta: 未先 setup_win");
    auto b = beta.request();
    if ((std::size_t)b.size != (std::size_t)c.n)
        throw std::runtime_error("qoejopt.set_win_beta: beta 需 (n,)");
    if (c.d_beta) cudaFree(c.d_beta);
    const int nseg = std::max(1, c.L - 1);
    std::vector<double> hb((std::size_t)nseg * c.n);
    const double *src = static_cast<const double *>(b.ptr);
    for (int s = 0; s < nseg; ++s)
        std::copy(src, src + c.n, hb.begin() + (std::size_t)s * c.n);
    cudaMalloc(&c.d_beta, (std::size_t)nseg * c.n * sizeof(double));
    cudaMemcpy(c.d_beta, hb.data(), hb.size() * sizeof(double), cudaMemcpyHostToDevice);
}

static void win_consensus_asm(py::array_t<double, py::array::c_style | py::array::forcecast> zcons,
                              py::array_t<double, py::array::c_style | py::array::forcecast> u,
                              py::object Lp, py::object zp,
                              double rho, int prior_mode,
                              uint64_t Aval_ptr, uint64_t b_ptr) {
    qoejopt::WinCtx &c = g_wctx;
    if (!c.d_X) throw std::runtime_error("qoejopt.win_consensus_asm: 未先 setup_win");
    auto bz = zcons.request(); auto bu = u.request();
    if ((std::size_t)bz.size != (std::size_t)c.L * c.n * 6 ||
        (std::size_t)bu.size != (std::size_t)c.L * c.n * 6)
        throw std::runtime_error("win_consensus_asm: zcons/u 需 (L,n,6)");
    double *dz = nullptr, *du = nullptr, *dLp = nullptr, *dzp = nullptr;
    cudaMalloc(&dz, bz.size * sizeof(double)); cudaMalloc(&du, bu.size * sizeof(double));
    cudaMemcpy(dz, bz.ptr, bz.size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(du, bu.ptr, bu.size * sizeof(double), cudaMemcpyHostToDevice);
    if (prior_mode && !Lp.is_none()) {
        auto aL = Lp.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        auto aZ = zp.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        cudaMalloc(&dLp, (std::size_t)c.n * 36 * sizeof(double));
        cudaMalloc(&dzp, (std::size_t)c.n * 6 * sizeof(double));
        cudaMemcpy(dLp, aL.request().ptr, (std::size_t)c.n * 36 * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dzp, aZ.request().ptr, (std::size_t)c.n * 6 * sizeof(double), cudaMemcpyHostToDevice);
    }
    {
        py::gil_scoped_release rel;
        qoejopt::winConsensusAsmDevice(c, dz, du, dLp, dzp, prior_mode, rho,
                                       reinterpret_cast<double *>(Aval_ptr),
                                       reinterpret_cast<double *>(b_ptr));
    }
    cudaFree(dz); cudaFree(du); if (dLp) cudaFree(dLp); if (dzp) cudaFree(dzp);
}



static void _solve6(double A[6][6], double b[6], double x[6]) {
    for (int i = 0; i < 6; ++i) {
        int p = i;
        for (int r = i + 1; r < 6; ++r)
            if (std::abs(A[r][i]) > std::abs(A[p][i])) p = r;
        if (std::abs(A[p][i]) < 1e-14) { for (int j = 0; j < 6; ++j) x[j] = 0.0; return; }
        if (p != i) { for (int j = 0; j < 6; ++j) std::swap(A[i][j], A[p][j]); std::swap(b[i], b[p]); }
        double d = A[i][i];
        for (int j = i; j < 6; ++j) A[i][j] /= d;
        b[i] /= d;
        for (int r = 0; r < 6; ++r) if (r != i) {
            double f = A[r][i];
            if (f == 0.0) continue;
            for (int j = i; j < 6; ++j) A[r][j] -= f * A[i][j];
            b[r] -= f * b[i];
        }
    }
    for (int j = 0; j < 6; ++j) x[j] = b[j];
}

static py::array_t<double> gts_global_update(
        py::array_t<double, py::array::c_style | py::array::forcecast> Z,
        py::array_t<double, py::array::c_style | py::array::forcecast> D,
        py::array_t<double, py::array::c_style | py::array::forcecast> M,
        py::array_t<double, py::array::c_style | py::array::forcecast> Q,
        py::array_t<int, py::array::c_style | py::array::forcecast> gs) {
    auto bz = Z.request(); auto bd = D.request(); auto bm = M.request();
    auto bq = Q.request(); auto bg = gs.request();
    const int K = (int)bz.shape[0], n = (int)bz.shape[1];
    const int G = (int)bg.size;
    if (bz.ndim != 3 || (int)bz.shape[2] != 6 || (int)bd.shape[0] != K ||
        (int)bd.shape[1] != G || (int)bm.shape[0] != K || (int)bm.shape[1] != G ||
        (int)bq.shape[0] != K || (int)bq.shape[1] != G)
        throw std::runtime_error("gts_global_update: shape mismatch");
    py::array_t<double> out({K, n, 6});
    std::memcpy(out.request().ptr, bz.ptr, (size_t)K * n * 6 * sizeof(double));
    double *z = static_cast<double *>(out.request().ptr);
    const double *Dp = static_cast<const double *>(bd.ptr);
    const double *Mp = static_cast<const double *>(bm.ptr);
    const double *Qp = static_cast<const double *>(bq.ptr);
    const int *gp = static_cast<const int *>(bg.ptr);
    std::vector<double> rv((size_t)n * 6), J((size_t)n * 36), dr((size_t)n * 6, 0.0);
    for (int f = 0; f < K; f += 6) {
        bool any = false;
        for (int g = 0; g < G; ++g) if (Mp[(size_t)f * G + g] > 0.5) { any = true; break; }
        if (!any) continue;
        qoejopt::oe2rvJacHost(z + (size_t)f * n * 6, n, rv.data(), J.data());
        std::fill(dr.begin(), dr.end(), 0.0);
        for (int g = 0; g < G; ++g) {
            if (Mp[(size_t)f * G + g] <= 0.5) continue;
            const int s = gp[g];
            const double *q = Qp + ((size_t)f * G + g) * 3;
            const double *p = rv.data() + (size_t)s * 6;
            const double *dd = dr.data() + (size_t)s * 6;
            double dv[3] = {p[0] + dd[0] - q[0], p[1] + dd[1] - q[1], p[2] + dd[2] - q[2]};
            double dist = std::sqrt(dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2]);
            if (dist < 1e-12) continue;
            double u[3] = {dv[0] / dist, dv[1] / dist, dv[2] / dist};
            double r = dist - Dp[(size_t)f * G + g];
            for (int c = 0; c < 3; ++c) dr[(size_t)s * 6 + c] -= r * u[c];
        }
        for (int s = 0; s < n; ++s) {
            const double *d = dr.data() + (size_t)s * 6;
            if (std::abs(d[0]) + std::abs(d[1]) + std::abs(d[2]) < 1e-12) continue;
            double A[6][6], b[6] = {d[0], d[1], d[2], 0.0, 0.0, 0.0}, x[6];
            const double *Js = J.data() + (size_t)s * 36;
            for (int a = 0; a < 6; ++a) for (int c = 0; c < 6; ++c) A[a][c] = Js[a * 6 + c];
            _solve6(A, b, x);
            for (int c = 0; c < 6; ++c) z[((size_t)f * n + s) * 6 + c] += x[c];
        }
    }
    return out;
}

static py::object win_batch_assemble(
        py::array_t<double, py::array::c_style | py::array::forcecast> X,
        py::array_t<double, py::array::c_style | py::array::forcecast> rv,
        py::array_t<double, py::array::c_style | py::array::forcecast> Ap,
        py::array_t<int, py::array::c_style | py::array::forcecast> isl_i,
        py::array_t<int, py::array::c_style | py::array::forcecast> isl_jj,
        py::array_t<double, py::array::c_style | py::array::forcecast> isl_D,
        py::array_t<double, py::array::c_style | py::array::forcecast> isl_W,
        py::array_t<int, py::array::c_style | py::array::forcecast> gts_a,
        py::array_t<double, py::array::c_style | py::array::forcecast> gts_D,
        py::array_t<double, py::array::c_style | py::array::forcecast> gts_W,
        py::array_t<double, py::array::c_style | py::array::forcecast> gts_Q,
        py::object beta, py::array_t<double, py::array::c_style | py::array::forcecast> qingt,
        py::array_t<double, py::array::c_style | py::array::forcecast> z,
        py::array_t<double, py::array::c_style | py::array::forcecast> u,
        py::object Lp, py::object zp,
        int n, int L, int E, int G, int nnz, int nrows,
        int off_isl, int off_gts, int off_dyn, int off_damp,
        double dt, double s_isl, double s_gts, double rho, int prior_mode, int prior_node,
        uint64_t A_device_ptr, uint64_t b_device_ptr) {
    auto bx = X.request(); auto br = rv.request(); auto ba = Ap.request();
    const int B = (int)bx.shape[0];
    if ((int)bx.shape[0] != B || bx.ndim != 4 ||
        (int)bx.shape[1] != L || (int)bx.shape[2] != n || (int)bx.shape[3] != 6)
        throw std::runtime_error("win_batch_assemble: X 需 (B,L,n,6)");
    auto cuda_up_d = [](py::array_t<double, py::array::c_style | py::array::forcecast> &a) {
        auto r = a.request(); double *d = nullptr;
        cudaMalloc(&d, (size_t)r.size * sizeof(double));
        cudaMemcpy(d, r.ptr, (size_t)r.size * sizeof(double), cudaMemcpyHostToDevice); return d;
    };
    auto cuda_up_i = [](py::array_t<int, py::array::c_style | py::array::forcecast> &a) {
        auto r = a.request(); int *d = nullptr;
        cudaMalloc(&d, (size_t)r.size * sizeof(int));
        cudaMemcpy(d, r.ptr, (size_t)r.size * sizeof(int), cudaMemcpyHostToDevice); return d;
    };
    qoejopt::WinBatchCtx c;
    c.B = B; c.n = n; c.L = L; c.E = E; c.G = G; c.ncol = L * n * 6;
    c.nnz = nnz; c.nrows = nrows; c.off_isl = off_isl; c.off_gts = off_gts;
    c.off_dyn = off_dyn; c.off_damp = off_damp; c.dt = dt; c.s_isl = s_isl;
    c.s_gts = s_gts; c.rho = rho; c.prior_mode = prior_mode; c.prior_node = prior_node;
    c.d_X = cuda_up_d(X); c.d_rv = cuda_up_d(rv); c.d_Ap = cuda_up_d(Ap);
    c.d_isl_i = cuda_up_i(isl_i); c.d_isl_jj = cuda_up_i(isl_jj);
    c.d_isl_D = cuda_up_d(isl_D); c.d_isl_W = cuda_up_d(isl_W);
    c.d_gts_a = cuda_up_i(gts_a); c.d_gts_D = cuda_up_d(gts_D);
    c.d_gts_W = cuda_up_d(gts_W); c.d_gts_Q = cuda_up_d(gts_Q);
    c.d_qingt = cuda_up_d(qingt);
    c.d_z = cuda_up_d(z); c.d_u = cuda_up_d(u);
    if (!Lp.is_none()) {
        auto a = Lp.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        c.d_Lp = cuda_up_d(a);
        auto b = zp.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        c.d_zp = cuda_up_d(b);
    }
    if (!beta.is_none()) {
        auto a = beta.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        c.d_beta = cuda_up_d(a);
    }
    const int nk = (L - 1) * n;
    double *d_end = nullptr, *d_phi = nullptr;
    if (nk > 0) {
        cudaMalloc(&d_end, (size_t)B * nk * 6 * sizeof(double));
        cudaMalloc(&d_phi, (size_t)B * nk * 36 * sizeof(double));
    }
    c.d_end = d_end; c.d_phi = d_phi;
    double *d_A = nullptr, *d_b = nullptr;
    const bool ext_out = (A_device_ptr != 0 || b_device_ptr != 0);
    if (A_device_ptr != 0 && b_device_ptr != 0) {
        d_A = reinterpret_cast<double *>(A_device_ptr);
        d_b = reinterpret_cast<double *>(b_device_ptr);
    } else {
        cudaMalloc(&d_A, (size_t)B * nnz * sizeof(double));
        cudaMalloc(&d_b, (size_t)B * nrows * sizeof(double));
    }
    cudaMemset(d_A, 0, (size_t)B * nnz * sizeof(double));
    cudaMemset(d_b, 0, (size_t)B * nrows * sizeof(double));
    c.d_A = d_A; c.d_b = d_b;
    {
        py::gil_scoped_release rel;
        qoejopt::winAssembleBatchDevice(c);
    }
    py::object out = py::none();
    if (!ext_out) {
        py::array_t<double> A_out({B, nnz});
        py::array_t<double> b_out({B, nrows});
        cudaMemcpy(A_out.request().ptr, d_A, (size_t)B * nnz * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(b_out.request().ptr, d_b, (size_t)B * nrows * sizeof(double), cudaMemcpyDeviceToHost);
        out = py::make_tuple(A_out, b_out);
    }
    if (!ext_out) { cudaFree(d_A); cudaFree(d_b); }
    cudaFree(d_end); cudaFree(d_phi);
    cudaFree(const_cast<double*>(c.d_X)); cudaFree(const_cast<double*>(c.d_rv));
    cudaFree(const_cast<double*>(c.d_Ap));
    cudaFree(const_cast<int*>(c.d_isl_i)); cudaFree(const_cast<int*>(c.d_isl_jj));
    cudaFree(const_cast<double*>(c.d_isl_D)); cudaFree(const_cast<double*>(c.d_isl_W));
    cudaFree(const_cast<int*>(c.d_gts_a)); cudaFree(const_cast<double*>(c.d_gts_D));
    cudaFree(const_cast<double*>(c.d_gts_W)); cudaFree(const_cast<double*>(c.d_gts_Q));
    cudaFree(const_cast<double*>(c.d_qingt));
    cudaFree(const_cast<double*>(c.d_z)); cudaFree(const_cast<double*>(c.d_u));
    if (c.d_Lp) cudaFree(const_cast<double*>(c.d_Lp));
    if (c.d_zp) cudaFree(const_cast<double*>(c.d_zp));
    if (c.d_beta) cudaFree(const_cast<double*>(c.d_beta));
    return out;
}

static double win_assemble(py::array_t<double, py::array::c_style | py::array::forcecast> X,
                           py::array_t<double, py::array::c_style | py::array::forcecast> rv,
                           py::array_t<double, py::array::c_style | py::array::forcecast> Ap,
                           uint64_t Aval_ptr, uint64_t b_ptr) {
    qoejopt::WinCtx &c = g_wctx;
    if (!c.d_X) throw std::runtime_error("qoejopt.win_assemble: 未先 setup_win");
    auto bx = X.request(); auto br = rv.request(); auto ba = Ap.request();
    if ((std::size_t)bx.size != (std::size_t)c.L * c.n * 6) throw std::runtime_error("win_assemble: X 长度");
    if ((std::size_t)ba.size != (std::size_t)c.L * c.n * 18) throw std::runtime_error("win_assemble: Ap 长度");
    py::gil_scoped_release rel;
    return qoejopt::winAssembleDevice(c, static_cast<const double *>(bx.ptr),
                                      static_cast<const double *>(br.ptr),
                                      static_cast<const double *>(ba.ptr),
                                      reinterpret_cast<double *>(Aval_ptr),
                                      reinterpret_cast<double *>(b_ptr));
}

// ---- 整弧单发：先验协方差 / STM 同源切换 / 段级子步 / 可观测性 ----
// host 6×6 对称特征分解（Jacobi），回 A_prior=chol(P0^-1)^T 的 (n,36) 因子（行 a 全 6 列）。
static void _host_info_fac(const double *P, int n, std::vector<double> &fac) {
    fac.assign((std::size_t)n * 36, 0.0);
    for (int i = 0; i < n; ++i) {
        double A[6][6], V[6][6];
        for (int a = 0; a < 6; ++a)
            for (int b = 0; b < 6; ++b) { A[a][b] = P[(std::size_t)i*36 + a*6 + b]; V[a][b] = (a == b) ? 1.0 : 0.0; }
        for (int sweep = 0; sweep < 40; ++sweep) {
            double off = 0.0;
            for (int p = 0; p < 6; ++p) for (int q = p + 1; q < 6; ++q) off += A[p][q] * A[p][q];
            if (off < 1e-30) break;
            for (int p = 0; p < 6; ++p) for (int q = p + 1; q < 6; ++q) {
                if (fabs(A[p][q]) < 1e-300) continue;
                double theta = (A[q][q] - A[p][p]) / (2.0 * A[p][q]);
                double t = (theta >= 0.0 ? 1.0 : -1.0) / (fabs(theta) + sqrt(theta * theta + 1.0));
                double c = 1.0 / sqrt(t * t + 1.0), s = t * c;
                for (int k = 0; k < 6; ++k) { double akp = A[k][p], akq = A[k][q]; A[k][p] = c*akp - s*akq; A[k][q] = s*akp + c*akq; }
                for (int k = 0; k < 6; ++k) { double apk = A[p][k], aqk = A[q][k]; A[p][k] = c*apk - s*aqk; A[q][k] = s*apk + c*aqk; }
                for (int k = 0; k < 6; ++k) { double vkp = V[k][p], vkq = V[k][q]; V[k][p] = c*vkp - s*vkq; V[k][q] = s*vkp + c*vkq; }
            }
        }
        // w[a] = A[a][a]（特征值，升序无妨）；A_prior[a][b] = V[b][a]/sqrt(clip(w[a]))
        for (int a = 0; a < 6; ++a) {
            double w = A[a][a] < 1e-3 ? 1e-3 : A[a][a];
            double si = 1.0 / sqrt(w);
            for (int b = 0; b < 6; ++b) fac[(std::size_t)i*36 + a*6 + b] = V[b][a] * si;
        }
    }
}

// FLS：逐星边缘协方差 P(n,6,6) → 白化因子 Lp(n,6,6)（Lpᵀ·Lp = P^{-1}），供 win_prior_asm。
static py::array_t<double> prior_fac_ss(py::array_t<double, py::array::c_style | py::array::forcecast> P) {
    auto b = P.request();
    if (b.ndim != 3 || b.shape[1] != 6 || b.shape[2] != 6)
        throw std::runtime_error("prior_fac_ss: P 需 (n,6,6)");
    const int n = (int)b.shape[0];
    std::vector<double> fac;
    _host_info_fac(static_cast<const double *>(b.ptr), n, fac);
    py::array_t<double> out({(py::ssize_t)n, (py::ssize_t)6, (py::ssize_t)6});
    std::memcpy(out.mutable_data(), fac.data(), fac.size() * sizeof(double));
    return out;
}

static void set_stm_gve(int flag) { g_ctx.stm_mode = flag ? 1 : 0; }
static void set_stm_mode(int mode) { g_ctx.stm_mode = (mode >= 0 && mode <= 2) ? mode : 2; }
static void set_sub_steps(int nsub) { g_ctx.sub_steps = nsub > 0 ? nsub : 1; }

static py::array_t<double> stm_fold_j2(py::array_t<double, py::array::c_style | py::array::forcecast> x,
                                       int nfr, double dt) {
    auto buf = x.request();
    if (buf.ndim != 2 || buf.shape[1] != 6) throw std::runtime_error("qoejopt.stm_fold_j2: x 需 (n,6)");
    const std::size_t n = (std::size_t)buf.shape[0];
    double *dx = nullptr, *dA = nullptr;
    cudaMalloc(&dx, n * 6 * sizeof(double));
    cudaMalloc(&dA, (std::size_t)nfr * n * 18 * sizeof(double));
    cudaMemcpy(dx, buf.ptr, n * 6 * sizeof(double), cudaMemcpyHostToDevice);
    { py::gil_scoped_release rel; qoejopt::stmFoldJ2Device(dx, (int)n, nfr, dt, dA, nullptr); }
    py::array_t<double> A({(py::ssize_t)nfr, (py::ssize_t)n, (py::ssize_t)18});
    cudaMemcpy(A.mutable_data(), dA, (std::size_t)nfr * n * 18 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(dx); cudaFree(dA);
    return A;
}

static py::array_t<double> stm_fold_gve(py::array_t<double, py::array::c_style | py::array::forcecast> x,
                                        int nfr, double dt, double beta) {
    auto buf = x.request();
    if (buf.ndim != 2 || buf.shape[1] != 6) throw std::runtime_error("qoejopt.stm_fold_gve: x 需 (n,6)");
    const std::size_t n = (std::size_t)buf.shape[0];
    double *dx = nullptr, *dA = nullptr;
    cudaMalloc(&dx, n * 6 * sizeof(double));
    cudaMalloc(&dA, (std::size_t)nfr * n * 18 * sizeof(double));
    cudaMemcpy(dx, buf.ptr, n * 6 * sizeof(double), cudaMemcpyHostToDevice);
    { py::gil_scoped_release rel; qoejopt::stmFoldGveDevice(dx, (int)n, nfr, dt, beta, dA); }
    py::array_t<double> A({(py::ssize_t)nfr, (py::ssize_t)n, (py::ssize_t)18});
    cudaMemcpy(A.mutable_data(), dA, (std::size_t)nfr * n * 18 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(dx); cudaFree(dA);
    return A;
}

static void set_prior_cov(py::array_t<double, py::array::c_style | py::array::forcecast> P0,
                          py::array_t<double, py::array::c_style | py::array::forcecast> zprior,
                          double sigma_a, int cov_mode) {
    Ctx &c = g_ctx;
    if (!c.d_X) throw std::runtime_error("qoejopt.set_prior_cov: 未先 setup");
    c.sigma_a = sigma_a; c.prior_mode = 1; c.cov_mode = cov_mode;
    if (!c.d_P0) cudaMalloc(&c.d_P0, (std::size_t)c.n * 36 * sizeof(double));
    if (!c.d_prior_fac) cudaMalloc(&c.d_prior_fac, (std::size_t)c.n * 36 * sizeof(double));
    if (!c.d_zprior) cudaMalloc(&c.d_zprior, (std::size_t)c.n * 6 * sizeof(double));
    if (!c.d_P0rv) cudaMalloc(&c.d_P0rv, (std::size_t)c.n * 36 * sizeof(double));
    if (!c.d_Pm) cudaMalloc(&c.d_Pm, (std::size_t)c.nfr * c.n * 36 * sizeof(double));
    if (!c.d_Pp) cudaMalloc(&c.d_Pp, (std::size_t)c.nfr * c.n * 36 * sizeof(double));
    if (!c.d_PpPrev) cudaMalloc(&c.d_PpPrev, (std::size_t)c.nfr * c.n * 36 * sizeof(double));
    if (!c.d_cov) cudaMalloc(&c.d_cov, (std::size_t)c.nfr * c.n * 36 * sizeof(double));
    if (!c.d_phi) cudaMalloc(&c.d_phi, (std::size_t)c.nfr * c.n * 36 * sizeof(double));
    _copy_to_dev(c.d_P0, P0, (std::size_t)c.n * 36);
    _copy_to_dev(c.d_zprior, zprior, (std::size_t)c.n * 6);
    std::vector<double> fac;
    _host_info_fac(static_cast<const double *>(P0.request().ptr), c.n, fac);
    cudaMemcpy(c.d_prior_fac, fac.data(), (std::size_t)c.n * 36 * sizeof(double), cudaMemcpyHostToDevice);
    { py::gil_scoped_release rel; qoejopt::jointSetCov0(c); }   // d_P0rv = J P0 Jᵀ
}

static py::array_t<double> win_prior_prop(py::array_t<double, py::array::c_style | py::array::forcecast> Pp,
                                          py::array_t<double, py::array::c_style | py::array::forcecast> Q,
                                          int stride) {
    Ctx &c = g_ctx;
    if (!c.d_phi_all) throw std::runtime_error("qoejopt.win_prior_prop: 未先 setup");
    const std::size_t sz = (std::size_t)c.n * 36;
    std::vector<double> Pn(sz);
    {
        py::gil_scoped_release rel;
        qoejopt::winPriorPropDevice(c, static_cast<const double *>(Pp.request().ptr),
                                    static_cast<const double *>(Q.request().ptr), stride, Pn.data());
    }
    py::array_t<double> out({(py::ssize_t)c.n, (py::ssize_t)6, (py::ssize_t)6});
    std::memcpy(out.mutable_data(), Pn.data(), sz * sizeof(double));
    return out;
}

static py::tuple obs_spectrum(int k) {
    Ctx &c = g_ctx;
    if (!c.d_X) throw std::runtime_error("qoejopt.obs_spectrum: 未先 setup");
    const int ncol = c.ncol;
    double *d_H = nullptr;
    cudaMalloc(&d_H, (std::size_t)ncol * ncol * sizeof(double));
    std::vector<double> eval, evec;
    {
        py::gil_scoped_release rel;
        qoejopt::obsFormHDevice(c, d_H);
        qoejopt::obsEigDevice(c, d_H, k, eval, evec);
    }
    cudaFree(d_H);
    py::array_t<double> EV((py::ssize_t)ncol);
    std::memcpy(EV.mutable_data(), eval.data(), eval.size() * sizeof(double));
    py::array_t<double> VEC({(py::ssize_t)ncol, (py::ssize_t)k});
    double *vp = VEC.mutable_data();
    for (int j = 0; j < k; ++j)
        for (int i = 0; i < ncol; ++i) vp[(std::size_t)i * k + j] = evec[(std::size_t)j * ncol + i];
    return py::make_tuple(EV, VEC);
}

static py::tuple obs_spectrum_range(int k) {
    Ctx &c = g_ctx;
    if (!c.d_X) throw std::runtime_error("qoejopt.obs_spectrum_range: 未先 setup");
    const int ncol = c.ncol;
    const int nk = std::max(1, std::min(k, ncol));
    double *d_H = nullptr;
    cudaMalloc(&d_H, (std::size_t)ncol * ncol * sizeof(double));
    std::vector<double> eval, evec;
    {
        py::gil_scoped_release rel;
        qoejopt::obsFormHDevice(c, d_H);
        qoejopt::obsEigRangeDevice(c, d_H, nk, eval, evec);
    }
    cudaFree(d_H);
    py::array_t<double> EV((py::ssize_t)eval.size());
    std::memcpy(EV.mutable_data(), eval.data(), eval.size() * sizeof(double));
    const int nk_out = (int)eval.size();
    py::array_t<double> VEC({(py::ssize_t)ncol, (py::ssize_t)nk_out});
    double *vp = VEC.mutable_data();
    for (int j = 0; j < nk_out; ++j)
        for (int i = 0; i < ncol; ++i) vp[(std::size_t)i * nk_out + j] = evec[(std::size_t)j * ncol + i];
    return py::make_tuple(EV, VEC);
}

static void obs_set_scale(py::array_t<double, py::array::c_style | py::array::forcecast> scale) {
    Ctx &c = g_ctx;
    if (!c.d_X) throw std::runtime_error("qoejopt.obs_set_scale: 未先 setup");
    auto b = scale.request();
    if ((int)b.size != c.ncol) throw std::runtime_error("qoejopt.obs_set_scale: 长度 != ncol");
    qoejopt::obsSetScaleDevice(c, static_cast<const double *>(b.ptr));
}

static py::array_t<double> obs_apply_h(py::array_t<double, py::array::c_style | py::array::forcecast> x,
                                       int mode) {
    Ctx &c = g_ctx;
    if (!c.d_X) throw std::runtime_error("qoejopt.obs_apply_h: 未先 setup");
    auto bx = x.request();
    if ((int)bx.size != c.ncol) throw std::runtime_error("qoejopt.obs_apply_h: x 长度 != ncol");
    double *dx = nullptr, *dy = nullptr;
    cudaMalloc(&dx, (std::size_t)c.ncol * sizeof(double));
    cudaMalloc(&dy, (std::size_t)c.ncol * sizeof(double));
    cudaMemcpy(dx, bx.ptr, (std::size_t)c.ncol * sizeof(double), cudaMemcpyHostToDevice);
    {
        py::gil_scoped_release rel;
        qoejopt::obsApplyHDevice(c, dx, dy, mode);
    }
    py::array_t<double> out((py::ssize_t)c.ncol);
    cudaMemcpy(out.mutable_data(), dy, (std::size_t)c.ncol * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(dx); cudaFree(dy);
    return out;
}

static py::array_t<double> copy_A() {
    Ctx &c = g_ctx;
    if (!c.d_A) throw std::runtime_error("qoejopt.copy_A: 未先 setup");
    py::array_t<double> A({(py::ssize_t)c.nfr, (py::ssize_t)c.n,
                           (py::ssize_t)3, (py::ssize_t)6});
    cudaMemcpy(A.mutable_data(), c.d_A,
               (std::size_t)c.nfr * c.n * 18 * sizeof(double), cudaMemcpyDeviceToHost);
    return A;
}

static py::array_t<double> obs_perstar() {
    Ctx &c = g_ctx;
    if (!c.d_X) throw std::runtime_error("qoejopt.obs_perstar: 未先 setup");
    std::vector<double> sv((std::size_t)c.n * 6);
    { py::gil_scoped_release rel; qoejopt::obsPerStarDevice(c, sv.data()); }
    py::array_t<double> out({(py::ssize_t)c.n, (py::ssize_t)6});
    std::memcpy(out.mutable_data(), sv.data(), sv.size() * sizeof(double));
    return out;
}

static py::array_t<double> copy_rv() {
    Ctx &c = g_ctx;
    if (!c.d_X) throw std::runtime_error("qoejopt.copy_rv: 未先 setup");
    std::vector<double> h((std::size_t)c.nfr * c.n * 6);
    { py::gil_scoped_release rel; qoejopt::jointCopyOut(c, h.data()); }
    py::array_t<double> out({(py::ssize_t)c.nfr, (py::ssize_t)c.n, (py::ssize_t)6});
    std::memcpy(out.mutable_data(), h.data(), h.size() * sizeof(double));
    return out;
}

// ---------------- 子块内环（144 块锚帧 ΔQOE）host 包装 ----------------
static qoejopt::BlkCtx g_blk;

static void blk_free() {
    qoejopt::BlkCtx &c = g_blk;
    if (c.d_ea) cudaFree(c.d_ea);
    if (c.d_lo) cudaFree(c.d_lo);
    if (c.d_gi) cudaFree(c.d_gi);
    double *pd[] = {c.d_islD, c.d_islM, c.d_gtsD, c.d_gtsM, c.d_gtsQ, c.d_base, c.d_dq,
                    c.d_Uti, c.d_priSig, c.d_r2P, c.d_truthA, c.d_truth4, c.d_y, c.d_q,
                    c.d_rv0, c.d_J0, c.d_I36, c.d_P, c.d_A, c.d_Pf, c.d_A4};
    for (auto p : pd)
        if (p) cudaFree(p);
    g_blk = qoejopt::BlkCtx{};
}

static void blk_setup(int B, int K, int n4, int gmax, int G, double s_isl, double s_gts,
                      double s_eph, double s_eph4, int prior_mode, int use_eph, double sigma_prior,
                      py::array_t<int, py::array::c_style | py::array::forcecast> ea,
                      py::array_t<double, py::array::c_style | py::array::forcecast> isl_D,
                      py::array_t<double, py::array::c_style | py::array::forcecast> isl_M,
                      py::array_t<int, py::array::c_style | py::array::forcecast> lo,
                      py::array_t<int, py::array::c_style | py::array::forcecast> gi,
                      py::array_t<double, py::array::c_style | py::array::forcecast> gts_D,
                      py::array_t<double, py::array::c_style | py::array::forcecast> gts_M,
                      py::array_t<double, py::array::c_style | py::array::forcecast> gts_Q) {
    auto pe = ea.request();
    auto pID = isl_D.request();
    auto pIM = isl_M.request();
    auto pl = lo.request();
    auto pg = gi.request();
    auto pGD = gts_D.request();
    auto pGM = gts_M.request();
    auto pGQ = gts_Q.request();
    qoejopt::blkSetupDevice(g_blk, B, K, n4, gmax, G, s_isl, s_gts, s_eph, s_eph4, prior_mode, use_eph,
                            sigma_prior, static_cast<const int *>(pe.ptr),
                            static_cast<const double *>(pID.ptr), static_cast<const double *>(pIM.ptr),
                            static_cast<const int *>(pl.ptr), static_cast<const int *>(pg.ptr),
                            static_cast<const double *>(pGD.ptr), static_cast<const double *>(pGM.ptr),
                            static_cast<const double *>(pGQ.ptr));
}

static void blk_set_window(py::array_t<double, py::array::c_style | py::array::forcecast> base,
                           py::array_t<double, py::array::c_style | py::array::forcecast> dq,
                           py::object Uti, py::object pri_sig, py::object r2P,
                           py::object truth_a, py::object truth_4) {
    auto pb = base.request();
    auto pd = dq.request();
    py::array_t<double, py::array::c_style | py::array::forcecast> aU, aS, aR, aTa, aT4;
    const double *pU = nullptr, *pS = nullptr, *pR = nullptr, *pTa = nullptr, *pT4 = nullptr;
    if (!Uti.is_none()) { aU = Uti.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(); pU = static_cast<const double *>(aU.request().ptr); }
    if (!pri_sig.is_none()) { aS = pri_sig.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(); pS = static_cast<const double *>(aS.request().ptr); }
    if (!r2P.is_none()) { aR = r2P.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(); pR = static_cast<const double *>(aR.request().ptr); }
    if (!truth_a.is_none()) { aTa = truth_a.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(); pTa = static_cast<const double *>(aTa.request().ptr); }
    if (!truth_4.is_none()) { aT4 = truth_4.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(); pT4 = static_cast<const double *>(aT4.request().ptr); }
    qoejopt::blkSetWindowDevice(g_blk, static_cast<const double *>(pb.ptr),
                                static_cast<const double *>(pd.ptr), pU, pS, pR, pTa, pT4);
}

static void blk_expand(py::array_t<double, py::array::c_style | py::array::forcecast> y,
                       uint64_t A24_ptr, uint64_t Apar_ptr, uint64_t r_ptr) {
    auto py_ = y.request();
    const double *hY = static_cast<const double *>(py_.ptr);
    py::gil_scoped_release rel;
    qoejopt::blkExpandDevice(g_blk, hY, reinterpret_cast<double *>(A24_ptr),
                             reinterpret_cast<double *>(Apar_ptr),
                             reinterpret_cast<double *>(r_ptr));
}

static py::tuple blk_copy_rv() {
    qoejopt::BlkCtx &c = g_blk;
    const int n5 = c.B * 5;
    py::array_t<double> rvK({(py::ssize_t)c.K, (py::ssize_t)n5, (py::ssize_t)6});
    py::array_t<double> rv4({(py::ssize_t)n5, (py::ssize_t)6});
    double *pK = rvK.mutable_data(), *p4 = rv4.mutable_data();
    { py::gil_scoped_release rel; qoejopt::blkCopyRvDevice(c, pK, p4); }
    return py::make_tuple(rvK, rv4);
}


static py::array_t<double> gts_weak_cuda(
        py::array_t<double, py::array::c_style | py::array::forcecast> Q,
        py::array_t<double, py::array::c_style | py::array::forcecast> D,
        py::array_t<double, py::array::c_style | py::array::forcecast> M,
        py::array_t<int, py::array::c_style | py::array::forcecast> edges,
        py::array_t<double, py::array::c_style | py::array::forcecast> Dg,
        py::array_t<double, py::array::c_style | py::array::forcecast> Mg,
        py::array_t<double, py::array::c_style | py::array::forcecast> Qg,
        py::array_t<int, py::array::c_style | py::array::forcecast> gsat,
        int nweak, double sigma_gts=19.0, double lam=1e-2) {
    auto qb = Q.request(); auto db = D.request(); auto mb = M.request();
    auto eb = edges.request(); auto dgb = Dg.request(); auto mgb = Mg.request();
    auto qgb = Qg.request(); auto gb = gsat.request();
    if (qb.ndim != 3 || qb.shape[2] != 6) throw std::runtime_error("gts_weak_cuda: Q 需 (K,n,6)");
    const int K = (int)qb.shape[0], n = (int)qb.shape[1], E = (int)eb.shape[0], G = (int)mgb.shape[1];
    const int ns = n * 6;
    std::vector<double> q((double*)qb.ptr, (double*)qb.ptr + (size_t)K * ns);
    const double *Dp = (const double*)db.ptr, *Mp = (const double*)mb.ptr;
    const int *ep = (const int*)eb.ptr;
    const double *Dgp = (const double*)dgb.ptr, *Mgp = (const double*)mgb.ptr, *Qgp = (const double*)qgb.ptr;
    const int *gsp = (const int*)gb.ptr;
    const double scale6[6] = {10.0, 1e-4, 1e-4, 1e-4, 1e-5, 1e-5};
    Eigen::VectorXd scale(ns);
    for (int i = 0; i < ns; ++i) scale[i] = scale6[i % 6];
    Eigen::MatrixXd F(ns, ns), Fw(ns, ns);
    std::vector<double> rv(n * 6), J(n * 36);
    cusolverDnHandle_t h = nullptr; cusolverDnCreate(&h);
    double *dA = nullptr, *dW = nullptr, *dEig = nullptr; int *dInfo = nullptr;
    cudaMalloc(&dA, (size_t)ns * ns * sizeof(double));
    cudaMalloc(&dEig, (size_t)ns * sizeof(double));
    cudaMalloc(&dInfo, sizeof(int));
    double *hEvec = (double*)malloc((size_t)ns * nweak * sizeof(double));
    int nwrite = 0;
    for (int f = 0; f < K; ++f) {
        bool has = false;
        for (int g = 0; g < G; ++g) if (Mgp[(size_t)f * G + g] > 0.5) { has = true; break; }
        if (!has) continue;
        qoejopt::oe2rvJacHost(q.data() + (size_t)f * ns, n, rv.data(), J.data());
        F.setZero();
        for (int e = 0; e < E; ++e) {
            if (Mp[(size_t)f * E + e] <= 0.5) continue;
            int i = ep[2 * e], j = ep[2 * e + 1];
            double dx = rv[6 * i] - rv[6 * j], dy = rv[6 * i + 1] - rv[6 * j + 1], dz = rv[6 * i + 2] - rv[6 * j + 2];
            double r = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (r < 1e-12) continue;
            double u[3] = {dx / r, dy / r, dz / r};
            double ri[6], rj[6];
            for (int a = 0; a < 6; ++a) {
                ri[a] = (u[0] * J[i * 36 + a] + u[1] * J[i * 36 + 6 + a] + u[2] * J[i * 36 + 12 + a]) / 30.0;
                rj[a] = -(u[0] * J[j * 36 + a] + u[1] * J[j * 36 + 6 + a] + u[2] * J[j * 36 + 12 + a]) / 30.0;
            }
            for (int a = 0; a < 6; ++a) for (int b = 0; b < 6; ++b) {
                F(i * 6 + a, i * 6 + b) += ri[a] * ri[b];
                F(i * 6 + a, j * 6 + b) += ri[a] * rj[b];
                F(j * 6 + a, i * 6 + b) += rj[a] * ri[b];
                F(j * 6 + a, j * 6 + b) += rj[a] * rj[b];
            }
        }
        for (int g = 0; g < G; ++g) {
            if (Mgp[(size_t)f * G + g] <= 0.5) continue;
            int i = gsp[g];
            double dx = rv[6 * i] - Qgp[(size_t)f * G * 3 + 3 * g];
            double dy = rv[6 * i + 1] - Qgp[(size_t)f * G * 3 + 3 * g + 1];
            double dz = rv[6 * i + 2] - Qgp[(size_t)f * G * 3 + 3 * g + 2];
            double r = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (r < 1e-12) continue;
            double u[3] = {dx / r, dy / r, dz / r};
            double ri[6];
            for (int a = 0; a < 6; ++a)
                ri[a] = (u[0] * J[i * 36 + a] + u[1] * J[i * 36 + 6 + a] + u[2] * J[i * 36 + 12 + a]) / sigma_gts;
            for (int a = 0; a < 6; ++a) for (int b = 0; b < 6; ++b)
                F(i * 6 + a, i * 6 + b) += ri[a] * ri[b];
        }
        Fw = scale.asDiagonal() * F * scale.asDiagonal();
        cudaMemcpy(dA, Fw.data(), (size_t)ns * ns * sizeof(double), cudaMemcpyHostToDevice);
        int lwork = 0, h_me = 0;
        cusolverDnDsyevdx_bufferSize(h, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_I,
                                     CUBLAS_FILL_MODE_LOWER, ns, dA, ns, 0.0, 0.0,
                                     1, nweak, &h_me, dEig, &lwork);
        if (!dW) cudaMalloc(&dW, (size_t)lwork * sizeof(double));
        cudaMemset(dA, 0, (size_t)ns * ns * sizeof(double));
        cudaMemcpy(dA, Fw.data(), (size_t)ns * ns * sizeof(double), cudaMemcpyHostToDevice);
        int h_m = 0;
        cusolverDnDsyevdx(h, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_I,
                          CUBLAS_FILL_MODE_LOWER, ns, dA, ns, 0.0, 0.0,
                          1, nweak, &h_m, dEig, dW, lwork, dInfo);
        cudaDeviceSynchronize();
        int h_info = -1;
        cudaMemcpy(&h_info, dInfo, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(hEvec, dA, (size_t)ns * nweak * sizeof(double), cudaMemcpyDeviceToHost);
        std::vector<double> V((size_t)ns * nweak);
        double vmax = 0.0;
        for (int k = 0; k < nweak; ++k)
            for (int i = 0; i < ns; ++i) {
                V[(size_t)k * ns + i] = scale[i] * hEvec[(size_t)k * ns + i];
                vmax = std::max(vmax, std::abs(V[(size_t)k * ns + i]));
            }
        if (nwrite < 2)
            fprintf(stderr, "[gts_cpp] f=%d meig=%d info=%d Vmax=%.3e\n", f, h_me, h_info, vmax);
        for (int g = 0; g < G; ++g) {
            if (Mgp[(size_t)f * G + g] <= 0.5) continue;
            int i = gsp[g];
            double dx = rv[6 * i] - Qgp[(size_t)f * G * 3 + 3 * g];
            double dy = rv[6 * i + 1] - Qgp[(size_t)f * G * 3 + 3 * g + 1];
            double dz = rv[6 * i + 2] - Qgp[(size_t)f * G * 3 + 3 * g + 2];
            double r = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (r < 1e-12) continue;
            double u[3] = {dx / r, dy / r, dz / r};
            Eigen::VectorXd gg(nweak);
            for (int k = 0; k < nweak; ++k) {
                double s = 0.0;
                for (int a = 0; a < 6; ++a)
                    s += u[0] * J[i * 36 + 0 * 6 + a] * V[(size_t)k * ns + i * 6 + a]
                       + u[1] * J[i * 36 + 1 * 6 + a] * V[(size_t)k * ns + i * 6 + a]
                       + u[2] * J[i * 36 + 2 * 6 + a] * V[(size_t)k * ns + i * 6 + a];
                gg[k] = s / sigma_gts;
            }
            // 每帧一个 GTS 行，累积到 A^T A；用逐行 rank-1 更新，避免保存所有行。
            // 这里只做单条 GTS 的弱写回近似会过拟合，故先在帧末统一解；见下面 frame A。
        }
        // 帧内全部 GTS 行联合弱坐标最小二乘
        Eigen::MatrixXd Af(nweak, nweak); Af.setZero();
        Eigen::VectorXd bf(nweak); bf.setZero();
        int m = 0;
        for (int g = 0; g < G; ++g) {
            if (Mgp[(size_t)f * G + g] <= 0.5) continue;
            int i = gsp[g];
            double dx = rv[6 * i] - Qgp[(size_t)f * G * 3 + 3 * g];
            double dy = rv[6 * i + 1] - Qgp[(size_t)f * G * 3 + 3 * g + 1];
            double dz = rv[6 * i + 2] - Qgp[(size_t)f * G * 3 + 3 * g + 2];
            double r = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (r < 1e-12) continue;
            double u[3] = {dx / r, dy / r, dz / r};
            Eigen::VectorXd gg(nweak);
            for (int k = 0; k < nweak; ++k) {
                double s = 0.0;
                for (int a = 0; a < 6; ++a)
                    s += u[0] * J[i * 36 + 0 * 6 + a] * V[(size_t)k * ns + i * 6 + a]
                       + u[1] * J[i * 36 + 1 * 6 + a] * V[(size_t)k * ns + i * 6 + a]
                       + u[2] * J[i * 36 + 2 * 6 + a] * V[(size_t)k * ns + i * 6 + a];
                gg[k] = s / sigma_gts;
            }
            double res = (r - Dgp[(size_t)f * G + g]) / sigma_gts;
            Af += gg * gg.transpose();
            bf += gg * (-res);
            ++m;
        }
        if (m > 0) {
            Af.diagonal().array() += lam;
            Eigen::VectorXd aa = Af.ldlt().solve(bf);
            for (int i = 0; i < n; ++i)
                for (int k = 0; k < nweak; ++k)
                    for (int a = 0; a < 6; ++a)
                        q[(size_t)f * ns + i * 6 + a] += V[(size_t)k * ns + i * 6 + a] * aa[k];
            ++nwrite;
            if (nwrite < 3) fprintf(stderr, "[gts_cpp] f=%d m=%d nwrite=%d aa0=%.3e\n", f, m, nwrite, aa.norm());
        }
    }
    cudaFree(dA); cudaFree(dW); cudaFree(dEig); cudaFree(dInfo);
    cusolverDnDestroy(h); free(hEvec);
    py::array_t<double> out({(py::ssize_t)K, (py::ssize_t)n, (py::ssize_t)6});
    std::memcpy(out.mutable_data(), q.data(), q.size() * sizeof(double));
    return out;
}

PYBIND11_MODULE(qoejopt, m) {
    m.doc() = "joint 单发全局联合内环 C++/CUDA 核心（device 常驻装配；A_val/b 写调用方 torch cuda 缓冲）";
    m.def("setup", &setup, py::arg("n"), py::arg("nfr"), py::arg("E"), py::arg("G"),
          py::arg("ncol"), py::arg("nnz"), py::arg("nrows"),
          py::arg("off_isl"), py::arg("off_gts"), py::arg("off_damp"),
          py::arg("dt"), py::arg("s_isl"), py::arg("s_gts"), py::arg("issq"),
          py::arg("isl_i"), py::arg("isl_jj"), py::arg("isl_D"), py::arg("isl_W"),
          py::arg("gts_a"), py::arg("gts_D"), py::arg("gts_W"), py::arg("gts_Q"),
          py::arg("ctr"), "一次注册结构/观测到 device");
    m.def("assemble", &assemble, py::arg("x"), py::arg("rv0"), py::arg("A0"),
          py::arg("A_val_ptr"), py::arg("b_ptr"),
          "展开+装配：x(n,6),rv0(n,6),A0(n,36)；A_val/b 写 device 指针；返回 res2");
    m.def("expand", &expand_res2, py::arg("x"), py::arg("rv0"), py::arg("A0"),
          "展开(GVE)+STM折叠+算 res2（线搜索；刷新 d_P/d_A/d_X）");
    m.def("assemble_cached", &assemble_cached, py::arg("A_val_ptr"), py::arg("b_ptr"),
          "用当前 d_P/d_A/d_X 装配 A_val/b（不重展开）");
    m.def("expand_rv", &expand_rv, py::arg("x"), "整弧 rv (nfr,n,6) → numpy");
    m.def("state_sens", &state_sens, py::arg("x"), py::arg("beta") = 1.0,
          "P4.2 前向敏度：(oe_all, S=∂oe/∂κ) 均 (nfr,n,6)");
    m.def("gve_rbf_sens", &gve_rbf_sens, py::arg("oe"), py::arg("nfr"), py::arg("dt"),
          py::arg("beta") = 1.0, py::arg("m") = 40,
          "整弧 RBF θ 前向敏度：oe(n,6) → (oe_all(nfr,n,6), S_all(nfr,n,6,m))（centers/s 需先 set_rbf_cuda）");
    m.def("set_pole", [](double x, double y, double z){ kep3::setPole(x, y, z); },
          py::arg("x"), py::arg("y"), py::arg("z"),
          "设置当日真极（ICRF 单位向量），与本模块 CUDA tbp 核共享");
    m.def("set_rbf_cuda", [](py::array_t<double, py::array::c_style | py::array::forcecast> centers, double s,
                             py::array_t<double, py::array::c_style | py::array::forcecast> w) {
        auto bc = centers.request();
        std::vector<std::array<double, 3>> C;
        if (bc.ndim == 2 && bc.shape[1] == 3) {
            const double *p = static_cast<const double *>(bc.ptr);
            for (py::ssize_t k = 0; k < bc.shape[0]; ++k) C.push_back({p[3 * k], p[3 * k + 1], p[3 * k + 2]});
        }
        auto bw = w.request();
        std::vector<double> wv(static_cast<double *>(bw.ptr), static_cast<double *>(bw.ptr) + bw.size);
        { py::gil_scoped_release rel; kep3::setRbfCuda(C, s, wv); }
    }, py::arg("centers"), py::arg("s"), py::arg("w"),
    "设定 qoejopt 侧 RBF 力场（静态库使 __constant__ 在 qoe/qoejopt 各一份，需各自设定）");
    m.def("gts_delta_chain", &gts_delta_chain, py::arg("Z"), py::arg("D"), py::arg("M"),
          py::arg("Qg"), py::arg("gsat"), py::arg("K"), py::arg("sigma_gts") = 19.0,
          py::arg("iters") = 2,
          "固定 ISL QOE 序列，所有窗加同一个 delta QOE，16T GTS 全局最小化");
    m.def("gts_z_chain", &gts_z_chain, py::arg("Z"), py::arg("D"), py::arg("M"),
          py::arg("Qg"), py::arg("gsat"), py::arg("K"), py::arg("sigma_gts") = 19.0,
          py::arg("sigma_dyn") = 10.0, py::arg("iters") = 2, py::arg("sigma_p") = 1e4,
          py::arg("mode") = 2,
          "16T QOE0 链 GTS 全局正规方程（mode 0=GTS off, 1=GTS only, 2=all）");
    m.def("gve_stm", &gve_stm, py::arg("x"), py::arg("dt"), py::arg("beta") = 1.0,
          "单步 GVE + 全动力学变分 STM：x(n,6) oe → (oef(n,6), Φ_oe(n,6,6))");
    m.def("gve_stm_ia", &gve_stm_ia, py::arg("x"), py::arg("dt"), py::arg("beta") = 1.0,
          "一阶同源 STM：Φ≈I+A·dt（A=∂f/∂oe 解析）→ (oef(n,6), Φ(n,6,6))");
    m.def("gts_weak_cuda", &gts_weak_cuda, py::arg("Q"), py::arg("D"), py::arg("M"),
          py::arg("edges"), py::arg("Dg"), py::arg("Mg"), py::arg("Qg"), py::arg("gsat"),
          py::arg("nweak"), py::arg("sigma_gts") = 19.0, py::arg("lam") = 1e-2,
          "CUDA/C++ GTS 逐帧弱子空间写回：全 1152 星、6912 维、cuSOLVER syevdx");
    m.def("oe2rv_jac", [](py::array_t<double, py::array::c_style | py::array::forcecast> X) {
            auto buf = X.request();
            if (buf.ndim != 2 || buf.shape[1] != 6)
                throw std::runtime_error("qoejopt.oe2rv_jac: 需 (n,6) 数组");
            const int n = (int)buf.shape[0];
            py::array_t<double> rv({(py::ssize_t)n, (py::ssize_t)6});
            py::array_t<double> J({(py::ssize_t)n, (py::ssize_t)36});
            { py::gil_scoped_release rel;
              qoejopt::oe2rvJacHost((const double *)buf.ptr, n, rv.mutable_data(), J.mutable_data()); }
            return py::make_tuple(rv, J);
          }, py::arg("X"), "CUDA 批量 QOE→rv 与解析 J=∂rv/∂oe：(n,6) → (rv (n,6), J (n,36))");
    m.def("gve_ctrl", &gve_ctrl, py::arg("x"),
          "GVE 控制矩阵 B=∂(oe_dot)/∂a(ECI)：x(n,6) oe → B(n,6,3)");
    m.def("gve_dyn_factor", &gve_dyn_factor, py::arg("x_lo"), py::arg("x_hi"),
          py::arg("dt"), py::arg("beta") = 1.0,
          "窗内 dyn 二因子（3N 并行）：x_lo/x_hi (nseg,n,6) → (res(nseg,n,6), Φ(nseg,n,6,6))");
    m.def("win_prior_asm", &win_prior_asm, py::arg("X"), py::arg("pri_fac"), py::arg("zp"),
          py::arg("pnode"), py::arg("off_pri"), py::arg("base_b"),
          py::arg("Aval_ptr"), py::arg("b_ptr"),
          "FLS 逐星边缘先验行：pri_fac(n,36)=chol(Λ_marg)^T、zp(n,6) → A_val/b 先验组");
    m.def("prior_fac_ss", &prior_fac_ss, py::arg("P"),
          "FLS: 逐星边缘协方差 P(n,6,6) → 白化因子 Lp(n,6,6)，LpᵀLp=P^{-1}");
    m.def("dbg_jac", [](py::array_t<double, py::array::c_style | py::array::forcecast> x, double beta) {
        auto buf = x.request();
        if ((std::size_t)buf.size % 6 != 0) throw std::runtime_error("dbg_jac: x 需 (n,6)");
        const int n = (int)(buf.size / 6);
        std::vector<double> Ao((std::size_t)n * 36), Fo((std::size_t)n * 6);
        { py::gil_scoped_release rel; qoejopt::dbgJac(static_cast<const double *>(buf.ptr), n, beta, Ao.data(), Fo.data()); }
        py::array_t<double> A({(py::ssize_t)n, (py::ssize_t)6, (py::ssize_t)6});
        py::array_t<double> F({(py::ssize_t)n, (py::ssize_t)6});
        std::memcpy(A.mutable_data(), Ao.data(), Ao.size() * sizeof(double));
        std::memcpy(F.mutable_data(), Fo.data(), Fo.size() * sizeof(double));
        return py::make_tuple(A, F);
    }, py::arg("x"), py::arg("beta") = 1.0, "调试：返回 A=∂f/∂oe (n,6,6) 与 Fk=∂f/∂κ (n,6)");
    m.def("get_timing", []() {
        auto v = qoejopt::jointGetTiming();
        py::list l; for (double x : v) l.append(x); return l;
    }, "细粒度计时 [expand(GVE), stmfold, assemble, res2]（秒）");
    m.def("reset_timing", &qoejopt::jointResetTiming, "清零 qoejopt 计时");
    m.def("free", &_free_ctx, "释放 device 缓冲");
    m.def("set_stm_gve", &set_stm_gve, py::arg("flag"),
          "joint d_A 折叠切换：1=同源全动力学变分 Φ_oe（stmFoldGveKernel），0=二体（旧路径）");
    m.def("set_stm_mode", &set_stm_mode, py::arg("mode"),
          "joint d_A 折叠：0=解析二体，1=同源全动力学变分，2=J2 解析（默认）");
    m.def("set_sub_steps", &set_sub_steps, py::arg("nsub"),
          "段级：每输出帧=一段、内部 nsub 个 dt/sub 子步（大步长段级递推）");
    m.def("stm_fold_gve", &stm_fold_gve, py::arg("x"), py::arg("nfr"), py::arg("dt"),
          py::arg("beta") = 1.0, "同源全动力学 STM 折叠：x(n,6) → d_A(nfr,n,18)=∂p_f/∂oe_0");
    m.def("stm_fold_j2", &stm_fold_j2, py::arg("x"), py::arg("nfr"), py::arg("dt"),
          "J2 解析 STM 折叠：x(n,6,QOE) → d_A(nfr,n,18)=Jp(oe_f)Φ_J2(t_f)");
    m.def("set_prior_cov", &set_prior_cov, py::arg("P0"), py::arg("zprior"), py::arg("sigma_a"),
          py::arg("cov_mode"),
          "整弧单发先验：每星 QOE 6×6 P0 + 先验均值 zprior → 信息先验（+ cov_mode=2 逐帧递推白化）");
    m.def("win_prior_prop", &win_prior_prop, py::arg("Pp"), py::arg("Q"), py::arg("stride") = 0,
          "K 帧单发窗口：P⁻ 跨窗递推 Pn = Φ_s Pp Φ_sᵀ + Q（(n,36)×2 → (n,6,6)）");
    m.def("obs_spectrum", &obs_spectrum, py::arg("k") = 32,
          "可观测性：装配 H=AᵀA（含 setup 配置的 ISL+GTS 行）的 (eval 升序 ncol, evec ncol×k)（cuSOLVER syevd）");
    m.def("obs_spectrum_range", &obs_spectrum_range, py::arg("k") = 32,
          "可观测性：装配 H=AᵀA（含 ISL+GTS 行）的最小 k 个特征对（cuSOLVER syevdx range I）");
    m.def("obs_apply_h", &obs_apply_h, py::arg("x"), py::arg("mode") = 7,
          "信息矩阵矩阵-free 作用 y=Hx；mode 位：1=ISL, 2=GTS, 4=prior");
    m.def("obs_set_scale", &obs_set_scale, py::arg("scale"),
          "设置状态列尺度；H 观测行按 S H S，先验行按 S^{-1} P^{-1} S^{-1}");
    m.def("obs_perstar", &obs_perstar, "逐星有效 Jacobian Gram 的 6 特征（奇异值²，升序）(n,6)");
    m.def("copy_rv", &copy_rv, "把当前 d_P（nfr,n,6）拷回 numpy（不重展开）");
    m.def("copy_A", &copy_A, "把当前 d_A（nfr,n,3,6）拷回 numpy（不重展开）");

    // 单发窗口（win_assemble_ss）：单发 L 帧 + 每星 6×6 先验 + 协方差白化
    m.def("setup_win_ss", &setup_win_ss, py::arg("n"), py::arg("L"), py::arg("E"), py::arg("G"),
          py::arg("ncol"), py::arg("nnz"), py::arg("nrows"),
          py::arg("off_isl"), py::arg("off_gts"), py::arg("off_damp"),
          py::arg("dt"), py::arg("s_isl"), py::arg("s_gts"), py::arg("sigma_a"),
          py::arg("isl_i"), py::arg("isl_jj"), py::arg("isl_D"), py::arg("isl_W"),
          py::arg("gts_a"), py::arg("gts_D"), py::arg("gts_W"), py::arg("gts_Q"),
          "单发窗口：一次注册结构/窗口0观测到 device");
    m.def("set_obs_ss", &set_obs_ss, py::arg("isl_D"), py::arg("isl_W"),
          py::arg("gts_D"), py::arg("gts_W"), py::arg("gts_Q"),
          "单发窗口：更新当前窗观测（结构不变）");
    m.def("set_prior_ss", &set_prior_ss, py::arg("prior_fac"), py::arg("zprior"), py::arg("P0"),
          "单发窗口：更新每星 6×6 先验（白化 P0 + 信息 A_prior/zprior）");
    m.def("set_beta_ss", &set_beta_ss, py::arg("beta"),
          "单发窗口：设置逐星阻力倍率 beta(n,)；缺省/不调用时 beta=1");
    m.def("win_assemble_ss", &win_assemble_ss, py::arg("X"), py::arg("rv0"), py::arg("A0"),
          py::arg("A_val_ptr"), py::arg("b_ptr"),
          "单发窗口装配：X(n,6) QOE、rv0(n,6)、A0(n,36)；单发 L 帧+STM折叠+白化装配；返回 res2");
    m.def("win_expand_ss", &win_expand_ss, py::arg("X"), py::arg("rv0"), py::arg("A0"),
          "单发窗口展开+STM折叠+res2（线搜索；刷新 d_P/d_A/d_X）");
    m.def("win_cached_ss", &win_cached_ss, py::arg("A_val_ptr"), py::arg("b_ptr"),
          "用当前 d_P/d_A/d_X 装配 A_val/b（不重展开）");
    m.def("win_rv_ss", &win_rv_ss, "上次装配的每帧 rv (nfr,n,6) → numpy（免 Python 重展开）");
    m.def("set_cov0_ss", &set_cov0_ss, py::arg("P0rv"), "iEKF：设置 rv 空间边界 P0(n,36)");
    m.def("set_adj_ss", &set_adj_ss, py::arg("adj_j"), py::arg("adj_w"),
          "iEKF：设置每 (帧,星) 的 ISL 邻接表 (nfr*n,D)");
    m.def("win_cov_rv_ss", &win_cov_rv_ss, py::arg("mode"),
          "iEKF 逐帧协方差滤波：mode=1 串行（迭代 1）、mode=2 Jacobi（迭代 ≥2）");
    m.def("win_asm_cov_ss", &win_asm_cov_ss, py::arg("A_val_ptr"), py::arg("b_ptr"),
          "装配 A_val/b + res2（口径 A：ISL/GTS 常数权重 + 每星 6×6 先验）");
    m.def("win_Pp_ss", &win_Pp_ss, "iEKF 每帧后验 P_k^+（rv 空间，(nfr,n,6,6)）");
    m.def("win_cov_ss", &win_cov_ss, py::arg("X"),
          "单发 L 步每帧 QOE 协方差与 Φ：(L,n,6,6)×2（用当前 P0）");
    m.def("free_win_ss", &_free_ctx, "释放 device 缓冲");

    // L 节点批量窗口（每帧一节点，ncol=6NL）
    m.def("setup_win", &setup_win, py::arg("n"), py::arg("L"), py::arg("E"), py::arg("G"),
          py::arg("ncol"), py::arg("nnz"), py::arg("nrows"),
          py::arg("off_isl"), py::arg("off_gts"), py::arg("off_dyn"), py::arg("off_damp"),
          py::arg("dt"), py::arg("s_isl"), py::arg("s_gts"), py::arg("issq"), py::arg("prior") = 0,
          py::arg("q_perstar") = 0,
          py::arg("isl_i"), py::arg("isl_jj"), py::arg("isl_D"), py::arg("isl_W"),
          py::arg("gts_a"), py::arg("gts_D"), py::arg("gts_W"), py::arg("gts_Q"),
          py::arg("ctr"), py::arg("qingt"), "L 节点批量窗口：一次注册结构/观测到 device（prior=1 时末组为 6×6 先验行）");
    m.def("set_win_beta", &set_win_beta, py::arg("beta"),
          "L 节点批量窗口：设置逐星阻力倍率 beta(n,)；缺省/不调用时 beta=1");
    m.def("win_consensus_asm", &win_consensus_asm, py::arg("zcons"), py::arg("u"),
          py::arg("Lp"), py::arg("zp"), py::arg("rho"), py::arg("prior_mode"),
          py::arg("A_val_ptr"), py::arg("b_ptr"),
          "全帧 consensus + 锚帧 FLS 先验行：写 A_val/b（需先 win_assemble 拷贝 X）");
    m.def("gts_global_update", &gts_global_update,
          py::arg("Z"), py::arg("D"), py::arg("M"), py::arg("Q"), py::arg("gs"),
          "全局 GTS 60 s 样本修正：每 6 帧一次，所有 GTS 全部进入");
    m.def("win_batch_assemble", &win_batch_assemble,
          py::arg("X"), py::arg("rv"), py::arg("Ap"),
          py::arg("isl_i"), py::arg("isl_jj"), py::arg("isl_D"), py::arg("isl_W"),
          py::arg("gts_a"), py::arg("gts_D"), py::arg("gts_W"), py::arg("gts_Q"),
          py::arg("beta"), py::arg("qingt"), py::arg("z"), py::arg("u"),
          py::arg("Lp"), py::arg("zp"),
          py::arg("n"), py::arg("L"), py::arg("E"), py::arg("G"),
          py::arg("nnz"), py::arg("nrows"),
          py::arg("off_isl"), py::arg("off_gts"), py::arg("off_dyn"), py::arg("off_damp"),
          py::arg("dt"), py::arg("s_isl"), py::arg("s_gts"), py::arg("rho"), py::arg("prior_mode"),
          py::arg("prior_node"), py::arg("A_device_ptr") = 0, py::arg("b_device_ptr") = 0,
          "144 块同图案批量装配（C++/CUDA）：返回 (B,nnz) A 与 (B,nrows) b");
    m.def("win_assemble", &win_assemble, py::arg("X"), py::arg("rv"), py::arg("Ap"),
          py::arg("A_val_ptr"), py::arg("b_ptr"),
          "L 节点批量窗口装配：X(Ln,6) QOE、rv(Ln,6)、Ap(Ln,18)；dyn 单步 φ/Φ + ISL/GTS/damp；返回 ISL res2");
    m.def("win_timing", []() {
        auto v = qoejopt::winGetTiming();
        py::list l; for (double x : v) l.append(x); return l;
    }, "L 节点窗口计时 [dyn, assemble, res2]（秒）");
    m.def("win_reset_timing", &qoejopt::winResetTiming, "清零窗口计时");
    m.def("win_free", &_free_wctx, "释放窗口 device 缓冲");

    // 子块内环（144 块锚帧 ΔQOE）device 常驻装配
    m.def("blk_setup", &blk_setup, py::arg("B"), py::arg("K"), py::arg("n4"), py::arg("gmax"),
          py::arg("G"), py::arg("s_isl"), py::arg("s_gts"), py::arg("s_eph"), py::arg("s_eph4"),
          py::arg("prior_mode"), py::arg("use_eph"), py::arg("sigma_prior"),
          py::arg("ea"), py::arg("isl_D"), py::arg("isl_M"), py::arg("lo"), py::arg("gi"),
          py::arg("gts_D"), py::arg("gts_M"), py::arg("gts_Q"),
          "子块内环结构/观测：ea(B,16,2) 局部节点、isl_D/M(B,16,K)、lo/gi(B,gmax)、gts_D/M(K,G)、gts_Q(K,G,3)");
    m.def("blk_set_window", &blk_set_window, py::arg("base"), py::arg("dq"), py::arg("Uti"),
          py::arg("pri_sig"), py::arg("r2P"), py::arg("truth_a"), py::arg("truth_4"),
          "子块内环每窗数据：base/dq(B,5,6)、Uti(B,6,6) 或 pri_sig(6,)、r2P(K,B*8,3)、Eph 真值(B,4,3)（None=部署）");
    m.def("blk_expand", &blk_expand, py::arg("y"), py::arg("A24_ptr"), py::arg("Apar_ptr"),
          py::arg("r_ptr"),
          "子块内环一轮：y(B,4,6) → pack+oe2rv+后向链+前向T/4+装配；A24/Apar/r 为 torch device 指针");
    m.def("blk_copy_rv", &blk_copy_rv, "拷出后向链 rv (K,B*5,6) 与前向 T/4 末帧 rv (B*5,6)");
    m.def("blk_timing", []() {
        auto v = qoejopt::blkGetTiming();
        py::list l; for (double x : v) l.append(x); return l;
    }, "子块内环计时 [pack, cartBwd, foldBwd, cartFwd, foldFwd, asm]（秒）");
    m.def("blk_reset_timing", &qoejopt::blkResetTiming, "清零子块内环计时");
    m.def("blk_free", &blk_free, "释放子块内环 device 缓冲");
}
