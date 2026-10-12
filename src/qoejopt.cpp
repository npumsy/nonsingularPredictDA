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
#include <cstdlib>
#include <cmath>
#include <stdexcept>
#include <string>
#include <array>
#include <vector>
#include "qoejopt.h"
#include "kepler.h"

namespace py = pybind11;
using qoejopt::Ctx;

static Ctx g_ctx;

static void _free_ctx() {
    Ctx &c = g_ctx;
    cudaFree(c.d_isl_i); cudaFree(c.d_isl_jj); cudaFree(c.d_gts_a);
    cudaFree(c.d_isl_D); cudaFree(c.d_isl_W);
    cudaFree(c.d_gts_D); cudaFree(c.d_gts_W); cudaFree(c.d_gts_Q);
    cudaFree(c.d_ctr);
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
    cudaMalloc(&c.d_cov, (std::size_t)L * n * 36 * sizeof(double));
    cudaMalloc(&c.d_phi, (std::size_t)L * n * 36 * sizeof(double));
    cudaMalloc(&c.d_P0rv, (std::size_t)n * 36 * sizeof(double));
    cudaMalloc(&c.d_Pm, (std::size_t)L * n * 36 * sizeof(double));
    cudaMalloc(&c.d_Pp, (std::size_t)L * n * 36 * sizeof(double));
    cudaMalloc(&c.d_PpPrev, (std::size_t)L * n * 36 * sizeof(double));
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
    if (!c.d_Pm) throw std::runtime_error("qoejopt.win_cov_rv_ss: 未先 setup_win_ss");
    qoejopt::ssFrameCovDevice(c, mode);
    c.cov_mode = 0;                 // 口径 A：ISL/GTS 用常数权重 1/σ（P_k 不进白化）
}

static py::array_t<double> win_Pp_ss() {
    Ctx &c = g_ctx;
    if (!c.d_Pp) throw std::runtime_error("qoejopt.win_Pp_ss: 未先 setup_win_ss");
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
    if (!c.d_X || !c.d_cov) throw std::runtime_error("qoejopt.win_cov_ss: 未先 setup_win_ss");
    auto bx = X.request();
    if ((std::size_t)bx.size != (std::size_t)c.n * 6)
        throw std::runtime_error("qoejopt.win_cov_ss: X 需 (n,6)");
    const std::size_t sz = (std::size_t)c.nfr * c.n * 36;
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
    cudaFree(c.d_ctr); cudaFree(c.d_qingt);
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

static void set_stm_gve(int flag) { g_ctx.stm_gve = flag ? 1 : 0; }
static void set_sub_steps(int nsub) { g_ctx.sub_steps = nsub > 0 ? nsub : 1; }

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
    std::memcpy(VEC.mutable_data(), evec.data(), evec.size() * sizeof(double));
    return py::make_tuple(EV, VEC);
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
    m.def("gve_stm", &gve_stm, py::arg("x"), py::arg("dt"), py::arg("beta") = 1.0,
          "单步 GVE + 全动力学变分 STM：x(n,6) oe → (oef(n,6), Φ_oe(n,6,6))");
    m.def("gve_stm_ia", &gve_stm_ia, py::arg("x"), py::arg("dt"), py::arg("beta") = 1.0,
          "一阶同源 STM：Φ≈I+A·dt（A=∂f/∂oe 解析）→ (oef(n,6), Φ(n,6,6))");
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
          "joint d_A 折叠切换：1=同源全动力学变分 Φ_oe（stmFoldGveKernel），0=二体");
    m.def("set_sub_steps", &set_sub_steps, py::arg("nsub"),
          "段级：每输出帧=一段、内部 nsub 个 dt/sub 子步（大步长段级递推）");
    m.def("stm_fold_gve", &stm_fold_gve, py::arg("x"), py::arg("nfr"), py::arg("dt"),
          py::arg("beta") = 1.0, "同源全动力学 STM 折叠：x(n,6) → d_A(nfr,n,18)=∂p_f/∂oe_0");
    m.def("set_prior_cov", &set_prior_cov, py::arg("P0"), py::arg("zprior"), py::arg("sigma_a"),
          py::arg("cov_mode"),
          "整弧单发先验：每星 QOE 6×6 P0 + 先验均值 zprior → 信息先验（+ cov_mode=2 逐帧递推白化）");
    m.def("win_prior_prop", &win_prior_prop, py::arg("Pp"), py::arg("Q"), py::arg("stride") = 0,
          "K 帧单发窗口：P⁻ 跨窗递推 Pn = Φ_s Pp Φ_sᵀ + Q（(n,36)×2 → (n,6,6)）");
    m.def("obs_spectrum", &obs_spectrum, py::arg("k") = 32,
          "可观测性：装配 H=AᵀA 的 (eval 升序 ncol, evec ncol×k)（cuSOLVER syevd）");
    m.def("obs_perstar", &obs_perstar, "逐星有效 Jacobian Gram 的 6 特征（奇异值²，升序）(n,6)");
    m.def("copy_rv", &copy_rv, "把当前 d_P（nfr,n,6）拷回 numpy（不重展开）");

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
    m.def("win_assemble", &win_assemble, py::arg("X"), py::arg("rv"), py::arg("Ap"),
          py::arg("A_val_ptr"), py::arg("b_ptr"),
          "L 节点批量窗口装配：X(Ln,6) QOE、rv(Ln,6)、Ap(Ln,18)；dyn 单步 φ/Φ + ISL/GTS/damp；返回 ISL res2");
    m.def("win_timing", []() {
        auto v = qoejopt::winGetTiming();
        py::list l; for (double x : v) l.append(x); return l;
    }, "L 节点窗口计时 [dyn, assemble, res2]（秒）");
    m.def("win_reset_timing", &qoejopt::winResetTiming, "清零窗口计时");
    m.def("win_free", &_free_wctx, "释放窗口 device 缓冲");
}
