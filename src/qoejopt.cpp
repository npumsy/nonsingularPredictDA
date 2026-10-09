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
#include <stdexcept>
#include <string>
#include "qoejopt.h"

namespace py = pybind11;
using qoejopt::Ctx;

static Ctx g_ctx;

static void _free_ctx() {
    Ctx &c = g_ctx;
    cudaFree(c.d_isl_i); cudaFree(c.d_isl_jj); cudaFree(c.d_gts_a);
    cudaFree(c.d_isl_D); cudaFree(c.d_isl_W);
    cudaFree(c.d_gts_D); cudaFree(c.d_gts_W); cudaFree(c.d_gts_Q);
    cudaFree(c.d_ctr);
    cudaFree(c.d_X); cudaFree(c.d_rv0); cudaFree(c.d_A0);
    cudaFree(c.d_P); cudaFree(c.d_A); cudaFree(c.d_res);
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
    m.def("gve_stm", &gve_stm, py::arg("x"), py::arg("dt"), py::arg("beta") = 1.0,
          "单步 GVE + 全动力学变分 STM：x(n,6) oe → (oef(n,6), Φ_oe(n,6,6))");
    m.def("get_timing", []() {
        auto v = qoejopt::jointGetTiming();
        py::list l; for (double x : v) l.append(x); return l;
    }, "细粒度计时 [expand(GVE), stmfold, assemble, res2]（秒）");
    m.def("reset_timing", &qoejopt::jointResetTiming, "清零 qoejopt 计时");
    m.def("free", &_free_ctx, "释放 device 缓冲");
}
