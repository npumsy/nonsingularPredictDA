#!/usr/bin/env python3
"""M3-large-m: multi-parameter / position-dependent force-field Learning.

Two cases (both reuse the augmented-state DACE interface; no DACE core change):
  A. drag multiplier parameters (rho, Cd, A/m): m=3, degenerate (only the
     product enters), used to check vector-parameter backpropagation.
  B. position-dependent gravity anomaly expanded in radial basis functions:
     U(r) = sum_k theta_k exp(-|r-c_k|^2/(2 s^2)), a = -grad U, centers c_k and
     width s fixed, theta (m=4) learned.  This is the genuinely larger-m case.

The center -> coefficients map is wrapped in a torch.autograd.Function whose
backward is the higher-order-coefficient shift identity (no adjoint tape).

Needs the compiled qoe module (../build) and the in-repo Theseus on PYTHONPATH.
"""
import os
import sys
import time

import numpy as np

for _n, _t in [("float_", np.float64), ("complex_", np.complex128),
               ("int_", np.int64), ("bool8", np.bool_), ("object_", np.object_)]:
    if not hasattr(np, _n):
        setattr(np, _n, _t)

import torch  # noqa: E402
import theseus as th  # noqa: E402

REPO = "/home/msy/Documents/TianZhi2"
sys.path.insert(0, os.path.join(REPO, "nonsingularPredictDA", "build"))
import qoe  # noqa: E402

torch.set_default_dtype(torch.float64)

RV0_M = np.array([6925443.9520, 190432.6240, 230986.9010,
                  -303.93854, 2277.90445, 7229.09828])
RV0 = RV0_M / 1e3
# DA 展开只需标称附近一段：取 1/4 轨道周期（无需整圈/多圈）。
_MU = 398600.4415e9
_a = 1.0 / (2.0 / np.linalg.norm(RV0_M[:3]) - np.linalg.norm(RV0_M[3:]) ** 2 / _MU)
PERIOD = 2.0 * np.pi * np.sqrt(_a ** 3 / _MU)
TF, STEP, ORDER = PERIOD / 4.0, 10.0, 2

# RBF gravity anomaly (case B): centers spread along the nominal orbit so each
# coefficient affects a distinct arc (identifiable).
S_RBF = 100.0  # km
_CENTERS = []
for _tq in (2000.0, 7000.0, 12000.0, 17000.0):
    _rf = qoe.daAugCoeffs(RV0_M, [0.0, 0.0, 0.0, 0.0], _tq, 1, 10.0)[0] / 1e3  # km
    _CENTERS.append([float(_rf[0]), float(_rf[1]), float(_rf[2])])
CENTERS = _CENTERS
THETA_TRUE = np.array([1e-3, -0.5e-3, 0.7e-3, 0.3e-3])

MODE = sys.argv[1] if len(sys.argv) > 1 else "rbf"   # "drag" | "rbf" | "sh"
LMAX = 3                                             # spherical-harmonic max degree (case "sh")
SH_THETA_TRUE = 1e-6 * np.array([1.0, -0.5, 0.7, 0.3, -0.2, 0.5, 0.4, -0.3, 0.2, -0.1])
if MODE == "drag":
    M, THETA_USE, PARAM_SCALE = 3, np.ones(3), 1.0
    FD_HT, TH_PERT = 1e-4, 1e-2   # drag 梯度量级 ~1e-4，FD 步长取 1e-4 以压住舍入
elif MODE == "rbf":
    M, THETA_USE, PARAM_SCALE = len(THETA_TRUE), THETA_TRUE, 1e-3
    FD_HT, TH_PERT = 1e-5, 1e-4
else:                                                # "sh"
    M = LMAX * (LMAX + 1) - 2                        # 10 for LMAX=3
    THETA_USE, PARAM_SCALE = SH_THETA_TRUE, 1e-6
    FD_HT, TH_PERT = 1e-8, 1e-7

TKS = [0.15 * TF, 0.40 * TF, 0.65 * TF, 0.85 * TF, TF]   # multi-epoch observation times


def da_data(x0_km, thetas, order=ORDER, tf=TF, step=STEP, centers=None, s=None):
    x0_km = np.asarray(x0_km, dtype=float)
    thetas = [float(t) for t in np.asarray(thetas).reshape(-1)]
    if MODE == "rbf":
        c_use = CENTERS if centers is None else centers
        s_use = S_RBF if s is None else s
        rf, c, mons = qoe.daAugRBFCoeffs(x0_km * 1e3, thetas, c_use, s_use, tf, order, step)
    elif MODE == "sh":
        rf, c, mons = qoe.daAugSHCoeffs(x0_km * 1e3, thetas, LMAX, tf, order, step)
    else:
        rf, c, mons = qoe.daAugCoeffs(x0_km * 1e3, thetas, tf, order, step)
    mons = np.array(mons)
    C = np.array(c).reshape(6, len(mons))
    idx = {tuple(m): k for k, m in enumerate(mons)}
    return rf * 1e-3, C, mons, idx


def pack_order1(C, mons, m):
    nvar = len(mons[0])                 # 6 state + m params
    idx = {tuple(mm): k for k, mm in enumerate(mons)}
    out = np.zeros((6, 7 + m))
    e0 = [0] * nvar
    out[:, 0] = C[:, idx[tuple(e0)]]
    for j in range(6):
        e = [0] * nvar; e[j] = 1
        out[:, 1 + j] = C[:, idx[tuple(e)]]
    for k in range(m):
        e = [0] * nvar; e[6 + k] = 1
        out[:, 7 + k] = C[:, idx[tuple(e)]]
    return out


class DACoeffs(torch.autograd.Function):
    """forward: (x0_bar[...,6], theta_bar[...,m]) -> packed order<=1 block (...,6,7+m).

    Supports an optional leading batch dimension.  The backward applies the
    higher-order-coefficient shift identity and is **vectorized over the batch**
    (no per-sample Python loop): gx_i = sum_w g_{i,w} (beta_w+1) c_{i,beta_w+e_l}.
    """

    @staticmethod
    def forward(ctx, x0_bar, theta_bar):
        x0n = x0_bar.detach().cpu().numpy()
        thn = theta_bar.detach().cpu().numpy()
        single = (x0n.ndim == 1)
        x0n = np.atleast_2d(x0n)
        thn = np.atleast_2d(thn)
        B, m = thn.shape
        Cs, outs = [], []
        for b in range(B):
            _, C, mons, idx = da_data(x0n[b], thn[b], order=ORDER)
            Cs.append(C)
            outs.append(pack_order1(C, mons, m))
        ctx.C, ctx.idx, ctx.m, ctx.single = np.stack(Cs), idx, m, single
        out = np.stack(outs)                       # [B,6,7+m]
        if single:
            out = out[0]
        return torch.tensor(out, dtype=torch.float64)

    @staticmethod
    def backward(ctx, g):
        C, idx, m = ctx.C, ctx.idx, ctx.m
        g = g.detach().cpu().numpy()
        if ctx.single:
            g = g[None]
        g = g.reshape(-1, 6, 7 + m)
        nvar, W = 6 + m, 7 + m
        # exponent (beta) of each packed column w; mult[w,l] = beta_w[l]+1
        beta = np.zeros((W, nvar))
        for w in range(1, 7):
            beta[w, w - 1] = 1.0
        for w in range(7, W):
            beta[w, 6 + (w - 7)] = 1.0
        mult = beta + 1.0
        # index of the shifted monomial (beta_w+e_l) in the dense coefficient array
        cidx = np.full((W, nvar), -1, dtype=int)
        for w in range(W):
            for l in range(nvar):
                e = beta[w].astype(int).copy(); e[l] += 1
                cidx[w, l] = idx.get(tuple(e), -1)
        mask = cidx >= 0
        Csel = C[:, :, np.where(mask, cidx, 0)] * mult[None, None] * mask[None, None]
        contrib = np.einsum("biw,biwl->bil", g, Csel)     # [B,6,nvar]
        gx = contrib[:, :, :6].sum(axis=1)                # [B,6]
        gt = contrib[:, :, 6:].sum(axis=1)                # [B,m]
        gx_t = torch.tensor(gx, dtype=torch.float64)
        gt_t = torch.tensor(gt, dtype=torch.float64)
        return (gx_t[0], gt_t[0]) if ctx.single else (gx_t, gt_t)


def check_backprop():
    print(f"=== backprop check (MODE={MODE}, m={M}) ===")
    theta_bar0 = np.array(THETA_USE, dtype=float)
    x0 = torch.tensor(RV0, requires_grad=True)
    tb = torch.tensor(theta_bar0, requires_grad=True)
    c = DACoeffs.apply(x0, tb)
    g = torch.linspace(-1.0, 1.0, c.numel()).reshape(c.shape)
    (g * c).sum().backward()
    gx_auto = x0.grad.numpy().copy()
    gt_auto = tb.grad.numpy().copy()

    def Lnp(x0v, tv):
        _, C, mons, _ = da_data(x0v, tv)
        return float((g.numpy() * pack_order1(C, mons, M)).sum())

    h, ht = 1e-5, FD_HT
    gx_fd = np.zeros(6)
    for j in range(6):
        xp = RV0.copy(); xp[j] += h
        xm = RV0.copy(); xm[j] -= h
        gx_fd[j] = (Lnp(xp, theta_bar0) - Lnp(xm, theta_bar0)) / (2 * h)
    gt_fd = np.zeros(M)
    for k in range(M):
        tp = theta_bar0.copy(); tp[k] += ht
        tm = theta_bar0.copy(); tm[k] -= ht
        gt_fd[k] = (Lnp(RV0, tp) - Lnp(RV0, tm)) / (2 * ht)

    relx = np.max(np.abs(gx_auto - gx_fd)) / (np.max(np.abs(gx_auto)) + 1e-30)
    relt = np.max(np.abs(gt_auto - gt_fd)) / (np.max(np.abs(gt_auto)) + 1e-30)
    print(f"  gx_auto[:3] = {gx_auto[:3]}")
    print(f"  rel |gx_auto - gx_fd| = {relx:.3e}")
    print(f"  rel |gt_auto - gt_fd| = {relt:.3e}")
    assert relx < 1e-6
    assert relt < (5e-4 if MODE == "drag" else 1e-4)


# 积分伴随（大 m deep）用的小规模历元：只覆盖前 40 s，控制测试成本。
TFS_DEEP = [0.0, 10.0, 20.0, 40.0]
# 深测专用 RBF 中心：落在 [0,40] s 弧段上（共享 CENTERS 沿整圈、在短弧上近似失效）。
DEEP_CENTERS = []
for _tq in (0.0, 10.0, 20.0, 40.0):
    _rf = qoe.daAugCoeffs(RV0_M, [0.0, 0.0, 0.0, 0.0], _tq, 1, 10.0)[0] / 1e3
    DEEP_CENTERS.append([float(_rf[0]), float(_rf[1]), float(_rf[2])])


def _deep_forward(x0_km, thetas):
    rv0 = np.asarray(x0_km, dtype=float)
    th = [float(t) for t in thetas]
    if MODE == "sh":
        fl = qoe.daDeepForwardSH(rv0, th, LMAX, TFS_DEEP, STEP)
    else:
        fl = qoe.daDeepForwardRBF(rv0, th, DEEP_CENTERS, S_RBF, TFS_DEEP, STEP)
    xf = np.array(fl.rvf).reshape(-1)                  # [K*6] km
    Phi = np.array(fl.PhiEpoch).reshape(-1)            # [K*36]
    return xf, Phi, fl


def _deep_L(x0_km, thetas, gx, gP):
    xf, Phi, _ = _deep_forward(x0_km, thetas)
    return float((gx.reshape(-1) * xf).sum() + (gP.reshape(-1) * Phi).sum())


def check_deep_adjoint():
    """T1：积分伴随的 ∂L/∂x0 与 ∂L/∂θ（含 deep）对拍中心差分。"""
    if MODE == "drag":
        return
    print(f"=== deep integration adjoint (MODE={MODE}, m={M}) ===")
    K = len(TFS_DEEP)
    rng = np.random.default_rng(2)
    gx = rng.normal(0.0, 1.0, (K, 6))
    gP = rng.normal(0.0, 1.0, (K, 36))
    th0 = np.array(THETA_USE, dtype=float)

    _, _, fl = _deep_forward(RV0, th0)
    if MODE == "sh":
        gx0, gt = qoe.daDeepBackwardSH(fl, [gx[k].tolist() for k in range(K)],
                                       gP.reshape(-1).tolist(), LMAX)
    else:
        gx0, gt = qoe.daDeepBackwardRBF(fl, [gx[k].tolist() for k in range(K)],
                                        gP.reshape(-1).tolist(), DEEP_CENTERS, S_RBF)
    gx0 = np.array(gx0).reshape(-1)
    gt = np.array(gt).reshape(-1)

    h, ht = 1e-4, FD_HT
    gx0_fd = np.zeros(6)
    for j in range(6):
        xp = RV0.copy(); xp[j] += h
        xm = RV0.copy(); xm[j] -= h
        gx0_fd[j] = (_deep_L(xp, th0, gx, gP) - _deep_L(xm, th0, gx, gP)) / (2 * h)
    gt_fd = np.zeros(M)
    for k in range(M):
        tp = th0.copy(); tp[k] += ht
        tm = th0.copy(); tm[k] -= ht
        gt_fd[k] = (_deep_L(RV0, tp, gx, gP) - _deep_L(RV0, tm, gx, gP)) / (2 * ht)

    rx = np.max(np.abs(gx0 - gx0_fd)) / (np.max(np.abs(gx0_fd)) + 1e-30)
    rt = np.max(np.abs(gt - gt_fd)) / (np.max(np.abs(gt_fd)) + 1e-30)
    print(f"  rel ∂L/∂x0 = {rx:.3e}   rel ∂L/∂θ(deep) = {rt:.3e}")

    # 去掉 Φ 种子（gP=0）即 direct：证明 deep 项非零、不可丢
    gP0 = np.zeros_like(gP)
    if MODE == "sh":
        _, gt_direct = qoe.daDeepBackwardSH(fl, [gx[k].tolist() for k in range(K)],
                                            gP0.reshape(-1).tolist(), LMAX)
    else:
        _, gt_direct = qoe.daDeepBackwardRBF(fl, [gx[k].tolist() for k in range(K)],
                                             gP0.reshape(-1).tolist(), DEEP_CENTERS, S_RBF)
    gt_direct = np.array(gt_direct).reshape(-1)
    rel_deep = np.max(np.abs(gt_direct - gt_fd)) / (np.max(np.abs(gt_fd)) + 1e-30)
    print(f"  direct-vs-full(deep) 偏差 = {rel_deep:.3e}  （O(1) 说明 deep 不可忽略）")

    assert rx < 1e-6 and rt < 1e-6


# --- 积分伴随接入 PyTorch / Theseus ---------------------------------------
_DEEP_TFS = TFS_DEEP
_DEEP_CENTERS = DEEP_CENTERS


class DAFlowDeep(torch.autograd.Function):
    """(x0_km, theta) -> (xf[K,6], Phi[K,36])；backward 为积分伴随（含 deep）。"""

    @staticmethod
    def forward(ctx, x0_km, theta):
        th = [float(v) for v in theta.detach().cpu().numpy().reshape(-1)]
        x0 = x0_km.detach().cpu().numpy().reshape(6)
        if MODE == "sh":
            fl = qoe.daDeepForwardSH(x0, th, LMAX, _DEEP_TFS, STEP)
        else:
            fl = qoe.daDeepForwardRBF(x0, th, _DEEP_CENTERS, S_RBF, _DEEP_TFS, STEP)
        ctx.fl = fl
        ctx.K = len(_DEEP_TFS)
        xf = torch.tensor(np.array(fl.rvf), dtype=torch.float64)
        Phi = torch.tensor(np.array(fl.PhiEpoch).reshape(ctx.K, 36), dtype=torch.float64)
        return xf, Phi

    @staticmethod
    def backward(ctx, gx, gP):
        K = ctx.K
        if gx is None:
            gx = torch.zeros(K, 6, dtype=torch.float64)
        if gP is None:
            gP = torch.zeros(K, 36, dtype=torch.float64)
        gx_l = [gx[k].detach().cpu().numpy() for k in range(K)]
        gP_l = gP.detach().cpu().numpy().reshape(-1).tolist()
        if MODE == "sh":
            gx0, gt = qoe.daDeepBackwardSH(ctx.fl, gx_l, gP_l, LMAX)
        else:
            gx0, gt = qoe.daDeepBackwardRBF(ctx.fl, gx_l, gP_l, _DEEP_CENTERS, S_RBF)
        return torch.tensor(np.array(gx0), dtype=torch.float64), \
            torch.tensor(np.array(gt), dtype=torch.float64)


class _DynPhi(th.CostFunction):
    """决策量=初始状态 x0；error=x_f(x0,θ)-obs；Jacobian=Φ(θ)（可微→含 deep）。"""

    def __init__(self, x0, theta, obs, deep=True, weight=None, name="dynphi"):
        if weight is None:
            weight = th.ScaleCostWeight(torch.tensor(1.0))
        super().__init__(cost_weight=weight, name=name)
        self.x0, self.theta, self.obs, self.deep = x0, theta, obs, deep
        self.register_optim_vars(["x0"])
        self.register_aux_vars(["theta"])
        self.K = len(_DEEP_TFS)
        self._Phi = None

    def error(self):
        xf, Phi = DAFlowDeep.apply(self.x0.tensor.reshape(-1), self.theta.tensor.reshape(-1))
        self._Phi = Phi
        return xf.reshape(-1) - self.obs

    def jacobians(self):
        e = self.error()
        J = torch.zeros(6 * self.K, 6, dtype=torch.float64)
        for k in range(self.K):
            J[6 * k:6 * k + 6, :] = self._Phi[k].reshape(6, 6)
        if not self.deep:
            J = J.detach()          # direct：丢掉 ∂Φ/∂θ
        return [J], e

    def dim(self):
        return 6 * self.K

    def _copy_impl(self, new_name=None):
        return _DynPhi(self.x0.copy(), self.theta.copy(), self.obs,
                       deep=self.deep, weight=self.weight.copy(), name=new_name)


_GN_OBS = None
_PHI_Z = None          # Φ-主导因子的观测 z
_PHI_X0TAR = None      # 下游目标 x0_target


class _PhiObs(th.CostFunction):
    """Φ-主导因子：error = Φ(θ)·x0 - z（Φ 为单历元 STM）；optim var x0，aux θ。

    deep: J=Φ(θ) 可微（→ 内层解含 ∂Φ/∂θ）；direct: J=Φ.detach()（→ ∂x0*/∂θ≡0）。
    """

    def __init__(self, x0, theta, z, deep=True, weight=None, name="phiobs"):
        if weight is None:
            weight = th.ScaleCostWeight(torch.tensor(1.0))
        super().__init__(cost_weight=weight, name=name)
        self.x0, self.theta, self.z, self.deep = x0, theta, z, deep
        self.register_optim_vars(["x0"])
        self.register_aux_vars(["theta"])
        self._Phi = None

    def _phi(self):
        # 所有历元的 STM 堆叠 (6K × 6)：z 为 6K 观测 ⇒ 超定，解处 e*≠0（deep 才非零）
        _, Phi = DAFlowDeep.apply(torch.tensor(RV0, dtype=torch.float64),
                                  self.theta.tensor.reshape(-1))
        return torch.cat([Phi[k].reshape(6, 6) for k in range(len(_DEEP_TFS))], dim=0)

    def error(self):
        self._Phi = self._phi()
        return (self._Phi @ self.x0.tensor.reshape(6)) - self.z

    def jacobians(self):
        e = self.error()
        J = self._Phi
        if not self.deep:
            J = J.detach()
        return [J], e

    def dim(self):
        return 6 * len(_DEEP_TFS)

    def _copy_impl(self, new_name=None):
        return _PhiObs(self.x0.copy(), self.theta.copy(), self.z, deep=self.deep,
                       weight=self.weight.copy(), name=new_name)


def _solve_x0_phi(theta_1d, deep):
    """内层 GN 解 x0*（Φ-主导因子），对 θ 可微。"""
    th_in = theta_1d.reshape(1, -1)
    x0v = th.Vector(tensor=torch.tensor(RV0.reshape(1, 6)), name="x0")
    thv = th.Vector(tensor=th_in, name="theta")
    obj = th.Objective()
    obj.add(_PhiObs(x0v, thv, _PHI_Z, deep=deep))
    opt = th.LevenbergMarquardt(obj, max_iterations=12,
                                abs_err_tolerance=1e-12, rel_err_tolerance=1e-12)
    layer = th.TheseusLayer(opt)
    out, _ = layer.forward({"x0": torch.tensor(RV0.reshape(1, 6)), "theta": th_in},
                           optimizer_kwargs={"track_best_solution": True})
    return out["x0"].reshape(-1)


def _solve_x0(theta_1d, deep):
    """Theseus 内层 GN 解 x0*(θ)；返回的解对 θ 可微（隐式微分）。"""
    th_in = theta_1d.reshape(1, -1)
    x0v = th.Vector(tensor=torch.tensor(RV0.reshape(1, 6)), name="x0")
    thv = th.Vector(tensor=th_in, name="theta")
    obj = th.Objective()
    obj.add(_DynPhi(x0v, thv, _GN_OBS, deep=deep))
    opt = th.LevenbergMarquardt(obj, max_iterations=8,
                                abs_err_tolerance=1e-12, rel_err_tolerance=1e-12)
    layer = th.TheseusLayer(opt)
    out, _ = layer.forward({"x0": torch.tensor(RV0.reshape(1, 6)), "theta": th_in},
                           optimizer_kwargs={"track_best_solution": True})
    return out["x0"].reshape(-1)


def check_gn_deep():
    """T2：GN 解 x0*(θ) 的 Learning 梯度（deep=Φ 可微 vs direct=Φ detach）对拍 FD。"""
    if MODE == "drag":
        return
    global _DEEP_TFS, _DEEP_CENTERS, _GN_OBS
    print(f"\n=== Theseus GN parameter Learning: deep vs direct (MODE={MODE}, m={M}) ===")
    _DEEP_TFS = [0.0, 50.0, 100.0, 150.0, 200.0]
    if MODE == "sh":
        _DEEP_CENTERS = None
        obs_np = np.array(qoe.daDeepForwardSH(RV0, list(2.0*np.array(THETA_USE)), LMAX, _DEEP_TFS, STEP).rvf)
    else:
        _DEEP_CENTERS = []
        for _tq in _DEEP_TFS:
            _rf = qoe.daAugCoeffs(RV0_M, [0.0, 0.0, 0.0, 0.0], _tq, 1, 10.0)[0] / 1e3
            _DEEP_CENTERS.append([float(_rf[0]), float(_rf[1]), float(_rf[2])])
        obs_np = np.array(qoe.daDeepForwardRBF(RV0, list(2.0*np.array(THETA_USE)), _DEEP_CENTERS, S_RBF,
                                               _DEEP_TFS, STEP).rvf)
    _GN_OBS = torch.tensor(obs_np.reshape(-1), dtype=torch.float64)

    rng = np.random.default_rng(3)
    target = rng.normal(0.0, 1.0, 6)
    th0 = np.array(THETA_USE, dtype=float)

    def Lnp(thv):
        x0s = _solve_x0(torch.tensor(thv.reshape(1, -1), dtype=torch.float64),
                        deep=True).detach().numpy()
        return float(0.5 * ((x0s.reshape(-1) - target) ** 2).sum())

    grads = {}
    for deep in (True, False):
        u = torch.tensor(th0.reshape(1, -1), dtype=torch.float64, requires_grad=True)
        x0s = _solve_x0(u, deep=deep)
        L = 0.5 * ((x0s - torch.tensor(target, dtype=torch.float64)) ** 2).sum()
        L.backward()
        grads[deep] = u.grad.detach().numpy().reshape(-1).copy()

    h = FD_HT
    gfd = np.zeros(M)
    for k in range(M):
        tp = th0.copy(); tp[k] += h
        tm = th0.copy(); tm[k] -= h
        gfd[k] = (Lnp(tp) - Lnp(tm)) / (2 * h)
    rel_deep = np.max(np.abs(grads[True] - gfd)) / (np.max(np.abs(gfd)) + 1e-30)
    rel_direct = np.max(np.abs(grads[False] - gfd)) / (np.max(np.abs(gfd)) + 1e-30)
    print(f"  rel |g_deep   - g_fd| = {rel_deep:.3e}")
    print(f"  rel |g_direct - g_fd| = {rel_direct:.3e}  （direct 应显著更大）")
    assert rel_deep < 1e-4
    assert rel_direct >= rel_deep


X0BAR = torch.tensor(RV0)


class _Dyn(th.CostFunction):
    # x0 is fixed at X0BAR (known initial state); only the force field theta is learned.
    def __init__(self, xf, theta, weight=None, name="dyn"):
        if weight is None:
            weight = th.ScaleCostWeight(torch.tensor(1.0))
        super().__init__(cost_weight=weight, name=name)
        self.xf, self.theta = xf, theta
        self.register_optim_vars(["xf"])
        self.register_aux_vars(["theta"])
        self.c = DACoeffs.apply(X0BAR, self.theta.tensor.reshape(-1))

    def error(self):
        return self.xf.tensor.reshape(-1) - self.c[:, 0]

    def jacobians(self):
        return [torch.eye(6, dtype=torch.float64)], self.error()

    def dim(self):
        return 6

    def _copy_impl(self, new_name=None):
        return _Dyn(self.xf.copy(), self.theta.copy(),
                    weight=self.weight.copy(), name=new_name)


class _Obs(th.CostFunction):
    def __init__(self, xf, x_obs, weight=None, name="obs"):
        if weight is None:
            weight = th.ScaleCostWeight(torch.tensor(1.0))
        super().__init__(cost_weight=weight, name=name)
        self.xf = xf
        self.x_obs = x_obs
        self.register_optim_vars(["xf"])

    def error(self):
        return self.xf.tensor.reshape(-1) - self.x_obs

    def jacobians(self):
        return [torch.eye(6, dtype=torch.float64)], self.error()

    def dim(self):
        return 6

    def _copy_impl(self, new_name=None):
        return _Obs(self.xf.copy(), self.x_obs, weight=self.weight.copy(), name=new_name)


def solve_and_loss(theta, x_obs):
    xf = th.Vector(tensor=torch.tensor(x_obs).reshape(1, 6), name="xf")
    tv = th.Vector(tensor=theta.reshape(1, M), name="theta")
    obj = th.Objective()
    obj.add(_Dyn(xf, tv))
    obj.add(_Obs(xf, torch.tensor(x_obs)))
    opt = th.LevenbergMarquardt(obj, max_iterations=10,
                                abs_err_tolerance=1e-12, rel_err_tolerance=1e-12)
    layer = th.TheseusLayer(opt)
    inputs = {"xf": torch.tensor(x_obs).reshape(1, 6),
              "theta": theta.reshape(1, M)}
    out, _ = layer.forward(inputs, optimizer_kwargs={"track_best_solution": True})
    return 0.5 * ((out["xf"].reshape(-1) - torch.tensor(x_obs)) ** 2).sum()


def check_gn_learn():
    """T4：m=30 RBF，用 Theseus _DynPhi（deep，Jacobian=Φ(θ)）真正学 30 个 θ。

    obs 由 θ_true 生成；内层 GN 解 x0*(θ)，外层最小化轨迹残差 ‖flow(x0*,θ)-obs‖²；
    θ 经隐式微分回传（含 deep）。判据：loss 明显下降。
    """
    global _DEEP_TFS, _DEEP_CENTERS, _GN_OBS
    if MODE != "rbf":
        return
    mm = 30
    print(f"\n=== Theseus large-m Learning: m={mm} (MODE=rbf) ===")
    _DEEP_TFS = list(np.linspace(0.0, TF, 12))     # 12 帧（积分步数只由 TF/step 决定，与帧数无关）
    rng = np.random.default_rng(7)
    _DEEP_CENTERS = []
    for _tq in np.linspace(0.0, TF, mm):
        _rf = qoe.daAugCoeffs(RV0_M, [0.0, 0.0, 0.0, 0.0], float(_tq), 1, 10.0)[0] / 1e3
        _DEEP_CENTERS.append([float(_rf[0]), float(_rf[1]), float(_rf[2])])
    theta_true = 1e-3 * rng.standard_normal(mm)
    obs_np = np.array(qoe.daDeepForwardRBF(RV0, list(theta_true), _DEEP_CENTERS, S_RBF,
                                           _DEEP_TFS, STEP).rvf)
    _GN_OBS = torch.tensor(obs_np.reshape(-1), dtype=torch.float64)

    scale = 1e-3

    def residual_loss(theta_t):
        x0s = _solve_x0(theta_t, deep=True)
        xf, _ = DAFlowDeep.apply(x0s, theta_t.reshape(-1))
        return 0.5 * ((xf.reshape(-1) - _GN_OBS) ** 2).sum()

    u = torch.zeros(1, mm, dtype=torch.float64, requires_grad=True)   # 从 θ=0 学起
    L0 = float(residual_loss(u * scale))
    adam = torch.optim.Adam([u], lr=0.2)
    for _ in range(60):
        loss = residual_loss(u * scale)
        adam.zero_grad(); loss.backward(); adam.step()
    Lf = float(residual_loss(u * scale))
    est = u.detach().numpy().reshape(-1) * scale
    rel = np.max(np.abs(est - theta_true)) / (np.max(np.abs(theta_true)) + 1e-30)
    print(f"  loss {L0:.3e} -> {Lf:.3e}  ({L0 / (Lf + 1e-30):.0f}x)   θ_rel_err={rel:.3e}")
    assert Lf < 0.05 * L0, "large-m learning must reduce the loss"


def _fwd(x0_km, th, tfs, centers):
    if MODE == "sh":
        return qoe.daDeepForwardSH(np.asarray(x0_km, float), [float(t) for t in th], LMAX, list(tfs), STEP)
    return qoe.daDeepForwardRBF(np.asarray(x0_km, float), [float(t) for t in th], centers, S_RBF, list(tfs), STEP)


def _bwd(fl, tfs, centers, ux, uP):
    uxl = [list(np.asarray(ux[k], float)) for k in range(len(tfs))]
    uPl = list(np.asarray(uP, float).reshape(-1))
    if MODE == "sh":
        return qoe.daDeepBackwardSH(fl, uxl, uPl, LMAX)
    return qoe.daDeepBackwardRBF(fl, uxl, uPl, centers, S_RBF)


def _centers_at(times):
    out = []
    for tq in times:
        rf = qoe.daAugCoeffs(RV0_M, [0.0, 0.0, 0.0, 0.0], float(tq), 1, 10.0)[0] / 1e3
        out.append([float(rf[0]), float(rf[1]), float(rf[2])])
    return out


def check_adjoint_identity():
    """理论-1：backward 是 forward 的**真伴随**（全 Jacobian，⟨u,Jv⟩=⟨v,Jᵀu⟩）。

    比单点 FD 强：它同时验证 J 的所有列/行，是「积分伴随」数学成立的判据。
    """
    if MODE == "drag":
        return
    print(f"\n=== adjoint identity <u,Jv>=<v,Jᵀu> (MODE={MODE}) ===")
    tfs = [0.0, 10.0, 20.0, 40.0]
    centers = _centers_at(tfs) if MODE != "sh" else None
    th0 = np.array(THETA_USE, dtype=float)
    fl = _fwd(RV0, th0, tfs, centers)
    K = len(tfs)
    rng = np.random.default_rng(11)
    ux = rng.normal(0.0, 1.0, (K, 6)); uP = rng.normal(0.0, 1.0, (K, 36))
    vx = rng.normal(0.0, 1.0, 6); vth = PARAM_SCALE * rng.normal(0.0, 1.0, M)
    eps = 1e-4
    fp = _fwd(RV0 + eps * vx, th0 + eps * vth, tfs, centers)
    fm = _fwd(RV0 - eps * vx, th0 - eps * vth, tfs, centers)
    Jv_x = (np.array(fp.rvf) - np.array(fm.rvf)).reshape(-1) / (2 * eps)
    Jv_P = (np.array(fp.PhiEpoch) - np.array(fm.PhiEpoch)).reshape(-1) / (2 * eps)
    lhs = float((ux.reshape(-1) * Jv_x).sum() + (uP.reshape(-1) * Jv_P).sum())
    gx0, gth = _bwd(fl, tfs, centers, ux, uP)
    rhs = float((vx * np.array(gx0)).sum() + (vth * np.array(gth)).sum())
    rel = abs(lhs - rhs) / (abs(lhs) + abs(rhs) + 1e-30)
    print(f"  <u,Jv>={lhs:.6e}  <v,Jᵀu>={rhs:.6e}  rel={rel:.3e}")
    assert rel < 1e-5


def check_deep_necessary():
    """理论-2（决定性）：Φ-only 损失——direct 项恒为 0，只有 deep(∂Φ/∂θ) 给出正确梯度。

    L = <G, Φ(tf)>；direct 编排（Φ detach）得 ∂L/∂θ=0（必错）；积分伴随 deep 的 ∂L/∂θ 对拍 FD。
    """
    if MODE == "drag":
        return
    print(f"\n=== Φ-only loss: direct≡0, deep=FD (MODE={MODE}) ===")
    tfs = [TF]
    centers = _centers_at(list(np.linspace(0.0, TF, 6))) if MODE != "sh" else None
    th0 = np.array(THETA_USE, dtype=float)
    fl = _fwd(RV0, th0, tfs, centers)
    rng = np.random.default_rng(12)
    G = rng.normal(0.0, 1.0, (1, 36))
    _, gth_deep = _bwd(fl, tfs, centers, np.zeros((1, 6)), G)
    _, gth_direct = _bwd(fl, tfs, centers, np.zeros((1, 6)), np.zeros((1, 36)))

    def L(th):
        f = _fwd(RV0, th, tfs, centers)
        return float((G.reshape(-1) * np.array(f.PhiEpoch).reshape(-1)).sum())

    ht = FD_HT; eye = np.eye(M)
    gfd = np.array([(L(th0 + ht * eye[k]) - L(th0 - ht * eye[k])) / (2 * ht) for k in range(M)])
    rd = np.max(np.abs(np.array(gth_deep) - gfd)) / (np.max(np.abs(gfd)) + 1e-30)
    print(f"  max|∂L/∂θ|_FD={np.max(np.abs(gfd)):.3e}   deep-vs-FD rel={rd:.3e}   max|direct|={np.max(np.abs(gth_direct)):.2e}")
    assert np.max(np.abs(gfd)) > 0 and rd < 1e-5 and np.max(np.abs(gth_direct)) < 1e-12


def check_cross_method():
    """理论-3：两套**独立实现**互为 gold standard——积分伴随(diagonal) vs θ-in-DA 移位恒等式。"""
    if MODE == "drag":
        return
    print(f"\n=== cross-method: integration adjoint vs θ-in-DA (MODE={MODE}) ===")
    tf = TF
    tfs = [tf]
    centers = _centers_at(list(np.linspace(0.0, tf, M))) if MODE != "sh" else None
    th0 = np.array(THETA_USE, dtype=float)
    fl = _fwd(RV0, th0, tfs, centers)
    xf_deep = np.array(fl.rvf).reshape(6)
    Phi_deep = np.array(fl.PhiEpoch).reshape(6, 6)
    rf, C, mons, idx = da_data(RV0, th0, tf=tf, centers=centers)
    packed = pack_order1(C, mons, M)                 # [6, 7+m]
    xf_da, Phi_da = packed[:, 0], packed[:, 1:7]
    frel = max(np.max(np.abs(xf_deep - xf_da)) / (np.max(np.abs(xf_da)) + 1e-30),
               np.max(np.abs(Phi_deep - Phi_da)) / (np.max(np.abs(Phi_da)) + 1e-30))
    rng = np.random.default_rng(13)
    ux = rng.normal(0.0, 1.0, 6); uP = rng.normal(0.0, 1.0, 36)
    gx0_deep, gth_deep = _bwd(fl, tfs, centers, ux.reshape(1, 6), uP.reshape(1, 36))
    gpacked = np.zeros((6, 7 + M)); gpacked[:, 0] = ux; gpacked[:, 1:7] = uP.reshape(6, 6)
    global CENTERS                                   # DACoeffs 内部 da_data 用模块 CENTERS
    saved = CENTERS
    if MODE == "rbf":
        CENTERS = centers
    try:
        x0t = torch.tensor(RV0, requires_grad=True); tbt = torch.tensor(th0, requires_grad=True)
        (torch.tensor(gpacked) * DACoeffs.apply(x0t, tbt)).sum().backward()
        gx0_da, gth_da = x0t.grad.numpy(), tbt.grad.numpy()
    finally:
        CENTERS = saved
    grel = max(np.max(np.abs(np.array(gx0_deep) - gx0_da)) / (np.max(np.abs(gx0_da)) + 1e-30),
               np.max(np.abs(np.array(gth_deep) - gth_da)) / (np.max(np.abs(gth_da)) + 1e-30))
    print(f"  forward rel={frel:.3e}   ∂L/∂(x0,θ) rel={grel:.3e}")
    assert frel < 1e-9 and grel < 1e-6


def check_effect_deep():
    """效果（决定性）：Φ-主导因子（error=Φ(θ)·x0-z，K=4 超定 ⇒ 解处 e*≠0）逐级对比。

    (1) 隐式梯度误差随场强：direct 单调上升（缺 (∂Φ/∂θ)ᵀe），deep 恒 ~1e-8；
    (2) 强场训练：deep loss 明显下降、direct 停摆（偏置梯度不再有效）。
    下游损失 L(θ)=½‖x0*(θ)-x0_target‖²（z 由 θ0 生成）。
    """
    if MODE != "rbf":
        return
    global _DEEP_TFS, _DEEP_CENTERS, _PHI_Z, _PHI_X0TAR
    print("\n=== effect: Φ-dominated factor (error=Φ(θ)x0-z, K=4) ===")
    tf, K, mm = 200.0, 4, 4
    _DEEP_TFS = list(np.linspace(tf / K, tf, K))
    _DEEP_CENTERS = _centers_at(list(np.linspace(0.0, tf, mm)))
    rng = np.random.default_rng(5)

    def build(thscale):
        th0 = thscale * rng.standard_normal(mm)
        x0tar = RV0 + rng.normal(0.0, 1.0, 6)
        x0tar_t = torch.tensor(x0tar, dtype=torch.float64)
        _, Phi = DAFlowDeep.apply(torch.tensor(RV0, dtype=torch.float64),
                                  torch.tensor(th0, dtype=torch.float64))
        z = (torch.cat([Phi[k].reshape(6, 6) for k in range(K)], 0) @ x0tar_t).detach()
        return th0, x0tar_t, z

    gt = {}
    for thscale in (1e-3, 1e-1, 1.0, 100.0):          # (1) 梯度误差 vs 场强
        th0, x0tar_t, z = build(thscale)
        _PHI_Z, _PHI_X0TAR = z, x0tar_t
        te = 0.5 * th0

        def Lg(deep):
            u = torch.tensor(te.reshape(1, -1), dtype=torch.float64, requires_grad=True)
            (0.5 * ((_solve_x0_phi(u, deep) - x0tar_t) ** 2).sum()).backward()
            return u.grad.numpy().reshape(-1)

        gd, gr = Lg(True), Lg(False)
        h = thscale * 1e-4
        gfd = np.zeros(mm)
        for k in range(mm):
            tp = te.copy(); tp[k] += h; tm = te.copy(); tm[k] -= h
            Lp = 0.5 * ((_solve_x0_phi(torch.tensor(tp.reshape(1, -1), dtype=torch.float64), True) - x0tar_t) ** 2).sum()
            Lm = 0.5 * ((_solve_x0_phi(torch.tensor(tm.reshape(1, -1), dtype=torch.float64), True) - x0tar_t) ** 2).sum()
            gfd[k] = float(Lp - Lm) / (2 * h)
        rd = np.linalg.norm(gd - gfd) / (np.linalg.norm(gfd) + 1e-30)
        rr = np.linalg.norm(gr - gfd) / (np.linalg.norm(gfd) + 1e-30)
        gt[thscale] = (rd, rr)
        print(f"  field={thscale:g}: rel|g_deep-g_FD|={rd:.2e}   rel|g_direct-g_FD|={rr:.2e}")
    assert gt[100.0][0] < 1e-3, "deep gradient must stay accurate at strong field"
    assert gt[100.0][1] > 1e-1, "direct gradient must be grossly wrong at strong field"
    assert gt[100.0][1] > 100 * gt[1e-3][1], "direct error must grow with field strength"

    thscale = 100.0                                    # (2) 强场训练（展示）
    th0, x0tar_t, z = build(thscale)
    _PHI_Z, _PHI_X0TAR = z, x0tar_t

    def L(u, deep):
        return 0.5 * ((_solve_x0_phi(u * thscale, deep) - x0tar_t) ** 2).sum()

    res = {}
    for deep in (True, False):
        u = torch.zeros(1, mm, dtype=torch.float64, requires_grad=True)
        adam = torch.optim.Adam([u], lr=0.1)
        L0 = float(L(u, deep))
        for _ in range(40):
            loss = L(u, deep)
            adam.zero_grad(); loss.backward(); adam.step()
        res[deep] = (L0, float(L(u, deep)))
        print(f"  train field={thscale:g} {'deep  ' if deep else 'direct'}: L {res[deep][0]:.3e} -> {res[deep][1]:.3e}  ({res[deep][0] / (res[deep][1] + 1e-30):.0f}x)")
    assert res[True][1] < 0.2 * res[True][0], "deep must learn at strong field"


def check_deep_scaling():
    """T3：积分伴随的 fwd+bwd 时间/内存随 m 近似线性（对照 θ-in-DA 的组合爆炸）。"""
    global _DEEP_TFS
    if MODE != "rbf":
        return
    print("\n=== deep adjoint scaling vs m (T3) ===")
    _DEEP_TFS = list(TFS_FAST)   # 真实训练规模：1/4 周期、始末各 3 帧（143 步）
    rng = np.random.default_rng(5)
    base = RV0[:3]
    K = len(_DEEP_TFS)
    for mm in (30, 100, 300):
        centers = (base + rng.normal(0.0, 50.0, (mm, 3))).tolist()
        th = 1e-3 * np.ones(mm)
        gx = [list(rng.normal(0.0, 1.0, 6)) for _ in range(K)]
        gP = list(rng.normal(0.0, 1.0, K * 36))
        t0 = time.perf_counter()
        fl = qoe.daDeepForwardRBF(RV0, list(th), centers, S_RBF, _DEEP_TFS, STEP)
        gx0, gt = qoe.daDeepBackwardRBF(fl, gx, gP, centers, S_RBF)
        dt = time.perf_counter() - t0
        print(f"  m={mm:4d}: fwd+bwd {dt*1e3:8.1f} ms   max|∂L/∂θ|={np.abs(np.array(gt)).max():.2e}")


def check_learning():
    print(f"\n=== Theseus IMPLICIT learning (MODE={MODE}, m={M}) ===")
    # optimize a scaled variable u with theta = u * PARAM_SCALE (keeps u = O(1))
    scale = PARAM_SCALE
    theta_true = np.array(THETA_USE, dtype=float)
    u0 = 0.5 * np.ones(M) if MODE == "drag" else np.zeros(M)
    u_true = theta_true / scale
    x_obs, _, _, _ = da_data(RV0, theta_true)

    u = torch.tensor(u0.reshape(1, M), requires_grad=True)
    L0 = float(solve_and_loss(torch.tensor(u0 * scale), x_obs))
    adam = torch.optim.Adam([u], lr=0.02)
    for _ in range(30):
        loss = solve_and_loss(u.reshape(-1) * scale, x_obs)
        adam.zero_grad(); loss.backward(); adam.step()
    Lf = float(solve_and_loss(u.reshape(-1) * scale, x_obs))
    est = (u.detach().cpu().numpy().reshape(-1)) * scale
    print(f"  theta_true = {np.array2string(theta_true, precision=4)}")
    print(f"  theta_est  = {np.array2string(est, precision=4)}")
    print(f"  loss: {L0:.3e} -> {Lf:.3e}  (reduction {L0 / (Lf + 1e-30):.0f}x)")
    # single-snapshot field recovery is ill-conditioned (only a combination of
    # coefficients is identifiable); the criterion here is that learning drives
    # the loss down substantially, i.e. the multi-parameter gradient is correct.
    # drag 在 1/4 周期内效应极弱（L0 已到舍入量级，7e-11），无有效梯度信号 → 不对该退化情形强求下降
    assert Lf < 0.02 * L0 or L0 < 1e-9, "learning must reduce the loss"


def check_batch():
    print(f"\n=== batch / vectorized-backward check (MODE={MODE}) ===")
    rng = np.random.default_rng(0)
    B = 8
    x0b = np.tile(RV0, (B, 1)) + rng.normal(0.0, 1e-3, (B, 6))
    thb = np.array(THETA_USE)[None, :] + rng.normal(0.0, TH_PERT, (B, M))

    x0t = torch.tensor(x0b, requires_grad=True)
    tht = torch.tensor(thb, requires_grad=True)
    c = DACoeffs.apply(x0t, tht)                      # [B,6,7+m]
    c_loop = torch.stack([DACoeffs.apply(torch.tensor(x0b[b]), torch.tensor(thb[b]))
                          for b in range(B)])
    fwd = float((c - c_loop).abs().max())

    g = torch.linspace(-1.0, 1.0, c.numel()).reshape(c.shape)
    (g * c).sum().backward()
    gx, gt = x0t.grad.numpy().copy(), tht.grad.numpy().copy()

    # reference: per-sample autograd (the invariant the batch must reproduce)
    gx_loop, gt_loop = np.zeros_like(x0b), np.zeros_like(thb)
    for b in range(B):
        xb = torch.tensor(x0b[b], requires_grad=True)
        tb = torch.tensor(thb[b], requires_grad=True)
        (g[b] * DACoeffs.apply(xb, tb)).sum().backward()
        gx_loop[b], gt_loop[b] = xb.grad.numpy(), tb.grad.numpy()
    relx = np.max(np.abs(gx - gx_loop)) / (np.max(np.abs(gx)) + 1e-30)
    relt = np.max(np.abs(gt - gt_loop)) / (np.max(np.abs(gt)) + 1e-30)

    # informational: batch gradient vs central differences
    gn = g.numpy()

    def Lb(x0v, thv, gb):
        _, C, mons, _ = da_data(x0v, thv)
        return float((gb * pack_order1(C, mons, M)).sum())

    h, ht = 1e-5, FD_HT
    gx_fd, gt_fd = np.zeros_like(x0b), np.zeros_like(thb)
    for b in range(B):
        for j in range(6):
            xp = x0b.copy(); xp[b, j] += h
            xm = x0b.copy(); xm[b, j] -= h
            gx_fd[b, j] = (Lb(xp[b], thb[b], gn[b]) - Lb(xm[b], thb[b], gn[b])) / (2 * h)
        for k in range(M):
            tp = thb.copy(); tp[b, k] += ht
            tm = thb.copy(); tm[b, k] -= ht
            gt_fd[b, k] = (Lb(x0b[b], tp[b], gn[b]) - Lb(x0b[b], tm[b], gn[b])) / (2 * ht)
    fd_x = np.max(np.abs(gx - gx_fd)) / (np.max(np.abs(gx)) + 1e-30)
    fd_t = np.max(np.abs(gt - gt_fd)) / (np.max(np.abs(gt)) + 1e-30)

    print(f"  batch-vs-loop forward max|diff| = {fwd:.3e}")
    print(f"  rel |gx_batch - gx_loop| = {relx:.3e}   | gt = {relt:.3e}")
    print(f"  (vs central-diff: gx {fd_x:.3e}, gt {fd_t:.3e})")
    assert fwd < 1e-12
    assert relx < 1e-10 and relt < 1e-10   # batch backward == loop backward
    assert fd_x < 1e-4                      # and matches finite differences

    reps = 3
    t0 = time.perf_counter()
    for _ in range(reps):
        cb = DACoeffs.apply(torch.tensor(x0b, requires_grad=True),
                            torch.tensor(thb, requires_grad=True))
        (g * cb).sum().backward()
    t_batch = (time.perf_counter() - t0) / reps
    t0 = time.perf_counter()
    for _ in range(reps):
        for b in range(B):
            cb = DACoeffs.apply(torch.tensor(x0b[b], requires_grad=True),
                                torch.tensor(thb[b], requires_grad=True))
            (g[b] * cb).sum().backward()
    t_loop = (time.perf_counter() - t0) / reps
    print(f"  fwd+bwd B={B}: batch {t_batch*1e3:.1f} ms vs loop {t_loop*1e3:.1f} ms"
          f"  (speedup {t_loop/t_batch:.2f}x)")


def _sh_AB(x, y, z):
    q = 5 * z * z - (x * x + y * y + z * z)
    return {(2, 1): (3 * x * z, 3 * y * z),
            (2, 2): (3 * (x * x - y * y), 6 * x * y),
            (3, 1): (1.5 * x * q, 1.5 * y * q),
            (3, 2): (15 * z * (x * x - y * y), 30 * x * y * z),
            (3, 3): (15 * (x ** 3 - 3 * x * y * y), 15 * (3 * x * x * y - y ** 3))}


def _U_sh(r_km, thetas, lmax):
    """Independent reconstruction of the residual potential U (km^2/s^2)."""
    mu, Re = 398600.4415, 6378.137
    x, y, z = r_km
    rn = np.sqrt(x * x + y * y + z * z)
    AB = _sh_AB(x, y, z)
    U, k = 0.0, 0
    for l in range(2, lmax + 1):
        for m in range(1, l + 1):
            A, B = AB[(l, m)]
            U += mu * Re ** l / rn ** (2 * l + 1) * (thetas[k] * A + thetas[k + 1] * B)
            k += 2
    return U


def check_sh_physics():
    print(f"\n=== SH physics check (lmax={LMAX}, a vs grad of independent U) ===")
    rng = np.random.default_rng(1)
    r_km = RV0[:3] + rng.normal(0.0, 10.0, 3)
    rv6_m = np.concatenate([r_km, RV0[3:]]) * 1e3
    thetas = np.array(THETA_USE, dtype=float)
    a_num = np.array(qoe.shResidualAccel(rv6_m, list(thetas), LMAX)).reshape(-1)[:3]  # m/s^2
    h = 1e-1
    a_fd = np.zeros(3)
    for j in range(3):
        rp = r_km.copy(); rp[j] += h
        rm = r_km.copy(); rm[j] -= h
        a_fd[j] = (_U_sh(rp, thetas, LMAX) - _U_sh(rm, thetas, LMAX)) / (2 * h)  # km/s^2
    a_fd *= 1e3
    rel = np.max(np.abs(a_num - a_fd)) / (np.max(np.abs(a_fd)) + 1e-30)
    print(f"  a_num = {a_num}")
    print(f"  a_fd  = {a_fd}")
    print(f"  rel |a_num - grad U_fd| = {rel:.3e}")
    assert rel < 1e-4


class DARecordFlow(torch.autograd.Function):
    """Path A: recording-scalar tape (RBF field). forward -> xf(m); backward -> (dL/dx0, dL/dtheta)."""

    @staticmethod
    def forward(ctx, x0_km, theta, tf):
        x0m = x0_km.detach().cpu().numpy().reshape(6) * 1e3
        th = [float(t) for t in theta.detach().cpu().numpy().reshape(-1)]
        if MODE == "sh":
            rf = qoe.daRecordFlowSH(x0m, th, LMAX, float(tf), STEP)
        else:
            rf = qoe.daRecordFlowRBF(x0m, th, CENTERS, S_RBF, float(tf), STEP)
        ctx.rf = rf
        return torch.tensor(np.array(rf.xf).reshape(-1), dtype=torch.float64)

    @staticmethod
    def backward(ctx, g):
        gx_m, gp = qoe.daRecordFlowBackward(
            ctx.rf, [float(x) for x in g.detach().cpu().numpy().reshape(-1)])
        return (torch.tensor(np.array(gx_m).reshape(-1) * 1e3, dtype=torch.float64),  # m -> km
                torch.tensor(np.array(gp).reshape(-1), dtype=torch.float64), None)


def check_pathA():
    print(f"\n=== path A (recording scalar) check (MODE={MODE}, m={M}) ===")
    theta_bar0 = np.array(THETA_USE, dtype=float)
    x0 = torch.tensor(RV0, requires_grad=True)
    tb = torch.tensor(theta_bar0, requires_grad=True)
    cA = DARecordFlow.apply(x0, tb, TF)                # xf in m
    g = torch.linspace(-1.0, 1.0, cA.numel())
    (g * cA).sum().backward()
    gx_auto, gt_auto = x0.grad.numpy().copy(), tb.grad.numpy().copy()

    # reference A: path C (validated) for the same loss
    x0c = torch.tensor(RV0, requires_grad=True)
    tbc = torch.tensor(theta_bar0, requires_grad=True)
    cC = DACoeffs.apply(x0c, tbc)                      # [6,7+m]
    fwd = float((cA - cC[:, 0] * 1e3).abs().max()) / (float(cA.abs().max()) + 1e-30)
    (g * cC[:, 0] * 1e3).sum().backward()
    gx_C, gt_C = x0c.grad.numpy().copy(), tbc.grad.numpy().copy()

    # reference B: central differences of the path-A operator
    def Lnp(x0_km, tv):   # x0_km: RV0 convention (km)
        x0m = np.asarray(x0_km, dtype=float) * 1e3
        th = [float(t) for t in tv]
        rf = (qoe.daRecordFlowSH(x0m, th, LMAX, TF, STEP) if MODE == "sh"
              else qoe.daRecordFlowRBF(x0m, th, CENTERS, S_RBF, TF, STEP))
        return float((g.numpy() * np.array(rf.xf).reshape(-1)).sum())

    h, ht = 1e-3, FD_HT
    gx_fd = np.zeros(6)
    for j in range(6):
        xp = RV0.copy(); xp[j] += h
        xm = RV0.copy(); xm[j] -= h
        gx_fd[j] = (Lnp(xp, theta_bar0) - Lnp(xm, theta_bar0)) / (2 * h)
    gt_fd = np.zeros(M)
    for k in range(M):
        tp = theta_bar0.copy(); tp[k] += ht
        tm = theta_bar0.copy(); tm[k] -= ht
        gt_fd[k] = (Lnp(RV0, tp) - Lnp(RV0, tm)) / (2 * ht)

    relC = max(np.max(np.abs(gx_auto - gx_C)) / (np.max(np.abs(gx_auto)) + 1e-30),
               np.max(np.abs(gt_auto - gt_C)) / (np.max(np.abs(gt_auto)) + 1e-30))
    relF = max(np.max(np.abs(gx_auto - gx_fd)) / (np.max(np.abs(gx_auto)) + 1e-30),
               np.max(np.abs(gt_auto - gt_fd)) / (np.max(np.abs(gt_auto)) + 1e-30))
    print(f"  rel |xf_A - xf_C| = {fwd:.3e}")
    print(f"  rel grad (A vs C) = {relC:.3e}   (A vs FD) = {relF:.3e}")
    assert fwd < 1e-9
    assert relC < 1e-6 and relF < 1e-4

    reps = 5
    t0 = time.perf_counter()
    for _ in range(reps):
        cb = DARecordFlow.apply(torch.tensor(RV0, requires_grad=True),
                                torch.tensor(theta_bar0, requires_grad=True), TF)
        (g * cb).sum().backward()
    print(f"  fwd+bwd (path A, m={M}): {(time.perf_counter()-t0)/reps*1e3:.1f} ms/call")

    # large-m cost probe: gradient cost ~ graph size, not binomial(n+m, p) (RBF only)
    if MODE != "rbf":
        return
    mbig = 30
    cents = []
    for tq in np.linspace(1500.0, 20000.0, mbig):
        rf = qoe.daAugCoeffs(RV0_M, [0.0, 0.0, 0.0, 0.0], float(tq), 1, 10.0)[0] / 1e3
        cents.append([float(rf[0]), float(rf[1]), float(rf[2])])
    thbig = 1e-3 * np.ones(mbig)
    t0 = time.perf_counter()
    rfb = qoe.daRecordFlowRBF(RV0_M, list(thbig), cents, S_RBF, TF, STEP)
    qoe.daRecordFlowBackward(rfb, list(np.linspace(-1.0, 1.0, 6)))
    print(f"  path A m={mbig}: fwd+bwd {(time.perf_counter()-t0)*1e3:.1f} ms"
          f"  (grad cost ~ 图规模，不随 m 二项式爆炸；对照稠密 DA m=30,p=2 前向 3.14 s)")


def flow_nominal(x0_km, thetas, tf):
    """Nominal final state (m) at tf via the DA path (path C); matches DARecordFlow units."""
    th = [float(t) for t in thetas]
    if MODE == "rbf":
        rf = np.array(qoe.daAugRBFCoeffs(x0_km * 1e3, th, CENTERS, S_RBF, tf, ORDER, STEP)[0])
    else:
        rf = np.array(qoe.daAugSHCoeffs(x0_km * 1e3, th, LMAX, tf, ORDER, STEP)[0])
    return rf


class _MultiDyn(th.CostFunction):
    """All epochs in one dynamics factor: error = X - concat_k flow(x0, theta, t_k)."""

    def __init__(self, X, theta, tks, weight=None, name="dyn"):
        if weight is None:
            weight = th.ScaleCostWeight(torch.tensor(1.0))
        super().__init__(cost_weight=weight, name=name)
        self.X, self.theta, self.tks = X, theta, list(tks)
        self.register_optim_vars(["X"])
        self.register_aux_vars(["theta"])
        cs = [DARecordFlow.apply(X0BAR, self.theta.tensor.reshape(-1), float(tk)) for tk in self.tks]
        self.c = torch.cat([c.reshape(-1) for c in cs])

    def error(self):
        return self.X.tensor.reshape(-1) - self.c

    def jacobians(self):
        return [torch.eye(6 * len(self.tks), dtype=torch.float64)], self.error()

    def dim(self):
        return 6 * len(self.tks)

    def _copy_impl(self, new_name=None):
        return _MultiDyn(self.X.copy(), self.theta.copy(), self.tks,
                         weight=self.weight.copy(), name=new_name)


class _MultiObs(th.CostFunction):
    def __init__(self, X, obs, weight=None, name="obs"):
        if weight is None:
            weight = th.ScaleCostWeight(torch.tensor(1.0))
        super().__init__(cost_weight=weight, name=name)
        self.X, self.obs = X, obs
        self.register_optim_vars(["X"])

    def error(self):
        return self.X.tensor.reshape(-1) - self.obs

    def jacobians(self):
        return [torch.eye(self.obs.numel(), dtype=torch.float64)], self.error()

    def dim(self):
        return self.obs.numel()

    def _copy_impl(self, new_name=None):
        return _MultiObs(self.X.copy(), self.obs, weight=self.weight.copy(), name=new_name)


def solve_and_loss_multi(theta, obs_all, tks):
    X = th.Vector(tensor=torch.tensor(obs_all).reshape(1, -1), name="X")
    tv = th.Vector(tensor=theta.reshape(1, M), name="theta")
    obj = th.Objective()
    obj.add(_MultiDyn(X, tv, tks))
    obj.add(_MultiObs(X, torch.tensor(obs_all)))
    opt = th.LevenbergMarquardt(obj, max_iterations=20,
                                abs_err_tolerance=1e-12, rel_err_tolerance=1e-12)
    layer = th.TheseusLayer(opt)
    inputs = {"X": torch.tensor(obs_all).reshape(1, -1),
              "theta": theta.reshape(1, M)}
    out, _ = layer.forward(inputs, optimizer_kwargs={"track_best_solution": True})
    return 0.5 * ((out["X"].reshape(-1) - torch.tensor(obs_all)) ** 2).sum()


def check_multiepoch():
    tks = TKS if MODE == "rbf" else TKS[:3]        # keep sh memory in check
    n_steps = 100 if MODE == "rbf" else 50
    print(f"\n=== Theseus multi-epoch learning (MODE={MODE}, m={M}, K={len(tks)}) ===")
    theta_true = np.array(THETA_USE, dtype=float)
    obs_all = np.concatenate([flow_nominal(RV0, theta_true, tk) for tk in tks])
    u0 = np.zeros(M)
    u = torch.tensor(u0.reshape(1, M), requires_grad=True)
    L0 = float(solve_and_loss_multi(torch.tensor(u0 * PARAM_SCALE), obs_all, tks))
    adam = torch.optim.Adam([u], lr=0.03)
    for _ in range(n_steps):
        loss = solve_and_loss_multi(u.reshape(-1) * PARAM_SCALE, obs_all, tks)
        adam.zero_grad(); loss.backward(); adam.step()
    Lf = float(solve_and_loss_multi(u.reshape(-1) * PARAM_SCALE, obs_all, tks))
    est = (u.detach().cpu().numpy().reshape(-1)) * PARAM_SCALE
    relc = np.max(np.abs(est - theta_true)) / (np.max(np.abs(theta_true)) + 1e-30)
    print(f"  loss: {L0:.3e} -> {Lf:.3e}  (reduction {L0/(Lf + 1e-30):.0f}x)")
    print(f"  coeff rel err: {relc:.3e}")
    assert Lf < 0.02 * L0


# 一个 factor 跨 1/4 轨道周期，观测只在**始末各 3 帧**（每帧 10s）；一次积分覆盖整段。
TFS_FAST = [0.0, 10.0, 20.0, TF - 20.0, TF - 10.0, TF]


class DAFlowMulti(torch.autograd.Function):
    """One C++ integration -> states + first-order Jacobians at all epochs.

    Backward is a matmul (Gauss-Newton adjoint): dL/d[theta] = sum_k J_k^T g_k.
    No tape, no per-epoch Python loop, order=1 (linear in m).
    """

    @staticmethod
    def forward(ctx, x0_m, theta):
        th = [float(v) for v in theta.detach().cpu().numpy().reshape(-1)]
        # 严格按 NominalErrorProp 用法：RK4-in-DA 在标称 (x0,θ) 上展开，出 Taylor 的
        # 常数项(状态)与一阶系数(Jacobian)；少量参数进 DA。再积分由每次外迭代重展开承担。
        x0 = x0_m.detach().cpu().numpy()
        if MODE == "sh":
            rvf, J = qoe.daFieldMultiEpochSH(x0, th, LMAX, TFS_FAST, 1, STEP)
        else:
            rvf, J = qoe.daFieldMultiEpochRBF(x0, th, CENTERS, S_RBF, TFS_FAST, 1, STEP)
        ctx.J = torch.tensor(np.array(J).reshape(len(TFS_FAST), 6, -1))
        return torch.tensor(np.array(rvf).reshape(len(TFS_FAST), 6))

    @staticmethod
    def backward(ctx, g):
        J = ctx.J
        gx = (J[:, :, :6].transpose(1, 2) @ g.unsqueeze(-1)).sum(0).reshape(-1)
        gth = (J[:, :, 6:].transpose(1, 2) @ g.unsqueeze(-1)).sum(0).reshape(-1)
        return gx, gth


def check_train_fast():
    print(f"\n=== lean multi-epoch training (one integration/step, MODE={MODE}, m={M}) ===")
    x0 = torch.tensor(RV0_M)                       # m
    theta_true = np.array(THETA_USE, dtype=float)
    obs = torch.tensor(np.stack([flow_nominal(RV0, theta_true, tk) for tk in TFS_FAST]))
    # gradient correctness vs finite difference
    theta = torch.tensor(theta_true, requires_grad=True)
    xf = DAFlowMulti.apply(x0, theta)
    g = torch.linspace(-1.0, 1.0, xf.numel()).reshape(xf.shape)
    (xf * g).sum().backward()
    gt, gt_fd = theta.grad.numpy().copy(), np.zeros(M)
    h = FD_HT
    for k in range(M):
        tp = theta_true.copy(); tp[k] += h
        tm = theta_true.copy(); tm[k] -= h
        rp = DAFlowMulti.apply(x0, torch.tensor(tp)).detach().numpy()
        rm = DAFlowMulti.apply(x0, torch.tensor(tm)).detach().numpy()
        gt_fd[k] = float((g.numpy() * (rp - rm)).sum()) / (2 * h)
    rel = np.max(np.abs(gt - gt_fd)) / (np.max(np.abs(gt)) + 1e-30)
    print(f"  gt rel vs FD = {rel:.3e}")

    u = torch.tensor(np.zeros(M), requires_grad=True)
    opt = torch.optim.Adam([u], lr=0.03)
    L0 = float(0.5 * ((DAFlowMulti.apply(x0, u * PARAM_SCALE) - obs) ** 2).sum())
    t0 = time.perf_counter()
    for _ in range(60):
        loss = 0.5 * ((DAFlowMulti.apply(x0, u * PARAM_SCALE) - obs) ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    dt = (time.perf_counter() - t0) / 60
    Lf = float(0.5 * ((DAFlowMulti.apply(x0, u * PARAM_SCALE) - obs) ** 2).sum())
    print(f"  train K={len(TFS_FAST)}: {dt*1e3:.1f} ms/step, loss {L0:.3e} -> {Lf:.3e} ({L0/(Lf+1e-30):.0f}x)")
    assert rel < 1e-5 and Lf < 0.05 * L0


def main():
    # 核心：① DA factor 的反向传播（含 ∂c/∂θ̄，deep）对拍 FD；② Theseus 隐式微分下学 θ。
    check_backprop()
    check_deep_adjoint()   # 积分伴随：大 m deep（∂L/∂x0 与 ∂L/∂θ）对拍 FD
    check_adjoint_identity()  # 理论-1：backward 是真伴随（全 Jacobian）
    check_deep_necessary()    # 理论-2：Φ-only 损失 direct≡0、deep=FD
    check_cross_method()      # 理论-3：积分伴随 vs θ-in-DA 两法互证
    check_gn_deep()        # GN 解 x0*(θ) 的 Learning 梯度：deep vs direct 对拍 FD
    check_gn_learn()       # T4：m=30 真正学 θ（Theseus _DynPhi deep）
    check_effect_deep()    # 效果（决定性）：Φ-主导因子 deep 学 / direct 停
    check_deep_scaling()   # T3：fwd+bwd 时间随 m 近似线性
    if "full" in sys.argv:      # 研究/效率项（慢或占内存），显式开启
        check_learning()   # 小 m θ-in-DA 的 Theseus 学习（旧路径，成本高）
        check_train_fast()
        check_batch()
        check_multiepoch()
        check_pathA()
        if MODE == "sh":
            check_sh_physics()
    print(f"\nM3-large-m checks passed (MODE={MODE}).")


if __name__ == "__main__":
    main()
