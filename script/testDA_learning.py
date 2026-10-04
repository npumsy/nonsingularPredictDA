#!/usr/bin/env python3
"""M3: DACE + Learning on the real J2/J3/J4 + drag dynamics.

Problem (params from testode.py testDA_class, l130):
    r0 = (6925443.952, 190432.624, 230986.901) m
    v0 = (-303.93854, 2277.90445, 7229.09828) m/s
    dx/dt = f_J234(x) + drag(x; kappa),  kappa = scale of rho*Cd*A/m (true 1.0)

DACE (qoe.daJ234DragAugCoeffs) integrates the augmented state [x, kappa]
(kappa' = 0) with kappa as the 7th DA variable and returns the dense Taylor
coefficients.  The coefficient map center -> coeffs is wrapped in a
torch.autograd.Function whose backward is the higher-order-coefficient identity
    d c_{b,g} / d xbar_j = (b_j+1) c_{b+e_j, g},   d c_{b,g}/d kbar = (g+1) c_{b,g+1}
(no reverse tape / adjoint needed; DACE core untouched).

Checks:
  1. forward + center sensitivity vs finite differences
  2. autograd.Function gradient vs finite differences
  3. Theseus IMPLICIT learning of kappa, direct (frozen center) vs deep (rebuilt)

Needs the compiled qoe module (../build) and the in-repo Theseus on PYTHONPATH.
"""
import os
import sys

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

# --- problem constants (testode.py testDA_class) ----------------------------
RV0_M = np.array([6925443.9520, 190432.6240, 230986.9010,
                  -303.93854, 2277.90445, 7229.09828])
RV0 = RV0_M / 1e3                 # km (DACE internal unit)
TF, STEP, ORDER = 21600.0, 10.0, 2   # ~3.8 orbits: drag accumulates to O(10 m)
KAPPA_TRUE = 1.0
SIGMA = 1e-6                       # km, x0 known to ~1 mm (tight)


def da_data(x0_km, kappa, order=ORDER, tf=TF, step=STEP):
    """Return (rf_km, C(6,nmono), mons, idx) at the center (x0_km, kappa)."""
    x0_km = np.asarray(x0_km, dtype=float)
    rf, c, mons = qoe.daJ234DragAugCoeffs(x0_km * 1e3, float(kappa), tf, order, step)
    mons = np.array(mons)
    C = np.array(c).reshape(6, len(mons))
    idx = {tuple(m): k for k, m in enumerate(mons)}
    return rf * 1e-3, C, mons, idx


def mono_index(mons, exp):
    idx = {tuple(m): k for k, m in enumerate(mons)}
    return idx.get(tuple(exp), None)


# coefficient block layout (order<=1): per output i, [c0, Phi11(6), Phik] -> 8
NBLK = 8


def pack_order1(C, mons):
    idx = {tuple(m): k for k, m in enumerate(mons)}
    out = np.zeros((6, NBLK))
    e0 = [0] * 7
    out[:, 0] = C[:, idx[tuple(e0)]]
    for j in range(6):
        e = [0] * 7; e[j] = 1
        out[:, 1 + j] = C[:, idx[tuple(e)]]
    e = [0] * 7; e[6] = 1
    out[:, 7] = C[:, idx[tuple(e)]]
    return out


class DACoeffs(torch.autograd.Function):
    """forward: (x0_bar(6), kappa_bar()) -> order<=1 coefficient block (6x8);
    backward: higher-order-coefficient identity (shift)."""

    @staticmethod
    def forward(ctx, x0_bar, kappa_bar):
        x0 = x0_bar.detach().cpu().numpy().reshape(6)
        kb = float(kappa_bar.reshape(()))
        _, C, mons, idx = da_data(x0, kb, order=ORDER)
        ctx.x0 = x0
        ctx.kb = kb
        ctx.mons = mons
        ctx.idx = idx
        ctx.C = C
        return torch.tensor(pack_order1(C, mons), dtype=torch.float64)

    @staticmethod
    def backward(ctx, g):
        C, idx = ctx.C, ctx.idx
        g = g.detach().cpu().numpy().reshape(6, NBLK)

        def cf(i, exp):
            k = idx.get(tuple(exp))
            return C[i, k] if k is not None else 0.0

        gx = np.zeros(6)
        gk = 0.0
        # packed entry w has exponent beta (beta[6] = kappa exponent); its center
        # derivative is (beta_l + 1) * c_{beta + e_l} (higher-order coefficient).
        for i in range(6):
            for w in range(NBLK):
                gw = g[i, w]
                if gw == 0.0:
                    continue
                beta = [0] * 7
                if w == 7:
                    beta[6] = 1
                elif w >= 1:
                    beta[w - 1] = 1
                for l in range(7):        # center = (x0_0..x0_5, kappa)
                    e = list(beta); e[l] += 1
                    val = (beta[l] + 1) * cf(i, e)
                    if l < 6:
                        gx[l] += gw * val
                    else:
                        gk += gw * val
        return (torch.tensor(gx, dtype=torch.float64),
                torch.tensor(gk, dtype=torch.float64).reshape(()))


def phi_torch(c, dx0, dk):
    """Evaluate the order<=1 polynomial from the packed block c (6x8)."""
    c0 = c[:, 0]
    Phi11 = c[:, 1:7]
    Phik = c[:, 7]
    return c0 + Phi11 @ dx0 + Phik * dk


def check_forward_and_center_sensitivity():
    print("=== 1. forward + center sensitivity ===")
    rf, C, mons, idx = da_data(RV0, KAPPA_TRUE)
    # Phi11 vs qoe STM
    Phi = np.zeros((6, 6))
    for j in range(6):
        e = [0] * 7; e[j] = 1
        Phi[:, j] = C[:, idx[tuple(e)]]
    da = qoe.NominalErrorProp(RV0_M, 2)
    Phi_ref = np.zeros((6, 6), order="F")
    rf_ref = da.propNomJ234Drag(Phi_ref, TF, True, STEP) * 1e-3
    print(f"  |rf - qoe rvf|        = {np.max(np.abs(rf - rf_ref)):.3e} km")
    print(f"  |Phi11 - qoe STM|     = {np.max(np.abs(Phi - Phi_ref)):.3e} (qoe cuts tiny coeffs)")
    # Phi11 columns vs full-reprop central difference
    hx = 1e-4  # km (small enough for the STM linearization)
    Phi_fd = np.zeros((6, 6))
    for j in range(6):
        xp = RV0.copy(); xp[j] += hx
        xm = RV0.copy(); xm[j] -= hx
        rp, _, _, _ = da_data(xp, KAPPA_TRUE)
        rm, _, _, _ = da_data(xm, KAPPA_TRUE)
        Phi_fd[:, j] = (rp - rm) / (2 * hx)
    # Phik vs full-reprop central difference
    h = 0.5
    rp, _, _, _ = da_data(RV0, KAPPA_TRUE + h)
    rm, _, _, _ = da_data(RV0, KAPPA_TRUE - h)
    phik_fd = (rp - rm) / (2 * h)
    e7 = [0] * 7; e7[6] = 1
    phik = C[:, idx[tuple(e7)]]
    print(f"  |Phi11 - FD|          = {np.max(np.abs(Phi - Phi_fd)):.3e}")
    print(f"  |Phik - FD|           = {np.max(np.abs(phik - phik_fd)):.3e} km/unit")
    assert np.max(np.abs(rf - rf_ref)) < 1e-9
    assert np.max(np.abs(Phi - Phi_ref)) < 1e-4
    assert np.max(np.abs(Phi - Phi_fd)) < 10.0
    assert np.max(np.abs(phik - phik_fd)) < 1e-6


def check_function_grad():
    print("\n=== 2. autograd.Function gradient vs FD ===")
    x0 = torch.tensor(RV0, requires_grad=True)
    k = torch.tensor(KAPPA_TRUE, requires_grad=True)
    c = DACoeffs.apply(x0, k)
    g = torch.linspace(-1.0, 1.0, c.numel()).reshape(c.shape)  # fixed cotangent
    L = (g * c).sum()
    L.backward()
    gx_auto = x0.grad.detach().cpu().numpy().copy()
    gk_auto = float(k.grad)

    # finite difference of L(x0, k) = <g, coeffs(x0,k)>
    def Lnp(x0v, kv):
        _, C, mons, _ = da_data(x0v, kv)
        cc = pack_order1(C, mons)
        return float((g.numpy() * cc).sum())

    h = 1e-5
    gx_fd = np.zeros(6)
    for j in range(6):
        xp = RV0.copy(); xp[j] += h
        xm = RV0.copy(); xm[j] -= h
        gx_fd[j] = (Lnp(xp, KAPPA_TRUE) - Lnp(xm, KAPPA_TRUE)) / (2 * h)
    gk_fd = (Lnp(RV0, KAPPA_TRUE + h) - Lnp(RV0, KAPPA_TRUE - h)) / (2 * h)
    relx = np.max(np.abs(gx_auto - gx_fd)) / (np.max(np.abs(gx_auto)) + 1e-30)
    relk = abs(gk_auto - gk_fd) / (abs(gk_auto) + 1e-30)
    print(f"  gx_auto[:3] = {gx_auto[:3]}")
    print(f"  rel |gx_auto - gx_fd| = {relx:.3e}")
    print(f"  rel |gk_auto - gk_fd| = {relk:.3e}  (FD noise floor)")
    assert relx < 1e-6
    assert relk < 1e-2


# --- Theseus IMPLICIT learning ---------------------------------------------
X0BAR = torch.tensor(RV0)          # fixed prior mean / expansion center (km)


class _Dyn(th.CostFunction):
    def __init__(self, x0, xf, kappa, deep, weight=None, name="dyn"):
        if weight is None:
            weight = th.ScaleCostWeight(torch.tensor(1.0))
        super().__init__(cost_weight=weight, name=name)
        self.x0, self.xf, self.kappa = x0, xf, kappa
        self.deep = deep
        self.register_optim_vars(["x0", "xf"])
        self.register_aux_vars(["kappa"])
        # coefficient block is fixed during the inner solve -> compute once
        kbar = self.kappa.tensor.reshape(()) if deep else torch.tensor(KAPPA_TRUE)
        self.c = DACoeffs.apply(X0BAR, kbar)

    def error(self):
        c = self.c
        dx0 = self.x0.tensor.reshape(-1) - X0BAR
        dk = (self.kappa.tensor.reshape(()) - KAPPA_TRUE) if not self.deep else 0.0
        phi = c[:, 0] + c[:, 1:7] @ dx0 + c[:, 7] * dk
        return self.xf.tensor.reshape(-1) - phi

    def jacobians(self):
        return [-self.c[:, 1:7], torch.eye(6, dtype=torch.float64)], self.error()

    def dim(self):
        return 6

    def _copy_impl(self, new_name=None):
        return _Dyn(self.x0.copy(), self.xf.copy(), self.kappa.copy(), self.deep,
                    weight=self.weight.copy(), name=new_name)


class _Prior(th.CostFunction):
    def __init__(self, x0, weight=None, name="prior"):
        if weight is None:
            weight = th.ScaleCostWeight(torch.tensor(1.0))
        super().__init__(cost_weight=weight, name=name)
        self.x0 = x0
        self.register_optim_vars(["x0"])

    def error(self):
        return (self.x0.tensor.reshape(-1) - X0BAR) / SIGMA

    def jacobians(self):
        return [torch.eye(6, dtype=torch.float64) / SIGMA], self.error()

    def dim(self):
        return 6

    def _copy_impl(self, new_name=None):
        return _Prior(self.x0.copy(), weight=self.weight.copy(), name=new_name)


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


def solve_and_loss(kappa, x_obs, deep):
    x0 = th.Vector(tensor=X0BAR.reshape(1, 6), name="x0")
    xf = th.Vector(tensor=torch.tensor(x_obs).reshape(1, 6), name="xf")
    kv = th.Vector(tensor=kappa.reshape(1, 1), name="kappa")
    obj = th.Objective()
    obj.add(_Dyn(x0, xf, kv, deep))
    obj.add(_Prior(x0))
    obj.add(_Obs(xf, torch.tensor(x_obs)))
    opt = th.LevenbergMarquardt(obj, max_iterations=30,
                                abs_err_tolerance=1e-12, rel_err_tolerance=1e-12)
    layer = th.TheseusLayer(opt)
    inputs = {"x0": X0BAR.reshape(1, 6),
              "xf": torch.tensor(x_obs).reshape(1, 6),
              "kappa": kappa.reshape(1, 1)}
    out, _ = layer.forward(inputs, optimizer_kwargs={"track_best_solution": True})
    return 0.5 * ((out["xf"].reshape(-1) - torch.tensor(x_obs)) ** 2).sum()


def check_learning():
    print("\n=== 3. Theseus IMPLICIT learning of kappa ===")
    x_obs, _, _, _ = da_data(RV0, KAPPA_TRUE)   # km, noise-free synthetic obs

    for deep in (False, True):
        kappa = torch.tensor([[0.5]], requires_grad=True)
        adam = torch.optim.Adam([kappa], lr=0.05)
        for _ in range(100):
            loss = solve_and_loss(kappa, x_obs, deep)
            adam.zero_grad(); loss.backward(); adam.step()
        tag = "deep  " if deep else "direct"
        err = abs(float(kappa) - KAPPA_TRUE)
        print(f"  {tag}: kappa* = {float(kappa):.8f}  |k-1| = {err:.3e}")
        assert err < 5e-3, "learning must recover kappa_true"


def main():
    check_forward_and_center_sensitivity()
    check_function_grad()
    check_learning()
    print("\nM3 checks passed.")


if __name__ == "__main__":
    main()
