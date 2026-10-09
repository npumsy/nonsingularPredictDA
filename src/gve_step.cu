// GVE（QOE 非奇异要素）单步 RK4 —— CUDA 版，对整星座 N 并行。
//
// 物理与 `src/dastate.cpp` 的 `gveRhsQOE<double>` / `noeGveRhs` / `OEOsc2rvT<double>` /
// `TBPfull` 逐式一致（二体 + J2/J3/J4 + 阻力；三体在 dastate 里默认关，此处同）。
// 积分用与 `rk4<Vector6d>` 相同的 **3/8 规则**（不是经典 1/6,2/6,2/6,1/6）。
// 无 DA、无高阶展开；每帧一次 `dt` 单步，链式得到整弧状态。
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <cstdio>
#include <chrono>
#include "kepler.h"   // kep3::Vector6d + gveStepNoeBatch 声明
#include "qoejopt.h"  // qoejopt::Ctx + joint* host 包装声明

#define PI 3.14159265358979323846
#define MU_M 3.986004415e14            // m^3/s^2 (bddd::MU)
#define MU_KM 398600.4415              // km^3/s^2 (= bddd::MU/1e9)
#define RE_KM 6378.137                 // km (= bddd::RE/1e3)
#define J2C 1082.626690598e-6
#define J3C (-2.532435345754e-6)
#define J4C (-1.619331205072e-6)
#define OMEGA_EARTH 7.2921159e-5
#define RHO_CDA_M 1.42812824e-12       // TBPfull 默认
#define H0_KM 530.0                    // TBPfull 默认
#define H1_KM 65.18534                 // TBPfull 默认

// Newton 解 Kepler 方程 M->E（同 KepEqtnET<double>：初值 M±e，60 次，1e-13）。
__device__ __forceinline__ double kepE_dev(double M, double e){
    double E = ((M > -PI && M < 0.0) || M > PI) ? M - e : M + e;
    #pragma unroll 1
    for(int it = 0; it < 60; ++it){
        double En = E;
        E = En + (M - En + e * sin(En)) / (1.0 - e * cos(En));
        if(fabs(E - En) < 1e-13) break;
    }
    return E;
}

// 非奇异要素(QOE)->rv（m, m/s）。逐式镜像 `OEOsc2rvT<double>`（同分支、同表达式）。
__device__ __forceinline__ void oe2rv_dev(const double* OE, double* x){
    const double a = OE[0], u = OE[1], ex = OE[2], ey = OE[3], i = OE[4], Om = OE[5];
    const double e = sqrt(ex * ex + ey * ey);
    double p = a * (1.0 - e * e);
    if(p < 1e-6) p = 1e-6;
    double omega, nu;
    if(e < 1e-5){
        omega = 0.0;
        nu = u;
    } else {
        omega = atan2(ey, ex);
        double M = u - omega;
        if(M < -PI) M = M + floor(fabs(M - PI) / (2.0 * PI)) * 2.0 * PI;
        else if(M > PI) M = M - floor((M + PI) / (2.0 * PI)) * 2.0 * PI;
        const double E = kepE_dev(M, e);
        double q = (1.0 + e) / (1.0 - e);
        if(q < 0.0) q = 0.0;
        nu = 2.0 * atan(sqrt(q) * tan(E / 2.0));
    }
    const double cnu = cos(nu), snu = sin(nu);
    const double vscale = sqrt(MU_M / p);
    const double rPQW[3] = { p * cnu / (1.0 + e * cnu), p * snu / (1.0 + e * cnu), 0.0 };
    const double vPQW[3] = { -vscale * snu, vscale * (e + cnu), 0.0 };
    const double cO = cos(Om), sO = sin(Om), ci = cos(i), si = sin(i), cw = cos(omega), sw = sin(omega);
    const double Trow[9] = { cO*cw - sO*sw*ci, -cO*sw - sO*cw*ci,  sO*si,
                             sO*cw + cO*sw*ci, -sO*sw + cO*cw*ci, -cO*si,
                             sw*si,             cw*si,             ci };
    for(int r = 0; r < 3; ++r){
        x[r]     = Trow[3*r]*rPQW[0] + Trow[3*r+1]*rPQW[1] + Trow[3*r+2]*rPQW[2];
        x[3 + r] = Trow[3*r]*vPQW[0] + Trow[3*r+1]*vPQW[1] + Trow[3*r+2]*vPQW[2];
    }
}

// 二体 + J2/J3/J4 + 阻力（km, km/s，km/s^2）。三体默认关，未计入（同 dastate 默认）。
__device__ __forceinline__ void tbp_dev(const double* xk, double beta, double* res){
    const double pos0 = xk[0], pos1 = xk[1], pos2 = xk[2];
    const double r = sqrt(pos0*pos0 + pos1*pos1 + pos2*pos2);
    const double z_r = pos2 / r, z2 = z_r*z_r, z3 = z2*z_r, z4 = z3*z_r;
    const double Re_r = RE_KM / r, Re2 = Re_r*Re_r, Re3 = Re2*Re_r, Re4 = Re3*Re_r;
    const double cf = -MU_KM / (r*r*r);

    double j2 = (3.0/2.0)*J2C*Re2*(1.0 - 5.0*z2);
    double j3 = (5.0/2.0)*J3C*Re3*(3.0*z_r - 7.0*z3);
    double j4 = (5.0/8.0)*J4C*Re4*(3.0 - 42.0*z2 + 63.0*z4);
    res[3] = cf*pos0*(1.0 + j2 + j3 - j4);
    res[4] = cf*pos1*(1.0 + j2 + j3 - j4);

    j2 = (3.0/2.0)*J2C*Re2*(3.0 - 5.0*z2);
    j3 = (5.0/2.0)*J3C*Re3*(6.0*z_r - 7.0*z3 - (3.0/5.0)*r/pos2);
    j4 = (5.0/8.0)*J4C*Re4*(15.0 - 70.0*z2 + 63.0*z4);
    res[5] = cf*pos2*(1.0 + j2 + j3 - j4);

    res[0] = xk[3]; res[1] = xk[4]; res[2] = xk[5];

    const double rvx = xk[3] + OMEGA_EARTH*xk[1], rvy = xk[4] - OMEGA_EARTH*xk[0], rvz = xk[5];
    const double v = sqrt(rvx*rvx + rvy*rvy + rvz*rvz);
    const double rho = exp(-(r - RE_KM - H0_KM) / H1_KM);
    const double dc = -0.5*RHO_CDA_M*beta*rho*v;
    res[3] += dc*rvx; res[4] += dc*rvy; res[5] += dc*rvz;
}

// 摄动加速度（去二体，ECI，m/s^2）：rv(m) -> TBPfull(km) 减二体后回 m。同 pertAccelT<double>。
__device__ __forceinline__ void pertAccel_dev(const double* rv_m, double beta, double* ap){
    double xk[6];
    #pragma unroll
    for(int k = 0; k < 6; ++k) xk[k] = rv_m[k] * 1e-3;
    double acc[6];
    tbp_dev(xk, beta, acc);
    const double rn = sqrt(xk[0]*xk[0] + xk[1]*xk[1] + xk[2]*xk[2]);
    const double c = MU_KM / (rn*rn*rn);
    #pragma unroll
    for(int k = 0; k < 3; ++k) ap[k] = (acc[3 + k] + c*xk[k]) * 1e3;
}

// GVE RHS：d/dt [a, u=M+w, ex, ey, i, Om]。同 gveRhsQOE<double>。
__device__ __forceinline__ void gveRhs_dev(const double* OE, double beta, double* out){
    const double a = OE[0], u = OE[1], ex = OE[2], ey = OE[3], inc = OE[4];
    const double e = sqrt(ex*ex + ey*ey);
    double p = a * (1.0 - e*e);
    if(p < 1e-6) p = 1e-6;
    const double w = (e > 0.0) ? atan2(ey, ex) : 0.0;
    double M = u - w;
    if(M < -PI) M = M + floor(fabs(M - PI) / (2.0*PI)) * 2.0*PI;
    else if(M > PI) M = M - floor((M + PI) / (2.0*PI)) * 2.0*PI;
    const double E = kepE_dev(M, e);
    const double cE = cos(E), sE = sin(E);
    const double cnu = (cE - e) / (1.0 - e*cE);
    double q = 1.0 - e*e;
    if(q < 0.0) q = 0.0;
    const double snu = sqrt(q) * sE / (1.0 - e*cE);
    const double r = p / (1.0 + e*cnu);
    const double h = sqrt(MU_M * p);

    double rv[6];
    oe2rv_dev(OE, rv);
    double ap[3];
    pertAccel_dev(rv, beta, ap);

    const double r3[3] = { rv[0], rv[1], rv[2] };
    const double v3[3] = { rv[3], rv[4], rv[5] };
    const double rn = sqrt(r3[0]*r3[0] + r3[1]*r3[1] + r3[2]*r3[2]);
    const double Rc[3] = { r3[0]/rn, r3[1]/rn, r3[2]/rn };
    const double hv[3] = { r3[1]*v3[2] - r3[2]*v3[1],
                           r3[2]*v3[0] - r3[0]*v3[2],
                           r3[0]*v3[1] - r3[1]*v3[0] };
    const double hvn = sqrt(hv[0]*hv[0] + hv[1]*hv[1] + hv[2]*hv[2]);
    const double Nc[3] = { hv[0]/hvn, hv[1]/hvn, hv[2]/hvn };
    const double Tc[3] = { Nc[1]*Rc[2] - Nc[2]*Rc[1],
                           Nc[2]*Rc[0] - Nc[0]*Rc[2],
                           Nc[0]*Rc[1] - Nc[1]*Rc[0] };
    double aR = 0.0, aT = 0.0, aN = 0.0;
    #pragma unroll
    for(int k = 0; k < 3; ++k){ aR += ap[k]*Rc[k]; aT += ap[k]*Tc[k]; aN += ap[k]*Nc[k]; }

    const double ef = (e < 1e-8) ? 1e-8 : e;
    double si = sin(inc);
    const double sif = (si < 1e-8) ? 1e-8 : si;
    const double uarg = w + atan2(snu, cnu);
    const double da  = (2.0*a*a/h) * (e*snu*aR + (p/r)*aT);
    const double de  = (1.0/h) * (p*snu*aR + ((p + r)*cnu + r*e)*aT);
    const double di  = (r*cos(uarg)/h) * aN;
    const double dOm = (r*sin(uarg)/(h*sif)) * aN;
    const double dw  = (1.0/(h*ef)) * (-p*cnu*aR + (p + r)*snu*aT)
                     - (r*sin(uarg)*cos(inc)/(h*sif)) * aN;
    const double fM  = sqrt(q) / (h*ef);
    const double dM  = sqrt(MU_M / (a*a*a)) + fM * ((p*cnu - 2.0*r*e)*aR - (p + r)*snu*aT);

    out[0] = da;
    out[1] = dM + dw;
    out[2] = cos(w)*de - e*sin(w)*dw;
    out[3] = sin(w)*de + e*cos(w)*dw;
    out[4] = di;
    out[5] = dOm;
}

// ---- 笛卡尔 ECI (x,y,z,vx,vy,vz) 全动力学 RHS（m, m/s）：dx=[v; a]，a=TBPfull(km)→m/s² ----
__device__ __forceinline__ void cartRhsDev(const double* x, double beta, double* dx){
    double xk[6];
    #pragma unroll
    for(int k = 0; k < 6; ++k) xk[k] = x[k] * 1e-3;
    double acc[6];
    tbp_dev(xk, beta, acc);                     // 二体+J234+阻力，km/s²
    #pragma unroll
    for(int c = 0; c < 3; ++c) dx[c] = x[3 + c];
    #pragma unroll
    for(int c = 0; c < 3; ++c) dx[3 + c] = acc[3 + c] * 1e3;   // → m/s²
}

// ---- 两体 STM（device）：逐式镜像 kep3::propagate_lagrangian(椭圆) + stm_lagrangian ----
__device__ __forceinline__ void _dot36(const double* u3, const double* A36, double* out6){
    for(int b = 0; b < 6; ++b){ double s = 0.0; for(int a = 0; a < 3; ++a) s += u3[a]*A36[a*6 + b]; out6[b] = s; }
}
__device__ __forceinline__ void _outer36(const double* u3, const double* r6, double* out36){
    for(int a = 0; a < 3; ++a) for(int b = 0; b < 6; ++b) out36[a*6 + b] = u3[a]*r6[b];
}

// Φ = ∂(r_f,v_f)/∂(r_0,v_0)（km 口径；非椭圆回退单位阵）。与 host `stateStmFoldBatch` 一致。
__device__ __forceinline__ void stmTwoBodyDev(const double* r0, const double* v0, double tof,
                                              double mu, double Phi[6][6]){
    #pragma unroll
    for(int a = 0; a < 6; ++a)
        #pragma unroll
        for(int b = 0; b < 6; ++b) Phi[a][b] = (a == b) ? 1.0 : 0.0;
    const double R0 = sqrt(r0[0]*r0[0] + r0[1]*r0[1] + r0[2]*r0[2]);
    const double V02 = v0[0]*v0[0] + v0[1]*v0[1] + v0[2]*v0[2];
    const double energy = V02/2.0 - mu/R0;
    const double a = -mu/2.0/energy;
    if(!(a > 0.0)) return;                       // 只做椭圆
    const double sqrta = sqrt(a);
    const double sqrtmu = sqrt(mu);
    const double sigma0 = (r0[0]*v0[0] + r0[1]*v0[1] + r0[2]*v0[2])/sqrtmu;
    const double DM = sqrt(mu/(a*a*a))*tof;
    const double sinDM = sin(DM), cosDM = cos(DM);
    double DMc = atan2(sinDM, cosDM);
    if(DMc < 0) DMc += 2.0*PI;
    const double s0 = sigma0/sqrta;
    const double c0 = 1.0 - R0/a;
    double IG = DMc + c0*sinDM - s0*(1.0 - cosDM)
                + (c0*cosDM - s0*sinDM)*(c0*sinDM + s0*cosDM - s0)
                + 0.5*(c0*sinDM + s0*cosDM - s0)
                      *(2.0*(c0*cosDM - s0*sinDM)*(c0*cosDM - s0*sinDM)
                        - (c0*sinDM + s0*cosDM - s0)*(c0*sinDM + s0*cosDM));
    double DE = IG;
    #pragma unroll 1
    for(int it = 0; it < 60; ++it){
        const double f = -DMc + DE + s0*(1.0 - cos(DE)) - c0*sin(DE);
        const double df = 1.0 + s0*sin(DE) - c0*cos(DE);
        if(df == 0.0) break;
        const double d = f/df;
        DE -= d;
        if(fabs(d) < 1e-15) break;
    }
    const double sinDE = sin(DE), cosDE = cos(DE);
    const double Rf = a + (R0 - a)*cosDE + sigma0*sqrta*sinDE;
    const double F = 1.0 - a/R0*(1.0 - cosDE);
    const double G = a*sigma0/sqrtmu*(1.0 - cosDE) + R0*sqrt(a/mu)*sinDE;
    const double Ft = -sqrt(mu*a)/(Rf*R0)*sinDE;
    const double Gt = 1.0 - a/Rf*(1.0 - cosDE);

    double dr0[18], dv0[18];
    for(int k = 0; k < 18; ++k){ dr0[k] = 0.0; dv0[k] = 0.0; }
    dr0[0] = 1.0; dr0[7] = 1.0; dr0[14] = 1.0;
    dv0[3] = 1.0; dv0[10] = 1.0; dv0[17] = 1.0;

    double dV02[6], dR0[6], denergy[6], dsigma0[6], da[6], t1[6], t2[6];
    _dot36(v0, dv0, dV02);
    for(int b = 0; b < 6; ++b) dV02[b] *= 2.0;
    _dot36(r0, dr0, dR0);  for(int b = 0; b < 6; ++b) dR0[b] /= R0;
    for(int b = 0; b < 6; ++b) denergy[b] = 0.5*dV02[b] + mu/(R0*R0)*dR0[b];
    _dot36(r0, dv0, t1); _dot36(v0, dr0, t2);
    for(int b = 0; b < 6; ++b) dsigma0[b] = (t1[b] + t2[b])/sqrtmu;
    for(int b = 0; b < 6; ++b) da[b] = mu/2.0/(energy*energy)*denergy[b];

    const double sqrta5 = sqrta*sqrta*sqrta*sqrta*sqrta;
    double ds0[6], dc0[6], dDM[6], dDE[6], dRf[6], dF[6], dG[6], dFt[6], dGt[6];
    for(int b = 0; b < 6; ++b) ds0[b] = dsigma0[b]/sqrta - 0.5*sigma0/(sqrta*sqrta*sqrta)*da[b];
    for(int b = 0; b < 6; ++b) dc0[b] = -1.0/a*dR0[b] + R0/(a*a)*da[b];
    for(int b = 0; b < 6; ++b) dDM[b] = -1.5*sqrtmu*tof/sqrta5*da[b];
    const double denom = 1.0 + s0*sinDE - c0*cosDE;
    for(int b = 0; b < 6; ++b) dDE[b] = (dDM[b] - (1.0 - cosDE)*ds0[b] + sinDE*dc0[b])/denom;
    for(int b = 0; b < 6; ++b)
        dRf[b] = (1.0 - cosDE + 0.5/sqrta*sigma0*sinDE)*da[b] + cosDE*dR0[b]
               + (sigma0*sqrta*cosDE - (R0 - a)*sinDE)*dDE[b] + sqrta*sinDE*dsigma0[b];
    for(int b = 0; b < 6; ++b)
        dF[b] = -(1.0 - cosDE)/R0*da[b] + a/(R0*R0)*(1.0 - cosDE)*dR0[b] - a/R0*sinDE*dDE[b];
    for(int b = 0; b < 6; ++b)
        dG[b] = (1.0 - F)*(R0*dsigma0[b] + sigma0*dR0[b]) - (sigma0*R0)*dF[b]
              + (sqrta*R0*cosDE)*dDE[b] + (sqrta*sinDE)*dR0[b] + (0.5*R0*sinDE/sqrta)*da[b];
    for(int b = 0; b < 6; ++b) dG[b] /= sqrtmu;
    for(int b = 0; b < 6; ++b)
        dFt[b] = (-sqrta/(R0*Rf)*cosDE)*dDE[b] - (0.5/(sqrta*R0*Rf)*sinDE)*da[b]
               + (sqrta/(Rf*R0*R0)*sinDE)*dR0[b] + (sqrta/(Rf*Rf*R0)*sinDE)*dRf[b];
    for(int b = 0; b < 6; ++b) dFt[b] *= sqrtmu;
    for(int b = 0; b < 6; ++b)
        dGt[b] = -(1.0 - cosDE)/Rf*da[b] + a/(Rf*Rf)*(1.0 - cosDE)*dRf[b] - a/Rf*sinDE*dDE[b];

    double r0dF[18], v0dG[18], r0dFt[18], v0dGt[18], Mr[18], Mv[18];
    _outer36(r0, dF, r0dF);   _outer36(v0, dG, v0dG);
    _outer36(r0, dFt, r0dFt); _outer36(v0, dGt, v0dGt);
    for(int k = 0; k < 18; ++k){
        Mr[k] = F*dr0[k] + r0dF[k] + G*dv0[k] + v0dG[k];
        Mv[k] = Ft*dr0[k] + r0dFt[k] + Gt*dv0[k] + v0dGt[k];
    }
    for(int k = 0; k < 18; ++k){
        double v = Mr[k];
        if(!isfinite(v)) for(int q = 0; q < 18; ++q) Mr[q] = 0.0;   // 退化回退
        Phi[k/6][k%6] = Mr[k];
        Phi[3 + k/6][k%6] = Mv[k];
    }
}

// 单步 3/8 RK4（与 dastate.cpp::rk4 的步进公式一致：h 为步长）。
__global__ void gveStepKernel(const double* oe0, int n, double dt, double beta, double* oef){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n) return;
    const double* y0 = oe0 + 6*idx;
    double y[6], k1[6], k2[6], k3[6], k4[6], tmp[6];
    #pragma unroll
    for(int c = 0; c < 6; ++c) y[c] = y0[c];

    gveRhs_dev(y, beta, k1);
    #pragma unroll
    for(int c = 0; c < 6; ++c) tmp[c] = y[c] + dt*k1[c]/3.0;
    gveRhs_dev(tmp, beta, k2);
    #pragma unroll
    for(int c = 0; c < 6; ++c) tmp[c] = y[c] + dt*(-k1[c]/3.0 + k2[c]);
    gveRhs_dev(tmp, beta, k3);
    #pragma unroll
    for(int c = 0; c < 6; ++c) tmp[c] = y[c] + dt*(k1[c] - k2[c] + k3[c]);
    gveRhs_dev(tmp, beta, k4);

    double* out = oef + 6*idx;
    #pragma unroll
    for(int c = 0; c < 6; ++c)
        out[c] = y[c] + dt*(k1[c] + 3.0*k2[c] + 3.0*k3[c] + k4[c])/8.0;
}

// 整弧多帧：每星一个线程，链式 nfr-1 次单步 3/8-RK4，输出每帧 rv（m, m/s），布局 [f][i]。
__global__ void gvePropagateKernel(const double* oe0, int n, int nfr, double dt, double beta, double* rv_all){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n) return;
    double y[6], k1[6], k2[6], k3[6], k4[6], tmp[6], rv[6];
    #pragma unroll
    for(int c = 0; c < 6; ++c) y[c] = oe0[6*idx + c];
    oe2rv_dev(y, rv);
    #pragma unroll
    for(int c = 0; c < 6; ++c) rv_all[((size_t)0*n + idx)*6 + c] = rv[c];
    for(int f = 1; f < nfr; ++f){
        gveRhs_dev(y, beta, k1);
        #pragma unroll
        for(int c = 0; c < 6; ++c) tmp[c] = y[c] + dt*k1[c]/3.0;
        gveRhs_dev(tmp, beta, k2);
        #pragma unroll
        for(int c = 0; c < 6; ++c) tmp[c] = y[c] + dt*(-k1[c]/3.0 + k2[c]);
        gveRhs_dev(tmp, beta, k3);
        #pragma unroll
        for(int c = 0; c < 6; ++c) tmp[c] = y[c] + dt*(k1[c] - k2[c] + k3[c]);
        gveRhs_dev(tmp, beta, k4);
        #pragma unroll
        for(int c = 0; c < 6; ++c) y[c] = y[c] + dt*(k1[c] + 3.0*k2[c] + 3.0*k3[c] + k4[c])/8.0;
        oe2rv_dev(y, rv);
        #pragma unroll
        for(int c = 0; c < 6; ++c) rv_all[((size_t)f*n + idx)*6 + c] = rv[c];
    }
}

// 整弧多帧（逐星 κ=betas[idx]）：P4.3 θ 学习的正向；与 gvePropagateKernel 仅 β 来源不同。
__global__ void gvePropagateBetaKernel(const double* oe0, const double* betas, int n, int nfr,
                                       double dt, double* rv_all){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n) return;
    const double beta = betas[idx];
    double y[6], k1[6], k2[6], k3[6], k4[6], tmp[6], rv[6];
    #pragma unroll
    for(int c = 0; c < 6; ++c) y[c] = oe0[6*idx + c];
    oe2rv_dev(y, rv);
    #pragma unroll
    for(int c = 0; c < 6; ++c) rv_all[((size_t)0*n + idx)*6 + c] = rv[c];
    for(int f = 1; f < nfr; ++f){
        gveRhs_dev(y, beta, k1);
        #pragma unroll
        for(int c = 0; c < 6; ++c) tmp[c] = y[c] + dt*k1[c]/3.0;
        gveRhs_dev(tmp, beta, k2);
        #pragma unroll
        for(int c = 0; c < 6; ++c) tmp[c] = y[c] + dt*(-k1[c]/3.0 + k2[c]);
        gveRhs_dev(tmp, beta, k3);
        #pragma unroll
        for(int c = 0; c < 6; ++c) tmp[c] = y[c] + dt*(k1[c] - k2[c] + k3[c]);
        gveRhs_dev(tmp, beta, k4);
        #pragma unroll
        for(int c = 0; c < 6; ++c) y[c] = y[c] + dt*(k1[c] + 3.0*k2[c] + 3.0*k3[c] + k4[c])/8.0;
        oe2rv_dev(y, rv);
        #pragma unroll
        for(int c = 0; c < 6; ++c) rv_all[((size_t)f*n + idx)*6 + c] = rv[c];
    }
}

// 笛卡尔单步 3/8-RK4（全动力学）：x0 (n×6, m/m·s⁻¹) → xf。
__global__ void cartStepKernel(const double* x0, int n, double dt, double beta, double* xf){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n) return;
    double y[6], k1[6], k2[6], k3[6], k4[6], tmp[6];
    #pragma unroll
    for(int c = 0; c < 6; ++c) y[c] = x0[6*idx + c];
    cartRhsDev(y, beta, k1);
    #pragma unroll
    for(int c = 0; c < 6; ++c) tmp[c] = y[c] + dt*k1[c]/3.0;
    cartRhsDev(tmp, beta, k2);
    #pragma unroll
    for(int c = 0; c < 6; ++c) tmp[c] = y[c] + dt*(-k1[c]/3.0 + k2[c]);
    cartRhsDev(tmp, beta, k3);
    #pragma unroll
    for(int c = 0; c < 6; ++c) tmp[c] = y[c] + dt*(k1[c] - k2[c] + k3[c]);
    cartRhsDev(tmp, beta, k4);
    #pragma unroll
    for(int c = 0; c < 6; ++c) xf[6*idx + c] = y[c] + dt*(k1[c] + 3.0*k2[c] + 3.0*k3[c] + k4[c])/8.0;
}

// 笛卡尔整弧多帧：每星线程链式 nfr-1 次单步，输出每帧 x（平铺 nfr*n*6）。
__global__ void cartPropagateKernel(const double* x0, int n, int nfr, double dt, double beta, double* x_all){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n) return;
    double y[6], k1[6], k2[6], k3[6], k4[6], tmp[6];
    #pragma unroll
    for(int c = 0; c < 6; ++c) y[c] = x0[6*idx + c];
    #pragma unroll
    for(int c = 0; c < 6; ++c) x_all[((size_t)0*n + idx)*6 + c] = y[c];
    for(int f = 1; f < nfr; ++f){
        cartRhsDev(y, beta, k1);
        #pragma unroll
        for(int c = 0; c < 6; ++c) tmp[c] = y[c] + dt*k1[c]/3.0;
        cartRhsDev(tmp, beta, k2);
        #pragma unroll
        for(int c = 0; c < 6; ++c) tmp[c] = y[c] + dt*(-k1[c]/3.0 + k2[c]);
        cartRhsDev(tmp, beta, k3);
        #pragma unroll
        for(int c = 0; c < 6; ++c) tmp[c] = y[c] + dt*(k1[c] - k2[c] + k3[c]);
        cartRhsDev(tmp, beta, k4);
        #pragma unroll
        for(int c = 0; c < 6; ++c) y[c] = y[c] + dt*(k1[c] + 3.0*k2[c] + 3.0*k3[c] + k4[c])/8.0;
        #pragma unroll
        for(int c = 0; c < 6; ++c) x_all[((size_t)f*n + idx)*6 + c] = y[c];
    }
}

// 解析两体 STM 折叠（GPU 版 stateStmFoldBatch）：每 (星,帧) 算 Φ_f，A_f=(Φ_f·A0)[0:3,:]。
__global__ void stmFoldKernel(const double* rv0, int n, int nfr, double dt, const double* A0, double* A_all){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n) return;
    const double r0[3] = { rv0[idx*6+0]*1e-3, rv0[idx*6+1]*1e-3, rv0[idx*6+2]*1e-3 };
    const double v0[3] = { rv0[idx*6+3]*1e-3, rv0[idx*6+4]*1e-3, rv0[idx*6+5]*1e-3 };
    const double mu = MU_M * 1e-9;                      // km^3/s^2
    const double* a0i = A0 + (size_t)idx*36;
    for(int f = 0; f < nfr; ++f){
        double Phi[6][6];
        if(f == 0){
            #pragma unroll
            for(int a = 0; a < 6; ++a)
                #pragma unroll
                for(int b = 0; b < 6; ++b) Phi[a][b] = (a == b) ? 1.0 : 0.0;
        } else {
            stmTwoBodyDev(r0, v0, (double)f*dt, mu, Phi);
        }
        double* Ao = A_all + ((size_t)f*n + idx)*18;
        for(int a = 0; a < 3; ++a)
            for(int b = 0; b < 6; ++b){
                double s = 0.0;
                for(int cc = 0; cc < 6; ++cc) s += Phi[a][cc]*a0i[cc*6 + b];
                Ao[a*6 + b] = s;
            }
    }
}

// Host 包装：对 n 个 QOE 各做一次 dt 单步（链式调用即得整弧）。t0 保留接口（RHS 自治，未用）。
namespace kep3 {
// 全流程细粒度计时（秒）：pack=host 装/解包、h2d=输入拷贝、kernel=CUDA 计算、d2h=输出拷贝。
// 全流程细粒度计时（秒）：pack=host 装/解包、h2d=输入拷贝、kernel=CUDA 计算、d2h=输出拷贝、
// stm=两体 STM 折叠 kernel。
double g_gve_pack = 0.0, g_gve_h2d = 0.0, g_gve_kernel = 0.0, g_gve_d2h = 0.0, g_gve_stm = 0.0;
void gveResetTiming(){ g_gve_pack = g_gve_h2d = g_gve_kernel = g_gve_d2h = g_gve_stm = 0.0; }
std::vector<double> gveGetTiming(){ return { g_gve_pack, g_gve_h2d, g_gve_kernel, g_gve_d2h, g_gve_stm }; }

void gveStepNoeBatch(const std::vector<Vector6d> &oe0s, double t0, double dt,
                     int nthreads, double beta, std::vector<Vector6d> &oef){
    (void)t0; (void)nthreads;
    using clk = std::chrono::steady_clock;
    auto _now = []{ return std::chrono::duration<double>(clk::now().time_since_epoch()).count(); };
    const int n = (int)oe0s.size();
    oef.assign(n, Vector6d::Zero());
    if(n == 0) return;

    double _t = _now();
    static thread_local std::vector<double> h0, hf;
    h0.resize((size_t)n * 6);
    hf.resize((size_t)n * 6);
    for(int i = 0; i < n; ++i)
        for(int c = 0; c < 6; ++c) h0[(size_t)i*6 + c] = oe0s[i](c);
    g_gve_pack += _now() - _t;

    // 设备缓冲复用（避免每帧 cudaMalloc/cudaFree 抖动）
    static thread_local double *d0 = nullptr, *df = nullptr;
    static thread_local size_t cap = 0;
    const size_t bytes = (size_t)n * 6 * sizeof(double);
    if(bytes > cap){
        if(d0) cudaFree(d0);
        if(df) cudaFree(df);
        d0 = df = nullptr;
        cudaMalloc(&d0, bytes);
        cudaMalloc(&df, bytes);
        cap = bytes;
    }
    _t = _now();
    cudaMemcpy(d0, h0.data(), bytes, cudaMemcpyHostToDevice);
    g_gve_h2d += _now() - _t;

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    _t = _now();
    gveStepKernel<<<blocks, threads>>>(d0, n, dt, beta, df);
    cudaError_t err = cudaDeviceSynchronize();
    g_gve_kernel += _now() - _t;
    if(err != cudaSuccess)
        fprintf(stderr, "[gveStepNoeBatch] kernel error: %s\n", cudaGetErrorString(err));

    _t = _now();
    cudaMemcpy(hf.data(), df, bytes, cudaMemcpyDeviceToHost);
    for(int i = 0; i < n; ++i)
        for(int c = 0; c < 6; ++c) oef[i](c) = hf[(size_t)i*6 + c];
    g_gve_d2h += _now() - _t;   // D2H + 解包
}

// 整弧多帧 GVE：nfr 帧一次调用（内部链式单步），输出每帧 rv（平铺 nfr*n*6，布局 [f][i][6]）。
void gvePropagateNoeBatch(const std::vector<Vector6d> &oe0s, int nfr, double dt, int nthreads,
                          double beta, std::vector<double> &rv_all){
    (void)nthreads;
    using clk = std::chrono::steady_clock;
    auto _now = []{ return std::chrono::duration<double>(clk::now().time_since_epoch()).count(); };
    const int n = (int)oe0s.size();
    rv_all.assign((size_t)nfr * n * 6, 0.0);
    if(n == 0 || nfr <= 0) return;

    static thread_local std::vector<double> h0;
    h0.resize((size_t)n * 6);
    double _t = _now();
    for(int i = 0; i < n; ++i)
        for(int c = 0; c < 6; ++c) h0[(size_t)i*6 + c] = oe0s[i](c);
    g_gve_pack += _now() - _t;

    static thread_local double *d0 = nullptr, *drv = nullptr;
    static thread_local size_t cap0 = 0, caprv = 0;
    const size_t b0 = (size_t)n*6*sizeof(double), brv = (size_t)nfr*n*6*sizeof(double);
    if(b0 > cap0){ if(d0) cudaFree(d0); d0 = nullptr; cudaMalloc(&d0, b0); cap0 = b0; }
    if(brv > caprv){ if(drv) cudaFree(drv); drv = nullptr; cudaMalloc(&drv, brv); caprv = brv; }

    _t = _now();
    cudaMemcpy(d0, h0.data(), b0, cudaMemcpyHostToDevice);
    g_gve_h2d += _now() - _t;

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    _t = _now();
    gvePropagateKernel<<<blocks, threads>>>(d0, n, nfr, dt, beta, drv);
    cudaError_t err = cudaDeviceSynchronize();
    g_gve_kernel += _now() - _t;
    if(err != cudaSuccess)
        fprintf(stderr, "[gvePropagateNoeBatch] kernel error: %s\n", cudaGetErrorString(err));

    _t = _now();
    cudaMemcpy(rv_all.data(), drv, brv, cudaMemcpyDeviceToHost);
    g_gve_d2h += _now() - _t;
}

// 整弧多帧（逐星 κ）：oe0s(n,6) + betas(n) --多帧链式--> rv_all（平铺 nfr*n*6）。
void gvePropagateNoeBatchBeta(const std::vector<Vector6d> &oe0s, const std::vector<double> &betas,
                              int nfr, double dt, int nthreads, std::vector<double> &rv_all){
    (void)nthreads;
    using clk = std::chrono::steady_clock;
    auto _now = []{ return std::chrono::duration<double>(clk::now().time_since_epoch()).count(); };
    const int n = (int)oe0s.size();
    rv_all.assign((size_t)nfr * n * 6, 0.0);
    if(n == 0 || nfr <= 0) return;
    static thread_local std::vector<double> h0, hb;
    h0.resize((size_t)n * 6); hb.resize((size_t)n);
    double _t = _now();
    for(int i = 0; i < n; ++i){
        for(int c = 0; c < 6; ++c) h0[(size_t)i*6 + c] = oe0s[i](c);
        hb[(size_t)i] = (betas.size() == (size_t)n) ? betas[i] : (betas.empty() ? 1.0 : betas[0]);
    }
    g_gve_pack += _now() - _t;
    static thread_local double *d0 = nullptr, *db = nullptr, *drv = nullptr;
    static thread_local size_t c0 = 0, cb = 0, crv = 0;
    const size_t b0 = (size_t)n*6*sizeof(double), bb = (size_t)n*sizeof(double),
                 brv = (size_t)nfr*n*6*sizeof(double);
    if(b0 > c0){ if(d0) cudaFree(d0); d0 = nullptr; cudaMalloc(&d0, b0); c0 = b0; }
    if(bb > cb){ if(db) cudaFree(db); db = nullptr; cudaMalloc(&db, bb); cb = bb; }
    if(brv > crv){ if(drv) cudaFree(drv); drv = nullptr; cudaMalloc(&drv, brv); crv = brv; }
    _t = _now();
    cudaMemcpy(d0, h0.data(), b0, cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb.data(), bb, cudaMemcpyHostToDevice);
    g_gve_h2d += _now() - _t;
    const int threads = 256, blocks = (n + threads - 1) / threads;
    _t = _now();
    gvePropagateBetaKernel<<<blocks, threads>>>(d0, db, n, nfr, dt, drv);
    cudaError_t err = cudaDeviceSynchronize();
    g_gve_kernel += _now() - _t;
    if(err != cudaSuccess) fprintf(stderr, "[gvePropagateNoeBatchBeta] %s\n", cudaGetErrorString(err));
    _t = _now();
    cudaMemcpy(rv_all.data(), drv, brv, cudaMemcpyDeviceToHost);
    g_gve_d2h += _now() - _t;
}

// 解析两体 STM 折叠（GPU 版 stateStmFoldBatch）：rv0(n,6) + A0(n,36) → A（平铺 nfr*n*18）。
void stmFoldGpuBatch(const std::vector<Vector6d> &rv0s, int nfr, double dt,
                     const std::vector<double> &A0flat, int nthreads, std::vector<double> &A_all){
    (void)nthreads;
    using clk = std::chrono::steady_clock;
    auto _now = []{ return std::chrono::duration<double>(clk::now().time_since_epoch()).count(); };
    const int n = (int)rv0s.size();
    A_all.assign((size_t)nfr * n * 18, 0.0);
    if(n == 0 || nfr <= 0) return;

    static thread_local std::vector<double> hrv, ha0;
    hrv.resize((size_t)n * 6); ha0.resize((size_t)n * 36);
    double _t = _now();
    for(int i = 0; i < n; ++i){
        for(int c = 0; c < 6; ++c) hrv[(size_t)i*6 + c] = rv0s[i](c);
        for(int c = 0; c < 36; ++c) ha0[(size_t)i*36 + c] = A0flat[(size_t)i*36 + c];
    }
    g_gve_pack += _now() - _t;

    static thread_local double *drv = nullptr, *da0 = nullptr, *dA = nullptr;
    static thread_local size_t crv = 0, ca0 = 0, cA = 0;
    const size_t brv = (size_t)n*6*sizeof(double), ba0 = (size_t)n*36*sizeof(double),
                 bA = (size_t)nfr*n*18*sizeof(double);
    if(brv > crv){ if(drv) cudaFree(drv); drv = nullptr; cudaMalloc(&drv, brv); crv = brv; }
    if(ba0 > ca0){ if(da0) cudaFree(da0); da0 = nullptr; cudaMalloc(&da0, ba0); ca0 = ba0; }
    if(bA > cA){ if(dA) cudaFree(dA); dA = nullptr; cudaMalloc(&dA, bA); cA = bA; }

    _t = _now();
    cudaMemcpy(drv, hrv.data(), brv, cudaMemcpyHostToDevice);
    cudaMemcpy(da0, ha0.data(), ba0, cudaMemcpyHostToDevice);
    g_gve_h2d += _now() - _t;

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    _t = _now();
    stmFoldKernel<<<blocks, threads>>>(drv, n, nfr, dt, da0, dA);
    cudaError_t err = cudaDeviceSynchronize();
    g_gve_stm += _now() - _t;
    if(err != cudaSuccess)
        fprintf(stderr, "[stmFoldGpuBatch] kernel error: %s\n", cudaGetErrorString(err));

    _t = _now();
    cudaMemcpy(A_all.data(), dA, bA, cudaMemcpyDeviceToHost);
    g_gve_d2h += _now() - _t;
}

// 笛卡尔单步（GPU，全动力学）：x0(n,6) → xf。Jac 用同一两体 STM（stmFoldGpuBatch）。
void cartStepBatch(const std::vector<Vector6d> &x0s, double dt, int nthreads, double beta,
                   std::vector<Vector6d> &xf){
    (void)nthreads;
    using clk = std::chrono::steady_clock;
    auto _now = []{ return std::chrono::duration<double>(clk::now().time_since_epoch()).count(); };
    const int n = (int)x0s.size();
    xf.assign(n, Vector6d::Zero());
    if(n == 0) return;
    static thread_local std::vector<double> h0, hf;
    h0.resize((size_t)n*6); hf.resize((size_t)n*6);
    double _t = _now();
    for(int i = 0; i < n; ++i)
        for(int c = 0; c < 6; ++c) h0[(size_t)i*6 + c] = x0s[i](c);
    g_gve_pack += _now() - _t;
    static thread_local double *d0 = nullptr, *df = nullptr; static thread_local size_t cap = 0;
    const size_t bytes = (size_t)n*6*sizeof(double);
    if(bytes > cap){ if(d0) cudaFree(d0); if(df) cudaFree(df); d0 = df = nullptr;
                     cudaMalloc(&d0, bytes); cudaMalloc(&df, bytes); cap = bytes; }
    _t = _now(); cudaMemcpy(d0, h0.data(), bytes, cudaMemcpyHostToDevice); g_gve_h2d += _now() - _t;
    const int threads = 256, blocks = (n + threads - 1) / threads;
    _t = _now(); cartStepKernel<<<blocks, threads>>>(d0, n, dt, beta, df);
    cudaError_t err = cudaDeviceSynchronize(); g_gve_kernel += _now() - _t;
    if(err != cudaSuccess) fprintf(stderr, "[cartStepBatch] kernel error: %s\n", cudaGetErrorString(err));
    _t = _now(); cudaMemcpy(hf.data(), df, bytes, cudaMemcpyDeviceToHost);
    for(int i = 0; i < n; ++i) for(int c = 0; c < 6; ++c) xf[i](c) = hf[(size_t)i*6 + c];
    g_gve_d2h += _now() - _t;
}

// 笛卡尔整弧多帧（GPU，全动力学）：x0(n,6) → x_all（平铺 nfr*n*6）。
void cartPropagateBatch(const std::vector<Vector6d> &x0s, int nfr, double dt, int nthreads,
                        double beta, std::vector<double> &x_all){
    (void)nthreads;
    using clk = std::chrono::steady_clock;
    auto _now = []{ return std::chrono::duration<double>(clk::now().time_since_epoch()).count(); };
    const int n = (int)x0s.size();
    x_all.assign((size_t)nfr * n * 6, 0.0);
    if(n == 0 || nfr <= 0) return;
    static thread_local std::vector<double> h0;
    h0.resize((size_t)n*6);
    double _t = _now();
    for(int i = 0; i < n; ++i)
        for(int c = 0; c < 6; ++c) h0[(size_t)i*6 + c] = x0s[i](c);
    g_gve_pack += _now() - _t;
    static thread_local double *d0 = nullptr, *dx = nullptr; static thread_local size_t c0 = 0, cx = 0;
    const size_t b0 = (size_t)n*6*sizeof(double), bx = (size_t)nfr*n*6*sizeof(double);
    if(b0 > c0){ if(d0) cudaFree(d0); d0 = nullptr; cudaMalloc(&d0, b0); c0 = b0; }
    if(bx > cx){ if(dx) cudaFree(dx); dx = nullptr; cudaMalloc(&dx, bx); cx = bx; }
    _t = _now(); cudaMemcpy(d0, h0.data(), b0, cudaMemcpyHostToDevice); g_gve_h2d += _now() - _t;
    const int threads = 256, blocks = (n + threads - 1) / threads;
    _t = _now(); cartPropagateKernel<<<blocks, threads>>>(d0, n, nfr, dt, beta, dx);
    cudaError_t err = cudaDeviceSynchronize(); g_gve_kernel += _now() - _t;
    if(err != cudaSuccess) fprintf(stderr, "[cartPropagateBatch] kernel error: %s\n", cudaGetErrorString(err));
    _t = _now(); cudaMemcpy(x_all.data(), dx, bx, cudaMemcpyDeviceToHost); g_gve_d2h += _now() - _t;
}
} // namespace kep3

// ===================== qoejopt：joint 装配（device 常驻） =====================
// 复用本 TU 的 device 助手（oe2rv_dev/gveRhs_dev/stmTwoBodyDev）。A_val/b 由 Python 提供
// torch cuda 缓冲指针直写，P/A 中间量只驻显存。布局与 host `_lift_struct`/`_lift_assemble` 严格一致：
//   ISL 行 r=f*E+e：12 非零 [sat_i 的 6，sat_j 的 6]，b[r] = -(rr-D)*wi
//   GTS 行 r=f*G+a：6 非零，b[nISL+r] = -(rg-D)*wg
//   damp 行 c：1 非零（对角 issq），b[nISL+nGTS+c] = -issq*(X[c]-ctr[c])
namespace qoejopt {

double jt_expand = 0.0, jt_stm = 0.0, jt_asm = 0.0, jt_res2 = 0.0;
void jointResetTiming() { jt_expand = jt_stm = jt_asm = jt_res2 = 0.0; }
std::vector<double> jointGetTiming() { return { jt_expand, jt_stm, jt_asm, jt_res2 }; }
static inline double _jnow() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
}

__global__ void jointIslAsmKernel(const double *d_P, const double *d_A,
                                  const int *isl_i, const int *isl_jj,
                                  const double *isl_D, const double *isl_W,
                                  int n, int nfr, int E, double s_isl, int off_isl,
                                  double *A_val, double *b) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int)((size_t)nfr * E)) return;
    const int f = idx / E, e = idx % E;
    const int i = isl_i[idx], j = isl_jj[idx];
    const double *pi = d_P + ((size_t)f * n + i) * 6;
    const double *pj = d_P + ((size_t)f * n + j) * 6;
    const double d0 = pi[0] - pj[0], d1 = pi[1] - pj[1], d2 = pi[2] - pj[2];
    const double rr = sqrt(d0 * d0 + d1 * d1 + d2 * d2);
    const double ir = (rr > 1e-12) ? 1.0 / rr : 0.0;
    const double u0 = d0 * ir, u1 = d1 * ir, u2 = d2 * ir;
    const double *Ai = d_A + ((size_t)f * n + i) * 18;
    const double *Aj = d_A + ((size_t)f * n + j) * 18;
    const double wit = isl_W[idx] * s_isl;
    double *av = A_val + off_isl + (size_t)idx * 12;
#pragma unroll
    for (int k = 0; k < 6; ++k) {
        av[k]     =  wit * (u0 * Ai[0 * 6 + k] + u1 * Ai[1 * 6 + k] + u2 * Ai[2 * 6 + k]);
        av[6 + k] = -wit * (u0 * Aj[0 * 6 + k] + u1 * Aj[1 * 6 + k] + u2 * Aj[2 * 6 + k]);
    }
    b[idx] = -(rr - isl_D[idx]) * wit;
}

__global__ void jointGtsAsmKernel(const double *d_P, const double *d_A,
                                  const int *gts_a, const double *gts_D,
                                  const double *gts_W, const double *gts_Q,
                                  int n, int G, int nGTS, double s_gts, int nISL,
                                  int off_gts, double *A_val, double *b) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nGTS) return;
    const int f = idx / G;
    const int a = gts_a[idx];
    const double *pf = d_P + ((size_t)f * n + a) * 6;
    const double *q = gts_Q + (size_t)idx * 3;
    const double d0 = pf[0] - q[0], d1 = pf[1] - q[1], d2 = pf[2] - q[2];
    const double rg = sqrt(d0 * d0 + d1 * d1 + d2 * d2);
    const double ir = (rg > 1e-12) ? 1.0 / rg : 0.0;
    const double u0 = d0 * ir, u1 = d1 * ir, u2 = d2 * ir;
    const double *Ag = d_A + ((size_t)f * n + a) * 18;
    const double wg = gts_W[idx] * s_gts;
    double *av = A_val + off_gts + (size_t)idx * 6;
#pragma unroll
    for (int k = 0; k < 6; ++k)
        av[k] = wg * (u0 * Ag[0 * 6 + k] + u1 * Ag[1 * 6 + k] + u2 * Ag[2 * 6 + k]);
    b[nISL + idx] = -(rg - gts_D[idx]) * wg;
}

__global__ void jointDampKernel(const double *d_X, const double *ctr, int ncol,
                                double issq, int off_damp, int base_b,
                                double *A_val, double *b) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncol) return;
    A_val[off_damp + c] = issq;
    b[base_b + c] = -issq * (d_X[c] - ctr[c]);
}

__global__ void jointRes2Kernel(const double *d_P, const int *isl_i, const int *isl_jj,
                                const double *isl_D, const double *isl_W,
                                int n, int E, int nISL, double *acc) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nISL) return;
    if (isl_W[idx] <= 0.0) return;
    const int f = idx / E;
    const int i = isl_i[idx], j = isl_jj[idx];
    const double *pi = d_P + ((size_t)f * n + i) * 6;
    const double *pj = d_P + ((size_t)f * n + j) * 6;
    const double d0 = pi[0] - pj[0], d1 = pi[1] - pj[1], d2 = pi[2] - pj[2];
    const double rr = sqrt(d0 * d0 + d1 * d1 + d2 * d2);
    const double r = rr - isl_D[idx];
    atomicAdd(acc, r * r);
}

void jointExpandDevice(Ctx &c) {
    const int threads = 256;
    const int blocks = (c.n + threads - 1) / threads;
    const double t0 = _jnow();
    gvePropagateKernel<<<blocks, threads>>>(c.d_X, c.n, c.nfr, c.dt, 1.0, c.d_P);
    cudaError_t err = cudaDeviceSynchronize();
    jt_expand += _jnow() - t0;
    if (err != cudaSuccess)
        fprintf(stderr, "[qoejopt::jointExpandDevice] %s\n", cudaGetErrorString(err));
}

void jointStmFoldDevice(Ctx &c) {
    const int threads = 256;
    const int blocks = (c.n + threads - 1) / threads;
    const double t0 = _jnow();
    stmFoldKernel<<<blocks, threads>>>(c.d_rv0, c.n, c.nfr, c.dt, c.d_A0, c.d_A);
    cudaError_t err = cudaDeviceSynchronize();
    jt_stm += _jnow() - t0;
    if (err != cudaSuccess)
        fprintf(stderr, "[qoejopt::jointStmFoldDevice] %s\n", cudaGetErrorString(err));
}

static double jointRes2Impl(Ctx &c) {
    const int nISL = c.nfr * c.E;
    if (nISL <= 0) return 0.0;
    const double t0 = _jnow();
    cudaMemset(c.d_res, 0, sizeof(double));
    const int threads = 256;
    const int blocks = (nISL + threads - 1) / threads;
    jointRes2Kernel<<<blocks, threads>>>(c.d_P, c.d_isl_i, c.d_isl_jj, c.d_isl_D,
                                         c.d_isl_W, c.n, c.E, nISL, c.d_res);
    double h = 0.0;
    cudaMemcpy(&h, c.d_res, sizeof(double), cudaMemcpyDeviceToHost);
    jt_res2 += _jnow() - t0;
    return h;
}

static void _asmKernels(Ctx &c, double *A_val, double *b) {
    if (!(A_val && b)) return;
    const int threads = 256;
    const int nISL = c.nfr * c.E;
    const int nGTS = c.nfr * c.G;
    if (nISL > 0) {
        const int blocks = (nISL + threads - 1) / threads;
        jointIslAsmKernel<<<blocks, threads>>>(c.d_P, c.d_A, c.d_isl_i, c.d_isl_jj,
                                               c.d_isl_D, c.d_isl_W, c.n, c.nfr, c.E,
                                               c.s_isl, c.off_isl, A_val, b);
    }
    if (nGTS > 0) {
        const int blocks = (nGTS + threads - 1) / threads;
        jointGtsAsmKernel<<<blocks, threads>>>(c.d_P, c.d_A, c.d_gts_a, c.d_gts_D,
                                               c.d_gts_W, c.d_gts_Q, c.n, c.G, nGTS,
                                               c.s_gts, nISL, c.off_gts, A_val, b);
    }
    if (c.ncol > 0) {
        const int blocks = (c.ncol + threads - 1) / threads;
        jointDampKernel<<<blocks, threads>>>(c.d_X, c.d_ctr, c.ncol, c.issq,
                                             c.off_damp, nISL + nGTS, A_val, b);
    }
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        fprintf(stderr, "[qoejopt::_asmKernels] %s\n", cudaGetErrorString(err));
}

double jointAssembleInto(Ctx &c, double *A_val, double *b) {
    jointExpandDevice(c);
    jointStmFoldDevice(c);
    const double t_asm0 = _jnow();
    _asmKernels(c, A_val, b);
    jt_asm += _jnow() - t_asm0;
    return jointRes2Impl(c);
}

double jointAssembleCachedInto(Ctx &c, double *A_val, double *b) {
    const double t_asm0 = _jnow();
    _asmKernels(c, A_val, b);
    jt_asm += _jnow() - t_asm0;
    return jointRes2Impl(c);
}

double jointIslRes2(Ctx &c) {
    jointExpandDevice(c);
    jointStmFoldDevice(c);
    return jointRes2Impl(c);
}

void jointCopyOut(const Ctx &c, double *host_rv) {
    cudaMemcpy(host_rv, c.d_P, (size_t)c.nfr * c.n * 6 * sizeof(double),
               cudaMemcpyDeviceToHost);
}

// ---- P4.2 前向敏度：S_f = ∂oe_f/∂κ（变分方程 + 同一 3/8-RK4）----
// A=∂f/∂oe、Fκ=∂f/∂κ 用设备端中心差分（可后续替换为解析式，接口不变；β 只乘阻力项、对其线性，故 Fκ 精确）。
__device__ __forceinline__ void gveRhsJacFD_dev(const double *OE, double beta, double A[6][6]) {
    // 中心差分（2 阶）；继续提精度需解析 ∂f/∂oe（接口不变）。
    double fp[6], fm[6];
    for (int j = 0; j < 6; ++j) {
        const double hh = 1e-7 * (fabs(OE[j]) + 1.0);
        double xp[6], xm[6];
        #pragma unroll
        for (int k = 0; k < 6; ++k) { xp[k] = OE[k]; xm[k] = OE[k]; }
        xp[j] += hh; xm[j] -= hh;
        gveRhs_dev(xp, beta, fp);
        gveRhs_dev(xm, beta, fm);
        const double inv = 1.0 / (2.0 * hh);
        for (int i = 0; i < 6; ++i) A[i][j] = (fp[i] - fm[i]) * inv;
    }
}
__device__ __forceinline__ void gveRhsKappaFD_dev(const double *OE, double beta, double Fk[6]) {
    const double hb = 1e-6;
    double fp[6], fm[6];
    gveRhs_dev(OE, beta + hb, fp);
    gveRhs_dev(OE, beta - hb, fm);
    #pragma unroll
    for (int i = 0; i < 6; ++i) Fk[i] = (fp[i] - fm[i]) / (2.0 * hb);
}
__device__ __forceinline__ void sensRhs_dev(const double *oe, const double *S, double beta,
                                            double *doe, double *dS) {
    gveRhs_dev(oe, beta, doe);
    double A[6][6]; gveRhsJacFD_dev(oe, beta, A);
    double Fk[6];   gveRhsKappaFD_dev(oe, beta, Fk);
    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        double s = Fk[i];
        for (int j = 0; j < 6; ++j) s += A[i][j] * S[j];
        dS[i] = s;
    }
}
__global__ void gveSensPropagateKernel(const double *oe0, int n, int nfr, double dt, double beta,
                                       double *oe_all, double *S_all) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    double y[6], S[6];
    #pragma unroll
    for (int c = 0; c < 6; ++c) { y[c] = oe0[6 * idx + c]; S[c] = 0.0; }
    #pragma unroll
    for (int c = 0; c < 6; ++c) {
        oe_all[((size_t)0 * n + idx) * 6 + c] = y[c];
        S_all[((size_t)0 * n + idx) * 6 + c] = S[c];
    }
    double k1y[6], k2y[6], k3y[6], k4y[6], s1[6], s2[6], s3[6], s4[6], ty[6], ts[6];
    for (int f = 1; f < nfr; ++f) {
        sensRhs_dev(y, S, beta, k1y, s1);
        #pragma unroll
        for (int c = 0; c < 6; ++c) { ty[c] = y[c] + dt * k1y[c] / 3.0; ts[c] = S[c] + dt * s1[c] / 3.0; }
        sensRhs_dev(ty, ts, beta, k2y, s2);
        #pragma unroll
        for (int c = 0; c < 6; ++c) { ty[c] = y[c] + dt * (-k1y[c] / 3.0 + k2y[c]); ts[c] = S[c] + dt * (-s1[c] / 3.0 + s2[c]); }
        sensRhs_dev(ty, ts, beta, k3y, s3);
        #pragma unroll
        for (int c = 0; c < 6; ++c) { ty[c] = y[c] + dt * (k1y[c] - k2y[c] + k3y[c]); ts[c] = S[c] + dt * (s1[c] - s2[c] + s3[c]); }
        sensRhs_dev(ty, ts, beta, k4y, s4);
        #pragma unroll
        for (int c = 0; c < 6; ++c) {
            y[c] += dt * (k1y[c] + 3.0 * k2y[c] + 3.0 * k3y[c] + k4y[c]) / 8.0;
            S[c] += dt * (s1[c] + 3.0 * s2[c] + 3.0 * s3[c] + s4[c]) / 8.0;
        }
        #pragma unroll
        for (int c = 0; c < 6; ++c) {
            oe_all[((size_t)f * n + idx) * 6 + c] = y[c];
            S_all[((size_t)f * n + idx) * 6 + c] = S[c];
        }
    }
}

void jointStateSensBatch(const Ctx &c, const double *x_host, double beta, double *oe_all, double *S_all) {
    const int n = c.n, nfr = c.nfr;
    if (n <= 0 || nfr <= 0) return;
    const size_t bx = (size_t)n * 6 * sizeof(double), bs = (size_t)nfr * n * 6 * sizeof(double);
    double *dx = nullptr, *doe = nullptr, *dS = nullptr;
    cudaMalloc(&dx, bx); cudaMalloc(&doe, bs); cudaMalloc(&dS, bs);
    cudaMemcpy(dx, x_host, bx, cudaMemcpyHostToDevice);
    const int threads = 256, blocks = (n + threads - 1) / threads;
    const double t0 = _jnow();
    gveSensPropagateKernel<<<blocks, threads>>>(dx, n, nfr, c.dt, 1.0, doe, dS);
    cudaError_t err = cudaDeviceSynchronize();
    jt_expand += _jnow() - t0;
    if (err != cudaSuccess) fprintf(stderr, "[jointStateSensBatch] %s\n", cudaGetErrorString(err));
    cudaMemcpy(oe_all, doe, bs, cudaMemcpyDeviceToHost);
    cudaMemcpy(S_all, dS, bs, cudaMemcpyDeviceToHost);
    cudaFree(dx); cudaFree(doe); cudaFree(dS);
}

// ---- 单步 GVE + 全动力学变分 STM：Φ_oe = ∂oe(dt)/∂oe(0)，与状态同 3/8-RK4（供 EKF-qoe 预测）----
__global__ void gveStmKernel(const double *oe0, int n, double dt, double beta,
                             double *oef, double *Phi_all) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    double y[6], k1[6], k2[6], k3[6], k4[6], tmp[6];
    double P[6][6], K1[6][6], K2[6][6], K3[6][6], K4[6][6], A[6][6];
    #pragma unroll
    for (int c = 0; c < 6; ++c) y[c] = oe0[6 * idx + c];
    #pragma unroll
    for (int a = 0; a < 6; ++a)
        #pragma unroll
        for (int b = 0; b < 6; ++b) P[a][b] = (a == b) ? 1.0 : 0.0;
    gveRhs_dev(y, beta, k1); gveRhsJacFD_dev(y, beta, A);
    for (int a = 0; a < 6; ++a) for (int b = 0; b < 6; ++b) { double s = 0; for (int c = 0; c < 6; ++c) s += A[a][c] * P[c][b]; K1[a][b] = s; }
    #pragma unroll
    for (int c = 0; c < 6; ++c) tmp[c] = y[c] + dt * k1[c] / 3.0;
    gveRhs_dev(tmp, beta, k2); gveRhsJacFD_dev(tmp, beta, A);
    for (int a = 0; a < 6; ++a) for (int b = 0; b < 6; ++b) { double s = 0; for (int c = 0; c < 6; ++c) s += A[a][c] * (P[c][b] + dt * K1[c][b] / 3.0); K2[a][b] = s; }
    #pragma unroll
    for (int c = 0; c < 6; ++c) tmp[c] = y[c] + dt * (-k1[c] / 3.0 + k2[c]);
    gveRhs_dev(tmp, beta, k3); gveRhsJacFD_dev(tmp, beta, A);
    for (int a = 0; a < 6; ++a) for (int b = 0; b < 6; ++b) { double s = 0; for (int c = 0; c < 6; ++c) s += A[a][c] * (P[c][b] + dt * (-K1[c][b] / 3.0 + K2[c][b])); K3[a][b] = s; }
    #pragma unroll
    for (int c = 0; c < 6; ++c) tmp[c] = y[c] + dt * (k1[c] - k2[c] + k3[c]);
    gveRhs_dev(tmp, beta, k4); gveRhsJacFD_dev(tmp, beta, A);
    for (int a = 0; a < 6; ++a) for (int b = 0; b < 6; ++b) { double s = 0; for (int c = 0; c < 6; ++c) s += A[a][c] * (P[c][b] + dt * (K1[c][b] - K2[c][b] + K3[c][b])); K4[a][b] = s; }
    double *of = oef + 6 * idx; double *Pa = Phi_all + (size_t)idx * 36;
    #pragma unroll
    for (int c = 0; c < 6; ++c) of[c] = y[c] + dt * (k1[c] + 3.0 * k2[c] + 3.0 * k3[c] + k4[c]) / 8.0;
    for (int a = 0; a < 6; ++a) for (int b = 0; b < 6; ++b)
        Pa[a * 6 + b] = P[a][b] + dt * (K1[a][b] + 3.0 * K2[a][b] + 3.0 * K3[a][b] + K4[a][b]) / 8.0;
}
void gveStmDevice(const double *d_oe, int n, double dt, double beta, double *d_oef, double *d_Phi) {
    const int threads = 256, blocks = (n + threads - 1) / threads;
    gveStmKernel<<<blocks, threads>>>(d_oe, n, dt, beta, d_oef, d_Phi);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) fprintf(stderr, "[gveStmDevice] %s\n", cudaGetErrorString(err));
}

}  // namespace qoejopt
