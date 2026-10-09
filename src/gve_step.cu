// GVE（QOE 非奇异要素）单步 RK4 —— CUDA 版，对整星座 N 并行。
//
// 物理与 `src/dastate.cpp` 的 `gveRhsQOE<double>` / `noeGveRhs` / `OEOsc2rvT<double>` /
// `TBPfull` 逐式一致（二体 + J2/J3/J4 + 阻力；三体在 dastate 里默认关，此处同）。
// 积分用与 `rk4<Vector6d>` 相同的 **3/8 规则**（不是经典 1/6,2/6,2/6,1/6）。
// 无 DA、无高阶展开；每帧一次 `dt` 单步，链式得到整弧状态。
#include <cuda_runtime.h>
#include <vector>
#include <array>
#include <cmath>
#include <cstdio>
#include <chrono>
#include <stdexcept>
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

// ---- 位置相关 RBF 残差力场（整星共享；与 dastate.cpp basisAccel/fieldResidualAccel 同式）----
// 基 a_k(r) = (w/s^2)(r-c_k) [km/s^2 / 单位 θ]，w=exp(-|r-c_k|^2/(2 s^2))；a_res=Σ_k θ_k a_k。
// 中心/宽度/个数放 constant（一次设定），权重 θ 放 device 全局（每外环迭代更新）。
// θ 力场**可插拔**：c_field_kind 选择形式（RBF 已实现；SH/其它预留——加分支即接入）。
// 统一接口（供状态传播与 θ 敏度核共用）：
//   fieldAccelKmDev(r_km, a3)       ：a_res(r;θ)=Σ_k θ_k b_k(r)，km/s²
//   fieldBasisKmDev(r_km, k, a3)    ：第 k 基 b_k(r)=∂a_res/∂θ_k，km/s²/单位 θ
// 数据：θ 权重 g_rbf_w（长度 m）；RBF 的中心/宽度 c_rbf_C/c_rbf_s2。
#define RBF_MAX 256
#define FIELD_KIND_RBF 0
#define FIELD_KIND_SH  1                         // 预留（球谐；参数面 c_sh_lmax 等）
__constant__ int    c_field_kind = FIELD_KIND_RBF;
__constant__ double c_rbf_s2 = 1.0;          // s^2（km^2）
__constant__ double c_rbf_cut2 = 1e300;      // 局部性裁剪阈值 r^2（km^2）；r^2>此值 → 基exp≈0 跳过
__constant__ int    c_rbf_m  = 0;            // 基函数个数（0 = 关闭）
__constant__ double c_rbf_C[3 * RBF_MAX];    // 中心（km）
__device__   double *g_rbf_w = nullptr;      // 权重（θ，长度 m）

__device__ __forceinline__ void rbfAccelKmDev(const double *r_km, double *a3){
    a3[0] = 0.0; a3[1] = 0.0; a3[2] = 0.0;
    if(c_rbf_m <= 0 || g_rbf_w == nullptr) return;
    const double inv2s2 = 0.5 / c_rbf_s2;
    for(int k = 0; k < c_rbf_m; ++k){
        const double dx = r_km[0] - c_rbf_C[3*k], dy = r_km[1] - c_rbf_C[3*k+1],
                     dz = r_km[2] - c_rbf_C[3*k+2];
        const double r2 = dx*dx + dy*dy + dz*dz;
        if(r2 > c_rbf_cut2) continue;            // 局部性裁剪：远离的基 exp≈0，跳过
        const double wv = exp(-r2 * inv2s2) / c_rbf_s2;
        const double wk = g_rbf_w[k];
        a3[0] += wk*wv*dx; a3[1] += wk*wv*dy; a3[2] += wk*wv*dz;
    }
}

// 第 k 个 RBF 基的加速度 a_k=(w/s²)(r-c_k) [km/s² / 单位 θ]；供敏度核 ∂f/∂θ_k 用。
__device__ __forceinline__ void rbfBasisKmDev(const double *r_km, int k, double *a3){
    a3[0] = 0.0; a3[1] = 0.0; a3[2] = 0.0;
    if(c_rbf_m <= 0 || k < 0 || k >= c_rbf_m) return;
    const double dx = r_km[0] - c_rbf_C[3*k], dy = r_km[1] - c_rbf_C[3*k+1],
                 dz = r_km[2] - c_rbf_C[3*k+2];
    const double r2 = dx*dx + dy*dy + dz*dz;
    if(r2 > c_rbf_cut2) return;                  // 局部性裁剪
    const double wv = exp(-r2 * (0.5 / c_rbf_s2)) / c_rbf_s2;
    a3[0] = wv*dx; a3[1] = wv*dy; a3[2] = wv*dz;
}

// ---- θ 力场统一接口（可插拔）：新增形式只在此加分支，其余代码不变 ----
__device__ __forceinline__ void fieldBasisKmDev(const double *r_km, int k, double *a3){   // b_k=∂a_res/∂θ_k
    a3[0] = 0.0; a3[1] = 0.0; a3[2] = 0.0;
    if(c_field_kind == FIELD_KIND_RBF) rbfBasisKmDev(r_km, k, a3);
    // else if(c_field_kind == FIELD_KIND_SH) shBasisKmDev(r_km, k, a3);   // 预留
}
__device__ __forceinline__ void fieldAccelKmDev(const double *r_km, double *a3){           // a_res=Σ_k θ_k b_k
    a3[0] = 0.0; a3[1] = 0.0; a3[2] = 0.0;
    if(c_field_kind == FIELD_KIND_RBF) rbfAccelKmDev(r_km, a3);
    // else if(c_field_kind == FIELD_KIND_SH) shAccelKmDev(r_km, a3);       // 预留
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

    // θ 残差力场（若 c_rbf_m>0）：km/s^2 加到加速度行（可插拔，见 fieldAccelKmDev）
    double ar[3]; fieldAccelKmDev(xk, ar);
    res[3] += ar[0]; res[4] += ar[1]; res[5] += ar[2];
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
// 由给定**去二体摄动加速度** ap(m/s²) 得 QOE 的 Gauss 变分 RHS（与摄动来源无关；
// gveRhs_dev 供状态传播用，敏度核用同一函数取 ∂f/∂θ_k = RHS(ap=基加速度)）。
__device__ __forceinline__ void gveRhsFromAp_dev(const double* OE, const double* ap, double* out){
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

// 摄动源 -> GVE 速率，**不含** dM 里的 Kepler 平运动 n=√(MU/a³)：用于 ∂f/∂θ（forcing）。
// 因 f = G(x)R(x)a_p + [0,n,0,0,0,0] 且 n 与摄动源无关，故 ∂f/∂(摄动源) 就是本函数；
// 直接用 gveRhsFromAp_dev 会把常数 n 误当进 ∂f/∂κ（实测多出 1.08e-3 = n，致敏度发散）。
__device__ __forceinline__ void gvePertRhsFromAp_dev(const double* OE, const double* ap, double* out){
    gveRhsFromAp_dev(OE, ap, out);
    const double a = OE[0];
    out[1] -= sqrt(MU_M / (a*a*a));
}

// W = ∂f/∂a_p（6×3，Gauss 映射含 RTN 框架）——每帧算一次；于是 ∂f/∂θ_k = W·(b_k·1e3)，
// 避免对每个基 k 重跑 oe2rv/Kepler/RTN（RBF 敏度核的瓶颈）。f = B·R·a_p + [0,n,0,0,0,0]。
__device__ __forceinline__ void gveGaussW_dev(const double* OE, double W[6][3]){
    static const double e1[3] = {1.0, 0.0, 0.0}, e2[3] = {0.0, 1.0, 0.0}, e3[3] = {0.0, 0.0, 1.0};
    double o1[6], o2[6], o3[6];
    gveRhsFromAp_dev(OE, e1, o1);
    gveRhsFromAp_dev(OE, e2, o2);
    gveRhsFromAp_dev(OE, e3, o3);
    const double a = OE[0];
    const double n = sqrt(MU_M / (a*a*a));
    #pragma unroll
    for(int i = 0; i < 6; ++i){ W[i][0] = o1[i]; W[i][1] = o2[i]; W[i][2] = o3[i]; }
    W[1][0] -= n; W[1][1] -= n; W[1][2] -= n;
}

// 全动力学 GVE RHS：ap = (J234+阻力+RBF) 去二体摄动（m/s²）。状态传播用。
__device__ __forceinline__ void gveRhs_dev(const double* OE, double beta, double* out){
    double rv[6]; oe2rv_dev(OE, rv);
    double ap[3]; pertAccel_dev(rv, beta, ap);
    gveRhsFromAp_dev(OE, ap, out);
}

// ============================================================================
// 解析 Jacobian A=∂f/∂oe 与 ∂f/∂θ：**前向对偶数**（值 + 6 个 ∂/∂oe），
// 逐式镜像上面的 double 物理；无有限差分、无 12× 重复求值（这正是 nvcc 卡死的根源）。
// FD 版只保留在 Python 侧做校验（validate_*），设备端不再有 FD。
// ============================================================================
struct Dual6 { double v; double d[6]; };
__device__ __forceinline__ Dual6 d6_c(double v){ Dual6 r; r.v=v;
    #pragma unroll
    for(int i=0;i<6;i++) r.d[i]=0.0; return r; }
__device__ __forceinline__ Dual6 d6_var(double v,int j){ Dual6 r; r.v=v;
    #pragma unroll
    for(int i=0;i<6;i++) r.d[i]=0.0; r.d[j]=1.0; return r; }
__device__ __forceinline__ Dual6 d6_add(const Dual6&x,const Dual6&y){ Dual6 r; r.v=x.v+y.v;
    #pragma unroll
    for(int i=0;i<6;i++) r.d[i]=x.d[i]+y.d[i]; return r; }
__device__ __forceinline__ Dual6 d6_sub(const Dual6&x,const Dual6&y){ Dual6 r; r.v=x.v-y.v;
    #pragma unroll
    for(int i=0;i<6;i++) r.d[i]=x.d[i]-y.d[i]; return r; }
__device__ __forceinline__ Dual6 d6_mul(const Dual6&x,const Dual6&y){ Dual6 r; r.v=x.v*y.v;
    #pragma unroll
    for(int i=0;i<6;i++) r.d[i]=x.d[i]*y.v+x.v*y.d[i]; return r; }
__device__ __forceinline__ Dual6 d6_div(const Dual6&x,const Dual6&y){ Dual6 r; const double iv=1.0/y.v;
    r.v=x.v*iv;
    #pragma unroll
    for(int i=0;i<6;i++) r.d[i]=(x.d[i]-r.v*y.d[i])*iv; return r; }
__device__ __forceinline__ Dual6 d6_sqrt(const Dual6&x){ Dual6 r; r.v=sqrt(x.v); const double h=0.5/r.v;
    #pragma unroll
    for(int i=0;i<6;i++) r.d[i]=h*x.d[i]; return r; }
__device__ __forceinline__ Dual6 d6_sin(const Dual6&x){ Dual6 r; const double c=cos(x.v); r.v=sin(x.v);
    #pragma unroll
    for(int i=0;i<6;i++) r.d[i]=c*x.d[i]; return r; }
__device__ __forceinline__ Dual6 d6_cos(const Dual6&x){ Dual6 r; const double s=-sin(x.v); r.v=cos(x.v);
    #pragma unroll
    for(int i=0;i<6;i++) r.d[i]=s*x.d[i]; return r; }
__device__ __forceinline__ Dual6 d6_tan(const Dual6&x){ Dual6 r; const double c=cos(x.v); r.v=tan(x.v); const double g=1.0/(c*c);
    #pragma unroll
    for(int i=0;i<6;i++) r.d[i]=g*x.d[i]; return r; }
__device__ __forceinline__ Dual6 d6_atan(const Dual6&x){ Dual6 r; r.v=atan(x.v); const double g=1.0/(1.0+x.v*x.v);
    #pragma unroll
    for(int i=0;i<6;i++) r.d[i]=g*x.d[i]; return r; }
__device__ __forceinline__ Dual6 d6_exp(const Dual6&x){ Dual6 r; r.v=exp(x.v);
    #pragma unroll
    for(int i=0;i<6;i++) r.d[i]=r.v*x.d[i]; return r; }
__device__ __forceinline__ Dual6 d6_atan2(const Dual6&y,const Dual6&x){ const double den=x.v*x.v+y.v*y.v;
    Dual6 r; r.v=atan2(y.v,x.v);
    #pragma unroll
    for(int i=0;i<6;i++) r.d[i]=(x.v*y.d[i]-y.v*x.d[i])/den; return r; }
__device__ __forceinline__ Dual6 d6_floor(const Dual6&x){ return d6_c(floor(x.v)); }

// Kepler M->E 的对偶 Newton（同一迭代，收敛到对偶根）
__device__ __forceinline__ Dual6 kepE_d_dev(Dual6 M, Dual6 e){
    Dual6 E = ((M.v > -PI && M.v < 0.0) || M.v > PI) ? d6_sub(M,e) : d6_add(M,e);
    #pragma unroll 1
    for(int it=0; it<60; ++it){
        Dual6 En = E;
        Dual6 num = d6_add(d6_sub(M,En), d6_mul(e, d6_sin(En)));
        Dual6 den = d6_sub(d6_c(1.0), d6_mul(e, d6_cos(En)));
        E = d6_add(En, d6_div(num, den));
        if(fabs(E.v - En.v) < 1e-13) break;
    }
    return E;
}

// QOE->rv 的对偶版（逐式镜像 oe2rv_dev）
__device__ __forceinline__ void oe2rv_d_dev(const Dual6* OE, Dual6* x){
    Dual6 a=OE[0], u=OE[1], ex=OE[2], ey=OE[3], inc=OE[4], Om=OE[5];
    Dual6 e = d6_sqrt(d6_add(d6_mul(ex,ex), d6_mul(ey,ey)));
    Dual6 p = d6_mul(a, d6_sub(d6_c(1.0), d6_mul(e,e)));
    if(p.v < 1e-6) p = d6_c(1e-6);
    Dual6 omega, nu;
    if(e.v < 1e-5){ omega = d6_c(0.0); nu = u; }
    else {
        omega = d6_atan2(ey, ex);
        Dual6 M = d6_sub(u, omega);
        if(M.v < -PI) M = d6_add(M, d6_mul(d6_floor(d6_div(d6_c(fabs(M.v-PI)), d6_c(2.0*PI))), d6_c(2.0*PI)));
        else if(M.v > PI) M = d6_sub(M, d6_mul(d6_floor(d6_div(d6_add(M, d6_c(PI)), d6_c(2.0*PI))), d6_c(2.0*PI)));
        Dual6 E = kepE_d_dev(M, e);
        Dual6 q = d6_div(d6_add(d6_c(1.0), e), d6_sub(d6_c(1.0), e));
        if(q.v < 0.0) q = d6_c(0.0);
        nu = d6_mul(d6_c(2.0), d6_atan(d6_mul(d6_sqrt(q), d6_tan(d6_mul(d6_c(0.5), E)))));
    }
    Dual6 cnu = d6_cos(nu), snu = d6_sin(nu);
    Dual6 vscale = d6_sqrt(d6_div(d6_c(MU_M), p));
    Dual6 denr = d6_add(d6_c(1.0), d6_mul(e, cnu));
    Dual6 rP0 = d6_div(d6_mul(p, cnu), denr);
    Dual6 rP1 = d6_div(d6_mul(p, snu), denr);
    Dual6 vP0 = d6_sub(d6_c(0.0), d6_mul(vscale, snu));
    Dual6 vP1 = d6_mul(vscale, d6_add(e, cnu));
    Dual6 cO=d6_cos(Om), sO=d6_sin(Om), ci=d6_cos(inc), si=d6_sin(inc), cw=d6_cos(omega), sw=d6_sin(omega);
    Dual6 T[3][3];
    T[0][0]=d6_sub(d6_mul(cO,cw), d6_mul(d6_mul(sO,sw),ci));
    T[0][1]=d6_sub(d6_sub(d6_c(0.0), d6_mul(cO,sw)), d6_mul(d6_mul(sO,cw),ci));
    T[0][2]=d6_mul(sO,si);
    T[1][0]=d6_add(d6_mul(sO,cw), d6_mul(d6_mul(cO,sw),ci));
    T[1][1]=d6_add(d6_sub(d6_c(0.0), d6_mul(sO,sw)), d6_mul(d6_mul(cO,cw),ci));
    T[1][2]=d6_sub(d6_c(0.0), d6_mul(cO,si));
    T[2][0]=d6_mul(sw,si); T[2][1]=d6_mul(cw,si); T[2][2]=ci;
    Dual6 rP[3]={rP0,rP1,d6_c(0.0)}, vP[3]={vP0,vP1,d6_c(0.0)};
    #pragma unroll
    for(int r=0;r<3;r++){
        x[r]   = d6_add(d6_add(d6_mul(T[r][0],rP[0]), d6_mul(T[r][1],rP[1])), d6_mul(T[r][2],rP[2]));
        x[3+r] = d6_add(d6_add(d6_mul(T[r][0],vP[0]), d6_mul(T[r][1],vP[1])), d6_mul(T[r][2],vP[2]));
    }
}

// RBF 残差加速度的对偶版（逐式镜像 rbfAccelKmDev）
__device__ __forceinline__ void rbfAccelKm_d_dev(const Dual6* r_km, Dual6* a3){
    Dual6 ax=d6_c(0.0), ay=d6_c(0.0), az=d6_c(0.0);
    if(c_rbf_m <= 0 || g_rbf_w == nullptr){ a3[0]=ax; a3[1]=ay; a3[2]=az; return; }
    for(int k=0;k<c_rbf_m;++k){
        Dual6 dx=d6_sub(r_km[0], d6_c(c_rbf_C[3*k]));
        Dual6 dy=d6_sub(r_km[1], d6_c(c_rbf_C[3*k+1]));
        Dual6 dz=d6_sub(r_km[2], d6_c(c_rbf_C[3*k+2]));
        Dual6 rr=d6_add(d6_add(d6_mul(dx,dx), d6_mul(dy,dy)), d6_mul(dz,dz));
        if(rr.v > c_rbf_cut2) continue;              // 局部性裁剪（远基贡献及其导数≈0）
        Dual6 wv=d6_mul(d6_c(1.0/c_rbf_s2), d6_exp(d6_mul(d6_c(-0.5/c_rbf_s2), rr)));
        Dual6 wk=d6_mul(d6_c(g_rbf_w[k]), wv);
        ax=d6_add(ax, d6_mul(wk,dx)); ay=d6_add(ay, d6_mul(wk,dy)); az=d6_add(az, d6_mul(wk,dz));
    }
    a3[0]=ax; a3[1]=ay; a3[2]=az;
}
__device__ __forceinline__ void fieldAccelKm_d_dev(const Dual6* r_km, Dual6* a3){
    a3[0]=d6_c(0.0); a3[1]=d6_c(0.0); a3[2]=d6_c(0.0);
    if(c_field_kind == FIELD_KIND_RBF) rbfAccelKm_d_dev(r_km, a3);
}

// 二体+J234+阻力+RBF 的对偶版（逐式镜像 tbp_dev）
__device__ __forceinline__ void tbp_d_dev(const Dual6* xk, double beta, Dual6* res){
    Dual6 pos0=xk[0], pos1=xk[1], pos2=xk[2];
    Dual6 r = d6_sqrt(d6_add(d6_add(d6_mul(pos0,pos0), d6_mul(pos1,pos1)), d6_mul(pos2,pos2)));
    Dual6 z_r = d6_div(pos2, r);
    Dual6 z2 = d6_mul(z_r,z_r), z3=d6_mul(z2,z_r), z4=d6_mul(z3,z_r);
    Dual6 Re_r = d6_div(d6_c(RE_KM), r);
    Dual6 Re2=d6_mul(Re_r,Re_r), Re3=d6_mul(Re2,Re_r), Re4=d6_mul(Re3,Re_r);
    Dual6 cf = d6_div(d6_c(-MU_KM), d6_mul(d6_mul(r,r),r));

    Dual6 j2 = d6_mul(d6_mul(d6_c((3.0/2.0)*J2C), Re2), d6_sub(d6_c(1.0), d6_mul(d6_c(5.0), z2)));
    Dual6 j3 = d6_mul(d6_mul(d6_c((5.0/2.0)*J3C), Re3), d6_sub(d6_mul(d6_c(3.0),z_r), d6_mul(d6_c(7.0),z3)));
    Dual6 j4 = d6_mul(d6_mul(d6_c((5.0/8.0)*J4C), Re4), d6_add(d6_sub(d6_c(3.0), d6_mul(d6_c(42.0),z2)), d6_mul(d6_c(63.0),z4)));
    Dual6 fac1 = d6_sub(d6_add(d6_add(d6_c(1.0), j2), j3), j4);
    res[3] = d6_mul(d6_mul(cf, pos0), fac1);
    res[4] = d6_mul(d6_mul(cf, pos1), fac1);

    Dual6 j2b = d6_mul(d6_mul(d6_c((3.0/2.0)*J2C), Re2), d6_sub(d6_c(3.0), d6_mul(d6_c(5.0), z2)));
    Dual6 j3b = d6_mul(d6_mul(d6_c((5.0/2.0)*J3C), Re3),
                       d6_sub(d6_sub(d6_mul(d6_c(6.0), z_r), d6_mul(d6_c(7.0), z3)), d6_mul(d6_c(3.0/5.0), d6_div(r, pos2))));
    Dual6 j4b = d6_mul(d6_mul(d6_c((5.0/8.0)*J4C), Re4), d6_add(d6_sub(d6_c(15.0), d6_mul(d6_c(70.0),z2)), d6_mul(d6_c(63.0),z4)));
    Dual6 fac2 = d6_sub(d6_add(d6_add(d6_c(1.0), j2b), j3b), j4b);
    res[5] = d6_mul(d6_mul(cf, pos2), fac2);

    res[0]=xk[3]; res[1]=xk[4]; res[2]=xk[5];

    Dual6 rvx = d6_add(xk[3], d6_mul(d6_c(OMEGA_EARTH), xk[1]));
    Dual6 rvy = d6_sub(xk[4], d6_mul(d6_c(OMEGA_EARTH), xk[0]));
    Dual6 rvz = xk[5];
    Dual6 v = d6_sqrt(d6_add(d6_add(d6_mul(rvx,rvx), d6_mul(rvy,rvy)), d6_mul(rvz,rvz)));
    Dual6 rho = d6_exp(d6_mul(d6_c(-1.0/H1_KM), d6_sub(r, d6_c(RE_KM+H0_KM))));
    Dual6 dc = d6_mul(d6_mul(d6_c(-0.5*RHO_CDA_M*beta), rho), v);
    res[3]=d6_add(res[3], d6_mul(dc,rvx));
    res[4]=d6_add(res[4], d6_mul(dc,rvy));
    res[5]=d6_add(res[5], d6_mul(dc,rvz));

    Dual6 ar[3]; fieldAccelKm_d_dev(xk, ar);
    res[3]=d6_add(res[3],ar[0]); res[4]=d6_add(res[4],ar[1]); res[5]=d6_add(res[5],ar[2]);
}

// 去二体摄动加速度的对偶版（逐式镜像 pertAccel_dev）
__device__ __forceinline__ void pertAccel_d_dev(const Dual6* rv_m, double beta, Dual6* ap){
    Dual6 xk[6];
    #pragma unroll
    for(int k=0;k<6;k++) xk[k]=d6_mul(rv_m[k], d6_c(1e-3));
    Dual6 acc[6]; tbp_d_dev(xk, beta, acc);
    Dual6 rn = d6_sqrt(d6_add(d6_add(d6_mul(xk[0],xk[0]), d6_mul(xk[1],xk[1])), d6_mul(xk[2],xk[2])));
    Dual6 c = d6_div(d6_c(MU_KM), d6_mul(d6_mul(rn,rn),rn));
    #pragma unroll
    for(int k=0;k<3;k++) ap[k]=d6_mul(d6_add(acc[3+k], d6_mul(c,xk[k])), d6_c(1e3));
}

// Gauss 变分 RHS 的对偶版（逐式镜像 gveRhsFromAp_dev）
__device__ __forceinline__ void gveRhsFromAp_d_dev(const Dual6* OE, const Dual6* ap, Dual6* out){
    Dual6 a=OE[0], u=OE[1], ex=OE[2], ey=OE[3], inc=OE[4];
    Dual6 e = d6_sqrt(d6_add(d6_mul(ex,ex), d6_mul(ey,ey)));
    Dual6 p = d6_mul(a, d6_sub(d6_c(1.0), d6_mul(e,e)));
    if(p.v < 1e-6) p = d6_c(1e-6);
    Dual6 w;
    if(e.v > 0.0) w = d6_atan2(ey, ex); else w = d6_c(0.0);
    Dual6 M = d6_sub(u, w);
    if(M.v < -PI) M = d6_add(M, d6_mul(d6_floor(d6_div(d6_c(fabs(M.v-PI)), d6_c(2.0*PI))), d6_c(2.0*PI)));
    else if(M.v > PI) M = d6_sub(M, d6_mul(d6_floor(d6_div(d6_add(M, d6_c(PI)), d6_c(2.0*PI))), d6_c(2.0*PI)));
    Dual6 E = kepE_d_dev(M, e);
    Dual6 cE = d6_cos(E), sE = d6_sin(E);
    Dual6 dencE = d6_sub(d6_c(1.0), d6_mul(e, cE));
    Dual6 cnu = d6_div(d6_sub(cE, e), dencE);
    Dual6 q = d6_sub(d6_c(1.0), d6_mul(e,e));
    if(q.v < 0.0) q = d6_c(0.0);
    Dual6 snu = d6_div(d6_mul(d6_sqrt(q), sE), dencE);
    Dual6 r = d6_div(p, d6_add(d6_c(1.0), d6_mul(e, cnu)));
    Dual6 h = d6_sqrt(d6_mul(d6_c(MU_M), p));

    Dual6 rv[6]; oe2rv_d_dev(OE, rv);
    Dual6 r3v[3]={rv[0],rv[1],rv[2]}, v3[3]={rv[3],rv[4],rv[5]};
    Dual6 rn = d6_sqrt(d6_add(d6_add(d6_mul(r3v[0],r3v[0]), d6_mul(r3v[1],r3v[1])), d6_mul(r3v[2],r3v[2])));
    Dual6 Rc[3]={d6_div(r3v[0],rn), d6_div(r3v[1],rn), d6_div(r3v[2],rn)};
    Dual6 hv[3]={d6_sub(d6_mul(r3v[1],v3[2]), d6_mul(r3v[2],v3[1])),
                 d6_sub(d6_mul(r3v[2],v3[0]), d6_mul(r3v[0],v3[2])),
                 d6_sub(d6_mul(r3v[0],v3[1]), d6_mul(r3v[1],v3[0]))};
    Dual6 hvn = d6_sqrt(d6_add(d6_add(d6_mul(hv[0],hv[0]), d6_mul(hv[1],hv[1])), d6_mul(hv[2],hv[2])));
    Dual6 Nc[3]={d6_div(hv[0],hvn), d6_div(hv[1],hvn), d6_div(hv[2],hvn)};
    Dual6 Tc[3]={d6_sub(d6_mul(Nc[1],Rc[2]), d6_mul(Nc[2],Rc[1])),
                 d6_sub(d6_mul(Nc[2],Rc[0]), d6_mul(Nc[0],Rc[2])),
                 d6_sub(d6_mul(Nc[0],Rc[1]), d6_mul(Nc[1],Rc[0]))};
    Dual6 aR=d6_c(0.0), aT=d6_c(0.0), aN=d6_c(0.0);
    #pragma unroll
    for(int k=0;k<3;k++){ aR=d6_add(aR,d6_mul(ap[k],Rc[k])); aT=d6_add(aT,d6_mul(ap[k],Tc[k])); aN=d6_add(aN,d6_mul(ap[k],Nc[k])); }

    Dual6 efD  = (e.v < 1e-8) ? d6_c(1e-8) : e;
    Dual6 siD  = d6_sin(inc);
    Dual6 sifD = (siD.v < 1e-8) ? d6_c(1e-8) : siD;
    Dual6 uarg = d6_add(w, d6_atan2(snu, cnu));
    Dual6 da  = d6_mul(d6_div(d6_mul(d6_c(2.0), d6_mul(a,a)), h),
                       d6_add(d6_mul(d6_mul(e,snu),aR), d6_mul(d6_div(p,r), aT)));
    Dual6 de  = d6_mul(d6_div(d6_c(1.0), h),
                       d6_add(d6_mul(d6_mul(p,snu),aR),
                              d6_mul(d6_add(d6_mul(d6_add(p,r),cnu), d6_mul(r,e)), aT)));
    Dual6 di  = d6_mul(d6_div(d6_mul(r, d6_cos(uarg)), h), aN);
    Dual6 dOm = d6_mul(d6_div(d6_mul(r, d6_sin(uarg)), d6_mul(h, sifD)), aN);
    Dual6 dw  = d6_sub(d6_mul(d6_div(d6_c(1.0), d6_mul(h, efD)),
                              d6_add(d6_mul(d6_sub(d6_c(0.0), d6_mul(p,cnu)), aR),
                                     d6_mul(d6_add(p,r), d6_mul(snu,aT)))),
                       d6_mul(d6_div(d6_mul(r, d6_mul(d6_sin(uarg), d6_cos(inc))), d6_mul(h, sifD)), aN));
    Dual6 fM  = d6_div(d6_sqrt(q), d6_mul(h, efD));
    Dual6 dM  = d6_add(d6_sqrt(d6_div(d6_c(MU_M), d6_mul(d6_mul(a,a),a))),
                       d6_mul(fM, d6_sub(d6_mul(d6_sub(d6_mul(p,cnu), d6_mul(d6_c(2.0), d6_mul(r,e))), aR),
                                         d6_mul(d6_add(p,r), d6_mul(snu,aT)))));
    Dual6 cw = d6_cos(w), sw = d6_sin(w);
    out[0]=da;
    out[1]=d6_add(dM, dw);
    out[2]=d6_sub(d6_mul(cw,de), d6_mul(d6_mul(e,sw), dw));
    out[3]=d6_add(d6_mul(sw,de), d6_mul(d6_mul(e,cw), dw));
    out[4]=di;
    out[5]=dOm;
}

// 全动力学 RHS 的对偶版
__device__ __forceinline__ void gveRhs_d_dev(const Dual6* OE, double beta, Dual6* out){
    Dual6 rv[6]; oe2rv_d_dev(OE, rv);
    Dual6 ap[3]; pertAccel_d_dev(rv, beta, ap);
    gveRhsFromAp_d_dev(OE, ap, out);
}

// 解析：同时给 f(oe) 值与 A=∂f/∂oe（对偶传播，一次求值）
__device__ __forceinline__ void gveRhsJacVal_dev(const double* OE, double beta,
                                                 double A[6][6], double fval[6]){
    Dual6 x[6], out[6];
    #pragma unroll
    for(int j=0;j<6;j++) x[j]=d6_var(OE[j], j);
    gveRhs_d_dev(x, beta, out);
    #pragma unroll
    for(int i=0;i<6;i++){
        fval[i]=out[i].v;
        #pragma unroll
        for(int j=0;j<6;j++) A[i][j]=out[i].d[j];
    }
}
__device__ __forceinline__ void gveRhsJac_dev(const double* OE, double beta, double A[6][6]){
    double fv[6]; gveRhsJacVal_dev(OE, beta, A, fv);
}

// 解析 ∂f/∂β：阻力对 β 线性，β=1 的阻力加速度即 ∂ap/∂β（m/s²）
__device__ __forceinline__ void gveRhsKappa_dev(const double* OE, const double* rv_m, double Fk[6]){
    double xk[6];
    #pragma unroll
    for(int k=0;k<6;k++) xk[k]=rv_m[k]*1e-3;
    const double rvx=xk[3]+OMEGA_EARTH*xk[1], rvy=xk[4]-OMEGA_EARTH*xk[0], rvz=xk[5];
    const double v=sqrt(rvx*rvx+rvy*rvy+rvz*rvz);
    const double rn=sqrt(xk[0]*xk[0]+xk[1]*xk[1]+xk[2]*xk[2]);
    const double rho=exp(-(rn-RE_KM-H0_KM)/H1_KM);
    const double dc=-0.5*RHO_CDA_M*rho*v;
    const double apb[3]={dc*rvx*1e3, dc*rvy*1e3, dc*rvz*1e3};
    gvePertRhsFromAp_dev(OE, apb, Fk);
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

// 二体解析传播（Lagrange F,G；**km 口径**）：r0,v0 -> rf,vf。用于多重打靶 chunk 的解析热启动。
__device__ __forceinline__ void twoBodyRvDev(const double* r0, const double* v0, double tof,
                                             double mu, double* rf, double* vf){
    const double R0 = sqrt(r0[0]*r0[0] + r0[1]*r0[1] + r0[2]*r0[2]);
    const double V02 = v0[0]*v0[0] + v0[1]*v0[1] + v0[2]*v0[2];
    const double energy = V02/2.0 - mu/R0;
    const double a = -mu/2.0/energy;
    if(!(a > 0.0)){ for(int k=0;k<3;k++){ rf[k]=r0[k]; vf[k]=v0[k]; } return; }   // 非椭圆回退
    const double sqrta = sqrt(a), sqrtmu = sqrt(mu);
    const double sigma0 = (r0[0]*v0[0] + r0[1]*v0[1] + r0[2]*v0[2])/sqrtmu;
    const double DM = sqrt(mu/(a*a*a))*tof;
    const double sinDM = sin(DM), cosDM = cos(DM);
    double DMc = atan2(sinDM, cosDM); if(DMc < 0) DMc += 2.0*PI;
    const double s0 = sigma0/sqrta, c0 = 1.0 - R0/a;
    double DE = DMc + c0*sinDM - s0*(1.0 - cosDM)
              + (c0*cosDM - s0*sinDM)*(c0*sinDM + s0*cosDM - s0)
              + 0.5*(c0*sinDM + s0*cosDM - s0)*(2.0*(c0*cosDM - s0*sinDM)*(c0*cosDM - s0*sinDM)
                - (c0*sinDM + s0*cosDM - s0)*(c0*sinDM + s0*cosDM));
    #pragma unroll 1
    for(int it=0; it<60; ++it){
        const double ff = -DMc + DE + s0*(1.0 - cos(DE)) - c0*sin(DE);
        const double df = 1.0 + s0*sin(DE) - c0*cos(DE);
        if(df == 0.0) break;
        const double d = ff/df; DE -= d;
        if(fabs(d) < 1e-15) break;
    }
    const double sinDE = sin(DE), cosDE = cos(DE);
    const double Rf = a + (R0 - a)*cosDE + sigma0*sqrta*sinDE;
    const double F = 1.0 - a/R0*(1.0 - cosDE);
    const double G = a*sigma0/sqrtmu*(1.0 - cosDE) + R0*sqrt(a/mu)*sinDE;
    const double Ft = -sqrt(mu*a)/(Rf*R0)*sinDE;
    const double Gt = 1.0 - a/Rf*(1.0 - cosDE);
    for(int k=0;k<3;k++){ rf[k] = F*r0[k] + G*v0[k]; vf[k] = Ft*r0[k] + Gt*v0[k]; }
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

// 设定 CUDA RBF 力场（整星共享）：中心（km）/宽度 s（km）/权重 θ。centers 空 → 关闭。
// 中心/宽度/个数写 constant（一次），权重写 device 全局（每外环迭代调用一次即可）。
void setRbfCuda(const std::vector<std::array<double,3>> &centers, double s,
                const std::vector<double> &w){
    const int m = (int)centers.size();
    if(m > RBF_MAX) throw std::runtime_error("setRbfCuda: m > RBF_MAX");
    int kind = FIELD_KIND_RBF;
    cudaMemcpyToSymbol(c_field_kind, &kind, sizeof(int));
    double s2 = (s > 0.0 ? s * s : 1.0);
    cudaMemcpyToSymbol(c_rbf_s2, &s2, sizeof(double));
    double cut2 = (s > 0.0 ? 18.0 * s2 : 1e300);   // 3s 半径外基的 exp < 1.2e-4，裁剪
    cudaMemcpyToSymbol(c_rbf_cut2, &cut2, sizeof(double));
    cudaMemcpyToSymbol(c_rbf_m, &m, sizeof(int));
    if(m <= 0){ double *nul = nullptr; cudaMemcpyToSymbol(g_rbf_w, &nul, sizeof(double*)); return; }
    std::vector<double> C(3 * RBF_MAX, 0.0);
    for(int k = 0; k < m; ++k){ C[3*k] = centers[k][0]; C[3*k+1] = centers[k][1]; C[3*k+2] = centers[k][2]; }
    cudaMemcpyToSymbol(c_rbf_C, C.data(), 3 * RBF_MAX * sizeof(double));
    static double *dw = nullptr; static std::size_t capw = 0;
    const std::size_t bw = (std::size_t)m * sizeof(double);
    if(bw > capw){ if(dw) cudaFree(dw); dw = nullptr; cudaMalloc(&dw, bw); capw = bw; }
    std::vector<double> ww(w.begin(), w.end());
    if(ww.size() < (std::size_t)m) ww.resize(m, 0.0);
    cudaMemcpy(dw, ww.data(), bw, cudaMemcpyHostToDevice);
    double *dp = dw;
    cudaMemcpyToSymbol(g_rbf_w, &dp, sizeof(double*));
}

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
// 多重打靶 chunk 并行前向：grid=(卫星 block, chunk)。每 chunk 用二体解析热启动到帧 cL，
// 再用全动力学(笛卡尔 cartRhsDev，含 J2/阻力/力场) 推 L 帧。L=18 → 串行链 144→18、并行度 ×8。
__global__ void gvePropagateChunkKernel(const double *rv0_all, int n, int nfr, int L, double dt, double beta,
                                        double *rv_all){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y;
    const int f0 = c * L;
    if(idx >= n || f0 >= nfr) return;
    const double *rv0 = rv0_all + (size_t)idx * 6;
    double r0km[3], v0km[3], rf[3], vf[3];
    #pragma unroll
    for(int k=0;k<3;k++){ r0km[k]=rv0[k]*1e-3; v0km[k]=rv0[3+k]*1e-3; }   // m -> km
    twoBodyRvDev(r0km, v0km, (double)f0*dt, MU_KM, rf, vf);               // 解析热启动
    double y[6];
    #pragma unroll
    for(int k=0;k<3;k++){ y[k]=rf[k]*1e3; y[3+k]=vf[k]*1e3; }             // 回 m
    #pragma unroll
    for(int k=0;k<6;k++) rv_all[((size_t)f0*n + idx)*6 + k] = y[k];
    for(int i=1;i<L;++i){
        const int f = f0 + i; if(f >= nfr) break;
        double k1[6],k2[6],k3[6],k4[6], ty[6];
        cartRhsDev(y, beta, k1);
        #pragma unroll
        for(int k=0;k<6;k++) ty[k]=y[k]+dt*k1[k]/3.0; cartRhsDev(ty,beta,k2);
        #pragma unroll
        for(int k=0;k<6;k++) ty[k]=y[k]+dt*(-k1[k]/3.0+k2[k]); cartRhsDev(ty,beta,k3);
        #pragma unroll
        for(int k=0;k<6;k++) ty[k]=y[k]+dt*(k1[k]-k2[k]+k3[k]); cartRhsDev(ty,beta,k4);
        #pragma unroll
        for(int k=0;k<6;k++) y[k]+=dt*(k1[k]+3.0*k2[k]+3.0*k3[k]+k4[k])/8.0;
        #pragma unroll
        for(int k=0;k<6;k++) rv_all[((size_t)f*n + idx)*6 + k] = y[k];
    }
}

// Host：多重打靶 chunk 并行前向。rv0s=(n,6, m)，输出 rv_all (nfr,n,6)。
void gvePropagateChunkBatch(const std::vector<Vector6d> &rv0s, int nfr, int L, double dt, double beta,
                            std::vector<double> &rv_all){
    const int n = (int)rv0s.size();
    if(n <= 0 || nfr <= 0 || L <= 0) return;
    rv_all.assign((size_t)nfr*n*6, 0.0);
    static thread_local std::vector<double> h0;
    h0.resize((size_t)n*6);
    for(int i=0;i<n;i++) for(int c=0;c<6;c++) h0[(size_t)i*6+c] = rv0s[i](c);
    static thread_local double *d0=nullptr, *drv=nullptr;
    static thread_local size_t c0=0, crv=0;
    const size_t b0=(size_t)n*6*sizeof(double), brv=(size_t)nfr*n*6*sizeof(double);
    if(b0>c0){ if(d0) cudaFree(d0); d0=nullptr; cudaMalloc(&d0,b0); c0=b0; }
    if(brv>crv){ if(drv) cudaFree(drv); drv=nullptr; cudaMalloc(&drv,brv); crv=brv; }
    cudaMemcpy(d0, h0.data(), b0, cudaMemcpyHostToDevice);
    const int threads=256, blocks=(n+threads-1)/threads;
    const int nchunk=(nfr + L - 1)/L;
    dim3 grid(blocks, nchunk);
    gvePropagateChunkKernel<<<grid, threads>>>(d0, n, nfr, L, dt, beta, drv);
    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess) fprintf(stderr, "[gvePropagateChunkBatch] %s\n", cudaGetErrorString(err));
    cudaMemcpy(rv_all.data(), drv, brv, cudaMemcpyDeviceToHost);
}

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

// ---- 单发窗口（win_assemble_ss）：每星 6×6 边界先验 + 协方差白化 ----
__device__ __forceinline__ void inv6_dev(const double *A, double *Ai) {
    double M[6][12];
    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        #pragma unroll
        for (int j = 0; j < 6; ++j) { M[i][j] = A[i * 6 + j]; M[i][6 + j] = (i == j) ? 1.0 : 0.0; }
    }
    #pragma unroll
    for (int col = 0; col < 6; ++col) {
        int piv = col; double mx = fabs(M[col][col]);
        #pragma unroll
        for (int r = col + 1; r < 6; ++r) { double v = fabs(M[r][col]); if (v > mx) { mx = v; piv = r; } }
        if (piv != col) {
            #pragma unroll
            for (int j = 0; j < 12; ++j) { double t = M[col][j]; M[col][j] = M[piv][j]; M[piv][j] = t; }
        }
        double pv = M[col][col];
        if (fabs(pv) < 1e-300) pv = (pv < 0.0 ? -1e-300 : 1e-300);
        const double ip = 1.0 / pv;
        #pragma unroll
        for (int j = 0; j < 12; ++j) M[col][j] *= ip;
        #pragma unroll
        for (int r = 0; r < 6; ++r) {
            if (r == col) continue;
            const double fct = M[r][col];
            if (fct == 0.0) continue;
            #pragma unroll
            for (int j = 0; j < 12; ++j) M[r][j] -= fct * M[col][j];
        }
    }
    #pragma unroll
    for (int i = 0; i < 6; ++i)
        #pragma unroll
        for (int j = 0; j < 6; ++j) Ai[i * 6 + j] = M[i][6 + j];
}

// 白化 ISL：权重 = mask/√(σ² + J_i P0_i J_iᵀ + J_j P0_j J_jᵀ)，J=û·(∂p/∂z0)。
__global__ void ssIslAsmKernel(const double *d_P, const double *d_A, const double *d_P0,
                               const int *isl_i, const int *isl_jj,
                               const double *isl_D, const double *isl_W,
                               int n, int nfr, int E, double sig_isl, int off_isl,
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
    double Ji[6], Jj[6];
    #pragma unroll
    for (int k = 0; k < 6; ++k) {
        Ji[k] = u0 * Ai[0 * 6 + k] + u1 * Ai[1 * 6 + k] + u2 * Ai[2 * 6 + k];
        Jj[k] = u0 * Aj[0 * 6 + k] + u1 * Aj[1 * 6 + k] + u2 * Aj[2 * 6 + k];
    }
    const double *P0i = d_P0 + (size_t)i * 36;
    const double *P0j = d_P0 + (size_t)j * 36;
    double var = sig_isl * sig_isl;
    for (int a = 0; a < 6; ++a) {
        double si = 0.0, sj = 0.0;
        for (int bb = 0; bb < 6; ++bb) { si += P0i[a * 6 + bb] * Ji[bb]; sj += P0j[a * 6 + bb] * Jj[bb]; }
        var += Ji[a] * si + Jj[a] * sj;
    }
    const double wit = isl_W[idx] / sqrt(var > 1e-30 ? var : 1e-30);
    double *av = A_val + off_isl + (size_t)idx * 12;
    #pragma unroll
    for (int k = 0; k < 6; ++k) { av[k] = wit * Ji[k]; av[6 + k] = -wit * Jj[k]; }
    b[idx] = -(rr - isl_D[idx]) * wit;
}

__global__ void ssGtsAsmKernel(const double *d_P, const double *d_A, const double *d_P0,
                               const int *gts_a, const double *gts_D,
                               const double *gts_W, const double *gts_Q,
                               int n, int G, int nGTS, double sig_gts, int nISL,
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
    double Jg[6];
    #pragma unroll
    for (int k = 0; k < 6; ++k) Jg[k] = u0 * Ag[0 * 6 + k] + u1 * Ag[1 * 6 + k] + u2 * Ag[2 * 6 + k];
    const double *P0g = d_P0 + (size_t)a * 36;
    double var = sig_gts * sig_gts;
    for (int aa = 0; aa < 6; ++aa) {
        double s = 0.0;
        for (int bb = 0; bb < 6; ++bb) s += P0g[aa * 6 + bb] * Jg[bb];
        var += Jg[aa] * s;
    }
    const double wg = gts_W[idx] / sqrt(var > 1e-30 ? var : 1e-30);
    double *av = A_val + off_gts + (size_t)idx * 6;
    #pragma unroll
    for (int k = 0; k < 6; ++k) av[k] = wg * Jg[k];
    b[nISL + idx] = -(rg - gts_D[idx]) * wg;
}

// 每星 6×6 信息先验：A_prior=chol(P0^-1)^T（行 a 全 6 列），b=-A_prior(z-zprior)。
__global__ void ssPriorAsmKernel(const double *fac, const double *z, const double *zp,
                                 int n, int off_damp, int base_b, double *A_val, double *b) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double *F = fac + (size_t)i * 36;
    const double *zi = z + (size_t)i * 6;
    const double *zpi = zp + (size_t)i * 6;
    double dz[6];
    #pragma unroll
    for (int c = 0; c < 6; ++c) dz[c] = zi[c] - zpi[c];
    #pragma unroll
    for (int a = 0; a < 6; ++a) {
        double *av = A_val + off_damp + (size_t)(i * 6 + a) * 6;
        double s = 0.0;
        #pragma unroll
        for (int bb = 0; bb < 6; ++bb) { av[bb] = F[a * 6 + bb]; s += F[a * 6 + bb] * dz[bb]; }
        b[base_b + i * 6 + a] = -s;
    }
}

// 单发窗口协方差（rv 空间，**状态与 STM 同源=二体解析**）：边界 QOE oe0 + QOE 先验 P0 →
// 每帧 rv_f(m) 与 P_f^rv = Φ_rv,f P0rv Φ_rv,fᵀ + Q_f（过程噪声在 rv 空间，物理）。
// 状态用 twoBodyRvDev、协方差用 stmTwoBodyDev，二者同一二体模型 → 一致、不发散、保留全 6×6。
__global__ void ssCovKernel(const double *oe0, const double *P0b, int n, int nfr, double dt,
                            double sigma_a, double *rv_all, double *P_all) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    double y[6], rv0[6], J0[36];
    #pragma unroll
    for (int c = 0; c < 6; ++c) y[c] = oe0[6 * idx + c];
    oe2rv_dev(y, rv0);
    {
        Dual6 OD[6];
        #pragma unroll
        for (int j = 0; j < 6; ++j) OD[j] = d6_var(y[j], j);
        Dual6 xd[6]; oe2rv_d_dev(OD, xd);
        #pragma unroll
        for (int c = 0; c < 6; ++c)
            #pragma unroll
            for (int j = 0; j < 6; ++j) J0[c * 6 + j] = xd[c].d[j];
    }
    double P0i[36];
    #pragma unroll
    for (int a = 0; a < 36; ++a) P0i[a] = P0b[idx * 36 + a];
    double tmp[36], P0rv[36];
    #pragma unroll
    for (int a = 0; a < 6; ++a)
        #pragma unroll
        for (int bb = 0; bb < 6; ++bb) {
            double s = 0.0;
            #pragma unroll
            for (int c = 0; c < 6; ++c) s += J0[a * 6 + c] * P0i[c * 6 + bb];
            tmp[a * 6 + bb] = s;
        }
    #pragma unroll
    for (int a = 0; a < 6; ++a)
        #pragma unroll
        for (int bb = 0; bb < 6; ++bb) {
            double s = 0.0;
            #pragma unroll
            for (int c = 0; c < 6; ++c) s += tmp[a * 6 + c] * J0[bb * 6 + c];
            P0rv[a * 6 + bb] = s;
        }
    const double r0k[3] = { rv0[0] * 1e-3, rv0[1] * 1e-3, rv0[2] * 1e-3 };
    const double v0k[3] = { rv0[3] * 1e-3, rv0[4] * 1e-3, rv0[5] * 1e-3 };
    for (int f = 0; f < nfr; ++f) {
        const double tof = (double)f * dt;
        double Phi[6][6]; stmTwoBodyDev(r0k, v0k, tof, MU_KM, Phi);
        double rf[3], vf[3]; twoBodyRvDev(r0k, v0k, tof, MU_KM, rf, vf);
        double *rvp = rv_all + ((size_t)f * n + idx) * 6;
        rvp[0] = rf[0] * 1e3; rvp[1] = rf[1] * 1e3; rvp[2] = rf[2] * 1e3;
        rvp[3] = vf[0] * 1e3; rvp[4] = vf[1] * 1e3; rvp[5] = vf[2] * 1e3;
        const double q00 = sigma_a * sigma_a * tof * tof * tof / 3.0;
        const double q01 = sigma_a * sigma_a * tof * tof / 2.0;
        const double q11 = sigma_a * sigma_a * tof;
        double A1[36], Prv[36];
        #pragma unroll
        for (int a = 0; a < 6; ++a)
            #pragma unroll
            for (int bb = 0; bb < 6; ++bb) {
                double s = 0.0;
                #pragma unroll
                for (int c = 0; c < 6; ++c) s += Phi[a][c] * P0rv[c * 6 + bb];
                A1[a * 6 + bb] = s;
            }
        #pragma unroll
        for (int a = 0; a < 6; ++a)
            #pragma unroll
            for (int bb = 0; bb < 6; ++bb) {
                double s = 0.0;
                #pragma unroll
                for (int c = 0; c < 6; ++c) s += A1[a * 6 + c] * Phi[bb][c];
                if (a < 3 && bb < 3) s += q00;
                else if (a >= 3 && bb >= 3) s += q11;
                else s += q01;
                Prv[a * 6 + bb] = s;
            }
        double *Po = P_all + ((size_t)f * n + idx) * 36;
        #pragma unroll
        for (int a = 0; a < 36; ++a) Po[a] = Prv[a];
    }
}

// 从 X(n,6) QOE 设备端算 rv0(n,6) 与 A0=∂rv/∂oe(n,36)（对偶），免 host DACE。
__global__ void oeJacKernel(const double *X, int n, double *rv0, double *A0) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double y[6];
    #pragma unroll
    for (int c = 0; c < 6; ++c) y[c] = X[i*6+c];
    Dual6 OD[6];
    #pragma unroll
    for (int j = 0; j < 6; ++j) OD[j] = d6_var(y[j], j);
    Dual6 xd[6]; oe2rv_d_dev(OD, xd);
    #pragma unroll
    for (int c = 0; c < 6; ++c) {
        rv0[i*6+c] = xd[c].v;
        #pragma unroll
        for (int j = 0; j < 6; ++j) A0[i*36 + c*6 + j] = xd[c].d[j];
    }
}

void jointExpandDevice(Ctx &c) {
    const int threads = 256;
    const int blocks = (c.n + threads - 1) / threads;
    const double t0 = _jnow();
    oeJacKernel<<<blocks, threads>>>(c.d_X, c.n, c.d_rv0, c.d_A0);   // rv0/A0 设备端（免 host DACE）
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

// 前向声明（定义在 host 包装段；_asmKernels 先引用）
__global__ void ssIslAsmCovKernel(const double *, const double *, const double *, const int *,
    const int *, const double *, const double *, int, int, int, double, int, double *, double *);
__global__ void ssGtsAsmCovKernel(const double *, const double *, const double *, const int *,
    const double *, const double *, const double *, int, int, int, double, int, int, double *, double *);

static void _asmKernels(Ctx &c, double *A_val, double *b) {
    if (!(A_val && b)) return;
    const int threads = 256;
    const int nISL = c.nfr * c.E;
    const int nGTS = c.nfr * c.G;
    if (nISL > 0) {
        const int blocks = (nISL + threads - 1) / threads;
        if (c.cov_mode == 2)
            ssIslAsmCovKernel<<<blocks, threads>>>(c.d_P, c.d_A, c.d_Pm, c.d_isl_i, c.d_isl_jj,
                                                   c.d_isl_D, c.d_isl_W, c.n, c.nfr, c.E,
                                                   1.0 / c.s_isl, c.off_isl, A_val, b);
        else if (c.cov_mode == 1)
            ssIslAsmKernel<<<blocks, threads>>>(c.d_P, c.d_A, c.d_P0, c.d_isl_i, c.d_isl_jj,
                                                c.d_isl_D, c.d_isl_W, c.n, c.nfr, c.E,
                                                1.0 / c.s_isl, c.off_isl, A_val, b);
        else
            jointIslAsmKernel<<<blocks, threads>>>(c.d_P, c.d_A, c.d_isl_i, c.d_isl_jj,
                                                   c.d_isl_D, c.d_isl_W, c.n, c.nfr, c.E,
                                                   c.s_isl, c.off_isl, A_val, b);
    }
    if (nGTS > 0) {
        const int blocks = (nGTS + threads - 1) / threads;
        if (c.cov_mode == 2)
            ssGtsAsmCovKernel<<<blocks, threads>>>(c.d_P, c.d_A, c.d_Pm, c.d_gts_a, c.d_gts_D,
                                                   c.d_gts_W, c.d_gts_Q, c.n, c.G, nGTS,
                                                   1.0 / c.s_gts, nISL, c.off_gts, A_val, b);
        else if (c.cov_mode == 1)
            ssGtsAsmKernel<<<blocks, threads>>>(c.d_P, c.d_A, c.d_P0, c.d_gts_a, c.d_gts_D,
                                                c.d_gts_W, c.d_gts_Q, c.n, c.G, nGTS,
                                                1.0 / c.s_gts, nISL, c.off_gts, A_val, b);
        else
            jointGtsAsmKernel<<<blocks, threads>>>(c.d_P, c.d_A, c.d_gts_a, c.d_gts_D,
                                                   c.d_gts_W, c.d_gts_Q, c.n, c.G, nGTS,
                                                   c.s_gts, nISL, c.off_gts, A_val, b);
    }
    if (c.ncol > 0) {
        const int blocks = (c.ncol + threads - 1) / threads;
        if (c.prior_mode)
            ssPriorAsmKernel<<<blocks, threads>>>(c.d_prior_fac, c.d_X, c.d_zprior,
                                                  c.n, c.off_damp, nISL + nGTS, A_val, b);
        else
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
// A=∂f/∂oe、Fκ=∂f/∂κ **解析**（gveRhsJacVal_dev / gveRhsKappa_dev；设备端无有限差分）。
// **积分器稳定化**：非奇异要素在 e→0 附近使 A 非正规（非对角元可达 ~1e2，特征值 ~1e-6）；
//   直接对 dt=10 用 3/8-RK4 积分线性变分方程 \dot S=A S+Fκ 会被非正规瞬态放大而发散。
//   故每帧在帧首冻结 (A,Fκ)，对 S 做 SENS_SUBSTEPS 个 dt 子步的线性 RK4（实测 rel ~1e-3）。
#define SENS_SUBSTEPS 8
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
    const double hs = dt / (double)SENS_SUBSTEPS;
    double k1y[6], k2y[6], k3y[6], k4y[6], ty[6];
    for (int f = 1; f < nfr; ++f) {
        double A[6][6], Fk[6];
        gveRhsJacVal_dev(y, beta, A, k1y);                 // fval=f(y), A=∂f/∂y（帧首冻结）
        double rv[6]; oe2rv_dev(y, rv);
        gveRhsKappa_dev(y, rv, Fk);
        for (int s = 0; s < SENS_SUBSTEPS; ++s) {          // 冻结 A,Fκ 的线性子步
            double s1[6], s2[6], s3[6], s4[6], z[6];
            #pragma unroll
            for (int i = 0; i < 6; ++i) { double t = Fk[i]; for (int j = 0; j < 6; ++j) t += A[i][j] * S[j]; s1[i] = t; }
            #pragma unroll
            for (int i = 0; i < 6; ++i) z[i] = S[i] + hs * s1[i] / 3.0;
            #pragma unroll
            for (int i = 0; i < 6; ++i) { double t = Fk[i]; for (int j = 0; j < 6; ++j) t += A[i][j] * z[j]; s2[i] = t; }
            #pragma unroll
            for (int i = 0; i < 6; ++i) z[i] = S[i] + hs * (-s1[i] / 3.0 + s2[i]);
            #pragma unroll
            for (int i = 0; i < 6; ++i) { double t = Fk[i]; for (int j = 0; j < 6; ++j) t += A[i][j] * z[j]; s3[i] = t; }
            #pragma unroll
            for (int i = 0; i < 6; ++i) z[i] = S[i] + hs * (s1[i] - s2[i] + s3[i]);
            #pragma unroll
            for (int i = 0; i < 6; ++i) { double t = Fk[i]; for (int j = 0; j < 6; ++j) t += A[i][j] * z[j]; s4[i] = t; }
            #pragma unroll
            for (int i = 0; i < 6; ++i) S[i] += hs * (s1[i] + 3.0 * s2[i] + 3.0 * s3[i] + s4[i]) / 8.0;
        }
        #pragma unroll
        for (int c = 0; c < 6; ++c) ty[c] = y[c] + dt * k1y[c] / 3.0;
        gveRhs_dev(ty, beta, k2y);
        #pragma unroll
        for (int c = 0; c < 6; ++c) ty[c] = y[c] + dt * (-k1y[c] / 3.0 + k2y[c]);
        gveRhs_dev(ty, beta, k3y);
        #pragma unroll
        for (int c = 0; c < 6; ++c) ty[c] = y[c] + dt * (k1y[c] - k2y[c] + k3y[c]);
        gveRhs_dev(ty, beta, k4y);
        #pragma unroll
        for (int c = 0; c < 6; ++c) y[c] += dt * (k1y[c] + 3.0 * k2y[c] + 3.0 * k3y[c] + k4y[c]) / 8.0;
        #pragma unroll
        for (int c = 0; c < 6; ++c) {
            oe_all[((size_t)f * n + idx) * 6 + c] = y[c];
            S_all[((size_t)f * n + idx) * 6 + c] = S[c];
        }
    }
}

// 调试/校验钩子：输出 A=∂f/∂oe 与 Fk=∂f/∂κ（单步，给定 oe）。
__global__ void dbgJacKernel(const double *oe, double beta, double *Aout, double *Fout) {
    double A[6][6], fv[6]; gveRhsJacVal_dev(oe, beta, A, fv);
    double rv[6]; oe2rv_dev(oe, rv);
    double Fk[6]; gveRhsKappa_dev(oe, rv, Fk);
    for (int i = 0; i < 6; ++i) { Fout[i] = Fk[i]; for (int j = 0; j < 6; ++j) Aout[i * 6 + j] = A[i][j]; }
}
void dbgJac(const double *oe_host, int n, double beta, double *Aout, double *Fout) {
    const size_t bo = (size_t)n * 6 * sizeof(double), bA = (size_t)n * 36 * sizeof(double), bF = (size_t)n * 6 * sizeof(double);
    double *d_oe = nullptr, *d_A = nullptr, *d_F = nullptr;
    cudaMalloc(&d_oe, bo); cudaMalloc(&d_A, bA); cudaMalloc(&d_F, bF);
    cudaMemcpy(d_oe, oe_host, bo, cudaMemcpyHostToDevice);
    dbgJacKernel<<<(n + 255) / 256, 256>>>(d_oe, beta, d_A, d_F);
    cudaDeviceSynchronize();
    cudaMemcpy(Aout, d_A, bA, cudaMemcpyDeviceToHost);
    cudaMemcpy(Fout, d_F, bF, cudaMemcpyDeviceToHost);
    cudaFree(d_oe); cudaFree(d_A); cudaFree(d_F);
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
    gveSensPropagateKernel<<<blocks, threads>>>(dx, n, nfr, c.dt, beta, doe, dS);
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
    gveRhsJacVal_dev(y, beta, A, k1);
    for (int a = 0; a < 6; ++a) for (int b = 0; b < 6; ++b) { double s = 0; for (int c = 0; c < 6; ++c) s += A[a][c] * P[c][b]; K1[a][b] = s; }
    #pragma unroll
    for (int c = 0; c < 6; ++c) tmp[c] = y[c] + dt * k1[c] / 3.0;
    gveRhsJacVal_dev(tmp, beta, A, k2);
    for (int a = 0; a < 6; ++a) for (int b = 0; b < 6; ++b) { double s = 0; for (int c = 0; c < 6; ++c) s += A[a][c] * (P[c][b] + dt * K1[c][b] / 3.0); K2[a][b] = s; }
    #pragma unroll
    for (int c = 0; c < 6; ++c) tmp[c] = y[c] + dt * (-k1[c] / 3.0 + k2[c]);
    gveRhsJacVal_dev(tmp, beta, A, k3);
    for (int a = 0; a < 6; ++a) for (int b = 0; b < 6; ++b) { double s = 0; for (int c = 0; c < 6; ++c) s += A[a][c] * (P[c][b] + dt * (-K1[c][b] / 3.0 + K2[c][b])); K3[a][b] = s; }
    #pragma unroll
    for (int c = 0; c < 6; ++c) tmp[c] = y[c] + dt * (k1[c] - k2[c] + k3[c]);
    gveRhsJacVal_dev(tmp, beta, A, k4);
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

// ---- RBF θ 敏度：S_f = ∂oe_f/∂θ（逐基 k，变分方程 + 同一 3/8-RK4；与 κ 敏度同结构）----
// ∂f/∂θ_k = gveRhsFromAp_dev(oe, b_k(rv)·1e3)（b_k 为第 k 基加速度 km/s²→m/s²）；
// A=∂f/∂oe 解析（gveRhsJacVal_dev；含 RBF 的位置依赖）。S 为 (6×m) 每星，(nfr,n,6,m)。
// 说明：S 逐基（6 向量）在寄存器中做 RK4，状态 S_k 存全局（S_all）跨帧；**不用大动态本地数组**
// （此前 `double S[6*64]` 之类使 nvcc 编译爆掉/卡死）。A=∂f/∂oe 每帧 4 个阶段各算一次（共享）。
#define RBF_SENS_MAX 64
__device__ __forceinline__ void rbfSensFk_dev(const double *oe, const double *r_km, int k, double *Fk){
    double bk[3]; fieldBasisKmDev(r_km, k, bk);
    const double apk[3] = { bk[0]*1e3, bk[1]*1e3, bk[2]*1e3 };      // km/s² → m/s²
    gvePertRhsFromAp_dev(oe, apk, Fk);
}

// 阶段1：逐星传播（1D grid，n 线程），落每帧 A=∂f/∂x(36)、W=∂f/∂a_p(18)、r_km(3)，索引 (f-1)。
// 这些量与基 k 无关，故只在阶段1算一次（阶段2 各基复用），避免 m 倍的对偶/Kepler 冗余。
// 阶段1：逐星状态传播（时间维串行，n 线程只出状态轨迹 oe_all）。
__global__ void gveRbfSensStateKernel(const double *oe0, int n, int nfr, double dt, double beta,
                                      double *oe_all){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n) return;
    double y[6];
    #pragma unroll
    for(int c=0;c<6;c++) y[c] = oe0[6*idx + c];
    #pragma unroll
    for(int c=0;c<6;c++) oe_all[((size_t)0*n + idx)*6 + c] = y[c];
    for(int f=1; f<nfr; ++f){
        double k1[6],k2[6],k3[6],k4[6], ty[6];
        gveRhs_dev(y, beta, k1);
        #pragma unroll
        for(int c=0;c<6;c++) ty[c]=y[c]+dt*k1[c]/3.0; gveRhs_dev(ty,beta,k2);
        #pragma unroll
        for(int c=0;c<6;c++) ty[c]=y[c]+dt*(-k1[c]/3.0+k2[c]); gveRhs_dev(ty,beta,k3);
        #pragma unroll
        for(int c=0;c<6;c++) ty[c]=y[c]+dt*(k1[c]-k2[c]+k3[c]); gveRhs_dev(ty,beta,k4);
        #pragma unroll
        for(int c=0;c<6;c++) y[c]+=dt*(k1[c]+3.0*k2[c]+3.0*k3[c]+k4[c])/8.0;
        #pragma unroll
        for(int c=0;c<6;c++) oe_all[((size_t)f*n + idx)*6 + c] = y[c];
    }
}

// 阶段1b：**帧维并行**（grid=(sat-block, frame)）：由状态轨迹算每帧 Φ=exp(A dt)、Ψ=∫Φ、W=∂f/∂a_p、r。
// 每帧彼此独立（无串行依赖）→ 约 165888 线程填满 GPU，消除阶段1 只有 5 个 block 的占用瓶颈。
__global__ void gveRbfSensJacKernel(const double *oe_all, int n, int nfr, double dt, double beta,
                                    double *A_all, double *W_all, double *r_all){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int f = blockIdx.y;
    if(idx >= n || f >= nfr-1) return;
    double y[6];
    #pragma unroll
    for(int c=0;c<6;c++) y[c] = oe_all[((size_t)f*n + idx)*6 + c];
    double A[6][6], fv[6], rv[6], W[6][3];
    gveRhsJacVal_dev(y, beta, A, fv);  oe2rv_dev(y, rv);
    gveGaussW_dev(y, W);
    const size_t fbase = (size_t)f;
    const double hs = dt / (double)SENS_SUBSTEPS;
    double Z[6][12];                                   // [0..5]=Φ, [6..11]=Ψ
    #pragma unroll
    for(int i=0;i<6;i++) for(int j=0;j<12;j++) Z[i][j] = (j<6) ? ((i==j)?1.0:0.0) : 0.0;
    for(int s=0;s<SENS_SUBSTEPS;s++){
        double K1[6][12],K2[6][12],K3[6][12],K4[6][12];
        #pragma unroll
        for(int i=0;i<6;i++) for(int j=0;j<12;j++){ double t=(j>=6 && i==j-6)?1.0:0.0; for(int l=0;l<6;l++) t+=A[i][l]*Z[l][j]; K1[i][j]=t; }
        #pragma unroll
        for(int i=0;i<6;i++) for(int j=0;j<12;j++){ double t=(j>=6 && i==j-6)?1.0:0.0; for(int l=0;l<6;l++) t+=A[i][l]*(Z[l][j]+hs*K1[l][j]/3.0); K2[i][j]=t; }
        #pragma unroll
        for(int i=0;i<6;i++) for(int j=0;j<12;j++){ double t=(j>=6 && i==j-6)?1.0:0.0; for(int l=0;l<6;l++) t+=A[i][l]*(Z[l][j]+hs*(-K1[l][j]/3.0+K2[l][j])); K3[i][j]=t; }
        #pragma unroll
        for(int i=0;i<6;i++) for(int j=0;j<12;j++){ double t=(j>=6 && i==j-6)?1.0:0.0; for(int l=0;l<6;l++) t+=A[i][l]*(Z[l][j]+hs*(K1[l][j]-K2[l][j]+K3[l][j])); K4[i][j]=t; }
        #pragma unroll
        for(int i=0;i<6;i++) for(int j=0;j<12;j++) Z[i][j]+=hs*(K1[i][j]+3.0*K2[i][j]+3.0*K3[i][j]+K4[i][j])/8.0;
    }
    #pragma unroll
    for(int i=0;i<6;i++) for(int j=0;j<12;j++) A_all[(fbase*72 + i*12 + j)*n + idx] = Z[i][j];
    #pragma unroll
    for(int i=0;i<6;i++) for(int j=0;j<3;j++) W_all[(fbase*18 + i*3 + j)*n + idx] = W[i][j];
    #pragma unroll
    for(int j=0;j<3;j++) r_all[(fbase*3 + j)*n + idx] = rv[j]*1e-3;
}

// 阶段2：二维 grid (x=卫星 idx, y=基 k)。读帧首 A,W,r，做冻结 A 的线性子步；无 Kepler/无对偶。
// 阶段2：二维 grid（x=卫星，y=基 k）。读帧首 Φ/Ψ/W/r，每基每帧 2 次 matvec（无 Kepler/对偶）。
__global__ void gveRbfSensPropagateKernel(const double *A_all, const double *W_all, const double *r_all,
                                          int n, int nfr, int m, double *S_all){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y;
    if(idx >= n || k >= m || m <= 0 || m > RBF_SENS_MAX) return;
    #pragma unroll
    for(int c=0;c<6;c++) S_all[((size_t)c*m + k)*n + idx] = 0.0;      // 帧0 = 0（布局 (帧,6,m,n)）
    double S6[6];
    #pragma unroll
    for(int c=0;c<6;c++) S6[c] = 0.0;
    for(int f=1; f<nfr; ++f){
        const size_t fbase = (size_t)(f-1);
        double Phi[6][6], Psi[6][6], r_km[3];
        #pragma unroll
        for(int i=0;i<6;i++) for(int j=0;j<6;j++){
            Phi[i][j] = A_all[(fbase*72 + i*12 + j)*n + idx];
            Psi[i][j] = A_all[(fbase*72 + i*12 + 6 + j)*n + idx];
        }
        #pragma unroll
        for(int j=0;j<3;j++) r_km[j] = r_all[(fbase*3 + j)*n + idx];
        double bk[3]; fieldBasisKmDev(r_km, k, bk);
        double Fk[6], Sn[6];
        #pragma unroll
        for(int i=0;i<6;i++)
            Fk[i] = W_all[(fbase*18+i*3+0)*n+idx]*bk[0]*1e3 + W_all[(fbase*18+i*3+1)*n+idx]*bk[1]*1e3 + W_all[(fbase*18+i*3+2)*n+idx]*bk[2]*1e3;
        #pragma unroll
        for(int i=0;i<6;i++){ double t=0.0; for(int j=0;j<6;j++) t += Phi[i][j]*S6[j] + Psi[i][j]*Fk[j]; Sn[i]=t; }
        #pragma unroll
        for(int i=0;i<6;i++) S6[i]=Sn[i];
        #pragma unroll
        for(int c=0;c<6;c++) S_all[(((size_t)f*6 + c)*m + k)*n + idx] = S6[c];
    }
}

// ---- 分段并行 scan（阶段2 的并行-in-time）：S_f=Φ_f S_{f-1}+Ψ_f F_f ----
// Pass1：段级 Φ_c=ΠΦ 与 forcing 响应 G_c^{(k)}（从段首 S=0 跑段内递推）。grid=(sat-block, chunk)
__global__ void gveRbfSensChunkMapKernel(const double *A_all, const double *W_all, const double *r_all,
                                         int n, int nfr, int L, int m, double *PhiC_all, double *G_all){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y;
    if(idx >= n) return;
    const int a = c*L, b = min((c+1)*L, nfr-1);      // 段覆盖帧 [a,b]
    if(a >= b) return;
    double Pc[6][6];
    #pragma unroll
    for(int i=0;i<6;i++) for(int j=0;j<6;j++) Pc[i][j] = (i==j)?1.0:0.0;
    for(int f=a+1; f<=b; ++f){                        // Φ_c = Φ_{b-1}···Φ_a
        const size_t fb=(size_t)(f-1); double P[6][6], N[6][6];
        #pragma unroll
        for(int i=0;i<6;i++) for(int j=0;j<6;j++) P[i][j]=A_all[(fb*72+i*12+j)*n+idx];
        #pragma unroll
        for(int i=0;i<6;i++) for(int j=0;j<6;j++){ double t=0; for(int l=0;l<6;l++) t+=P[i][l]*Pc[l][j]; N[i][j]=t; }
        #pragma unroll
        for(int i=0;i<6;i++) for(int j=0;j<6;j++) Pc[i][j]=N[i][j];
    }
    #pragma unroll
    for(int i=0;i<6;i++) for(int j=0;j<6;j++) PhiC_all[(c*n+idx)*36 + i*6+j] = Pc[i][j];
    for(int k=0;k<m;k++){                             // G_c^{(k)}：段内 forcing 响应（S_a=0）
        double S[6];
        #pragma unroll
        for(int i=0;i<6;i++) S[i]=0.0;
        for(int f=a+1; f<=b; ++f){
            const size_t fb=(size_t)(f-1);
            double Phi[6][6], Psi[6][6], r_km[3];
            #pragma unroll
            for(int i=0;i<6;i++) for(int j=0;j<6;j++){
                Phi[i][j]=A_all[(fb*72+i*12+j)*n+idx];
                Psi[i][j]=A_all[(fb*72+i*12+6+j)*n+idx];
            }
            #pragma unroll
            for(int j=0;j<3;j++) r_km[j]=r_all[(fb*3+j)*n+idx];
            double bk[3]; fieldBasisKmDev(r_km, k, bk);
            double Fk[6], Sn[6];
            #pragma unroll
            for(int i=0;i<6;i++) Fk[i]=W_all[(fb*18+i*3+0)*n+idx]*bk[0]*1e3 + W_all[(fb*18+i*3+1)*n+idx]*bk[1]*1e3 + W_all[(fb*18+i*3+2)*n+idx]*bk[2]*1e3;
            #pragma unroll
            for(int i=0;i<6;i++){ double t=0.0; for(int j=0;j<6;j++) t += Phi[i][j]*S[j] + Psi[i][j]*Fk[j]; Sn[i]=t; }
            #pragma unroll
            for(int i=0;i<6;i++) S[i]=Sn[i];
        }
        #pragma unroll
        for(int i=0;i<6;i++) G_all[(((size_t)c*n+idx)*m + k)*6 + i] = S[i];
    }
}

// Pass2：段首 scan（每星,每基，串行 C 段）：S^{(c+1)}_{start}=Φ_c S^{(c)}_{start}+G_c^{(k)}。
__global__ void gveRbfSensScanKernel(int n, int C, int m, const double *PhiC_all, const double *G_all,
                                     double *Ss_all){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n) return;
    for(int k=0;k<m;k++){
        double S[6];
        #pragma unroll
        for(int i=0;i<6;i++) S[i]=0.0;
        #pragma unroll
        for(int i=0;i<6;i++) Ss_all[(((size_t)0*n+idx)*m + k)*6 + i] = 0.0;   // 段0 首 = 帧0 = 0
        for(int c=1;c<C;c++){
            double Pc[6][6], Sn[6];
            #pragma unroll
            for(int i=0;i<6;i++) for(int j=0;j<6;j++) Pc[i][j]=PhiC_all[((size_t)(c-1)*n+idx)*36 + i*6+j];
            #pragma unroll
            for(int i=0;i<6;i++){ double t=G_all[(((size_t)(c-1)*n+idx)*m + k)*6 + i]; for(int j=0;j<6;j++) t += Pc[i][j]*S[j]; Sn[i]=t; }
            #pragma unroll
            for(int i=0;i<6;i++) S[i]=Sn[i];
            #pragma unroll
            for(int i=0;i<6;i++) Ss_all[(((size_t)c*n+idx)*m + k)*6 + i] = S[i];
        }
    }
}

// Pass3：从段首并行回代逐帧 S。grid=(sat-block, chunk)
__global__ void gveRbfSensApplyKernel(const double *A_all, const double *W_all, const double *r_all,
                                      const double *Ss_all, int n, int nfr, int L, int m, double *S_all){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y;
    if(idx >= n) return;
    const int a = c*L, b = min((c+1)*L, nfr-1);
    if(a >= b) return;
    for(int k=0;k<m;k++){
        double S[6];
        #pragma unroll
        for(int i=0;i<6;i++) S[i]=Ss_all[(((size_t)c*n+idx)*m + k)*6 + i];    // 段首
        #pragma unroll
        for(int i=0;i<6;i++) S_all[(((size_t)a*6 + i)*m + k)*n + idx] = S[i]; // 帧 a
        for(int f=a+1; f<=b; ++f){
            const size_t fb=(size_t)(f-1);
            double Phi[6][6], Psi[6][6], r_km[3];
            #pragma unroll
            for(int i=0;i<6;i++) for(int j=0;j<6;j++){
                Phi[i][j]=A_all[(fb*72+i*12+j)*n+idx];
                Psi[i][j]=A_all[(fb*72+i*12+6+j)*n+idx];
            }
            #pragma unroll
            for(int j=0;j<3;j++) r_km[j]=r_all[(fb*3+j)*n+idx];
            double bk[3]; fieldBasisKmDev(r_km, k, bk);
            double Fk[6], Sn[6];
            #pragma unroll
            for(int i=0;i<6;i++) Fk[i]=W_all[(fb*18+i*3+0)*n+idx]*bk[0]*1e3 + W_all[(fb*18+i*3+1)*n+idx]*bk[1]*1e3 + W_all[(fb*18+i*3+2)*n+idx]*bk[2]*1e3;
            #pragma unroll
            for(int i=0;i<6;i++){ double t=0.0; for(int j=0;j<6;j++) t += Phi[i][j]*S[j] + Psi[i][j]*Fk[j]; Sn[i]=t; }
            #pragma unroll
            for(int i=0;i<6;i++) S[i]=Sn[i];
            #pragma unroll
            for(int i=0;i<6;i++) S_all[(((size_t)f*6 + i)*m + k)*n + idx] = S[i];
        }
    }
}

// Host：整弧 RBF θ 敏度，输出 oe_all(nfr*n*6) 与 S_all(nfr*n*6*m)。centers/s 需已 setRbfCuda 预设。
void gveRbfSensBatch(const std::vector<kep3::Vector6d> &oe0s, int nfr, double dt, double beta, int m,
                     std::vector<double> &oe_all, std::vector<double> &S_all){
    const int n = (int)oe0s.size();
    oe_all.assign((size_t)nfr*n*6, 0.0);
    S_all.assign((size_t)nfr*n*6*(size_t)(m>0?m:1), 0.0);
    if(n == 0 || nfr <= 0 || m <= 0) return;
    static thread_local std::vector<double> h0;
    h0.resize((size_t)n*6);
    for(int i=0;i<n;i++) for(int c=0;c<6;c++) h0[(size_t)i*6+c] = oe0s[i](c);
    const int CH = 18; const int C = (nfr - 1 + CH - 1) / CH;     // 段长 18 帧
    static thread_local double *d0=nullptr, *doe=nullptr, *dS=nullptr, *dA=nullptr, *dW=nullptr, *dr=nullptr,
                              *dPc=nullptr, *dG=nullptr, *dSs=nullptr;
    static thread_local size_t c0=0, coe=0, cS=0, cA=0, cW=0, cr=0, cPc=0, cG=0, cSs=0;
    const size_t b0=(size_t)n*6*sizeof(double), boe=(size_t)nfr*n*6*sizeof(double),
                 bS=(size_t)nfr*n*6*(size_t)m*sizeof(double),
                 bA=(size_t)nfr*n*72*sizeof(double), bW=(size_t)nfr*n*18*sizeof(double),
                 br=(size_t)nfr*n*3*sizeof(double),
                 bPc=(size_t)C*n*36*sizeof(double), bG=(size_t)C*n*m*6*sizeof(double), bSs=bG;
    if(b0>c0){ if(d0) cudaFree(d0); d0=nullptr; cudaMalloc(&d0,b0); c0=b0; }
    if(boe>coe){ if(doe) cudaFree(doe); doe=nullptr; cudaMalloc(&doe,boe); coe=boe; }
    if(bS>cS){ if(dS) cudaFree(dS); dS=nullptr; cudaMalloc(&dS,bS); cS=bS; }
    if(bA>cA){ if(dA) cudaFree(dA); dA=nullptr; cudaMalloc(&dA,bA); cA=bA; }
    if(bW>cW){ if(dW) cudaFree(dW); dW=nullptr; cudaMalloc(&dW,bW); cW=bW; }
    if(br>cr){ if(dr) cudaFree(dr); dr=nullptr; cudaMalloc(&dr,br); cr=br; }
    if(bPc>cPc){ if(dPc) cudaFree(dPc); dPc=nullptr; cudaMalloc(&dPc,bPc); cPc=bPc; }
    if(bG>cG){ if(dG) cudaFree(dG); dG=nullptr; cudaMalloc(&dG,bG); cG=bG; }
    if(bSs>cSs){ if(dSs) cudaFree(dSs); dSs=nullptr; cudaMalloc(&dSs,bSs); cSs=bSs; }
    cudaMemcpy(d0, h0.data(), b0, cudaMemcpyHostToDevice);
    const int threads=64, blocks=(n+threads-1)/threads;            // 小 block → 更多 SM（n=1152 线程）
    gveRbfSensStateKernel<<<blocks, threads>>>(d0, n, nfr, dt, beta, doe);
    dim3 gridj(blocks, nfr - 1);                                  // 帧维并行：每帧独立算 Φ/Ψ/W/r
    gveRbfSensJacKernel<<<gridj, threads>>>(doe, n, nfr, dt, beta, dA, dW, dr);
    (void)CH; (void)C; (void)dPc; (void)dG; (void)dSs;            // 分段 scan 实测更慢，保留 2D-grid 版
    dim3 grid(blocks, m);
    gveRbfSensPropagateKernel<<<grid, threads>>>(dA, dW, dr, n, nfr, m, dS);
    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess) fprintf(stderr, "[gveRbfSensBatch] %s\n", cudaGetErrorString(err));
    cudaMemcpy(oe_all.data(), doe, boe, cudaMemcpyDeviceToHost);
    static thread_local std::vector<double> hS;      // dS 布局 (nfr,6,m,n) → 输出 (nfr,n,6,m)
    hS.resize((size_t)nfr * 6 * (size_t)m * n);
    cudaMemcpy(hS.data(), dS, bS, cudaMemcpyDeviceToHost);
    for(int f=0; f<nfr; ++f)
        for(int i=0; i<n; ++i)
            for(int c=0; c<6; ++c)
                for(int k=0; k<m; ++k)
                    S_all[(((size_t)f*n + i)*6 + c)*(size_t)m + k] = hS[(((size_t)f*6 + c)*m + k)*(size_t)n + i];
}

// ===================== 逐帧协方差（单发 ncol=6N）：rv 空间，状态与 STM 同源=二体 =====================
// P_k^- = Φ_k P_{k-1}^+ Φ_kᵀ + Q(dt)（二体单步 Φ）；ISL/GTS 后验注入 H=[û,0]（rv 空间，测量仅位置）：
// S_k = Σ Hᵀ R⁻¹ H；P_k^+ = (P_k^{-,-1}+S_k)^{-1}。迭代间 P_{k-1}^+ 取上一轮（Jacobi）→ 时间并行。
__device__ __forceinline__ void mm6_dev(const double *A, const double *B, double *C) {
    #pragma unroll
    for (int a = 0; a < 6; ++a)
        #pragma unroll
        for (int b = 0; b < 6; ++b) { double s = 0.0;
            #pragma unroll
            for (int c = 0; c < 6; ++c) s += A[a*6+c]*B[c*6+b]; C[a*6+b] = s; }
}
__device__ __forceinline__ void mm6t_dev(const double *A, const double *B, double *C) {
    #pragma unroll
    for (int a = 0; a < 6; ++a)
        #pragma unroll
        for (int b = 0; b < 6; ++b) { double s = 0.0;
            #pragma unroll
            for (int c = 0; c < 6; ++c) s += A[a*6+c]*B[b*6+c]; C[a*6+b] = s; }
}
// 逐 (星 i, 帧 k) 的协方差步：Pprev(rv 6×6) → Pm=P_k^-、Pp=P_k^+（rv 空间）。
__device__ __forceinline__ void frameCovStep_rv(
        const double *d_P, int i, int k, int n, double dt, double sigma_a,
        const int *adj_j, const double *adj_w, int adjD, double sig_isl,
        int G, const int *gts_a, const double *gts_W, const double *gts_Q, double sig_gts,
        const double *Pprev, double *Pm, double *Pp) {
    double Pmv[36];
    if (k == 0) {
        #pragma unroll
        for (int a = 0; a < 36; ++a) Pmv[a] = Pprev[a];
    } else {
        const double *rv0 = d_P + (size_t)((k - 1) * n + i) * 6;
        const double r0k[3] = { rv0[0]*1e-3, rv0[1]*1e-3, rv0[2]*1e-3 };
        const double v0k[3] = { rv0[3]*1e-3, rv0[4]*1e-3, rv0[5]*1e-3 };
        double Phi[6][6]; stmTwoBodyDev(r0k, v0k, dt, MU_KM, Phi);
        double tmp[36]; mm6t_dev((const double*)Phi, Pprev, tmp); mm6_dev(tmp, (const double*)Phi, Pmv);
        double q00 = sigma_a*sigma_a*dt*dt*dt/3.0, q01 = sigma_a*sigma_a*dt*dt/2.0, q11 = sigma_a*sigma_a*dt;
        #pragma unroll
        for (int a = 0; a < 6; ++a)
            #pragma unroll
            for (int b = 0; b < 6; ++b) {
                if (a < 3 && b < 3) Pmv[a*6+b] += q00;
                else if (a >= 3 && b >= 3) Pmv[a*6+b] += q11;
                else Pmv[a*6+b] += q01;
            }
    }
    double S[36];
    #pragma unroll
    for (int a = 0; a < 36; ++a) S[a] = 0.0;
    const double *rvi = d_P + (size_t)(k * n + i) * 6;
    const int arow = (k * n + i) * adjD;
    for (int d = 0; d < adjD; ++d) {
        const int j = adj_j[arow + d];
        if (j < 0 || adj_w[arow + d] <= 0.0) continue;
        const double *pv = d_P + (size_t)(k * n + j) * 6;
        double d0 = rvi[0]-pv[0], d1 = rvi[1]-pv[1], d2 = rvi[2]-pv[2];
        double r = sqrt(d0*d0 + d1*d1 + d2*d2); if (r < 1e-12) continue;
        double u0 = d0/r, u1 = d1/r, u2 = d2/r, w = 1.0/(sig_isl*sig_isl);
        double H[6] = { u0, u1, u2, 0.0, 0.0, 0.0 };
        #pragma unroll
        for (int a = 0; a < 6; ++a)
            #pragma unroll
            for (int b = 0; b < 6; ++b) S[a*6+b] += w*H[a]*H[b];
    }
    for (int g = 0; g < G; ++g) {
        const int idx = k * G + g;
        if (gts_W[idx] <= 0 || gts_a[idx] != i) continue;
        const double *q = gts_Q + (size_t)idx * 3;
        double d0 = rvi[0]-q[0], d1 = rvi[1]-q[1], d2 = rvi[2]-q[2];
        double r = sqrt(d0*d0 + d1*d1 + d2*d2); if (r < 1e-12) continue;
        double u0 = d0/r, u1 = d1/r, u2 = d2/r, w = 1.0/(sig_gts*sig_gts);
        double H[6] = { u0, u1, u2, 0.0, 0.0, 0.0 };
        #pragma unroll
        for (int a = 0; a < 6; ++a)
            #pragma unroll
            for (int b = 0; b < 6; ++b) S[a*6+b] += w*H[a]*H[b];
    }
    double Pmi[36], A2[36], Ppv[36]; inv6_dev(Pmv, Pmi);
    #pragma unroll
    for (int a = 0; a < 36; ++a) A2[a] = Pmi[a] + S[a];
    inv6_dev(A2, Ppv);
    #pragma unroll
    for (int a = 0; a < 36; ++a) { Pm[a] = Pmv[a]; Pp[a] = Ppv[a]; }
}

// 迭代 1（串行扫描）：每星一线程，k=0..L-1 顺序预测+注入。
__global__ void ssFrameCovSeqKernel(const double *d_P, int n, int L, double dt, double sigma_a,
        const int *adj_j, const double *adj_w, int adjD, double sig_isl,
        int G, const int *gts_a, const double *gts_W, const double *gts_Q, double sig_gts,
        const double *d_P0, double *d_Pm, double *d_Pp) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double P[36];
    #pragma unroll
    for (int a = 0; a < 36; ++a) P[a] = d_P0[i*36+a];
    for (int k = 0; k < L; ++k) {
        double Pm[36], Pp[36];
        frameCovStep_rv(d_P, i, k, n, dt, sigma_a, adj_j, adj_w, adjD, sig_isl,
                        G, gts_a, gts_W, gts_Q, sig_gts, P, Pm, Pp);
        double *po = d_Pm + (size_t)(k*n+i)*36;
        #pragma unroll
        for (int a = 0; a < 36; ++a) po[a] = Pm[a];
        po = d_Pp + (size_t)(k*n+i)*36;
        #pragma unroll
        for (int a = 0; a < 36; ++a) { po[a] = Pp[a]; P[a] = Pp[a]; }
    }
}

// 迭代 ≥2（并行 Jacobi）：每 (k,i) 独立，P_{k-1}^+ 取上一轮 Pprev（k=0 用 d_P0）。
__global__ void ssFrameCovJacKernel(const double *d_P, int n, int L, double dt, double sigma_a,
        const int *adj_j, const double *adj_w, int adjD, double sig_isl,
        int G, const int *gts_a, const double *gts_W, const double *gts_Q, double sig_gts,
        const double *d_P0, const double *d_Pprev, double *d_Pm, double *d_Pp) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * L) return;
    const int k = idx / n, i = idx % n;
    const double *Pprev = (k == 0) ? (d_P0 + (size_t)i*36)
                                   : (d_Pprev + (size_t)((k-1)*n + i)*36);
    double Pm[36], Pp[36];
    frameCovStep_rv(d_P, i, k, n, dt, sigma_a, adj_j, adj_w, adjD, sig_isl,
                    G, gts_a, gts_W, gts_Q, sig_gts, Pprev, Pm, Pp);
    double *po = d_Pm + (size_t)idx*36;
    #pragma unroll
    for (int a = 0; a < 36; ++a) po[a] = Pm[a];
    po = d_Pp + (size_t)idx*36;
    #pragma unroll
    for (int a = 0; a < 36; ++a) po[a] = Pp[a];
}

// 白化 ISL（逐帧 P_k^-，rv 空间）：var = σ² + uᵢPᵢuᵢ + uⱼPⱼuⱼ（位置块）；行 Jacobian=û·(∂p/∂z0)。
__global__ void ssIslAsmCovKernel(const double *d_P, const double *d_A, const double *d_Pm,
        const int *isl_i, const int *isl_jj, const double *isl_D, const double *isl_W,
        int n, int nfr, int E, double sig_isl, int off_isl, double *A_val, double *b) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int)((size_t)nfr * E)) return;
    const int f = idx / E, e = idx % E;
    const int i = isl_i[idx], j = isl_jj[idx];
    const double *pi = d_P + ((size_t)f*n + i)*6;
    const double *pj = d_P + ((size_t)f*n + j)*6;
    const double d0 = pi[0]-pj[0], d1 = pi[1]-pj[1], d2 = pi[2]-pj[2];
    const double rr = sqrt(d0*d0 + d1*d1 + d2*d2);
    const double ir = (rr > 1e-12) ? 1.0/rr : 0.0;
    const double u0 = d0*ir, u1 = d1*ir, u2 = d2*ir;
    const double *Ai = d_A + ((size_t)f*n + i)*18;
    const double *Aj = d_A + ((size_t)f*n + j)*18;
    double Ji[6], Jj[6];
    #pragma unroll
    for (int k = 0; k < 6; ++k) {
        Ji[k] = u0*Ai[0*6+k] + u1*Ai[1*6+k] + u2*Ai[2*6+k];
        Jj[k] = u0*Aj[0*6+k] + u1*Aj[1*6+k] + u2*Aj[2*6+k];
    }
    const double *Pmi = d_Pm + ((size_t)f*n + i)*36;
    const double *Pmj = d_Pm + ((size_t)f*n + j)*36;
    const double uu[3] = { u0, u1, u2 };
    double vi = 0.0, vj = 0.0;
    #pragma unroll
    for (int a = 0; a < 3; ++a) {
        double si = 0.0, sj = 0.0;
        #pragma unroll
        for (int bb = 0; bb < 3; ++bb) { si += Pmi[a*6+bb]*uu[bb]; sj += Pmj[a*6+bb]*uu[bb]; }
        vi += uu[a]*si; vj += uu[a]*sj;
    }
    double var = sig_isl*sig_isl + vi + vj;
    const double wit = isl_W[idx] / sqrt(var > 1e-30 ? var : 1e-30);
    double *av = A_val + off_isl + (size_t)idx*12;
    #pragma unroll
    for (int k = 0; k < 6; ++k) { av[k] = wit*Ji[k]; av[6+k] = -wit*Jj[k]; }
    b[idx] = -(rr - isl_D[idx]) * wit;
}

__global__ void ssGtsAsmCovKernel(const double *d_P, const double *d_A, const double *d_Pm,
        const int *gts_a, const double *gts_D, const double *gts_W, const double *gts_Q,
        int n, int G, int nGTS, double sig_gts, int nISL, int off_gts, double *A_val, double *b) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nGTS) return;
    const int f = idx / G, a = gts_a[idx];
    const double *pf = d_P + ((size_t)f*n + a)*6;
    const double *q = gts_Q + (size_t)idx*3;
    const double d0 = pf[0]-q[0], d1 = pf[1]-q[1], d2 = pf[2]-q[2];
    const double rg = sqrt(d0*d0 + d1*d1 + d2*d2);
    const double ir = (rg > 1e-12) ? 1.0/rg : 0.0;
    const double u0 = d0*ir, u1 = d1*ir, u2 = d2*ir;
    const double *Ag = d_A + ((size_t)f*n + a)*18;
    double Jg[6];
    #pragma unroll
    for (int k = 0; k < 6; ++k) Jg[k] = u0*Ag[0*6+k] + u1*Ag[1*6+k] + u2*Ag[2*6+k];
    const double *Pmg = d_Pm + ((size_t)f*n + a)*36;
    const double uu[3] = { u0, u1, u2 };
    double vg = 0.0;
    #pragma unroll
    for (int aa = 0; aa < 3; ++aa) {
        double s = 0.0;
        #pragma unroll
        for (int bb = 0; bb < 3; ++bb) s += Pmg[aa*6+bb]*uu[bb];
        vg += uu[aa]*s;
    }
    const double var = sig_gts*sig_gts + vg;
    const double wg = gts_W[idx] / sqrt(var > 1e-30 ? var : 1e-30);
    double *av = A_val + off_gts + (size_t)idx*6;
    #pragma unroll
    for (int k = 0; k < 6; ++k) av[k] = wg*Jg[k];
    b[nISL + idx] = -(rg - gts_D[idx]) * wg;
}

// ===================== 单发窗口（win_assemble_ss）host 包装 =====================
void setObsDevice(Ctx &c, const double *hIslD, const double *hIslW,
                  const double *hGtsD, const double *hGtsW, const double *hGtsQ) {
    const size_t nISL = (size_t)c.nfr * c.E, nGTS = (size_t)c.nfr * c.G;
    if (nISL && c.d_isl_D && hIslD) {
        cudaMemcpy(c.d_isl_D, hIslD, nISL * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(c.d_isl_W, hIslW, nISL * sizeof(double), cudaMemcpyHostToDevice);
    }
    if (nGTS && c.d_gts_D && hGtsD) {
        cudaMemcpy(c.d_gts_D, hGtsD, nGTS * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(c.d_gts_W, hGtsW, nGTS * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(c.d_gts_Q, hGtsQ, nGTS * 3 * sizeof(double), cudaMemcpyHostToDevice);
    }
}

void setPriorDevice(Ctx &c, const double *hPriorFac, const double *hZprior, const double *hP0) {
    if (c.d_prior_fac && hPriorFac)
        cudaMemcpy(c.d_prior_fac, hPriorFac, (size_t)c.n * 36 * sizeof(double), cudaMemcpyHostToDevice);
    if (c.d_zprior && hZprior)
        cudaMemcpy(c.d_zprior, hZprior, (size_t)c.n * 6 * sizeof(double), cudaMemcpyHostToDevice);
    if (c.d_P0 && hP0)
        cudaMemcpy(c.d_P0, hP0, (size_t)c.n * 36 * sizeof(double), cudaMemcpyHostToDevice);
}

void setCov0Device(Ctx &c, const double *hP0rv) {
    if (c.d_P0rv && hP0rv)
        cudaMemcpy(c.d_P0rv, hP0rv, (size_t)c.n * 36 * sizeof(double), cudaMemcpyHostToDevice);
}

void setAdjDevice(Ctx &c, const int *hAdjJ, const double *hAdjW, int D, int total) {
    if (c.d_adj_j) { cudaFree(c.d_adj_j); c.d_adj_j = nullptr; }
    if (c.d_adj_w) { cudaFree(c.d_adj_w); c.d_adj_w = nullptr; }
    c.adjD = D;
    const std::size_t sz = (std::size_t)total * D;
    if (sz == 0) return;
    cudaMalloc(&c.d_adj_j, sz * sizeof(int));
    cudaMalloc(&c.d_adj_w, sz * sizeof(double));
    cudaMemcpy(c.d_adj_j, hAdjJ, sz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(c.d_adj_w, hAdjW, sz * sizeof(double), cudaMemcpyHostToDevice);
}

// d_P0(n,36) 广播到 d_PpPrev(nfr,n,36) 的每帧（Jacobi 首扫的入参）。
__global__ void bcastP0Kernel(const double *P0, double *Pp, int n, int nfr) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nfr * n) return;
    const int i = idx % n;
    #pragma unroll
    for (int a = 0; a < 36; ++a) Pp[(size_t)idx*36+a] = P0[(size_t)i*36+a];
}

void ssFrameCovDevice(Ctx &c, int mode) {
    if (!c.d_P || !c.d_Pm || !c.d_Pp) return;
    const int threads = 256;
    const double sig_isl = 1.0 / c.s_isl, sig_gts = 1.0 / c.s_gts;
    const int total = c.nfr * c.n, tblocks = (total + threads - 1) / threads;
    if (mode == 1) {
        const int blocks = (c.n + threads - 1) / threads;
        ssFrameCovSeqKernel<<<blocks, threads>>>(c.d_P, c.n, c.nfr, c.dt, c.sigma_a,
            c.d_adj_j, c.d_adj_w, c.adjD, sig_isl,
            c.G, c.d_gts_a, c.d_gts_W, c.d_gts_Q, sig_gts, c.d_P0rv, c.d_Pm, c.d_Pp);
    } else if (mode == 3) {                                  // 并行 Jacobi 多扫（≈顺序滤波，快）
        bcastP0Kernel<<<tblocks, threads>>>(c.d_P0rv, c.d_PpPrev, c.n, c.nfr);
        for (int s = 0; s < 3; ++s) {
            ssFrameCovJacKernel<<<tblocks, threads>>>(c.d_P, c.n, c.nfr, c.dt, c.sigma_a,
                c.d_adj_j, c.d_adj_w, c.adjD, sig_isl,
                c.G, c.d_gts_a, c.d_gts_W, c.d_gts_Q, sig_gts, c.d_P0rv, c.d_PpPrev, c.d_Pm, c.d_Pp);
            cudaMemcpy(c.d_PpPrev, c.d_Pp, (size_t)total*36*sizeof(double),
                       cudaMemcpyDeviceToDevice);
        }
    } else {
        cudaMemcpy(c.d_PpPrev, c.d_Pp, (size_t)total*36*sizeof(double), cudaMemcpyDeviceToDevice);
        ssFrameCovJacKernel<<<tblocks, threads>>>(c.d_P, c.n, c.nfr, c.dt, c.sigma_a,
            c.d_adj_j, c.d_adj_w, c.adjD, sig_isl,
            c.G, c.d_gts_a, c.d_gts_W, c.d_gts_Q, sig_gts, c.d_P0rv, c.d_PpPrev, c.d_Pm, c.d_Pp);
    }
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) fprintf(stderr, "[ssFrameCovDevice] %s\n", cudaGetErrorString(err));
}

void ssCovDevice(const Ctx &c, const double *hX, double *hCov, double *hRv) {
    if (!c.d_X || !c.d_cov) return;
    cudaMemcpy(c.d_X, hX, (size_t)c.n * 6 * sizeof(double), cudaMemcpyHostToDevice);
    const int threads = 256, blocks = (c.n + threads - 1) / threads;
    ssCovKernel<<<blocks, threads>>>(c.d_X, c.d_P0, c.n, c.nfr, c.dt, c.sigma_a, c.d_P, c.d_cov);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) fprintf(stderr, "[ssCovDevice] %s\n", cudaGetErrorString(err));
    cudaMemcpy(hCov, c.d_cov, (size_t)c.nfr * c.n * 36 * sizeof(double), cudaMemcpyDeviceToHost);
    if (hRv && c.d_P)
        cudaMemcpy(hRv, c.d_P, (size_t)c.nfr * c.n * 6 * sizeof(double), cudaMemcpyDeviceToHost);
}

// ===================== L 节点批量窗口（每帧一节点，ncol=6NL）：帧间 dyn 连续性装配 =====================
// 每 (k,i)，k=0..L-2：e = x_{k+1,i} - end_{k,i}（QOE 6 维）；行 = qingt·e，b=-行；
// Jacobian 块 = qingt·[-Φ, I]（12 非零：节点 k 的 -Φ，节点 k+1 的 I）。行序 = (k,i,a)，与 _lift_struct 一致。
__global__ void winDynAsmKernel(const double *d_X, const double *d_end, const double *d_phi,
                                const double *qingt, int n, int L, int off_dyn, int base_b,
                                double *A_val, double *b) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;      // (k*n+i)
    const int nk = (L - 1) * n;
    if (idx >= nk) return;
    const double *xnext = d_X + ((size_t)(idx + n)) * 6;        // (k+1,i)
    const double *xe = d_end + (size_t)idx * 6;
    const double *Ph = d_phi + (size_t)idx * 36;
    double e[6];
#pragma unroll
    for (int c = 0; c < 6; ++c) e[c] = xnext[c] - xe[c];
    for (int a = 0; a < 6; ++a) {
        const size_t row = (size_t)idx * 6 + a;
        double *av = A_val + off_dyn + row * 12;
        double bb = 0.0;
#pragma unroll
        for (int j = 0; j < 6; ++j) bb += qingt[a * 6 + j] * e[j];
        b[base_b + row] = -bb;
#pragma unroll
        for (int cc = 0; cc < 6; ++cc) {                        // 节点 k 列：-Φ 加权
            double s = 0.0;
#pragma unroll
            for (int j = 0; j < 6; ++j) s += qingt[a * 6 + j] * (-Ph[j * 6 + cc]);
            av[cc] = s;
        }
#pragma unroll
        for (int cc = 0; cc < 6; ++cc) av[6 + cc] = qingt[a * 6 + cc];   // 节点 k+1 列：I 加权
    }
}

double wt_dyn = 0.0, wt_asm = 0.0, wt_res2 = 0.0;
void winResetTiming() { wt_dyn = wt_asm = wt_res2 = 0.0; }
std::vector<double> winGetTiming() { return { wt_dyn, wt_asm, wt_res2 }; }

double winAssembleDevice(WinCtx &c, const double *hX, const double *hRv, const double *hAp,
                         double *A_val, double *b) {
    using clk = std::chrono::steady_clock;
    auto _now = [] { return std::chrono::duration<double>(clk::now().time_since_epoch()).count(); };
    const size_t bRv = (size_t)c.L * c.n * 6 * sizeof(double);
    const size_t bAp = (size_t)c.L * c.n * 18 * sizeof(double);
    cudaMemcpy(c.d_X, hX, bRv, cudaMemcpyHostToDevice);
    cudaMemcpy(c.d_rv, hRv, bRv, cudaMemcpyHostToDevice);
    cudaMemcpy(c.d_Ap, hAp, bAp, cudaMemcpyHostToDevice);

    const int threads = 256;
    const int nk = (c.L - 1) * c.n;
    double td = _now();
    if (nk > 0) gveStmDevice(c.d_X, nk, c.dt, 1.0, c.d_end, c.d_phi);   // 单步 φ + Φ（同一 3/8-RK4）
    wt_dyn += _now() - td;

    double ta = _now();
    const int nISL = c.L * c.E, nGTS = c.L * c.G;
    if (nISL > 0)
        jointIslAsmKernel<<<(nISL + threads - 1) / threads, threads>>>(
            c.d_rv, c.d_Ap, c.d_isl_i, c.d_isl_jj, c.d_isl_D, c.d_isl_W,
            c.n, c.L, c.E, c.s_isl, c.off_isl, A_val, b);
    if (nGTS > 0)
        jointGtsAsmKernel<<<(nGTS + threads - 1) / threads, threads>>>(
            c.d_rv, c.d_Ap, c.d_gts_a, c.d_gts_D, c.d_gts_W, c.d_gts_Q,
            c.n, c.G, nGTS, c.s_gts, nISL, c.off_gts, A_val, b);
    const int nDYN = nk * 6;
    if (nDYN > 0)
        winDynAsmKernel<<<(nk + threads - 1) / threads, threads>>>(
            c.d_X, c.d_end, c.d_phi, c.d_qingt, c.n, c.L, c.off_dyn, nISL + nGTS, A_val, b);
    const int ncol = c.ncol;
    if (ncol > 0)
        jointDampKernel<<<(ncol + threads - 1) / threads, threads>>>(
            c.d_X, c.d_ctr, ncol, c.issq, c.off_damp, nISL + nGTS + nDYN, A_val, b);
    cudaDeviceSynchronize();
    wt_asm += _now() - ta;

    double tr = _now();
    cudaMemset(c.d_res, 0, sizeof(double));
    if (nISL > 0)
        jointRes2Kernel<<<(nISL + threads - 1) / threads, threads>>>(
            c.d_rv, c.d_isl_i, c.d_isl_jj, c.d_isl_D, c.d_isl_W, c.n, c.E, nISL, c.d_res);
    double h = 0.0;
    cudaMemcpy(&h, c.d_res, sizeof(double), cudaMemcpyDeviceToHost);
    wt_res2 += _now() - tr;
    return h;
}

}  // namespace qoejopt
