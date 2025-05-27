#include <math.h>
#include <chrono>
#include <stdexcept>
#include <iostream>
#include <Eigen/Dense>
#include "kepler.h"
#include "elements.h"

// #include <utility>
// #include <functional>

using std::function;
using std::runtime_error;
using Eigen::Vector3d;

namespace kepler {
    double period(double sma, double mu) {
        return M_PI * 2 * sqrt(pow(sma, 3) / mu);
    }

    double newton(
            double p0,
            function<double(double)> const &func,
            function<double(double)> const &deriv,
            int maxiter = 50,
            double tol = 1e-8
    ) {
        for (auto i = 1; i < maxiter; i++) {
            auto p = p0 - func(p0) / deriv(p0);
            if (fabs(p - p0) < tol) {
                return p;
            }
            p0 = p;
        }
        throw runtime_error("Not converged.");
    }

    double mean2ecc(double M, double ecc) {
        auto E = newton(M, [ecc, M](double E) -> double {
            return E - ecc * sin(E) - M;
        }, [ecc](double E) -> double {
            return 1 - ecc * cos(E);
        });
        return E;
    }

    double ecc2true(double E, double ecc) {
        return 2 * atan2(sqrt(1 + ecc) * sin(E / 2), sqrt(1 - ecc) * cos(E / 2));
    }

    void benchmark(int times) {
        auto mu = 3.986004418e5;
        Vector3d r(8.59072560e+02, -4.13720368e+03, 5.29556871e+03);
        Vector3d v(7.37289205e+00, 2.08223573e+00, 4.39999794e-01);
        auto el = elements::elements(r, v, mu);

        auto best = std::numeric_limits<double>::infinity();
        auto worst = -std::numeric_limits<double>::infinity();
        double all = 0;
        for (auto i=0; i < times; i++) {
            auto begin = std::chrono::high_resolution_clock::now();

            mean2ecc(M_PI, el[1]);

            auto end = std::chrono::high_resolution_clock::now();
            auto current = std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()/1e9;
            all += current;
            if (current < best) {
                best = current;
            }
            if (current > worst) {
                worst = current;
            }
        }
        std::cout << "[" << all/times << "," << best << "," << worst << "]" << std::endl;
    }
    
    // Computes the satellite state vector from osculating Keplerian elements for elliptic orbits
    Eigen::VectorXd State(double gm, const Eigen::VectorXd& Kep, double dt) {
        // Keplerian elements at epoch
        double a = Kep(0), Omega = Kep(3);
        double e = Kep(1), omega = Kep(4);
        double i = Kep(2), M0 = Kep(5);

        // Mean anomaly
        double M;
        if (dt == 0) {
            M = M0;
        } else {
            double n = sqrt(gm / (a * a * a));
            M = M0 + n * dt;
        }

        // Eccentric anomaly
        double E = kepler::mean2ecc(M, e);
        double cosE = cos(E);
        double sinE = sin(E);

        // Perifocal coordinates
        double fac = sqrt((1 - e) * (1 + e));

        double R = a * (1 - e * cosE);  // Distance
        double V = sqrt(gm * a) / R;    // Velocity

        Eigen::Vector3d r(a * (cosE - e), a * fac * sinE, 0);
        Eigen::Vector3d v(-V * sinE, V * fac * cosE, 0);

        // Transformation to reference system (Gaussian vectors)
        Eigen::Matrix3d PQW = R_z(-Omega) * R_x(-i) * R_z(-omega);
        // Eigen::Matrix3d PQW = R_z(Omega) * R_x(i) * R_z(omega);

        r = PQW * r;
        v = PQW * v;

        Eigen::VectorXd Y(6);
        Y << r(0), r(1), r(2), v(0), v(1), v(2);
        return Y;
    }


    // Rotation matrices
    Eigen::Matrix3d R_x(double angle) {
        double C = cos(angle);
        double S = sin(angle);
        Eigen::Matrix3d rotmat;
        rotmat << 1.0, 0.0, 0.0,
                0.0, C, S,
                0.0, -S, C;
        return rotmat;
    }

    Eigen::Matrix3d R_y(double angle) {
        double C = cos(angle);
        double S = sin(angle);
        Eigen::Matrix3d rotmat;
        rotmat << C, 0.0, -S,
                0.0, 1.0, 0.0,
                S, 0.0, C;
        return rotmat;
    }

    Eigen::Matrix3d R_z(double angle) {
        double C = cos(angle);
        double S = sin(angle);
        Eigen::Matrix3d rotmat;
        rotmat << C, S, 0.0,
                -S, C, 0.0,
                0.0, 0.0, 1.0;
        return rotmat;
    }
}

namespace kep3
{

/// Lagrangian propagation
/**
 * This function propagates an initial Cartesian state for a time t assuming a
 * central body and a keplerian motion. Lagrange coefficients are used as basic
 * numerical technique. All units systems can be used, as long
 * as the input parameters are all expressed in the same system.
 */
Vector6d propagate_lagrangian(
    //std::pair<std::array<std::array<double, 3>, 2>, std::optional<std::array<double, 36>>>
    const Vector3d &r0, const Vector3d &v0,
    const double tof, const double mu, bool stm,
    Eigen::Ref<Eigen::Matrix<double, 6, 6>> Phi0f
    )
{
    // const auto &[r0, v0] = pos_vel0;
    auto rf = r0,vf = v0;
    // auto rf = r0.clone().data(),vf = v0.data();
    double R0 = std::sqrt(r0[0] * r0[0] + r0[1] * r0[1] + r0[2] * r0[2]);
    double Rf = 0.;
    double V02 = v0[0] * v0[0] + v0[1] * v0[1] + v0[2] * v0[2];
    double DX = 0.;
    double energy = (V02 / 2 - mu / R0);
    double a = -mu / 2.0 / energy; // will be negative for hyperbolae
    double sqrta = 0.;
    double F = 0., G = 0., Ft = 0., Gt = 0.;
    double s0 = 0., c0 = 0.;
    double sigma0 = (r0[0] * v0[0] + r0[1] * v0[1] + r0[2] * v0[2]) / std::sqrt(mu);

    if (a > 0) { // Solve Kepler's equation in DE, elliptical case
        sqrta = std::sqrt(a);
        double DM = std::sqrt(mu / std::pow(a, 3)) * tof;
        double sinDM = std::sin(DM), cosDM = std::cos(DM);
        // Here we use the atan2 to recover the mean anomaly difference in the
        // [0,2pi] range. This makes sure that for high value of M no catastrophic
        // cancellation occurs, as would be the case using std::fmod(DM, 2pi)
        double DM_cropped = std::atan2(sinDM, cosDM);
        if (DM_cropped < 0) {
            DM_cropped += 2 * M_PI;
        }
        s0 = sigma0 / sqrta;
        c0 = (1 - R0 / a);
        // This initial guess was developed applying Lagrange expansion theorem to
        // the Kepler's equation in DE. We stopped at 3rd order.
        double IG = DM_cropped + c0 * sinDM - s0 * (1 - cosDM)
                    + (c0 * cosDM - s0 * sinDM) * (c0 * sinDM + s0 * cosDM - s0)
                    + 0.5 * (c0 * sinDM + s0 * cosDM - s0)
                          * (2 * std::pow(c0 * cosDM - s0 * sinDM, 2)
                             - (c0 * sinDM + s0 * cosDM - s0) * (c0 * sinDM + s0 * cosDM));

        // Solve Kepler Equation for ellipses in DE (eccentric anomaly difference)
        const int digits = std::numeric_limits<double>::digits;
        uint max_iter = 100u;
        // NOTE: Halley iterates may result into instabilities (specially with a
        // poor IG)
// 作为函数句柄求解方程的三种实现
        std::function<kep3::Cf1f2 (double)> f = [DM_cropped, sigma0, sqrta, a, R0](double DE) {
            return kepDE_dKepDE(DE, DM_cropped, sigma0, sqrta, a, R0);
        };
        double DE = newton_raphson_iterate<decltype(f), double>(f,IG, IG - M_PI, IG + M_PI, digits, max_iter);

        // double DE = newton_raphson_iterate(
        //     [DM_cropped, sigma0, sqrta, a, R0](double DE)->kep3::Cf1f2 {
        //         return kepDE_dKepDE(DE, DM_cropped, sigma0, sqrta, a, R0);
        //     },
        //     IG, IG - M_PI, IG + M_PI, digits, max_iter);

        // double DE = kep3::newton_raphson_iterate(
        //     [DM_cropped, sigma0, sqrta, a, R0](double DE) {
        //         return std::make_tuple(
        //             kepDE(DE, DM_cropped, sigma0, sqrta, a, R0),
        //             d_kepDE(DE, sigma0, sqrta, a, R0)
        //         );
        //     },
        //     IG, IG - M_PI, IG + M_PI, digits, max_iter
        // );

        // LCOV_EXCL_START
        if (max_iter == 100u) {
            printf("Maximum number of iterations exceeded when solving Kepler's equation for the eccentric anomaly in propagate_lagrangian.\n  DM=%6.3f\nsigma0=%6.3f\nsqrta=%6.3f\na=%6.3f\nR=%6.3f\nDE=%6.3f",
                DM, sigma0, sqrta, a, R0, DE);
            throw std::domain_error("error");
        }
        // LCOV_EXCL_STOP
        Rf = a + (R0 - a) * std::cos(DE) + sigma0 * sqrta * std::sin(DE);

        // Lagrange coefficients
        F = 1 - a / R0 * (1 - std::cos(DE));
        G = a * sigma0 / std::sqrt(mu) * (1 - std::cos(DE)) + R0 * std::sqrt(a / mu) * std::sin(DE);
        Ft = -std::sqrt(mu * a) / (Rf * R0) * std::sin(DE);
        Gt = 1 - a / Rf * (1 - std::cos(DE));
        DX = DE;
    } else { // Solve Kepler's equation in DH, hyperbolic case
        std::cout<<"Error, Hyperbola. \n";
    }

    for (auto i = 0u; i < 3; i++) {
        rf[i] = F * r0[i] + G * v0[i];
        vf[i] = Ft * r0[i] + Gt * v0[i];
    }
    // 使用 Eigen::Map 将数组转换为向量
    // Eigen::Map<Vector6d> pos_velf(rf.data());
    // pos_velf.tail<3>() = Eigen::Map<Vector3d>(vf.data());
    Vector6d pos_velf;
    pos_velf.head<3>() = rf;pos_velf.tail<3>() = vf;

    if (stm) {
        Phi0f = kep3::stm_lagrangian(r0,v0, tof, mu, R0, Rf, energy, sigma0, a, s0, c0, DX, F, G, Ft, Gt);
    } 
    return pos_velf;
}
Cf1f2 kepDE_dKepDE(double DE, double DM_cropped, double sigma0, double sqrta, double a, double R0){
    Cf1f2 rt;
    rt.f = kepDE(DE, DM_cropped, sigma0, sqrta, a, R0); rt.df =  d_kepDE(DE, sigma0, sqrta, a, R0);
    return rt;
}

template <class F, class T>
T newton_raphson_iterate(F f, T guess, T min, T max, int digits, uint& max_iter)
{
   T f0(0), f1, last_f0(0);
   T result = guess;

   T factor = static_cast<T>(ldexp(1.0, 1 - digits));
   T delta = 1e6;
   T delta1 = 1e6;
   T delta2 = 1e6;

   uint count = max_iter;
#ifdef BOOST_MATH_INSTRUMENT
	 std::cout << "Newton_raphson_iterate, guess = " << guess << ", min = " << min << ", max = " << max 
		 << ", digits = " << digits << ", max_iter = " << max_iter << std::endl;
#endif
   do{
      last_f0 = f0;
      delta2 = delta1;
      delta1 = delta;
      Cf1f2 f1f2 = f(result);
      f0 = f1f2.f; f1 = f1f2.df;
        //   detail::unpack_tuple(f(result), f0, f1);
      --count;
      if(0 == f0)
         break;
      if(f1 == 0)
      {
#ifdef BOOST_MATH_INSTRUMENT
         std::cout << "Newton iteration, zero derivative found!" << std::endl;
#endif
         // Oops zero derivative!!!
         kep3::handle_zero_derivative(f, last_f0, f0, delta, result, guess, min, max);
      }
      else
      {
         delta = f0 / f1;
      }
#ifdef BOOST_MATH_INSTRUMENT
      std::cout << "Newton iteration " << max_iter - count << ", delta = " << delta << std::endl;
#endif
      if(fabs(delta * 2) > fabs(delta2))
      {
         // Last two steps haven't converged.
         T shift = (delta > 0) ? (result - min) / 2 : (result - max) / 2;
         if ((result != 0) && (fabs(shift) > fabs(result)))
         {
            delta = sign(delta) * fabs(result) * 0.9; // Protect against huge jumps!
            //delta = sign(delta) * result; // Protect against huge jumps! Failed for negative result. https://github.com/boostorg/math/issues/216
         }
         else
            delta = shift;
         // reset delta1/2 so we don't take this branch next time round:
         delta1 = 3 * delta;
         delta2 = 3 * delta;
      }
      guess = result;
      result -= delta;
      if(result <= min)
      {
         delta = 0.5F * (guess - min);
         result = guess - delta;
         if((result == min) || (result == max))
            break;
      }
      else if(result >= max)
      {
         delta = 0.5F * (guess - max);
         result = guess - delta;
         if((result == min) || (result == max))
            break;
      }
      // Update brackets:
      if(delta > 0)
         max = guess;
      else
         min = guess;
   }while(count && (fabs(result * factor) < fabs(delta)));

   max_iter -= count;
#ifdef BOOST_MATH_INSTRUMENT
   std::cout << "Newton Raphson final iteration count = " << max_iter << std::endl;

   static uint max_count = 0;
   if(max_iter > max_count)
   {
      max_count = max_iter;
      // std::cout << "Maximum iterations: " << max_iter << std::endl;
	    // Puzzled what this tells us, so commented out for now?
   }
#endif
   return result;
}

template <class F, class T>
void handle_zero_derivative(F f,
                            T& last_f0,
                            const T& f0,
                            T& delta,
                            T& result,
                            T& guess,
                            const T& min,
                            const T& max)
{
   if(last_f0 == 0)
   {
      // this must be the first iteration, pretend that we had a
      // previous one at either min or max:
      if(result == min)
      {
         guess = max;
      }
      else
      {
         guess = min;
      }
      Cf1f2 f1f2= f(guess);last_f0 = f1f2.f;
    //   unpack_0(f(guess), last_f0);
      delta = guess - result;
   }
   if(sign(last_f0) * sign(f0) < 0)
   {
      // we've crossed over so move in opposite direction to last step:
      if(delta < 0)
      {
         delta = (result - min) / 2;
      }
      else
      {
         delta = (result - max) / 2;
      }
   }
   else
   {
      // move in same direction as last step:
      if(delta < 0)
      {
         delta = (result - max) / 2;
      }
      else
      {
         delta = (result - min) / 2;
      }
   }
}

// Here we take the lagrangian coefficient expressions for rf and vf as function of r0 and v0, and manually,
// differentiate it to obtain the state transition matrix.
mat66 stm_lagrangian(const Vector3d &r0, const Vector3d &v0, double tof, // NOLINT
                                      double mu,                                                        // NOLINT
                                      double R0, double Rf, double energy,                              // NOLINT
                                      double sigma0,                                                    // NOLINT
                                      double a, double s0, double c0,                                   // NOLINT
                                      double DX, double F, double G, double Ft, double Gt)
{
    // Create xtensor fixed arrays from input (we avoid adapt as its slower in this case since all is fixed size)
    // We use row vectors (not column) as its then more conventional for gradients as the differential goes to the
    // end

    // We seed the gradients with the initial dr0/dx0 and dv0/dx0
    mat36 dr0 = Eigen::Matrix<double, 3, 6>::Zero();
    mat36 dv0 = Eigen::Matrix<double, 3, 6>::Zero();
    dr0(0, 0) = 1;
    dr0(1, 1) = 1;
    dr0(2, 2) = 1;
    dv0(0, 3) = 1;
    dv0(1, 4) = 1;
    dv0(2, 5) = 1;

    // 1 - We start computing the differentials of basic quantities. A differential for a scalar will be a 16 mat
    // (gradient).
    double sqrtmu = std::sqrt(mu);
    Vector6dr dV02 = 2. * _dot(v0, dv0);
    Vector6dr dR0 = 1. / R0 * _dot(r0, dr0);
    Vector6dr denergy = 0.5 * dV02 + mu / R0 / R0 * dR0;
    Vector6dr dsigma0 = ((_dot(r0, dv0) + _dot(v0, dr0))) / sqrtmu;
    Vector6dr da = mu / 2. / energy / energy * denergy; // a = -mu / 2 / energy
    Vector6dr dF, dFt, dG, dGt;

    if (a > 0) { // ellipses
        double sqrta = std::sqrt(a);
        double sinDE = std::sin(DX);
        double cosDE = std::cos(DX);

        Vector6dr ds0 = dsigma0 / sqrta - 0.5 * sigma0 / sqrta / sqrta / sqrta * da; // s0 = sigma0 / sqrta
        Vector6dr dc0 = -1. / a * dR0 + R0 / a / a * da;                             // c0 = (1- R/a)
        Vector6dr dDM = -1.5 * sqrtmu * tof / std::pow(sqrta, 5) * da;               // M = sqrt(mu/a**3) tof
        Vector6dr dDE = (dDM - (1 - cosDE) * ds0 + sinDE * dc0) / (1 + s0 * sinDE - c0 * cosDE);
        Vector6dr dRf = (1 - cosDE + 0.5 / sqrta * sigma0 * sinDE) * da + cosDE * dR0
                    + (sigma0 * sqrta * cosDE - (R0 - a) * sinDE) * dDE
                    + sqrta * sinDE * dsigma0; // r = a + (r0 - a) * cosDE + sigma0 * sqrta * sinDE

        // 2 - We may now compute the differentials of the Lagrange coefficients
        dF = -(1 - cosDE) / R0 * da + a / R0 / R0 * (1 - cosDE) * dR0 - a / R0 * sinDE * dDE;
        dG = (1 - F) * (R0 * dsigma0 + sigma0 * dR0) - (sigma0 * R0) * dF + (sqrta * R0 * cosDE) * dDE
             + (sqrta * sinDE) * dR0 + (0.5 * R0 * sinDE / sqrta) * da; // sqrtmu G = sigma0 r0 (1-F) + r0 sqrta sinDE
        dG = dG / sqrtmu;
        dFt = (-sqrta / R0 / Rf * cosDE) * dDE - (0.5 / sqrta / R0 / Rf * sinDE) * da
              + (sqrta / Rf / R0 / R0 * sinDE) * dR0 + (sqrta / Rf / Rf / R0 * sinDE) * dRf;
        dFt = dFt * sqrtmu;
        dGt = -(1 - cosDE) / Rf * da + a / Rf / Rf * (1 - cosDE) * dRf - a / Rf * sinDE * dDE;
    } else { // hyperbolas (sqrta is sqrt(-a))
        double sqrta = std::sqrt(-a);
        double sinhDH = std::sinh(DX); // DX is here the hyperbolic anomaly.
        double coshDH = std::cosh(DX);

        Vector6dr ds0 = dsigma0 / sqrta + 0.5 * sigma0 / sqrta / sqrta / sqrta * da; // s0 = sigma0 / sqrta
        Vector6dr dc0 = -1. / a * dR0 + R0 / a / a * da;                             // c0 = (1- R/a)
        Vector6dr dDN = 1.5 * sqrtmu * tof / std::pow(sqrta, 5) * da;                // N = sqrt(-mu/a**3) tof
        Vector6dr dDH = (dDN - (coshDH - 1) * ds0 - sinhDH * dc0) / (s0 * sinhDH + c0 * coshDH - 1);
        Vector6dr dRf = (1 - coshDH - 0.5 / sqrta * sigma0 * sinhDH) * da + coshDH * dR0
                    + (sigma0 * sqrta * coshDH + (R0 - a) * sinhDH) * dDH
                    + sqrta * sinhDH * dsigma0; // r = a + (r0 - a) * coshDH + sigma0 * sqrta * sinhDH

        // 2 - We may now compute the differentials of the Lagrange coefficients
        dF = -(1 - coshDH) / R0 * da + a / R0 / R0 * (1 - coshDH) * dR0 + a / R0 * sinhDH * dDH;
        dG = (1 - F) * (R0 * dsigma0 + sigma0 * dR0) - (sigma0 * R0) * dF + (sqrta * R0 * coshDH) * dDH
             + (sqrta * sinhDH) * dR0
             - (0.5 * R0 * sinhDH / sqrta) * da; // sqrtmu G = sigma0 r0 (1-F) + r0 sqrta sinhDH
        dG = dG / sqrtmu;
        dFt = (-sqrta / R0 / Rf * coshDH) * dDH + (0.5 / sqrta / R0 / Rf * sinhDH) * da
              + (sqrta / Rf / R0 / R0 * sinhDH) * dR0 + (sqrta / Rf / Rf / R0 * sinhDH) * dRf;
        dFt = dFt * sqrtmu;
        dGt = -(1 - coshDH) / Rf * da + a / Rf / Rf * (1 - coshDH) * dRf + a / Rf * sinhDH * dDH;
    }
    // 3 - And finally assemble the state transition matrix
    mat36 Mr = F * dr0 + _dot(r0.transpose(), dF) + G * dv0 + _dot(v0.transpose(), dG);
    mat36 Mv = Ft * dr0 + _dot(r0.transpose(), dFt) + Gt * dv0 + _dot(v0.transpose(), dGt);
    mat66 M{};
    // 将两个 mat36 赋值给 mat66 的左上角和右下角
    M.topLeftCorner<3, 6>() = Mr;
    M.bottomRightCorner<3, 6>() = Mv;
    return M;
}

}// kep3

namespace osculating{
    // https://github.com/decenter2021/osculating2mean/
Eigen::VectorXd rv2OEOsc(const Eigen::VectorXd &x) {
    // Input
    Eigen::Vector3d r0 = x.segment<3>(0);
    Eigen::Vector3d v0 = x.segment<3>(3);
    
    // Compute classical orbit elements 
    // a
    double a = -(MU / 2) / ((v0.norm() * v0.norm()) / 2 - MU / r0.norm());
    
    // Eccentricity vector 
    Eigen::Vector3d e_vec = ((v0.norm() * v0.norm() - MU / r0.norm()) * r0 - r0.dot(v0) * v0) / MU;
    double e = e_vec.norm();
    
    // Angular momentum vector
    Eigen::Vector3d h = r0.cross(v0);
    
    // Line of nodes unit vector
    Eigen::Vector3d n = Eigen::Vector3d::UnitZ().cross(h);
    n.normalize();
    
    Eigen::Vector3d n_cross_h = h.normalized().cross(n);
    n_cross_h.normalize();
    
    // Compute OE
    double omega, nu, E, M, u;
    
    if (e < 1e3 * std::numeric_limits<double>::epsilon()) {
        omega = 0;
        nu = std::acos(n.dot(r0.normalized()));
        if (r0.normalized().dot(h.normalized().cross(n)) < 0) {
            nu = 2 * M_PI - nu;
        }
        E = 2 * std::atan(std::sqrt((1 - e) / (1 + e)) * std::tan(nu / 2));
        M = E - e * std::sin(E);
        u = M + omega;
    } else {
        omega = std::acos(n.dot(e_vec / e));
        if (e_vec(2) < 0) {
            omega = 2 * M_PI - omega;
        }
        nu = std::acos((e_vec / e).dot(r0.normalized()));
        if (r0.dot(v0) < 0) {
            nu = 2 * M_PI - nu;
        }
        E = 2 * std::atan(std::sqrt((1 - e) / (1 + e)) * std::tan(nu / 2));
        M = E - e * std::sin(E);
        u = M + omega;
    }
    
    if (u > 2 * M_PI) {
        u = u - std::floor(u / (2 * M_PI)) * 2 * M_PI;
    } else if (u < 0) {
        u = u + std::ceil(-u / (2 * M_PI)) * 2 * M_PI;
    }
    
    double ex = n.dot(e_vec);
    double ey = n_cross_h.dot(e_vec);
    double i = std::acos(h(2) / h.norm());
    double Omega = std::acos(n(0));
    if (n(1) < 0) {
        Omega = 2 * M_PI - Omega;
    }
    if (Omega > 2 * M_PI) {
        Omega = Omega - std::floor(Omega / (2 * M_PI)) * 2 * M_PI;
    } else if (Omega < 0) {
        Omega = Omega + std::ceil(-Omega / (2 * M_PI)) * 2 * M_PI;
    }
    
    // Output vector
    Eigen::VectorXd OE(6);
    OE << a, u, ex, ey, i, Omega;
    
    return OE;
}

double KepEqtnE(double M, double e, int MaxIt = 100, double epsl = 1e-13) {
    double E_n1;
    if ((M > -M_PI && M < 0) || M > M_PI) {
        E_n1 = M - e;
    } else {
        E_n1 = M + e;
    }

    int count = 0;
    while (true) {
        double E_n = E_n1;
        E_n1 = E_n + (M - E_n + e * std::sin(E_n)) / (1 - e * std::cos(E_n));
        if (std::abs(E_n1 - E_n) < epsl) {
            return E_n1;
        }else{
            if(0)
                printf("E_n1 = %1.6f\n",E_n1);
        }
        count++;
        if (count >= MaxIt) {
            std::cerr << "Warning: Maximum number of iterations for KepEqtnE reached." << std::endl;
            return E_n1;
        }
    }
}

Eigen::VectorXd OEOsc2rv(const Eigen::VectorXd &OE, int MaxIt = 100, double epsl = 1e-5) {
    double a = OE(0);
    double u = OE(1);
    double ex = OE(2);
    double ey = OE(3);
    double i = OE(4);
    double Omega = OE(5);
    double e = std::sqrt(ex * ex + ey * ey);
    double p = a * (1 - e * e);

    double omega, nu;
    if (e < 1e-5) {
        omega = 0;
        nu = u;
    } else {
        omega = std::atan2(ey, ex);
        double M = u - omega;
        // Fix angle difference
        if (M < -M_PI) {
            M += std::floor(std::abs(M - M_PI) / (2 * M_PI)) * 2 * M_PI;
        } else if (M > M_PI) {
            M -= std::floor((M + M_PI) / (2 * M_PI)) * 2 * M_PI;
        }
        double E = KepEqtnE(M, e, MaxIt, epsl);
        nu = 2 * std::atan(std::sqrt((1 + e) / (1 - e)) * std::tan(E / 2));
    }

    Eigen::Vector3d rPQW;
    rPQW << p * std::cos(nu) / (1 + e * std::cos(nu)), 
            p * std::sin(nu) / (1 + e * std::cos(nu)), 
            0;
    
    Eigen::Vector3d vPQW;
    vPQW << -std::sqrt(MU / p) * std::sin(nu), 
            std::sqrt(MU / p) * (e + std::cos(nu)), 
            0;

    Eigen::Matrix3d T;
    T << std::cos(Omega) * std::cos(omega) - std::sin(Omega) * std::sin(omega) * std::cos(i),
         -std::cos(Omega) * std::sin(omega) - std::sin(Omega) * std::cos(omega) * std::cos(i),
         std::sin(Omega) * std::sin(i),
         std::sin(Omega) * std::cos(omega) + std::cos(Omega) * std::sin(omega) * std::cos(i),
         -std::sin(Omega) * std::sin(omega) + std::cos(Omega) * std::cos(omega) * std::cos(i),
         -std::cos(Omega) * std::sin(i),
         std::sin(omega) * std::sin(i),
         std::cos(omega) * std::sin(i),
         std::cos(i);

    Eigen::VectorXd x(6);
    x.segment<3>(0) = T * rPQW;
    x.segment<3>(3) = T * vPQW;

    return x;
}
void testOEosc() {
    // Example usage
    Eigen::VectorXd x(6);
    x << 6925443.952, 190432.624, 230986.901, -303.938541, 2277.904454, 7229.098276;

    Eigen::VectorXd OE = rv2OEOsc(x);
    
    std::cout << "Orbital Elements (a, u, ex, ey, i, Omega):" << std::endl;
    std::cout << OE.transpose() << std::endl;

// // Example usage
//     Eigen::VectorXd OE(6);
//     OE << 7000e3, 0.1, 0.01, 0.01, 0.1, 0.1;

    Eigen::VectorXd rv = OEOsc2rv(OE);
    
    std::cout << "Position and Velocity Vector:" << std::endl;
    std::cout << rv.transpose() << std::endl;
}

//输入: 非奇异轨道元素向量 OEMean。
// 输出: Eckstein-Ustinov 摄动的 6 元素向量 EUPerturbations。
Eigen::VectorXd EcksteinUstinovPerturbations(const Eigen::VectorXd &OEMean) {
    // Process input parameters
    double a0 = OEMean(0);
    double lambda_0 = OEMean(1);
    double l0 = OEMean(2);  // ex
    double h0 = OEMean(3);  // ey
        double e0 = std::sqrt(OEMean(2) * OEMean(2) + OEMean(3) * OEMean(3));
    double i0 = OEMean(4);
    double Omega_0 = OEMean(5);

    // Compute parameters
    double G2 = -J2 * std::pow(RE / a0, 2);
    double beta_0 = std::sin(i0);
    double lambda_star = 1 - (3.0 / 2.0) * G2 * (3 - 4 * beta_0 * beta_0);
    double xi_0 = std::cos(i0);

    // Compute Eckstein-Ustinov perturbations
    double da = -(3.0 / 2.0) * (a0 / lambda_star) * G2 * ((2 - (7.0 / 2.0) * beta_0 * beta_0) * l0 * std::cos(lambda_0) +
               (2 - (5.0 / 2.0) * beta_0 * beta_0) * h0 * std::sin(lambda_0) + beta_0 * beta_0 * std::cos(2 * lambda_0) +
               (7.0 / 2.0) * beta_0 * beta_0 * (l0 * std::cos(3 * lambda_0) + h0 * std::sin(3 * lambda_0))) +
               (3.0 / 4.0) * a0 * G2 * G2 * beta_0 * beta_0 * (7 * (2 - 3 * beta_0 * beta_0) * std::cos(2 * lambda_0) +
               beta_0 * beta_0 * std::cos(4 * lambda_0));

    double dh = -(3.0 / (2 * lambda_star)) * G2 * ((1 - (7.0 / 4.0) * beta_0 * beta_0) * std::sin(lambda_0) +
              (1 - 3 * beta_0 * beta_0) * l0 * std::sin(2 * lambda_0) +
              ((-3.0 / 2.0) + 2 * beta_0 * beta_0) * h0 * std::cos(2 * lambda_0) +
              (7.0 / 12.0) * beta_0 * beta_0 * std::sin(3 * lambda_0) +
              (17.0 / 8.0) * beta_0 * beta_0 * (l0 * std::sin(4 * lambda_0) - h0 * std::cos(4 * lambda_0)));

    double dl = -(3.0 / (2 * lambda_star)) * G2 * ((1 - (5.0 / 4.0) * beta_0 * beta_0) * std::cos(lambda_0) +
              (1.0 / 2.0) * (3 - 5 * beta_0 * beta_0) * l0 * std::cos(2 * lambda_0) +
              (2 - (3.0 / 2.0) * beta_0 * beta_0) * h0 * std::sin(2 * lambda_0) +
              (7.0 / 12.0) * beta_0 * beta_0 * std::cos(3 * lambda_0) +
              (17.0 / 8.0) * beta_0 * beta_0 * (l0 * std::cos(4 * lambda_0) + h0 * std::sin(4 * lambda_0)));

    double di = -(3.0 / (4 * lambda_star)) * G2 * beta_0 * xi_0 * (-l0 * std::cos(lambda_0) + h0 * std::sin(lambda_0) +
              std::cos(2 * lambda_0) + (7.0 / 3.0) * l0 * std::cos(3 * lambda_0) + (7.0 / 3.0) * h0 * std::sin(3 * lambda_0));

    double dOmega = (3.0 / (2 * lambda_star)) * G2 * xi_0 * ((7.0 / 2.0) * l0 * std::sin(lambda_0) -
                  (5.0 / 2.0) * h0 * std::cos(lambda_0) - (1.0 / 2.0) * std::sin(2 * lambda_0) -
                  (7.0 / 6.0) * l0 * std::sin(3 * lambda_0) + (7.0 / 6.0) * h0 * std::cos(3 * lambda_0));

    double dlambda = -(3.0 / (2 * lambda_star)) * G2 * ((10 - (119.0 / 8.0) * beta_0 * beta_0) * l0 * std::sin(lambda_0) +
                  ((85.0 / 8.0) * beta_0 * beta_0 - 9) * h0 * std::cos(lambda_0) +
                  (2 * beta_0 * beta_0 - (1.0 / 2.0)) * std::sin(2 * lambda_0) +
                  ((-7.0 / 6.0) + (119.0 / 24.0) * beta_0 * beta_0) * (l0 * std::sin(3 * lambda_0) - h0 * std::cos(3 * lambda_0)) -
                  (3 - (21.0 / 4.0) * beta_0 * beta_0) * l0 * std::sin(lambda_0) +
                  (3 - (15.0 / 4.0) * beta_0 * beta_0) * h0 * std::cos(lambda_0) -
                  (3.0 / 4.0) * beta_0 * beta_0 * std::sin(2 * lambda_0) -
                  (21.0 / 12.0) * beta_0 * beta_0 * (l0 * std::sin(3 * lambda_0) - h0 * std::cos(3 * lambda_0)));

    // Output
    Eigen::VectorXd EUPerturbations(6);
    EUPerturbations << da, dlambda, dl, dh, di, dOmega;

    return EUPerturbations;
}
//使用 OEOsc2rv 函数计算位置-速度向量 x。
// 初始化 mean 轨道元素 OEMean 为 osculating 元素。
Eigen::VectorXd OEOsc2OEMeanEU(const Eigen::VectorXd &OEosc, int MaxIt, double epslPos, double epslVel) {
    // Compute position-velocity vector
//     接受 osculating 轨道元素 OEosc，最大迭代次数 MaxIt，位置误差 epslPos 和速度误差 epslVel 作为参数。
// 在循环中计算 Eckstein-Ustinov 摄动，并更新和修正角度范围。
    Eigen::VectorXd x = OEOsc2rv(OEosc);

    // Initialization: Mean elements are equal to osculating elements 
    Eigen::VectorXd OEMean = OEosc;

    // Iterate the Eckstein-Ustinov corrections
    for (int i = 0; i < MaxIt; ++i) {
        // Compute perturbation
        Eigen::VectorXd EUPerturbation = EcksteinUstinovPerturbations(OEMean);

        // Update and fix angle ranges
        Eigen::VectorXd OEoscIt = OEMean + EUPerturbation;
        OEoscIt(1) = fmod(OEoscIt(1), 2 * M_PI);
        OEoscIt(5) = fmod(OEoscIt(5), 2 * M_PI);

        // Update and fix angle ranges for mean elements
        OEMean = OEosc - EUPerturbation;
        OEMean(1) = fmod(OEMean(1), 2 * M_PI);
        OEMean(5) = fmod(OEMean(5), 2 * M_PI);

        // Check stopping criterion
        Eigen::VectorXd xIt = OEOsc2rv(OEoscIt);
        double posError = (xIt.segment<3>(0) - x.segment<3>(0)).norm();
        double velError = (xIt.segment<3>(3) - x.segment<3>(3)).norm();

        if (posError < epslPos && velError < epslVel) {
            break;
        }

        if (i == MaxIt - 1) {
            throw std::runtime_error("Maximum number of iterations reached for OEOsc2OEMeanEU.");
        }
    }

    return OEMean;
}

// Function to convert mean elements to osculating elements with perturbations
Eigen::VectorXd OEMeanEU2OEOsc(const Eigen::VectorXd &OEMean) {
    // Compute perturbation
    Eigen::VectorXd EUPerturbation = EcksteinUstinovPerturbations(OEMean);
    
    // Update osculating elements
    Eigen::VectorXd OEosc = OEMean + EUPerturbation;
    
    // Fix angles (assuming the angles are in radians)
    if (OEosc(1) > 2 * M_PI)
        OEosc(1) -= floor(OEosc(1) / (2 * M_PI)) * (2 * M_PI);
    else if (OEosc(1) < 0)
        OEosc(1) += ceil(-OEosc(1) / (2 * M_PI)) * (2 * M_PI);
    
    if (OEosc(5) > 2 * M_PI)
        OEosc(5) -= floor(OEosc(5) / (2 * M_PI)) * (2 * M_PI);
    else if (OEosc(5) < 0)
        OEosc(5) += ceil(-OEosc(5) / (2 * M_PI)) * (2 * M_PI);
    
    return OEosc;
}

int testOEMean() {
    Eigen::VectorXd x(6);
    x << 6925443.952, 190432.624, 230986.901, -303.938541, 2277.904454, 7229.098276;

    Eigen::VectorXd OEosc = rv2OEOsc(x);
std::cout << "Osculatimg Orbital Elements by RV:" << std::endl;
        std::cout << OEosc.transpose() << std::endl;

    // Example usage
    Eigen::VectorXd OEMean = OEOsc2OEMeanEU(OEosc);
std::cout << "Mean Orbital Elements converted:" << std::endl;
        std::cout << OEMean.transpose() << std::endl;

std::cout << "----------------------" << std::endl;

Eigen::VectorXd OEosc2 = OEMeanEU2OEOsc(OEMean);
std::cout << "Osculatimg Orbital Elements converted:" << std::endl;
        std::cout << OEosc2.transpose() << std::endl;

        std::cout << "===================" << std::endl;
        Eigen::VectorXd rv = OEOsc2rv(OEosc2);
    
    std::cout << "Position and Velocity Vector:" << std::endl;
    std::cout << rv.transpose() << std::endl;

    std::cout << "Conversion errpo:" << std::endl;
    for(int i=0;i<6;i++)std::cout << rv[i]-x[i] << std::endl;

    return 0;
}

int testAll(){
    Eigen::MatrixXd x(10,6);
    x<<
        6925443.952 , 190432.624 , 230986.901 , -303.938541 , 2277.904454 , 7229.098276 ,
        6892287.168165196 , 326597.9811288715 , 663922.8343111346 , -800.8863777932581 , 2259.3097423724694 , 7196.913409921307 ,
        6829407.366007369 , 461354.8897545596 , 1093995.6187574891 , -1294.3461537385338 , 2230.9734445513236 , 7133.695357598292 ,
        6737078.405941692 , 594122.3964290187 , 1519351.0089608086 , -1782.1688216724021 , 2193.0207849047265 , 7039.725194243644 ,
        6615702.360106298 , 724328.3450485808 , 1938155.7286206526 , -2262.231346188976 , 2145.619664773123 , 6915.420524948245 ,
        6465807.65765267 , 851411.9139328094 , 2348605.6035639415 , -2732.446475188839 , 2088.9798431280215 , 6761.33341519237 ,
        6288046.6494301725 , 974826.097551565 , 2748933.5507154725 , -3190.7723130224194 , 2023.3519118265497 , 6578.147660306201 ,
        6083192.60555117 , 1094040.120765485 , 3137417.383841274 , -3635.221642626703 , 1949.0260720111414 , 6366.67541179209 ,
        5852136.162424382 , 1208541.7738766817 , 3512387.3981167097 , -4063.870946267338 , 1866.3307194502058 , 6127.85318248683 ,
        5595881.238803249 , 1317839.6572921136 , 3872233.6970134056 , -4474.869076379788 , 1775.6308479127215 , 5862.737256750369;

    double Ts = 10; // (orbits)

    // Compute osculating orbital elements
    Eigen::MatrixXd OE_osc(6, x.rows());
    for (int t = 0; t < x.rows(); ++t) {
        OE_osc.col(t) = rv2OEOsc(x.row(t));
        printf("R: %7.1f,%7.1f,%7.1f \t",x.row(t)[0],x.row(t)[1],x.row(t)[2]);
        printf("OE_osc:a, lambda, ex, ey, i, Omega: %7.1f,%1.2f,%1.4f,%1.4f ,%1.3f,%1.3f  \n",OE_osc.col(t)[0],OE_osc.col(t)[1],OE_osc.col(t)[2],OE_osc.col(t)[3],OE_osc.col(t)[4],OE_osc.col(t)[5]);
    }

    // Compute mean orbital elements using Eckstein-Ustinov method
    Eigen::MatrixXd OE_mean_EcksteinUstinov(6, x.rows());
    for (int t = 0; t < x.rows(); ++t) {
        OE_mean_EcksteinUstinov.col(t) = OEOsc2OEMeanEU(OE_osc.col(t));
    }

    
    // Compute parameters
    double semiMajorAxis = OE_osc.row(0).mean();
    double incl = OE_osc.row(4).mean();
    double n = sqrt(MU / pow(semiMajorAxis, 3));
    double gamma = (J2 / 2) * pow(RE / semiMajorAxis, 2);
    double Omega_dot = -3 * gamma * n * cos(incl);
    double arg_perigee_dot = (3.0 / 2) * gamma * n * (5 * pow(cos(incl), 2) - 1);
    double M_dot = (3.0 / 2) * gamma * n * (3 * pow(cos(incl), 2) - 1);
    double u_dot = n + M_dot + arg_perigee_dot;

    // Output the computed parameters
    std::cout << "Semi-major Axis: " << semiMajorAxis << std::endl;
    std::cout << "Inclination: " << incl << std::endl;
    std::cout << "Omega_dot: " << Omega_dot << std::endl;
    std::cout << "arg_perigee_dot: " << arg_perigee_dot << std::endl;
    std::cout << "M_dot: " << M_dot << std::endl;
    std::cout << "u_dot: " << u_dot << std::endl;

    return 0;
}
// 定义求阶乘的函数
double factorial(int n) {
    if (n == 0) return 1;
    double result = 1;
    for (int i = 1; i <= n; ++i) {
        result *= i;
    }
    return result;
}

// 定义伽马函数的近似计算
double gammaFun(int n) {
    return factorial(n - 1);
}

// 实现第一类虚变量的贝塞尔函数
double modified_bessel_first_kind(int n, double x) {
    double sum = 0.0;
    for (int k = 0; k < 100; ++k) { // 我们取前100项求和
        double term = std::pow(x / 2.0, 2 * k + n) / (factorial(k) * gammaFun(n + k + 1));
        sum += term;
        if (term < 1e-10) { // 当项的大小足够小时停止迭代
            break;
        }
    }
    return sum;
}

// OE: a, lambda (mean anomaly + arg perigee), ex, ey, i, longitude of asceding node
//    cond_c(1)= a,    cond_c(2)= lambda,    cond_c(4)= q1,    cond_c(5)= q2,  
//    cond_c(3)= i  cond_c(6)= Omega
Eigen::VectorXd OscElemsLongpropagate( double tf, const Eigen::VectorXd OEm, double RE=osculating::RE, double mu=osculating::MU, double tol=1e-9,double J2=osculating::J2,double J3=osculating::J3,double J4=osculating::J4){
    Eigen::VectorXd OE_osc(6);

    // Constants
    double gamma = 3 * J2 * RE * RE;
    double t0 = 0;
    // double tf = t(1);
    // double q10 = ex,q20=ey,argLat0=u;
    double a0 = OEm(0), lambda0 = OEm(1), q10 = OEm(2), q20 = OEm(3);
    double i0 = OEm(4), RAAN0 = OEm(5);

    // Initial values
    double n0 = sqrt(mu / (a0 * a0 * a0));
        double e2 = q10 * q10 + q20 * q20; // e_x^2 + e_y^2
    double p0 = a0 * (1 -e2);
    double gamma_J2 = J2 * pow(RE / p0, 2);
    double gamma_J3 = J3 * pow(RE / p0, 3);
    double eta0 = sqrt(1 - e2);

    // 定义一些中间变量

    double eta = sqrt(1 - e2); // sqrt(1 - (e_x^2 + e_y^2))
    double sin_i = sin(i0); // sin(i)
    double sin_i2 = sin_i * sin_i; // sin^2(i)
    double sin2_i = sin(2 * i0); // sin(2i)
    double cos_i = cos(i0); // cos(i)
    double cos_i2 = cos_i * cos_i; // cos^2(i)
    double tan_i = tan(i0); // tan(i)
    double e = sqrt(e2); // sqrt(e_x^2 + e_y^2)
    double e_x2_minus_e_y2 = q10 * q10 - q20 * q20; // e_x^2 - e_y^2

    // 导数公式
    // Secular variations by J2
    // 导数公式
    // J2
    double aDot = 0;
    double argLatDot = (3.0 / 4.0) * n0 * gamma_J2 * (eta * (3 * cos_i2 - 1) + (5 * cos_i2 - 1));
    double q10Dot = -(3.0 / 4.0) * n0 * gamma_J2 * q20 * (5 * cos_i2 - 1);
    double q20Dot = (3.0 / 4.0) * n0 * gamma_J2 * q10 * (5 * cos_i2 - 1);
    double iDot = 0;
    double RAANDot = -(3.0 / 2.0) * n0 * gamma_J2 * cos_i;

    // J3
    aDot += 0;

    argLatDot += (3.0 / 8.0) * n0 * gamma_J3 * (
        ((4 - 5 * sin_i2) * ((sin_i2 - e2 * cos_i2) / (e * sin_i)) + 2 * sin_i * (13 - 15 * sin_i2) * e) * (q20 / e) 
        - sin_i * (4 - 5 * sin_i2) * ((1 - 4 * e2) / e2) * q20 * eta
    );

    q10Dot += -(3.0 / 8.0) * n0 * gamma_J3 * (
        sin_i * (4 - 5 * sin_i2) * (1 - e2) * (q10 * q10 / e2) 
        + ((4 - 5 * sin_i2) * ((sin_i2 - e2 * cos_i2) / (e * sin_i)) + 2 * sin_i * (13 - 15 * sin_i2) * e) * (q20 * q20 / e)
    );

    q20Dot += -(3.0 / 8.0) * n0 * gamma_J3 * (
        sin_i * (4 - 5 * sin_i2) * (1 - e2) * (q10 * q20 / e2) 
        - ((4 - 5 * sin_i2) * ((sin_i2 - e2 * cos_i2) / (e * sin_i)) + 2 * sin_i * (13 - 15 * sin_i2) * e) * (q10 * q20 / e)
    );

    iDot += (3.0 / 8.0) * n0 * gamma_J3 * cos_i * (4 - 5 * sin_i2) * q10;

    RAANDot += -(3.0 / 8.0) * n0 * gamma_J3 * (15 * sin_i2 - 4) * q20 / tan_i;

    double gamma_J4 = J4 * pow(RE / p0, 4);

// J4
aDot += 0;

argLatDot += -(45.0 / 128.0) * n0 * gamma_J4 * (
    (8 - 40 * sin_i2 + 35 * sin_i2 * sin_i2) * e2 * eta
    - (2.0 / 3.0) * sin_i2 * (6 - 7 * sin_i2) * (2 - 5 * e2) * eta * (e_x2_minus_e_y2 / e2)
    + (4.0 / 3.0) * (
        (16 - 62 * sin_i2 + 49 * sin_i2 * sin_i2 + 0.75 * (24 - 84 * sin_i2 + 63 * sin_i2 * sin_i2) * e2)
        + (sin_i2 * (6 - 7 * sin_i2) - 0.5 * (12 - 70 * sin_i2 + 63 * sin_i2 * sin_i2) * e2) * (e_x2_minus_e_y2 / e2)
    )
);

q10Dot += -(15.0 / 32.0) * n0 * gamma_J4 * (
    sin_i2 * (6 - 7 * sin_i2) * (1 - e2) * (2 * q10 * q20 * q20 / e2)
    - (16 - 62 * sin_i2 + 49 * sin_i2 * sin_i2 + 0.75 * (24 - 84 * sin_i2 + 63 * sin_i2 * sin_i2) * e2
    + (sin_i2 * (6 - 7 * sin_i2) - 0.5 * (12 - 70 * sin_i2 + 63 * sin_i2 * sin_i2) * e2) * (e_x2_minus_e_y2 / e2)) * q20
);

q20Dot += -(15.0 / 32.0) * n0 * gamma_J4 * (
    sin_i2 * (6 - 7 * sin_i2) * (1 - e2) * (2 * q20 * q20 * q10 / e2)
    + (16 - 62 * sin_i2 + 49 * sin_i2 * sin_i2 + 0.75 * (24 - 84 * sin_i2 + 63 * sin_i2 * sin_i2) * e2
    + (sin_i2 * (6 - 7 * sin_i2) - 0.5 * (12 - 70 * sin_i2 + 63 * sin_i2 * sin_i2) * e2) * (e_x2_minus_e_y2 / e2)) * q10
);

iDot += (15.0 / 64.0) * n0 * gamma_J4 * sin2_i * (6 - 7 * sin_i2) * 2 * q10 * q20;

RAANDot += (15.0 / 16.0) * n0 * gamma_J4 * cos_i * (
    (4 - 7 * sin_i2) * (1 + 1.5 * e2)
    - (3 - 7 * sin_i2) * e_x2_minus_e_y2
);

    // Drag


    double a = a0 + aDot * (tf - t0);
    double lambda = lambda0+(argLatDot + n0)*(tf-t0);
    double q1 = q10+q10Dot*(tf-t0);
    double q2 = q20+q20Dot*(tf-t0);
    double i = i0 + iDot * (tf - t0);
    double Omega = RAAN0 + RAANDot * (tf - t0);

    OE_osc<<  a, lambda, q1, q2, i, Omega;
return OE_osc;
}
// double rho0 = 3.003075e-4, double CDA_m=0.0044,
Eigen::VectorXd OscElemsLongDrag_BCrho0( const Eigen::VectorXd OEm,double dt,
    double rp0 = 535e3, double H0 =65.35644970323516e3, double mu=osculating::MU){
    Eigen::VectorXd dOE_osc(6);
    // double tf = t(1);
    // double q10 = ex, q20=ey, lambda0=u=Mean anomaly + omega;
    double a0 = OEm(0), lambda0 = OEm(1), q10 = OEm(2), q20 = OEm(3);
    double i0 = OEm(4), RAAN0 = OEm(5);

    double n0 = sqrt(mu / (a0 * a0 * a0));
    double e2 = q10 * q10 + q20 * q20; // e_x^2 + e_y^2
    double p0 = a0 * (1 - e2);
    double eta = sqrt(1 - e2); // sqrt(1 - (e_x^2 + e_y^2))

    double sin_i = sin(i0); // sin(i)
    double sin_i2 = sin_i * sin_i; // sin^2(i)
    double omega = std::atan2(q20, q10);
    double sin_w2 = sin(omega)*sin(omega); // sin^2(i)

    double e = sqrt(e2); // sqrt(e_x^2 + e_y^2)
    double z;
    double theta= oscstm::lam2theta(lambda0, q10, q20, 1e-6,z); 
    
    // double Hp0 = a0*(1-e)-(1-osculating::epsilon*sin_i2*sin_w2);
    // // double rp0 = p0 / (1 + q10 * cos(theta) + q20 * sin(theta));
    // double H = rp0-osculating::RE;
    z = a0*e/H0;

    double eeee = exp(-(a0-rp0)/H0);
    double I0z  = osculating::modified_bessel_first_kind(0,z);
    
    double da = -a0*a0 *I0z* eeee*n0 * dt;
    double dM = 3.0/4.0 /a0*da *(n0 * dt );
    dOE_osc << da,dM,0,0,0,0;
    return dOE_osc;
}

Eigen::Matrix3d PQW2GCRF(double Omega, double omega, double i) {
    Eigen::Matrix3d T;

    // 计算三角函数值
    double cos_Omega = cos(Omega);
    double sin_Omega = sin(Omega);
    double cos_omega = cos(omega);
    double sin_omega = sin(omega);
    double cos_i = cos(i);
    double sin_i = sin(i);

    // 填充矩阵
    T(0, 0) = cos_Omega * cos_omega - sin_Omega * sin_omega * cos_i;
    T(0, 1) = -cos_Omega * sin_omega - sin_Omega * cos_omega * cos_i;
    T(0, 2) = sin_Omega * sin_i;
    T(1, 0) = sin_Omega * cos_omega + cos_Omega * sin_omega * cos_i;
    T(1, 1) = -sin_Omega * sin_omega + cos_Omega * cos_omega * cos_i;
    T(1, 2) = -cos_Omega * sin_i;
    T(2, 0) = sin_omega * sin_i;
    T(2, 1) = cos_omega * sin_i;
    T(2, 2) = cos_i;

    return T;
}
Eigen::MatrixXd Jacobian_RV2OscElems(const Eigen::VectorXd OE, const Eigen::VectorXd &RV,  double mu){

    // Constants (these should be defined or calculated appropriately)
    double a=OE(0);
    double lambda0 = OE(1),xi = OE(2), eta = OE(3);
    double i = OE(4),Omega = OE(5);
    double omega = std::atan2(eta, xi);
    double Ew;
    double u = oscstm::lam2theta(lambda0,xi,eta,tol,Ew);
    double nu = u - omega;
    double e2 = xi * xi + eta * eta; // e_x^2 + e_y^2
    double e = sqrt(e2); // sqrt(e_x^2 + e_y^2)

    Eigen::MatrixXd M(6, 6);
    Eigen::Matrix3d p;
    p = PQW2GCRF(Omega,omega,i);

    // Helper variables
    double cos_nu = cos(nu);
    double sin_nu = sin(nu);
    double cos_omega = cos(omega);
    double sin_omega = sin(omega);
    double cos_Omega = cos(Omega);
    double sin_Omega = sin(Omega);
    double cos_i = cos(i);
    double sin_i = sin(i);
    double one_minus_e2 = 1 - e * e;
    double one_plus_e_cos_nu = 1 + e * cos_nu;
    double a_sqrt_mu_one_minus_e2 = sqrt(mu / (a * one_minus_e2));

    double term1 = (1 - e * e) / one_plus_e_cos_nu;
    double term2 = - (2 * a * e + a * cos_nu + a * cos_nu * e * e) / (one_plus_e_cos_nu * one_plus_e_cos_nu);
    double term3 = a * one_minus_e2 / one_plus_e_cos_nu;
    double term4 = (sin_nu * sin_omega + cos_nu * cos_omega);

    double term5 = 1 / (2 * a) * a_sqrt_mu_one_minus_e2;
    double term6 = 1 / one_minus_e2 * a_sqrt_mu_one_minus_e2;

    // Define left side of the matrix (6x3)
    M(0, 0) = term1 * (cos_nu * p(0, 0) + sin_nu * p(0, 1));
    M(0, 1) = term2 * (cos_nu * p(0, 0) + sin_nu * p(0, 1));
    M(0, 2) = term3 * p(0, 2) * term4;

    M(1, 0) = term1 * (cos_nu * p(1, 0) + sin_nu * p(1, 1));
    M(1, 1) = term2 * (cos_nu * p(1, 0) + sin_nu * p(1, 1));
    M(1, 2) = term3 * p(1, 2) * term4;

    M(2, 0) = term1 * (cos_nu * p(2, 0) + sin_nu * p(2, 1));
    M(2, 1) = term2 * (cos_nu * p(2, 0) + sin_nu * p(2, 1));
    M(2, 2) = term3 * cos_i * term4;

    M(3, 0) = term5 * (sin_nu * p(0, 0) - (e + cos_nu) * p(0, 1));
    M(3, 1) = term6 * (e * sin_nu * p(0, 0) + (e + cos_nu) * p(0, 1));
    M(3, 2) = a_sqrt_mu_one_minus_e2 * sin_Omega * (sin_nu * p(2, 0) + (e + cos_nu) * p(2, 1));

    M(4, 0) = term5 * (sin_nu * p(1, 0) - (e + cos_nu) * p(1, 1));
    M(4, 1) = term6 * (e * sin_nu * p(1, 0) + (e + cos_nu) * p(1, 1));
    M(4, 2) = a_sqrt_mu_one_minus_e2 * cos_Omega * (sin_nu * p(2, 0) - (e + cos_nu) * p(2, 1));

    M(5, 0) = term5 * (sin_nu * p(2, 0) - (e + cos_nu) * p(2, 1));
    M(5, 1) = term6 * (e * sin_nu * p(2, 0) + (e + cos_nu) * p(2, 1));
    M(5, 2) = -a_sqrt_mu_one_minus_e2 * (sin_nu * sin_omega * cos_i + (e + cos_nu) * cos_omega * cos_i);

    // Define right side of the matrix (6x3)
    M(0, 3) = term3 * (-cos_nu * p(1, 0) - sin_nu * p(1, 1));
    M(0, 4) = term3 * (cos_nu * p(0, 1) - sin_nu * p(0, 0));
    M(0, 5) = term3 * (sin_nu * p(0, 1) - (e + cos_nu) * p(0, 0)) / one_plus_e_cos_nu;

    M(1, 3) = term3 * (cos_nu * p(1, 0) + sin_nu * p(1, 1));
    M(1, 4) = term3 * (cos_nu * p(1, 1) - sin_nu * p(1, 0));
    M(1, 5) = term3 * (sin_nu * p(1, 1) - (e + cos_nu) * p(1, 0)) / one_plus_e_cos_nu;

    M(2, 3) = 0;
    M(2, 4) = term3 * sin_i * (cos_nu * cos_omega - sin_nu * sin_omega);
    M(2, 5) = term3 * (sin_nu * p(2, 1) - (e + cos_nu) * p(2, 0)) / one_plus_e_cos_nu;

    M(3, 3) = a_sqrt_mu_one_minus_e2 * (sin_nu * p(1, 0) - (e + cos_nu) * p(1, 1));
    M(3, 4) = a_sqrt_mu_one_minus_e2 * (sin_nu * p(0, 1) + (e + cos_nu) * p(1, 0));
    M(3, 5) = a_sqrt_mu_one_minus_e2 * (-cos_nu * p(0, 0) - sin_nu * p(0, 1));

    M(4, 3) = a_sqrt_mu_one_minus_e2 * (-sin_nu * p(0, 0) + (e + cos_nu) * p(0, 1));
    M(4, 4) = a_sqrt_mu_one_minus_e2 * (-sin_nu * p(1, 1) - (e + cos_nu) * p(1, 0));
    M(4, 5) = a_sqrt_mu_one_minus_e2 * (-cos_nu * p(1, 0) - sin_nu * p(1, 1));

    M(5, 3) = 0;
    M(5, 4) = a_sqrt_mu_one_minus_e2 * (-sin_nu * p(2, 1) - (e + cos_nu) * p(2, 0));
    M(5, 5) = a_sqrt_mu_one_minus_e2 * (-cos_nu * p(2, 0) - sin_nu * p(2, 1));

    return M;
}
Eigen::MatrixXd STM_ONsElemsWarppedByCOE(const Eigen::VectorXd OE, double tf, double Re, double mu, double J2){
    // Constants (these should be defined or calculated appropriately)
    double a=OE(0);
    double lambda0 = OE(1),xi = OE(2), eta = OE(3);
    double i = OE(4),Omega = OE(5);
    double omega = std::atan2(eta, xi);
    double e2 = xi * xi + eta * eta; // e_x^2 + e_y^2
    double e = sqrt(e2); // sqrt(e_x^2 + e_y^2)
    double p = a * (1 - e2);
    double n= sqrt(mu / (a * a * a));
    
    // Define trigonometric terms
    double cos_omega = cos(omega);
    double sin_omega = sin(omega);
    double e_cos2_sin2 = e * (cos_omega * cos_omega + sin_omega * sin_omega);

    // Define matrix J
    Eigen::MatrixXd J(6, 6);
    J << 1, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 1, 1,
         0, cos_omega, 0, 0, -e * sin_omega, 0,
         0, sin_omega, 0, 0, e * cos_omega, 0,
         0, 0, 1, 0, 0, 0,
         0, 0, 0, 1, 0, 0;

    // Define matrix JInv
    Eigen::MatrixXd JInv(6, 6);
    JInv << 1, 0, 0, 0, 0, 0,
            0, 0, e * cos_omega / e_cos2_sin2, e * sin_omega / e_cos2_sin2, 0, 0,
            0, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 1,
            0, 0, -sin_omega / e_cos2_sin2, cos_omega / e_cos2_sin2, 0, 0,
            0, 1, sin_omega / e_cos2_sin2, -cos_omega / e_cos2_sin2, 0, 0;
    
    // Constants (these should be defined or calculated appropriately)
    double Omega_1, omega_1, M_1;

    // Compute the Omega_1, omega_1, and M_1 values
    Omega_1 = -3.0 * J2 * n * cos(i) / (2 * p * p);
    omega_1 = 3.0 * J2 * n * (2 - 2.5 * sin(i) * sin(i)) / (2 * p * p);
    M_1 = 3.0 * J2 * n * (1 - 1.5 * sin(i) * sin(i)) * sqrt(1 - e * e) / (2 * p * p);

    // Create the matrix with the given values
    Eigen::Matrix3d phi1;
    phi1(0, 0) = -7.0 / (2 * a) * Omega;
    phi1(0, 1) = 4 * e / (1 - e * e) * Omega;
    phi1(0, 2) = -tan(i) * Omega;
    phi1(1, 0) = -7.0 / (2 * a) * omega_1;
    phi1(1, 1) = 4 * e / (1 - e * e) * omega_1;
    phi1(1, 2) = -5 * sin(2 * i) / (4 - 5 * sin(i) * sin(i)) * omega_1;
    phi1(2, 0) = -7.0 / (2 * a) * M_1;
    phi1(2, 1) = 3 * e / (1 - e * e) * M_1;
    phi1(2, 2) = -3 * sin(2 * i) / (2 - 3 * sin(i) * sin(i)) * M_1;

    // Construct the 6x6 matrix
    Eigen::MatrixXd Phi_coe(6, 6);
    Phi_coe.block<3, 3>(3, 0) = phi1 * tf;
    Phi_coe(5,0) =-3.0/2.0*n/a * tf;

    Phi_coe = Phi_coe + Eigen::MatrixXd::Identity(6, 6);

    Eigen::MatrixXd Phi_nse(6, 6);
    Phi_nse = J*Phi_coe*JInv;
    return Phi_nse;
}
int testOEMeanSTM(){
    double J2 = osculating::J2;
    double Re = osculating::RE;
    double mu = osculating::MU;
    double tol = 1e-10; // Tolerance
    double    tf = 3600.0; // Example time vector (t0, tf)
    
    Eigen::VectorXd x(6);
    x << 6925443.952, 190432.624, 230986.901, -303.938541, 2277.904454, 7229.098276;

    Eigen::VectorXd OE = osculating::rv2OEOsc(x);
    Eigen::VectorXd OEMean = osculating::OEOsc2OEMeanEU(OE);
    std::cout << "Mean Elements ref: " << OEMean.transpose() << std::endl;

    Eigen::MatrixXd PhiOE(6,6);
    PhiOE = STM_ONsElemsWarppedByCOE(OEMean,3600,RE, MU,J2);
    std::cout << ">>State Transition Matrix phi_J2:\n"
              << PhiOE << std::endl;
    Eigen::VectorXd OEf(6);
    // OEf = OscElemsLongpropagate(3600, OEMean,RE, MU, 1e-9,J2,J3,J4);
    OEf = OscElemsLongpropagate(3600, OEMean,RE, MU, 1e-9,J2,0,0);
    std::cout << "Mean Elements ref: " << OEf.transpose() << std::endl;

    return 0;
}
}

namespace oscstm{
// OE   平纬度幅角 M + w
// Osc  纬度幅角   f + w
// 这两种轨道要素在圆轨道下仍然是奇异的
Eigen::VectorXd OE2Osc(const Eigen::VectorXd &OE){
// % Non-singular orbital elements are employed: 
//      a: semi-major axis
//      lambda: mean anomaly + argument of perigee
//      ex: e*cos(argument of perigee) xi
//      ey: e*sin(argument of perigee) eta
//      i: inclination
//      Omega: longitude of ascending node
///////////////////////////////////////////////
// nsElems(1) = a
// nsElems(2) = w + f = theat = argLat;
// nsElems(3) = i;
// nsElems(4) = e*cos(w) = q1;
// nsElems(5) = e*sin(w) = q2;
// nsElems(6) = Omega;
    Eigen::VectorXd ICSc(6);
    double F;
    double theta= lam2theta(OE[1],OE[2],OE[3],1e-6,F); 
    ICSc<< OE[0],theta,OE[4],OE[2],OE[3],OE[5];
    return ICSc;
}
Eigen::VectorXd Osc2OE(const Eigen::VectorXd &Osc){
    Eigen::VectorXd OE(6);
    double lambda=theta2lam(Osc[0],Osc[1],Osc[3],Osc[4]);
    OE<< Osc[0],lambda,Osc[3],Osc[4],Osc[2],Osc[5];
    return OE;
}

// Calculation of true longitude theta = f + w
// from mean longitude lambda = M + w
// input :
//    lambda = mean longitude = M + w
//    q1 = e * cos(w)
//    q2 = e * sin(w)
// output :
//    theta = true longitude = f + w
//    F = eccentric longitude = E + w
double lam2theta(double lambda, double q1, double q2, double Tol, double &F) {
    double eta = std::sqrt(1 - q1 * q1 - q2 * q2);
    double beta = 1 / (eta * (1 + eta));

    // Modified Kepler's Equation
    F = lambda;
    double FF = 1;
    while (std::abs(FF) > Tol) {
        FF = lambda - (F - q1 * std::sin(F) + q2 * std::cos(F));
        double dFFdF = -(1 - q1 * std::cos(F) - q2 * std::sin(F));
        double del_F = -FF / dFFdF;
        F = F + del_F;
    }

    // True Longitude
    double num = (1 + eta) * (eta * std::sin(F) - q2) + q2 * (q1 * std::cos(F) + q2 * std::sin(F));
    double den = (1 + eta) * (eta * std::cos(F) - q1) + q1 * (q1 * std::cos(F) + q2 * std::sin(F));
    double theta = std::atan2(num, den);
    
    while (theta < 0) {
        theta += 2 * M_PI;
    }
    while (theta >= 2 * M_PI) {
        theta -= 2 * M_PI;
    }

    if (lambda < 0) {
        int kk_plus = 0;
        int quad_plus = 0;
        while (lambda < 0) {
            kk_plus++;
            lambda += 2 * M_PI;
        }
        if (lambda < M_PI / 2 && theta > M_PI) {
            quad_plus = 1;
        } else if (theta < M_PI / 2 && lambda > M_PI) {
            quad_plus = -1;
        }
        theta = theta - (kk_plus + quad_plus) * 2 * M_PI;
    } else {
        int kk_minus = 0;
        int quad_minus = 0;
        while (lambda >= 2 * M_PI) {
            kk_minus++;
            lambda -= 2 * M_PI;
        }
        if (lambda < M_PI / 2 && theta > M_PI) {
            quad_minus = -1;
        } else if (theta < M_PI / 2 && lambda > M_PI) {
            quad_minus = 1;
        }
        theta = theta + (kk_minus + quad_minus) * 2 * M_PI;
    }
    return theta;
}

// Calculation of mean longitude lambda = M + w
// from true longitude theta = f + w
// input :
//    a = semi major axis
//    theta = true longitude
//    q1 = e * cos(w)
//    q2 = e * sin(w)
// output :
//    lambda = mean longitude
double theta2lam(double a, double theta, double q1, double q2) {
    double eta = std::sqrt(1 - q1 * q1 - q2 * q2);
    double beta = 1 / (eta * (1 + eta));
    double R = (a * eta*eta) / (1 + q1 * std::cos(theta) + q2 * std::sin(theta));

    double num = R * (1 + beta * q1 * q1) * std::sin(theta) - beta * R * q1 * q2 * std::cos(theta) + a * q2;
    double den = R * (1 + beta * q2 * q2) * std::cos(theta) - beta * R * q1 * q2 * std::sin(theta) + a * q1;

    double F = std::atan2(num, den);
    double lambda = F - q1 * std::sin(F) + q2 * std::cos(F);

    while (lambda < 0) {
        lambda += 2 * M_PI;
    }
    while (lambda >= 2 * M_PI) {
        lambda -= 2 * M_PI;
    }

    if (theta < 0) {
        int kk_plus = 0;
        int quad_plus = 0;
        while (theta < 0) {
            kk_plus++;
            theta += 2 * M_PI;
        }
        if (theta < M_PI / 2 && lambda > M_PI) {
            quad_plus = 1;
        } else if (lambda < M_PI / 2 && theta > M_PI) {
            quad_plus = -1;
        }
        lambda -= (kk_plus + quad_plus) * 2 * M_PI;
    } else {
        int kk_minus = 0;
        int quad_minus = 0;
        while (theta >= 2 * M_PI) {
            kk_minus++;
            theta -= 2 * M_PI;
        }
        if (theta < M_PI / 2 && lambda > M_PI) {
            quad_minus = -1;
        } else if (lambda < M_PI / 2 && theta > M_PI) {
            quad_minus = 1;
        }
        lambda += (kk_minus + quad_minus) * 2 * M_PI;
    }
    return lambda;
}

bool debuggingOrn=  true;
// formation matrix D_J2 in closed form
// between mean and osculating new set of elements with the perturbation by only J2
// input :
//    mean_c(1) = a_mean
//    mean_c(2) = theta_mean
//    mean_c(3) = i_mean
//    mean_c(4) = q1_mean
//    mean_c(5) = q2_mean
//    mean_c(6) = Omega_mean
// output :
//    osc Osculating Elements from Mean elements
Eigen::VectorXd OscMeanToOsculatingElements(double J2, Eigen::VectorXd meanElems, double Re, double mu) {
    double gamma = -J2 * Re * Re;
    double a = meanElems(0);
    double argLat = meanElems(1);
    double inc = meanElems(2);
    double q1 = meanElems(3);
    double q2 = meanElems(4);
    double RAAN = meanElems(5);

    double si = std::sin(inc);
    double ci = std::cos(inc);
    double s2i = std::sin(2 * inc);
    double c2i = std::cos(2 * inc);
    double sth = std::sin(argLat);
    double cth = std::cos(argLat);
    double s2th = std::sin(2 * argLat);
    double c2th = std::cos(2 * argLat);
    double s3th = std::sin(3 * argLat);
    double c3th = std::cos(3 * argLat);
    double s4th = std::sin(4 * argLat);
    double c4th = std::cos(4 * argLat);
    double s5th = std::sin(5 * argLat);
    double c5th = std::cos(5 * argLat);
double eta = std::sqrt(1 - (q1 * q1 + q2 * q2));

    double ci2th = ci*ci;
    double ci4th = ci2th*ci2th;

    double p = a * (1 - (q1 * q1 + q2 * q2));
    double R = p / (1 + q1 * cth + q2 * sth);
    double Vr = std::sqrt(mu / p) * (q1 * sth - q2 * cth);
    double Vt = std::sqrt(mu / p) * (1 + q1 * cth + q2 * sth);

    double Ttheta = 1 / (1 - 5 * ci2th);
    double eps1 = std::sqrt(q1 * q1 + q2 * q2);
    double eps2 = q1 * cth + q2 * sth;
    double eps3 = q1 * sth - q2 * cth;
    double eta2th = eta*eta;
    double eta3th = eta2th*eta;
    double eta4th = eta3th*eta;

    // Placeholder function for theta2lam. Need to implement this function.
    double lambda = theta2lam(a, argLat, q1, q2);  
    double argLatLam = argLat - lambda;
    double lam_q1 = (q1 * Vr) / (eta * Vt) + q2 / (eta * (1 + eta)) - eta * R * (a + R) * (q2 + std::sin(argLat)) / (p * p);
    double lam_q2 = (q2 * Vr) / (eta * Vt) - q1 / (eta * (1 + eta)) + eta * R * (a + R) * (q1 + std::cos(argLat)) / (p * p);

    // Placeholder matrices DLP, D_sp1, and D_sp2. Need to fill in these matrices.

    // % Long period part,  D_lp
// lamLp, aLp, argLatLp, incLp, q1Lp, q2Lp, RAANLp 计算
    double lamLp = (si * si / (8 * a * a * eta2th * (1 + eta))) * (1 - 10 * Ttheta * ci2th) * q1 * q2 +
                   (q1 * q2 / (16 * a * a * eta4th)) * (3 - 55 * ci2th- 280 * Ttheta * ci4th - 400 * Ttheta * Ttheta * ci4th * ci2th);

    double aLp = 0;
    double argLatLp = lamLp - (si * si / (16 * a * a * eta4th)) * (1 - 10 * Ttheta * ci2th) * ((3 + 2 * eta2th / (1 + eta)) * q1 * q2 + 2 * q1 * sth + 2 * q2 * cth + (1 / 2) * (q1 * q1 + q2 * q2) * s2th);
    double incLp = (s2i / (32 * a * a * eta4th)) * (1 - 10 * Ttheta * ci2th) * (q1 * q1 - q2 * q2);
    double q1Lp = -(q1 * si * si / (16 * a * a * eta2th)) * (1 - 10 * Ttheta * ci2th) - (q1 * q2 * q2 / (16 * a * a * eta4th)) * (3 - 55 * ci2th- 280 * Ttheta * ci4th - 400 * Ttheta * Ttheta * ci4th * ci2th);
    double q2Lp = (q2 * si * si / (16 * a * a * eta2th)) * (1 - 10 * Ttheta * ci2th) + (q1 * q1 * q2 / (16 * a * a * eta4th)) * (3 - 55 * ci2th- 280 * Ttheta * ci4th - 400 * Ttheta * Ttheta * ci4th * ci2th);
    double RAANLp = (q1 * q2 * ci / (8 * a * a * eta4th)) * (11 + 80 * Ttheta * ci2th+ 200 * Ttheta * Ttheta * ci4th);

    // 填充 DLP 矩阵


// First short period part, D_sp1
    double lamSp1 = (eps3 * (1 - 3 * ci2th) / (4 * a * a * eta4th * (1 + eta))) * ((1 + eps2) * (1 + eps2) + (1 + eps2) + eta2th) 
                    + (3 * (1 - 5 * ci2th) / (4 * a * a * eta4th)) * (argLatLam + eps3);
    double aSp1 = ((1 - 3 * ci2th) / (2 * a * eta4th * eta2th)) * ((1 + eps2) * (1 + eps2) * (1 + eps2) - eta2th * eta);
    double argLatSp1 = lamSp1 - (eps3 * (1 - 3 * ci2th) / (4 * a * a * eta4th * (1 + eta))) * ((1 + eps2) * (1 + eps2) + eta * (1 + eta));
    double IncSp1 = 0;
    double q1Sp1 = -(3 * q2 * (1 - 5 * ci2th) / (4 * a * a * eta4th)) * (argLatLam + eps3) 
                   + ((1 - 3 * ci2th) / (4 * a * a * eta4th * (1 + eta))) * (((1 + eps2) * (1 + eps2) + eta2th) * (q1 + (1 + eta) * cth) + (1 + eps2) * ((1 + eta) * cth + q1 * (eta - eps2)));
    double q2Sp1 = (3 * q1 * (1 - 5 * ci2th) / (4 * a * a * eta4th)) * (argLatLam + eps3) 
                   + ((1 - 3 * ci2th) / (4 * a * a * eta4th * (1 + eta))) * (((1 + eps2) * (1 + eps2) + eta2th) * (q2 + (1 + eta) * sth) + (1 + eps2) * ((1 + eta) * sth + q2 * (eta - eps2)));
    double RAANSp1 = (3 * ci / (2 * a * a * eta4th)) * (argLatLam + eps3);
    
    // Populate DSP1 matrix


// Compute each element of D_sp2
    double lamSp2 = -(3 * eps3 * si * si * c2th / (4 * a * a * eta4th * (1 + eta))) * (1 + eps2) * (2 + eps2)
        -(si * si / (8 * a * a * eta2th * (1 + eta))) * (3 * (q1 * si + q2 * cth) + (q1 * s3th - q2 * c3th))
        -((3 - 5 * ci2th) / (8 * a * a * eta4th)) * (3 * (q1 * si + q2 * cth) + 3 * s2th + (q1 * s3th - q2 * c3th));

    double aSp2 = -(3 * si * si / (2 * a * eta4th * eta)) * std::pow(1 + eps2, 3) * c2th;
    
    double argLatSp2 = lamSp2 - (si * si / (32 * a * a * eta2th * (1 + eta))) * (36 * q1 * q2 - 4 * (3 * eta2th + 5 * eta - 1) * (q1 * sth + q2 * cth)
        + 12 * eps2 * q1 * q2 - 32 * (1 + eta) * s2th - (eta2th + 12 * eta + 39) * (q1 * s3th - q2 * c3th)
        + 36 * q1 * q2 * c4th - 18 * (q1 * q1 - q2 * q2) * s4th + 3 * q2 * (3 * q1 * q1 - q2 * q2) * c5th - 3 * q1 * (q1 * q1 - 3 * q2 * q2) * s5th);

    double incSp2 = -(s2i / (8 * a * a * eta4th)) * (3 * (q1 * cth - q2 * sth) + 3 * c2th + (q1 * c3th + q2 * s3th));

    double q1Sp2 = (q2 * (3 - 5 * ci2th) / (8 * a * a * eta4th)) * (3 * (q1 * sth + q2 * cth) + 3 * s2th + (q1 * s3th - q2 * c3th))
        + (si * si / (8 * a * a * eta4th)) * (3 * (eta2th - q1 * q1) * cth + 3 * q1 * q2 * sth - (eta2th + 3 * q1 * q1) * c3th - 3 * q1 * q2 * s3th)
        - (3 * si * si * c2th / (16 * a * a * eta4th)) * (10 * q1 + (8 + 3 * q1 * q1 + q2 * q2) * cth + 2 * q1 * q2 * sth
            + 6 * (q1 * c2th + q2 * s2th) + (q1 * q1 - q2 * q2) * c3th + 2 * q1 * q2 * s3th);

    double q2Sp2 = -(q1 * (3 - 5 * ci2th) / (8 * a * a * eta4th)) * (3 * (q1 * sth + q2 * cth) + 3 * s2th + (q1 * s3th - q2 * c3th))
        - (si * si / (8 * a * a * eta4th)) * (3 * (eta2th - q2 * q2) * sth + 3 * q1 * q2 * cth + (eta2th + 3 * q2 * q2) * s3th + 3 * q1 * q2 * c3th)
        - (3 * si * si * c2th / (16 * a * a * eta4th)) * (10 * q2 + (8 + q1 * q1 + 3 * q2 * q2) * sth + 2 * q1 * q2 * cth
            + 6 * (q1 * sth - q2 * cth) + (q1 * q1 - q2 * q2) * s3th - 2 * q1 * q2 * c3th);

    double RAANSp2 = -(ci / (4 * a * a * eta4th)) * (3 * (q1 * sth + q2 * cth) + 3 * s2th + (q1 * s3th - q2 * c3th));

    // Fill D_sp2 matrix elements

    // Osculating Elements calculations
    double aOsc = a + gamma * (aLp + aSp1 + aSp2);  // aLp, aSp1, aSp2 need to be defined.
    double argLatOsc = argLat + gamma * (argLatLp + argLatSp1 + argLatSp2);  // argLatLp, argLatSp1, argLatSp2 need to be defined.
    double iOsc = inc + gamma * (incLp + IncSp1 + incSp2);  // incLp, IncSp1, incSp2 need to be defined.
    double q1Osc = q1 + gamma * (q1Lp + q1Sp1 + q1Sp2);  // q1Lp, q1Sp1, q1Sp2 need to be defined.
    double q2Osc = q2 + gamma * (q2Lp + q2Sp1 + q2Sp2);  // q2Lp, q2Sp1, q2Sp2 need to be defined.
    double OmegaOsc = RAAN + gamma * (RAANLp + RAANSp1 + RAANSp2);  // RAANLp, RAANSp1, RAANSp2 need to be defined.

    // Osculating Elements from Mean elements
    Eigen::VectorXd osc_c(6);
    osc_c << aOsc, argLatOsc, iOsc, q1Osc, q2Osc, OmegaOsc;

    return osc_c;
}

// Calculation of the state transition matrix
// for the mean non-singular variables with perturbation by J2
// input :
//    t_in(1) = t0
//    t_in(2) = t
//    ICs_c(1)= a0,    ICs_c(2)= theta0,    ICs_c(3)= i0
//    ICs_c(4)= q10,    ICs_c(5)= q20,    ICs_c(6)= Omega0
// output :
//    6x6 state transition matrix, phi_J2
Eigen::MatrixXd OscMeanElemsSTM(double J2, double tf, const Eigen::VectorXd ICSc, double Re, double mu, double tol=1e-9) {
    // System_matrix, phi_J2 in mean non-singular variables with perturbation by J2

    // Constants
    double gamma = 3 * J2 * Re * Re;
    double t0 = 0;
    // double tf = t(1);
    double a0 = ICSc(0), argLat0 = ICSc(1), i0 = ICSc(2);
    double q10 = ICSc(3), q20 = ICSc(4), RAAN0 = ICSc(5);

    // Initial values
    double n0 = sqrt(mu / (a0 * a0 * a0));
    double p0 = a0 * (1 - q10 * q10 - q20 * q20);
    double R0 = p0 / (1 + q10 * cos(argLat0) + q20 * sin(argLat0));
    double Vr0 = sqrt(mu / p0) * (q10 * sin(argLat0) - q20 * cos(argLat0));
    double Vt0 = sqrt(mu / p0) * (1 + q10 * cos(argLat0) + q20 * sin(argLat0));
    double eta0 = sqrt(1 - q10 * q10 - q20 * q20);

    double lambda0 =   theta2lam(a0, argLat0, q10, q20);

    // Secular variations by J2
    double aDot = 0.0;
    double incDot = 0.0;
    double argPerDot = gamma * (1.0 / 4.0) * (n0 / (p0 * p0)) * (5 * cos(i0) * cos(i0) - 1);
    double sDot = sin(argPerDot * (tf - t0));
    double cDot = cos(argPerDot * (tf - t0));
    double lamDot = n0 + gamma * (1.0 / 4.0) * (n0 / (p0 * p0)) * ((5 + 3 * eta0) * cos(i0) * cos(i0) - (1 + eta0));
    double RAANDot = -gamma * (1.0 / 2.0) * (n0 / (p0 * p0)) * (cos(i0));

    // Perturbed orbital elements
    double a = a0 + aDot * (tf - t0);
    double i = i0 + incDot * (tf - t0);
    double Omega = RAAN0 + RAANDot * (tf - t0);
    double q1 = q10 * cos(argPerDot * (tf - t0)) - q20 * sin(argPerDot * (tf - t0));
    double q2 = q10 * sin(argPerDot * (tf - t0)) + q20 * cos(argPerDot * (tf - t0));

    double temp;
    double lambda =  lam2theta(lambda0 + lamDot * (tf - t0), q1, q2,tol, temp);

    // Mean orbital elements
    double n = sqrt(mu / (a * a * a));
    double p = a * (1 - q1 * q1 - q2 * q2);
    double R = p / (1 + q1 * cos(lambda) + q2 * sin(lambda));
    double Vr = sqrt(mu / p) * (q1 * sin(lambda) - q2 * cos(lambda));
    double Vt = sqrt(mu / p) * (1 + q1 * cos(lambda) + q2 * sin(lambda));
    double eta = sqrt(1 - q1 * q1 - q2 * q2);

    // Partial derivatives
    double G_theta = n * R / Vt;
    double G_theta0 = -n0 * R0 / Vt0;
    double G_q1 = (q1 * Vr) / (eta * Vt) + q2 / (eta * (1 + eta)) - eta * R * (a + R) * (q2 + sin(lambda)) / (p * p);
    double G_q10 = -(q10 * Vr0) / (eta0 * Vt0) - q20 / (eta0 * (1 + eta0)) + eta0 * R0 * (a0 + R0) * (q20 + sin(argLat0)) / (p0 * p0);
    double G_q2 = (q2 * Vr) / (eta * Vt) - q1 / (eta * (1 + eta)) + eta * R * (a + R) * (q1 + cos(lambda)) / (p * p);
    double G_q20 = -(q20 * Vr0) / (eta0 * Vt0) + q10 / (eta0 * (1 + eta0)) - eta0 * R0 * (a0 + R0) * (q10 + cos(argLat0)) / (p0 * p0);
    double K = 1 + G_q1 * (q10 * sDot + q20 * cDot) - G_q2 * (q10 * cDot - q20 * sDot);

    // Transformation Matrix phi_J2
    Eigen::MatrixXd phi_J2(6, 6);
    phi_J2 << 1, 0, 0, 0, 0, 0,
            //
             -((tf - t0) / G_theta) * ((3 * n0 / (2 * a0)) +
             (7 * gamma / 8) * (n0 / (a0 * p0 * p0)) * (eta0 * (3 * cos(i0) * cos(i0) - 1) + K * (5 * cos(i0) * cos(i0) - 1))), 
             -(G_theta0 / G_theta), -((tf - t0) / G_theta) * (gamma / 2) * (n0 * sin(i0) * cos(i0) / (p0 * p0)) * (3 * eta0 + 5 * K), 
             -((G_q10 + cDot * G_q1 + sDot * G_q2) / G_theta) + ((tf - t0) / G_theta) * (gamma / 4) * (n0 * a0 * q10 / (p0 * p0 * p0)) * (3 * eta0 * (3 * cos(i0) * cos(i0) - 1) + 4 * K * (5 * cos(i0) * cos(i0) - 1)), 
             -((G_q20 - sDot * G_q1 + cDot * G_q2) / G_theta) + ((tf - t0) / G_theta) * (gamma / 4) * (n0 * a0 * q20 / (p0 * p0 * p0)) * (3 * eta0 * (3 * cos(i0) * cos(i0) - 1) + 4 * K * (5 * cos(i0) * cos(i0) - 1)), 
             0,
             //
             0, 0, 1, 0, 0, 0,
             //
             (7 * gamma / 8) * (n0 * (q10 * sDot + q20 * cDot) * (5 * cos(i0) * cos(i0) - 1) / (a0 * p0 * p0)) * (tf - t0), 0, 
             (5 * gamma / 2) * (n0 * (q10 * sDot + q20 * cDot) * (sin(i0) * cos(i0)) / (p0 * p0)) * (tf - t0), 
             cDot - gamma * (n0 * a0 * q10 * (q10 * sDot + q20 * cDot) * (5 * cos(i0) * cos(i0) - 1) / (p0 * p0 * p0)) * (tf - t0), 
             -sDot - gamma * (n0 * a0 * q20 * (q10 * sDot + q20 * cDot) * (5 * cos(i0) * cos(i0) - 1) / (p0 * p0 * p0)) * (tf - t0), 
             0,
             //
             -(7 * gamma / 8) * (n0 * (q10 * cDot - q20 * sDot) * (5 * cos(i0) * cos(i0) - 1) / (a0 * p0 * p0)) * (tf - t0), 0, 
             -(5 * gamma / 2) * (n0 * (q10 * cDot - q20 * sDot) * (sin(i0) * cos(i0)) / (p0 * p0)) * (tf - t0), 
             sDot + gamma * (n0 * a0 * q10 * (q10 * cDot - q20 * sDot) * (5 * cos(i0) * cos(i0) - 1) / (p0 * p0 * p0)) * (tf - t0), 
             cDot + gamma * (n0 * a0 * q20 * (q10 * cDot - q20 * sDot) * (5 * cos(i0) * cos(i0) - 1) / (p0 * p0 * p0)) * (tf - t0), 
             0,
             //
             (7 * gamma / 4) * (n0 * cos(i0) / (a0 * p0 * p0)) * (tf - t0), 0, 
             (gamma / 2) * (n0 * sin(i0) / (p0 * p0)) * (tf - t0), 
             -(2 * gamma) * (n0 * a0 * q10 * cos(i0) / (p0 * p0 * p0)) * (tf - t0), 
             -(2 * gamma) * (n0 * a0 * q20 * cos(i0) / (p0 * p0 * p0)) * (tf - t0), 
             1;

    return phi_J2;
}

// OE: a, theta (true anomaly + arg perigee), ex, ey, i, longitude of asceding node
//    cond_c(1)= a,    cond_c(2)= theta,    cond_c(3)= i
//    cond_c(4)= q1,    cond_c(5)= q2,    cond_c(6)= Omega
Eigen::VectorXd OscMeanElemspropagate(double J2, double tf, const Eigen::VectorXd ICSc, double Re, double mu, double tol=1e-9){
    Eigen::VectorXd OE_osc(6);

    // Constants
    double gamma = 3 * J2 * Re * Re;
    double t0 = 0;
    // double tf = t(1);
    double a0 = ICSc(0), argLat0 = ICSc(1), i0 = ICSc(2);
    double q10 = ICSc(3), q20 = ICSc(4), RAAN0 = ICSc(5);

    // Initial values
    double n0 = sqrt(mu / (a0 * a0 * a0));
    double p0 = a0 * (1 - q10 * q10 - q20 * q20);
    double R0 = p0 / (1 + q10 * cos(argLat0) + q20 * sin(argLat0));
    double Vr0 = sqrt(mu / p0) * (q10 * sin(argLat0) - q20 * cos(argLat0));
    double Vt0 = sqrt(mu / p0) * (1 + q10 * cos(argLat0) + q20 * sin(argLat0));
    double eta0 = sqrt(1 - q10 * q10 - q20 * q20);

    double lambda0 =   theta2lam(a0, argLat0, q10, q20);

    // Secular variations by J2
    double aDot = 0.0;
    double incDot = 0.0;
    double argPerDot = gamma * (1.0 / 4.0) * (n0 / (p0 * p0)) * (5 * cos(i0) * cos(i0) - 1);
    double sDot = sin(argPerDot * (tf - t0));
    double cDot = cos(argPerDot * (tf - t0));
    double lamDot = n0 + gamma * (1.0 / 4.0) * (n0 / (p0 * p0)) * ((5 + 3 * eta0) * cos(i0) * cos(i0) - (1 + eta0));
    double RAANDot = -gamma * (1.0 / 2.0) * (n0 / (p0 * p0)) * (cos(i0));

    // Perturbed orbital elements
    double a = a0 + aDot * (tf - t0);
    double i = i0 + incDot * (tf - t0);
    double Omega = RAAN0 + RAANDot * (tf - t0);
    double q1 = q10 * cos(argPerDot * (tf - t0)) - q20 * sin(argPerDot * (tf - t0));
    double q2 = q10 * sin(argPerDot * (tf - t0)) + q20 * cos(argPerDot * (tf - t0));

    double temp;
    double lambda =  lambda0 + lamDot * (tf - t0);
    double theta= lam2theta(lambda, q1, q2, tol,temp);

    OE_osc<<  a, theta, i, q1, q2, Omega;
return OE_osc;
}

// formation matrix D_J2 in closed form
// from mean to osculating new set of elements with the perturbation by only J2
// input :
//    mean_c(1) = a_mean
//    mean_c(2) = theta_mean
//    mean_c(3) = i_mean
//    mean_c(4) = q1_mean
//    mean_c(5) = q2_mean
//    mean_c(6) = Omega_mean
// output :
//    osc Osculating Elements from Mean elements
//    D_J2 = I + (-J2*Re^2)*(D_lp+D_sp1+D_sp2) = 6x6 transformation matrix D_J2
Eigen::VectorXd DMeanToOsculatingElements(double J2, Eigen::VectorXd meanElems, double Re, double mu,Eigen::Ref<Eigen::Matrix<double, 6, 6>> DJ2) {
    double gamma = -J2 * Re * Re;
    double a = meanElems(0);
    double argLat = meanElems(1);
    double inc = meanElems(2);
    double q1 = meanElems(3);
    double q2 = meanElems(4);
    double RAAN = meanElems(5);

    double si = std::sin(inc);
    double ci = std::cos(inc);
    double s2i = std::sin(2 * inc);
    double c2i = std::cos(2 * inc);
    double sth = std::sin(argLat);
    double cth = std::cos(argLat);
    double s2th = std::sin(2 * argLat);
    double c2th = std::cos(2 * argLat);
    double s3th = std::sin(3 * argLat);
    double c3th = std::cos(3 * argLat);
    double s4th = std::sin(4 * argLat);
    double c4th = std::cos(4 * argLat);
    double s5th = std::sin(5 * argLat);
    double c5th = std::cos(5 * argLat);

    double eta = std::sqrt(1 - (q1 * q1 + q2 * q2));
    double eta2th = eta*eta;
    double eta3th = eta2th*eta;
    double eta4th = eta3th*eta;
    double ci2th = ci*ci;
    double ci4th = ci2th*ci2th;

    double p = a * (1 - (q1 * q1 + q2 * q2));
    double R = p / (1 + q1 * cth + q2 * sth);
    double Vr = std::sqrt(mu / p) * (q1 * sth - q2 * cth);
    double Vt = std::sqrt(mu / p) * (1 + q1 * cth + q2 * sth);

    double Ttheta = 1 / (1 - 5 * ci2th);

    double eps1 = std::sqrt(q1 * q1 + q2 * q2);
    double eps2 = q1 * cth + q2 * sth;
    double eps3 = q1 * sth - q2 * cth;
    


    // Placeholder function for theta2lam. Need to implement this function.
    double lambda = theta2lam(a, argLat, q1, q2);  
    double argLatLam = argLat - lambda;
    double lam_q1 = (q1 * Vr) / (eta * Vt) + q2 / (eta * (1 + eta)) - eta * R * (a + R) * (q2 + std::sin(argLat)) / (p * p);
    double lam_q2 = (q2 * Vr) / (eta * Vt) - q1 / (eta * (1 + eta)) + eta * R * (a + R) * (q1 + std::cos(argLat)) / (p * p);

    // Placeholder matrices DLP, D_sp1, and D_sp2. Need to fill in these matrices.
    Eigen::MatrixXd DLP(6, 6);DLP.setZero();
    Eigen::MatrixXd DSP1(6, 6);
    Eigen::MatrixXd D_sp2(6, 6);

    // % Long period part,  D_lp
// lamLp, aLp, argLatLp, incLp, q1Lp, q2Lp, RAANLp 计算
    double lamLp = (si * si / (8 * a * a * eta2th * (1 + eta))) * (1 - 10 * Ttheta * ci2th) * q1 * q2 +
                   (q1 * q2 / (16 * a * a * eta4th)) * (3 - 55 * ci2th- 280 * Ttheta * ci4th - 400 * Ttheta * Ttheta * ci4th * ci2th);

    double aLp = 0;
    double argLatLp = lamLp - (si * si / (16 * a * a * eta4th)) * (1 - 10 * Ttheta * ci2th) * ((3 + 2 * eta2th / (1 + eta)) * q1 * q2 + 2 * q1 * sth + 2 * q2 * cth + (1 / 2) * (q1 * q1 + q2 * q2) * s2th);
    double incLp = (s2i / (32 * a * a * eta4th)) * (1 - 10 * Ttheta * ci2th) * (q1 * q1 - q2 * q2);
    double q1Lp = -(q1 * si * si / (16 * a * a * eta2th)) * (1 - 10 * Ttheta * ci2th) - (q1 * q2 * q2 / (16 * a * a * eta4th)) * (3 - 55 * ci2th- 280 * Ttheta * ci4th - 400 * Ttheta * Ttheta * ci4th * ci2th);
    double q2Lp = (q2 * si * si / (16 * a * a * eta2th)) * (1 - 10 * Ttheta * ci2th) + (q1 * q1 * q2 / (16 * a * a * eta4th)) * (3 - 55 * ci2th- 280 * Ttheta * ci4th - 400 * Ttheta * Ttheta * ci4th * ci2th);
    double RAANLp = (q1 * q2 * ci / (8 * a * a * eta4th)) * (11 + 80 * Ttheta * ci2th+ 200 * Ttheta * Ttheta * ci4th);

    // 填充 DLP 矩阵
    DLP(0, 0) = -(1 / a) * aLp;
    DLP(1, 0) = -(2 / a) * argLatLp;
    DLP(1, 1) = -(si * si / (16 * a * a * eta4th)) * (1 - 10 * Ttheta * ci2th) * (2 * (q1 * cth - q2 * sth) + eta * c2th);
    DLP(1, 2) = (s2i / (16 * a * a * eta4th)) * (5 * q1 * q2 * (11 + 112 * Ttheta * ci2th+ 520 * Ttheta * Ttheta * ci4th + 800 * Ttheta * Ttheta * Ttheta * ci4th * ci2th) - (2 * q1 * q2 + (2 + (q1 * cth + q2 * sth)) * (q1 * sth + q2 * cth)) * ((1 - 10 * Ttheta * ci2th) + 10 * Ttheta * si * si * (1 + 5 * Ttheta * ci2th)));
    DLP(1, 3) = (1 / (16 * a * a * eta4th * eta2th)) * ((eta2th + 4 * q1 * q1) * (q2 * (3 - 55 * ci2th- 280 * Ttheta * ci4th - 400 * Ttheta * Ttheta * ci4th * ci2th) - si * si * (1 - 10 * Ttheta * ci2th) * (3 * q2 + 2 * sth)) - 2 * si * si * (1 - 10 * Ttheta * ci2th) * (4 * q2 + sth * (1 + eta)) * q1 * cth);
    DLP(1, 4) = (1 / (16 * a * a * eta4th * eta2th)) * ((eta2th + 4 * q2 * q2) * (q1 * (3 - 55 * ci2th- 280 * Ttheta * ci4th - 400 * Ttheta * Ttheta * ci4th * ci2th) - si * si * (1 - 10 * Ttheta * ci2th) * (3 * q1 + 2 * cth)) - 2 * si * si * (1 - 10 * Ttheta * ci2th) * (4 * q1 + cth * (1 + eta)) * q2 * sth);
    DLP(2, 0) = -(2 / a) * incLp;
    DLP(2, 2) = ((q1 * q1 - q2 * q2) / (16 * a * a * eta4th)) * (c2i * (1 - 10 * Ttheta * ci2th) + 5 * Ttheta * s2i * s2i * (1 + 5 * Ttheta * ci2th));
    DLP(2, 3) = (q1 * s2i / (16 * a * a * eta4th * eta2th)) * (1 - 10 * Ttheta * ci2th) * (eta2th + 2 * (q1 * q1 - q2 * q2));
    DLP(2, 4) = -(q2 * s2i / (16 * a * a * eta4th * eta2th)) * (1 - 10 * Ttheta * ci2th) * (eta2th - 2 * (q1 * q1 - q2 * q2));
    DLP(3, 0) = -(2 / a) * q1Lp;
    DLP(3, 2) = -(q1 * s2i / (16 * a * a * eta4th)) * (eta2th * ((1 - 10 * Ttheta * ci2th) + 10 * Ttheta * si * si * (1 + 5 * Ttheta * ci2th)) + 5 * q2 * q2 * (11 + 112 * Ttheta * ci2th+ 520 * Ttheta * Ttheta * ci4th + 800 * Ttheta * Ttheta * Ttheta * ci4th * ci2th));
    DLP(3, 3) = -(1 / (16 * a * a * eta4th * eta2th)) * (eta2th * si * si * (1 - 10 * Ttheta * ci2th) * (eta2th + 2 * q1 * q1) + q2 * q2 * (eta2th + 4 * q1 * q1) * (3 - 55 * ci2th- 280 * Ttheta * ci4th - 400 * Ttheta * Ttheta * ci4th * ci2th));
    DLP(3, 4) = -(q1 * q2 / (8 * a * a * eta4th * eta2th)) * (eta2th * si * si * (1 - 10 * Ttheta * ci2th) + (eta2th + 2 * q2 * q2) * (3 - 55 * ci2th- 280 * Ttheta * ci4th - 400 * Ttheta * Ttheta * ci4th * ci2th));
    DLP(4, 0) = -(2 / a) * q2Lp;
    DLP(4, 2) = (q2 * s2i / (16 * a * a * eta4th)) * (eta2th * (1 - 10 * Ttheta * ci2th) + 10 * Ttheta *eta2th  * si * si * (1 + 5 * Ttheta * ci2th) + 5 * q1 * q1 * (11 + 112 * Ttheta * ci2th+ 520 * Ttheta * Ttheta * ci4th + 800 * Ttheta * Ttheta * Ttheta * ci4th * ci2th));
    DLP(4, 3) = (q1 * q2 / (8 * a * a * eta4th * eta2th)) * (eta2th * si * si * (1 - 10 * Ttheta * ci2th) + (3 - 55 * ci2th- 280 * Ttheta * ci4th - 400 * Ttheta * Ttheta * ci4th * ci2th) * (eta2th + 2 * q1 * q1));
    DLP(4, 4) = (1 / (16 * a * a * eta4th * eta2th)) * (eta2th * si * si * (1 - 10 * Ttheta * ci2th) * (eta2th + 2 * q2 * q2) + q1 * q1 * (3 - 55 * ci2th- 280 * Ttheta * ci4th - 400 * Ttheta * Ttheta * ci4th * ci2th) * (eta2th + 4 * q2 * q2));
    DLP(5, 0) = -(2 / a) * RAANLp;
    DLP(5, 2) = -(q1 * q2 * si / (8 * a * a * eta4th)) * ((11 + 80 * Ttheta * ci2th+ 200 * Ttheta * Ttheta * ci4th) + 160 * Ttheta * ci2th* (1 + 5 * Ttheta * ci2th) * (1 + 5 * Ttheta * ci2th));
    DLP(5, 3) = (q2 * ci / (8 * a * a * eta4th * eta2th)) * (eta2th + 4 * q1 * q1) * (11 + 80 * Ttheta * ci2th+ 200 * Ttheta * Ttheta * ci4th);
    DLP(5, 4) = (q1 * ci / (8 * a * a * eta4th * eta2th)) * (eta2th + 4 * q2 * q2) * (11 + 80 * Ttheta * ci2th+ 200 * Ttheta * Ttheta * ci4th);

    if(debuggingOrn){printf("\n");printmat66(DLP);}
// First short period part, D_sp1
    double lamSp1 = (eps3 * (1 - 3 * ci2th) / (4 * a * a * eta4th * (1 + eta))) * ((1 + eps2) * (1 + eps2) + (1 + eps2) + eta2th) 
                    + (3 * (1 - 5 * ci2th) / (4 * a * a * eta4th)) * (argLatLam + eps3);
    double aSp1 = ((1 - 3 * ci2th) / (2 * a * eta4th * eta2th)) * ((1 + eps2) * (1 + eps2) * (1 + eps2) - eta2th * eta);
    double argLatSp1 = lamSp1 - (eps3 * (1 - 3 * ci2th) / (4 * a * a * eta4th * (1 + eta))) * ((1 + eps2) * (1 + eps2) + eta * (1 + eta));
    double IncSp1 = 0;
    double q1Sp1 = -(3 * q2 * (1 - 5 * ci2th) / (4 * a * a * eta4th)) * (argLatLam + eps3) 
                   + ((1 - 3 * ci2th) / (4 * a * a * eta4th * (1 + eta))) * (((1 + eps2) * (1 + eps2) + eta2th) * (q1 + (1 + eta) * cth) + (1 + eps2) * ((1 + eta) * cth + q1 * (eta - eps2)));
    double q2Sp1 = (3 * q1 * (1 - 5 * ci2th) / (4 * a * a * eta4th)) * (argLatLam + eps3) 
                   + ((1 - 3 * ci2th) / (4 * a * a * eta4th * (1 + eta))) * (((1 + eps2) * (1 + eps2) + eta2th) * (q2 + (1 + eta) * sth) + (1 + eps2) * ((1 + eta) * sth + q2 * (eta - eps2)));
    double RAANSp1 = (3 * ci / (2 * a * a * eta4th)) * (argLatLam + eps3);
    
    // Populate DSP1 matrix
    DSP1(0, 0) = -(1 / a) * aSp1;
    DSP1(0, 1) = -(3 * eps3 / (2 * a * eta4th * eta2th)) * (1 - 3 * ci2th) * (1 + eps2) * (1 + eps2);
    DSP1(0, 2) = (3 * s2i / (2 * a * eta4th * eta2th)) * ((1 + eps2) * (1 + eps2) * (1 + eps2) - eta2th * eta);
    DSP1(0, 3) = (3 * (1 - 3 * ci2th) / (2 * a * eta4th * eta4th)) * (2 * q1 * (1 + eps2) * (1 + eps2) * (1 + eps2) + eta2th * (1 + eps2) * (1 + eps2) * cth - eta2th * eta * q1);
    DSP1(0, 4) = (3 * (1 - 3 * ci2th) / (2 * a * eta4th * eta4th)) * (2 * q2 * (1 + eps2) * (1 + eps2) * (1 + eps2) + eta2th * (1 + eps2) * (1 + eps2) * sth - eta2th * eta * q2);
    DSP1(0, 5) = 0;
    
    DSP1(1, 0) = -(2 / a) * argLatSp1;
    DSP1(1, 1) = ((1 - 3 * ci2th) / (4 * a * a * eta4th * (1 + eta))) * (eps2 * (1 + eps2 - eta) - eps3 * eps3) 
                 + (3 * (1 - 5 * ci2th) / (4 * a * a * eta4th * (1 + eps2) * (1 + eps2))) * ((1 + eps2) * (1 + eps2) * (1 + eps2) - eta2th * eta);
    DSP1(1, 2) = (3 * eps3 * s2i / (4 * a * a * eta4th * (1 + eta))) * ((1 + eps2) + (5 + 4 * eta)) 
                 + (15 * s2i / (4 * a * a * eta4th)) * (argLatLam);
    DSP1(1, 3) = ((1 - 3 * ci2th) / (4 * a * a * eta4th * (1 + eta) * (1 + eta))) * (eta2th * (eps1 * sth + (1 + eta) * (eps2 * sth + eps3 * cth)) 
                 + q1 * eps3 * (4 * (eps1 + eps2) + eta * (2 + 5 * eps2))) + (3 * (1 - 5 * ci2th) / (4 * a * a * eta4th * eta2th)) * (4 * q1 * (argLatLam + eps3) + eta2th * sth) 
                 - (3 * (1 - 5 * ci2th) / (4 * a * a * eta4th)) * (lam_q1);
    DSP1(1, 4) = -((1 - 3 * ci2th) / (4 * a * a * eta4th * (1 + eta) * (1 + eta))) * (eta2th * (eps1 * cth + (1 + eta) * (eps2 * cth - eps3 * sth)) 
                 - q2 * eps3 * (4 * (eps1 + eps2) + eta * (2 + 5 * eps2))) + (3 * (1 - 5 * ci2th) / (4 * a * a * eta4th * eta2th)) * (4 * q2 * (argLatLam + eps3) - eta2th * cth) 
                 - (3 * (1 - 5 * ci2th) / (4 * a * a * eta4th)) * (lam_q2);
    DSP1(1, 5) = 0;
    
    DSP1(2, 0) = -(2 / a) * IncSp1;
    DSP1(2, 1) = 0;
    DSP1(2, 2) = 0;
    DSP1(2, 3) = 0;
    DSP1(2, 4) = 0;
    DSP1(2, 5) = 0;
    
    DSP1(3, 0) = -(2 / a) * q1Sp1;
    DSP1(3, 1) = -((1 - 3 * ci2th) / (4 * a * a * eta4th)) * ((1 + eps2) * (2 * sth + eps2 * sth + 2 * eps3 * cth) + eps3 * (q1 + cth) + eta2th * sth)
                - (3 * q2 * (1 - 5 * ci2th) / (4 * a * a * eta4th * (1 + eps2) * (1 + eps2))) * ((1 + eps2) * (1 + eps2) * (1 + eps2) - eta2th * eta);
    DSP1(3, 2) = (3 * q1 * s2i / (4 * a * a * eta2th * (1 + eta))) 
                + (3 * s2i / (4 * a * a * eta4th)) * ((1 + eps2) * (q1 + (2 + eps2) * cth) - 5 * q2 * eps3 + eta2th * cth) 
                - (15 * q2 * s2i / (4 * a * a * eta4th)) * (argLatLam);
    DSP1(3, 3) = ((1 - 3 * ci2th) / (4 * a * a * eta2th * (1 + eta))) 
                + ((1 - 3 * ci2th) * q1 * q1 * (4 + 5 * eta) / (4 * a * a * eta4th * eta2th * (1 + eta) * (1 + eta))) 
                + ((1 - 3 * ci2th) / (8 * a * a * eta4th * eta2th)) * (eta2th * (5 + 2 * (5 * q1 * cth + 2 * q2 * sth) + (3 + 2 * eps2) * c2th) 
                + 2 * q1 * (4 * (1 + eps2) * (2 + eps2) * cth + (3 * eta + 4 * eps2) * q1)) 
                - (3 * q2 * (1 - 5 * ci2th) / (4 * a * a * eta4th * eta2th)) * (4 * q1 * eps3 + eta2th * sth) 
                - (3 * q1 * q2 * (1 - 5 * ci2th) / (a * a * eta4th * eta2th)) * (argLatLam) 
                + (3 * q2 * (1 - 5 * ci2th) / (4 * a * a * eta4th)) * (lam_q1);
    DSP1(3, 4) = ((1 - 3 * ci2th) / (8 * a * a * eta4th * eta2th)) * (eta2th * (2 * (q1 * sth + 2 * q2 * cth) + (3 + 2 * eps2) * s2th) 
                + 2 * q2 * (4 * (1 + eps2) * (2 + eps2) * cth + (3 * eta + 4 * eps2) * q1)) 
                + ((1 - 3 * ci2th) * q1 * q2 * (4 + 5 * eta) / (4 * a * a * eta4th * eta2th * (1 + eta) * (1 + eta))) 
                - (3 * (1 - 5 * ci2th) / (4 * a * a * eta4th)) * (eps3 * (eta2th + 4 * q2 * q2) - eta2th * q2 * cth) 
                - (3 * (1 - 5 * ci2th) / (4 * a * a * eta4th * eta2th)) * (argLatLam * (eta2th + 4 * q2 * q2)) 
                + (3 * q2 * (1 - 5 * ci2th) / (4 * a * a * eta4th)) * (lam_q2);
    DSP1(3, 5) = 0;

    
    DSP1(4, 0) = -(2 / a) * q2Sp1;
    DSP1(4, 1) = ((1 - 3 * ci2th) / (4 * a * a * eta4th * (1 + eta))) * ((1 + eps2) * (2 * cth + eps2 * cth - 2 * eps3 * sth) - eps3 * (q2 + sth) + eta2th * cth)
                + (3 * q1 * (1 - 5 * ci2th) / (4 * a * a * eta4th * (1 + eps2) * (1 + eps2))) * ((1 + eps2) * (1 + eps2) * (1 + eps2) - eta2th * eta);
    DSP1(4, 2) = (3 * q2 * s2i / (4 * a * a * eta2th * (1 + eta))) + (3 * s2i / (4 * a * a * eta4th)) * ((1 + eps2) * (q2 + (2 + eps2) * sth) + 5 * q1 * eps3 + eta2th * sth) 
                - (15 * q1 * s2i / (4 * a * a * eta4th)) * (argLatLam);
    DSP1(4, 3) = ((1 - 3 * ci2th) / (8 * a * a * eta4th * (1 + eta) * (1 + eta))) * (eta2th * (2 * (2 * q1 * sth + q2 * cth) + (3 + 2 * eps2) * s2th)
                + 2 * q1 * (4 * (1 + eps2) * (2 + eps2) * sth + (3 * eta + 4 * eps2) * q2)) + ((1 - 3 * ci2th) * q1 * q2 * (4 + 5 * eta) / (4 * a * a * eta4th * (1 + eta) * (1 + eta)))
                + (3 * (1 - 5 * ci2th) / (4 * a * a * eta4th)) * (eps3 * (eta2th + 4 * q1 * q1) + eta2th * q1 * sth)
                + (3 * (1 - 5 * ci2th) / (4 * a * a * eta4th)) * (argLatLam * (eta2th + 4 * q1 * q1)) - (3 * q1 * (1 - 5 * ci2th) / (4 * a * a * eta4th)) * (lam_q1);
    DSP1(4, 4) = ((1 - 3 * ci2th) / (4 * a * a * eta2th * (1 + eta))) + ((1 - 3 * ci2th) * q2 * q2 * (4 + 5 * eta) / (4 * a * a * eta4th * (1 + eta) * (1 + eta)))
                + ((1 - 3 * ci2th) / (8 * a * a * eta4th)) * (eta2th * (5 + 2 * (2 * q1 * cth + 5 * q2 * sth) - (3 + 2 * eps2) * c2th)
                + 2 * q2 * (4 * (1 + eps2) * (2 + eps2) * sth + (3 * eta + 4 * eps2) * q2)) + (3 * q1 * (1 - 5 * ci2th) / (4 * a * a * eta4th))
                * (4 * q2 * eps3 - eta2th * cth) + (3 * q1 * q2 * (1 - 5 * ci2th) / (a * a * eta4th)) * (argLatLam)
                - (3 * q1 * (1 - 5 * ci2th) / (4 * a * a * eta4th)) * (lam_q2);
    DSP1(4, 5) = 0;

    
    DSP1(5, 0) = -(2 / a) * RAANSp1;
    DSP1(5, 1) = (3 * ci / (2 * a * a * eta4th * (1 + eps2) * (1 + eps2))) * ((1 + eps2) * (1 + eps2) * (1 + eps2) - eta2th * eta);
    DSP1(5, 2) = -(3 * eps3 * si / (2 * a * a * eta4th)) - (3 * si / (2 * a * a * eta4th)) * (argLatLam);
    DSP1(5, 3) = (3 * ci / (2 * a * a * eta4th * (1 + eta))) * (4 * q1 * eps3 + eta2th * sth) + (6 * q1 * ci / (a * a * eta4th)) * (argLatLam)
                - (3 * ci / (2 * a * a * eta4th)) * (lam_q1);
    DSP1(5, 4) = (3 * ci / (2 * a * a * eta4th * (1 + eta))) * (4 * q2 * eps3 - eta2th * cth) + (6 * q2 * ci / (a * a * eta4th)) * (argLatLam)
                - (3 * ci / (2 * a * a * eta4th)) * (lam_q2);
    DSP1(5, 5) = 0;
    if(debuggingOrn){printf("\n");printmat66(DSP1);}



// Compute each element of D_sp2
    double lamSp2 = -(3 * eps3 * si * si * c2th / (4 * a * a * eta4th * (1 + eta))) * (1 + eps2) * (2 + eps2)
        - (si * si / (8 * a * a * eta2th * (1 + eta))) * (3 * (q1 * sth + q2 * cth) + (q1 * s3th - q2 * c3th))
        - ((3 - 5 * ci2th) / (8 * a * a * eta4th)) * (3 * (q1 * sth + q2 * cth) + 3 * s2th + (q1 * s3th - q2 * c3th));

    double aSp2 = -(3 * si * si / (2 * a * eta4th * eta)) * std::pow(1 + eps2, 3) * c2th;

    double argLatSp2 = lamSp2 - (si * si / (32 * a * a * eta2th * (1 + eta))) * (36 * q1 * q2 - 4 * (3 * eta2th + 5 * eta - 1) * (q1 * sth + q2 * cth)
        + 12 * eps2 * q1 * q2 - 32 * (1 + eta) * s2th - (eta2th + 12 * eta + 39) * (q1 * s3th - q2 * c3th)
        + 36 * q1 * q2 * c4th - 18 * (q1 * q1 - q2 * q2) * s4th + 3 * q2 * (3 * q1 * q1 - q2 * q2) * c5th - 3 * q1 * (q1 * q1 - 3 * q2 * q2) * s5th);

    double incSp2 = -(s2i / (8 * a * a * eta4th)) * (3 * (q1 * cth - q2 * sth) + 3 * c2th + (q1 * c3th + q2 * s3th));

    double q1Sp2 = (q2 * (3 - 5 * ci2th) / (8 * a * a * eta4th)) * (3 * (q1 * sth + q2 * cth) + 3 * s2th + (q1 * s3th - q2 * c3th))
        + (si * si / (8 * a * a * eta4th)) * (3 * (eta2th - q1 * q1) * cth + 3 * q1 * q2 * sth - (eta2th + 3 * q1 * q1) * c3th - 3 * q1 * q2 * s3th)
        - (3 * si * si * c2th / (16 * a * a * eta4th)) * (10 * q1 + (8 + 3 * q1 * q1 + q2 * q2) * cth + 2 * q1 * q2 * sth
            + 6 * (q1 * c2th + q2 * s2th) + (q1 * q1 - q2 * q2) * c3th + 2 * q1 * q2 * s3th);

    double q2Sp2 = -(q1 * (3 - 5 * ci2th) / (8 * a * a * eta4th)) * (3 * (q1 * sth + q2 * cth) + 3 * s2th + (q1 * s3th - q2 * c3th))
        - (si * si / (8 * a * a * eta4th)) * (3 * (eta2th - q2 * q2) * sth + 3 * q1 * q2 * cth + (eta2th + 3 * q2 * q2) * s3th + 3 * q1 * q2 * c3th)
        - (3 * si * si * c2th / (16 * a * a * eta4th)) * (10 * q2 + (8 + q1 * q1 + 3 * q2 * q2) * sth + 2 * q1 * q2 * cth
            + 6 * (q1 * s2th - q2 * c2th) + (q1 * q1 - q2 * q2) * s3th - 2 * q1 * q2 * c3th);

    double RAANSp2 = -(ci / (4 * a * a * eta4th)) * (3 * (q1 * sth + q2 * cth) + 3 * s2th + (q1 * s3th - q2 * c3th));


    // Fill D_sp2 matrix elements
    D_sp2(0, 0) = -(1 / a) * aSp2;
    D_sp2(0, 1) = (3 * si * si / (2 * a * eta4th * eta2th)) * std::pow(1 + eps2, 2) * (3 * eps3 * c2th + 2 * (1 + eps2) * s2th);
    D_sp2(0, 2) = -(3 * s2i * c2th / (2 * a * eta4th * eta)) * std::pow(1 + eps2, 3);
    D_sp2(0, 3) = -(9 * si * si * c2th / (2 * a * eta4th * eta4th)) * std::pow(1 + eps2, 2) * (2 * q1 * (1 + eps2) + eta2th * cth);
    D_sp2(0, 4) = -(9 * si * si * c2th / (2 * a * eta4th * eta4th)) * std::pow(1 + eps2, 2) * (2 * q2 * (1 + eps2) + eta2th * sth);
    D_sp2(0, 5) = 0;


    D_sp2(1, 0) = -(2 / a) * argLatSp2;
    D_sp2(1, 1) = -(1 / (8 * a * a * eta4th)) * (3 * (3 - 5 * ci2th) * ((q1 * cth - q2 * sth) + 2 * c2th + (q1 * c3th + q2 * s3th))
                - si * si * (5 * (q1 * cth - q2 * sth) + 16 * c2th + 9 * (q1 * c3th + q2 * s3th)));
    D_sp2(1, 2) = -(s2i / (8 * a * a * eta4th)) * (10 * (q1 * sth + q2 * cth) + 7 * s2th + 2 * (q1 * s3th - q2 * c3th));
    D_sp2(1, 3) = -((3 - 5 * ci2th) / (8 * a * a * eta4th * eta)) * (4 * q1 * (3 * s2th + q2 * (3 * cth - c3th)) + (eta2th + 4 * q1 * q1) * (3 * sth + s3th))
                - (si * si * (3 * sth + s3th) / (8 * a * a * eta2th * (1 + eta)))
                - (si * si / (32 * a * a * eta4th * (1 + eta))) * (36 * q2 - 4 * (2 + 3 * eta) * sth - (eta * (12 + eta) + 39) * s3th + 9 * eps1 * s5th
                + 12 * q2 * (2 * q1 * cth + q2 * sth) + 9 * q1 * (q1 * s3th - q2 * c3th) + 18 * (3 * q1 * s4th + 2 * q2 * c4th)
                - 3 * q1 * (q1 * s5th - 11 * q2 * c5th) + 24 * ((1 + eps2) * (2 + eps2) * sth + eps3 * (3 + 2 * eps2) * cth) * c2th)
                - (3 * si * si / (32 * a * a * eta4th * (1 + eta) * (1 + eta))) * (4 * sth - 6 * q1 * s4th - q1 * (q1 * s5th + q2 * c5th))
                + (q1 * si * si / (8 * a * a * eta4th * (1 + eta))) * (20 * (1 + eta) * (q1 * sth + q2 * cth) + 32 * (1 + eta) * s2th
                + 3 * (4 + 3 * eta) * (q1 * s3th - q2 * c3th)) - (q1 * si * si * (4 + 5 * eta) / (32 * a * a * eta4th * (1 + eta) * (1 + eta))) * (24 * (q1 * sth + q2 * cth)
                + 24 * eps3 * (1 + eps2) * (2 + eps2) * c2th - (27 + 3 * eta) * (q1 * s3th - q2 * c3th) - 18 * s4th - 3 * (q1 * s5th + q2 * c5th)
                + 12 * q2 * ((3 + eps2) * q1 + 3 * (q1 * c4th + q2 * s4th) + q1 * (q1 * c5th + q2 * s5th)));
    D_sp2(1, 4) = -((3 - 5 * ci2th) / (8 * a * a * eta4th * eta)) * (4 * q2 * (3 * s2th + q1 * (3 * sth + s3th)) + (eta2th + 4 * q2 * q2) * (3 * cth - c3th))
                - (si * si * (3 * cth - c3th) / (8 * a * a * eta2th * (1 + eta)))
                - (si * si / (32 * a * a * eta4th * (1 + eta))) * (36 * q1 - 4 * (2 + 3 * eta) * cth + (eta * (12 + eta) + 39) * c3th + 9 * eps1 * c5th
                + 12 * q1 * (q1 * cth + 2 * q2 * sth) + 9 * q2 * (q1 * s3th - q2 * c3th) + 18 * (2 * q1 * c4th + 7 * q2 * s4th)
                + 3 * q2 * (11 * q1 * s5th - q2 * c5th) + 24 * (eps3 * (3 + 2 * eps2) * sth - (1 + eps2) * (2 + eps2) * cth) * c2th)
                - (3 * si * si / (32 * a * a * eta4th * (1 + eta) * (1 + eta))) * (4 * cth - 6 * q2 * s4th - q2 * (q1 * s5th + q2 * c5th))
                + (q2 * si * si / (8 * a * a * eta4th * (1 + eta))) * (20 * (1 + eta) * (q1 * sth + q2 * cth) + 32 * (1 + eta) * s2th
                + 3 * (4 + 3 * eta) * (q1 * s3th - q2 * c3th)) - (q2 * si * si * (4 + 5 * eta) / (32 * a * a * eta4th * (1 + eta) * (1 + eta))) * (24 * (q1 * sth + q2 * cth)
                + 24 * eps3 * (1 + eps2) * (2 + eps2) * c2th - (27 + 3 * eta) * (q1 * s3th - q2 * c3th) - 18 * s4th - 3 * (q1 * s5th + q2 * c5th)
                + 12 * q2 * ((3 + eps2) * q1 + 3 * (q1 * c4th + q2 * s4th) + q1 * (q1 * c5th + q2 * s5th)));
    D_sp2(1, 5) = 0;


   D_sp2(2, 0) = -(2 / a) * incSp2;
    D_sp2(2, 1) = (3 * s2i / (8 * a * a * eta4th)) * ((q1 * sth + q2 * cth) + 2 * s2th + (q1 * s3th - q2 * c3th));
    D_sp2(2, 2) = -(c2i / (4 * a * a * eta4th)) * (3 * (q1 * cth - q2 * sth) + 3 * c2th + (q1 * c3th + q2 * s3th));
    D_sp2(2, 3) = -(s2i / (8 * a * a * eta4th * eta)) * (4 * q1 * (3 * c2th - q2 * (3 * sth - s3th)) + (eta2th + 4 * q1 * q1) * (3 * cth + c3th));
    D_sp2(2, 4) = -(s2i / (8 * a * a * eta4th * eta)) * (4 * q2 * (3 * c2th + q1 * (3 * cth + c3th)) - (eta2th + 4 * q2 * q2) * (3 * sth - s3th));
    D_sp2(2, 5) = 0;

    D_sp2(3, 0) = -(2 / a) * q1Sp2;
    D_sp2(3, 1) = (3 * q2 * (3 - 5 * ci2th) / (8 * a * a * eta4th)) * ((q1 * cth - q2 * sth) + 2 * c2th + (q1 * c3th + q2 * s3th))
        + (3 * si * si / (16 * a * a * eta4th)) * ((2 * eps2 * q2 - 9 * q2 * (q1 * c3th + q2 * s3th) + 12 * (q1 * s4th - q2 * c4th) - 5 * q2 * (q1 * c5th + q2 * s5th))
            + (1 / 2) * (4 * (1 + 3 * q1 * q1) * sth + 40 * q1 * s2th + (28 + 17 * eps1) * s3th + 5 * eps1 * s5th));
    D_sp2(3, 2) = -(s2i / (16 * a * a * eta4th)) * ((36 * q1 * (q1 * cth - q2 * sth) + 30 * (q1 * c2th - q2 * s2th) - q2 * (q1 * s3th - q2 * c3th)
            + 9 * (q1 * c4th + q2 * s4th) + 3 * q2 * (q1 * s5th - q2 * c5th))
            + (1 / 2) * (6 * q1 * (3 + 2 * q1 * cth) + 12 * (1 - 4 * eps1) * cth + (28 + 17 * eps1) * c3th + 3 * eps1 * c5th));
    D_sp2(3, 3) = (q2 * (3 - 5 * ci2th) / (8 * a * a * eta4th)) * (4 * q1 * (3 * s2th + q2 * (3 * cth - c3th)) + (eta2th + 4 * q1 * q1) * (3 * sth + s3th))
        - (si * si / (8 * a * a * eta4th)) * ((8 * q1 * c3th - 3 * q2 * (sth - s3th)) + 3 * (5 + eps2 + 3 * c2th + 3 * (q1 * c3th + q2 * s3th)) * c2th)
        - (3 * q1 * si * si / (4 * a * a * eta4th)) * (2 * q1 * ((q1 * cth - q2 * sth) + (q1 * c3th + q2 * s3th))
            + (9 * cth - c3th + 2 * q1 * (5 + eps2) + 6 * (q1 * c2th + q2 * s2th) + 2 * q1 * (q1 * c3th + q2 * s3th)) * c2th);
    D_sp2(3, 4) = ((3 - 5 * ci2th) / (8 * a * a * eta4th)) * ((eta2th + 4 * q2 * q2) * (3 * s2th + q1 * (3 * sth + s3th))
        + 2 * (eta2th + 2 * q2 * q2) * q2 * (3 * cth - c3th))
        + (si * si / (16 * a * a * eta4th)) * (6 * (q1 * sth + 2 * q2 * cth) - (9 * q1 * s3th + q2 * c3th) - 9 * s4th - 3 * (q1 * s5th + q2 * c5th))
        - (3 * q2 * si * si / (8 * a * a * eta4th)) * (2 * q1 * (3 + 2 * (2 * q1 * cth - q2 * sth) + 10 * c2th + 3 * (q1 * c3th + q2 * s3th) + (q1 * c5th + q2 * s5th)));
    D_sp2(3, 5) = 0;


   D_sp2(4, 0) = -(2 / a) * q2Sp2;
    D_sp2(4, 1) = -(3 * q1 * (3 - 5 * ci2th) / (8 * a * a * eta4th))
                * ((q1 * cth - q2 * sth) + 2 * c2th + (q1 * c3th + q2 * s3th))
                + (3 * si * si / (16 * a * a * eta4th)) * ((2 * eps2 * q1 + 9 * q1 * (q1 * c3th + q2 * s3th) - 12 * (q1 * c4th + q2 * s4th) - 5 * q1 * (q1 * c5th + q2 * s5th))
                    + (1 / 2) * (4 * (1 + 3 * q2 * q2) * cth + 40 * q2 * s2th - (28 + 17 * eps1) * c3th + 5 * eps1 * c5th));
    D_sp2(4, 2) = -(s2i / (16 * a * a * eta4th))
                * ((36 * q1 * (q1 * sth + q2 * cth) + 30 * (q1 * s2th + q2 * c2th) + q1 * (q1 * s3th - q2 * c3th)
                    + 9 * (q1 * s4th - q2 * c4th) + 3 * q1 * (q1 * s5th - q2 * c5th))
                    - (1 / 2) * (6 * q2 * (3 + 2 * q2 * sth) + 12 * (1 + 2 * eps1) * sth - (28 + 17 * eps1) * s3th + 3 * eps1 * s5th));
    D_sp2(4, 3) = -((3 - 5 * ci2th) / (8 * a * a * eta4th))
                * ((eta2th + 4 * q1 * q1) * (3 * s2th + q2 * (3 * cth - c3th))
                    + 2 * (eta2th + 2 * q1 * q1) * q1 * (3 * sth + s3th))
                - (si * si / (16 * a * a * eta4th))
                * (6 * (2 * q1 * sth + q2 * cth) + (q1 * s3th + 9 * q2 * c3th) + 9 * s4th - 3 * (q1 * s5th + q2 * c5th))
                + (3 * q1 * si * si / (8 * a * a * eta4th))
                * (2 * q2 * (3 - 2 * (q1 * cth - 2 * q2 * sth) - 10 * c2th - 3 * (q1 * c3th + q2 * s3th) + (q1 * c5th + q2 * s5th))
                    + (8 * sth - 9 * s3th - 6 * (q1 * s4th - q2 * c4th) - s5th));
    D_sp2(4, 4) = 0;
    D_sp2(4, 5) = 0;


    D_sp2(5, 0) = -(2 / a) * RAANSp2;
    D_sp2(5, 1) = -(3 * ci / (4 * a * a * eta4th)) * ((q1 * cth - q2 * sth) + 2 * c2th + (q1 * c3th + q2 * s3th));
    D_sp2(5, 2) = (si / (4 * a * a * eta4th)) * (3 * (q1 * sth + q2 * cth) + 3 * s2th + (q1 * s3th - q2 * c3th));
    D_sp2(5, 3) = -(ci / (4 * a * a * eta4th)) * (4 * q1 * (3 * s2th + q2 * (3 * cth - c3th)) + (eta2th + 4 * q1 * q1) * (3 * sth + s3th));
    D_sp2(5, 4) = -(ci / (4 * a * a * eta4th)) * (4 * q2 * (3 * s2th + q1 * (3 * sth + s3th)) + (eta2th + 4 * q2 * q2) * (3 * cth - c3th));
    D_sp2(5, 5) = 0;


    if(debuggingOrn){printf("\n");printmat66(D_sp2);}

    // Osculating Elements calculations
    double aOsc = a + gamma * (aLp + aSp1 + aSp2);  // aLp, aSp1, aSp2 need to be defined.
    double argLatOsc = argLat + gamma * (argLatLp + argLatSp1 + argLatSp2);  // argLatLp, argLatSp1, argLatSp2 need to be defined.
    double iOsc = inc + gamma * (incLp + IncSp1 + incSp2);  // incLp, IncSp1, incSp2 need to be defined.
    double q1Osc = q1 + gamma * (q1Lp + q1Sp1 + q1Sp2);  // q1Lp, q1Sp1, q1Sp2 need to be defined.
    double q2Osc = q2 + gamma * (q2Lp + q2Sp1 + q2Sp2);  // q2Lp, q2Sp1, q2Sp2 need to be defined.
    double OmegaOsc = RAAN + gamma * (RAANLp + RAANSp1 + RAANSp2);  // RAANLp, RAANSp1, RAANSp2 need to be defined.

    // Transformation Matrix D_J2
    // Eigen::MatrixXd DJ2(6,6);
    DJ2 = Eigen::MatrixXd::Identity(6, 6) + gamma * (DLP + DSP1 + D_sp2);

    printf("DJ2 = \n");
    if(debuggingOrn)printmat66(DJ2);
    printf("\n");
    // Osculating Elements from Mean elements
    Eigen::VectorXd osc_c(6);
    osc_c << aOsc, argLatOsc, iOsc, q1Osc, q2Osc, OmegaOsc;

    return osc_c;
}



// Calculation of the inverse of system_matrix at t0, inv_AA_BB_t0
// with perturbed osculating elements by J2
// input :
//    elems(1)= a0
//    elems(2)= theta0
//    elems(3)= i0
//    elems(4)= q10
//    elems(5)= q20
//    elems(6)= Omega0
// output :
//    6x6 inverse of system matrix
//    SigmaInverse = inv(T)*(T*inv(A+gamma*B))
Eigen::MatrixXd SigmaInverseMatrix(double J2, Eigen::VectorXd elems, double Re, double mu, double tol) {
    double gamma = 3 * J2 * Re * Re;
    double a = elems(0);
    double argLat = elems(1);
    double inc = elems(2);
    double q1 = elems(3);
    double q2 = elems(4);
    double RAAN = elems(5);
    
    double Hamiltonian = -mu / (2 * a);
    double p = a * (1 - q1 * q1 - q2 * q2);
    double R = p / (1 + q1 * cos(argLat) + q2 * sin(argLat));
    double Vr = sqrt(mu / p) * (q1 * sin(argLat) - q2 * cos(argLat));
    double Vt = sqrt(mu / p) * (1 + q1 * cos(argLat) + q2 * sin(argLat));
    
    // New non-singular elements
    double q1tilde = q1 * cos(RAAN) - q2 * sin(RAAN);
    double q2tilde = q1 * sin(RAAN) + q2 * cos(RAAN);
    double p1 = tan(inc / 2) * cos(RAAN);
    double p2 = tan(inc / 2) * sin(RAAN);
    
    double p1p2 = (p1 == 0 && p2 == 0) ? p1 * p1 + p2 * p2 + tol : p1 * p1 + p2 * p2;
    // Matrix T
    //   TT = [ 1  0  0  0  0  0;
    //          0  1  0  0  0  1;
    //          0  0  0  cos(Omega)  -sin(Omega)  -(q1*sin(Omega)+q2*cos(Omega));
    //          0  0  0  sin(Omega)   cos(Omega)   q1*cos(Omega)-q2*sin(Omega);
    //          0  0  cos(Omega)/(1+cos(i))  0  0  -sin(Omega)*sin(i)/(1+cos(i));
    //          0  0  sin(Omega)/(1+cos(i))  0  0   cos(Omega)*sin(i)/(1+cos(i)) ];
        // Inverse Matrix of T for non-singular elements
    Eigen::MatrixXd InvT(6, 6);
    InvT << 1, 0, 0, 0, 0, 0,
           0, 1, 0, 0, p2 / p1p2, -p1 / p1p2,
           0, 0, 0, 0, 2 * p1 / (sqrt(p1p2) * (1 + p1p2)), 2 * p2 / (sqrt(p1p2) * (1 + p1p2)),
           0, 0, p1 / sqrt(p1p2), p2 / sqrt(p1p2), -p2 * (p1 * q2tilde - p2 * q1tilde) / pow(p1p2, 1.5),  p1 * (p1 * q2tilde - p2 * q1tilde) / pow(p1p2, 1.5), 
           0,  0, -p2 / sqrt(p1p2), p1 / sqrt(p1p2), p2 * (p1 * q1tilde + p2 * q2tilde) / pow(p1p2, 1.5),-p1 * (p1 * q1tilde + p2 * q2tilde) / pow(p1p2, 1.5), 
           0,0,0,0,-p2/(p1p2), p1 / p1p2;
    
    // Calculate InvTA matrix
    Eigen::MatrixXd InvTA(6, 6);
    InvTA << (1 / (R * Hamiltonian)) * ((mu / R) * (3 * a - 2 * R) - a * (2 * Vr * Vr + 3 * Vt * Vt)),
             -a * Vr / Hamiltonian,
             -(Vr / Hamiltonian) * ((Vt / p) * (2 * a - R) - (a / (R * Vt)) * (Vr * Vr + 2 * Vt * Vt)),
             (R / Hamiltonian) * ((Vt / p) * (2 * a - R) - (a / (R * Vt)) * (Vr * Vr + 2 * Vt * Vt)),
             0, 0,
             //
             0, 0, 1 / R, 0, -((Vr * sin(argLat) + Vt * cos(argLat)) / (R * Vt)) * (sin(inc) / (1 + cos(inc))), (sin(argLat) / Vt) * (sin(inc) / (1 + cos(inc))),
            //
            p * (cos(RAAN) * (2 * Vr * sin(argLat) + 3 * Vt * cos(argLat)) + sin(RAAN) * (2 * Vr * cos(argLat) - 3 * Vt * sin(argLat))) / (R * R * Vt), 
             sqrt(p / mu) * (sin(RAAN) * cos(argLat) + cos(RAAN) * sin(argLat)),
             (sin(RAAN) * cos(argLat) + cos(RAAN) * sin(argLat)) * ((1 / R) - (Vr * Vr + Vt * Vt) / mu) - (cos(RAAN) * cos(argLat) - sin(RAAN) * sin(argLat)) * (Vr * Vt / mu), 
             2 * sqrt(p / mu) * (cos(RAAN) * cos(argLat) - sin(RAAN) * sin(argLat)) + (sin(RAAN) * cos(argLat) + cos(RAAN) * sin(argLat)) * (R * Vr / mu), 
             ((q1 * sin(RAAN) + q2 * cos(RAAN)) * (q1 + cos(argLat)) * sin(inc)) / (p * (1 + cos(inc))),
             -((q1 * sin(RAAN) + q2 * cos(RAAN)) * sin(argLat) * sin(inc)) / (Vt * (1 + cos(inc))),
            //
             p * (sin(RAAN) * (2 * Vr * sin(argLat) + 3 * Vt * cos(argLat)) - cos(RAAN) * (2 * Vr * cos(argLat) - 3 * Vt * sin(argLat))) / (R * R * Vt), 
             -sqrt(p / mu) * (cos(RAAN) * cos(argLat) - sin(RAAN) * sin(argLat)),
             -(cos(RAAN) * cos(argLat) - sin(RAAN) * sin(argLat)) * ((1 / R) - (Vr * Vr + Vt * Vt) / mu) - (sin(RAAN) * cos(argLat) + cos(RAAN) * sin(argLat)) * (Vr * Vt / mu), 
             2 * sqrt(p / mu) * (sin(RAAN) * cos(argLat) + cos(RAAN) * sin(argLat)) - (cos(RAAN) * cos(argLat) - sin(RAAN) * sin(argLat)) * (R * Vr / mu), 
             -((q1 * cos(RAAN) - q2 * sin(RAAN)) * (q1 + cos(argLat)) * sin(inc)) / (p * (1 + cos(inc))),
             ((q1 * cos(RAAN) - q2 * sin(RAAN)) * sin(argLat) * sin(inc)) / (Vt * (1 + cos(inc))),
             //
             0, 0, 0, 0, -(cos(RAAN) * (Vr * cos(argLat) - Vt * sin(argLat)) - sin(RAAN) * (Vr * sin(argLat) + Vt * cos(argLat))) / (R * Vt * (1 + cos(inc))),
             (cos(RAAN) * cos(argLat) - sin(RAAN) * sin(argLat)) / (Vt * (1 + cos(inc))),
             //
             0, 0, 0, 0, -(sin(RAAN) * (Vr * cos(argLat) - Vt * sin(argLat)) + cos(RAAN) * (Vr * sin(argLat) + Vt * cos(argLat))) / (R * Vt * (1 + cos(inc))),
             (sin(RAAN) * cos(argLat) + cos(RAAN) * sin(argLat)) / (Vt * (1 + cos(inc)));
             
    // The rest of inverse Matrix
    Eigen::MatrixXd InvTD(6, 6);
    InvTD << 0, 0, 0, 0, (sin(inc) * cos(inc) * sin(argLat) / (Hamiltonian * p * R * R)) * ((mu / R) * (2 * a - R) - a * (Vr * Vr + 2 * Vt * Vt)), 0,
             0, 0, -cos(inc) * (1 - cos(inc)) * sin(argLat) * sin(argLat) / (p * R * R), 0, 0, 0,
             0, 0, (cos(inc) * cos(inc) * sin(argLat) * sin(argLat) / (Vt * Vt * R * R * R)) * (Vr * Vt * (cos(RAAN) * cos(argLat) - sin(RAAN) * sin(argLat)) - Vt * Vt * (sin(RAAN) * cos(argLat) + cos(RAAN) * sin(argLat))) + (cos(inc) * sin(argLat) * sin(argLat) / (p * R * R)) * (cos(inc) * (sin(RAAN) * cos(argLat) + cos(RAAN) * sin(argLat)) + (q1 * sin(RAAN) + q2 * cos(RAAN))), 0, (sin(inc) * cos(inc) * sin(argLat) / (Vt * R * R * R)) * (Vr * (sin(RAAN) * cos(argLat) + cos(RAAN) * sin(argLat)) + 2 * Vt * (cos(RAAN) * cos(argLat) - sin(RAAN) * sin(argLat))), 0,
             0, 0, (cos(inc) * cos(inc) * sin(argLat) * sin(argLat) / (Vt * Vt * R * R * R)) * (Vr * Vt * (sin(RAAN) * cos(argLat) + cos(RAAN) * sin(argLat)) + Vt * Vt * (cos(RAAN) * cos(argLat) - sin(RAAN) * sin(argLat))) - (cos(inc) * sin(argLat) * sin(argLat) / (p * R * R)) * (cos(inc) * (cos(RAAN) * cos(argLat) - sin(RAAN) * sin(argLat)) + (q1 * cos(RAAN) - q2 * sin(RAAN))), 0, -(sin(inc) * cos(inc) * sin(argLat) / (Vt * R * R * R)) * (Vr * (cos(RAAN) * cos(argLat) - sin(RAAN) * sin(argLat)) - 2 * Vt * (sin(RAAN) * cos(argLat) + cos(RAAN) * sin(argLat))), 0,
             0, 0, -(sin(inc) * cos(inc) * sin(argLat) / ((1 + cos(inc)) * p * R * R)) * (cos(RAAN) * cos(argLat) - sin(RAAN) * sin(argLat)), 0, 0, 0,
             0, 0, -(sin(inc) * cos(inc) * sin(argLat) / ((1 + cos(inc)) * p * R * R)) * (sin(RAAN) * cos(argLat) + cos(RAAN) * sin(argLat)), 0, 0, 0;
             
    // Calculate SigmaInverse
    Eigen::MatrixXd SigmaInverse(6,6);
    SigmaInverse = InvT * (InvTA + gamma * InvTD);
    
    return SigmaInverse;
}
// System_matrix, Sigma in osculating element with perturbation by J2
// input :
//    elems(1)= a
//    elems(2)= theta
//    elems(3)= i
//    elems(4)= q1
//    elems(5)= q2
//    elems(6)= Omega
// output :
//    6x6 system matrix
//    Sigma = A + (3*J2*Re^2)*B
Eigen::MatrixXd SigmaMatrix(double J2, Eigen::VectorXd elems, double Re, double mu) {
    // System_matrix, Sigma in osculating element with perturbation by J2

    // Constants
    double gamma = 3 * J2 * Re * Re;
    double a = elems(0);
    double argLat = elems(1);
    double inc = elems(2);
    double q1 = elems(3);
    double q2 = elems(4);

    // Evaluations from the inputs
    double p = a * (1 - q1*q1 - q2*q2);
    double R = p / (1 + q1 * cos(argLat) + q2 * sin(argLat));
    double Vr = sqrt(mu / p) * (q1 * sin(argLat) - q2 * cos(argLat));
    double Vt = sqrt(mu / p) * (1 + q1 * cos(argLat) + q2 * sin(argLat));

    // Transformation Matrix A
    Eigen::MatrixXd A(6, 6);
    A << R/a, R*Vr/Vt, 0, -(2*a*R*q1/p) - (R*R/p)*cos(argLat), -(2*a*R*q2/p) - (R*R/p)*sin(argLat), 0,
         -(0.5*Vr/a), sqrt(mu/p)*((p/R) - 1), 0, (Vr*a*q1/p) + sqrt(mu/p)*sin(argLat), (Vr*a*q2/p) - sqrt(mu/p)*cos(argLat), 0,
         0, R, 0, 0, 0, R*cos(inc),
         -(1.5*Vt/a), -Vr, 0, (3*Vt*a*q1/p) + 2*sqrt(mu/p)*cos(argLat), (3*Vt*a*q2/p) + 2*sqrt(mu/p)*sin(argLat), Vr*cos(inc),
         0, 0, R*sin(argLat), 0, 0, -R*cos(argLat)*sin(inc),
         0, 0, Vt*cos(argLat) + Vr*sin(argLat), 0, 0, (Vt*sin(argLat) - Vr*cos(argLat))*sin(inc);

    // Transformation Matrix B
    Eigen::MatrixXd B(6, 6);
    B << 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0,
         0, 0, -Vt*sin(inc)*cos(inc)*sin(argLat)*sin(argLat)/(p*R), 0, 0, Vt*sin(inc)*sin(inc)*cos(inc)*sin(argLat)*cos(argLat)/(p*R),
         0, 0, 0, 0, 0, 0,
         0, Vt*sin(inc)*cos(inc)*sin(argLat)/(p*R), 0, 0, 0, Vt*sin(inc)*cos(inc)*cos(inc)*sin(argLat)/(p*R);

    // System Matrix Sigma = A + gamma * B
    Eigen::MatrixXd Sigma(6,6);
    Sigma = A + gamma * B;

    return Sigma;
}

void printmat66(Eigen::MatrixXd mat){
    for(int i=0;i<mat.rows();i++){
        for(int j=0;j<mat.cols();j++)printf("%1.3e\t",mat(i,j));
        printf("\n");
    }
}

// Example usage
int testlam2theta() {
    double a = 6930.0;
    double lambda = 1.0; // Example values
    double q1 = 0.1; // Example values
    double q2 = 0.1; // Example values
    double Tol = 1e-6; // Example tolerance
    double theta, F;

    theta = lam2theta(lambda, q1, q2, Tol,  F);

    std::cout << "Theta: " << theta << std::endl;
    std::cout << "F: " << F << std::endl;

    std::cout << "Lam ref: " << lambda << std::endl;
    lambda=theta2lam(a, theta, q1, q2);

    std::cout << "Lambda: " << lambda << std::endl;

    return 0;
}

int testSigmaMat(){
    // Example usage
    double J2 = 0.00108263;
    double Re = 6378.137;
    double mu = 398600.4418;
    Eigen::VectorXd elems(6);
    elems << 7000, 0.5, 1.3, 0.1, 0.1, 2.1; // Example elements

    Eigen::MatrixXd result = SigmaMatrix(J2, elems, Re, mu);

    std::cout << "Sigma Matrix:\n" << result << std::endl;

    return 0;
}


int testSigmaInverseMatrix(){
    Eigen::VectorXd x(6);
    x << 6925443.952, 190432.624, 230986.901, -303.938541, 2277.904454, 7229.098276;

Eigen::VectorXd OE = osculating::rv2OEOsc(x);
    Eigen::VectorXd ICSc_ref = OE2Osc(OE);
    double J2 = osculating::J2;
    double Re = osculating::RE;
    double mu = osculating::MU;

    double tol = 1e-10; // Tolerance
    
    Eigen::MatrixXd Sig(6,6);
    Sig = SigmaMatrix(J2, ICSc_ref, Re, mu);
    Eigen::MatrixXd SigInv(6,6);
    SigInv = SigmaInverseMatrix(J2, ICSc_ref, Re, mu, tol);
    std::cout << ">>Sigma Matrix:\n"
              << Sig << std::endl;
    std::cout << ">>Sigma Matrix Inv:\n"
              << SigInv << std::endl;
    Eigen::MatrixXd expectedI(6,6);
    expectedI = SigInv * Sig;
    std::cout << ">>SigmaInv*Sigma:\n"
              << expectedI << std::endl;
return 0;
/*
>>Sigma Matrix:
    0.999324     -170.488            0 -6.93234e+06      -243076            0
 1.34479e-08      5.12112            0      265.265     -7575.82            0
           0  6.93191e+06            0            0            0   2.0873e+06
 -0.00164034     0.186565  -0.00732892        15167      531.629     0.143418
           0            0       242571            0            0 -6.60614e+06
           0     0.209438      7580.93            0            0      253.366
>>Sigma Matrix Inv:
     4.00676   -0.0450417  4.92891e-05      1831.36   5.5332e-05            0
           0            0   1.4426e-07            0  4.55251e-08 -1.45669e-06
           0            0 -3.98058e-12            0  5.05169e-09  0.000131748
 4.32808e-07  4.61625e-06  1.37886e-13  0.000263674  5.77056e-12  7.02657e-11
 1.51618e-08 -0.000131837  9.75223e-11  9.23574e-06  3.09763e-11 -9.82235e-10
           0            0 -1.46163e-13            0 -1.51189e-07  4.83766e-06
>>SigmaInv*Sigma:
           1  2.27374e-13            0            0  2.32831e-10            0
           0            1 -3.46945e-18            0            0  1.83013e-16
           0 -6.77626e-21            1            0            0            0
 5.29396e-23  3.34091e-18  9.52912e-22            1 -2.77556e-17  1.68303e-18
-6.61744e-24 -9.50475e-17 -1.27055e-20  2.77556e-17            1 -4.83497e-17
           0 -2.11758e-22  6.93889e-18            0            0            1
*/
}

int testOscMeanToOsculatingElements() {
    // Example usage
    double J2 = osculating::J2;
    double Re = osculating::RE;
    double mu = osculating::MU;

    Eigen::VectorXd x(6);
    x << 6925443.952, 190432.624, 230986.901, -303.938541, 2277.904454, 7229.098276;

Eigen::VectorXd OE = osculating::rv2OEOsc(x);
Eigen::VectorXd OEMean = osculating::OEOsc2OEMeanEU(OE);
Eigen::VectorXd ICM_ref = OE2Osc(OEMean);
    Eigen::VectorXd ICSc_ref = OE2Osc(OE);
    std::cout << "Osculating Elements set: " << ICSc_ref.transpose() << std::endl;

    // Eigen::VectorXd ICM =  (OEMean);
    // std::cout << "Mean Elements converted: " << ICM.transpose() << std::endl;

    Eigen::MatrixXd DJ2(6, 6);
    Eigen::VectorXd ICSc = DMeanToOsculatingElements(J2, ICM_ref, Re, mu, DJ2);
    // DJ2 = OscMeanToOsculatingElements(J2, ICM_ref, Re, mu);
    std::cout << ">> Osculating Elements converted: " << ICSc.transpose() << std::endl;
    return 0;
/*
Osculating Elements set:            6.9366e+06   0.0350005     1.26494 0.000674293 4.82366e-05   0.0169624
>> Osculating Elements converted:   6.93661e+06   0.0349423     1.26494 0.000673839 4.81777e-05   0.0169624
*/
}
int testOscMeanSTM() {
    double J2 = osculating::J2;
    double Re = osculating::RE;
    double mu = osculating::MU;
    double tol = 1e-10; // Tolerance
    double t1 = 10,t2=20;
    
    Eigen::VectorXd x0_t0(6),x1_t0(6),x31_t0(6);
    Eigen::VectorXd x0_t1(6),x1_t1(6),x31_t1(6);
    Eigen::VectorXd x0_t2(6),x1_t2(6),x31_t2(6);
    x0_t0 << 6925443.952, 190432.624, 230986.901, -303.938541, 2277.904454, 7229.098276;
    x0_t1 << 6921989.6884, 213199.8038, 303262.5598,	-386.90722, 2275.48604, 7225.88873;
    x0_t2 << 6917705.9636, 235941.4359, 375501.7815,	-469.82912, 2272.79496, 7221.81096;

    x1_t0 << 6742506.7850, 582990.7040, 1496003.1690,	-1755.23833, 2203.24326, 7043.84991;
    x1_t1 << 6724550.7486, 604987.7644, 1566350.3739,	-1835.93278, 2196.12491, 7025.45024;
    x1_t2 << 6705788.8690, 626912.3255, 1636509.3611,	-1916.40535, 2188.74355, 7006.20678;

    x31_t0 << 6848665.9010, -220295.8230, -1058536.4800,	1176.99883, 2274.29060, 7138.17868;
    x31_t1 << 6860025.6559, -197540.1827, -987092.5773,	1094.92923, 2276.79205, 7150.45895;
    x31_t2 << 6870564.0437, -174760.8905, -915530.1620,	1012.72705, 2279.02094, 7161.88095;


    Eigen::VectorXd OEOsc0 = osculating::rv2OEOsc(x0_t0);
    Eigen::VectorXd OEOMean0 = osculating::OEOsc2OEMeanEU(OEOsc0);
    Eigen::VectorXd ICM_ref0 = OE2Osc(OEOMean0);
    Eigen::MatrixXd Doe2Drv = SigmaMatrix(J2, ICM_ref0, Re, mu);

    Eigen::VectorXd ICS_ref0 = OE2Osc(OEOsc0);
    Eigen::MatrixXd Does2Drv = SigmaMatrix(J2, ICS_ref0, Re, mu);
    std::cout << "Mean Elements ref: " << ICM_ref0.transpose() << std::endl;

    Eigen::VectorXd OEOsc01 = osculating::rv2OEOsc(x0_t1);
    Eigen::VectorXd ICS_ref1 = OE2Osc(OEOsc01);
    std::cout << "Mean Elements ref(next,by osc): " << ICS_ref1.transpose() << std::endl;

    Eigen::MatrixXd meanSTM = OscMeanElemsSTM(J2, t1, ICS_ref0, Re, mu, tol);
    std::cout << ">>State Transition Matrix phi_J2(osc0):\n"
              << meanSTM << std::endl;
    Eigen::MatrixXd meanSTM2 = OscMeanElemsSTM(J2, t1, ICM_ref0, Re, mu, tol);
    std::cout << ">>State Transition Matrix phi_J2(Mean):\n"
              << meanSTM2 << std::endl;
    
    std::cout << "下面不再使用Mean OE:\n";
    Eigen::VectorXd OEOsc10 = osculating::rv2OEOsc(x1_t0);
    Eigen::VectorXd ICS_10 = OE2Osc(OEOsc10);
    Eigen::VectorXd dICS_10 = ICS_10 - ICS_ref0;

    Eigen::VectorXd OEOsc11 = osculating::rv2OEOsc(x1_t1);
    Eigen::VectorXd ICS_11 = OE2Osc(OEOsc11);
    Eigen::VectorXd dICS_11 = ICS_11 - ICS_ref0;
    
    Eigen::VectorXd dICS_11_pred = meanSTM2*dICS_10;
    std::cout << "Predicted OE error(t1): " << dICS_11_pred.transpose() << std::endl;
    std::cout << "Real      OE error(t1): " << dICS_11.transpose() << std::endl;

    Eigen::VectorXd drv_11_pred = Does2Drv*dICS_11_pred;
    Eigen::VectorXd drv_11 = x1_t1 - x0_t1;
    std::cout << "Predicted XYZ error(t1)/m: " << drv_11_pred.transpose() << std::endl;
    std::cout << "Real      XYZ error(t1)/m: " << drv_11.transpose() << std::endl;
    return 0;
}


}//osc