#include <iostream>
#include <chrono>
#include <Eigen/Dense>
#include "elements.h"


// 读取CSV文件并存储数据点
std::vector<DensityInterpolator::DataPoint> DensityInterpolator::read_csv(const std::string& filename) const {
        std::vector<DataPoint> data;
        std::ifstream file(filename);
        if (!file.is_open()) {
                throw std::runtime_error("Could not open file");
        }

        std::string line;
        std::getline(file, line); // 跳过标题行
        while (std::getline(file, line)) {
                std::stringstream ss(line);
                std::string height_str, density_str;
                std::getline(ss, height_str, ',');
                std::getline(ss, density_str, ',');

                DataPoint point;
                point.height = std::stod(height_str);
                point.density = std::stod(density_str);
                data.push_back(point);
        }

        file.close();
        return data;
}

// 根据目标高度进行线性插值计算
double DensityInterpolator::interpolate_density(double target_height) const {
        if (target_height < data.front().height || target_height > data.back().height) {
                throw std::out_of_range("Target height is out of the data range.");
        }

        double interval = 0.5;
        int index = static_cast<int>((target_height - data.front().height) / interval);

        if (index < 0 || index >= data.size() - 1) {
                throw std::out_of_range("Calculated index is out of range.");
        }

        double h1 = data[index].height;
        double h2 = data[index + 1].height;
        double rho1 = data[index].density;
        double rho2 = data[index + 1].density;

        double density = rho1 + (rho2 - rho1) * (target_height - h1) / (h2 - h1);
        return density;
}
int testDensity() {
    try {
        DensityInterpolator interpolator("density_data.csv");
        double target_height = 520.0;
        double density_at_target = interpolator.getRho(target_height);
        std::cout << "Density at " << target_height << " km: " << density_at_target << " kg/m^3" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
using Eigen::Vector3d;

namespace elements
{
    VectorXd elements(Vector3d r, Vector3d v, double mu)
    {
        auto r_mag = r.norm();
        auto v_mag = v.norm();
        auto v_mag2 = v_mag * v_mag;
        auto h = r.cross(v);
        auto h_mag = h.norm();
        Vector3d k(0, 0, 1);
        auto n = k.cross(h);
        auto n_mag = n.norm();
        auto xi = v_mag2 / 2 - mu / r_mag;
        auto e = ((v_mag2 - mu / r_mag) * r - v * r.dot(v)) / mu;
        auto ecc = e.norm();
        double sma;
        if (ecc != 1) {
            sma = -mu / (2 * xi);
        } else {
            sma = pow(h_mag, 2) / mu;
        }
        auto inc = acos(h.z() / h_mag);

        double node;
        if (n_mag > 1e-16) { // 如果n_mag接近于零，说明倾角接近于零
            node = acos(n.x() / n_mag);
            if (n.y() < 0) {
                node = M_PI * 2 - node;
            }
        } else {
            node = 0; // 倾角为零时，升交点经度定义为零
        }

        double peri;
        if (ecc > 1e-10) { // 如果ecc接近于零，说明轨道接近于圆形
            if (n_mag > 1e-10) {
                peri = acos(n.dot(e) / (ecc * n_mag));
                if (e.z() < 0) {
                    peri = M_PI * 2 - peri;
                }
            } else {
                peri = acos(e.x() / ecc);
                if (e.y() < 0) {
                    peri = M_PI * 2 - peri;
                }
            }
        } else {
            peri = 0; // 轨道接近于圆形时，近地点幅角定义为零
        }

        auto ano = acos(e.dot(r) / (ecc * r_mag));
        if (r.dot(v) < 0) {
            ano = M_PI * 2 - ano;
        }

        VectorXd out(6);
        out << sma, ecc, inc, node, peri, ano;
        return out;
    }
    double  orbitalPeriod(Vector3d r, Vector3d v, double mu, double &n){
        auto r_mag = r.norm();
        auto v_mag = v.norm();
        auto v_mag2 = v_mag * v_mag;
        // auto h = r.cross(v);
        // auto h_mag = h.norm();
        auto xi = v_mag2 / 2 - mu / r_mag;
        // auto e = ((v_mag2 - mu / r_mag) * r - v * r.dot(v)) / mu;
        // auto ecc = e.norm();
        double sma;
        // if (ecc != 1) {
            sma = -mu / (2 * xi);
        // }  else {
        //     sma = pow(h_mag, 2) / mu;
        // }
        n = std::sqrt(mu / std::pow(sma, 3));
        return  2 * M_PI / n;
    }

    void benchmark(int times) {
        auto mu = 3.986004418e5;
        Vector3d r(8.59072560e+02, -4.13720368e+03, 5.29556871e+03);
        Vector3d v(7.37289205e+00, 2.08223573e+00, 4.39999794e-01);

        auto best = std::numeric_limits<double>::infinity();
        auto worst = -std::numeric_limits<double>::infinity();
        double all = 0;
        for (auto i=0; i < times; i++) {
            auto begin = std::chrono::high_resolution_clock::now();

            elements(r, v, mu);

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
}
