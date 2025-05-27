#include <Eigen/Dense>
#include <string>
#include <fstream>
#include <vector>
#include <sstream>
#include <stdexcept>
#ifndef ICATT_ELEMENTS_H
#define ICATT_ELEMENTS_H
class DensityInterpolator {
public:
    // 构造函数，自动初始化并保存数据
    DensityInterpolator(const std::string& filename){
        data = read_csv(filename);
    }

    // 成员函数，根据目标高度进行线性插值计算
    double getRho(double target_height) const {
        return interpolate_density(target_height);
    }

private:
    struct DataPoint {
        double height;
        double density;
    };

    std::vector<DataPoint> data;

    // 读取CSV文件并存储数据点
    std::vector<DataPoint> read_csv(const std::string& filename) const;
    // 根据目标高度进行线性插值计算
    double interpolate_density(double target_height) const;
};
int testDensity();
namespace elements {
    using namespace Eigen;
    VectorXd elements(Vector3d r, Vector3d v, double mu);
    double  orbitalPeriod(Vector3d r, Vector3d v, double mu, double &n);
    void benchmark(int times);
}

#endif //ICATT_ELEMENTS_H
