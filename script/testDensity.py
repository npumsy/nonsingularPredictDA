import sys
sys.path.insert(0, '/mnt/d/Docker/shared/TZ/nonsingularPredict/build')
import qoe


import matplotlib.pyplot as plt
import math

# csv_file = "/mnt/d/Develop/OrbitUQ/SpaceDSL/nonsingularPredict/src/AtmosphericDensity1976.csv"
csv_file = "/mnt/d/Docker/shared/TZ/nonsingularPredict/src/AtmosphericDensity1976.csv"
# 创建 DensityInterpolator 实例
interpolator = qoe.DensityInterpolator(csv_file)
def test_density_interpolator():
    # 使用提供的CSV文件路径

    
    # 测试不同高度处的大气密度
    test_heights = [0.0, 200.0, 540.0, 1100.0]
    for height in test_heights:
        try:
            density = interpolator.getRho(height)
            print(f"Density at {height} km: {density:.6e} kg/km^3")
        except Exception as e:
            print(f"Error calculating density at {height} km: {str(e)}")
def testexpH():
    h0 = 530
    rho0 =  interpolator.getRho(h0)
    h1 = 560
    rho1 = interpolator.getRho(h1)
    H0 = (h1-h0)/(math.log(rho0)-math.log(rho1))
    print(H0)
    print('rho0',rho0)
    def expH(h): 
        return rho0*math.exp(-(h-h0)/H0)
    
    test_heights = list(range(h0-5,h1+5,1))
    den_search = []
    den_analytic = []
    for height in test_heights:
        den_search.append( interpolator.getRho(height))
        rho = expH(height)
        den_analytic.append(rho)

    plt.figure()
    plt.plot(test_heights,den_analytic)
    plt.plot(test_heights,den_search)
    plt.legend(['atmo1976','expon()'])
    plt.show()
   
if __name__ == "__main__":
    test_density_interpolator()
    # testexpH()