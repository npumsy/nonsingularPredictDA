import sys
sys.path.insert(0, '/mnt/d/Docker/shared/TZ/nonsingularPredict/build')

sys.path.append("/mnt/d/Docker/shared/TZ")
# from SatOrbitProp.orbit_propagation import satOrbit_PROP

import qoe 
import numpy as np
csv_file = "/mnt/d/Docker/shared/TZ/nonsingularPredict/src/AtmosphericDensity1976.csv"
# 创建 DensityInterpolator 实例
aeroDensity = qoe.DensityInterpolator(csv_file)
def test_propagate_lagrangian():
    # Define initial position and velocity vectors
    pos0 = np.array([6925443.952, 190432.624, 230986.901])
    vel0 = np.array([-303.938541, 2277.904454, 7229.098276])
    tof = 1.0  # Time of flight in seconds
    mu =qoe.GM_Earth()  # Gravitational parameter for Earth in m^3/s^2
    # mu = 398600
    stm = True
    
    # Create a 6x6 zero matrix for Phi0f with required flags
    Phi0f = np.zeros((6, 6), order='F')
    
    # Call the propagate_lagrangian function
    rvf_kep = qoe.propagate_lagrangian(pos0, vel0, tof, mu, stm, Phi0f)
    
    # Print the results
    print("Propagated state vector (6x1):")
    print(rvf_kep)
    print("State transition matrix (6x6):")
    print(Phi0f)

    rv_pred = Phi0f@(np.concatenate((pos0,vel0))[:,np.newaxis])[:,0]
    print("result=")
    print(rv_pred)

    err = np.array(rvf_kep)-rv_pred
    print("error=",err)

def printRV(t,rvf_kep,km=1e3):
    # Print the results
    pas =f"{t:04d}"+'\t'+", ".join(f"{x:4.4f}" for x in km*rvf_kep[:3]) + ',\t'+ ", ".join(f"{v:1.5f}" for v in km*rvf_kep[3:])
    print(pas)

def testRK4():
    # Define initial position and velocity vectors
    pos0 = np.array([6925443.952, 190432.624, 230986.901])
    vel0 = np.array([-303.938541, 2277.904454, 7229.098276])
    rv0 = np.concatenate((pos0,vel0))
    rv0 = np.array([6925443.9520, 190432.6240, 230986.9010,	-303.93854, 2277.90445, 7229.09828])
    rv1 = np.array([6921989.7470, 213199.4765, 303261.2250,	-386.89500, 2275.42060, 7225.62180])
    rv2 = np.array([6917706.2178, 235940.1275, 375496.4435,	-469.80169, 2272.66418, 7221.27727])
    tof = list(range(0,5400,10))  # Time of flight in seconds

    h = np.linalg.norm(pos0)-qoe.EarthRadius()
    density = aeroDensity.getRho(h/1e3)/1e9

    CdA_m = 0.0044

    # print("Propagated state vector (6x1):")
    t=5400
    # rvf_kep = qoe.intJ234DragRV_RK4Step(rv0, t, CdA_m*density,0.05)
    # printRV(t,rvf_kep)

    t0 = [2025,10,1,0,0,0]
    Sat_prop = satOrbit_PROP(pos0,vel0,t0, t,1.0)
    pos_array, vel_array, _, epoch_array = Sat_prop.orbit_prop(True,0.001)
    print("reference Pythoin state vector (6x1):")
    printRV(t,np.concatenate((pos_array[-1],vel_array[-1])) )

    print("Propagated state vector (6x1):")
    for t in tof:
        rvf_kep = qoe.intJ234DragRV_RK4Step(rv0/1e3, 10, CdA_m*density)
        rv0 = np.array(rvf_kep)
        if(not t%100):
            printRV(t,rvf_kep)

def testIntRK4():
    rv0 = np.array([6925443.9520, 190432.6240, 230986.9010,	-303.93854, 2277.90445, 7229.09828])
    t0=0
    step=10
    while t0<30:
        tf = step+t0
        # rvf_kep = qoe.intJ234DragRV_RK4Step(rv0/1e3, tf, 1.4281E-12,True, 0.1)
        rvf_kep = qoe.intJ234DragRV_RK4Step(rv0/1e3, tf, 0,False, 0.1)
        printRV(tf,rvf_kep)
        t0=tf

def testDARK4():
    # Define initial position and velocity vectors
    pos0 = np.array([6925443.952, 190432.624, 230986.901])
    vel0 = np.array([-303.938541, 2277.904454, 7229.098276])
    rv0 = np.concatenate((pos0,vel0))
    tof = list(range(0,5400,10))  # Time of flight in seconds

    # print("Propagated state vector (6x1):")
    t=1000
    rvf_kep = qoe.intJ234DragRV_RK4Step(rv0/1e3, t, 1.4281E-6/1e3,True, 0.01)*1e3
    print("reference Pythoin state vector (6x1):")
    printRV(t,rvf_kep)

    Phi0f = np.zeros((6, 6), order='F')
    order = 1
    # rvf_kep = qoe.daJ234DragRV_RK4Step(rv0,Phi0f, t, scale_rhoCdA_m=1.0, givePhi=True, step=10.0,order=order,scale_derive=0.1)
    rvf_kep = qoe.daJ234DragRV_RK4Step(rv0,Phi0f, t,1.0, True,0.1,order,1.0)
    print(f"DA order-{order:d} state vector (6x1):")
    printRV(t,rvf_kep)


def testDA_ode():
    # Define initial position and velocity vectors
    pos0 = np.array([6925443.952, 190432.624, 230986.901])
    vel0 = np.array([-303.938541, 2277.904454, 7229.098276])
    rv0 = np.concatenate((pos0,vel0))
    tof = list(range(0,5400,10))  # Time of flight in seconds

    # print("Propagated state vector (6x1):")
    t=0
    drv0 = qoe.EigenwarpIntOrbitJ234DragODE(rv0/1e3, t, 1.4281E-12)*1e3
    print("reference d state (6x1):")
    printRV(t,drv0)

    drv0_ = qoe.EigenwarpDAOrbitJ234DragODE(rv0, t, scale_rhoCdA_m=1.0)
    print(f"DA d state (6x1):")
    printRV(t,drv0_)
    print("error: ",drv0-drv0_)

def testDA_class():

    rv0 =np.array([6925443.9520, 190432.6240, 230986.9010,	-303.93854, 2277.90445, 7229.09828])
    rv1 =np.array([6921989.6884,   213199.8038,    303262.5598,    -386.90722,     2275.48603,     7225.88874])

    # print("Propagated state vector (6x1):")
    t=0
    da = qoe.NominalErrorProp(rv0,2)
    Phi0f = np.zeros((6, 6), order='F')
    print(f"DA d state (6x1):")
    rv1_ = da.propNomJ234Drag(Phi0f,0,True)
    printRV(0,rv1_,1)
    rv1_ = da.propNomJ234Drag(Phi0f,10,True)
    printRV(10,rv1_,1)
    rv2_ = da.propNomJ234Drag(Phi0f,20,True)
    printRV(20,rv2_,1)
    
    erv0=np.array([1000,-10000,0,0,1.0,-1.0])

    erv1_ = Phi0f@erv0[:,np.newaxis]
    print("STM error;\n")
    print(erv1_[:,0])
    # printRV(erv1_[:],10,1)
    erv1_da= da.evaldXf(erv0)
    erv1_da = np.array(erv1_da)-rv1
    # printRV(erv1_da,10,1)
    print(erv1_da)

    da = qoe.NominalErrorProp(rv0+erv0,2)
    rv1_ = da.propNomJ234Drag(Phi0f,10,False)
    print(f"正向递推erv0求误差演化 erv1(6x1):")
    drv1 = rv1_-rv1
    printRV(10,drv1,1)
    print('')
    
    da = qoe.NominalErrorProp(rv1,1)
    rv0_0 = da.bkpropNomJ234Drag(Phi0f,10,False,2.0)
    print("Backward prop state(6x1):")
    printRV(0,rv0_0,1)
    erv0_da= da.evaldXp(-drv1)-rv0
    print(f"反向递推erv1误差得出erv0(6x1):",rv0_0-rv0)
    print(f"DA反向估计 estimate(6x1):")
    printRV(0,-erv0_da,1)
    print('')

    da = qoe.NominalErrorProp(rv1_,2)
    rv0_ = da.bkpropNomJ234Drag(Phi0f,10,False,2.0)
    erv0_da= da.evaldXp(-drv1)-rv0
    print(f"t0加误差状态正向递推的结果，反向递推恢复t0误差(6x1):\n",rv0_-rv0)
    print(f"Backward prop estimate(6x1):",erv0_da)


if __name__ == "__main__":
    # test_propagate_lagrangian()
    # testRK4()
    # testDA_ode()
    # testDARK4()
    testDA_class()
    # testIntRK4()