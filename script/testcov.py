import qoe 
import numpy as np
def test_eci2coe():
    # 使用 CartState 类
    o_rv = qoe.CartState(6925443.952, 190432.624, 230986.901, -303.938541, 2277.904454, 7229.098276)

    pos = o_rv.Pos()
    vel = o_rv.Vel()
    print(pos)

    o_coe =qoe.OrbitElem()
    qoe.CartToOrbitElem(o_rv,qoe.GM_Earth(),o_coe)

    print(o_coe.SMajAx())

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

def testOSCstm3():
    J2 = qoe.Earth_J2()
    Re = qoe.EarthRadius()
    mu = qoe.GM_Earth()
    x_ref = np.array([6925443.952, 190432.624, 230986.901, -303.938541, 2277.904454, 7229.098276])

    OE_ref = qoe.rv2OEOsc(x_ref)
    print("Osculating Elements set:", OE_ref)

    OEMean = qoe.OEOsc2OEMeanEU(OE_ref,100,1e-5,1e-4)
    print("Mean Elements ref:", OEMean)

    tol = 1e-10
    t = np.array([0.0, 3600.0])
    PhiOE  = qoe.STM_ONsElemsWarppedByCOE(OEMean,t[1])
    OEf = qoe.OscElemsLongpropagate(t[1],OEMean)

    print("Final Elements funLong:", OEf)
    OEf2= PhiOE@(OEMean[:,np.newaxis])[:,0]
    print("Final Elements STM:", OEf2)

    
def testAllosc():
    # qoe.testOEMean()

    # qoe.testAllosc()


# # OSC STM
# qoe.testlam2theta()
# qoe.testSigmaMat()
    qoe.testSigmaInverseMatrix()
    # qoe.testOscMeanToOsculatingElements()

        # qoe.testOscMeanSTM()
def testOSCstm():
    J2 = qoe.Earth_J2()
    Re = qoe.EarthRadius()
    mu = qoe.GM_Earth()
    x = np.array([6925443.952, 190432.624, 230986.901, -303.938541, 2277.904454, 7229.098276])

    OE = qoe.rv2OEOsc(x)
    OEMean = qoe.OEOsc2OEMeanEU(OE,100,1e-5,1e-4)
    ICM_ref = qoe.OE2Osc(OEMean)
    ICSc_ref = qoe.OE2Osc(OE)
    print("Osculating Elements set:", ICSc_ref)

    DJ2 = np.zeros((6, 6), order='F')
    ICSc = qoe.OscMeanToOsculatingElements(J2, ICM_ref, Re, mu)
    _    = qoe.DMeanToOsculatingElements  (J2, ICM_ref, Re, mu, DJ2)
    print(">> Osculating Elements converted:", ICSc)
    print(">> Osculating Elements converted2:", np.matmul(DJ2,ICM_ref))

    tol = 1e-10
    t = np.array([0.0, 3600.0])

    OEMean_ref = qoe.OE2Osc(OEMean)
    print("Mean Elements ref:", OEMean_ref)

    meanSTM = qoe.OscMeanElemsSTM(J2, t, ICM_ref, Re, mu, tol)
    print(">> State Transition Matrix phi_J2:\n", meanSTM)

def testOsculating():
    J2 = qoe.Earth_J2()
    Re = qoe.EarthRadius()
    mu = qoe.GM_Earth()
    x_ref = np.array([6925443.952, 190432.624, 230986.901, -303.938541, 2277.904454, 7229.098276])

    OE_ref = qoe.rv2OEOsc(x_ref)
    print("Osculating Elements set:", OE_ref)

    OEMean = qoe.OEOsc2OEMeanEU(OE_ref,100,1e-5,1e-4)
    print("Mean Elements ref:", OEMean)

    OE = qoe.OEMeanEU2OEOsc(OEMean)
    print("Osculating Elements converted:", OE)
    x = qoe.OEOsc2rv(OE,100, 1e-6)
    print("Position velocity converted:", x)



if __name__ == "__main__":
    # test_propagate_lagrangian()
    testOSCstm3()
    # # testAllosc()
    # testOSCstm()
    # # testOsculating()