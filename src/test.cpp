
// #include "SpPropagator.h"
#include "elements.h"
#include "kepler.h"
#include "integrater.h"
#include "dastates.h"
void printRV( const Eigen::VectorXd& rvf_kep, int t=10, double m = 1000.0) {
    printf("%04d\t", t);
    for (int i = 0; i < 3; i++) {
        printf("%.4f,\t",  rvf_kep(i)*m);
    }
    for (int i = 3; i < rvf_kep.size(); i++) {
        printf("%.5f,\t",  rvf_kep(i)*m);
    }
    printf("(m,m/s)\n");
}
// 打印 6x6 矩阵的函数
void pmat(const Eigen::Matrix<double, 6, 6>& matrix, double scale = 1.0) {
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            printf("%.3f, ", scale * matrix(i, j));
        }
        printf("\n");
    }
}
// using namespace osculating;
int basicTest(){
    // oscstm::testSigmaInverseMatrix();
    // oscstm::testOscMeanToOsculatingElements();
    // oscstm::testOscMeanSTM();
    // osculating::testOEMeanSTM();
    Vector6d rv0;rv0<<6925443.9520, 190432.6240, 230986.9010,	-303.93854, 2277.90445, 7229.09828;
    Vector6d rv1;rv1<<6921989.6884,   213199.8038,    303262.5598,    -386.90722,     2275.48603,     7225.88874;
    Vector6d rv2;rv2<<6917705.9636,   235941.4358,    375501.7816,    -469.82911,     2272.79496,     7221.81096;
    
    double tf =10.0;
    mat66 p0;
    // DensityInterpolator aeroDensity("/mnt/d/Docker/shared/TZ/nonsingularPredict/src/AtmosphericDensity1976.csv");
    // double rho = aeroDensity.getRho(530);
    Vector6d drv0,drv0_da;
    drv0 = EigenwarpIntOrbitJ234DragODE(rv0/1e3,tf);
    drv0_da= EigenwarpDAOrbitJ234DragODE(rv0, tf,1,true);
    printf("d state \n");
    printRV(drv0);
    printRV(drv0_da,0,1);


// rho0 =3.245746e-4 kg/km^3
// A 2m^2 = 2e-6km^2
// Cd 2.2
// m 1000
// km-1, 改成单位m，需要乘1e3
    Vector6d rv1_,rv1_da;
    rv1_ = intJ234DragRV_RK4Step(rv0/1e3, tf,  1.42812824E-12);// output km
    bool givePhi = true;
    rv1_da = daJ234DragRV_RK4Step(rv0, p0, tf, 1.0,givePhi);// output m
    printf("RK45 state final;\n");
    printRV(rv1_);
    printRV(rv1_da,10,1);

    printf("RK45 state more(20-30s);\n");
    rv1_ = intJ234DragRV_RK4Step(rv1_, tf,  1.42812824E-12);
    printRV(rv1_,20);
    rv1_ = intJ234DragRV_RK4Step(rv1_, tf,  1.42812824E-12);
    printRV(rv1_,30);


    Vector6d erv1_,erv1_da;
    erv1_ = rv1_*1e3 - rv1;
    erv1_da = rv1_da - rv1;
    printf("Error state final;\n");
    printRV(erv1_,10,1);
    printRV(erv1_da,10,1);

    erv1_da = p0*rv0-rv1_da;
    printf("STM error;\n");
    printRV(erv1_da,10,1);

    return 0;
}
int testErrorState(){
    Vector6d rv0;rv0<<6925443.9520, 190432.6240, 230986.9010,	-303.93854, 2277.90445, 7229.09828;
    Vector6d rv1;rv1<<6921989.6884,   213199.8038,    303262.5598,    -386.90722,     2275.48603,     7225.88874;
    Vector6d rv2;rv2<<6917705.9636,   235941.4358,    375501.7816,    -469.82911,     2272.79496,     7221.81096;
    
    double tf =10.0;
    mat66 p0;
    // DensityInterpolator aeroDensity("/mnt/d/Docker/shared/TZ/nonsingularPredict/src/AtmosphericDensity1976.csv");
    // double rho = aeroDensity.getRho(530);

    AlgebraicVector<double> Deltax0(6);
    Vector6d rv1_,rv1_da;
    bool givePhi = true;
    rv1_da = daJ234DragRV_RK4Step(rv0, p0, tf, 1.0,givePhi,   1.0, 2, 1.0);// output m
    printf("RK45 state final;\n");
    printRV(rv1_da,10,1);
    
    NominalErrorProp orb_ref(rv0,2);
    rv1_ = orb_ref.propNomJ234Drag(p0,10.0,false);
    printRV(rv1_,10,1);

    Vector6d erv0;erv0<<1000,-1000,0,0,0,0;
    Vector6d erv1_,erv1_da;

    erv1_da = p0*erv0;
    printf("STM error;\n");
    printRV(erv1_da,10,1);
    erv1_ = orb_ref.evaldXf(erv0)-rv1_da;

    printRV(erv1_,10,1);
    return 0;
}

int main(){
    // basicTest();
    // testErrorState();
    oscstm::testOscMeanSTM();

    return 0;
}