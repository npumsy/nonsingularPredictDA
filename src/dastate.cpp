#include "dastates.h"
#include <Eigen/Core>
// #include <Eigen/Dense>
using namespace std; 
using namespace DACE;


template<typename T>
AlgebraicVector<T> TBPfullwarp(AlgebraicVector<T> x, double t, double arg1){
    return TBPfull(x,t ,arg1);
}
// Exercise 6.2.1: 3/8 rule RK4 integrator
template<typename T> T rk4( T x0, double t0, double t1, T (*f)(T,double,double) ,double arg1, double hmax)
{
	int steps = ceil( (t1-t0)/hmax );
	double h = (t1-t0)/steps;
    double t = t0;

    T k1, k2, k3, k4;
	for( int i = 0; i < steps; i++ )
	{
        k1 = f( x0, t ,arg1);
        k2 = f( x0 + h*k1/3.0, t + h/3.0 ,arg1 );
        k3 = f( x0 + h*(-k1/3.0 + k2), t + 2.0*h/3.0 ,arg1);
        k4 = f( x0 + h*(k1 - k2 + k3), t + h ,arg1);
        x0 = x0 + h*(k1 + 3*k2 + 3*k3 +k4)/8.0;
		t += h;
	}

    return x0;
}
template<typename T> T rk4b( T x0, double t0, double t1, T (*f)(T,double,double) ,double arg1, double hmax)
{
	int steps = ceil( (t1-t0)/hmax );
	double h = (t1-t0)/steps;
    double t = t0;

    T k1, k2, k3, k4;
	for( int i = 0; i < steps; i++ )
	{
        k1 = f( x0, t ,arg1);
        k2 = f( x0 - h*k1/3.0, t - h/3.0 ,arg1 );
        k3 = f( x0 - h*(-k1/3.0 + k2), t - 2.0*h/3.0 ,arg1);
        k4 = f( x0 - h*(k1 - k2 + k3), t - h ,arg1);
        x0 = x0 - h*(k1 + 3*k2 + 3*k3 +k4)/8.0;
		t += h;
	}

    return x0;
}
NominalErrorProp::NominalErrorProp(const Vector6d &rv0,int order){
    const int N = 6;
     DA::init( order, N );       // initialize DACE for 1st-order computations in 2 variables

    x0 = AlgebraicVector<DA>(6);
    xf = AlgebraicVector<DA>(6);
    xp = AlgebraicVector<DA>(6);
   
    for(int i=0;i<6;i++)x0[i]=rv0(i)/1e3 + DA(i+1);

    DA::pushTO( 1 );    // only first order derivative needed

}
void NominalErrorProp::updateX0(const Vector6d &rv0){
    for(int i=0;i<6;i++)x0[i]=rv0(i)/1e3 + DA(i+1);
}
Vector6d NominalErrorProp::propNomJ234Drag( Eigen::Ref<Eigen::Matrix<double, 6, 6>> Phi0f, double tf,
                                    bool givePhi, double step){
    xf = rk4( x0, 0, tf, TBPfullwarp ,1.0 ,step);
    
    if(givePhi)
        for( int i = 0; i < 6; i++ )
        {
            for( int j = 1; j <= 6; j++ )
            {
                Phi0f(i,j-1)=cons(xf[i].deriv(j));
            }
        }
    Vector6d rvf;
    for(int i=0;i<6;i++)rvf[i]=cons(xf[i])*1e3;
    return rvf;
}
Vector6d NominalErrorProp::bkpropNomJ234Drag( Eigen::Ref<Eigen::Matrix<double, 6, 6>> Phi0f, double tp,
                                    bool givePhi, double step){
    xp = rk4b( x0, 0, tp, TBPfullwarp ,1.0 ,step);
    
    if(givePhi)
        for( int i = 0; i < 6; i++ )
        {
            for( int j = 1; j <= 6; j++ )
            {
                Phi0f(i,j-1)=cons(xp[i].deriv(j));
            }
        }
    Vector6d rvf;
    for(int i=0;i<6;i++)rvf[i]=cons(xp[i])*1e3;
    return rvf;
}
// // 函数返回一个 6x6x6 的张量
// Eigen::Tensor<double, 6> returnTensor() {
//     // 创建一个 3x4x5 的张量
//     Eigen::Tensor<double, 6> tensor(6,6,6);

//     // 初始化张量的值
//     for (int i = 0; i < 3; ++i) {
//         for (int j = 0; j < 4; ++j) {
//             for (int k = 0; k < 5; ++k) {
//                 tensor(i, j, k) = i * 100 + j * 10 + k;
//             }
//         }
//     }

//     return tensor;
// }
Vector6d NominalErrorProp::evaldXf(const Vector6d &drv0,double scale_rhoCdA_m){
    Vector6d drvf;
    AlgebraicVector<double> Deltax0(6),Deltaxf(6);
    for(int i=0;i<6;i++){Deltax0[i]=drv0[i]/1e3;}
    Deltaxf=xf.eval(Deltax0)*1e3;
    for(int i=0;i<6;i++)drvf[i]=Deltaxf[i];
    return drvf;
}
Vector6d NominalErrorProp::evaldXp(const Vector6d &drv0,double scale_rhoCdA_m){
    Vector6d drvf;
    AlgebraicVector<double> Deltax0(6),Deltaxf(6);
    for(int i=0;i<6;i++){Deltax0[i]=drv0[i]/1e3;}
    Deltaxf=xp.eval(Deltax0)*1e3;
    for(int i=0;i<6;i++)drvf[i]=Deltaxf[i];
    return drvf;
}
NominalErrorProp::~NominalErrorProp(){
    DA::popTO( );
}


template<typename T> AlgebraicVector<T> TBP( AlgebraicVector<T> x, double t )
{
    
    AlgebraicVector<T> pos(3), res(6);
    
    pos[0] = x[0]; pos[1] = x[1]; pos[2] = x[2];
    
    T r = pos.vnorm();
    
    const double mu = 398600; // km^3/s^2
    
    res[0] = x[3];
    res[1] = x[4];
    res[2] = x[5];
    
    res[3] = -mu*pos[0]/(r*r*r);
    res[4] = -mu*pos[1]/(r*r*r);
    res[5] = -mu*pos[2]/(r*r*r);
    
    return res;
    
}

template<typename T>
AlgebraicVector<T> TBPfull(AlgebraicVector<T> x, double t,double beta, double mu, double Re, double rhoCdA_m,double h0,double H0) {
    AlgebraicVector<T> pos(3),vel(3), res(6);
    
    pos[0] = x[0]; pos[1] = x[1]; pos[2] = x[2];
    // vel[0]=x[3];vel[1]=x[4];vel[2]=x[5];
    
    T r = pos.vnorm();
    T z_r =pos[2] / r;
    T z2_r2 = z_r*z_r;
    T z3_r3 = z2_r2 * z_r;
    T z4_r4 = z3_r3 * z_r;
    T Re_r = Re/r;
    T Re2_r2 = Re_r * Re_r;
    T Re3_r3 = Re2_r2 * Re_r;
    T Re4_r4 = Re3_r3 * Re_r;

    T common_factor = -mu /pow(r,3);

    T J2_term = (3.0 / 2.0) * bddd::J2 * Re2_r2 * (1.0 - 5.0 * z2_r2);
    T J3_term = (5.0 / 2.0) * bddd::J3 * Re3_r3 * (3.0 * z_r - 7.0 * z3_r3);
    T J4_term = (5.0 / 8.0) * bddd::J4 * Re4_r4 * (3.0 - 42.0 * z2_r2 + 63.0 * z4_r4);

    res[3] = common_factor * pos[0] * (1.0 + J2_term + J3_term - J4_term);
    res[4] = common_factor * pos[1] * (1.0 + J2_term + J3_term - J4_term);


    J2_term = (3.0 / 2.0) * bddd::J2 * Re2_r2 * (3.0 - 5.0 * z2_r2);
    J3_term = (5.0 / 2.0) * bddd::J3 * Re3_r3 * (6.0 * z_r- 7.0 * z3_r3 - (3.0 / 5.0) * r / pos[2]);
    J4_term = (5.0 / 8.0) * bddd::J4 * Re4_r4 * (15.0 - 70.0 * z2_r2 + 63.0 * z4_r4);

    res[5] = common_factor * pos[2] * (1.0 + J2_term + J3_term - J4_term);
 if(printEachStepPertub_da)cout<<"DA---J234: (m/s^2)="<<cons(res[3]-common_factor * pos[0] )*1e3<<",\t"<<cons(res[4]-common_factor * pos[1] )*1e3<<",\t"<<cons(res[5]-common_factor * pos[2] )*1e3<<endl;
    // cout<<"DA---a_x, a_y, a_z(m/s^2)="<<cons(res[3]-common_factor * pos[0] )*1e3<<",\t"<<cons(res[4]-common_factor * pos[1] )*1e3<<",\t"<<cons(res[5]-common_factor * pos[2] )*1e3<<endl;

    // Calculate drag acceleration
    // T v = vel.vnorm(); // velocity magnitude
    AlgebraicVector<T> rel_vel = {x[3] +bddd::OMEGA_EARTH * x[1], x[4] -bddd::OMEGA_EARTH * x[0], x[5]}; // Earth rotation correction
    T v = rel_vel.vnorm();
    AlgebraicVector<T> drag_acc = {
        -0.5 * rhoCdA_m * beta*exp(-(r-Re - h0) / H0) * v * rel_vel[0],
        -0.5 * rhoCdA_m * beta*exp(-(r-Re - h0) / H0) * v * rel_vel[1],
        -0.5 * rhoCdA_m * beta*exp(-(r-Re - h0) / H0) * v * rel_vel[2]
    };
// cout<<"DA---v, v_x,v_y, v_z(m/s^2)="<<cons(v)*1e3<<",\t"<<cons(rel_vel[0])*1e3<<",\t"<<cons(rel_vel[1])*1e3<<",\t"<<cons(rel_vel[2])*1e3<<endl;

    // Combine accelerations
    res[0] = x[3];
    res[1] = x[4];
    res[2] = x[5];
    res[3] +=drag_acc[0];
    res[4] +=drag_acc[1];
    res[5] +=drag_acc[2];

 if(printTotalAccleration_da)cout<<"DA-- a_x, a_y, a_z(m/s^2)="<<1e3*cons(res[3])<<",\t"<<1e3*cons(res[4])<<",\t"<<1e3*cons(res[5])<<endl;
  if(printEachStepPertub_da)cout<<"DA---drag: (m/s^2)="<<cons(drag_acc[0])*1e3<<",\t"<<cons(drag_acc[1])*1e3<<",\t"<<cons(drag_acc[2])*1e3<<endl;

    return res;
}


Vector6d EigenwarpDAOrbitJ234DragODE(const Vector6d &rv0, double t, double arg1,bool J234){
    AlgebraicVector<double> x(6),dx(6);
    for(int i=0;i<6;i++)x[i]=rv0[i]/1e3;
    dx =  TBPfull(x,t ,arg1);
    Vector6d rdx(6);
    for(int i=0;i<6;i++)rdx[i]=dx[i]*1e3;
    return rdx;
}

void ex6_2_3(double T=10 )
{
    AlgebraicVector<double> x0(6);
    AlgebraicVector<DA>  xf(6);
    x0[0] = 6716.3932; 
    x0[1] = -1389.0295; 
    x0[2] = -992.5427; 
    x0[3] = 1.48411; 
    x0[4] = 2.06038; 
    x0[5] = 7.15010;

    double scale=0.01;
    AlgebraicVector<DA> x = x0 + scale*AlgebraicVector<DA>::identity( );

    DA::pushTO( 1 );    // only first order computation needed

    x = rk4( x, 0, T, TBPfullwarp ,0,1.0);

    cout << "Exercise 6.2.3: CR3BP STM" << endl;
    cout.precision( 6 );
    cout << cons(x);

    Eigen::MatrixXd Phi0f(6,6);

    for( int i = 0; i < 6; i++ )
    {
        for( int j = 1; j <= 6; j++ )
        {
            cout << cons(x[i].deriv(j))/scale << "  ";
            Phi0f(i,j-1)=cons(x[i].deriv(j))/scale;
        }
        cout << endl;
    }
    cout << endl;
    cout<<"Eigen\n";
    cout<<Phi0f<<endl;
    DA::popTO( );
}
// 函数内部计算单位为km
Vector6d daJ234DragRV_RK4Step(const Vector6d &rv0, Eigen::Ref<Eigen::Matrix<double, 6, 6>> Phi0f, double tf,
                                    double scale_rhoCdA_m,bool givePhi, double step,int order,double scale){
    const int N = 6;
    DA::init( order, N );       // initialize DACE for 1st-order computations in 2 variables
    AlgebraicVector<double> x0(6);
    for(int i=0;i<6;i++)x0[i]=rv0(i)/1e3;

    AlgebraicVector<DA>  xf(6);

    AlgebraicVector<DA> x = x0 + scale*AlgebraicVector<DA>::identity( );

    DA::pushTO( 1 );    // only first order derivative needed
    x = rk4( x, 0, tf, TBPfullwarp ,scale_rhoCdA_m,step);

    Vector6d rvf;
    for(int i=0;i<6;i++)rvf[i]=cons(x[i])*1e3;
if(givePhi)
    for( int i = 0; i < 6; i++ )
    {
        for( int j = 1; j <= 6; j++ )
        {
            Phi0f(i,j-1)=cons(x[i].deriv(j))/scale;
        }
    }
    DA::popTO( );
    return rvf;
}
int main_(int argc, char *argv[])
{
    int order = 3;
    double tf=10;
    if (argc > 1) {
        try {
            order= std::stoi(argv[1]);
            if(argc>2){
                tf= std::stoi(argv[2]);
            }
        } catch (const std::invalid_argument& ia) {
            std::cerr << "Invalid argument, please provide an integer." << std::endl;
        }
    } 
    DA::init( order, 6 );       // initialize DACE for 1st-order computations in 2 variables

    ex6_2_3( tf);
    return 0;
}

