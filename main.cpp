#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

int main() {
	Vector2d x_es;
	x_es<<-1.0,-1.0;
	
	//first example
    Matrix2d mat_1;
    mat_1 << 5.547001962252291e-01, -3.770900990025203e-02,
           8.320502943378437e-01, -9.992887623566787e-01;
    Vector2d b_1;
    b_1 << -5.169911863249772e-01, 
         1.672384680188350e-01;  
	// QR	
	Vector2d x_1 = mat_1.colPivHouseholderQr().solve(b_1);
	cout<<"x solution for first ex , QR factorization is equal to "<<x_1<<endl;
	double err_rel_qr_1=(x_1-x_es).norm()/x_es.norm();
	cout<<"Relative error for example 1 with qr is :"<<abs(err_rel_qr_1)<<endl;
	//LU partialpivlu first example
	FullPivLU<Matrix2d> lu(mat_1);
	Vector2d x_Lu_1 = lu.solve(b_1);
	//cout<<"x LU first example is equal to"<<x_Lu_1<<endl;
	double err_rel_lu_1=(x_Lu_1-x_es).norm()/x_es.norm();
	cout<<"x solution for first ex , PALU factorization is equal to "<<x_Lu_1<<endl;
	cout<<"Relative error with Palu first example is :"<<abs(err_rel_lu_1)<<endl;
	
	//second example
    Matrix2d mat_2;
    mat_2 << 5.547001962252291e-01, -5.540607316466765e-01, 8.320502943378437e-01,
-8.324762492991313e-01;
    Vector2d b_2;
    b_2 << -6.394645785530173e-04, 4.259549612877223e-04;  
	// QR	
	Vector2d x_2 = mat_2.colPivHouseholderQr().solve(b_2);
	double err_rel_qr_2=(x_2-x_es).norm()/x_es.norm();
	cout<<"x solution for second ex , QR factorization is equal to "<<x_2<<endl;
	cout<<"Relative error for example 2 with qr is :"<<abs(err_rel_qr_2)<<endl;
	//LU partialpivlu first example
	FullPivLU<Matrix2d> lu_2(mat_2);
	Vector2d x_Lu_2 = lu_2.solve(b_2);
	double err_rel_lu_2=(x_Lu_2-x_es).norm()/x_es.norm();
	cout<<"x solution for second ex , PALU factorization is equal to "<<x_Lu_2<<endl;
	cout<<"Relative error with Palu second example is :"<<abs(err_rel_lu_2)<<endl;
	
	
	//third example
    Matrix2d mat_3;
    mat_3 << 5.547001962252291e-01, -5.547001955851905e-01, 8.320502943378437e-01,
-8.320502947645361e-01;
    Vector2d b_3;
    b_3 << -6.400391328043042e-10, 4.266924591433963e-10;  
	// QR	
	Vector2d x_3 = mat_3.colPivHouseholderQr().solve(b_3);
	double err_rel_qr_3=(x_3-x_es).norm()/x_es.norm();
	cout<<"x solution for third ex , QR factorization is equal to "<<x_3<<endl;
	cout<<"Relative error for example 3 with qr is :"<<abs(err_rel_qr_3)<<endl;
	FullPivLU<Matrix2d> lu_3(mat_3);
	Vector2d x_Lu_3 = lu_3.solve(b_3);
	double err_rel_lu_3=(x_Lu_3-x_es).norm()/x_es.norm();
	cout<<"x solution for third ex , PALU factorization is equal to "<<x_Lu_3<<endl;
	cout<<"Relative error with Palu third example is :"<<abs(err_rel_lu_3)<<endl;
	
	
    return 0;
}
