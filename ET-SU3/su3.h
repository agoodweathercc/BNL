#ifndef	SU3_MATRIX
	#define	SU3_MATRIX
	#include<complex>

	using namespace std;

	namespace su3 {
		template<class Float>
		class	Su3 {

			private:
			complex<Float> A[9];

			public:

//					 Su3(const Su3<Float> &copyme)=default;
					 Su3(const Float &copyMe) { 
						 for(int i=0; i<9; i++) { A[i] = copyMe; }
						 #pragma acc enter data copyin(this,A[0:9])
					}

					 Su3() { 
						 for(int i=0; i<9; i++) { A[i] = 0.; }
						 A[0] = A[4] = A[8] = 1.;
					}

					~Su3() { 
						#pragma acc exit data delete(this,A[0:9])
					}

			#pragma acc routine seq
			Su3<Float> operator=(const Su3<Float> &rhs) { for(int i=0; i<9; i++) { A[i] = rhs.A[i]; } return (*this); }

			#pragma acc routine seq
			friend inline Su3<Float> operator+=(Su3<Float> lhs, const Su3<Float> &rhs) { for(int i=0; i<9; i++) { lhs.A[i] += rhs.A[i]; } return lhs; }
			#pragma acc routine seq
			friend inline Su3<Float> operator-=(Su3<Float> lhs, const Su3<Float> &rhs) { for(int i=0; i<9; i++) { lhs.A[i] -= rhs.A[i]; } return lhs; }
			#pragma acc routine seq
			friend inline Su3<Float> operator*=(Su3<Float> lhs, const Su3<Float> &rhs) {

				
				complex <Float>  tmp[9];

				for(int i=0; i<3; i++)
			        for(int k=0; k<3; k++) 
				    for(int j=0; j<3; j++)
						tmp[3*i+k] += lhs.A[3*i+j]*rhs.A[3*j+k];

				for(int i=0; i<3; i++)
				   for(int k=0; k<3; k++)
					lhs.A[3*i+k] = tmp[3*i+k];
				return lhs;
			}

			#pragma acc routine seq
			friend inline Su3<Float> operator+ (const Su3<Float> &lhs, const Su3<Float> &rhs) { return (lhs += rhs); }
			#pragma acc routine seq
			friend inline Su3<Float> operator- (const Su3<Float> &lhs, const Su3<Float> &rhs) { return (lhs -= rhs); }
			#pragma acc routine seq
			friend inline Su3<Float> operator* (const Su3<Float> &lhs, const Su3<Float> &rhs) { return (lhs *= rhs); }

			#pragma acc routine seq
			inline complex<Float>	Trace () { return (A[0] + A[4] + A[8]); }

			friend ostream& operator<<(ostream& os, const Su3<Float> &rhs) {
				for(int i=0; i<3; i++) {
					for(int j=0; j<3; j++) 
						os << "  " << rhs.A[i*3+j];

					os << endl;
				}
				os << endl;

				return	os;
			}

			#pragma acc routine seq
			inline Su3<Float>& operator= (const complex<Float> &rhs) {
				for (int i=0; i<9; i++)
					A[i] = rhs;
				return *this;
			}
		};

		typedef Su3<float>  Su3f;
		typedef Su3<double> Su3d;
/*
		template<class Float> std::ostream& operator<< (std::ostream &os, const Su3<Float> &rhs) {

			for(int i=0; i<3; i++) {
				for(int j=0; j<3; j++) 
					os << "  " << rhs.A[i*3+j];

				os << endl;
			}

			os << endl;
			return os;
		}
*/
	}
#endif
