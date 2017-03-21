#include <iostream>
#include <cstdio>
#include <cmath>
#include <random>


void vec2mat(const uint64_t I, const uint64_t J, double* vec, double** mat)
{
    for (uint64_t i = 0; i < I; i++)
        mat[i] = &(vec[i * J]);
}


void fill_uniform_rand(const uint64_t length, double* M,
                       const double start, const double end)
{
    std::random_device rd;
    std::mt19937_64 rng(rd());
    std::uniform_real_distribution<> dis(start, end);

    for (uint64_t i = 0; i < length; i++) {
	M[i] = dis(rng);
    }
}

int main()
{
 double error = 1;
 double tol = 0.0001;
 int iter = 1;
 int iter_max = 5001;
 int m=100;
 int n=100;
 double* _A = new double[m * n];
 double* _Anew = new double[m * n];
double** A = new double*[m];
double** Anew = new double*[m];


vec2mat(m, n, _A, A);
vec2mat(m, n, _Anew, Anew);

fill_uniform_rand(100 * 100, _Anew, -1, 1);
fill_uniform_rand(100 * 100, _A, -1, 1);


 while (error > tol && iter< iter_max){
 error = 0.0;
#pragma acc parallel loop reduction(max:error)
 for(int j=1; j < n-1; j++){
  #pragma acc loop reduction(max:error)
  for(int i =1; i<m-1; i++){
    Anew[j][i] = 0.25*(A[j][i+1]+A[j][i-1]+A[j-1][i]+A[j+1][i]);
    error = fmax(error, fabs(Anew[j][i]-A[j][i]));
   }
 }

#pragma acc parallel loop
for(int j=1; j<n-1; j++){
   #pragma acc loop
   for(int i=1; i<m-1; i++){
     A[j][i]=Anew[j][i];
  }
}

if(iter % 100 ==0) printf("%5d, %0.6f\n",iter, error);

iter++;
	}

std::cout << "Done!\n";

}
