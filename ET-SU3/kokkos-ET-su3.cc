#include <iostream>
//#include <accel.h>
#include <vector>
#include <chrono>
#include <iomanip>
#include <Kokkos_Core.hpp>
#include <complex>
#include "su3.h"
using namespace std;
using namespace su3;

namespace ET {

  ////////////////////////////////////////////////////////////////////////////////////////
  // These either do offload or omp loops depending on GPU vs. Multicore
  // Basic expressions used in Expression Template
  ////////////////////////////////////////////////

  template<class T> using Vector =  std::vector<T>;

  template<class T1, class T2,class T3> struct LatticeBinaryExpression
  {
    T1 Op;               
    T2 arg1;
    T3 arg2;
    LatticeBinaryExpression() : Op(), arg1(), arg2() { }
    LatticeBinaryExpression(T1 ff,T2 ss,T3 tt) : Op(ff), arg1(ss), arg2(tt) { }
    LatticeBinaryExpression(const LatticeBinaryExpression<T1, T2, T3> &p) : Op(p.Op), arg1(p.arg1),arg2(p.arg2) { }
  };

  template<class T1,class T2,class T3>  
  inline LatticeBinaryExpression<T1,T2,T3> makeLatticeBinaryExpression(T1 ff,T2 ss,T3 tt){
    LatticeBinaryExpression<T1,T2,T3> tmp(ff,ss,tt);
    return tmp;
  }

  class Grid { 
  public:
    Grid(int sites) : _sites(sites){ };
    int Osites(void) { return _sites ; };
  private:
    int _sites;
  };

}

namespace ET {
  template<class obj> 
  class Lattice : public Grid { 
  public: 
     typedef Kokkos::CudaUVMSpace MemSpace;
     typedef Kokkos::LayoutLeft Layout;
     typedef Kokkos::View<obj*, Layout, MemSpace> ViewVectorType;
    // ViewVectorType _odatavec('_odatavec',this->Osites());  
     obj * _odata;
     Grid *_grid;
     Vector<obj> *_odatavec;

    Lattice(const Lattice<obj> &copyme)=default; 

    Lattice(Grid *gp) : Grid(*gp), _grid(gp)
    { 
           
     typedef Kokkos::View<obj*, Layout, MemSpace> ViewVectorType;
     ViewVectorType _odatavec('_odatavec', this->Osites());  
    //_odatavec = new Vector<obj>;
    //_odatavec->resize(this->Osites());
     _odata = & (*_odatavec)[0];
    }
    ~Lattice () {}


   inline Lattice<obj> & operator= (const obj & splatme)
    {
      int _osites=this->Osites();
      Kokkos::parallel_for(_osites, [=](const size_t ss){_odata[ss]=splatme;});
      //for(int ss=0;ss<_osites;ss++){
      //	    _odata[ss] = splatme;
      // }
      // }
      //  return *this;
    }

    template <typename Op, typename T1,typename T2> inline Lattice<obj> & operator=(const LatticeBinaryExpression<Op,T1,T2> &expr)
    {
      int _osites=this->Osites();
      Kokkos::parallel_for(_osites, [=](const size_t ss){_odata[ss] = eval(ss, expr);});
      //for(int ss=0;ss<_osites;ss++){
      //	    _odata[ss] = eval(ss,expr);
      // }
      // }
      //  return *this;
    }
  };
  ////////////////////////////////////////////
  //leaf eval of lattice 
  ////////////////////////////////////////////
  template<class obj>  inline obj eval(const unsigned int ss, const Lattice<obj> &arg)
  {
    return arg._odata[ss];
  }

  template<class obj> std::ostream& operator<< (std::ostream& stream, const Lattice<obj> &o){
    int N=o._grid->Osites();
    stream<<"{";
    for(int s=0;s<N-1;s++){
      stream<<(* o._odatavec)[s]<<",";
    }
    stream<<(* o._odatavec)[N-1]<<"}";
    return stream;
  }

  ////////////////////////////////////////////
  // Evaluation of expressions
  ////////////////////////////////////////////
  template <typename Op, typename T1, typename T2> 
  auto inline eval (const unsigned int ss, const LatticeBinaryExpression<Op,T1,T2> &expr) 
    -> decltype(expr.Op.func(eval(ss,expr.arg1),eval(ss,expr.arg2)))
  {
    return expr.Op.func(eval(ss,expr.arg1),eval(ss,expr.arg2));
  }

  ////////////////////////////////////////////
  // Binary operators
  ////////////////////////////////////////////
#define BinOpClass(name,combination)\ 
  template <class left,class right>  struct name {			\
   static auto inline func(const left &lhs,const right &rhs)-> decltype(combination) {\
  return combination;\
  }\
  };
  BinOpClass(BinaryAdd,lhs+rhs);
  BinOpClass(BinarySub,lhs-rhs);
  BinOpClass(BinaryMul,lhs*rhs);

  ////////////////////////////////////////////
  // Operator syntactical glue
  ////////////////////////////////////////////

#define BINOP(name)  name<decltype(eval(0, lhs)), decltype(eval(0, rhs))>
#define DEFINE_BINOP(op, name)                                  \
  template <typename T1,typename T2>   inline auto op(const T1 &lhs,const T2 &rhs)  \
    -> decltype(LatticeBinaryExpression<BINOP(name), T1 , T2 >(makeLatticeBinaryExpression(BINOP(name)(),lhs,rhs))) \
  {\
    return LatticeBinaryExpression<BINOP(name), T1 , T2 >(makeLatticeBinaryExpression(BINOP(name)(),lhs, rhs)); \
  }
  ////////////////////////
  //Operator definitions
  ////////////////////////

#define OPERATOR_PLUS operator+
#define OPERATOR_MINUS operator-
#define OPERATOR_STAR operator*
DEFINE_BINOP(OPERATOR_PLUS ,BinaryAdd);
DEFINE_BINOP(OPERATOR_MINUS,BinarySub);
DEFINE_BINOP(OPERATOR_STAR ,BinaryMul);
}

typedef Kokkos::View<double*[3]> view_type;
struct InitView{
 view_type a;
 InitView (view_type a_):
  a(a_)
  {}

 KOKKOS_INLINE_FUNCTION
 void operator () (const int i) const{
 a(i,0)=1.0*i;
 a(i,1)=1.0*i*i;
 a(i,2)=1.0*i*i*i;
 }
};

struct ReduceFunctor{
 view_type a;
 ReduceFunctor (view_type a_): a (a_){}
 typedef double value_type;
 KOKKOS_INLINE_FUNCTION
 void operator() (int i, double &lsum) const{
   lsum += a(i,0)*a(i,1)/((a(i,2)+0.1));
}
};




///////////////////// The Tests /////////////////////
using namespace ET;
              
int main(int argc,char **argv) {
  Kokkos::initialize(argc, argv);
  
  typedef Kokkos::Cuda ExecSpace;
  typedef Kokkos::CudaUVMSpace MemSpace;
  typedef Kokkos::LayoutLeft Layout;


  typedef Kokkos::View<double*, Layout, MemSpace> ViewVectorType;
 // ViewVectorType Lattice::_odatavec


  std::chrono::high_resolution_clock::time_point start, stop;
  Su3f I;
  int Nloop=1000;

  //Kokkos::initialize(argc, argv);
  const int N = 10;
  view_type a("A", N);
  Kokkos::parallel_for(N, InitView(a));
  double sum=0;
  Kokkos::parallel_reduce(N, ReduceFunctor(a), sum);
  printf("Result: %f\n", sum);


  #ifdef _OPENACC
    int threads = 1;
  #elif defined(_OPENMP)
    int threads = 1;
  #else
    int threads = 1;
  #endif

  std::cout << "===================================================================================================="<<std::endl;
  std::cout << "DISCLAIMER: THIS IS NOT Grid, but definitely looks like Grid" << std::endl;
  std::cout << "===================================================================================================="<<std::endl;
  std::cout << "Grid is setup to use " << threads << " threads" << std::endl;

  std::cout << "===================================================================================================="<<std::endl;
  std::cout << "= Benchmarking SU3xSU3  x= x*y"<<std::endl;
  std::cout << "===================================================================================================="<<std::endl;
  std::cout << "  L  "<<"\t\t"<<"bytes"<<"\t\t\t"<<"GB/s\t\t GFlop/s"<<std::endl;
  std::cout << "----------------------------------------------------------"<<std::endl;
 // #pragma acc kernels
 // {
  for (int lat=2;lat<=8;lat+=2) {

    int vol = lat*lat*lat*lat;
   //// int vol = 4*4*4*4;
    Grid grid(vol);
    Lattice<Su3f> z(&grid);
    Lattice<Su3f> x(&grid);
    Lattice<Su3f> y(&grid);
    x = I;
    y = I;
    z = I;
    start = std::chrono::high_resolution_clock::now();
    
   // #pragma acc parallel loop
    for(int i=0;i<Nloop;i++) {
	x=x*y;
    }
    
    stop = std::chrono::high_resolution_clock::now();

    double time = (std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count())/Nloop*1000.0;
      
    double bytes	= 3.0*vol*9*sizeof(complex<float>);
    double footprint	= 2.0*vol*9*sizeof(complex<float>);
    double flops	= 9*(6.0+8.0+8.0)*vol;

    std::cout << std::setprecision(3) << lat <<"\t\t" << footprint << "    \t\t" << bytes/time << "\t\t" << flops/time << std::endl;
  }
   Kokkos::finalize();
   return 0;
}
//}

