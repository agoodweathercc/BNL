#include <iostream>
#include <accel.h>
#include <vector>
#include <chrono>
#include <iomanip>

#include"su3.h"

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
    obj * _odata;
    Grid *_grid;
    Vector<obj> *_odatavec;

  public:
    Lattice(const Lattice<obj> &copyme)=default; // NOTE doesn't create array copy
                                                         // Copies pointer and I'm using this to pass
                                                         // Lattice info through the lambda capture to 
                                                         // expr.second._odata  be the device pointer

    Lattice(Grid *gp) : Grid(*gp), _grid(gp)
    { 
      _odatavec = new Vector<obj>;
      _odatavec->resize(this->Osites());
      _odata = & (*_odatavec)[0];
    }
    ~Lattice () {}


   inline Lattice<obj> & operator= (const obj & splatme)
    {
      int _osites=this->Osites();
      #pragma acc parallel loop independent copyin(splatme[0:1])
      for(int ss=0;ss<_osites;ss++){
	    _odata[ss] = splatme;
      }

      return *this;
    }

    template <typename Op, typename T1,typename T2> inline Lattice<obj> & operator=(const LatticeBinaryExpression<Op,T1,T2> &expr)
    {
      int _osites=this->Osites();
      #pragma acc parallel loop gang copyin(expr[0:1])
      for(int ss=0;ss<_osites;ss++){
	    _odata[ss] = eval(ss,expr);
      }

      return *this;
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

///////////////////// The Tests /////////////////////
using namespace ET;
              
int main(int argc,char **argv) {

  std::chrono::high_resolution_clock::time_point start, stop;
  Su3f I;
  int Nloop=1000;

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

  for (int lat=2;lat<=32;lat+=2) {

    int vol = lat*lat*lat*lat;
    Grid grid(vol);
    Lattice<Su3f> z(&grid);
    Lattice<Su3f> x(&grid);
    Lattice<Su3f> y(&grid);
    x = I;
    y = I;
    z = I;
    start = std::chrono::high_resolution_clock::now();

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
}


