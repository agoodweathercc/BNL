#include <cstdio>
#include <iostream>
#ifdef _GPU
#include <accel.h>
#endif
#include <vector>
#include <tuple>
#include <openacc.h>

namespace ET {

  ////////////////////////////////////////////////////////////////////////////////////////
  // These either do offload or omp loops depending on GPU vs. Multicore
  // Basic expressions used in Expression Template
  ////////////////////////////////////////////////

  template<class T> using Vector =  std::vector<T>;
  template<class T> using Matrix =  std::vector<std::vector<T> >;

  template<class T1, class T2,class T3> struct LatticeBinaryExpression
  {
    T1 Op;               
    T2 arg1;
    T3 arg2;

    LatticeBinaryExpression() : Op(), arg1(), arg2() {
    }

    LatticeBinaryExpression(T1 ff,T2 ss,T3 tt) : Op(ff), arg1(ss), arg2(tt) {
    }

    LatticeBinaryExpression(const LatticeBinaryExpression<T1, T2, T3> &p) : Op(p.Op), arg1(p.arg1), arg2(p.arg2) {
    }

    ~LatticeBinaryExpression() {
    }
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

#if 0
  template<class Expr> __global__
  void ETapply(int N,double *_odata,Expr Op)
  {
    int ss = blockIdx.x;    
    _odata[ss]=Op.func(eval(ss,Op.arg1),eval(ss,Op.arg2));
  }
#endif

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

    ~Lattice () 
    {
    }


   inline Lattice<obj> & operator= (const obj & splatme)
    {
      int _osites=this->Osites();
      #pragma acc parallel loop copyin(splatme[0:1])// Without this no kernels run on the GPU
      for(int ss=0;ss<_osites;ss++){
	 _odata[ss] = splatme;
      }
      return *this;
    }

    template <typename Op, typename T1,typename T2> inline Lattice<obj> & operator=(const LatticeBinaryExpression<Op,T1,T2> &expr)
    {
      int _osites=this->Osites();
      #pragma acc parallel loop independent copyin(expr)
      for(int ss=0;ss<_osites;ss++){
	_odata[ss] = eval(ss,expr);
      }
      return *this;
    }
  };

/* Template specialization */

  template<class T1> struct LatticeBinaryExpression<T1, Lattice<double>,Lattice<double>>
  {
    T1 Op;               
    Lattice<double> arg1;
    Lattice<double> arg2;

    LatticeBinaryExpression() : Op(), arg1(), arg2() {
    }

    LatticeBinaryExpression(T1 ff, Lattice<double> ss, Lattice<double> tt) : Op(ff), arg1(ss), arg2(tt) {
    }

    LatticeBinaryExpression(const LatticeBinaryExpression<T1, Lattice<double>, Lattice<double>> &p) : Op(p.Op), arg1(p.arg1), arg2(p.arg2) {
    }

    ~LatticeBinaryExpression() {
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
  template <typename T1,typename T2>   \
   inline auto op(const T1 &lhs,const T2 &rhs)  \
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
              
int main(int argc,char **argv){
  
  const int length = 2400;
  typedef Matrix<double>(3,3) U3;
   
 
  Grid grid(length);
  Lattice<U3> v1(&grid); 
  Lattice<U3> v2(&grid);
  Lattice<U3> v3(&grid);
   
  const U3 m1,m2;
  for(int i=0; i<m1.size(); i++) 
    for(int j=0; i<m1[i].size; j++) {
      m1[i][j] = 1.0;
      m2[i][j] = 2.0;
  }
  v1=m1;
  v2=m2;
//  v3=0.0;
  std::cout<<"initialised arrays "<<v1<<" "<<v2<<std::endl;
//  acc_present_dump();
  v3=v1+v2; 
//  acc_present_dump();
  std::cout<<"v3 = v1+v2"<< v3<<std::endl;
  
  v3=v1+v2+v1*v2+v2;
  //acc_present_dump();
  std::cout<<"v3 = v1+v2+v1*v2"<< v3<<std::endl;
};


