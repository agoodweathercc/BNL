#include <cstdio>
#include <iostream>
#ifdef _GPU
#include <accel.h>
#endif
#include <vector>
#include <tuple>
#include <typeinfo>
//#include <openacc.h>
#include<Kokkos_Core.hpp>

namespace ET {

  // These either do offload or omp loops depending on GPU vs. Multicore
  // Basic expressions used in Expression Template

  template<class T> using Vector =  std::vector<T>;
  template<class T> using Matrix =  std::vector<std::vector<T> >;

  template<class T1, class T2,class T3> struct LBE
  {
    T1 Op;               
    T2 arg1;
    T3 arg2;

    LBE() : Op(), arg1(), arg2() { }
    LBE(T1 ff,T2 ss,T3 tt) : Op(ff), arg1(ss), arg2(tt) { }
    LBE(const LBE<T1, T2, T3> &p) : Op(p.Op), arg1(p.arg1), arg2(p.arg2) { }
    ~LBE() {}
  };

  template<class T1,class T2,class T3>  
  inline LBE<T1,T2,T3> makeLBE(T1 ff,T2 ss,T3 tt){
    LBE<T1,T2,T3> tmp(ff,ss,tt);
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
    Lattice(const Lattice<obj> &copyme)=default; 
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
     // #pragma acc parallel loop copyin(splatme[0:1])// Without this no kernels run on the GPU
     Kokkos::parallel_for(_osites, [=](const size_t ss){_odata[ss]=splatme;} );
     // for(int ss=0;ss<_osites;ss++){
     //	 _odata[ss] = splatme;
     // }
     // return *this;
    }

    template <typename Op, typename T1,typename T2> inline Lattice<obj> & operator=(const LBE<Op,T1,T2> &expr)
    {
      int _osites=this->Osites();
     // #pragma acc parallel loop independent copyin(expr[0:1])
     Kokkos::parallel_for(_osites, [=](const size_t ss){_odata[ss]=eval(ss, expr);});
     //   for(int ss=0;ss<_osites;ss++){
     //	_odata[ss] = eval(ss,expr);
     // }
     // return *this;
    }
  };

  //leaf eval of lattice 

  template<class obj> 
  inline obj eval(const unsigned int ss, const Lattice<obj> &arg)
   { return arg._odata[ss]; }
   

  template<class obj> std::ostream& operator<< (std::ostream& stream, const Lattice<obj> &o){
    int N=o._grid->Osites();
    stream<<"{";
    for(int s=0;s<N-1;s++){
      stream<<(* o._odatavec)[s]<<",";
    }
    stream<<(* o._odatavec)[N-1]<<"}";
    return stream;
  }

  // Evaluation of expressions
  template <typename Op, typename T1, typename T2> 
  auto inline eval (const unsigned int ss, const LBE<Op,T1,T2> &expr) -> decltype(expr.Op.func(eval(ss,expr.arg1),eval(ss,expr.arg2)))
   {
     return expr.Op.func(eval(ss,expr.arg1),eval(ss,expr.arg2));
   }
 
 
  // Binary operators
#define BinOpClass(name,combination)\ 
  template <class left,class right>  struct name {\
   static auto inline func(const left &lhs,const right &rhs)-> decltype(combination) {\
    return combination;\
    }\
  };

  BinOpClass(BinaryAdd,lhs+rhs);
  BinOpClass(BinarySub,lhs-rhs);
  BinOpClass(BinaryMul,lhs*rhs);

  // Operator syntactical glue

#define BINOP(name)  name<decltype(eval(0, lhs)), decltype(eval(0, rhs))>

#define DEFINE_BINOP(op, name)                                  \
  template <typename T1,typename T2>   \
   inline auto op(const T1 &lhs,const T2 &rhs)  \
    -> decltype( LBE<BINOP(name),T1,T2> (makeLBE(BINOP(name)(),lhs,rhs))) \
  {\
    return LBE<BINOP(name),T1,T2>(makeLBE(BINOP(name)(),lhs, rhs)); \
  } 

//Operator definitions
//#define OPERATOR_PLUS operator+
//#define OPERATOR_MINUS operator-
//#define OPERATOR_STAR operator*
DEFINE_BINOP(operator+ ,BinaryAdd);
DEFINE_BINOP(operator-,BinarySub);
DEFINE_BINOP(operator* ,BinaryMul);
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
              
int main(int argc,char **argv){
  
  const int length = 240;
  Kokkos::initialize(argc, argv); 
  Grid grid(length);
  Lattice<double> v1(&grid); 
  Lattice<double> v2(&grid);
  Lattice<double> v3(&grid);
   
  v1=1.2;
  v2=2.3;
  v3=0.0;
  std::cout<<"initialised arrays "<<v1<<" "<<v2<<std::endl;
//  acc_present_dump();
  v3=v1+v2; 
//  acc_present_dump();
  std::cout<<"v3 = v1+v2"<< v3<<std::endl;
  
  auto V = v1 + v2 + v1 * v2 + v2;
  std::cout << typeid(V).name() << '\n';

  v3=v1+v2+v1*v2+v2;
  //acc_present_dump();
  std::cout<<"v3 = v1+v2+v1*v2"<< v3<<std::endl;


  // Kokkos::initialize  (argc, argv);
  const int N = 10;
  view_type a("A", N);
  Kokkos::parallel_for(N, InitView(a));
  double sum=0;
  Kokkos::parallel_reduce (N, ReduceFunctor (a), sum);
  printf("Result: %f\n", sum);
  Kokkos::finalize();


};





