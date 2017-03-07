gpu-su3.x: ET-su3.cc su3.h
	pgc++ -I. -w -acc -fast -ta=tesla:cc35,managed -Minline --c++11 -Mlarge_arrays -Minfo=accel --no_exceptions -O3 ET-su3.cc -o gpu-su3.x

clean:
	rm *.x
