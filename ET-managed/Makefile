gpu.x: ET.cc
	pgc++ -I. -w -acc -fast -ta=tesla:cc35,managed --c++11 -Mlarge_arrays -Minfo=accel -O3 ET.cc -o gpu.x
clean:
	rm *.x
