default:
	nvcc gemm.cu -O2 -arch=sm_80 -std=c++17  -I ../ -I ../../thirdparty/cutlass/include -I ../../thirdparty/cutlass/tools/util/include --expt-relaxed-constexpr -lcublas
test:
	nvcc Epilogue_test.cu -O2 -o test -arch=sm_80 -std=c++17  -I ../ -I ../../thirdparty/cutlass/include -I ../../thirdparty/cutlass/tools/util/include --expt-relaxed-constexpr