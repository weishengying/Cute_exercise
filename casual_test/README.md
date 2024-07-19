# Cute_exercise
Cute_exercise

demo:
```bash
nvcc vectorized_copy_1.cu -arch=sm_80 -I ./thirdparty/cutlass/include -I ./thirdparty/cutlass/tools/util/include --expt-relaxed-constexpr


nvcc -ptx vectorized_copy_1.cu -o cute_copy_1.ptx -arch=sm_80 -I ./thirdparty/cutlass/include -I ./thirdparty/cutlass/tools/util/include --expt-relaxed-constexpr
```

