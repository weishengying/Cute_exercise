#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"

#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <random>

void checkCublasError(cublasStatus_t status, const char* msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << msg << ": ";
        switch (status) {
            case CUBLAS_STATUS_NOT_INITIALIZED:
                std::cerr << "CUBLAS_STATUS_NOT_INITIALIZED";
                break;
            case CUBLAS_STATUS_ALLOC_FAILED:
                std::cerr << "CUBLAS_STATUS_ALLOC_FAILED";
                break;
            case CUBLAS_STATUS_INVALID_VALUE:
                std::cerr << "CUBLAS_STATUS_INVALID_VALUE";
                break;
            case CUBLAS_STATUS_ARCH_MISMATCH:
                std::cerr << "CUBLAS_STATUS_ARCH_MISMATCH";
                break;
            case CUBLAS_STATUS_MAPPING_ERROR:
                std::cerr << "CUBLAS_STATUS_MAPPING_ERROR";
                break;
            case CUBLAS_STATUS_EXECUTION_FAILED:
                std::cerr << "CUBLAS_STATUS_EXECUTION_FAILED";
                break;
            case CUBLAS_STATUS_INTERNAL_ERROR:
                std::cerr << "CUBLAS_STATUS_INTERNAL_ERROR";
                break;
            default:
                std::cerr << "Unknown error";
                break;
        }
        std::cerr << std::endl;
        exit(EXIT_FAILURE);
    }
}

template<class T, int bM, int bN, int bK,
        class TiledMMA>
__global__ void gemm_device(const T* Aptr, const T* Bptr, T* Cptr, 
                            int m, int n, int k,
                            TiledMMA tiled_mma) {
  using namespace cute;
  using TA = float;
  using TB = float;

  Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k), make_stride(Int<1>{}, m)); //(m,k) col-major
  Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k), make_stride(k, Int<1>{})); //(k,n) row-major
  Tensor C = make_tensor(make_gmem_ptr(Cptr), make_shape(m, n), make_stride(Int<1>{}, m)); //(m,n) col-major

  // Get the appropriate blocks for this thread block
  int ix = blockIdx.x;
  int iy = blockIdx.y;             
  Tensor gA = local_tile(A, make_tile(Int<bM>{}, Int<bK>{}), make_coord(ix, _));  // (b_M,b_K,num_tile_k)
  Tensor gB = local_tile(B, make_tile(Int<bN>{}, Int<bK>{}), make_coord(iy, _));  // (b_N,b_K,num_tile_k)
  Tensor gC = local_tile(C, make_tile(Int<bM>{}, Int<bN>{}), make_coord(ix, iy)); // (b_M,b_N)
  
  // if(ix == 0 && iy==0 && threadIdx.x==0) {
  //   print(gA); printf("\n");
  //   print(gB); printf("\n");
  //   print(gC); printf("\n");
  // }

  ThrMMA thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
  Tensor tAgA = thr_mma.partition_A(gA); // (MMA, MMA_M, MMA_K, num_tile_k)
  Tensor tBgB = thr_mma.partition_B(gB); // (MMA, MMA_N, MMA_K, num_tile_k)
  Tensor tCgC = thr_mma.partition_C(gC); // (MMA, MMA_M, MMA_N)
  
  // if(ix == 0 && iy==0 && threadIdx.x==0) {
  //   print(tAgA); printf("\n");
  //   print(tBgB); printf("\n");
  //   print(tCgC); printf("\n");
  // }

  auto tArA = thr_mma.partition_fragment_A(gA(_, _, 0));  // (MMA, MMA_M, MMA_K)
  auto tBrB = thr_mma.partition_fragment_B(gB(_, _, 0));  // (MMA, MMA_K, MMA_N)
  auto tCrC = thr_mma.partition_fragment_C(gC(_, _));     // (MMA, MMA_M, MMA_N)

  clear(tCrC); 
  int num_tile_k = size<2>(gA);

  #pragma unroll 1
  for(int itile = 0; itile < num_tile_k; ++itile) {
    cute::copy(tAgA(_, _, _, itile), tArA);
    cute::copy(tBgB(_, _, _, itile), tBrB);

    cute::gemm(tiled_mma, tCrC, tArA, tBrB, tCrC);
  }

  cute::copy(tCrC, tCgC); 
//   gemm(tiled_mma, ta, tb, tc);
}


int main(int argc, char** argv)
{

  using namespace cute;
  using Element = float;


  // Use a 1x1x1 FMA on the types TC += TA * TB. Each atom requires a single thread.
  // Reproduce that atom 16x16x1 times (m-major) across threads so that we use 256 threads.
  using TA = Element;
  using TB = Element;
  using TC = Element;
  
  static constexpr int kNWarps = 4;

  using mma_op = SM80_16x8x16_F16F16F16F16_TN;
  using mma_traits = MMA_Traits<mma_op>;
  using MMA_Atom_Arch  = MMA_Atom<mma_traits>;
  using TiledMma = TiledMMA<
                        MMA_Atom_Arch,
                        Layout<Shape<Int<kNWarps>,_1,_1>>,
                        Tile<Int<16 * kNWarps>, _16, _32>>;
make_tiled_mma
  TiledMMA mmaC = TiledMma{};
  print(size(mmaC)); printf("\n");
  print_latex(mmaC);
  // TiledMMA mmaC = make_tiled_mma(UniversalFMA<TC,TA,TB>{},
  //                                Layout<Shape<_4,_4,_4>>{});  // 16x16x1 TiledMMA
//   ThrMMA thr_mma = mmaC.get_thread_slice(0);
}

