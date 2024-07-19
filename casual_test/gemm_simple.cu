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
#include <cuda_fp16.h>

using namespace cute;
using Element = __half;

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

template<class T, 
        int bM, int bN, int bK,
        class TiledMMA>
__global__ void gemm_device(const T* Aptr, const T* Bptr, T* Cptr, 
                            int m, int n, int k) {
  using namespace cute;
  using TA = float;
  using TB = float;

  Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k), make_stride(k, Int<1>{})); //(m,k) row-major
  Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k), make_stride(k, Int<1>{})); //(n,k) row-major
  Tensor C = make_tensor(make_gmem_ptr(Cptr), make_shape(m, n), make_stride(n, Int<1>{})); //(m,n) row-major

  // Get the appropriate blocks for this thread block
  int ix = blockIdx.x;
  int iy = blockIdx.y;             
  Tensor gA = local_tile(A, make_tile(Int<bM>{}, Int<bK>{}), make_coord(ix, _));  // (b_M,b_K,num_tile_k)
  Tensor gB = local_tile(B, make_tile(Int<bN>{}, Int<bK>{}), make_coord(iy, _));  // (b_N,b_K,num_tile_k)
  Tensor gC = local_tile(C, make_tile(Int<bM>{}, Int<bN>{}), make_coord(ix, iy)); // (b_M,b_N)
  
  TiledMMA tiled_mma;
  ThrMMA thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
  Tensor tAgA = thr_mma.partition_A(gA); // (MMA, MMA_M, MMA_K, num_tile_k)
  Tensor tBgB = thr_mma.partition_B(gB); // (MMA, MMA_N, MMA_K, num_tile_k)
  Tensor tCgC = thr_mma.partition_C(gC); // (MMA, MMA_M, MMA_N)
  

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
}

template<class T>
void gemm(T* A, T* B, T* C, int M, int N, int K) {
  const int bM = 128;
  const int bN = 128;
  const int bK = 32;

  using mma_op = SM80_16x8x16_F16F16F16F16_TN;
  using mma_traits = MMA_Traits<mma_op>;
  using MMA_Atom_Arch = MMA_Atom<mma_traits>;
  // 直接可以简写为下面这句
  // using MMA_Atom_Arch  = MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>;
  static constexpr int kNWarps = 4;
  using TiledMMA = TiledMMA<MMA_Atom_Arch,
                      Layout<Shape<Int<kNWarps>,_1,_1>>,
                      Tile<Int<16 * kNWarps>, _16, _16>>;
  // print(MMA{});
  dim3 dimGrid(size(ceil_div(M, bM)), 
               size(ceil_div(N, bN)));
  dim3 dimBlock(size(TiledMMA{}));

  gemm_device<T, bM, bN, bK, TiledMMA><<<dimGrid, dimBlock, 0, 0>>>(A, B, C,  M, N, K);
}

int main(int argc, char** argv)
{
  
  constexpr int M = 4096;
  constexpr int N = 1024;
  constexpr int K = 512;

  // Define a tensor shape with dynamic extents (m, n)
  // Allocate and initialize
  thrust::host_vector<Element> h_A(M*K);
  thrust::host_vector<Element> h_B(K*N);
  thrust::host_vector<Element> h_C(M*N);
  thrust::host_vector<Element> h_C_ref(M*N);

  for (size_t i = 0; i < h_A.size(); ++i) {
    auto rand_value = rand() % 10 - 5;
    h_A[i] = static_cast<Element>(rand_value);
  }
  for (size_t i = 0; i < h_B.size(); ++i) {
    auto rand_value = rand() % 10 - 5;
    h_B[i] = static_cast<Element>(rand_value);
  }
  for (size_t i = 0; i < h_C.size(); ++i) {
    h_C[i] = static_cast<Element>(0.0f);
    h_C_ref[i] = static_cast<Element>(0.0f);
  }

  thrust::device_vector<Element> d_A = h_A;
  thrust::device_vector<Element> d_B = h_B;
  thrust::device_vector<Element> d_C = h_C;
  thrust::device_vector<Element> d_C_ref = h_C_ref;

  const Element* Aptr = thrust::raw_pointer_cast(d_A.data());
  const Element* Bptr = thrust::raw_pointer_cast(d_B.data());
  Element* Cptr = thrust::raw_pointer_cast(d_C.data());
  gemm(Aptr, Bptr, Cptr, M, N, K);
  cudaDeviceSynchronize();


  // 使用 cublas 库计算
  // Initialize cuBLAS
  cudaSetDevice(0);
  cublasHandle_t handle;
  checkCublasError(cublasCreate(&handle), "cuBLAS initialization failed");
  half alpha = half(1.0f);
  half beta = half(0.0f);
  Element* Cptr_ref = thrust::raw_pointer_cast(d_C_ref.data());
  checkCublasError(cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    N, M, K,
                    &alpha,
                    Bptr, K,
                    Aptr, K,
                    &beta,
                    Cptr_ref, N), "cuBLAS SGEMM failed");

  h_C = d_C;
  h_C_ref = d_C_ref;
  for (int i = 0; i < M*N; i++) {
    if (std::abs(__half2float(h_C[i]) - __half2float(h_C_ref[i])) > 0.01) {
      std::cerr << "Error. h_C[" << i << "]: " << __half2float(h_C[i]) << ",   h_C_ref[" << i << "]: " << __half2float(h_C_ref[i]) << std::endl;
      return -1;
    }
  }
  printf("Success!\n");
  cudaDeviceSynchronize();
  return 0;
}

