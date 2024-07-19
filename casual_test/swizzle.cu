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


template<class T, class SmemLayoutA, class G2SCopyA>
__global__ void gemm_device(const T* Aptr, int m, int k) 
{
  using namespace cute;
  Tensor gA = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k), make_stride(k, Int<1>{}));
  // Shared memory buffers
  __shared__ T smemA[size(SmemLayoutA{})];
  Tensor sA = make_tensor(make_smem_ptr(smemA), SmemLayoutA{});        // (BLK_M,BLK_K)

  G2SCopyA g2s_tiled_copy_a;
  auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(threadIdx.x);
  const auto tAgA_copy = g2s_thr_copy_a.partition_S(gA);  // (CPY_M, CPY_K)
  auto tAsA_copy = g2s_thr_copy_a.partition_D(sA);  // (CPY_M, CPY_K)
  cute::copy(G2SCopyA{}, tAgA_copy, tAsA_copy);

  cp_async_fence();
  cp_async_wait<0>();
  __syncthreads();
  if(threadIdx.x==0){
    print(gA); printf("\n");
    for (int i = 0; i < sA.size(); i++) {
        if(i % k == 0)
            printf("\n\n");
        printf("%f ", __half2float(sA.data()[i]));
    }
    printf("\n");
  }
}

int main(int argc, char** argv)
{

  using namespace cute;
  using T = half;
  
  int M = 8;
  int K = 32;
  thrust::host_vector<T> h_A(M*K);
  for (int i = 0; i < h_A.size(); ++i) {
    h_A[i] = static_cast<T>(i);
  }
  thrust::device_vector<T> d_A = h_A;
  const T* Aptr = thrust::raw_pointer_cast(d_A.data());

  using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
  using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
  using g2s_copy_atom = Copy_Atom<g2s_copy_traits, T>;

  using G2SCopyA =
      decltype(make_tiled_copy(g2s_copy_atom{},
                               make_layout(make_shape(Int<8>{}, Int<4>{}),
                                           make_stride(Int<4>{}, Int<1>{})),
                               make_layout(make_shape(Int<1>{}, Int<8>{}))));
  
  static constexpr int kShmLoadSwizzleM = 3;
  static constexpr int kShmLoadSwizzleS = 3;
  static constexpr int kShmLoadSwizzleB = 3;
//   使用 Swizzle 语义的 smem layout
  using SmemLayoutAtom_swillze = decltype(composition(
      Swizzle<kShmLoadSwizzleB, kShmLoadSwizzleM, kShmLoadSwizzleS>{},
      make_layout(make_shape(Int<16>{}, Int<32>{}),
                  make_stride(Int<32>{}, Int<1>{}))));
  
//   这是未使用 Swizzle 语义的 smem layout
  using SmemLayoutAtom = decltype(
      make_layout(make_shape(Int<16>{}, Int<32>{}),
                  make_stride(Int<32>{}, Int<1>{})));
  
  print_layout(SmemLayoutAtom{});
  print_layout(SmemLayoutAtom_swillze{});
//   using SmemLayoutA = decltype(
//       tile_to_shape(SmemLayoutAtom{},
//                     make_shape(Int<8>{}, Int<32>{})));
//   dim3 gridDim(1);
//   dim3 blockDim(size(G2SCopyA{}));

//   print(size(G2SCopyA{})); printf("\n");
//   gemm_device<T, SmemLayoutA, G2SCopyA>
//               <<<gridDim, blockDim>>>(Aptr, M, K);
//   cudaDeviceSynchronize();
}

