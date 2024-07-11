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


template<class T, class SmemLayoutA, class G2SCopyA, class S2RCopyAtomA, class TiledMMA, int m ,int k>
__global__ void gemm_device(const T* Aptr) 
{
  using namespace cute;
  Tensor gA = make_tensor(make_gmem_ptr(Aptr), make_shape(Int<m>{}, Int<k>{}), make_stride(Int<k>{}, Int<1>{}));
  // Shared memory buffers
  __shared__ T smemA[size(SmemLayoutA{})];
  Tensor sA = make_tensor(make_smem_ptr(smemA), SmemLayoutA{});        // (M,K)
  
  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(threadIdx.x);
  Tensor tCrA = thr_mma.partition_fragment_A(gA(_, _));  // (MMA, MMA_M, MMA_K)


  G2SCopyA g2s_tiled_copy_a;
  auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(threadIdx.x);
  const auto tAgA_copy = g2s_thr_copy_a.partition_S(gA);  // (CPY_M, CPY_K)
  auto tAsA_copy = g2s_thr_copy_a.partition_D(sA);  // (CPY_M, CPY_K)
  
  auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
  auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(threadIdx.x);
  cute::copy(G2SCopyA{}, tAgA_copy, tAsA_copy);

  cp_async_fence();
  cp_async_wait<0>();
  __syncthreads();
  
  auto tAsA = s2r_thr_copy_a.partition_S(sA);  // (CPY, CPY_M, CPY_K)
  auto tCrA_view = s2r_thr_copy_a.retile_D(tCrA);  // (CPY, CPY_M, CPY_K)

  cute::copy(S2RCopyAtomA{}, tAsA, tCrA_view);

  cp_async_fence();
  cp_async_wait<0>();
  __syncthreads();

  if(threadIdx.x==0){
    print_tensor(tCrA_view);
  }
}

int main(int argc, char** argv)
{

  using namespace cute;
  using T = half;
  
  static constexpr int M = 16;
  static constexpr int K = 32;
  thrust::host_vector<T> h_A(M*K);
  for (int i = 0; i < h_A.size(); ++i) {
    h_A[i] = static_cast<T>(i);
  }
  thrust::device_vector<T> d_A = h_A;
  const T* Aptr = thrust::raw_pointer_cast(d_A.data());
  
  using mma_op = SM80_16x8x16_F16F16F16F16_TN;
  using mma_traits = MMA_Traits<mma_op>;
  using mma_atom = MMA_Atom<mma_traits>;

  static constexpr int kMmaEURepeatM = 1;
  static constexpr int kMmaEURepeatN = 1;
  static constexpr int kMmaEURepeatK = 1;

  static constexpr int kMmaVRepeatM = 1;
  static constexpr int kMmaVRepeatN = 1;
  static constexpr int kMmaVRepeatK = 1;

  using MMA_EU_RepeatT = decltype(make_layout(make_shape(
      Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})));
  using MMA_V_RepeatT = decltype(make_layout(make_shape(
      Int<kMmaVRepeatM>{}, Int<kMmaVRepeatN>{}, Int<kMmaVRepeatK>{})));
  using TiledMMA =
      decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_V_RepeatT{}));

  using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
  using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
  using g2s_copy_atom = Copy_Atom<g2s_copy_traits, T>;

  using G2SCopyA =
      decltype(make_tiled_copy(g2s_copy_atom{},
                               make_layout(make_shape(Int<8>{}, Int<4>{}),
                                           make_stride(Int<4>{}, Int<1>{})),
                               make_layout(make_shape(Int<1>{}, Int<8>{}))));
  
  // shared memory to register copy
  using s2r_copy_op = SM75_U32x4_LDSM_N;
  using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
  using s2r_copy_atom = Copy_Atom<s2r_copy_traits, T>;

  using S2RCopyAtomA = s2r_copy_atom;
  

  static constexpr int kShmLoadSwizzleM = 3;
  static constexpr int kShmLoadSwizzleS = 3;
  static constexpr int kShmLoadSwizzleB = 3;

  using SmemLayoutAtom = decltype(composition(
      Swizzle<kShmLoadSwizzleB, kShmLoadSwizzleM, kShmLoadSwizzleS>{},
      make_layout(make_shape(Int<8>{}, Int<64>{}),
                  make_stride(Int<64>{}, Int<1>{}))));
  
  // using SmemLayoutAtom = decltype(
  //     make_layout(make_shape(Int<16>{}, Int<32>{}),
  //                 make_stride(Int<32>{}, Int<1>{})));
  using SmemLayoutA = decltype(
      tile_to_shape(SmemLayoutAtom{},
                    make_shape(Int<16>{}, Int<32>{})));
  static_assert(size(TiledMMA{}) == size(G2SCopyA{}));
  dim3 gridDim(1);
  dim3 blockDim(size(TiledMMA{}));

  print(size(G2SCopyA{})); printf("\n");
  gemm_device<T, SmemLayoutA, G2SCopyA, S2RCopyAtomA, TiledMMA, M, K>
              <<<gridDim, blockDim>>>(Aptr);
  cudaDeviceSynchronize();
  
  // print_layout(SmemLayoutA{});

}

