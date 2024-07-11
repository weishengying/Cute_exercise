#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"


/*
nvcc tiled_copy.cu -arch=sm_80 -I ./thirdparty/cutlass/include -I ./thirdparty/cutlass/tools/util/include --expt-relaxed-constexpr

*/
int main(int argc, char** argv)
{
  using namespace cute;
  using Element = float;

  Layout thr_layout = make_layout(make_shape(Int<16>{}, Int<8>{}), make_stride(Int<8>{}, Int<1>{}));
  Layout vec_layout = make_layout(make_shape(Int<1>{}, Int<4>{}));
//   using AccessType = Element;
  using AccessType = cutlass::AlignedArray<Element, 4>;
  using Atom = Copy_Atom<UniversalCopy<AccessType>, Element>;

  auto tiled_copy =
    make_tiled_copy(
      Atom{},                       // access size
      thr_layout,                  // thread layout
      vec_layout);                 // vector layout (e.g. 4x1)
    
  print(tiled_copy);
  auto tensor_shape = make_shape(32, 32);
  auto tensor_stride = make_stride(Int<32>{}, Int<1>{});

  // Allocate and initialize
  thrust::host_vector<Element> h_S(size(tensor_shape));

  for (size_t i = 0; i < h_S.size(); ++i) {
    h_S[i] = static_cast<Element>(i);
  }

  Tensor tensor_S = make_tensor(h_S.data(), make_layout(tensor_shape, tensor_stride));
  
  auto thr_copy = tiled_copy.get_thread_slice(0);

  Tensor thr_tile_S = thr_copy.partition_S(tensor_S);             // (CopyOp, CopyM, CopyN)
//   print_tensor(thr_tile_S);
  return 0;
}

