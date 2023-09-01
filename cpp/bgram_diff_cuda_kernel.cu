#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/core/Scalar.h>
#include <ATen/core/TensorBase.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cuda/DeviceUtils.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/core/Scalar.h>
#include <c10/cuda/CUDAMathCompat.h>

#include <ATen/native/cuda/block_reduce.cuh>
#include <ATen/native/cuda/DeviceSqrt.cuh>
#include <ATen/native/Distance.h>
#include <ATen/native/Fill.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/sum.h>
#endif

#include <c10/macros/Macros.h>
#include <type_traits>
#include <iostream>
#include <vector>

#include "bgram_diff.h"

constexpr int kCUDANumThreads = 256;

template <typename scalar_t>
struct grams {

  static __forceinline__ __device__ scalar_t sign(scalar_t val) {
    return (0 < val) - (val < 0);
  }

  static __forceinline__ __device__ scalar_t neg_sign(scalar_t val) {
    return  (val < 0) - (0 < val);
  }

  // @NOTE: the forward pass/backward pass are very different
  // during forward pass we simply do L1/L2, since we require median value of distance to estimate gamma/sigma
  // when doing backward we already know gamma/sigma so we can fuse operations together

  // Laplacian Gram
  struct lapl_op {
    static __forceinline__ __device__ void inc(scalar_t& agg, const scalar_t diff, const scalar_t /*p*/) { agg += diff; }
    static __forceinline__ __device__ scalar_t finish(const scalar_t agg, const scalar_t /*p*/) { return agg; }
    static __forceinline__ __device__ scalar_t dist_apply(scalar_t& dist, const scalar_t param, const scalar_t /*p*/) {
      dist = exp(param * dist); 
    }
    static __forceinline__ __device__ void agg(scalar_t& update, const scalar_t other) { update += other; }
    static __forceinline__ __device__ scalar_t backward_agg(const scalar_t param, const scalar_t kern) {
      return param * kern;  // -1/b e^(-1/b * |x|) if x > 0 else 1/b e^(-1/b * |x|) gradient of laplacian
    }
    static __forceinline__ __device__ scalar_t backward(const scalar_t diff, const scalar_t grad, const scalar_t back_agg, const scalar_t /* kern */) {
      return grad * sign(diff) * back_agg;  // includes backward aggregate grad
    }
  };

  // Radial Basis (RBF) Gram @TODO
  struct rbf_op {
    static __forceinline__ __device__ void inc(scalar_t& agg, const scalar_t diff, const scalar_t /*p*/) { agg += diff * diff; }
    static __forceinline__ __device__ scalar_t finish(const scalar_t agg, const scalar_t /*p*/) { return at::native::device_sqrt<scalar_t>(agg); }
    static __forceinline__ __device__ scalar_t dist_apply(scalar_t& dist, const scalar_t param, const scalar_t /*p*/) {
      dist = exp(param * dist); 
    }
    static __forceinline__ __device__ void agg(scalar_t& update, const scalar_t other) { update += other; }
    static __forceinline__ __device__ scalar_t backward_agg(const scalar_t param, const scalar_t kern) {
      return param * kern;  // -1/b e^(-1/b * |x|) if x > 0 else 1/b e^(-1/b * |x|) gradient of laplacian
    }
    static __forceinline__ __device__ scalar_t backward(const scalar_t diff, const scalar_t grad, const scalar_t param, const scalar_t kern) {
      return 0.0;  // dist == 0.0 ? 0 : grad * diff; // / dist;
    }
  };
};

template <typename scalar_t, typename F>
struct DistReduceOp {
  __forceinline__ __device__ scalar_t combine(scalar_t a, scalar_t b) const {
    F::agg(a, b);
    return a;
  }

  __forceinline__ __device__ scalar_t warp_shfl_down(scalar_t data, int offset) const {
    return WARP_SHFL_DOWN(data, offset);
  }
};


template <typename scalar_t, typename F>
__global__ static void bgram_backward_kernel_cuda_impl(scalar_t * buffer, const scalar_t * grad, const scalar_t * x1, const scalar_t * x2, const scalar_t * kern, const scalar_t * params,
                                                       const scalar_t p, const int64_t r1, const int64_t r2, const int64_t m, const int64_t count, const int64_t r_size, const int64_t l1_size, const int64_t l2_size) {
  const int y = (blockIdx.y * gridDim.z + blockIdx.z) * blockDim.y + threadIdx.y;
  const int init = blockIdx.x * blockDim.x + threadIdx.x;
  if (y >= count || init >= m) {
    return;
  }
  const int l = y / r_size;
  const int k = y % r_size;
  const int stride = blockDim.x * gridDim.x;
  const int l_size = r_size * m;

  int64_t i = k / r2;
  int64_t j = k % r2;

  const scalar_t grad_k = grad[y];
  const scalar_t kern_k = kern[y];
  const scalar_t param_l = params[l];  // batch corrected parameter
  const scalar_t kern_agg_k = F::backward_agg(param_l, kern_k);

  const scalar_t * const start = x1 + l * l1_size + i * m;
  const scalar_t * const end = start + m;
  const scalar_t * self_i = start + init;
  const scalar_t * self_j = x2 + l * l2_size + j * m + init;

  scalar_t * buff_i = buffer + l * l_size + (r1 * j + i) * m + init;

  for (; self_i < end; self_i += stride, self_j += stride, buff_i += stride) {
    const scalar_t res = F::backward(*self_i - *self_j, grad_k, kern_agg_k, kern_k);
    *buff_i = res;
  }
}


template <typename scalar_t, typename F>
__global__ static void bgram_kernel_cuda_impl(scalar_t * result, const scalar_t * x1, const scalar_t * x2,
    const scalar_t p, const int64_t r2, const int64_t m, const int64_t r_size, const int64_t l1_size, const int64_t l2_size) {
  
  // dimensions:
  //   * note r_size is number of elements in batched dim (so rows of x1 times rows of x2)
  //   * l1_size is number of elements in x1 (similar for l2 size)
  const int64_t l = blockIdx.x / r_size;  // l indicates batch number
  const int64_t k = blockIdx.x % r_size;  //  
  const int64_t i = k / r2;
  const int64_t j = k % r2;
  const int stride = blockDim.x;

  const scalar_t * const start = x1 + l * l1_size + i * m;
  const scalar_t * const end = start + m;
  const scalar_t * a = start + threadIdx.x;
  const scalar_t * b = x2 + l * l2_size + j * m + threadIdx.x;

  scalar_t agg = 0.0;
  for (; a < end; a += stride, b += stride) {
    F::inc(agg, std::abs(*a - *b), p);
  }
  __shared__ scalar_t agg_smem[kCUDANumThreads];
  scalar_t agg_init{0.0};
  agg = at::native::cuda_utils::BlockReduce(agg, DistReduceOp<scalar_t, F>{}, agg_init, agg_smem);
  if (threadIdx.x == 0) {
    result[blockIdx.x] = F::finish(agg, p);
  }
}


template <typename scalar_t, typename F>
__global__ static void bgram_elem_kernel_cuda_impl(scalar_t * dist, const scalar_t * params,
    const scalar_t p, const int64_t numel, const int64_t r_size) {
  
  // common grid-stride approach
  const int64_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
  const int64_t stride = blockDim.x * gridDim.x;

  for (int64_t i = index; i < numel; i += stride) {
    const int64_t l = i / r_size;  // batch index for parameter adjustment
    F::dist_apply(dist[i], params[l], p);  // apply final kernel transformation
  }
}


void bgram_backward_kernel_impl(at::Tensor& grad_x1, at::Tensor& grad_x2, const at::Tensor& grad, const at::Tensor& x1, const at::Tensor& x2, const double p, const at::Tensor& kern, const at::Tensor& params) {
  if (p == 0.0 || grad.numel() == 0 || x1.numel() == 0 || x2.numel() == 0) {
    grad_x1.fill_(0);
    grad_x2.fill_(0);
    return;
  }

  const int64_t r1 = x1.size(-2);
  const int64_t r2 = x2.size(-2);
  const int64_t m = x1.size(-1);  // shared feature dimension
  // Just like we do in the CPU code, assume that result is always batched
  int64_t batch = grad_x1.size(0);
  const int block_x = 64;
  const int block_y = 16;
  const int grid_x = (m + block_x * 8 - 1) / (block_x * 8);

  const int64_t count = kern.numel();
  const int64_t grid_temp = (count + block_y - 1) / block_y;

  const int grid_y = (grid_temp - 1) / 65535 + 1;
  const int grid_z = (grid_temp - 1) / grid_y + 1;

  const dim3 grid(grid_x, grid_y, grid_z);
  const dim3 block(block_x, block_y);

  const int64_t r_size = r1 * r2;  // size of result there will be r1 rows in x1 and r2 rows in x2 so output is r1 * r2
  const int64_t l1_size = r1 * m;  // length of x1
  const int64_t l2_size = r2 * m;  // lenght of x2
  //current implementation supports only gradient that can be collapsed to 1D. However, to avoid checking this assumption,
  //we call grad.contiguous() before backward, so stride is guaranteed to be 1

  at::Tensor buffer = at::empty({batch, r2, r1, m}, grad_x1.options());
  AT_DISPATCH_FLOATING_TYPES(grad_x1.scalar_type(), "bgram_cuda_backward", [&] {
    auto impl_fptr = bgram_backward_kernel_cuda_impl<scalar_t, grams<scalar_t>::lapl_op>;
    if (p == 2.0) {
       impl_fptr = bgram_backward_kernel_cuda_impl<scalar_t, grams<scalar_t>::rbf_op>;
    }
    impl_fptr<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(buffer.data_ptr<scalar_t>(),
      grad.data_ptr<scalar_t>(), x1.data_ptr<scalar_t>(), x2.data_ptr<scalar_t>(), kern.data_ptr<scalar_t>(), params.data_ptr<scalar_t>(),
      p, r1, r2, m, count, r_size, l1_size, l2_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });

  // gradient for elements of x1
  at::sum_out(grad_x1, buffer, 1);

  // gradients for elements of x2
  at::sum_out(grad_x2, buffer, 2);
  at::negative_(grad_x2);
}


// @TODO remove when verified blocked cuda stream working properly
/* void bgram_elem_kernel_impl(at::Tensor& result, const at::Tensor& params, const double p) {
  const int64_t r1 = result.size(-2);
  const int64_t r2 = result.size(-1);
  const int64_t numel = result.numel();
  const int64_t r_size = r1 * r2;  // size of pairwise distance matrix
  const dim3 block(kCUDANumThreads);
  const int64_t grid_x = (numel + kCUDANumThreads - 1) / kCUDANumThreads;
  const dim3 grid(grid_x);

  AT_DISPATCH_FLOATING_TYPES(result.scalar_type(), "bgram_elem_cuda", [&] {
    auto impl_fptr = bgram_elem_kernel_cuda_impl<scalar_t, grams<scalar_t>::lapl_op>;
    if (p == 2.0) {
      impl_fptr = bgram_elem_kernel_cuda_impl<scalar_t, grams<scalar_t>::rbf_op>;
    }
    impl_fptr<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(result.data_ptr<scalar_t>(), params.data_ptr<scalar_t>(), p, numel, r_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
} */


void bgram_kernel_impl(at::Tensor& result, at::Tensor& params, const at::Tensor& x1, const at::Tensor& x2, const at::Scalar& param, const double p) {
  const int64_t r1 = x1.size(-2);
  const int64_t r2 = x2.size(-2);
  const int64_t m = x1.size(-1);
  const int64_t r_size = r1 * r2;
  const int64_t l1_size = r1 * m;
  const int64_t l2_size = r2 * m;
  const int64_t batch = result.size(0);
  const dim3 grid(result.numel());
  const dim3 block(kCUDANumThreads);
  TORCH_CHECK(batch == params.numel(), "bgram forward params length must be equal to the batch size");

  AT_DISPATCH_FLOATING_TYPES(x1.scalar_type(), "bgram_cuda", [&] {
    auto cuda_stream = at::cuda::getCurrentCUDAStream();
    auto impl_fptr = bgram_kernel_cuda_impl<scalar_t, grams<scalar_t>::lapl_op>;
    if (p == 2.0) {
      impl_fptr = bgram_kernel_cuda_impl<scalar_t, grams<scalar_t>::rbf_op>;
    }
    impl_fptr<<<grid, block, 0, cuda_stream>>>(result.data_ptr<scalar_t>(), x1.data_ptr<scalar_t>(), x2.data_ptr<scalar_t>(), p, r2, m, r_size, l1_size, l2_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // calculate median distances across the batch
    // depending on size of matrix median could be zero which... is not great
    // diag do not accurately contribute to median so we evaluate non-diag values
    // to do this we inplace fill diagonal with nan calculate nan-median then refill with zeros
    for (int64_t i = 0; i < batch; i++) {
      result[i].fill_diagonal_(std::numeric_limits<scalar_t>::quiet_NaN());
    }
    
    // calculate accurate median
    const at::Tensor medians = at::zeros({batch}, x1.options()); // std::get<0>(at::nanmedian(result.view({batch, r_size}), 1));
    
    const scalar_t diag = 0.0;
    const scalar_t init_param = 1.0;
    for (int64_t i = 0; i < batch; i++) {
      result[i].fill_diagonal_(diag); // refill with zeros since diag distance is always zero
    }

    params.fill_(init_param);  // initialize parameters
    medians.multiply_(param);  // scale median
    // params = torch::ones_like(medians);

    // precompute adjustment parameter depending on implementation
    if (p == 1.0) {
      medians.neg_();

      params = at::divide(params, medians);
    } else if (p == 2.0) {
      // @TODO
    }

    // @TODO handle zero median

    // now we have computed the parameters for the exp we can apply a kernel to update the kernel matrix
    const int64_t numel = result.numel();
    const dim3 blockelem(kCUDANumThreads);
    const int64_t grid_x = (numel + kCUDANumThreads - 1) / kCUDANumThreads;
    const dim3 gridelem(grid_x);
    auto impl_fptr_elem = bgram_elem_kernel_cuda_impl<scalar_t, grams<scalar_t>::lapl_op>;
    if (p == 2.0) {
      impl_fptr_elem = bgram_elem_kernel_cuda_impl<scalar_t, grams<scalar_t>::rbf_op>;
    }
    impl_fptr_elem<<<gridelem, blockelem, 0, cuda_stream>>>(result.data_ptr<scalar_t>(), params.data_ptr<scalar_t>(), p, numel, r_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
}