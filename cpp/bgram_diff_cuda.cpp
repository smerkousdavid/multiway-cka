#include <torch/extension.h>
#include <ATen/core/Scalar.h>
#include <c10/core/Scalar.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/grad_mode.h>
#include <ATen/ExpandUtils.h>
#include <ATen/ExpandBase.h>
#include <c10/util/irange.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/TensorOperators.h>
#include <ATen/native/Distance.h>
#include <c10/util/accumulate.h>
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include <utility>
#include <vector>
#include <torch/custom_class.h>
#include <torch/custom_class_detail.h>
#include "bgram_diff.h"
// #include "bind.h"

// std::vector<torch::Tensor> lapl_gram_cuda_backward(
//     torch::Tensor features);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_NONEMPTY(x) TORCH_CHECK(x.is_nonempty(), #x " must be non-empty")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// pulled from https://github.com/pytorch/pytorch/blob/eab3b2637a1bbaea7539238c8fe803f69f3ea5ae/aten/src/ATen/ExpandUtils.cpp
// NOTE: are_expandable did a similar check, please keep them sync if change is needed
template <typename Container, typename ArrayType>
Container infer_size_impl(ArrayType a, ArrayType b) {
  size_t dimsA = a.size();
  size_t dimsB = b.size();
  size_t ndim = dimsA > dimsB ? dimsA : dimsB;
  Container expandedSizes(ndim);

  // Use ptrdiff_t to ensure signed comparison.
  for (ptrdiff_t i = (ptrdiff_t)ndim - 1; i >= 0; --i) {
    ptrdiff_t offset = ndim - 1 - i;
    ptrdiff_t dimA = dimsA - 1 - offset;
    ptrdiff_t dimB = dimsB - 1 - offset;
    auto sizeA = (dimA >= 0) ? a[dimA] : 1;
    auto sizeB = (dimB >= 0) ? b[dimB] : 1;

    TORCH_CHECK(
        sizeA == sizeB || sizeA == 1 || sizeB == 1,
        "The size of tensor a (", sizeA,
        ") must match the size of tensor b (", sizeB,
        ") at non-singleton dimension ", i);

      // 1s map to the other size (even 0).
      expandedSizes[i] = sizeA == 1 ? std::move(sizeB) : std::move(sizeA);
  }

  return expandedSizes;
}

std::vector<int64_t> infer_size(at::IntArrayRef a, at::IntArrayRef b) {
  return infer_size_impl<std::vector<int64_t>>(a, b);
}

std::vector<at::SymInt> infer_size_symint(at::SymIntArrayRef a, at::SymIntArrayRef b) {
  return infer_size_impl<std::vector<at::SymInt>>(a, b);
}


std::vector<torch::Tensor> bgram_dist_based_forward(const torch::Tensor& x1, const torch::Tensor& x2, const torch::Scalar param, const double p) {
  TORCH_CHECK(at::isFloatingType(x1.scalar_type()), "bgram only supports floating-point dtypes, X1 got: ", x1.scalar_type());
  auto device1 = x1.device().type();
  TORCH_CHECK(at::isFloatingType(x2.scalar_type()), "bgram only supports floating-point dtypes, X2 got: ", x2.scalar_type());
  auto device2 = x2.device().type();
  TORCH_CHECK(p == 1 || p == 2, "bgram only supports p=1 (laplacian) and p=2 (rbf) gram matrices");
  TORCH_CHECK(device1 == device2, "X1 and X2 must have the same device type. X1: ", device1, " X2: ", device2);
  TORCH_CHECK(!x1.is_cuda() || x1.get_device() == x2.get_device(), "device of X1 (", x1.get_device(), ") must match device of X2 (", x2.get_device(), ")");
  torch::SymInt c1 = x1.sym_size(-1);
  torch::SymInt c2 = x2.sym_size(-1);
  torch::SymInt r1 = x1.sym_size(-2);
  torch::SymInt r2 = x2.sym_size(-2);

  auto dim1 = x1.dim();
  auto dim2 = x2.dim();

  //For batch calculation we expand all dimensions(except the last two) to one, with size that equals to product of them.
  //The last two dimensions will stay the same
  torch::SymIntArrayRef batch_tensor1(x1.sym_sizes().data(), dim1 - 2);
  torch::SymIntArrayRef batch_tensor2(x2.sym_sizes().data(), dim2 - 2);
  std::vector<at::SymInt> expand_batch_portion = infer_size_symint(batch_tensor1, batch_tensor2);
  std::vector<torch::SymInt> tensor1_expand_size(expand_batch_portion);
  tensor1_expand_size.insert(tensor1_expand_size.end(), {r1, c1});
  std::vector<torch::SymInt> tensor2_expand_size(expand_batch_portion);
  tensor2_expand_size.insert(tensor2_expand_size.end(), {r2, c2});

  const torch::SymInt expand_batch_product = c10::multiply_integers(expand_batch_portion);
  std::vector<torch::SymInt> tensor1_view{expand_batch_product, r1, c1};
  std::vector<torch::SymInt> tensor2_view{expand_batch_product, r2, c2};

  torch::Tensor tensor1_expanded = x1.expand_symint(tensor1_expand_size).contiguous().view_symint(tensor1_view);
  torch::Tensor tensor2_expanded = x2.expand_symint(tensor2_expand_size).contiguous().view_symint(tensor2_view);

  std::vector<torch::SymInt> output_shape(std::move(expand_batch_portion));
  output_shape.insert(output_shape.end(), {r1, r2});

  torch::Tensor result;
  auto batch_size = output_shape.at(0);
  torch::Tensor params = at::zeros_symint({batch_size}, x1.options());
  if (r1 == 0 || r2 == 0 || expand_batch_product == 0) {
    result = at::empty_symint(output_shape, x1.options());
  } else if (c1 == 0) {
    result = at::zeros_symint(output_shape, x1.options());
  } else {
    result = at::empty_symint(output_shape, x1.options());
    bgram_kernel_impl(result, params, tensor1_expanded, tensor2_expanded, param, p);
  }
  return {result, params};
}

std::vector<torch::Tensor> bgram_dist_based_backward(const torch::Tensor& _grad, const torch::Tensor& _x1, const torch::Tensor& _x2, const torch::Tensor& _params, const double p, const torch::Tensor& _kern) {
  // Broadcasting might generate non-contiguous Tensors, so handle it before doing checks
  int64_t c1 = _x1.size(-1);
  int64_t c2 = _x2.size(-1);
  int64_t r1 = _x1.size(-2);
  int64_t r2 = _x2.size(-2);
  auto dim1 = _x1.dim();
  auto dim2 = _x2.dim();
  torch::IntArrayRef batch_tensor1(_x1.sizes().data(), dim1 - 2);
  torch::IntArrayRef batch_tensor2(_x2.sizes().data(), dim2 - 2);
  std::vector<int64_t> expand_batch_portion = at::infer_size(batch_tensor1, batch_tensor2);
  std::vector<int64_t> tensor1_expand_size(expand_batch_portion);
  tensor1_expand_size.insert(tensor1_expand_size.end(), {r1, c1});
  std::vector<int64_t> tensor2_expand_size(expand_batch_portion);
  tensor2_expand_size.insert(tensor2_expand_size.end(), {r2, c2});

  // Compute the linearized batch size
  const int64_t batch_product = c10::multiply_integers(expand_batch_portion);

  // Gracefully handle empty Tensors
  if (r1 == 0 || r2 == 0 || c1 == 0 || batch_product == 0) {
    return {at::zeros_like(_x1, _x1.options()), at::zeros_like(_x2, _x2.options())};
  }

  torch::Tensor x1 = _x1;
  if (tensor1_expand_size != x1.sizes()) {
    x1 = x1.expand(tensor1_expand_size);
  }
  torch::Tensor x2 = _x2;
  if (tensor2_expand_size != x2.sizes()) {
    x2 = x2.expand(tensor2_expand_size);
  }

  x1 = x1.contiguous();
  x2 = x2.contiguous();
  auto params = _params.contiguous();
  auto kern = _kern.contiguous();
  auto grad = _grad.contiguous();
  int64_t n = x1.size(-2);
  int64_t n2 = x2.size(-2);
  int64_t m = x1.size(-1);
  TORCH_CHECK(x1.device().is_cuda(), "bgram_backward only supports CUDA devices, X1 is not CUDA");
  TORCH_CHECK(x2.device().is_cuda(), "bgram_backward only supports CUDA devices, X2 is not CUDA");

  torch::Tensor grad_x1 = at::empty({batch_product, n, m}, x1.options(), LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  torch::Tensor grad_x2 = at::empty({batch_product, n2, m}, x2.options(), LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  bgram_backward_kernel_impl(grad_x1, grad_x2, grad, x1, x2, p, kern, params);

  // Use x1.size() here and not the original size of _x1.size() as this gradient is not taking broadcasting into account
  // Broadcasting will be handled automatically by the autograd engine
  return {grad_x1.view(x1.sizes()), grad_x2.view(x2.sizes())};
}


std::vector<torch::Tensor> bgram_forward(const torch::Tensor &x1, const torch::Tensor &x2, const double param, const int64_t kernel=0) {
  // ensure cuda, contiguous, and non-empty
  CHECK_INPUT(x1);
  CHECK_INPUT(x2);
  TORCH_CHECK(kernel == 1 || kernel == 2, "kernel must be a valid kernel")

  at::NoNamesGuard guard;

  // specify kernel to use
  switch (kernel)
  {
  case 1:
    return bgram_dist_based_forward(x1, x2, param, 1.0);
  case 2:
    return bgram_dist_based_forward(x1, x2, param, 2.0);
  default:
    return std::vector<torch::Tensor>();
  }
}

std::vector<torch::Tensor> bgram_backward(const torch::Tensor& grad, const torch::Tensor &x1, const torch::Tensor &x2, const torch::Tensor& kern, const torch::Tensor& params, const int64_t kernel=0) {
  // ensure cuda, contiguous, and non-empty
  CHECK_INPUT(params);
  CHECK_INPUT(x1);
  CHECK_INPUT(x2);
  CHECK_INPUT(kern);
  TORCH_CHECK(kernel == 1 || kernel == 2, "kernel must be a valid kernel")

  at::NoNamesGuard guard;
  // specify kernel to use
  switch (kernel)
  {
  case 1:
    return bgram_dist_based_backward(grad, x1, x2, params, 1.0, kern);
  case 2:
    return bgram_dist_based_backward(grad, x1, x2, params, 2.0, kern);
  default:
    return std::vector<torch::Tensor>();
  }
}

// std::vector<torch::Tensor> lapl_gram_backward(
//     torch::Tensor features) {
//   CHECK_INPUT(features);

//   return lapl_gram_cuda_backward(
//     features  
//   );
// }


// @TODO move autograd into C++
// using namespace torch::autograd;
// class MulConstant : public Function<MulConstant> {
//  public:
//   static torch::Tensor forward(AutogradContext *ctx, torch::Tensor tensor, double constant) {
//     // ctx is a context object that can be used to stash information
//     // for backward computation
//     ctx->saved_data["constant"] = constant;
//     return tensor * constant;
//   }

//   static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
//     // We return as many input gradients as there were arguments.
//     // Gradients of non-tensor arguments to forward must be `torch::Tensor()`.
//     return {grad_outputs[0] * ctx->saved_data["constant"].toDouble(), torch::Tensor()};
//   }
// };

// using py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &bgram_forward, "Batched Gram forward (CUDA)");
  m.def("backward", &bgram_backward, "Batched Gram backward (CUDA)");

  // will add default binds
  // bind_autograd_function<MulConstant>(m, "MulConstant");
  // torch::class_<MulConstant>("MulConstant
  // torch::python::bind_module<Net>
  // py::class_<MulConstant>(m, "MulConstant")
  //     .def(py::init<>())
  //     .def("backward", &MulConstant::backward);
}