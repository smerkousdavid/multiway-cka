#ifndef BGRAM_DIFF_H
#define BGRAM_DIFF_H
#include <ATen/core/Scalar.h>
#include <c10/core/Scalar.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/grad_mode.h>

// CUDA declarations
void bgram_kernel_impl(at::Tensor&, at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Scalar&, double);
void bgram_backward_kernel_impl(at::Tensor&, at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const double, const at::Tensor&, const at::Tensor&);

#endif // ndef BGRAM_DIFF_H