#ifndef KERNEL_BIND_H
#define KERNEL_BIND_H
#include <ATen/core/Scalar.h>
#include <c10/core/Scalar.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/grad_mode.h>
#include <torch/detail/static.h>
#include <torch/nn/module.h>
#include <torch/ordered_dict.h>
#include <torch/types.h>

#include <torch/csrc/Device.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/python_tuples.h>


// template <typename ModuleType>
// using PySharedClass =
//     pybind11::class_<ModuleType, std::shared_ptr<ModuleType>>;

// template <typename ModuleType>
// PySharedClass<ModuleType> bind_autograd_function(pybind11::module module, const char* name) {
//   return PySharedClass<ModuleType>(module, name)
//     .def("forward", &ModuleType::forward)
//     .def("__call__", &ModuleType::forward)
//     .def("backward", &ModuleType::backward);
// }

#endif // ndef KERNEL_BIND_H