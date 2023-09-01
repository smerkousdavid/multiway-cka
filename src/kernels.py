""" Optimized kernel computation using CUDA """
import torch
from torch.autograd import Function
from .kernels_py import laplacian_gram, rbf_gram, linear_gram

# valid kernel list
AVAIL_CUDA_KERNELS = {
   'laplacian': 1,
   'rbf': 2
}
AVAIL_PY_KERNELS = {
  'linear': linear_gram,
  'laplacian': laplacian_gram,
  'rbf': rbf_gram
}

# attempt to import CUDA kernels
# if it fails then use backup functions
try:
  import bgram_cuda
  CUDA_KERNELS = True

  if not torch.cuda.is_available():
    CUDA_KERNELS = False
    raise RuntimeError("CUDA is not available")
except (ImportError, RuntimeError) as err:
  CUDA_KERNELS = False

  print('WARNING! pytorch kernels missing CUDA extensions. Please make sure you have installed/compiled this library correctly!')
  print('If you do not have a CUDA GPU and/or missing the compiled extensions then note the python gram functions used will be slower')
  print(f'Error: {str(err)}')


# define autograd functions for c++ functions if available
if CUDA_KERNELS:
  class BatchedGram(Function):
    @staticmethod
    def forward(ctx, x1, x2, param, kernel):
        kern_num = AVAIL_CUDA_KERNELS[kernel]
        gram, params = bgram_cuda.forward(x1, x2, param, kern_num)
        ctx.save_for_backward(gram, x1, x2, params, torch.tensor([kern_num], dtype=x1.dtype, device=x1.device) )
        return gram

    @staticmethod
    def backward(ctx, grad):
        gram, x1, x2, params, kern_num = ctx.saved_variables
        kern_num = int(kern_num.cpu().item())
        outputs = bgram_cuda.backward(grad.contiguous(), x1, x2, gram, params, kern_num)
        return outputs[0], outputs[1], None, None


def bgram_matrix(X1: torch.Tensor, X2: torch.Tensor, kernel='linear', detach_diag: bool=False, force_py_kernel: bool=False, **kwargs):
  """ Creates a gram matrix using the specified kernel, detach_diag is useful when you don't care about gradients between self-features """
  
  # call python version (supports any device) or CUDA kernel not defined for specified device
  if force_py_kernel or (not CUDA_KERNELS) or (kernel not in AVAIL_CUDA_KERNELS):
    func = AVAIL_PY_KERNELS[kernel]
    return func(X1, X2, detach_diag=detach_diag, **kwargs)

  assert 'param' in kwargs, 'Currently CUDA kernel requires parameter'

  # call CUDA optimized version otherwise
  return BatchedGram.apply(X1, X2, kwargs['param'], kernel)
