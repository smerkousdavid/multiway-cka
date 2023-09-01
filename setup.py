
from functools import partial
import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

CPP_FOLDER = 'cpp/'
join_cpp = partial(os.path.join, CPP_FOLDER)
join_cpp_list = lambda x: list(map(join_cpp, x))

setup(
  name='torch_kernel',
  ext_modules=[
    CUDAExtension('bgram_cuda',
      join_cpp_list([
        'bgram_diff_cuda.cpp',
        'bgram_diff_cuda_kernel.cu',
      ]),
      extra_compile_args={
          'cxx': ['-O3'],
          'nvcc': ['-O3']
      }
    )
  ],
  cmdclass={
    'build_ext': BuildExtension
  }
)