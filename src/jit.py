""" For testing/compat use pytorch's JIT functions """
from functools import partial
from torch.utils.cpp_extension import load
import os

CPATH = os.path.dirname(os.path.abspath(__file__))
CPP_FOLDER = os.path.join(CPATH, '..', 'cpp')
join_cpp = partial(os.path.join, CPP_FOLDER)
join_cpp_list = lambda x: list(map(join_cpp, x))

bgram_cuda = load('bgram_cuda', join_cpp_list(['bgram_diff_cuda.cpp', 'bgram_diff_cuda_kernel.cu']), verbose=True)