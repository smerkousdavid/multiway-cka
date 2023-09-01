from __future__ import division
from __future__ import print_function

import argparse
import math
import time

import torch

TIME_SCALES = {'s': 1, 'ms': 1000, 'us': 1000000}

parser = argparse.ArgumentParser()
parser.add_argument('example', choices=['py', 'cuda'])
parser.add_argument('-b', '--batch-size', type=int, default=3)
parser.add_argument('-l', '--row-size', type=int, default=400)
parser.add_argument('-f', '--features', type=int, default=6000)
parser.add_argument('-r', '--runs', type=int, default=100)
parser.add_argument('--scale', choices=['s', 'ms', 'us'], default='ms')
parser.add_argument('-c', '--cuda', action='store_true')
parser.add_argument('-d', '--double', action='store_true')
options = parser.parse_args()

from src.kernels import bgram_matrix
laplacian = lambda X : bgram_matrix(X, X, kernel='laplacian', force_py_kernel=(options.example == 'py'), param=1.0)
# lambda X: torch.bmm(X, torch.transpose(X, 1, 2))  # 

device = torch.device("cuda") if options.cuda else torch.device("cpu")
dtype = torch.float64 if options.double else torch.float32

kwargs = {'dtype': dtype,
          'device': device,
          'requires_grad': True}
X = torch.randn(options.batch_size, options.row_size, options.features, **kwargs)
print('Feature matrix size', X.shape)

# Force CUDA initialization
laplacian(X)

forward_min = math.inf
forward_time = 0
backward_min = math.inf
backward_time = 0
for _ in range(options.runs):
    if X.grad is not None:
      X.grad.zero_()

    start = time.time()
    out = laplacian(X)
    elapsed = time.time() - start
    forward_min = min(forward_min, elapsed)
    forward_time += elapsed

    start = time.time()
    torch.sum(out).backward()
    elapsed = time.time() - start
    backward_min = min(backward_min, elapsed)
    backward_time += elapsed

scale = TIME_SCALES[options.scale]
forward_min *= scale
backward_min *= scale
forward_average = forward_time / options.runs * scale
backward_average = backward_time / options.runs * scale

print('Forward: {0:.3f}/{1:.3f} {4} | Backward {2:.3f}/{3:.3f} {4}'.format(
    forward_min, forward_average, backward_min, backward_average,
    options.scale))