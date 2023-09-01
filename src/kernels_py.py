""" Kernel functions and gram matrices written in pytorch/python (not CUDA optimized) """
import torch


def linear_gram(X1: torch.Tensor, X2: torch.Tensor, detach_diag: bool=False) -> torch.Tensor:
  """ Computes the linear kernel Gram matrix with examples (Mercer/PSD)  

  Args:
      X (torch.Tensor): input

  Returns:
    torch.Tensor: the linear Gram matrix that is n by n
  """
  gram = torch.bmm(X1, torch.transpose(X2, 2, 1))

  if detach_diag:
    gram_diag = torch.diagonal(gram).clone().detach()
    diag = torch.ones_like(gram)
    diag.fill_diagonal_(0.0)
    return gram.multiply(diag) + gram_diag   # zero out current diagonal and add diag back in (not sure of gradients with fill_diagonal_)
  return gram


def rbf_gram(X: torch.Tensor, eps: float=1e-8, detach_diag: bool=False) -> torch.Tensor:
  """ Computes the RBF kernel Gram matrix with examples (Mercer/PSD)  

  Args:
      X (torch.Tensor): input
      eps (torch.Tensor): smoothing constant to prevent zero div

  Returns:
    torch.Tensor: the RBF Gram matrix that is n by n
  """

  # note that ||x-y||^2 = (||x||^2  + ||y||^2) (< norms) - 2 * x^T * y  (< matrix mult here)
  dots = torch.mm(X, X.t())
  sq_norms = torch.diag(dots)
  sq_distances = -2.0 * dots + sq_norms[:, None] + sq_norms[None, :]
  sq_median_distance = torch.median(sq_distances)

  # detach/mask in a non-inplace way by zeroing out diag then adding it back
  if detach_diag:
    diag = torch.eye(sq_distances.shape[0], device=sq_distances.device, dtype=sq_distances.dtype)
    fsq_distances = sq_distances.multiply(1.0 - diag)  # zero out diagonal
  else:
    fsq_distances = sq_distances

  return torch.exp(-fsq_distances / ((2.0 * sq_median_distance) + eps))


def laplacian_gram(X1: torch.Tensor, X2: torch.Tensor, param: float=2.0, detach_diag: bool=False):
  """ Calculates laplacian gram matrix from Nxp matrix """
  # p1_dist = torch.concatenate([torch.pdist(X1[i], p=1) for i in range(X1.shape[0])])
  # return p1_dist
  p1_dist = torch.cdist(X1, X2, p=1)

  # detach in a non-inplace way by zeroing out diag then adding it back
  if detach_diag:
    diag = torch.eye(p1_dist.shape[0], device=p1_dist.device, dtype=p1_dist.dtype)
    fp1_dist = p1_dist.multiply(1.0 - diag)  # zero out diagonal
  else:
    fp1_dist = p1_dist
  
  # get median distance for gamma (terribly slow method so switch to c++ variant if possible)
  # see cuda.cpp for description of this method:
  fp_gammas = []
  for i in range(X1.shape[0]):
    dist = fp1_dist[i].clone().detach()
    dist.fill_diagonal_(torch.nan)
    gamma_m = (torch.nanmedian(dist) * param).detach().item()
    fp_gammas.append(torch.exp(-fp1_dist[i] / gamma_m).unsqueeze(0))

  # return laplacian gram matrix
  return torch.concatenate(fp_gammas)