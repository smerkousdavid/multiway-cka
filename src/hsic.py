""" Development of a differentiable multiway HSIC """
import torch

# from .kernels import gram_matrix
from .kernels import bgram_matrix


""" @TODO if this is useful create autograd function/optimize implementations """


def unscaled_hsic(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
  """ Compute the unscaled version of HSIC (to scale use 1/((m-1)**2))
  Assumed X and Y are already centered

  Args:
      X (torch.Tensor): input gram X
      Y (torch.Tensor): input gram Y

  Returns:
      torch.Tensor: unscaled HSIC
  """

  # left is faster/fewer operations
  return torch.dot(X.ravel(), Y.ravel())


def hsic(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
  """ Compute the scaled version of HSIC (to scale use 1/((m-1)**2))
  Assumed X and Y are already centered

  Args:
      X (torch.Tensor): input gram X
      Y (torch.Tensor): input gram Y

  Returns:
      torch.Tensor: HSIC
  """

  m = float(X.shape[0])
  return (1.0 / ((m-1.0)**2)) * unscaled_hsic(X, Y)


def mhsic(Ks: list, eps=1e-6):
    """ Multivariate HSIC[K_1, .... K_N] where Ks is a list of N examples by p_i feature matrices
    
    params:
      kernel - construct K_i for each X_i using specified kernel
      detach_diag - when using backprop we don't want to minimize kernel features on the same features. Thus detach the diag
    """

    # prepare one col
    one = torch.ones((Ks[0].shape[0], 1), device=Ks[0].device, dtype=Ks[0].dtype)

    # accumulate hsic
    E_1 = Ks[0] + eps  # first accumulator E_{x_i=1...N, x'_i=1...N}
    E_2 = torch.sum(Ks[0] @ one)  # middle product E_{x_i,x'_i}
    E_3 = Ks[0] @ one  # second accumulator E_{x_i=1...N}
    
    # apply to rest
    for i in range(1, len(Ks)):
        # first elementwise mult
        E_1 = E_1.multiply(Ks[i] + eps)
        
        # middle product
        Kp_i = Ks[i] @ one
        E_2 = E_2 * torch.sum(Kp_i)
        
        # second accumulator
        E_3 = E_3.multiply(Kp_i + eps)

    # finalize first accumulator
    E_1f = torch.sum(E_1 @ one)

    # finalize second accumulator
    E_3f = 2.0*torch.sum(E_2)

    # create mhsic
    hs = E_1f + E_2 + E_3f

    print('E1', E_1f, E_2, E_3f)
    return hs

