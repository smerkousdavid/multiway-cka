import torch
from src.hsic import bgram_matrix, unscaled_hsic, hsic, mhsic

if __name__ == '__main__':
  print('Testing')

  base = 30 * torch.randn((1, 300, 100))

  Xs = torch.concatenate([base.clone() for _ in range(5)])
  Xs_n = torch.concatenate([torch.randn_like(base) for _ in range(len(Xs))])
  for i in range(len(Xs)):
     Xs_n[i][0, :] = 0.0

  # create gram matrices (this returns a list if provided a list)
  Ks = bgram_matrix(Xs, Xs, kernel='laplacian', force_py_kernel=True)  # , param=1.0, detach_diag=False)
  Ks_n = bgram_matrix(Xs_n, Xs_n, kernel='laplacian', force_py_kernel=True)  # , param=1.0, detach_diag=False)
  print(Ks.shape, Ks_n.shape)
  m = Ks[0].shape[0]
  H = torch.eye(m, device=Ks[0].device) - (1.0 / m)
  # Ks = [H @ K for K in Ks]
  # Ks_n = [H @ K for K in Ks_n]

  # raw test
  # one = torch.ones((Ks[0].shape[0], 1), device=Ks[0].device, dtype=Ks[0].dtype)
  # first = torch.multiply(Ks[0], Ks[1])
  # hs = one.t() @ first @ one + ((one.t() @ Ks[0] @ one) * (one.t() @ Ks[1] @ one)) - 2.0 * one.t() @ (torch.multiply(Ks[0] @ one, Ks[1] @ one))
  # print('RAW MHSIC', hs)

  print('HSIC', unscaled_hsic(Ks[0], Ks[1]), unscaled_hsic(Ks_n[0], Ks_n[1]) / (torch.sqrt(unscaled_hsic(Ks_n[0], Ks_n[0])) * torch.sqrt(unscaled_hsic(Ks_n[1], Ks_n[1]))))
  print('MHSIC', mhsic(Ks), mhsic(Ks_n))