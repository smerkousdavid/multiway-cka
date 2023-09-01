import unittest
from src.kernels import bgram_cuda

# class TestCalculations(unittest.TestCase):

#     def test_sum(self):
#         calculation = Calculations(8, 2)
#         self.assertEqual(calculation.get_sum(), 10, 'The sum is wrong.')
features = torch.tensor([
  [
    [0.0, 0.0, -10.0, 0.0],
    [10.0, 1.0, 1.0, 10.0]
  ]
], requires_grad=True).cuda()
features.retain_grad()
features2 = features.clone().detach()
features2.requires_grad = True
features2.retain_grad()
param = 1.0

lapl = bgram_matrix(features, features, 'laplacian', param=param)
lapl2 = bgram_matrix(features2, features2, 'laplacian', force_py_kernel=True, gamma=param)

# get 1 - 0 laplacian
print(lapl, lapl[0, 1, 0])
print(lapl2, lapl2[0, 1, 0])
single = lapl[0, 1, 0]
single.backward()
single = lapl2[0, 1, 0]
single.backward()

print(features.grad, '\n', features2.grad)
# print(bgram_cuda.MulConstant(features, 1.0))
if __name__ == '__main__':
  print('ok')
  # unittest.main()