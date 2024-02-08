import torch
from torch.autograd.functional import jacobian

# def translate_points(points, translation):
#     return points + translation
#
#
# points = torch.ones((10, 3), dtype=torch.float64, requires_grad=False)
# translation_1D = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64, requires_grad=True)
# translation_2D = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float64, requires_grad=True)
#
# translated_points_1D = translate_points(points, translation_1D)
# translated_points_2D = translate_points(points, translation_2D)
#
# jacobian_1D = jacobian(translate_points, (points, translation_1D))
# jacobian_2D = jacobian(translate_points, (points, translation_2D))
#
# print('Jacobian 1D shapes')
# print(jacobian_1D[0].shape)
# print(jacobian_1D[1].shape)
#
# print('Jacobian 2D shapes')
# print(jacobian_2D[0].shape)
# print(jacobian_2D[1].shape)



def add_vars(*args):

    combined = torch.stack(args)

    return torch.sum(args)

a = torch.tensor([[1.0]], requires_grad=True)
b = torch.tensor([[1.0], [2.0]], requires_grad=True)
c = torch.tensor([[1.0]], requires_grad=True)

abc = (a, b, c)

print(add_vars(*abc))

jac = jacobian(add_vars, abc)