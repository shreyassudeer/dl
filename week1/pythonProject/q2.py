# import pytorch library
import torch

# create a tensor of size 2 x 4
input_var = torch.randn(2,4)

# print size
print(input_var.size())

print(input_var)

# dimensions permuted
input_var = input_var.permute(1, 0)

# print size
print(input_var.size())

print(input_var)
