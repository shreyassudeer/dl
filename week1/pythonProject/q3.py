import torch
import numpy as np

tensor=torch.tensor([[1,2,3],[4,5,6]])
print(tensor)

numpy_array=tensor.numpy()
print(numpy_array)
print(type(numpy_array))