import numpy as np

import torch
import torch.nn as nn

x = torch.Tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]])
x_view = x.view(2, 8)
print(x_view)