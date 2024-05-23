# from torchtext.data import Field, TabularDataset, BucketIterator
import numpy as np
import torch

data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# to numpy with dtype float32
np_data = np.array(data, dtype=np.float32)
print(np_data)

# to torch tensor
torch_data = torch.from_numpy(data)
print(torch_data)


