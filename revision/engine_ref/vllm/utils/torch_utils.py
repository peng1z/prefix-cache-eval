import torch
def get_dtype_size(dtype):
    return torch.tensor([], dtype=dtype).element_size()
