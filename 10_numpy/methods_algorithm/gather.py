
# %%

import torch 
import numpy as np 

from torch import Tensor 
from numpy import ndarray 

# %%

a = torch.arange(100).reshape(10, 10)

idx = torch.tensor([
    [5, 5, 3, 3],  # (0, 5), (0, 5), (0, 3), (0, 3)
    [1, 1, 9, 8],  # (1, 1), (1, 1), (1, 9), (1, 8)
    [2, 6, 5, 8],  # (2, 2), (2, 6), (2, 5), (2, 8)
])

torch.gather(a, 1, idx)

# %%

a = torch.arange(100).reshape(10, 10)

idx = torch.tensor([
    [5, 5, 3, 3],  # (5, 0), (5, 1), (3, 2), (3, 3)
    [1, 1, 9, 8],  # (1, 0), (1, 1), (9, 2), (8, 3)
    [2, 6, 5, 8],  # (2, 0), (6, 1), (5, 2), (8, 3)
])

torch.gather(a, 0, idx)

# %%

"""
torch.gather 和 torch.scatter 是一对 API: 前者是从数组中取 item, 后者是给数组中的 item 赋值。
torch.index_select 和 torch.gather 是相似的 API:
    index_select: 获取 `input` 第 `dim` 层列表中的元素, `index` 就是该层列表的索引值。
    gather: `index` 依然是 `input` 数组第 `dim` 层列表的索引, 但是会考虑位置信息, 建议结合下面的代码理解。
"""


def gather_np(input: ndarray, dim: int, index: ndarray) -> ndarray:
    assert index.ndim == input.ndim
    indices = list(np.indices(index.shape, sparse=True))
    indices[dim] = index 
    return input[*indices]


def index_select_np(input: ndarray, dim: int, index: ndarray) -> ndarray:
    assert index.ndim == 1
    indices = [slice(None, None, None), ] * input.ndim 
    indices[dim] = index 
    return input[*indices]


# %%

a_np = np.arange(100).reshape(10, 10)

idx_np = np.array([
    [5, 5, 3, 3],  # (0, 5), (0, 5), (0, 3), (0, 3)
    [1, 1, 9, 8],  # (1, 1), (1, 1), (1, 9), (1, 8)
    [2, 6, 5, 8],  # (2, 2), (2, 6), (2, 5), (2, 8)
])

gather_np(a_np, 1, idx_np)


# %%

a_np = np.arange(100).reshape(10, 10)

idx_np = np.array([
    [5, 5, 3, 3],  # (5, 0), (5, 1), (3, 2), (3, 3)
    [1, 1, 9, 8],  # (1, 0), (1, 1), (9, 2), (8, 3)
    [2, 6, 5, 8],  # (2, 0), (6, 1), (5, 2), (8, 3)
])

gather_np(a_np, 0, idx_np)
