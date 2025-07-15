
# %%

import numpy as np 
from numpy import ndarray

# %%

a = np.arange(720).reshape(2, 3, 4, 5, 6)

print(a.shape)

# swap: 交换 axis1 和 axis2 的位置
# print(np.swapaxes(a, axis1=1, axis2=3).shape)

# moveaxis: 当 src_axis < dst_axis 时, src_axis 不断和右边 axis 交换, 直到到达 dst_axis
# moveaxis: 当 src_axis > dst_axis 时, src_axis 不断和左边 axis 交换, 直到到达 dst_axis
# reference: https://stackoverflow.com/a/39878857 & https://github.com/scipy/scipy/issues/14491 
print("moveaxis:", np.moveaxis(a, source=1, destination=3).shape)
print("moveaxis:", np.moveaxis(a, source=1, destination=0).shape)

# rollaxis: 当 start <= src_axis 时, src_axis 不断和左边的 axis 交换位置, 直到到达 start 的位置
# rollaxis: 当 start > src_axis 时, src_axis 不断和右边的 axis 交换位置, 直到到达 start - 1 的位置
print("rollaxis:", np.rollaxis(a, axis=1, start=4).shape)
print("rollaxis:", np.rollaxis(a, axis=1, start=0).shape)

# %%

# [n_samples, n_features] ==> [n_sample_groups, n_samples, n_features]
np.stack([np.arange(25).reshape(5,5), np.arange(25, 50).reshape(5, 5)], axis=0)

# [n_samples, n_features] ==> [n_samples, n_feature_groups, n_features]
np.stack([np.arange(25).reshape(5,5), np.arange(25, 50).reshape(5, 5)], axis=1)

# axis=0: matrix 作为 底面, 向 "上" (顶面) 堆叠
np.stack([np.arange(25).reshape(5,5), np.arange(25, 50).reshape(5, 5)], axis=0)

# axis=1: matrix 作为 背面, 向 "前" (正面) 堆叠
np.stack([np.arange(25).reshape(5,5), np.arange(25, 50).reshape(5, 5)], axis=1)

# axis=2: matrix 作为 左侧面, 向 "右" (右侧面) 堆叠
np.stack([np.arange(25).reshape(5,5), np.arange(25, 50).reshape(5, 5)], axis=2)

# %%

a = np.array([
    ["0, 0", "0, 1", "0, 2", "0, 3"],
    ["1, 0", "1, 1", "1, 2", "1, 3"],
    ["2, 0", "2, 1", "2, 2", "2, 3"],
    ["3, 0", "3, 1", "3, 2", "3, 3"],
])

# %%


def _choose(array: ndarray, choices: ndarray):

    result = np.empty_like(array)

    for idx in np.ndindex(array.shape):
        subarray = choices[array[idx]]

        if not isinstance(subarray, ndarray):
            print(idx, array[idx], choices, subarray)
            result[idx] = subarray
        elif array.ndim == choices.ndim:
            result[idx] = subarray[idx]
        elif array.ndim > choices.ndim:
            result[idx] = subarray[idx[:choices.ndim]]
    
    return result 

# %%

a = np.array([[[2,0,1],
              [2,1,0],
              [1,2,2]]])

np.choose(a, np.arange(27).reshape(3, 3, 3))
# np.choose(a, np.arange(3).reshape(3,))

# %%
