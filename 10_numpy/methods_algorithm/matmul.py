
# %%

import numpy as np 

# %% 使用 vectorize 实现 matmul 

matmul = np.vectorize(np.dot, signature="(m,k),(k,n)->(m,n)")

a = np.random.randn(4, 10, 20)
b = np.random.randn(4, 20, 30)

r1 = matmul(a, b)
r2 = np.matmul(a, b)

assert r1.shape == r2.shape 
np.max(np.abs(r1 - r2))

# %% 使用 einsum 实现 matmul

def matmul(arr1, arr2):
    return np.einsum("...mk,...kn->...mn", arr1, arr2)

a = np.random.randn(4, 10, 20)
b = np.random.randn(4, 20, 30)

r1 = matmul(a, b)
r2 = np.matmul(a, b)

assert r1.shape == r2.shape 
np.max(np.abs(r1 - r2))

# %% 使用 expand_dims + multiply + sum 实现 matmul

def matmul(arr1, arr2):
    # (m, n, 1) * (1, n, k) -> (m, n, k)
    arr1 = np.expand_dims(arr1, axis=-1)
    arr2 = np.expand_dims(arr2, axis=-3)

    return np.sum(np.multiply(arr1, arr2), axis=-2)


a = np.random.randn(4, 10, 20)
b = np.random.randn(4, 20, 30)

r1 = matmul(a, b)
r2 = np.matmul(a, b)

assert r1.shape == r2.shape 
np.max(np.abs(r1 - r2))

# %% 使用 nditer 实现 matmul 
