
# %% 

import numpy as np 
from numpy import ndarray 

# %%

def elementwise_squared(arr: ndarray, out: ndarray = None) -> ndarray:
    """ 一元标量运算 """
    iterator = np.nditer(
        op=[arr, out], 
        flags=["external_loop", "buffered",],
        op_flags=[["readonly"], ["writeonly", "allocate"]],
        order="C",
    )

    with iterator:
        for sub_input, sub_output in iterator:
            sub_output[...] = sub_input * sub_input
        
        return iterator.operands[1]

# %%

def elementwise_add(arr1: ndarray, arr2: ndarray, out: ndarray = None) -> ndarray:
    """ 二元标量运算 """
    iterator = np.nditer(
        op=[arr1, arr2, out],
        flags=["external_loop", "buffered", ],
        op_flags=[["readonly", ], ["readonly", ], ["writeonly", "allocate", ], ],
        order="C"
    )

    with iterator:
        for sub_arr1, sub_arr2, sub_out in iterator:
            sub_out[...] = sub_arr1 + sub_arr2
        
        return iterator.operands[2]

# %%

def outer_product(vec1: ndarray, vec2: ndarray, out: ndarray = None) -> ndarray:
    """ 向量张量积 (https://zh.wikipedia.org/zh-cn/外积_(张量积)) """
    iterator = np.nditer(
        op=[vec1, vec2, out],
        flags=["external_loop", ],
        op_flags=[["readonly", ], ["readonly", ], ["writeonly", "allocate", ], ],
        op_axes=[[0, -1], [-1, 0], None, ],
        order="C"
    )

    with iterator:
        for sub_vec1, sub_vec2, sub_out in iterator:
            sub_out[...] = sub_vec1 * sub_vec2
        
        return iterator.operands[2]

# %%

def matmul_v0(mat1: ndarray, mat2: ndarray) -> ndarray:
    """ 矩阵乘法 (使用 elementwise_mul 和 reduce_add [sum] 的实现方式) """
    # (m, k) @ (k, n) = (m, n)
    # (m, k, 1) * (1, k, n) ==> (m, k, n)
    return np.sum(
        np.expand_dims(mat1, axis=-1) * np.expand_dims(mat2, axis=-3),
        axis=-2
    )


def matmul(mat1: ndarray, mat2: ndarray, out: ndarray = None) -> ndarray:
    """ 矩阵乘法, 和 matmul_v0 的实现思路是一致的 """
    assert mat1.ndim == mat2.ndim == 2

    if out is None:
        out_shape = list(np.shape(mat1))
        out_shape[-1] = mat2.shape[-1]
        out = np.zeros(shape=out_shape, dtype=mat1.dtype)

    iterator = np.nditer(
        op=[mat1, mat2, out],
        flags=["reduce_ok", "external_loop", ],
        op_flags=[["readonly", ], ["readonly", ], ["readwrite", "allocate", ], ],
        op_axes=[[0, 1, -1], [-1, 0, 1], [0, -1, 1]],
        order="C" 
    )

    with iterator:
        for sub_mat1, sub_mat2, sub_out in iterator:
            sub_out[...] += sub_mat1 * sub_mat2

        return iterator.operands[2]

# %%

def softmax_v0(arr: ndarray, axis: int = -1) -> ndarray:
    """ softmax 运算 """
    arr = arr - np.max(arr, axis=axis, keepdims=True)
    arr = np.exp(arr, out=arr)
    arr /= np.sum(arr, axis=axis, keepdims=True)
    return arr 


def softmax_v1(arr: ndarray, axis: int = -1) -> ndarray:
    """ softmax 运算: np.nditer 没有办法迭代向量 """

    def single_vec_cal(vec: ndarray) -> ndarray:
        vec = vec - np.max(vec)
        vec = np.exp(vec)
        vec = vec / np.sum(vec)
        return vec 
    
    return np.apply_along_axis(single_vec_cal, axis=axis, arr=arr)


def softmax(arr: ndarray, axis: int = -1, out: ndarray = None) -> ndarray:
    """ softmax 运算 """
    if out is None:
        out = np.empty_like(arr)
    outer_axes = list(range(arr.ndim))
    axis = axis + arr.ndim if axis < 0 else axis 
    outer_axes.pop(axis)
    inner_axes = [axis, ]

    outer_iter, inner_iter = np.nested_iters(
        op=[arr, out], 
        axes=[outer_axes, inner_axes, ],
        flags=["external_loop", ],
        op_flags=[["readonly", ], ["writeonly", ], ],
    )

    with outer_iter, inner_iter:
        for _ in outer_iter:
            for inner_arr, inner_out in inner_iter:
                np.subtract(inner_arr, np.max(inner_arr), out=inner_out)
                np.exp(inner_out, out=inner_out)
                np.divide(inner_out, np.sum(inner_out), out=inner_out)

    return out 

# %%
