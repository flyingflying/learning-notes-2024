# %%

import numpy as np 
from types import EllipsisType
from numpy import ndarray 


def advanced_indexing(array: ndarray, indices: list[int | slice | ndarray | None | EllipsisType]):
    """ 使用 批量多维索引 实现 advanced indexing """

    # ## step1: 处理布尔数组的情况
    # 所有的布尔数组都会使用 `np.nonzero` 平铺开
    _indices = []
    for index in indices:
        if isinstance(index, ndarray) and np.issubdtype(index.dtype, np.bool):
            _indices.extend(np.nonzero(index))
        else:
            _indices.append(index)
    indices = _indices

    # ## step2: 处理 Ellipsis 的情况
    ellipsis_axis = -1
    for axis, index in enumerate(indices):
        if index is Ellipsis:
            if ellipsis_axis != -1:  # Ellipsis 只能出现一次
                raise IndexError("an index can only have a single ellipsis ('...')")
            ellipsis_axis = axis 

    if ellipsis_axis == -1:
        # 当 Ellipsis 出现在最后时, 也是可以省略的
        indices.append(Ellipsis)
        ellipsis_axis = len(indices) - 1

    array_ndim = array.ndim 
    num_indices = sum([index is not None for index in indices]) - 1  # 排除掉 None 和 Ellipsis 之后的索引数

    if array_ndim < num_indices:
        raise IndexError(f"too many indices for array: array is {array_ndim}-dimensional, but {num_indices} were indexed")

    indices = indices[:ellipsis_axis] + [slice(None, None, None), ] * (array_ndim - num_indices) + indices[ellipsis_axis+1:]

    # ## step3: 标准化 indices 数组, 标准化之后只有 slice 和 整形数组 两种类型
    intarr_axes, slice_axes = [], []
    for axis, index in enumerate(indices):

        if isinstance(index, slice):
            slice_axes.append(axis)
            continue

        if isinstance(index, ndarray):
            if np.issubdtype(index.dtype, np.integer):
                indices[axis] = np.astype(index, np.int32)
                intarr_axes.append(axis)
                continue

            raise IndexError("arrays used as indices must be of integer (or boolean) type")

        if index is None:  # 转换成 slice
            array = np.expand_dims(array, axis=axis)
            indices[axis] = slice(None, None, None)
            slice_axes.append(axis)
            continue
        
        if isinstance(index, int):  # 转换成 整形数组
            indices[axis] = np.array(index, dtype=np.int32)
            intarr_axes.append(axis)
            continue

        raise IndexError(
            "only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and "
            "integer or boolean arrays are valid indices"
        )

    assert len(indices) == array.ndim 
    assert len(intarr_axes) > 0

    # ## step4: 将所有的 整数数组 broadcast 成相同 shape 
    intarrs = np.broadcast_arrays(*[indices[axis] for axis in intarr_axes])
    for axis, intarr in zip(intarr_axes, intarrs):
        indices[axis] = intarr
    
    # ## step5: 将 slice 转换成 整形数组
    intarr_ndim = intarrs[0].ndim  # 整形数组的维度数
    first_intarr_axis = intarr_axes[0]   # 第一个整形数组的位置

    output_ndim = intarr_ndim + len(slice_axes)  # 输出数组的维度数 (所有整形数组之间是联动的!)

    # 所有的 intarr 都要 reshape 成 `intarr_shape`!
    before_ndim = first_intarr_axis  # 第一个整形数组之前的维度数
    after_ndim = len(slice_axes) - first_intarr_axis  # 第一个整形数组之后的维度数
    intarr_shape = [1, ] * before_ndim + list(intarrs[0].shape) + [1, ] * after_ndim

    target_axis = 0
    for axis, index in enumerate(indices):
        if isinstance(index, slice):
            size = np.size(array, axis)
            start, stop, step = index.indices(size)

            shape = [1, ] * output_ndim
            shape[target_axis] = -1

            indices[axis] = np.arange(start, stop, step, dtype=np.int32).reshape(shape)

            target_axis += 1 

        elif isinstance(index, ndarray):
            indices[axis] = index.reshape(intarr_shape)

            # 只有第一个整形数组需要增加, 后续是联动的, 不用添加
            if axis == first_intarr_axis:
                target_axis += intarr_ndim
    
    output = array[tuple(indices)]
    return output 


# %%

a = np.arange(100000).reshape(10, 10, 10, 10, 10)

idx = np.array([[1, 2, 3], [4, 5, 6]])

np.array_equal(
    a[None, 2:9, idx, idx, 8, idx],
    advanced_indexing(a, [None, slice(2, 9), idx, idx, 8, idx])
) 

# 个人认为, 这是 NumPy 的 BUG
np.array_equal(
    a[None, ..., idx, 2:9, idx, 8, idx],
    advanced_indexing(a, [None, Ellipsis, idx, slice(2, 9), idx, 8, idx])
) 

# %%

idx1 = np.arange(1, 4)
idx2 = np.arange(2, 6)
idx3 = np.arange(3, 7)
idx4 = np.arange(4, 9)

arr1 = list(np.meshgrid(idx1, idx2, idx3, idx4, indexing='xy'))
arr1 = np.stack(arr1, axis=0)

arr2 = list(np.meshgrid(idx2, idx1, idx3, idx4, indexing='ij'))
arr2[0], arr2[1] = arr2[1], arr2[0]
arr2 = np.stack(arr1, axis=0)

np.array_equal(arr1, arr2)

# %%
