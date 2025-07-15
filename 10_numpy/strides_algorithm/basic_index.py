
import math 
from types import EllipsisType, NoneType

from core import CoreStruct


def cdiv(a: int, b: int):
    if a > 0 and b > 0:
        return (a + b - 1) // b
    
    return math.ceil(a / b)


def slice_indexing(old_struct: CoreStruct, slices: list[slice]):
    """ basic_indexing 的特殊情况: 所有的 index 都是 slice """
    old_strides = old_struct.strides
    old_shape = old_struct.shape 
    ndim = len(old_shape)
    assert len(slices) == ndim 

    new_start_ptr = old_struct.start_ptr
    new_strides = []
    new_shape = []

    for old_size, old_stride, index in zip(old_shape, old_strides, slices):
        start, stop, step = index.indices(old_size)  # 修正 start, stop 和 step 值
        new_size = cdiv(stop - start, step)  # 多余保留
        if new_size <= 0:  # 空列表
            new_size = 0
            new_stride = old_stride  # 和 NumPy 保持一致
        else:  # 核心代码
            new_start_ptr += start * old_stride
            new_stride = old_stride * step

        new_strides.append(new_stride)
        new_shape.append(new_size)
    
    return CoreStruct(tuple(new_shape), new_start_ptr, tuple(new_strides))


def basic_indexing(old_struct: CoreStruct, indices: list[int | slice | NoneType | EllipsisType]):

    indices = list(indices)

    num_ellipsis = sum([index is Ellipsis for index in indices])
    if num_ellipsis > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")
    if num_ellipsis == 0:  # a[0] ==> a[0, ...]
        num_ellipsis += 1
        indices.append(Ellipsis)

    # 填充 Ellipsis 值
    num_indices = sum([index is not None for index in indices]) - num_ellipsis
    ndim = len(old_struct.shape)
    replace_part = [slice(None, None, None), ] * (ndim - num_indices)
    sep = indices.index(Ellipsis)
    indices = indices[:sep] + replace_part + indices[sep+1:]

    num_indices = sum([index is not None for index in indices])
    if num_indices > ndim:
        raise IndexError(
            f"too many indices for array: array is {ndim}-dimensional, "
            f"but {num_indices} were indexed")
    
    new_start_ptr, new_shape, new_strides = old_struct.start_ptr, [], []
    cur_dim = 0

    for index in indices:
        if index is None:
            new_shape.append(1)
            new_strides.append(0)
            continue

        old_size = old_struct.shape[cur_dim]
        old_stride = old_struct.strides[cur_dim]
    
        if isinstance(index, int):
            if index >= old_size or index < -old_size:  # 越界报错
                raise IndexError(
                    f"index {index} is out of bounds for axis {cur_dim} "
                    f"with size {old_size}")
            index = index + old_size if index < 0 else index  # 修正负 index 情况
            new_start_ptr += index * old_stride

        elif isinstance(index, slice):
            start, stop, step = index.indices(old_size)  # 修正 负 index, 越界, 默认值
            new_size = cdiv(stop - start, step)
            if new_size <= 0:
                new_size = 0
                new_stride = old_stride
            else:
                new_start_ptr += start * old_stride
                new_stride = old_stride * step

            new_strides.append(new_stride)
            new_shape.append(new_size)
        
        cur_dim += 1

    return CoreStruct(tuple(new_shape), new_start_ptr, tuple(new_strides))


if __name__ == "__main__":

    import numpy as np

    from core import build_struct_from_ndarray

    def test_slice_indexing():

        case_no = 0

        a = np.arange(100).reshape(10, 10)
        _, struct0 = build_struct_from_ndarray(a)

        def _test_template(*slices):
            nonlocal case_no
            case_no += 1

            _, struct1 = build_struct_from_ndarray(a[slices])
            struct2 = slice_indexing(struct0, slices)

            print(f"case{case_no}  gold:", struct1)
            print(f"case{case_no} check:", struct2)
            print(struct1 == struct2)

        _test_template(  # case1
            slice(1, 9, 2), slice(-1, -9, -2)
        )

        _test_template(  # case2
            slice(1, 9, 2), slice(1, 9, -2)
        )

        _test_template(  # case3
            slice(5, 5, 2), slice(-1, -9, -2)
        )

        _test_template(  # case4
            slice(None, None, -1), slice(None, None, -1)
        )

        a = np.arange(10000).reshape(10, 10, 10, 10).swapaxes(1, 3)
        _, struct0 = build_struct_from_ndarray(a)

        _test_template(  # case5
            slice(1, 9, 1), slice(1, 9, 2), slice(-1, -9, -1), slice(-1, -9, -2)
        )

        a = np.moveaxis(np.arange(30000).reshape(10, 30, 10, 10)[:, 1:35:3], 1, -1)
        _, struct0 = build_struct_from_ndarray(a)

        _test_template(  # case6
            slice(1, 9, 3), slice(-2, -18, -3), slice(-1, -9, -1), slice(-1, -9, -2)
        )

    # test_slice_indexing()

    def test_basic_indexing():

        case_no = 0

        a = np.arange(100).reshape(10, 10)
        _, struct0 = build_struct_from_ndarray(a)
        print(struct0)

        def _test_template(*indices):
            nonlocal case_no
            case_no += 1

            _, struct1 = build_struct_from_ndarray(a[indices])
            struct2 = basic_indexing(struct0, indices)

            print(f"case{case_no}  gold:", struct1)
            print(f"case{case_no} check:", struct2)
            print(struct1 == struct2)

        _test_template(4)  # case1
        _test_template(4, None, ...)  # case2
        _test_template(..., None, None, None)  # case3
        _test_template(slice(2, 8, 2))  # case4
        _test_template(slice(2, 8, 2), None, ...)  # case5
        _test_template(slice(2, 8, 2), 4)  # case6

        a = np.arange(40000).reshape(10, 20, 10, 20)[:, ::2, :, ::2].transpose((0, 3, 1, 2))
        _, struct0 = build_struct_from_ndarray(a)

        _test_template(slice(2, 8, 2), 4, slice(3, 9, 3), 5, ...)  # case7
    
    test_basic_indexing()
