
from dataclasses import dataclass

import numpy as np 
from numpy import ndarray 


@dataclass
class CoreStruct:
    shape: tuple[int, ...]
    start_ptr: int 
    strides: tuple[int, ...]

    def __post_init__(self):
        assert len(self.shape) == len(self.strides)
        assert all([axis_size >= 0 for axis_size in self.shape])
        assert self.start_ptr >= 0


def build_struct_from_ndarray(array: ndarray):
    base_array = array.base if array.base is not None else array
    # base_array 一定是 C order 或者 F order 的
    assert base_array.flags.c_contiguous or base_array.flags.f_contiguous
    base_array = base_array.ravel(order="k")

    itemsize = array.itemsize
    # 这里用 start_ptr 模拟 C 语言中的指针 (NumPy 中首元素的指针是 void * 类型)
    start_ptr = (array.ctypes.data - base_array.ctypes.data) // itemsize
    strides = tuple([stride // itemsize for stride in array.strides])
    struct = CoreStruct(array.shape, start_ptr, strides)

    return base_array, struct
