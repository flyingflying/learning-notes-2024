
"""
简单用 一维数组 构建 多维数组。需要注意的是, 对于 多维数组 而言, elements_ptr, shape 和 strides 是核心, 其直接影响 transpose 和 indexing (__getitem__)

题外话: "多维数组" 也是我的一个心结了, 在 17 年的时候, 由于我对于 "知识" 的定位不对, 在研究 numpy 的 ndarray 时吃了很多苦头。现在感概, 对于 "知识" 的定位真的很重要!
"""

import math 
import warnings
import operator
import functools
import itertools
from copy import copy
from typing import Callable
from dataclasses import dataclass

import torch 
from torch import Tensor 


@dataclass
class _MDArrayStruct:
    elements_ptr: int 
    ndim: int 
    shape: tuple[int, ...]
    orders: tuple[int, ...]
    strides: tuple[int, ...]


class MDArray:
    _elements: list[float]
    _struct: _MDArrayStruct

    def __init__(self, elements: list[float], struct: _MDArrayStruct):
        self._elements = elements
        self._struct = struct

    @classmethod
    def from_torch(cls, tensor: Tensor):
        assert tensor.dtype == torch.float64
        elements = tensor.flatten().tolist()
        struct = _MDArrayStruct(
            elements_ptr=0, ndim=tensor.ndim, shape=tuple(tensor.shape),
            orders=tensor.dim_order(), strides=tensor.stride()
        )
        return MDArray(elements, struct)

    @property
    def ndim(self) -> int:
        return self._struct.ndim

    @property
    def shape(self) -> tuple[int, ...]: 
        return self._struct.shape

    def size(self, dim: int = None) -> tuple[int, ...] | int:
        if dim is None:
            return self._struct.shape

        self._check_dim_index(dim)
        return self._struct.shape[dim]

    def numel(self) -> int:
        return functools.reduce(operator.mul, self._struct.shape, 1)

    def stride(self, dim: int = None) -> tuple[int, ...] | int:
        if dim is None:
            return self._struct.strides

        self._check_dim_index(dim)
        return self._struct.strides[dim]

    def dim_order(self, ) -> tuple[int, ...]:
        return self._struct.orders

    def is_contiguous(self) -> bool: 
        # TODO: 实现方案有待商榷, 目前的实现方案过于草率
        raise NotImplementedError
        return tuple(range(self._struct.ndim)) == self._struct.orders

    def transpose(self, dim0: int, dim1: int) -> 'MDArray':
        self._check_dim_index(dim0)
        self._check_dim_index(dim1)

        """ 下面是 核心代码, 一定要理解! """

        def swap(tuple_: tuple):
            tuple_ = list(tuple_)
            tuple_[dim0], tuple_[dim1] = tuple_[dim1], tuple_[dim0]
            return tuple(tuple_)

        new_struct = copy(self._struct)
        new_struct.shape = swap(self._struct.shape)
        new_struct.orders = swap(self._struct.orders)
        new_struct.strides = swap(self._struct.strides)

        """ 核心代码 警告结束 """

        return MDArray(self._elements, new_struct)

    def item(self) -> float:
        num_elements = self.numel()
        if num_elements != 1:
            raise RuntimeError(f"a Tensor with {num_elements} elements cannot be converted to Scalar")
        return self._elements[self._struct.elements_ptr]

    def sin(self) -> 'MDArray':
        return self._uni_ufunc(math.sin)

    def exp(self) -> 'MDArray':
        return self._uni_ufunc(math.exp)

    def __str__(self) -> str:
        # 多维数组输出有很多细节需要处理, 也不是这里想展示的内容, 这里直接使用 torch 中的输出方案
        tensor = torch.tensor(self._ravel_elements(), dtype=torch.float64)
        tensor = tensor.reshape(self._struct.shape)
        return tensor.__str__()

    def __getitem__(self, range_indices: tuple[int | slice]) -> 'MDArray':
        struct = self._struct

        if not isinstance(range_indices, tuple):
            range_indices = (range_indices, )

        # 每一个维度的索引范围, 用元组形式表示: (start, stop, step, keepdim)
        srange_indices: list[tuple[int, int, int]] = []

        # 将所有的索引标准化, 负索引转化成正索引, 并进行类型检查
        for i, range_index in enumerate(range_indices):
            if isinstance(range_index, int):
                # 负索引 转化为 正索引
                if range_index < 0:
                    range_index += struct.shape[i] 

                # 进行索引边界检查
                if range_index < 0 or range_index >= struct.shape[i]:
                    raise IndexError

                srange_indices.append((range_index, range_index+1, 1, False))

            elif range_index == Ellipsis:
                srange_indices.append((0, struct.shape[i], 1, True))

            elif isinstance(range_index, slice):
                srange_index = range_index.indices(struct.shape[i])

                # TODO: 如果 start 大于等于 stop, 应该返回空数组的, 这里暂不实现。
                if srange_index[0] >= srange_index[1]:
                    raise NotImplementedError("不支持返回空数组")

                # NumPy 中实现了 负步数, PyTorch 中直接不支持 负步数, 这里暂不实现。
                if srange_index[2] < 0:
                    raise ValueError("step must be greater than zero")

                srange_indices.append(srange_index + (True, ))

            else:
                raise NotImplementedError("只支持使用 int 和 slice 进行索引")

        """ 下面是 核心代码, 一定要理解! """
    
        # ndexing 操作主要改变的是数组的 shape 和 elements_ptr, 不影响数组的 stride, 只有 step 会影响到数组的 stride。
        new_element_ptr = struct.elements_ptr
        new_shape, new_strides, new_orders, new_ndim = [], [], [], 0
        for i, (start, stop, step, keepdim) in enumerate(srange_indices):
            new_element_ptr += start * struct.strides[i]

            if keepdim:
                new_shape.append((stop - start - 1) // step + 1 )
                new_strides.append(struct.strides[i] * step)
                new_orders.append(struct.orders[i])
                new_ndim += 1

        """ 核心代码 警告结束 """

        # 下面两行代码是为了适配 PyTorch 中的结果, 其实 order 不是 多维数组 的核心部分, 可以忽略的
        index_map = {v: i for i, v in enumerate(sorted(new_orders))}
        new_orders = [index_map[v] for v in new_orders]

        return MDArray(
            elements=self._elements, 
            struct=_MDArrayStruct(
                new_element_ptr, new_ndim, 
                tuple(new_shape), tuple(new_orders), tuple(new_strides)
            )
        )

    def _check_dim_index(self, dim: int):
        # 检查数组的维度的索引
        struct = self._struct

        if struct.ndim == 0:
            raise IndexError(f"Dimension specified as {dim} but tensor has no dimensions")

        min_dim = -struct.ndim
        max_dim = struct.ndim - 1

        if dim < min_dim or dim > max_dim:
            raise IndexError(f"Dimension out of range (expected to be in range of [{min_dim}, {max_dim}], but got {dim})")

    def _get_exact_item(self, indices: list[int]) -> float:
        warnings.warn("this function is deprecated", DeprecationWarning)
        struct = self._struct

        global_index = 0
        for i in range(struct.ndim):
            if indices[i] >= struct.shape[i]:
                raise IndexError

            global_index += indices[i] * struct.strides[i]

        return self._elements[global_index]

    def _ravel_elements(self) -> list[float]:
        struct = self._struct
        new_elements = []
        ranges = [range(dim_size) for dim_size in struct.shape]

        # 这里使用 itertools.product 实现任意维度的 for 循环
        for indices in itertools.product(*ranges):
            offset = 0
            for index, stride in zip(indices, struct.strides):
                offset += index * stride
            new_elements.append(self._elements[struct.elements_ptr + offset])

        return new_elements

    def _uni_ufunc(self, func: Callable) -> 'MDArray':  # 一元 ufunc
        elements = self._ravel_elements()
        elements = [func(element) for element in elements]

        struct = copy(self._struct)
        struct.elements_ptr = 0
        # 这里的实现方式和 PyTorch 中不一致。
        # 在 PyTorch 中, 实际的计算量可能大于数组中的元素个数
        struct.strides = tuple([
            functools.reduce(operator.mul, struct.shape[i+1:], 1) for i in range(struct.ndim)
        ])
        struct.orders = tuple(range(struct.ndim))

        return MDArray(elements, struct)

    def _bi_ufunc(self, func: Callable, other: 'MDArray') -> 'MDArray':  # 二元 ufunc
        raise NotImplementedError("暂不进行探索")

    def _reduce_ufunc(self, func: Callable, dim: int = None) -> 'MDArray':  # reduce ufunc
        raise NotImplementedError("暂不进行探索")

if __name__ == "__main__":
    def check_transpose():
        tensor = torch.randn(4, 6, 7, 8, dtype=torch.float64)
        md_array = MDArray.from_torch(tensor)

        print(tensor[1, 2, 3, 4].item())
        print(md_array[1, 2, 3, 4].item())

        tensor_t = tensor.transpose(1, 2)
        md_array_t = md_array.transpose(1, 2)

        print(tuple(tensor_t.shape), tensor_t.stride(), tensor_t.dim_order())
        print(md_array_t.shape, md_array_t.stride(), md_array_t.dim_order())
        
        print(tensor_t[1, 2, 3, 4].item())
        print(md_array_t[1, 2, 3, 4].item())

    def check_getitem_v1():
        tensor = torch.randn(4, 6, 7, 9, dtype=torch.float64)
        md_array = MDArray.from_torch(tensor)

        tensor = tensor[1:3, 2, 1, 1]
        md_array = md_array[1:3, 2, 1, 1]

        print(tuple(tensor.shape), tensor.stride(), tensor.dim_order(),)
        print(md_array.shape, md_array.stride(), md_array.dim_order(),)

        print(tensor)
        print(md_array)

    def check_getitem_v2():
        tensor = base_tensor = torch.randn(4, 6, 7, 9, dtype=torch.float64)
        md_array = MDArray.from_torch(tensor)

        tensor = tensor.transpose(1, 3)  # [4, 9, 7, 6]
        md_array = md_array.transpose(1, 3)

        tensor = tensor[1, 1:8:2, 1:7:3, 5]
        md_array = md_array[1, 1:8:2, 1:7:3, 5]

        print(tuple(tensor.shape), tensor.stride(), (tensor.data_ptr() - base_tensor.data_ptr()) // 8, tensor.dim_order(), tensor.ndim)
        print(md_array.shape, md_array.stride(), md_array._struct.elements_ptr, md_array.dim_order(), md_array.ndim)

        print(tensor)
        print(md_array)
        print(str(tensor) == str(md_array))

        print(tensor[1, 1].item(), md_array[1, 1].item())

        sin_tensor = tensor.sin()
        sin_md_array = md_array.sin()
        print(tuple(sin_tensor.shape), sin_tensor.stride(), sin_tensor.dim_order(), sin_tensor.ndim)
        print(sin_md_array.shape, sin_md_array.stride(), sin_md_array.dim_order(), sin_md_array.ndim)

        print(sin_tensor)
        print(sin_md_array)
        print(str(sin_tensor) == str(sin_md_array))

    # check_transpose()

    check_getitem_v2()
