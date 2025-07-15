
from numpy import ndarray 

from core import CoreStruct


def rebuild_struct_for_transpose(struct: CoreStruct, axes: tuple[int, ...]):
    ndims = len(struct.shape)
    normal_axes = []

    if len(axes) != ndims:
        raise ValueError("axes don't match array")
    
    for axis_idx in axes:
        if not -ndims <= axis_idx <= ndims - 1:
            raise ValueError(f"axis {axis_idx} is out of bounds for array of dimension {ndims}")
        if axis_idx < 0:
            axis_idx += ndims
        normal_axes.append(axis_idx)
    
    if tuple(sorted(normal_axes)) != tuple(range(ndims)):
        raise ValueError("repeated axis in transpose")
    
    new_shape, new_strides = [], []

    for axis_idx in normal_axes:
        new_shape.append(struct.shape[axis_idx])
        new_strides.append(struct.strides[axis_idx])

    return CoreStruct(tuple(new_shape), struct.start_ptr, tuple(new_strides))


if __name__ == '__main__':
    import numpy as np 
    from core import build_struct_from_ndarray

    a = np.arange(120).reshape(2, 3, 4, 5)
    _, struct0 = build_struct_from_ndarray(a)
    _, struct1 = build_struct_from_ndarray(a.transpose(0, 2, 1, -1))
    struct2 = rebuild_struct_for_transpose(struct0, (0, 2, 1, -1))
    print(struct1)
    print(struct2)
