

def attempt_nocopy_reshape(
        old_shape: list[int],
        old_strides: list[int],
        new_shape: list[int],
        order: str = "f"
) -> list[int]:

    # 将 old_shape 中 size = 1 的 axis 都去除掉
    old_shape = [axis_size for axis_size in old_shape if axis_size != 1]

    old_num_axes = len(old_shape)
    new_num_axes = len(new_shape)

    new_strides = [0, ] * len(new_shape)

    # [old_i, old_j) 和 [new_i, new_j) 合并
    old_i, old_j, new_i, new_j = 0, 1, 0, 1

    while old_i < old_num_axes and new_i < new_num_axes:
        old_size, new_size = old_shape[old_i], new_shape[new_i]

        while new_size != old_size:
            if new_size < old_size:
                new_size *= new_shape[new_j]
                new_j += 1
            else:
                old_size *= old_shape[old_j]
                old_j += 1

        for k in range(old_i, old_j - 1):
            if order == "f":
                if old_strides[k+1] != old_shape[k] * old_strides[k]:
                    return None
            else:
                if old_strides[k] != old_shape[k+1] * old_strides[k+1]:
                    return None
        
        if order == "f":
            new_strides[new_i] = old_strides[old_i]
            for k in range(new_i + 1, new_j):
                new_strides[k] = new_strides[k - 1] * new_shape[k - 1]
        else:
            new_strides[new_j - 1] = old_strides[old_j - 1]
            for k in range(new_j - 1, new_i, -1):
                new_strides[k - 1] = new_strides[k] * new_shape[k]

        new_i = new_j
        old_i = old_j
        new_j += 1
        old_j += 1
    
    if new_i >= 1:
        last_stride = new_strides[new_i - 1]
    else:
        last_stride = 8
    if order == "f":
        last_stride *= new_shape[new_i - 1]

    for k in range(new_i, new_num_axes):
        new_strides[k] = last_stride

    return new_strides


print(
    attempt_nocopy_reshape(
        old_shape=[15, 10], old_strides=[10, 1], new_shape=[3, 5, 10], order="f"
    )
)

print(
    attempt_nocopy_reshape(
        old_shape=[15, 10], old_strides=[10, 1], new_shape=[10, 15], order="c"
    )
)
