
# %%

import numpy as np 
from numpy import ndarray 

# %%


def concat(*arrays: ndarray, axis: int = 0) -> ndarray:
    """ 使用 basic indexing 实现 concat """
    assert len(arrays) != 0, "need at least one array to concatenate"
    
    ndim = arrays[0].ndim 
    dtype = arrays[0].dtype 
    shape = list(arrays[0].shape)

    axis_ = axis + ndim if axis < 0 else axis  
    assert axis_ < ndim, f"axis is out of bounds for array of dimension {ndim}"

    num_elements = sum(np.size(array, axis_) for array in arrays)
    shape[axis_] = num_elements
    out = np.empty(shape=shape, dtype=dtype)

    start = 0
    slices = [slice(None, None, None) for _ in range(ndim)]

    for array in arrays:
        size = np.size(array, axis_)
        slices[axis_] = slice(start, start + size, 1)
        out[*slices] = array
        start += size 
    
    return out


# %%


def stack(*arrays: ndarray, axis: int = 0) -> ndarray:
    """ 使用 basic indexing 实现 stack """
    assert len(arrays) != 0, "need at least one array to stack"

    ndim = arrays[0].ndim 
    dtype = arrays[0].dtype
    shape = list(arrays[0].shape)

    axis_ = axis + ndim if axis < 0 else axis 
    assert axis_ <= ndim, f"axis {axis} is out of bounds for array of dimension {ndim}"

    shape.insert(axis_, len(arrays))
    out = np.empty(shape=shape, dtype=dtype)

    slices = [slice(None, None, None) for _ in range(ndim + 1)]

    for idx, array in enumerate(arrays):
        slices[axis_] = idx
        out[*slices] = array 
    
    return out 

# %%


def analysis(n: int):
    """
    斐波那契数列 (Fibonacci Sequence): 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, ...
    F(1) = 1, F(2) = 1, F(n) = F(n-1) + F(n-2)
        F(n)   调用 1 = F(1) 次,
        F(n-1) 调用 1 = F(2) 次,
        F(n-2) 调用 2 = F(3) 次,
        F(n-3) 调用 3 = F(4) 次,
        F(i)   调用 F(n-i+1) 次,
        F(3)   调用 F(n-2) 次
        F(2)   调用 F(n-1) 次,
    需要注意的是, F(1) 调用次数和 F(3) 调用次数是相同的, 等于 F(n-2), 并不等于 F(n)。
    最终, 叶结点的个数等于 F(1) 和 F(2) 调用次数之和, 即 F(n); 非叶结点的个数等于叶结点个数减一, 即 F(n) - 1。
    因此, 此时递归总调用函数数等于 2 * F(n) - 1, 递归深度最大为 n。
    解决问题的先后顺序: 运算过慢 (lru_cache) -> 运算结果无法用 int64 表示 (大数运算) -> 栈内存爆炸 (提高递归深度限制)
    递归顺序就是 DFS 顺序。整个过程和 2048 游戏很像, 叶结点的数量是 F(n) 就很难蹦。
    """
    # from functools import lru_cache
    from collections import defaultdict

    assert isinstance(n, int) and n > 0, 'n must be a positive integer.'
    n_dist = defaultdict(int)

    # @lru_cache
    def _recursion(n: int, depth: int = 1):
        nonlocal n_dist

        n_dist[n] += 1

        if n == 1 or n == 2:  # leaf nodes
            ret = 1
        else:  # non leaf nodes
            ret = _recursion(n-1, depth+1) + _recursion(n-2, depth+1)
        
        print(f"depth {depth}: _recursion({n}) = {ret}")
        return ret 
    
    result = _recursion(n)
    print(f"fib({n}) = {result}: n_dist = {dict(n_dist)}")

# %%

def block(arrays):

    """
    block 是将一个 内嵌的 ndarray 列表组合成一个 ndarray 对象。
    实现方式是两次 DFS (递归) 遍历:
        1. 第一次遍历: 获取 max_ndim 和 list_ndim 值, 并保证 内嵌列表 的维度数正确 (不会去检查 ndarray 的 shape 正确性)
            max_ndim: 所有 ndarray 的 ndim 最大值
            list_ndim: 内嵌列表 的维度
        2. 第二次遍历: 先将最内层列表的 ndarray 在最后一个维度上 concat, 依次向前, 最外层列表的 ndarray 在倒数第 list_ndim 维度上 concat
    """

    max_ndim, total_size, list_ndim = 0, 0, 0

    def _check_recursion(arrays, depth: int = 0):
        nonlocal max_ndim, total_size, list_ndim

        # leaf nodes 条件: 当前结点是 scalar 或者 ndarray 对象
        if not isinstance(arrays, list):
            # 内嵌列表的维度 和 DFS 的深度是一致的
            if list_ndim == 0:
                list_ndim = depth 
            assert list_ndim == depth 

            if np.ndim(arrays) > max_ndim:
                max_ndim = np.ndim(arrays)
            total_size += np.size(arrays)
            return 

        # non-leaf nodes
        assert len(arrays) != 0
        for array in arrays:
            _check_recursion(array, depth + 1)
    
    _check_recursion(arrays)
    
    def _concat_recursion(arrays, depth: int = 0):

        if depth == list_ndim:  # leaf nodes
            return np.array(arrays, ndmin=max_ndim)  # 确保所有的 ndarray 对象维度都是 max_ndim
        
        arrays_ = [_concat_recursion(array, depth + 1) for array in arrays]
        try:
            return np.concat(arrays_, axis=-(list_ndim-depth))
        except ValueError:
            return np.stack(arrays_, axis=0)
    
    return _concat_recursion(arrays)

# %%

np.unstack
