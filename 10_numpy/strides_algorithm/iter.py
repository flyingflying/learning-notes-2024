
from numpy import ndarray

from core import CoreStruct


def gen_corder_index(shape: tuple[int, ...]):

    # 树构建: 每一个 axis 作为树的一层, 每一层的结点数等于 axis 的 size, 相邻层的结点两两全连接
    # 这样, 我们可以使用深度优先算法实现 corder index 索引生成
    ndim = len(shape)
    path = []

    # 在这里, 递归的深度最大值是 ndim + 1
    # 在 NumPy 中, ndim 最大值是 64, 因此不用担心递归深度的问题
    def _recursion(depth):

        if len(path) == ndim:
            yield tuple(path)
            return 

        for i in range(shape[depth]):
            path.append(i)
            yield from _recursion(depth + 1)
            path.pop(-1)
    
    yield from _recursion(0)


def gen_forder_index(shape: tuple[int, ...]):

    # F order 需要反向建树, 并将 path 反过来输出
    ndim = len(shape)
    path = []

    def _recursion(depth):

        if len(path) == ndim:
            yield tuple(reversed(path))
            return 

        for i in range(shape[depth]):
            path.append(i)
            yield from _recursion(depth - 1)
            path.pop(-1)
    
    yield from _recursion(ndim - 1)


def corder_iterator(array: ndarray, struct: CoreStruct):
    for grid_index in gen_corder_index(struct.shape):
        global_index = sum([i * j for i, j in zip(grid_index, struct.strides)])
        yield array[struct.start_ptr + global_index]


def forder_iterator(array: ndarray, struct: CoreStruct):
    for grid_index in gen_forder_index(struct.shape):
        global_index = sum([i * j for i, j in zip(grid_index, struct.strides)])
        yield array[struct.start_ptr + global_index]


def korder_iterator(array: ndarray, struct: CoreStruct):
    from numpy import argsort, abs

    # NumPy 文档中的原话:
    # 'k' means to read the elements in the order they occur in memory, 
    # except for reversing the data when strides are negative.

    # axes = argsort(abs(struct.strides), stable=True)[::-1].tolist()
    axes = argsort(abs(struct.strides), stable=True).tolist()
    shape = [struct.shape[axis] for axis in axes]
    strides = [struct.strides[axis] for axis in axes]

    struct = CoreStruct(shape=shape, start_ptr=struct.start_ptr, strides=strides)

    # yield from corder_iterator(array, struct)
    yield from forder_iterator(array, struct)


if __name__ == "__main__":

    import numpy as np 
    from core import build_struct_from_ndarray

    def test_gen_corder_index():
        shape = (2, 3, 4)
        for index in gen_corder_index(shape):
            print(index)

    # test_gen_corder_index()

    def test_gen_forder_index():
        shape = (2, 3, 4)
        for index in gen_forder_index(shape):
            print(index)
    
    # test_gen_forder_index()

    def advance_test():
        import numpy as np 

        shape = (2, 3, 4, 1)
        a = np.random.randn(*shape)

        r1 = a.ravel(order="c")
        r2 = np.array([a[idx] for idx in gen_corder_index(shape)])
        print("gen_corder_index:", np.all(r1 == r2))

        r1 = a.ravel(order="f")
        r2 = np.array([a[idx] for idx in gen_forder_index(shape)])
        print("gen_forder_index:", np.all(r1 == r2))
    
    # advance_test()

    def show_test():
        import numpy as np 

        shape = (2, 3, 4)
        a = np.arange(np.prod(shape)).reshape(*shape)
        print(a)

        rc = np.array([a[idx] for idx in gen_corder_index(shape)])
        print("C order:", rc, a.ravel("c"))

        rf = np.array([a[idx] for idx in gen_forder_index(shape)])
        print("F order:", rf, a.ravel("f"))
    
    # show_test()


    def test_corder_iterator():
        a = np.arange(100).reshape(10, 10)[1:9:2, -1:-9:-2]

        print(a)
        print(np.ravel(a, order="c"))

        base_array, struct = build_struct_from_ndarray(a)
        print(np.array(
            list(corder_iterator(base_array, struct))
        ))
    
    # test_corder_iterator()

    def test_forder_iterator():
        a = np.arange(100).reshape(10, 10)[1:9:2, -1:-9:-2]

        print(a)
        print(np.ravel(a, order="f"))

        base_array, struct = build_struct_from_ndarray(a)
        print(np.array(
            list(forder_iterator(base_array, struct))
        ))

    # test_forder_iterator()

    def test_korder_iterator():
        # a = np.arange(100).reshape(10, 10).T[1:9:2, -1:-9:-2]
        # a = np.asfortranarray(np.arange(100).reshape(10, 10)[::2, ::2])
        a = np.arange(7776).reshape(6, 6, 6, 6, 6).swapaxes(1, 3)[::-3, ::-3, ::3, ::3, ::-3]

        r1 = np.ravel(a, order="k")

        base_array, struct = build_struct_from_ndarray(a)
        r2 = np.array(
            list(korder_iterator(base_array, struct))
        )

        print(r1)
        print(r2)
        print(r1 == r2)
        print(np.all(r1 == r2).item())

    test_korder_iterator()
