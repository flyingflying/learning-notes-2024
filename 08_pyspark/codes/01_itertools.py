
"""
使用 Python 的生成器实现 itertools 模块。

itertools 模块的文档: https://docs.python.org/3/library/itertools.html 
"""

# %%

import operator 
from typing import *
from dataclasses import dataclass


# %% 元素级别操作


def map_(func: Callable, *iterables: List[Iterable]):
    iterators = [iter(iterable) for iterable in iterables]
    while True:
        try:
            arguments = [next(iterator) for iterator in iterators]
        except StopIteration:
            return 
        yield func(*arguments)


def starmap(function: Callable, iterable: Iterable):
    """ iterable 中的元素是 function 中的参数 """
    iterator = iter(iterable)

    for arguments in iterator:
        yield function(*arguments)


def zip_(*iterables: List[Iterable]):
    iterators = [iter(iterable) for iterable in iterables]

    while True:
        try:
            yield tuple([next(iterator) for iterator in iterators])
        except StopIteration:
            return 


def zip_longest(*iterables: List[Iterable], fillvalue: Any = None):
    """ 所有的迭代器 "按位" 配对构成元组, 如果有迭代器先被耗尽, 则用 fillvalue 来填充 """
    num_actived = len(iterables)

    def warp_iterator(iterable):
        nonlocal num_actived

        yield from iter(iterable)
        num_actived -= 1

        if num_actived == 0:
            return 

        while True:
            yield fillvalue
    
    iterators = [warp_iterator(iterable) for iterable in iterables]

    while True:
        try:
            yield tuple([next(iterator) for iterator in iterators])
        except StopIteration:
            return 


def filter_(function: Callable, iterable: Iterable):
    iterator = iter(iterable)
    function = bool if function is None else function

    for element in iterator:
        if function(element):
            yield element 


def filterfalse(predicate: Callable, iterable: Iterable):
    predicate = bool if predicate is None else predicate
    iterator = iter(iterable)

    for element in iterator:
        if not predicate(element):
            yield element


def compress(data: Iterable, selectors: Iterable):
    iterator = iter(data)
    selectors = iter(selectors)

    while True:
        try:
            element = next(iterator)
            selector = next(selectors)
        except StopIteration:
            return 
        
        if selector:
            yield element 


def dropwhile(predicate: Callable, iterable: Iterable):
    iterator = iter(iterable)

    for element in iterator:
        if not predicate(element):
            yield element
            break

    for element in iterator:
        yield element


def takewhile(predicate: Callable, iterable: Iterable):
    iterator = iter(iterable)

    for element in iterator:
        if not predicate(element):
            break
        yield element


# %% 无限迭代器


def count(start: int = 0, step: int = 1):
    """ 和 range 功能相似, 只是没有了 stop 参数 """

    cur_idx = start 
    while True:
        yield cur_idx
        cur_idx += step 


def cycle(iterable: Iterable):
    """ 无限重复生成 iterable 中的元素 """

    elements = list(iterable)
    if len(elements) == 0:
        return 

    while True:
        for element in elements:
            yield element 


def repeat(object, times: int = None):
    """ 重复生成 object 对象 times 次。如果 times 为 None, 就是无限迭代器。"""
    
    if times is not None:
        for _ in range(times):
            yield object 
        return 
    
    while True:
        yield object 


# %% 迭代器级别操作


def chain(*iterables: List[Iterable]):
    """ 将多个迭代器合成一个迭代器, 和 RDD 的 flatMap 的功能相似。 """

    for iterable in iterables:
        iterator = iter(iterable)
        yield from iterator


def islice(iterable: Iterable, start: int = None, stop: int = None, step: int = None):
    """
    islice 的含义是 iterator slice, 即 "迭代器切片", 不是 is_slice 的意思。
    slice 主要有三个参数: start, stop 和 step, 不支持参数为负值的情况。
    需要注意的是, islice 消耗 迭代器 的元素个数是 max(start, stop), 和 step 参数无关。
    如果 stop 是 None, 那就是消耗整个迭代器中的元素。
    """

    try:
        assert (start is None) or (isinstance(start, int) and start >= 0), "start 参数只能是 None 或者非负整数"
        assert (stop is None) or (isinstance(stop, int) and stop >= 0), "stop 参数只能是 None 或者非负整数"
        assert (step is None) or (isinstance(step, int) and step >= 1), "step 参数只能是 None 或者正整数"
    except AssertionError as ae:
        # raise ValueError(*ae.args).with_traceback(ae.__traceback__) from None 
        raise ValueError(*ae.args) from None 

    try:
        iterator = iter(iterable)

        start = 0 if start is None else start 
        # 先将 start 位置之前的所有元素 "消耗" 完, 这样可以让 start 值归零, 后续功能实现更加方便
        for _ in range(start):
            next(iterator)

        stop = float("inf") if stop is None else stop - start 
        if stop <= 0:
            return 

        step = 1 if step is None else step 

        cur_idx = 0
        while cur_idx < stop:
            if cur_idx % step == 0:
                yield next(iterator)
            else:
                next(iterator)
            cur_idx += 1
    
    except StopIteration:
        return 


@dataclass
class Node:
    """ 用于 LinkedIterator """
    element: Any 
    next: Optional['Node'] 

    @property
    def is_empty(self):
        return self.next is None
    
    @classmethod
    def get_empty_node(cls):
        return cls(None, None)
    
    def fill_empty(self, giter: Iterator):
        if not self.is_empty:
            return 
        self.element = next(giter)
        self.next = Node.get_empty_node()


class LinkedIterator:
    """ 用于 tee 迭代器 """
    giter: Iterator
    node: Node 

    def __init__(self, iterable: Iterable):
        iterator = iter(iterable)

        if isinstance(iterator, LinkedIterator):
            self.giter = iterator.giter
            self.node = iterator.node
        else:
            self.giter = iterator
            self.node = Node.get_empty_node()

    def __iter__(self):
        return self 

    def __next__(self):
        self.node.fill_empty(self.giter)  # 懒加载
        element = self.node.element
        self.node = self.node.next
        return element


def tee(iterable: Iterable, n: int = 2):
    """ 将一个迭代器复制 n 次, 底层使用链表实现的。"""

    assert isinstance(n, int) and n >= 0

    if n == 0:
        return tuple()

    iterator = LinkedIterator(iterable)
    results = [iterator, ]
    results.extend([LinkedIterator(iterator) for _ in range(n - 1)])
    return tuple(results) 


# %% 排列组合


def product(*iterables: List[Iterable], repeat: int = 1):
    """
    分类: 多元运算
    功能: 将多个集合中的元素进行 "全匹配"
    实现方式: 构建 Graph, 使用 DFS 算法实现。
    构建 Graph 的方式: 
        "节点" 是集合中的元素, 一个集合为一层, 一共有 n 层, 其中 n 是集合的数量。
        相邻层之间采取 "全连接" 的方式, 每一个 "节点" 的 "子节点" 是其下一层的所有元素。
    """
    collections = [tuple(iterable) for iterable in iterables] * repeat 
    n = len(collections)
    path = []

    def dfs():
        if len(path) == n:  # 叶节点, 直接抛出
            yield tuple(path)
            return 

        # 当前节点为 path[-1], 其叶节点为 collections[len(path)], 遍历所有的叶节点
        for element in collections[len(path)]:
            path.append(element)
            yield from dfs()  # 递归迭代器要用 yield from 
            path.pop()
    
    return dfs()


def product_v2(*iterables: List[Iterable], repeat: int = 1):
    """ 非递归的 DFS 实现方式, 速度较慢 """
    collections = [tuple(iterable) for iterable in iterables] * repeat

    stack = [(), ]
    while len(stack) != 0:
        cur_node = stack.pop(0)
        level = len(cur_node)
        child_nodes = collections[level]

        if level == len(collections) - 1:  # 子节点是叶节点
            for child_node in child_nodes:
                yield cur_node + (child_node, )
        else:  # 子节点不是叶节点: 将所有的节点放入 stack 
            stack.extend([cur_node + (child_node, ) for child_node in child_nodes])


def product_v3(*iterables: List[Iterable], repeat: int = 1):
    """ 
    双层循环 + 嵌套迭代器 的实现方式, 思路非常巧妙。
    reference: https://stackoverflow.com/a/12605800 
    """
    collections = [tuple(iterable) for iterable in iterables] * repeat
    
    def nested_loop(values: Iterable, upstream: Iterator):
        for prefix in upstream:
            for value in values:
                yield prefix + (value, )

    stack = [(), ]
    for collection in collections:
        stack = nested_loop(collection, stack)
    
    return stack 


def combinations(iterable: Iterable, r: int):
    """
    分类: 生成子集合 - 排列组合
    功能: 返回集合的所有 "组合", 组合数可以通过 math.comb(n, r) 计算, 其中 n 是集合中元素个数。
    实现方式: 构建 Graph, 使用 DFS 算法实现。
    构建 Graph 的方式: 
        "节点" 是集合中的元素, "节点" 的 "子节点" 是在该元素位置之后的所有元素,
        最多有 r 层, 每一个路径即为一个 "组合"。
    reference: https://walkccc.me/LeetCode/problems/77/#__tabbed_1_3 
    """
    # [''.join(subseq) for subseq in combinations('ABCDE', 3)]
    # ['ABC', 'ABD', 'ABE', 'ACD', 'ACE', 'ADE', 'BCD', 'BCE', 'BDE', 'CDE']

    collection = tuple(iterable)
    n = len(collection)
    path = []

    def dfs(s):  # collection[s-1] 即为当前 "节点"
        if len(path) == r:
            yield tuple(path)
            return 

        for i in range(s, n):  # 遍历当前 "节点" 的所有 "子节点"
            path.append(collection[i])
            yield from dfs(i+1)  # 递归迭代器要用 yield from 
            path.pop()
        
        return 
    
    return dfs(0)
    

def combinations_with_replacement(iterable: Iterable, r: int):
    """
    和 combinations 功能相似, 只是允许 "子集合" 中的元素可以重复。
    换言之, 就是找出所有长度为 r 的 "子集合", 并且 "子集合" 内的顺序无关, "子集合" 内元素可以重复。
    返回的 "子集合" 数量为 math.comb(n+r-1, r)。
    依旧使用 DFS 算法实现, Graph 的构建方式和 combinations 类似, 
    只是现在一个 "节点" 的 "子节点" 包含元素本身。
    """
    # [''.join(subseq) for subseq in combinations_with_replacement('ABCD', 3)]
    # ['AAA', 'AAB', 'AAC', 'AAD', 'ABB', 'ABC', 'ABD', 'ACC', 'ACD', 'ADD', 
    #  'BBB', 'BBC', 'BBD', 'BCC', 'BCD', 'BDD', 'CCC', 'CCD', 'CDD', 'DDD']

    collection = tuple(iterable)
    n = len(collection)
    path = []

    def dfs(s):  
        # 若 s = 0, 则当前节点为 "虚拟" 根节点
        # 若 s > 0, 则当前节点为 collection[i]
        if len(path) == r:  # 当前节点为 "叶节点"
            yield tuple(path)
            return 

        for i in range(s, n):  # 遍历当前节点的所有 "子节点"
            path.append(collection[i])
            yield from dfs(i)  # 递归迭代器要用 yield from 
            path.pop()
        
        return 
    
    return dfs(0)


def permutations(iterable: Iterable, r: int):
    """
    分类: 生成子集合 - 排列组合
    功能: 返回集合的所有 "排列", 组合数可以通过 math.perm(n, r) 计算, 其中 n 是集合中元素个数。
    实现方式: 构建 Tree, 使用 DFS 算法实现。
    构建 Tree 的方式: 
        "节点" 是集合中的元素, "节点" 的 "子节点" 是集合中的所有元素 (除去自身和 "祖先节点"),
        最多有 r 层, 每一个路径即为一个 "排列"。
    注意: permutations 也有 replacement 版本, 对应为 product(iterable, r)
    """
    # [''.join(subseq) for subseq in permutations('ABCD', 3)]
    # ['ABC', 'ABD', 'ACB', 'ACD', 'ADB', 'ADC', 'BAC', 'BAD', 'BCA', 'BCD', 'BDA', 'BDC', 
    #  'CAB', 'CAD', 'CBA', 'CBD', 'CDA', 'CDB', 'DAB', 'DAC', 'DBA', 'DBC', 'DCA', 'DCB']

    collection = tuple(iterable)
    n = len(collection)
    path = []  # 这里存的是 索引

    def dfs():
        if len(path) == r:
            yield tuple([collection[idx] for idx in path])
            return 
        
        for i in range(0, n):
            if i in path:  # 所有在 path 中的元素都不是 "子节点"
                continue
            path.append(i)
            yield from dfs()  # 递归迭代器要用 yield from 
            path.pop()

    return dfs()


# %% 分组运算


def groupby(iterable: Iterable, key: Callable = None):
    """
    将迭代器中的元素进行分组, 分组依据是 key 函数。
    这里实现的是 排序分组, 即将相邻相同的元素作为一组, 以迭代器的形式返回。
    换言之, 就是将一个迭代器拆分成多个迭代器, 在不相等的两个元素之间划分界限。
    因此, 使用其的前提是迭代器中的元素有序, 可以配合 sorted 函数使用。
    此时, 存在一个问题: 排序的前提是所有数据都在内存中, 一个迭代器经过 sorted 函数排序后就不是迭代器了。
    这里提供一个解决办法: 参考 advanced_group_by 函数。
    """

    if key is None:
        key_func = lambda x: x 
    else:
        key_func = key 

    iterator = iter(iterable)

    try:
        cur_val = next(iterator)
    except StopIteration:
        return 

    cur_key = key_func(cur_val)
    exhausted = False

    while not exhausted:
        target_key = cur_key

        def _generator():
            # cur_val 是此次迭代中的元素值, cur_key = key_func(cur_val)
            # target_key 是当前分组的 key 值, exhausted 表示当前迭代器元素是否耗尽
            nonlocal cur_key, cur_val, target_key, exhausted
            yield cur_val

            for cur_val in iterator:
                cur_key = key_func(cur_val)

                if cur_key != target_key:
                    return 
                
                yield cur_val
            # 如果执行到这里, 说明 iterator 中已经没有元素了
            exhausted = True
        
        val_generator = _generator()
        yield cur_key, val_generator

        # 按照正常的使用方式, 此时用户应该将 val_generator 中的元素都迭代完成
        # 但是用户不一定会将 val_generator 迭代完成, 那么就需要我们将其迭代完成
        for _ in val_generator:
            pass 


def advance_groupby(iterable: Iterable, key: Callable = None, batch_size: int = None):

    """
    参考 PySpark 中的实现方式:
        1. 使用 islice 将一个 迭代器 分成多个 "子迭代器", 
        2. 每一个 "子迭代器" 内的元素都加载到内存中进行排序, 然后存储到磁盘文件中, 构建新的 "子迭代器"
        3. 用 heapq.merge 将所有新构建的有序 "子迭代器" 合成一个 迭代器, 这个迭代器就是有序的
    """

    import heapq
    import pickle
    import struct
    import tempfile
    import itertools

    def sorted_batch_generator():
        iterator = iter(iterable)

        while True:
            sub_elements = list(itertools.islice(iterator, batch_size))
            sub_elements.sort(key=key)

            if len(sub_elements) == 0:
                break 

            yield sub_elements

    def build_disk_stream(elements: list):
        with tempfile.TemporaryFile("w+b") as file_stream:
            for element in elements:
                # 采用 长度+对象 的方式进行 "序列化"
                # "长度" 使用 struct 进行 "序列化", "对象" 使用 pickle 进行 "序列化"
                bin_element = pickle.dumps(element)
                length = len(bin_element)
                bin_length = struct.pack(">I", length)  # big-endian unsigned int
                file_stream.write(bin_length)
                file_stream.write(bin_element)
            file_stream.seek(0)
            elements.clear()

            while True:
                bin_length = file_stream.read(4)
                if len(bin_length) == 0:
                    break 
                length = struct.unpack(">I", bin_length)[0]
                bin_element = file_stream.read(length)
                element = pickle.loads(bin_element)
                yield element

    if batch_size is None:
        iterator = iter(iterable)
        sorted_iterator = sorted(iterator, key=key)
        return itertools.groupby(sorted_iterator, key=key)

    streams = [build_disk_stream(sorted_batch) for sorted_batch in sorted_batch_generator()]
    sorted_iterator = heapq.merge(*streams, key=key)
    return itertools.groupby(sorted_iterator, key=key)


# %% 特殊

def accumulate(iterable: Iterable, function: Callable = operator.add, initial: Any = None):
    """ 累计运算, 将 functools.reduce 运算的每一步结果都输出 """
    iterator = iter(iterable)

    if initial is None:
        try:
            total = next(iterator)
        except StopIteration:
            return 
    else:
        total = initial
    
    yield total 

    for element in iterator:
        total = function(total, element)
        yield total 


def pairwise(iterable: Iterable):
    """
    pairwise 有以下一些含义:
        1. 两个集合中的元素 "全部" 两两配对, 也被称为笛卡尔乘积。
            比方说 SQL 中的 cross join 语句 和 itertools 中的 product 函数。
            额外说一下, 我们可以将 SQL 中的 inner join 理解为分组 pairwise。
        2. 两个集合中的元素 "按位" 两两配对
            比方说 zip 函数和 itertools 中的 zip_longest 函数。
        3. 一个集合中 "全部" 元素 两两配对, 我们可以理解为 "排列组合" 中的 2-permutation。
        4. 一个集合中 "相邻" 元素 两两配对, 我们可以理解为 窗口大小为 2 的 "滑窗"。
    这里的 pairwise 指的是第 4 种。
    """

    iterator = iter(iterable)
    left = next(iterator, None)

    for right in iterator:
        yield (left, right)
        left = right 

