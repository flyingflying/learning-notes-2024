
"""
实现任意层的循环

之前一直在思索, 任意层的循环应该怎么实现。最近, 借助 ChatGPT, 想通了这个问题, 特此记录一下。

核心思想是一致的, 那就是构建 "全连接树"!

我们将每一个 迭代器 作为 "树结构" 的一层, 迭代器 中的每一个元素作为一个 "树结点", 然后相邻两层的所有结点 两两相连, 构成一个 "全连接树"。

构建完成后, 我们就可以采用 BFS 或者 DFS 算法来实现迭代!
"""

# %%

import itertools
from typing import Iterable

# %%

"""
第一种方式是 递归 (recursion), 非常标准的 BFS 实现方案, 如果不理解的可以看程序的输出。

需要注意的是, 这里采用 yield 生成器 + 递归 的方式实现的, yield 抛出的不一定给主程序, 还有可能在递归中被消耗。

如果有 n 个迭代器, 那么 递归深度 就是 n+1, 只要 迭代器 的数量不是特别多, 就不会出现 栈溢出 的问题。

主要问题: 循环次数过多! 我们用 c_i 表示第 i 个 迭代器 的元素个数, 那么一共要进行 n * c_1 * c_2 * ... * c_n 次循环, 比用代码写的循环多 n 倍!
"""


def recursion_nested_loop(iterators: list[Iterable], debug: bool = False):
    if len(iterators) == 0:
        yield tuple()
        return 

    cur_iterator = iterators[0]
    for cur_value in cur_iterator:
        cur_value = (cur_value, )
        for nested_value in recursion_nested_loop(iterators[1:], debug):
            if debug:  # 输出每一次 yield 的结果, 更容易理解程序的运行方式
                print(cur_value + nested_value)
            yield cur_value + nested_value


def check_recursion():
    iterators = [range(2), range(3), range(4), range(5)]
    result1 = list(recursion_nested_loop(iterators))
    result2 = list(itertools.product(*iterators))

    print(result1)
    print(result1 == result2)

    iterators = ["ABC", "DEF", "GHI"]
    for _ in recursion_nested_loop(iterators, True):
        print("===")


check_recursion()
# %%

"""
第二种方案是用 stack 实现的 DFS。和 递归 的方案相比, 没有 递归深度 的限制, 循环次数也更少。

一共进行了 c_1 + c_1 * c_2 + c_1 * c_2 * c_3 + ... + c_1 * c_2 * c_3 * ... * c_n 次 for 循环,

我们可以简写为 sum(cumprod(vec_c)), 其中 vec_c 是由 c_1, c_2, ..., c_n 构成的向量。

for 循环的次数就是往 stack 中添加元素的次数, 而 while 循环的次数应该和 stack 中的元素个数是相同的, 

因此, while 循环的次数为 sum(cumprod(vec_c)) + 1, 其中 1 表示 root 结点。

那么, while 循环 + for 循环一共进行了 2 * sum(cumprod(vec_c)) + 1 次。

递归方案的循环次数可以写成 n * prod(vec_c), 从直观上感觉, 肯定比 stack 方案的循环次数多 (两者都可以理解为 n 项求和)。

直接手写 方案的循环次数是 prod(vec_c), 是目前循环次数最少的方案!

stack 方案的最大问题是: 得到的结果是倒序的, 不是正序的。在 for 循环迭代时, 需要添加 reversed 操作 (Iterable 对象不一定能进行 reversed 操作)。
"""


def dfs_nested_loop(iterators: list[Iterable], debug: bool = False):
    count = 0
    stack = [(tuple(), 0), ]
    max_depth = len(iterators)

    while stack:
        if debug:  # 输出当前 stack 的信息, 更容易理解运行方式
            print([result for result, _ in stack])

        cur_result, cur_depth = stack.pop()

        if cur_depth == max_depth:
            yield cur_result
            continue

        next_depth = cur_depth + 1
        for item in reversed(iterators[cur_depth]):
            count += 1
            next_result = cur_result + (item, )
            stack.append((next_result, next_depth))

    if debug:
        print("the number of for loop is", count)


def check_dfs():
    iterators = [range(2), range(3), range(4), range(5)]
    result1 = list(dfs_nested_loop(iterators))
    # result1 = list(reversed(result1))
    result2 = list(itertools.product(*iterators))

    print(result1)
    print(result1 == result2)

    iterators = ["ABC", "DEF", "GHI"]
    for _ in dfs_nested_loop(iterators, True):
        pass 


check_dfs()
# %%

"""
第三种方案, 基于 queue 实现的 BFS。和第二种方案相比, 代码几乎没有发生变化, 主要变化有两个:

1. stack 变成了 queue, 抛出元素的方式从 `pop()` 变成了 `pop(0)`;
2. for 循环时不需要进行 reversed 操作。

问题: queue 中的元素个数过多。显然, queue 中元素个数最多为 prod(vec_c) 个, 此时 嵌套循环 的所有可能都在其中。

而第二种方案中, stack 中元素个数最多为 sum(vec_c) - n + 1, 此时处于第一次 yield 抛出前夕。
"""


def bfs_nested_loop(iterators: list[Iterable], debug: bool = False):
    count = 0
    queue = [(tuple(), 0), ]
    max_depth = len(iterators)

    while queue:
        if debug:  # 输出当前 queue 的信息, 更容易理解运行方式
            print([result for result, _ in queue])

        cur_result, cur_depth = queue.pop(0)

        if cur_depth == max_depth:
            yield cur_result
            continue

        next_depth = cur_depth + 1
        for item in iterators[cur_depth]:
            count += 1
            next_result = cur_result + (item, )
            queue.append((next_result, next_depth))

    if debug:
        print("the number of for loop is", count)


def check_bfs():
    iterators = [range(2), range(3), range(4), range(5)]
    result1 = list(bfs_nested_loop(iterators))
    result2 = list(itertools.product(*iterators))

    print(result1)
    print(result1 == result2)

    iterators = ["ABC", "DEF", "GHI"]
    for _ in bfs_nested_loop(iterators, True):
        pass 


check_bfs()

# %%

"""
总结一下, 第二种方案最佳, 内存消耗 和 循环次数 都是最佳的, 只是需要 reversed 操作。

在计算机中, "树结构" 真的是无处不在啊!
"""