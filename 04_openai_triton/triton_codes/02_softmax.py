
import os 

os.environ["TRITON_PTXAS_PATH"]="/home/lqxu/miniconda3/envs/torch2/lib/python3.10/site-packages/nvidia/cuda_nvcc/bin/ptxas"

import torch 
from torch import Tensor 

import triton  # type: ignore
import triton.language as tl  # type: ignore


# %%

@torch.jit.script
def torch_row_softmax(input: Tensor):
    assert input.ndim == 2
    return torch.softmax(input, dim=1)


@torch.jit.script
def native_row_softmax(input: Tensor):
    assert input.ndim == 2
    output = input - input.max(dim=1, keepdim=True)[0]
    output = torch.exp(output)
    output = output / output.sum(dim=1, keepdim=True)
    return output 


# %%

"""
如果用 PyTorch 实现, 那么每一步运算都会先从 HBM 中读取数据, 然后再存储到 HBM 中, 我们将这种方式称为 native softmax。
但是, 我们知道, 从 HBM 中读取和存储数据都是非常缓慢的, 我们希望将计算的中间变量保存到 shared memory 中, 以减少数据读取的时间。
然而 shared memory 的空间是有限的, 一般都是 KB 级别的。GeForce RTX 2060 上才 48KB 的 shared memory, 最多可以存放 48 * 1024 / 4 = 12,288 个单精度浮点数。
那么, 如果向量的元素超过 12,288 个, 那么只能采用 native softmax 的方式。

OneFlow 在其 [博客](https://zhuanlan.zhihu.com/p/341059988) 中给出的解决方案是:

1. 如果向量的元素个数小于 1024, 那么一个 warp 处理一个向量, 此时效率最高;
2. 如果向量的元素个数在 1024 到 4096 之间, 那么将向量预先加载到 shared memory 中, 一个 thread block 处理一个向量, 和刚刚所说的想法一致;
3. 如果向量的元素个数大于 4096, 依旧是一个 thread block 处理一个向量, 只是会将中间的计算结果保存到 HBM 中, 和 native softmax 差不多。

简单来说, 我们根据向量的元素个数来选择不同的计算策略, 这和排序算法的思想是一致的 (我们根据数组的大小选择不同的排序算法)。

OpenAI Triton 采用的策略应该是相似的, 只是上述过程变成 "自动" 的, 不需要人手动调整。
"""

"""
矩阵 在计算机底层就是 一维数组。在 C 语言中, 我们一般用 ptr, size/shape 和 stride 三个内容来描述一个数组。
row stride: 矩阵中元素的行索引加 1 时, 其在 一维数组 中索引的增加量。
column stride: 矩阵中元素的列索引加 1 时, 其在 一维数组 中索引的增加量。
"""

"""
这里实现的是 row softmax, 即对矩阵的每一个 行向量 进行 softmax 运算。
实现方式如下: 一个 program 处理一个 行向量, 不对一个 行向量 进行拆分! 一个 行向量 的拆分方式交给 Triton 编译器执行。

在 softmax 运算中, 包含了 max 和 sum 两个 reduce 运算, 以及 减法 和 除法 两个 boardcast 运算, 想分块还是比较麻烦的。
"""


@triton.autotune(configs=[triton.Config({}, num_warps=2 ** i) for i in range(0, 6)], key=["BLOCK_SIZE_N"])
@triton.jit 
def _row_softmax_kernel(
        # output = row_softmax(input), input 和 output 的 shape 都是 [M, N]
        input_ptr, output_ptr, M: int, N: int, 
        stride_input_m: int, stride_input_n: int, stride_output_m: int, stride_output_n: int, 
        BLOCK_SIZE_N: tl.constexpr,  # BLOCK_SIZE_N = triton.next_power_of_2(N)
    ):

    # 一个 program 处理一个 行向量, 因此 pid 就是元素的 "行坐标索引"
    pid = tl.program_id(axis=0)

    # 计算元素的 "列坐标索引"
    offsets_n = tl.arange(0, BLOCK_SIZE_N)

    # 计算元素的 input "指针"
    input_ptrs = input_ptr + (pid * stride_input_m + offsets_n * stride_input_n)
    mask = offsets_n < N

    # 加载数据
    vector = tl.load(input_ptrs, mask=mask, other=-float("inf"))

    # 计算
    vector = vector - tl.max(vector)
    vector = tl.exp(vector)
    vector = vector / tl.sum(vector)

    # 计算元素的 output "指针"
    output_ptrs = output_ptr + (pid * stride_output_m + offsets_n * stride_output_n)

    # 存储数据到 HBM 中
    tl.store(output_ptrs, vector, mask=mask)



def triton_softmax(input: Tensor):
    assert input.ndim == 2

    M, N = input.shape 
    BLOCK_SIZE_N = triton.next_power_of_2(N)
    assert BLOCK_SIZE_N < tl.TRITON_MAX_TENSOR_NUMEL

    output = torch.empty_like(input)

    _row_softmax_kernel[(M, )](
        input, output, M, N, 
        input.stride(0), input.stride(1), output.stride(0), output.stride(1), 
        BLOCK_SIZE_N
    )

    return output 


def is_same_tensor(t1: Tensor, t2: Tensor, eps: float = 1e-6):
    return (t1 - t2).abs().max().cpu().item() < eps 


def check_softmax_func():
    device = torch.device("cuda:0")
    input = torch.randn(10, 2047).to(device)
    output = triton_softmax(input)

    print(is_same_tensor(output, torch_row_softmax(input)))
    print(is_same_tensor(output, native_row_softmax(input)))

    input = torch.randn(10, 2048).to(device).T
    output = triton_softmax(input)

    print(is_same_tensor(output, torch_row_softmax(input)))
    print(is_same_tensor(output, native_row_softmax(input)))


check_softmax_func()
