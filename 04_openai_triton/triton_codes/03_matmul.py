
# %% 设置 ptxas 的路径 

import os 

os.environ["TRITON_PTXAS_PATH"]="/home/lqxu/miniconda3/envs/torch2/lib/python3.10/site-packages/nvidia/cuda_nvcc/bin/ptxas"

# %%

import time

import torch 
from torch import Tensor 

import triton  # type: ignore
import triton.language as tl  # type: ignore

# %%


@triton.jit
def _matmul_kernel(
        # c = a @ b, 其中, a 的 shape 是 [M, K], b 的 shape 是 [K, N], c 的 shape 的 [M, N]
        a_ptr, b_ptr, c_ptr, M, N, K, 
        # a, b, c 三个矩阵不同维度的 stride 值
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, 
        # meta 参数, 必须是 2 的幂
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr = None
):

    # ## step1: 对矩阵 a 和 b 进行分块
    # 一个 program 只负责计算 [BLOCK_SIZE_M, K] @ [K, BLOCK_SIZE_N] = [BLOCK_SIZE_M, BLOCK_SIZE_N],
    # 那么一共有 ceil(M / BLOCK_SIZE_M) * ceil(N / BLOCK_SIZE_N) 个 program
    # 我们记作 (num_programs_m, num_programs_n)
    pid = tl.program_id(axis=0)
    num_programs_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_programs_n = tl.cdiv(N, BLOCK_SIZE_N)

    if GROUP_SIZE_M is None:
        # program: 全局索引 pid, 坐标索引 (pid_m, pid_n), shape 为 [num_programs_m, num_programs_n], row-major
        # 则: pid = pid_m * num_programs_n + pid_n 
        pid_m = pid // num_programs_n
        pid_n = pid % num_programs_n

    else:
        # 现在, 我们沿着 M 维度进行分组, 将多个 block 分入一组中, 那么每一组有 GROUP_SIZE_M * num_programs_n 个 block
        num_programs_in_group = GROUP_SIZE_M * num_programs_n
        group_id = pid // num_programs_in_group  # 当前 program 所在 group 的 id
        first_pid_m = group_id * GROUP_SIZE_M  # 当前 group 第一个 program 的 M 维度索引
        group_size_m = min(num_programs_m - first_pid_m, GROUP_SIZE_M)  # 修正最后一组的 group_size_m

        # 第 0 组: pid = pid_n * group_size_m + pid_m
        # 其它组: pid = pid_n * group_size_m + (pid_m - first_pid_m) + (num_programs_in_group * group_id)
        # 那么: pid - (num_programs_in_group * group_id) = pid * group_size_m + (pid_m - first_pid_m)
        pid = pid - num_programs_in_group * group_id
        pid_m = pid % group_size_m + first_pid_m
        pid_n = pid // group_size_m

    # ## step2: 采用 向量化编程 的方式计算 坐标索引 (offsets)
    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)

    # ## step3: 采用 向量化编程 的方式计算 指针 (ptrs)
    # 接下来, 我们需要进行进一步的分块, 采用 for 循环的方式, 
    # 每一次循环计算 [BLOCK_SIZE_M, BLOCK_SIZE_K] @ [BLOCK_SIZE_K, BLOCK_SIZE_N], 然后结果相加即可。
    # 那么, 这里的 a_ptrs 是第一个 [BLOCK_SIZE_M, BLOCK_SIZE_K], 后续的再移动即可
    # 同理, 这里的 b_ptrs 是第一个 [BLOCK_SIZE_K, BLOCK_SIZE_N], 后续的再移动即可
    a_ptrs = a_ptr + (offsets_m[:, None] * stride_am + offsets_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offsets_k[:, None] * stride_bk + offsets_n[None, :] * stride_bn)

    # ## step4: 循环计算
    c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a_masks = (offsets_k[None, :] < K - i * BLOCK_SIZE_K) & (offsets_m[:, None] < M)
        a = tl.load(a_ptrs, mask=a_masks, other=0.0)

        b_masks = (offsets_k[:, None] < K - i * BLOCK_SIZE_K) & (offsets_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_masks, other=0.0)

        c += tl.dot(a, b, allow_tf32=False)
        # c += tl.sum(a[:, None, :] * tl.trans(b)[None, :, :], axis=2)  # 可以运行, 但是编译时间特别长

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    # ## step5: 存储
    c_ptrs = c_ptr + (offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn)
    c_masks = (offsets_m[:, None] < M) & (offsets_n[None, :] < N)
    tl.store(c_ptrs, c, c_masks)


def matmul(input1: Tensor, input2: Tensor, use_group: bool = False) -> Tensor:

    assert input1.is_cuda and input2.is_cuda, "only supported for cuda device."
    assert input1.size(1) == input2.size(0), "the shape of two matrices are mismatched."

    M, K = input1.shape 
    K, N = input2.shape 

    output = torch.empty((M, N), device=input1.device)

    def cal_programs_shape(meta):
        num_pid_m = triton.cdiv(M, meta["BLOCK_SIZE_M"])
        num_pid_n = triton.cdiv(N, meta["BLOCK_SIZE_N"])
        num_pid = num_pid_m * num_pid_n
        return (num_pid, )  # 返回的一定要是元组, 不能是数字

    if use_group:
        _matmul_kernel[cal_programs_shape](
            input1, input2, output, M, N, K, 
            input1.stride(0), input1.stride(1), input2.stride(0), input2.stride(1), output.stride(0), output.stride(1),
            BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64, num_stages=2
        )
    else:
        _matmul_kernel[cal_programs_shape](
            input1, input2, output, M, N, K, 
            input1.stride(0), input1.stride(1), input2.stride(0), input2.stride(1), output.stride(0), output.stride(1),
            BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64, GROUP_SIZE_M=8, num_stages=2
        )

    return output 


def check_matmul():

    def get_max_diff(t1: Tensor, t2: Tensor):
        assert t1.shape == t2.shape 
        max_diff = (t1 - t2).abs().max().cpu().item()
        return max_diff
    
    def single_exam(M, N, K, step = 1, need_transpose = False, use_group = True, print_tensor = False, eps: float = 1e-4):

        if need_transpose:
            input1 = torch.randn(K, M).cuda().transpose(0, 1)
            input2 = torch.randn(N, K).cuda().transpose(0, 1)
        else:
            input1 = torch.randn(M, K).cuda()
            input2 = torch.randn(K, N).cuda()

        if step != 1:
            input1 = input1[::2, ::2]
            input2 = input2[::2, ::2]
        
        result1 = matmul(input1, input2, use_group=use_group)
        result2 = torch.matmul(input1, input2)

        if print_tensor:
            print(result1)
            print(result2)

        max_diff = get_max_diff(result1, result2)

        print(
            f"M={M}, N={N}, K={K}, step={step}, need_transpose={need_transpose}, use_group={use_group}:", 
            f"max_diff={max_diff}", 
            "✅" if max_diff < eps else "❌"
        )
    
    # 方阵测试

    from itertools import product

    for need_transpose_, use_group_ in product([False, True], [False, True]):
        single_exam(64, 64, 64, need_transpose=need_transpose_, use_group=use_group_)
        single_exam(128, 128, 128, 2, need_transpose=need_transpose_, use_group=use_group_)

        # 向量的 size 过大时, 误差也会增加
        single_exam(1024, 1024, 1024, need_transpose=need_transpose_, use_group=use_group_, eps=1e-2)
        single_exam(1024, 1024, 512, need_transpose=need_transpose_, use_group=use_group_)

        single_exam(1025, 1026, 127, need_transpose=need_transpose_, use_group=use_group_)
        single_exam(65, 65, 64, need_transpose=need_transpose_, use_group=use_group_)
        single_exam(64, 64, 10, need_transpose=need_transpose_, use_group=use_group_)
        single_exam(65, 65, 65, need_transpose=need_transpose_, use_group=use_group_)


if __name__ == "__main__":
    check_matmul()
