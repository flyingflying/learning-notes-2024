
# %%

"""
设置 PTXAS 路径

pip install nvidia-cuda-nvcc-cu11

Reference: https://github.com/pytorch/pytorch/issues/119054 
"""

import os 

os.environ["TRITON_PTXAS_PATH"]="/home/lqxu/miniconda3/envs/torch2/lib/python3.10/site-packages/nvidia/cuda_nvcc/bin/ptxas"

# %%

import json 
from itertools import product

import torch
from torch import Tensor 

import triton  # 只能用于 主机函数  # type: ignore
import triton.language as tl  # 只能用于 核函数  # type: ignore

# %%

# reference: https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html 

@triton.jit
def _vector_add_kernel(
        # c = a + b, 其中, a, b 和 c 的 shape 都是 [M, ]
        a_ptr, b_ptr, c_ptr, M,
        stride_a, stride_b, stride_c, 
        BLOCK_SIZE: tl.constexpr
):
    
    """ 每一个 program 处理 [BLOCK_SIZE, ] 大小的数据 """

    # pid: 当前 program 的索引值
    pid = tl.program_id(axis=0)

    # offsets: 当前 program 需要处理元素的 坐标索引
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # ptrs: 当前 program 需要处理元素的内存地址
    a_ptrs = a_ptr + offsets * stride_a
    b_ptrs = b_ptr + offsets * stride_b 
    
    # 加载数据并计算
    a = tl.load(a_ptrs, mask=offsets < M, other=0.0)
    b = tl.load(b_ptrs, mask=offsets < M, other=0.0)
    c = a + b 

    # 保存数据
    c_ptrs = c_ptr + offsets * stride_c
    tl.store(c_ptrs, c, mask=offsets < M)


def vector_add(
        input1: Tensor, input2: Tensor, block_size: int = 1024,
        print_ir: bool = False, ir_type: str = "llir", print_meta: bool = False
) -> Tensor:

    assert input1.is_cuda and input2.is_cuda
    assert input1.ndim == 1 and input2.ndim == 1 and input1.size(0) == input2.size(0)

    vector_size = input1.size(0)
    output = torch.zeros_like(input1)

    block_size = triton.next_power_of_2(block_size)
    programs_size = triton.cdiv(vector_size, block_size)

    compiled_kernel: triton.compiler.CompiledKernel = _vector_add_kernel[(programs_size, )](
        input1, input2, output, vector_size, input1.stride(0), input2.stride(0), output.stride(0), block_size 
    )

    if print_ir:
        print("Triton GPU IR codes of add kernel are shown below:", )
        print(compiled_kernel.asm[ir_type])
        print()

    if print_meta:
        print("The meta parameters of add kernel are shown below:",)

        while True:  # 等待程序运行完成
            if isinstance(compiled_kernel.metadata, dict):
                print(json.dumps(compiled_kernel.metadata, ensure_ascii=False, indent=4, skipkeys=True))
                break 
    
    return output 


def check_add_func():
    input1 = torch.randn(20000).to("cuda:0")[::2]
    input2 = torch.randn(20000).to("cuda:0")[::2]
    output = vector_add(input1, input2, print_ir=True, print_meta=True).cpu() 
    gold_output = torch.add(input1, input2).cpu()
    max_diff = (output - gold_output).abs().max().item()

    print("The max absolute difference between output and gold_output is ", max_diff)


# %%


def find_best_meta_params(vector_size):

    can_block_size = [2 ** i for i in range(0, 14)]  # [1, 8192] 之间所有 2 的幂

    """
    num_warps: 
        一个 thread block 中的 thread warp 的数量, Triton 中设置必须是 2 的幂。
        由于一个 warp 固定有 32 个 thread, 而一个 block 最多有 1024 个 thread, 因此 warp 的数量最大是 32!

    num_stages: 
        官方说法:
            在 安培架构 (8.0+) 中, 英伟达引入了新的 异步拷贝 (asynchronous copy) 指令。
            Triton 编译器根据这些指令会将 核函数 中的 for 循环 "流水线" 化。num_stages 指的就是 "流水线" 的深度。
        我是这么理解的 (不一定对):
            如果 for 循环中有 "数据加载" 和 "计算" 两个任务, 那么在第一次循环 "数据加载" 任务结束时, 
            我们就可以开始下一次循环的 "数据加载" 任务, 而不用等到第一次循环 "计算" 任务结束时, 再开始下一次循环的 "数据加载" 任务。
            而 num_stages 指的就是 数据预加载 的最大次数: 即某一次循环 "数据加载" 任务结束后, 最多提前加载在这之后 num_stages 次循环的数据。
        References:
            1. https://github.com/openai/triton/discussions/512
            2. https://github.com/openai/triton/issues/2077 
        从测试效果来看, num_stages 没有大小限制, 只要是个整数就行, 官方教程中只测试了 3, 4, 5, 这里也只测试这三个数字。

    num_ctas:
        CTA (Cooperative Thread Array) 指的就是 线程块, 是其的另外一个名称。
        num_ctas 指的是一个 block cluster 中 block 的数量, 只有 9.0+ 的架构中才能使用。
        完整的线程层级: grid -> (thread block cluster) -> thread block / CTA -> (thread warp) -> thread 
    """
    can_num_warps = [2 ** i for i in range(0, 6)]  # [1, 32] 之间所有 2 的幂
    can_num_stages = [3, 4, 5]

    configs = []
    for block_size, num_warps, num_stages in product(can_block_size, can_num_warps, can_num_stages):
        configs.append(
            triton.Config({"BLOCK_SIZE": block_size}, num_warps=num_warps, num_stages=num_stages)
        )

    _autotune_vector_add_kernel = triton.autotune(
        configs=configs,  # 所有可能的 meta 参数配置
        key=["M", ]  # 当 M 发生变化时, 自动寻找, 如果不发生变化, 保持不变
    )(_vector_add_kernel)

    input1 = torch.randn(vector_size, ).cuda()
    input2 = torch.randn(vector_size, ).cuda()
    output = torch.empty_like(input1)

    def cal_programs_shape(meta):
        num_programs = triton.cdiv(vector_size, meta["BLOCK_SIZE"])
        return (num_programs, )

    _autotune_vector_add_kernel[cal_programs_shape](
        input1, input2, output, vector_size, 
        input1.stride(0), input2.stride(0), output.stride(0)
    )

    print(f"Best config of meta parameters for num_elements={vector_size} are shown below: ",)
    print(_autotune_vector_add_kernel.best_config)


# %%


def check_performance():
    def benchmark_func(vector_size: int, provider: str):
        # step1: 构建输入
        input1 = torch.randn(vector_size, ).float().to("cuda:0")
        input2 = torch.randn(vector_size, ).float().to("cuda:0")

        # step2: 测试运行时间
        quantiles = [0.5, 0.2, 0.8]
        if provider == "torch":
            # do_bench 中涉及到的时间都是 毫秒 (milliseconds), 而 1s=1000ms 
            # warmup 指的是大概的预热时间 (ms), rep 指的是大概的测试时间 (ms), 两者的原理如下:
            #   do_bench 中会先将 fn 函数跑 5 次, 然后计算 5 次的平均时间, 得到一个大致的运行时间 estimate_time
            #   接下来, 用 warmup 除以 estimate_time 得到 warmup 的次数, 用 rep 除以 estimate_time 得到 实际测试 的次数
            # quantiles 和 return_mode 设置一个即可, 两者的原理如下:
            #   假设一共测试了一万次, 我们得到一万次的运行时间, return_mode 表示我们用什么 "集中趋势" 来描述这一万次运行时间, 可以是 平均数, 中位数, 最大值 和 最小值
            #   如果设置 quantiles, 那么 return_mode 就失效了, 返回 一万次运行时间对应的分位点, 这里用 20 百分位点作为 最小值, 80 百分位点作为 最大值, 50 百分位点作为 实际值
            real_time_ms, min_time_ms, max_time_ms = triton.testing.do_bench(fn=lambda: torch.add(input1, input2), warmup=50, rep=1000, quantiles=quantiles)
        if provider == "triton":
            real_time_ms, min_time_ms, max_time_ms = triton.testing.do_bench(fn=lambda: vector_add(input1, input2), warmup=50, rep=1000, quantiles=quantiles)

        def cal_gbps(time_ms):
            # 计算处理性能的 GB/s
            # 一次计算涉及到 3 个 float32 张量, 一个 float32 占 4 个 Bytes, 3 个张量一共 3 * 4 * num_elements 个 Bytes, 然后乘以 1e-9 转换成 GB
            # 1ms = 1e-3 s, 因此性能计算公式为 (3 * 4 * num_elements * 1e-9) / (time * 1e-3) = (12 * num_elements * 1e-6) / time        
            return 12 * vector_size * 1e-6 / time_ms

        return cal_gbps(real_time_ms), cal_gbps(max_time_ms), cal_gbps(min_time_ms)


    triton.testing.perf_report(benchmarks=[
        triton.testing.Benchmark(
            x_names=["vector_size"], x_vals=[2 ** i for i in range(12, 28, 1)], x_log=True, 
            line_arg="provider", line_vals=["triton", "torch"], line_names=["Triton", "PyTorch"], styles=[("blue", "-"), ("green", "-")],
            ylabel="GB/s", plot_name="vector-add performance", args={}
        ),
    ])(benchmark_func).run(print_data=True)


# %%


@triton.jit
def _random_kernel(
        vec_ptr, vec_size, vec_stride, seed, BLOCK_SIZE: tl.constexpr
    ):

    pid = tl.program_id(axis=0)

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) 

    vec = tl.rand(seed, offsets)  # 根据 seed 和 offset 生成随机数!

    tl.store(vec_ptr + offsets * vec_stride, vec, mask=offsets < vec_size)


def random(vec_size, seed):
    output = torch.randn(vec_size).cuda()

    def cal_program_size(meta: dict):
        num_programs = triton.cdiv(vec_size, meta["BLOCK_SIZE"])
        return (num_programs, )

    _random_kernel[cal_program_size](output, vec_size, output.stride(0), seed, 8)

    return output


def check_random():
    result1 = random(10000, 42)
    result2 = random(10000, 42)

    print((result1 - result2).abs().max().cpu().item())

# %%


if __name__ == "__main__":
    # check_add_func()
    # find_best_meta_params(100000)
    # check_performance()
    check_random()
