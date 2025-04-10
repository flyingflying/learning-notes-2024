
import numpy as np 

import torch 
from torch import nn, Tensor 

import bitsandbytes as bnb  # 0.45.0 
from bitsandbytes import functional as F

"""
absmax 量化:
    对于一组数, 寻找它们绝对值的最大值, 记作 absmax。
    此时, 当最大值乘以 (127 / absmax) 时, 一定会被映射为 -127 或者 127。
    我们将所有的数字乘以 (127 / absmax), 就近取整, 所有的数字都在 [-127, 127] 之间, 可以转换成 int8 类型。
    在 反量化 时, 我们将数字除以 (127 / absmax), 就可以得到原本数字的近似值。
    需要注意的是, 如果这组数中有 离群值, 那么 量化 的效果会非常差。
LLM.int8():
    1. int8_absmax_linear 计算方式:
        (1) input 矩阵和 weight 矩阵都量化, 然后进行 矩阵乘法 (int8 @ int8 -> int32)
        (2) 得到的结果除以 (127 / input_absmax) * (127 / weight_absmax), 即为计算结果
    2. 上述过程的优化方式:
        (1) Vector-wise Quantization
            按行量化 (分块量化), 从而提高计算精度
        (2) Mixed-precision Decomposition
            将 input 矩阵中比 threshold 大的元素值 所在的 列向量 单独拆出来, 进行计算, 从而排除 离群值 的干扰。
    3. 注意事项:
        (1) Mixed-precision Decomposition 的原理是 分块矩阵乘法
        (2) 离群值只检测 input 矩阵, 不检测 weight 矩阵
        (3) 整个过程中, 输入和输出都是 float16, 实际计算都是 float32 (除了 int8 矩阵乘法外)
        (4) 整个项目使用 cuda 有两个目的:
            (a) 解决 int8 矩阵乘法问题, 至少到 2.5 版本的 PyTorch 是不支持整形乘法的
            (b) 算子融合: 量化 和 反量化 的过程可以融合成一次计算, 从而加快运算速度
        (5) 使用 LLM.int8() 之后:
            (a) 模型参数是以 "int8 参数值 + float32 的 absmax" 存在的
            (b) hidden_states 是以 float16 形式存在的 
        (6) 量化算法属于 "时间换空间" 的算法, 虽然 int8 矩阵乘法更快, 但是还需要添加 反量化 的计算时间
"""


def int8_matmul(A: Tensor, B: Tensor) -> Tensor:
    """
    两个 `int8` 类型的矩阵点乘得到 `int32` 类型的矩阵。\n
    out = A @ B.T
   
    源码路径:
        + bitsandbytes/functional.py: `int8_linear_matmul`
        + csrc/pythonInterface.cpp: `cigemmlt_32` -> `igemmlt_32` (Python-C 接口)
        + csrc/ops.cu: `igemmlt` -> `cublasLtMatmul` (调用 cuBLASLt 接口)
    """
    assert A.dtype == torch.int8  # [m, n]
    assert B.dtype == torch.int8  # [k, n]

    # PyTorch 2.5 不支持整型的计算, 使用 NumPy 代替
    # NumPy 不支持 int8 类型的运算直接得到 int32 类型, 所以我们先将 A 和 B 强转成 int32
    np_a = np.astype(A.detach().cpu().numpy(), np.int32)
    np_b = np.astype(B.detach().cpu().numpy(), np.int32)
    np_out = np.matmul(np_a, np_b.swapaxes(-1, -2))
    out = torch.from_numpy(np_out).to(A.device)

    return out  # [m, k]


def int8_vectorwise_quant(A: Tensor, threshold: float = 0.0) -> tuple[Tensor, Tensor, Tensor | None]:
    """
    对矩阵 A 按行量化, 每一个 行向量 执行 absmax 量化, 并排除异常值。

    源码路径:
        + bitsandbytes/functional.py: `int8_vectorwise_quant`
        + csrc/pythonInterface.cpp: `cint8_vector_quant` (Python-C 接口)
        + csrc/ops.cu: `int8VectorQuant` (cuda 主机函数)
        + csrc/kernels.cu: `kInt8VectorQuant` (cuda 核函数)
    """
    assert A.dtype == torch.float16  # [m, n]
    A = A.clone()

    # 1. 检测异常值: 将 A 矩阵中所有大于 0 的元素值都赋值为 0
    outlier_cols = None
    if threshold > 0.0:
        outliers = A.abs() >= threshold

        if outliers.any().item():
            outlier_cols = torch.argwhere(outliers.any(dim=0)).squeeze(dim=-1)
            A[outliers] = 0.0

    # 2. 求每一个 行向量 的 absmax
    row_stats = A.abs().max(dim=-1).values.float()  # [m, ] 

    # 3. 按行量化
    quant_a = A.float() * torch.div(127., row_stats).unsqueeze(-1)  # __fdividef
    # round: 四舍六入五平分, 平分方式为 奇入偶舍 (round-to-nearest-even mode / banker's rounding)
    # TODO: 部分 .5 结尾的浮点数计算会有差异, 原因未知
    quant_a = quant_a.round().to(torch.int8)  # __float2int_rn

    # 4. 按列排除异常值: 每一个列向量中只要有一个值高于 threshold, 就赋值为零
    if outlier_cols is not None:
        quant_a[:, outlier_cols] = 0.0

    return quant_a, row_stats, outlier_cols  # int8, float32, int64


def int8_mm_dequant(A: Tensor, row_stats: Tensor, col_stats: Tensor, bias: Tensor = None) -> Tensor:
    """
    对矩阵 A 进行反量化操作。

    源码路径:
        + bitsandbytes/functional.py: `int8_mm_dequant`
        + csrc/pythonInterface.cpp: `cdequant_mm_int32_fp16` (Python-C 接口)
        + csrc/ops.cu: `dequant_mm_int32_fp16` (cuda 主机函数)
        + csrc/kernels.cu: `kdequant_mm_int32_fp16` (cuda 核函数)
    """
    assert A.dtype == torch.int32  # [m, n]
    assert row_stats.dtype == torch.float32  # [m, ]
    assert col_stats.dtype == torch.float32  # [n, ]
    assert bias is None or bias.dtype == torch.float16  # [n, ]
    assert A.size(0) == row_stats.size(0) and A.size(1) == col_stats.size(0)

    scale = (row_stats.unsqueeze(1) * col_stats.unsqueeze(0)) / (127. ** 2)
    out = A.float() * scale.float()
    if bias is not None:
        out = out + bias 
    out = out.half()
    
    return out


@torch.no_grad()
def int8_absmax_linear(input: Tensor, weight_quant: Tensor, weight_stats: Tensor, bias: Tensor = None, threshold: float = 0.0) -> Tensor:
    """
    int8 absmax 量化线性层代码。注意, 只有 input 需要检测 离群值, weight 不需要检测 离群值, 且需要事先量化完成。

    源码路径:
        + bitsandbytes/autograd/_functions.py: `matmul` -> `MatMul8bitLt.forward`
    """

    # weight 量化: 代码位于 bnb.modules.Int8Params.cuda 中
    # weight_quant, weight_stats, _ = int8_vectorwise_quant(weight)
    assert input.dtype == torch.float16         # [batch_size, in_features]
    assert weight_quant.dtype == torch.int8     # [out_features, in_features]
    assert weight_stats.dtype == torch.float32  # [out_features, ]
    assert bias is None or bias.dtype == torch.float16  # [out_features, ]

    batch_shape = input.shape[:-1]
    input = input.reshape(-1, input.size(-1))

    input_quant, input_stats, outlier_cols = int8_vectorwise_quant(input, threshold=threshold)

    out32 = int8_matmul(input_quant, weight_quant)  # [batch_size, out_features]
    output = int8_mm_dequant(out32, input_stats, weight_stats, bias)

    if outlier_cols is not None:  # Mixed-precision Decomposition
        sub_input = input[:, outlier_cols]  # [batch_size, n_outliers]

        # 反量化
        sub_weight_quant = weight_quant[:, outlier_cols]  # [out_features, n_outliers]
        sub_weight = sub_weight_quant.float().T * weight_stats / 127.

        sub_weight = sub_weight.half()
        # TODO: 唯一一个 float16 运算, 个人认为是 bitsandbytes 中的 bug, 跟踪后续版本的代码
        output = output.addmm(sub_input, sub_weight)

    output = output.reshape(*batch_shape, -1)
    return output 


if __name__ == '__main__':

    def check_int8_absmax_linear():

        with torch.device("cuda:0"):
            input = torch.randn(4, 32).half()
            weight = torch.randn(64, 32).half()
            bias = torch.randn(64).half()
            threshold = 3.0

            weight_quant1, weight_stats1, _ = F.int8_vectorwise_quant(weight)
            state = bnb.MatmulLtState()
            state.has_fp16_weights = False
            output1: Tensor = bnb.matmul(input, weight, state=state, bias=bias, threshold=threshold)

            weight_quant2, weight_stats2, _ = int8_vectorwise_quant(weight)
            output2: Tensor = int8_absmax_linear(input, weight_quant2, weight_stats2, bias, threshold=threshold)

            is_same = torch.all(output1 == output2).item()
            print(is_same, torch.abs(output1 - output2).max().item(), torch.sum(output1 != output2).item())

            bool_idx = weight_quant1 == weight_quant2
            print(bool_idx.all().item(), weight_quant1[~bool_idx].tolist(), weight_quant2[~bool_idx].tolist())
    
    check_int8_absmax_linear()
