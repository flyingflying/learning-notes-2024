
# %% 准备

import torch 
from torch import nn, Tensor 
from torch.nn import functional as F 


def is_same_tensor(t1: Tensor, t2: Tensor, eps: float = 1e-8) -> bool:
    return t1.shape == t2.shape and (t1 - t2).abs().max().item() < eps


# %% 线性层求导


def linear(input: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    # input: [*, in_features] 每一个行向量表示一个样本
    # weight: [out_features, in_features] 每一个行向量表示一个线性函数的权重
    # bias: [out_features] 每一个元素表示一个线性函数的偏置
    # reference: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html 
    return torch.matmul(input, weight.transpose(0, 1)) + bias  


def linear_grad(input: Tensor, weight: Tensor, bias: Tensor, output_grad: Tensor) -> Tensor:
    # output_grad: [*, out_features]
    # reference: https://zhuanlan.zhihu.com/p/676212963 
    # 对于一个线性函数而言, bias 的梯度等于所有样本 输出值梯度 之和
    bias_grad = output_grad.flatten(start_dim=0, end_dim=-2).sum(0)
    # 对于一个线性函数而言, weight 的梯度等于每一个样本 输出值梯度 乘以 输入值, 再求和
    weight_grad = torch.matmul(
        output_grad.flatten(start_dim=0, end_dim=-2).transpose(0, 1),
        input.flatten(start_dim=0, end_dim=-2)
    )
    # 对于每一个样本而言, 输入 input 的梯度等于每一个线性函数 输出值梯度 乘以 其 weight, 再求和。
    input_grad = torch.matmul(output_grad, weight)
    return input_grad, weight_grad, bias_grad


def check_linear():
    input = nn.Parameter(torch.randn(3, 4, 10))
    layer = nn.Linear(in_features=10, out_features=20)
    weight = layer.weight
    bias = layer.bias
    output = layer(input)
    output_grad = torch.randn(3, 4, 20)
    output.backward(output_grad)

    with torch.no_grad():
        output_ = linear(input, weight, bias)
        input_grad_, weight_grad_, bias_grad_ = linear_grad(input, weight, bias, output_grad)

        print(is_same_tensor(output, output_))
        print(is_same_tensor(input.grad, input_grad_))
        print(is_same_tensor(weight.grad, weight_grad_))
        print(is_same_tensor(bias.grad, bias_grad_))


check_linear()


# %% 向量卷积


def vector_conv(
        vector: Tensor, kernel: Tensor, 
        padding: int = 0, stride: int = 1, dilation: int = 1
    ) -> Tensor:

    # reference: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html 

    output = F.conv1d(
        input=vector.unsqueeze(0),
        weight=kernel.unsqueeze(0).unsqueeze(0),
        padding=padding, stride=stride, dilation=dilation
    )
    return output.squeeze()


def vector_tconv(
        vector: Tensor, kernel: Tensor, 
        padding: int = 0, stride: int = 1, dilation: int = 1, output_padding: int = 0
    ) -> Tensor:

    # reference: https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d.html 

    output = F.conv_transpose1d(
        input=vector.unsqueeze(0),
        weight=kernel.unsqueeze(0).unsqueeze(0),
        padding=padding, stride=stride, dilation=dilation, output_padding=output_padding
    )
    return output.squeeze()


def matrix_conv(
        matrix: Tensor, kernel: Tensor, 
        padding: int = 0, stride: int = 1, dilation: int = 1
    ) -> Tensor:

    # reference: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html 

    output = F.conv2d(
        input=matrix.unsqueeze(0),
        weight=kernel.unsqueeze(0).unsqueeze(0),
        padding=padding, stride=stride, dilation=dilation
    )
    return output.squeeze()


def matrix_tconv(
        matrix: Tensor, kernel: Tensor, 
        padding: int = 0, stride: int = 1, dilation: int = 1
    ) -> Tensor:

    # reference: https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html 

    output = F.conv_transpose2d(
        input=matrix.unsqueeze(0),
        weight=kernel.unsqueeze(0).unsqueeze(0),
        padding=padding, stride=stride, dilation=dilation
    )
    return output.squeeze()

# %% 向量卷积 循环版本


def vector_conv_loop(
        vector: list, kernel: list, 
        padding: int = 0, stride: int = 1, dilation: int = 1
    ) -> list:

    vector = [0, ] * padding + vector + [0, ] * padding
    kernel_size = dilation * (len(kernel) - 1) + 1
    result = []

    for start_idx in range(0, len(vector), stride):
        end_idx = start_idx + kernel_size
        sub_vector = vector[start_idx:end_idx:dilation]

        if len(sub_vector) < len(kernel):
            break

        result.append(sum(
            [e1 * e2 for e1, e2 in zip(sub_vector, kernel)]
        ))

    return result 


def vector_tconv_loop(vector: list, kernel: list) -> list:
    output_size = len(vector) + len(kernel) - 1
    result = [0 for _ in range(output_size)]

    for idx_v, input_element in enumerate(vector):
        for idx_k, kernel_element in enumerate(kernel):
            result[idx_v+idx_k] += input_element * kernel_element

    return result 


def check_loop_version():
    def is_same_list(list1: list, list2: Tensor, eps: float = 1e-6) -> bool:
        return len(list1) == len(list2) and all([abs(ele1 - ele2) < eps for ele1, ele2 in zip(list1, list2)])

    vector = torch.randn(10)
    kernel = torch.randn(3)

    print(is_same_list(
        vector_conv_loop(vector.tolist(), kernel.tolist()),
        vector_conv(vector, kernel).tolist()
    ))

    print(is_same_list(
        vector_conv_loop(vector.tolist(), kernel.tolist(), padding=2),
        vector_conv(vector, kernel, padding=2).tolist()
    ))

    print(is_same_list(
        vector_conv_loop(vector.tolist(), kernel.tolist(), dilation=2),
        vector_conv(vector, kernel, dilation=2).tolist()
    ))

    print(is_same_list(
        vector_conv_loop(vector.tolist(), kernel.tolist(), stride=2),
        vector_conv(vector, kernel, stride=2).tolist()
    ))

    print(is_same_list(
        vector_conv_loop(vector.tolist(), kernel.tolist(), padding=1, stride=2, dilation=3),
        vector_conv(vector, kernel, padding=1, stride=2, dilation=3).tolist()
    ))

check_loop_version()


# %%


def vector_tconv_loop_v2(
        vector: list, kernel: list, padding: int = 0,
        stride: int = 1, dilation: int = 1, output_padding: int = 0, 
    ) -> list:

    assert len(vector) >= len(kernel)
    assert output_padding < stride or output_padding < dilation

    # step1: 在 vector 每两个元素之间添加 stride - 1 个零
    new_vector = []
    for idx, element in enumerate(vector):
        new_vector.append(element)
        if idx != len(vector) - 1:
            new_vector.extend([0 for _ in range(stride - 1)])
        else:
            new_vector.extend([0 for _ in range(output_padding)])

    # step2: 在 kernel 每两个元素之间添加 dilation - 1 个零
    new_kernel = []
    for idx, element in enumerate(kernel):
        new_kernel.append(element)
        if idx != len(kernel) - 1:
            new_kernel.extend([0 for _ in range(dilation - 1)])

    # step3: 对 vector 进行 full padding 操作
    fp_part = [0 for _ in range(len(new_kernel) - 1)]
    new_vector = fp_part + new_vector + fp_part

    # step4: 对 kernel 进行 "翻转" 操作
    new_kernel = list(reversed(new_kernel))

    # step5: 进行基本卷积运算
    result = []
    for start_idx in range(len(new_vector) - len(new_kernel) + 1):
        end_idx = start_idx + len(new_kernel)
        sub_vector = new_vector[start_idx:end_idx]
        result.append(sum(
            [ele1 * ele2 for ele1, ele2 in zip(sub_vector, new_kernel)]
        ))

    # step6: 删除两端多余的元素
    if padding != 0:
        result = result[padding:-padding]

    return result 


def check_loop_version():
    def is_same_list(list1: list, list2: Tensor, eps: float = 1e-6) -> bool:
        return len(list1) == len(list2) and all([abs(ele1 - ele2) < eps for ele1, ele2 in zip(list1, list2)])

    vector = torch.randn(10)
    kernel = torch.randn(3)

    print(is_same_list(
        vector_tconv_loop_v2(vector.tolist(), kernel.tolist()),
        vector_tconv(vector, kernel).tolist()
    ))

    print(is_same_list(
        vector_tconv_loop_v2(vector.tolist(), kernel.tolist(), padding=2),
        vector_tconv(vector, kernel, padding=2).tolist()
    ))

    print(is_same_list(
        vector_tconv_loop_v2(vector.tolist(), kernel.tolist(), dilation=2),
        vector_tconv(vector, kernel, dilation=2).tolist()
    ))

    print(is_same_list(
        vector_tconv_loop_v2(vector.tolist(), kernel.tolist(), dilation=2, output_padding=1),
        vector_tconv(vector, kernel, dilation=2, output_padding=1).tolist()
    ))

    print(is_same_list(
        vector_tconv_loop_v2(vector.tolist(), kernel.tolist(), stride=2),
        vector_tconv(vector, kernel, stride=2).tolist()
    ))

    print(is_same_list(
        vector_tconv_loop_v2(vector.tolist(), kernel.tolist(), stride=2, output_padding=1),
        vector_tconv(vector, kernel, stride=2, output_padding=1).tolist()
    ))

    print(is_same_list(
        vector_tconv_loop_v2(vector.tolist(), kernel.tolist(), padding=1, stride=2, dilation=3),
        vector_tconv(vector, kernel, padding=1, stride=2, dilation=3).tolist()
    ))

    print(is_same_list(
        vector_tconv_loop_v2(vector.tolist(), kernel.tolist(), padding=1, stride=2, dilation=3, output_padding=2),
        vector_tconv(vector, kernel, padding=1, stride=2, dilation=3, output_padding=2).tolist()
    ))

check_loop_version()

# %% 基础向量卷积


def check_basic_tconv():
    vec = torch.randn(6)
    vec_kernel = torch.randn(3)

    mat = torch.randn(6, 6)
    mat_kernel = torch.randn(3, 3)

    """ 基础的 向量卷积 反向过程 等价于 kernel "翻转" (filp) 的 full padding 卷积 """
    print(is_same_tensor(
        vector_tconv(vec, vec_kernel), 
        vector_conv(vec, torch.flip(vec_kernel, dims=[0, ]), padding=vec_kernel.size(0) - 1),
        eps=1e-6
    ))

    """ 矩阵 "翻转" 和 "转置" 是不同的两个概念 """
    print(is_same_tensor(
        matrix_tconv(mat, mat_kernel),
        matrix_conv(mat, torch.flip(mat_kernel, dims=[0, 1]), padding=mat_kernel.size(0) - 1), 
        eps=1e-6
    ))
    print(is_same_tensor(
        matrix_tconv(mat, mat_kernel),
        matrix_conv(mat, torch.rot90(mat_kernel, k=2), padding=mat_kernel.size(0) - 1), 
        eps=1e-6
    ))
check_basic_tconv()

# %% 卷积层 fold 和 unfold 实现
