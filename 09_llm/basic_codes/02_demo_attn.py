
# %%

import math 

import torch 
from torch import Tensor 

# %%

def basic_attention(query: Tensor, key: Tensor, value: Tensor) -> Tensor: 
    # query, key, value: [num_tokens, token_size]
    # reference: https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py
    token_size = query.size(1)

    """
    第一步, 计算每一个 q 词向量和每一个 k 词向量之间的相关性, 得到 attn_scores 矩阵。
    在 query 和 key 矩阵中, 每一个行向量是一个词向量, 我们分别称为 "q 词向量" 和 "k 词向量"。
    我们将 query 矩阵和 key 矩阵的转置相乘, 得到的 attn_scores 的 shape 是 [num_tokens, num_tokens]。
    此时, attn_scores[i, j] 表示第 i 个 q 词向量和第 j 个 k 词向量之间的相关性。
    attn_scores[i] 表示第 i 个 q 词向量和每一个 k 词向量之间的相关性。
    矩阵乘法视角: "左行向量" 乘以 "右矩阵" 得到 "结果行向量", "结果行向量" 中每一个元素值等于 "左行向量" 点乘 "右矩阵" 的列向量。
    """
    attn_scores = torch.matmul(query, key.transpose(0, 1))  # [num_tokens, num_tokens]
    attn_scores = attn_scores / math.sqrt(token_size)

    """
    第二步, 遍历每一个 q 词向量, 对所有 k 词向量的相关性分数进行 softmax 运算。
    换言之, 将 attn_scores 矩阵中的每一个行向量进行 softmax 运算, 得到 attn_probs 矩阵。
    此时, attn_probs[i] 表示第 i 个 q 词向量对每一个 k 词向量的 "关注度"。
    """
    # softmax(input, dim) = input.exp() / input.exp().sum(dim, keepdims=True)
    # softmax(input, dim) = torch.exp(input - input.max(dim, True)) / torch.exp(input - input.max(dim, True)).sum(dim, True)
    attn_probs = torch.softmax(attn_scores, dim=1)  # [num_tokens, num_tokens]
    attn_probs = torch.dropout(attn_probs, p=0.1, train=False)

    """
    第三步, 遍历每一个 q 词向量, 其和每一个 k 词向量的 "关注度" 作为线性组合的系数, 所有的 v 词向量作为线性组合的向量, 得到 结果词向量。
    矩阵乘法视角: "左行向量" 乘以 "右矩阵" 得到 "结果行向量", "结果行向量" 等于 "右矩阵" 行向量的线性组合, 线性组合的系数是 "左行向量" 的元素值。
    """
    result = torch.matmul(attn_probs, value)  # [num_tokens, token_size]
    return result 

# %%

# %%

def demo_sdap(
        query: Tensor, key: Tensor, value: Tensor, attn_mask: Tensor = None, 
        dropout_p: float = 0.0, is_causal: bool = False, scale: float = None
) -> Tensor:

    # query: [batch_size, num_heads, num_query_tokens, hidden_size]
    # key / value: [batch_size, num_heads, num_kv_tokens, hidden_size]
    # attn_mask: [batch_size, num_kv_tokens]
    batch_size = query.size(0)
    num_query_tokens = query.size(-2)  # tgt_len
    num_kv_tokens = key.size(-2)  # src_len

    if scale is None:
        hidden_size = query.size(-1)
        scale = 1. / math.sqrt(hidden_size)

    # ## step1: 生成 4D 样式的 attn_mask 
    # 0 抛弃, 需要掩码 ==> -inf; 1 保留, 不需要掩码 ==> 0.
    if attn_mask is None:
        attn_mask = torch.ones(batch_size, num_kv_tokens)
    attn_mask = attn_mask[:, None, None, :].float()  # [batch_size, 1, 1, num_kv_tokens]
    if is_causal:
        causal_attn_mask = torch.tril(torch.ones(num_query_tokens, num_kv_tokens))
        causal_attn_mask = causal_attn_mask[None, None, :, :]  # [1, 1, num_query_tokens, num_key_tokens]
        attn_mask = attn_mask * causal_attn_mask
    attn_mask = (1 - attn_mask) * torch.finfo(torch.float32).min
    # 传给 F.scaled_dot_product_attention 的 attn_mask 是这个

    # return torch.nn.functional.scaled_dot_product_attention(
    #     query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=False, scale=scale
    # )

    # ## step2: 计算 attn_scores: [batch_size, num_heads, num_query_tokens, num_kv_tokens]
    attn_scores = torch.matmul(query, key.transpose(-1, -2))
    attn_scores = attn_scores * scale + attn_mask

    # ## step3: 计算 attn_probs: [batch_size, num_heads, num_query_tokens, num_kv_tokens]
    attn_probs = torch.softmax(attn_scores, dim=-1)
    attn_probs = torch.dropout(attn_probs, p=dropout_p, train=True)

    # ## step4: 计算最终输出: [batch_size, num_heads, num_query_tokens, hidden_size]
    output = torch.matmul(attn_probs, value)
    return output 


def standard_sdap(
        query: Tensor, key: Tensor, value: Tensor, attn_mask: Tensor = None, 
        dropout_p: float = 0.0, is_causal: bool = False, scale: float = None
) -> Tensor:

    if attn_mask is not None:
        if is_causal:
            from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

            batch_size = query.size(0)
            num_query_tokens = query.size(2)
            # num_kv_tokens = key.size(2)
            input_shape = (batch_size, num_query_tokens)

            attn_mask = _prepare_4d_causal_attention_mask(attn_mask, input_shape, query, 0)
        else:
            from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask

            attn_mask = _prepare_4d_attention_mask(attn_mask, dtype=query.dtype, tgt_len=query.size(2))

    output = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask, dropout_p=dropout_p, is_causal=False, scale=scale
    )

    return output 

# %%

def check():
    query = torch.randn(2, 3, 5, 32)
    key = torch.randn(2, 3, 5, 32)
    value = torch.randn(2, 3, 5, 32)
    # attn_mask = torch.ones(2, 5)
    # attn_mask[1, 3:] = 0
    # attn_mask = torch.zeros(2, 5)
    # attn_mask = torch.tensor([
    #     [1, 0, 0, 0, 0], 
    #     [1, 0, 0, 0, 0]
    # ])
    attn_mask = torch.randint(low=0, high=2, size=(2, 5)).float()
    attn_mask[:, :1] = 1  # 防止全零的情况出现, 此时会有异常
    print(attn_mask)
    r1 = standard_sdap(query, key, value, attn_mask, is_causal=True)
    r2 = demo_sdap(query, key, value, attn_mask, is_causal=True)
    print(r1.shape, r2.shape, torch.abs(r1 - r2).max(), torch.abs(r1 - r2).mean())


check()
