
# %%

import torch 
from torch import Tensor 

def _compute_default_rope_parameters(base: float = 10000., dim: int = 128) -> Tensor:
    """ 计算 theta 值, 相关代码位于 transformers.modeling_rope_utils 中 """
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))  # theta 
    return inv_freq  # [hidden_size // 2]

def rope_forward(position_ids: Tensor) -> tuple[Tensor, Tensor]:
    """ 计算公式中的 cos(m * theta) 和 sin(m * theta), 相关代码位于 RotaryEmbedding 中  """

    # ## step1: 计算 theta 值, 一般在 __init__ 中预先算好
    inv_freq = _compute_default_rope_parameters()  # theta
    inv_freq = inv_freq[None, :, None].float()  # [1, hidden_size // 2, 1]

    # ## step2: 计算 m * theta 值, 注意这里的分组方式发生了变化
    position_ids = position_ids[:, None, :].float()  # [batch_size, 1, num_tokens]
    freqs = torch.matmul(inv_freq, position_ids).transpose(1, 2)  # [batch_size, num_tokens, hidden_size // 2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [batch_size, num_tokens, hidden_size]

    # ## step3: 取 cos 和 sin 值
    cos = emb.cos()
    sin = emb.sin()
    return cos, sin  # [batch_size, num_tokens, hidden_size] 

def rotate_half(qk_vec: Tensor):
    """ 将向量的后半部分添加负号, 拼接到前面 """
    hidden_size = qk_vec.size(-1)
    x1 = qk_vec[..., : hidden_size // 2]
    x2 = qk_vec[..., hidden_size // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(query: Tensor, key: Tensor, position_ids: Tensor, unsqueeze_dim: int = 1):
    """ 对 query 和 key 向量实施旋转变换 """
    cos, sin = rope_forward(position_ids)
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (query * cos) + (rotate_half(query) * sin)
    k_embed = (key * cos) + (rotate_half(key) * sin)
    return q_embed, k_embed

# %%

def check():

    from transformers.models.qwen2 import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
    config = Qwen2Config.from_pretrained("/home/lqxu/model-cache/DeepSeek-R1-Distill-Qwen-1.5B")
    rope = Qwen2RotaryEmbedding(config)
    position_ids = torch.arange(20).reshape(2, 10)  # [batch_size, num_tokens]
    gold_cos, gold_sin = rope.forward(torch.randn(1), position_ids)
    test_cos, test_sin = rope_forward(position_ids)
    print(torch.abs(gold_cos - test_cos).max().item(), torch.abs(gold_sin - test_sin).max().item())

    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb as gold_apply_rotary_pos_emb
    query = torch.randn(2, 3, 10, 128)
    key = torch.randn(2, 3, 10, 128)
    gold_query, gold_key = gold_apply_rotary_pos_emb(query, key, gold_cos, gold_sin, position_ids)
    test_query, test_key = apply_rotary_pos_emb(query, key, position_ids)
    print(torch.abs(gold_query - test_query).max().item(), torch.abs(gold_key - test_key).max().item())


check()

# %%
