
import torch 
from torch.nn import functional as F 
from torch import nn, Tensor, LongTensor

from transformers.models.qwen2 import Qwen2Config


class Qwen2RMSNorm(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config

        self.weight = nn.Parameter(torch.ones(config.hidden_size))
        self.variance_epsilon = config.rms_norm_eps

    def forward(self, hidden_states: Tensor):
        # 当 mean=0 时, variance 等价于 root mean square (RMS)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states


class Qwen2MLP(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config

        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: Tensor):
        x = F.silu(self.gate_proj(x)) * self.up_proj(x)  # SwiGLU 激活函数
        x = self.down_proj(x)
        return x 


class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config

        base = config.rope_theta
        if hasattr(config, "head_dim"):
            head_size = config.head_dim 
        else:
            head_size = config.hidden_size // config.num_attention_heads

        # 代码位于 transformers.modeling_rope_utils._compute_default_rope_parameters 中
        self.inv_freq = 1.0 / (base ** (torch.arange(0, head_size, 2).float() / head_size))  # theta 

    @torch.no_grad()
    def forward(self, position_ids: Tensor):
        inv_freq_expanded = self.inv_freq[None, :, None]  # [1, head_size // 2, 1]
        position_ids_expanded = position_ids[:, None, :]  # [batch_size, 1, num_tokens]

        freqs = torch.matmul(inv_freq_expanded, position_ids_expanded).transpose(1, 2)  # [batch_size, num_tokens, head_size // 2]
        emb = torch.cat((freqs, freqs), dim=-1)  # 注意: 分组方式改变了 [batch_size, num_tokens, head_size]
        return emb.cos(), emb.sin()


def rotate_half(x: Tensor) -> Tensor:  # 辅助 apply_rotary_pos_emb
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
        q: Tensor,  # [batch_size, num_heads, num_tokens, head_size]
        k: Tensor,  # [batch_size, num_heads, num_tokens, head_size]
        position_embeddings: tuple[Tensor, Tensor],  # [batch_size, num_tokens, head_size]
        unsqueeze_dim: Tensor = 1
    ) -> tuple[Tensor, Tensor]:

    cos_table, sin_table = position_embeddings
    cos_table = cos_table.unsqueeze(unsqueeze_dim)
    sin_table = sin_table.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos_table) + (rotate_half(q) * sin_table)
    k_embed = (k * cos_table) + (rotate_half(k) * sin_table)
    return q_embed, k_embed


def repeat_kv(hidden_states: Tensor, n_rep: int) -> Tensor:  # 辅助 eager_attention_forward 中的 GQA
    if n_rep == 1:
        return hidden_states

    batch_size, num_kv_heads, num_kv_tokens, head_size = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :]
    hidden_states = hidden_states.expand(batch_size, num_kv_heads, n_rep, num_kv_tokens, head_size)
    return hidden_states.reshape(batch_size, num_kv_heads * n_rep, num_kv_tokens, head_size)


def eager_attention_forward(
        query: Tensor,  # [batch_size, num_query_heads, num_query_tokens, head_size]
        key: Tensor,    # [batch_size, num_kv_heads, num_kv_tokens, head_size]
        value: Tensor,  # [batch_size, num_kv_heads, num_kv_tokens, head_size]
        attn_mask: Tensor,  # [batch_size, 1, num_query_tokens, num_kv_tokens]
        scaling: float,
        num_kv_in_group: int = 1,
        dropout: float = 0.0
    ) -> Tensor:

    key_states = repeat_kv(key, num_kv_in_group)
    value_states = repeat_kv(value, num_kv_in_group)

    # [batch_size, num_heads, num_query_tokens, num_key_tokens]
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    attn_weights = attn_weights + attn_mask[:, :, :, :key_states.size(-2)]

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=True)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output


class Qwen2Attention(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config

        self.head_dim = config.hidden_size // config.num_attention_heads  # num_query_heads
        self.num_kv_in_group = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(
            self,
            hidden_states: Tensor,
            position_embeddings: tuple[Tensor, Tensor],
            attention_mask: Tensor,
            past_key_value: tuple[Tensor, Tensor] = None,
        ) -> tuple[Tensor, tuple[Tensor, Tensor]]:

        batch_size, num_tokens, _ = hidden_states.shape 
        hidden_shape = (batch_size, num_tokens, -1, self.head_dim) 

        query_states = self.q_proj.forward(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj.forward(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj.forward(hidden_states).view(hidden_shape).transpose(1, 2)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, position_embeddings)

        if past_key_value is not None:
            # 相关代码位于 transformers.cache_utils.DynamicCache.update 中
            key_states = torch.cat([past_key_value[0], key_states], dim=-2)
            value_states = torch.cat([past_key_value[1], value_states], dim=-2)

        attn_output = eager_attention_forward(
            query_states, key_states, value_states, attention_mask, self.scaling,
            self.num_kv_in_group, self.attention_dropout
        )

        attn_output = attn_output.reshape(batch_size, num_tokens, -1).contiguous()
        attn_output = self.o_proj.forward(attn_output)
        return attn_output, (key_states, value_states)


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen2Attention(config)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config)
        self.post_attention_layernorm = Qwen2RMSNorm(config)

    def forward(
            self,
            hidden_states: Tensor,
            attention_mask: Tensor,
            position_embeddings: tuple[Tensor, Tensor],
            past_key_value: tuple[Tensor, Tensor] = None
        ) -> tuple[Tensor, tuple[Tensor, Tensor]]:

        # Attention Block
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, next_past_key_value = self.self_attn.forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value
        )
        hidden_states = residual + hidden_states

        # MLP Block
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, next_past_key_value


class Qwen2Model(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config 

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)

    def forward(
            self,
            input_ids: LongTensor,  # [batch_size, num_query_tokens]
            attention_mask: LongTensor = None,  # [batch_size, num_kv_tokens]
            position_ids: LongTensor = None,  # [batch_size, num_query_tokens]
            past_key_values: list[tuple[Tensor, Tensor]] = None,  # [batch_size, num_heads, past_seen_tokens, head_dim]
        ) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:

        num_query_tokens = input_ids.size(1)
        inputs_embeds: Tensor = self.embed_tokens(input_ids)  # [batch_size, num_query_tokens, hidden_size]

        if past_key_values is None:
            past_seen_tokens = 0
        else:
            past_seen_tokens = past_key_values[0][0].size(-2)
        num_kv_tokens = num_query_tokens + past_seen_tokens

        if position_ids is None:
            cache_position = torch.arange(past_seen_tokens, num_kv_tokens, device=inputs_embeds.device)
            position_ids = cache_position.unsqueeze(0)
        position_embeddings = self.rotary_emb.forward(position_ids.float())

        # 相关代码位于 transformers.modeling_attn_mask_utils 中
        if attention_mask is None:
            attention_mask = torch.ones(1, num_kv_tokens)
        attention_mask = attention_mask[:, None, None, :].float()  # [batch_size, 1, 1, num_kv_tokens]

        causal_attn_mask = torch.tril(torch.ones(num_query_tokens, num_kv_tokens), diagonal=past_seen_tokens)
        causal_attn_mask = causal_attn_mask[None, None, :, :]  # [1, 1, num_query_tokens, num_key_tokens]
        attention_mask = attention_mask * causal_attn_mask
        attention_mask = -10000. * (1 - attention_mask)

        hidden_states = inputs_embeds
        new_past_key_values = []
        
        for idx, decoder_layer in enumerate(self.layers):
            decoder_layer: Qwen2DecoderLayer
            if past_key_values is None:
                hidden_states, new_past_key_value = decoder_layer.forward(hidden_states, attention_mask, position_embeddings)
            else:
                hidden_states, new_past_key_value = decoder_layer.forward(hidden_states, attention_mask, position_embeddings, past_key_values[idx])
            new_past_key_values.append(new_past_key_value)
        
        hidden_states = self.norm(hidden_states)
        return hidden_states, new_past_key_values


class Qwen2ForCausalLM(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config 

        self.model = Qwen2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
            self,
            input_ids: Tensor = None,
            attention_mask: Tensor = None,
            position_ids: Tensor = None,
            past_key_values: list[tuple[Tensor, Tensor]] = None,
            num_logits_to_keep: int = 0,
        ) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:

        hidden_states, past_key_values = self.model.forward(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values
        )

        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        return logits, past_key_values


if __name__ == "__main__":
    import os 
    from safetensors.torch import load_model
    from transformers import AutoTokenizer
    from transformers.models.qwen2 import Qwen2ForCausalLM as GoldQwen2ForCausalLM

    # modelscope download --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B -f ./DeepSeek-R1-Distill-Qwen-1.5B/
    model_path = "/home/lqxu/model-cache/DeepSeek-R1-Distill-Qwen-1.5B"
    check_mode = False

    with torch.no_grad(), torch.device("cuda:0"):
        config = Qwen2Config.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        messages = [
            [{"role": "user", "content": "一加一等于几?"}],
        ]
        input_ids: Tensor = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True, padding=True
        )
        past_key_values = None

        test_model = Qwen2ForCausalLM(config).eval()
        load_model(test_model, filename=os.path.join(model_path, "model.safetensors"))
        test_model.eval()

        for _ in range(200):
            logits, past_key_values = test_model.forward(input_ids=input_ids, past_key_values=past_key_values, num_logits_to_keep=1)
            next_token_id = logits.argmax(-1).item()

            if next_token_id == config.eos_token_id:
                break 
            
            print(tokenizer.decode(next_token_id), end="", sep="", flush=True)

            if check_mode:
                logits = logits.cpu()
                break

            input_ids = logits.argmax(-1)
            # input_ids = torch.cat([input_ids, logits.argmax(-1)], dim=-1)
        
        print()

        del test_model
        torch.cuda.empty_cache()

        if check_mode:
            gold_model = GoldQwen2ForCausalLM.from_pretrained(model_path).eval().to(torch.float32)
            gold_logits = gold_model.forward(input_ids=input_ids, num_logits_to_keep=1).logits.cpu()
            print(tokenizer.decode(gold_logits.argmax(-1).item()), end="", sep="", flush=True)
            diff = torch.abs(gold_logits - logits)
            print(torch.max(diff).item(), torch.mean(diff).item())

    """
    首先，我需要明确问题：一加一等于几？

    根据基本的数学知识，1加1等于2。

    为了确保准确性，我可以回顾一下数学的基本原则，确认1加1确实等于2。

    此外，我可以从不同的角度来验证这个结果，例如通过数数法或代数运算，以确保答案的正确性。

    综上所述，1加1的结果是2。
    </think>

    **解答：**

    一加一等于几？

    根据数学的基本原则，1加1等于2。

    \[
    1 + 1 = 2
    \]

    因此，答案是：

    \[
    \boxed{2}
    \]
    """
