
import torch 
from torch import Tensor 
from transformers import AutoTokenizer
from transformers.models.qwen2 import Qwen2ForCausalLM

model_name = "/home/lqxu/model-cache/DeepSeek-R1-Distill-Qwen-1.5B"
max_length = 50


def do_sample(
        logits: Tensor,  # [batch_size, vocab_size]
        temperature: float = 1.0, top_k: int = None, 
        top_p: float = None, min_tokens_to_keep: int = 1
):
    # ## 1. temperature: 修改词表概率分布的形状
    temperature = max(temperature, 1e-5)  # 最小值为 1e-5, 此时等价于 贪心策略
    logits = logits / temperature

    # ## 2. top_k: 将概率最高的 k 个词语作为候选词
    if top_k is not None:
        top_k = min(top_k, logits.size(-1))  # 当 top_k = 1 时, 等价于 贪心策略
        min_logits = torch.topk(logits, top_k).values[..., -1].unsqueeze(-1)
        indices_to_remove = logits < min_logits
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

    # ## 3. top_p: 将累积概率在 top_p 范围内的词语作为候选词
    # 换言之, 将词表概率从高到低排序, 求累积概率, 保留累积概率低于 top_p 的词语
    if top_p is not None:
        # 将词表概率从低到高排序, 求累积概率, 然后将累积概率低于 1 - top_p 的都删除掉
        sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=False)  # 升序排列
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)  # 转换成概率, 累加
        sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
        sorted_indices_to_remove[..., -min_tokens_to_keep :] = False  # 至少保留 min_tokens_to_keep 个词语
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, -float("inf"))

    # ## 4. 采样
    probs = torch.softmax(logits, dim=-1)
    # HuggingFace Transformers 实现方式: 多项式分布
    next_tokens = torch.multinomial(probs, num_samples=1)
    # DeepSeek-V3 实现方式: 指数分布
    # next_tokens = probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1, keepdim=True)

    return next_tokens


with torch.no_grad(), torch.device("cuda:0"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = Qwen2ForCausalLM.from_pretrained(model_name).eval()
    eos_token = tokenizer.special_tokens_map['eos_token']
    # messages = [{"role": "user", "content": "草莓的英文单词有几个R字母?"}]
    # input_ids: Tensor = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)

    messages = [{"role": "user", "content": "广州的白云"}]
    input_ids: Tensor = tokenizer(text="广州的白云", return_tensors="pt")["input_ids"]

    for _ in range(max_length):
        logits = model.forward(input_ids, num_logits_to_keep=1).logits.squeeze(1)  # [batch_size, vocab_size]
        # output_id = logits.argmax(dim=-1, keepdim=True)  # 贪心策略 [batch_size, 1]
        output_id = do_sample(logits, temperature=1.5, top_k=50, top_p=0.95)  # 采样策略 [batch_size, 1]
        output_token = tokenizer.decode(output_id.item())

        if output_token != eos_token:
            print(output_token, end="", flush=True) 
        else:
            print()
            break 
        
        input_ids = torch.cat([input_ids, output_id], dim=-1)


def test_sort_and_scatter():
    a = torch.randn(4, 4, 4)
    sorted_a, sorted_indices = torch.sort(a, dim=2)
    re_sort_a = sorted_a.scatter(dim=1, index=sorted_indices, src=sorted_a)
    print(torch.all(a == re_sort_a).item())
