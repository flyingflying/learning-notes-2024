
"""
使用 RPC 设计 语言模型 示例代码。我们可以将 语言模型 分解成 三部分:

1. EmbeddingTable: 将离散的 token 映射成为 词向量, 我们也可以成为 Encoder 
2. model: 运算的模型部分, 可以是 RNN, 也可以是 Transformer, 根据需求确定
3. Decoder: 分类层, 将词向量转换成 下一个 token 的概率分布

假设 语言模型 的词表非常之大 (有一百万), 我们现在需要将模型放在 三个 GPU 之上。
其中, EmbeddingTable, model 和 Decoder 各占一个 GPU 设备。

这个示例代码就在 上述场景 下生成。

reference: https://pytorch.org/tutorials/intermediate/rpc_tutorial.html 
"""

import os 

import torch 
from torch import nn, optim
from torch.distributed import rpc
from torch import Tensor, LongTensor
from torch import multiprocessing as mp 
from torch.distributed import autograd as dist_autograd
from torch.distributed.optim import DistributedOptimizer


class EmbeddingTable(nn.Module):
    """ 将 词语 映射成 向量, 由于 词表 可能会非常大, 因此我们需要一个单独的 GPU 设备进行处理 """

    def __init__(self, vocab_size: int, input_size: int, device_id: int = None, dropout: float = 0.1) -> None:
        super().__init__()

        if device_id is None:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"cuda:{device_id}")

        self.encoder = nn.Embedding(vocab_size, input_size, padding_idx=0)
        self.dropout = nn.Dropout(p=dropout)
        self.to(self.device)

    def forward(self, input_ids: LongTensor):
        # 在 RPC 编程中, 默认所有的 张量 都在 CPU 上
        input_ids = input_ids.to(self.device)
        input_embeds: Tensor = self.dropout(self.encoder(input_ids))
        input_embeds = input_embeds.cpu()
        return input_embeds

    def rref_parameters(self) -> list[rpc.RRef]:
        rrefs = [rpc.RRef(param) for param in self.parameters()]
        return rrefs


class Decoder(nn.Module):
    """ 语言模型 的任务就是预测下一个词, 我们需要进行 词表级别 的预测, 非常的吃 显存, 因此我们也用一个单独的 GPU 设备进行处理 """
    def __init__(self, vocab_size: int, hidden_size: int, device_id: int = None, dropout: float = 0.1) -> None:
        super().__init__()

        if device_id is None:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"cuda:{device_id}")

        self.classifier = nn.Linear(hidden_size, vocab_size, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.to(self.device)

    def forward(self, output_embeds: Tensor):
        output_embeds = output_embeds.to(self.device)
        output_logits: Tensor = self.dropout(self.classifier(output_embeds))
        output_logits = output_logits.cpu()
        return output_logits

    def rref_parameters(self) -> list[rpc.RRef]:
        rrefs = [rpc.RRef(param) for param in self.parameters()]
        return rrefs


class RNNLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()

        self.device = torch.device("cuda:0")

        self.et_rref: rpc.RRef = rpc.remote(
            "et_agent", EmbeddingTable, args=(vocab_size, hidden_size, 0, 0.1)
        )

        self.rnn_model = nn.RNN(
            hidden_size, hidden_size, num_layers=2, 
            nonlinearity="relu", batch_first=True, dropout=0.1, bidirectional=False
        ).to(self.device)

        self.decoder_rref: rpc.RRef = rpc.remote(
            "decoder_agent", Decoder, args=(vocab_size, hidden_size, 0, 0.1)
        )

    def forward(self, input_ids: LongTensor) -> Tensor:

        input_embeds: Tensor = self.et_rref.rpc_sync().forward(input_ids)
        hidden_states: Tensor = self.rnn_model(input_embeds.to(self.device))[0].cpu()
        logits: Tensor = self.decoder_rref.rpc_sync().forward(hidden_states)

        return logits

    def rref_parameters(self) -> list[rpc.RRef]:
        rrefs = [rpc.RRef(param) for param in self.parameters()]
        rrefs.extend(self.et_rref.rpc_sync().rref_parameters())
        rrefs.extend(self.decoder_rref.rpc_sync().rref_parameters())
        return rrefs


def run_worker(cur_rank: int, vocab_size: int, hidden_size: int, num_tokens: int, batch_size: int):

    if cur_rank == 0:
        rpc.init_rpc("main_agent", rank=0, world_size=3)
    elif cur_rank == 1:
        rpc.init_rpc("et_agent", rank=1, world_size=3)
    elif cur_rank == 2:
        rpc.init_rpc("decoder_agent", rank=2, world_size=3)
    else:
        raise ValueError

    if cur_rank == 0:

        # 初始化语言模型
        rnn_lm = RNNLanguageModel(vocab_size, hidden_size)

        # 初始化优化器
        optimizer = DistributedOptimizer(optim.SGD, rnn_lm.rref_parameters(), lr=1e-5)

        mask_target = torch.tensor([-100, ] * batch_size).unsqueeze(-1)

        for _ in range(3):  # 进行迭代

            input_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, num_tokens))

            # 语言模型标签数据的生成方式
            target = torch.concat([input_ids[:, 1:], mask_target], dim=-1)

            with dist_autograd.context() as context_id:

                logits: Tensor = rnn_lm(input_ids)  # [batch_size, num_tokens, vocab_size]

                print(logits.shape)

                loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target.flatten(), ignore_index=-100)
                dist_autograd.backward(context_id, [loss])
                optimizer.step(context_id)

                """
                梯度在 distributed autograd context 中累积
                而每一次迭代, 我们都会创建一个 distributed autograd context
                因此, 这里不需要 zero_grad !!!
                """

            pass 

    rpc.shutdown(graceful=True)


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "5678"
    mp.spawn(run_worker, args=(100, 200, 300, 400), nprocs=3, )
