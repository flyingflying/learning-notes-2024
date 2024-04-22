
"""
DDP 程序示例代码

reference: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html 
"""

import os 
import torch 
from torch import nn, Tensor, optim
from torch import distributed as dist 
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import Dataset, DistributedSampler, DataLoader

SEED = 42
BATCH_SIZE, NUM_FEATURES, HIDDEN_SIZE = 9, 100, 10


class DemoDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.num_examples = BATCH_SIZE * 2
        self._processed_indices = []

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        self._processed_indices.append(idx)

        input = torch.randn(NUM_FEATURES)
        target = torch.randn(1)

        return input, target

    def __len__(self):
        return self.num_examples 

    def get_processed_indices(self) -> list[int]:
        return sorted(self._processed_indices)


def main():
    """ 每一个进程都运行此函数! """

    # region step1: 初始化进程组, 和其它进程之间建立联系
    # 注意, world_size 和 rank 是在初始化进程组之后才有的变量, 而 local_rank 在不进行初始化的情况下就有的变量
    local_rank = int(os.environ["LOCAL_RANK"])

    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    print(f"world_size={world_size}, rank={rank}, local_rank={local_rank}", flush=True)
    # endregion 

    # region step2: 确定并行化的方式
    # 这里, 我们认为一个进程对应一个 GPU 设备, 当然也可以是一个进程对应多个 GPU 设备
    device = torch.device(f"cuda:{local_rank}")

    # 如果一个进程对应两个 GPU 设备, 代码如下:
    # device1 = torch.device(f"cuda:{2 * local_rank}")
    # device2 = torch.device(f"cuda:{2 * local_rank + 1}")
    # endregion 

    # region step3: 初始化网络
    model = nn.Sequential(
        nn.Linear(NUM_FEATURES, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZE, 1),
    ).to(device)
    # 用 DistributedDataParallel 进行封装, 反向传播时梯度自动进行 all_reduce 操作
    """
    注意事项:
        1. 如果 model 中有 BatchNorm 层, 需要手动换成 SyncBatchNorm 层;
        2. 模型初始化参数不用保持一致, 在 DDP 中会自动将 rank 0 进程上的参数分发给其它进程的;
        3. 对于 Dropout 层而言, 每一个样本的 mask 没有必要保证相同, 这毫无意义;
        4. 对于 CV 领域, 如果要设置随机数种子, 一般建议每一个进程的随机数种子设置成不同的数字, 否则 图片数据增强 的结果具有 趋同性, 严重影响训练的效果。
    References:
        1. https://lightning.ai/forums/t/behaviour-of-dropout-over-multiple-gpu-setting/4562 
        2. https://zhuanlan.zhihu.com/p/250471767 
    """
    model = DistributedDataParallel(model)
    # endregion 

    # region extra: 下载数据
    if local_rank == 0:
        # 一般情况下, 每一个节点都需要下载完整的数据集
        # 这样可以保证一个节点只有一个进程在下载数据
        pass 

    """
    切记, dist.barrier() 函数必须在涉及到 GPU 的代码之后调用 (比方说 模型初始化)
    否则, 程序一定会卡死 (NCCL backend 不支持 CPU, 这也太坑了吧)
    """
    dist.barrier()  # 同步
    # endregion 

    # region step4: 初始化数据集
    dataset = DemoDataset()
    # 对于 sampler 而言, 每一个进程的随机数种子是一致的, 这一点非常重要
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=SEED)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=BATCH_SIZE // world_size)
    # endregion 

    # region step5: 开始训练
    optimizer = optim.SGD(model.parameters(), lr=1e-5)
    """
    每一次 epoch, sampler 都要调用 set_epoch 方法, 否则 shuffle 的意义不大。这里默认只有一个 epoch, 因此直接设置为 0
    reference: https://www.zhihu.com/question/67209417/answer/1017851899
    """
    sampler.set_epoch(0)
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        inputs: Tensor = inputs.to(device)
        targets: Tensor = targets.to(device)

        outputs = model(inputs)
        loss = nn.functional.mse_loss(outputs, targets)

        loss.backward()  # DistributedDataParallel 会自动对 模型参数 进行 all_reduce 操作
        optimizer.step()
    # endregion

    # region step6: 保存模型
    if rank == 0:
        # 保证只有主节点存储模型
        model: nn.Module = model.module
        model = model.cpu().eval()
        torch.save(model.state_dict(), "demo_ddp.bin")
    # endregion 

    # 检测 DistributedSampler 的合理性
    print(f"No. {rank} processes: {dataset.get_processed_indices()}", flush=True)


if __name__ == "__main__":
    """
    单节点的运行方式: torchrun --nnodes=1 --nproc_per_node=2 demo_ddp.py
    其中, nnodes 表示 计算机节点 的数量, nproc_per_node 表示 每一个节点 的进程数

    多节点的运行方式:
        1. 首先, 每一个节点都要有一份数据, 有一份代码, 有一份环境
        2. 然后, 启动主节点程序: torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 --master_addr=192.168.0.230 --master_port=29401 --rdzv_conf=is_host=true demo.py
        3. 接下来, 启动其它节点程序: torchrun --nnodes=2 --nproc_per_node=2 --node_rank=1 --master_addr=192.168.0.230 --master_port=29401 --rdzv_conf=is_host=false demo.py

    注意, 每一个节点的代码可以不一致, torchrun 只保证必要的数据交换是正常的, 不会检查代码的一致性。同时, 每一个节点的启动指令是不一样的。
    在这里, 192.168.0.230 是 主节点, 上面只启动一个进程, 因此 nproc_per_node 的值为 1, 同时在 rdzv_conf 中将 is_host 值设置为 true (防止自动检测失败)
    而在其它节点中, 启动两个进程, 因此 nproc_per_node 的值为 2

    Reference:
        1. https://github.com/pytorch/pytorch/issues/73656 
        2. https://zhuanlan.zhihu.com/p/510718081 
        3. https://stackoverflow.com/questions/58271635/in-distributed-computing-what-are-world-size-and-rank 
    """

    main()
