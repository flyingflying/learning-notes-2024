
"""
reference: https://pytorch.org/tutorials/intermediate/rpc_tutorial.html 

只有当 world_size=2 时运行才没有问题, 大于 2 就会报错, 报错信息可以参考:
https://github.com/pytorch/examples/issues/856 

吐槽一下, 这个教程和实际的 parameter server 有较大的差距, 感觉属于误导教程。

再吐槽一下, PyTorch Example 项目的更新速度极慢, 一些 示例代码 几年没有更新了, 甚至于版本还停留在 1.9,
另一些代码的 bug 几年都没有修复了, 真的不能理解。

parameter server: https://zhuanlan.zhihu.com/p/82116922 
"""

import os 

import torch 
from torch import Tensor
from torch import optim, nn  
from torch.distributed import rpc
from torch.nn import functional as F 
from torch import multiprocessing as mp
from torch.utils.data import DataLoader
from torch.distributed import autograd as dist_autograd
from torch.distributed.optim import DistributedOptimizer 

from torchvision import datasets, transforms

import numpy as np 
from sklearn.metrics import accuracy_score


class ParameterServer(nn.Module):
    __instance = None

    def __init__(self, device_id: int = None) -> None:
        super().__init__()

        if device_id is None:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"cuda:{device_id}")

        # 输入图片的高宽是 28 * 28
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1,  out_channels=32, kernel_size=3, stride=1),  # 高宽: 26 * 26
            nn.ReLU(), 
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),  # 高宽: 24 * 24
            nn.MaxPool2d(kernel_size=2),  # 高宽: 12 * 12
            nn.Dropout(p=0.25), 
            nn.Flatten(), 
            nn.Linear(in_features=9216, out_features=128),  # 输入 size: 12 * 12 * 64 = 9216
            nn.ReLU(), 
            nn.Dropout(p=0.5), 
            nn.Linear(in_features=128, out_features=10),
        )

        self.model.to(self.device)

    def forward(self, input_imgs: Tensor) -> Tensor:
        input_imgs = input_imgs.to(self.device)
        logits = self.model(input_imgs).cpu()
        return logits

    def get_param_rrefs(self) -> list[rpc.RRef]:
        rrefs = [rpc.RRef(param) for param in self.parameters()]
        return rrefs

    @classmethod
    def get_instance(cls, device_id: int = None) -> 'ParameterServer':
        """ 单例模式 (线程不安全) """
        if cls.__instance is None:
            cls.__instance = cls(device_id)
        return cls.__instance


def run_server(cur_rank: int, world_size: int):
    """ server 进程主要用于更新模型参数, 由 worker 进程进行调度 """
    param_server = ParameterServer.get_instance(device_id=0)

    rpc.init_rpc("param_server", rank=cur_rank, world_size=world_size)
    rpc.shutdown()

    torch.save(param_server.model.state_dict(), "model.bin")


class Trainer:
    def __init__(self, rank: int):
        self.ps_rref: rpc.RRef = rpc.remote("param_server", ParameterServer.get_instance)

        self.rank = rank 

    def train_one_epoch(self, train_loader: DataLoader, epoch_idx: int, verbose: bool = False):

        # 初始化 优化器
        params_rref = self.ps_rref.rpc_sync().get_param_rrefs()
        optimizer = DistributedOptimizer(optimizer_class=optim.SGD, params_rref=params_rref, lr=0.03)

        for iter_idx, (imgs, targets) in enumerate(train_loader):
            with dist_autograd.context() as context_id:
                logits = self.ps_rref.rpc_sync().forward(imgs)
                loss = F.cross_entropy(logits, targets)
                dist_autograd.backward(context_id, [loss, ])
                optimizer.step(context_id)

            if verbose:
                print(f"rank {self.rank}: completed epoch {epoch_idx} and iteration {iter_idx} training.", flush=True)

    @torch.no_grad()
    def eval(self, eval_loader: DataLoader):

        # TODO: 需要调用 model.eval() 方法, 怎么调用比较好?
        y_pred_list, y_true_list = [], []

        for imgs, targets in eval_loader:
            logits = self.ps_rref.rpc_sync().forward(imgs)
            y_pred_list.append(torch.argmax(logits, dim=-1).cpu().numpy())
            y_true_list.append(targets)

        y_pred = np.concatenate(y_pred_list, axis=0)
        y_true = np.concatenate(y_true_list, axis=0)
        return accuracy_score(y_true, y_pred)

    def train(self, minst_data_dir: str, download: bool = False, epoches: int = 3):
        train_loader = DataLoader(
            datasets.MNIST(
                minst_data_dir, train=True, download=download,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
            ),
            batch_size=32, shuffle=True
        )

        eval_loader = DataLoader(
            datasets.MNIST(
                minst_data_dir, train=False, download=download,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
            ), 
            batch_size=64, shuffle=False
        )

        for epoch_idx in range(epoches):
            self.train_one_epoch(train_loader, epoch_idx)
            accuracy = self.eval(eval_loader)
            print(f"rank {self.rank}: completed epoch {epoch_idx} training with accuracy score {accuracy:0.2f}")


def run_worker(cur_rank: int, world_size: int):
    rpc.init_rpc(name=f"worker{cur_rank}", rank=cur_rank, world_size=world_size)

    trainer = Trainer(cur_rank)
    trainer.train("~/image_generation/mnist_data/")

    rpc.shutdown()


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "5678"

    WORLD_SIZE = 2
    processes: list[mp.Process] = []

    assert WORLD_SIZE == 2

    processes.append(
        mp.Process(target=run_server, args=(0, WORLD_SIZE))
    )

    for rank in range(1, WORLD_SIZE):
        processes.append(
            mp.Process(target=run_worker, args=(rank, WORLD_SIZE))
        )

    for process in processes:
        process.start()

    for process in processes:
        process.join()
