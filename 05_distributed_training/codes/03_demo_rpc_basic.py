
"""
RPC (Remote Procedure Call), 对 底层通信 进行封装, 方面调用。

PyTorch 中的 RPC 支持并不是很好, 不支持 CUDA Tensor, 有待改进。

Reference: https://pytorch.org/docs/stable/rpc.html 
"""

import os 

import torch
from torch import nn, Tensor 
from torch.futures import Future
from torch.distributed import rpc 
from torch import multiprocessing as mp 


def block_matmul(lmat: Tensor, rmat: Tensor, lmat_split_size: int = 1, rmat_split_size: int = 1) -> Tensor:

    assert lmat.ndim == 2 and rmat.ndim == 2 and lmat.size(1) == rmat.size(0)
    result = torch.zeros(lmat.size(0), rmat.size(1))

    for part_lmat, part_rmat in zip(
            torch.split(lmat, lmat_split_size, dim=1), 
            torch.split(rmat, rmat_split_size, dim=0)
        ):

        result += torch.matmul(part_lmat, part_rmat)

    return result 


def is_same_tensor(t1: Tensor, t2: Tensor, eps: float = 1e-6, debug: float = False) -> bool:
    if not debug:
        return t1.shape == t2.shape and (t1 - t2).abs().max().cpu().item() < eps 
    
    if t1.shape != t2.shape:
        print("The shape of two tensors does not match!")
        return False

    max_diff = (t1 - t2).abs().max().cpu().item()

    if max_diff > eps:
        print(f"The maximum value of difference between two tensors is {max_diff}, which is greater than {eps}!")
        return False

    return True


def check_block_matmul():
    lmat = torch.randn(11, 21)
    rmat = torch.randn(21, 31)

    print(is_same_tensor(
        torch.matmul(lmat, rmat),
        block_matmul(lmat, rmat, 10, 10), 
        eps=1e-5, 
        debug=True
    ))


def run_worker(cur_rank: int):

    # ## 初始化 进程组 通信
    # 在 RPC 中, 一个进程称为一个 agent, 每一个进程都需要一个 名称!
    # main 进程对应 rank=0 进程, helper 进程对应 rank=1 进程
    if cur_rank == 0:
        rpc.init_rpc("main", rank=cur_rank, world_size=2)
    else:
        rpc.init_rpc("helper", rank=cur_rank, world_size=2)

    if cur_rank == 0:
        """ 我们只让 main 进程往 helper 进程分配任务, 因此 helper 进程没有多少代码, 只负责运算 """

        # ## rpc_sync 同步调用方式, 让 helper 进程调用 block_matmul 方法, 参数值在 args 中
        result: Tensor = rpc.rpc_sync("helper", block_matmul, args=(torch.randn(100, 200), torch.randn(200, 100)))
        print(result.shape)

        # ## rpc_async 异步的调用方式, 让 helper 进程调用 block_matmul 方法, 参数值在 args 中
        async_block: Future = rpc.rpc_async("helper", block_matmul, args=(torch.randn(100, 200), torch.randn(200, 100)))
        result: Tensor = async_block.wait()
        print(result.shape)

        # ## remote 调用方式
        rref: rpc.RRef = rpc.remote("helper", block_matmul, args=(torch.randn(100, 200), torch.randn(200, 100)))
        result: Tensor = rref.to_here()
        print(result.shape)

        # ## RRef 主要用于 对象 的远程调用
        linear_rref: rpc.RRef = rpc.remote("helper", nn.Linear, kwargs={"in_features": 20, "out_features": 20})
        # rpc_sync 返回的对象不能直接作为 Callable 对象, 否则会报错! 
        result: Tensor = linear_rref.rpc_sync().forward(torch.randn(100, 20))
        print(result.shape)

        result: Tensor = linear_rref.rpc_async().forward(torch.randn(100, 20)).wait()
        print(result.shape)

        # ## get_worker_info: 根据 进程名称 获得 相关信息
        print(rpc.get_worker_info("helper"))

    # ## 所有的进程都执行到 rpc.shutdown 时, 统一结束所有进程
    # helper 进程 执行到此处时, 辅助 main 进程的计算, 并等待 main 进程执行到此处 
    rpc.shutdown(graceful=True)


if __name__ == "__main__":
    check_block_matmul()

    """
    我们可以用 mp.spawn 启动多个进程, 可以用 torchrun 启动多个进程, 也可以手动启动多个进程。
    一般情况下, 单台机器: 用 mp.spawn / torchrun 启动多个进程; 多台机器: 每一台机器手动启动。
    这里使用 mp.spawn 启动, 而非 torchrun 启动!
    """

    """
    注意事项:
        1. mp.spawn 启动的函数必须在外部定义, 不能在 if 语句中定义, 否则会报错
        2. mp.spawn 启动的函数至少有一个参数, 即 rank / local_rank 
    """

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "5678"

    world_size = 2

    mp.spawn(run_worker, nprocs=world_size, join=True)
