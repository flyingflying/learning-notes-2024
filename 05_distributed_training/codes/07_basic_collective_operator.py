
import torch 
from torch import Tensor
from torch import distributed as dist 
from torch import multiprocessing as mp 


def flush_print(*args):
    print(*args, "\n", end="", flush=True)


def run_worker(rank: int, world_size: int):
    dist.init_process_group(
        backend="gloo", init_method="tcp://127.0.0.1:5679", 
        world_size=world_size, rank=rank, 
    )

    # region send & recv
    # send 和 recv 属于 P2P 通信, 是一组操作, send 在 src 进程上调用, recv 在 dst 进程上调用
    if rank == 0:
        tensor = torch.randn(2, )
        dist.send(tensor, dst=1)
        flush_print(f"rank 0 send {tensor} to rank 1")
    elif rank == 1:
        tensor = torch.zeros(2, )
        dist.recv(tensor, src=0)
        flush_print(f"rank 1 received {tensor} from rank 0")
    dist.barrier()
    # endregion 

    # region broadcast
    # 数据由 src 进程发送给其它所有进程
    tensor = torch.randn(3, )
    dist.broadcast(tensor, src=0)  # 不需要判断 rank, 指定 src 即可
    flush_print(f"broadcast op {rank}: {tensor}")
    dist.barrier()
    # endregion 

    # region all_reduce
    # 对所有进程的数据进行 reduce 操作, 结果保存在所有进程中
    tensor = torch.tensor([1., 2., 3., 4.])
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    flush_print(f"all_reduce op {rank}: {tensor}")
    dist.barrier()
    # endregion 

    # region reduce
    # 对所有进程的数据进行 reduce 操作, 结果保存在 dst 进程中
    tensor = torch.tensor([1., 2., 3., 4.])
    dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)
    # 需要注意的是, 非 dst 进程中的数据会发生变化, 因此是不可用的!
    flush_print(f"reduce op {rank}: {tensor}")
    dist.barrier()
    # endregion

    # region all_gather
    # 收集所有进程的数据, 结果保存在所有进程中
    tensor = torch.ones(2) * (rank + 1) 
    tensor_list = [torch.zeros(2) for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_list, tensor)
    flush_print(f"all_gather op {rank}: {tensor_list}")  # 结果按照 rank 顺序排列
    dist.barrier()
    # endregion 

    # region gather
    # 收集所有进程的数据, 结果保存在 dst 进程中
    tensor = torch.ones(2) * (rank + 1) 
    gather_list = [torch.zeros(2) for _ in range(dist.get_world_size())]
    if rank == 0:
        dist.gather(tensor, gather_list, dst=0)
    else:
        # 在非 dst 进程中, gather_list 必须是 None, 否则会报错
        dist.gather(tensor, None, dst=0)
    flush_print(f"gather op {rank}: {gather_list}")  # 结果按照 rank 顺序排列
    dist.barrier()
    # endregion

    # region scatter
    # src 进程分发数据给其它进程
    tensor = torch.zeros(2)
    scatter_list = [torch.ones(2) * (i + 1) for i in range(dist.get_world_size())]
    if rank == 0:
        dist.scatter(tensor, scatter_list, src=0)
    else:
        # 在非 src 进程中, scatter_list 必须是 None, 否则会报错
        dist.scatter(tensor, None, src=0)
    flush_print(f"scatter op {rank}: {tensor}")  # 按照 rank 顺序发送
    dist.barrier()
    # endregion 

    # reduce_scatter 和 all_to_all 只在 NCCL 上实现, 没有再 Gloo 上实现, 而 NCCL 每一个进程必须对应一个 GPU 上执行 ...
    # 即使 new_group 中只有 2 个进程, 它也会等待所有进程执行到此处!
    nccl_group = dist.new_group(ranks=[0, 1], backend="nccl")

    if rank != 2:
        world_size = 2
        device = torch.device(f"cuda:{rank}")

        # region reduce_scatter
        # input_list 中的张量个数必须和 world_size 保持一致, 每一个张量的 shape 必须和 output 张量保持一致
        # 将 input_list 中的每一个张量进行进程间的 reduce 操作, 再分发给每一个进程的 output 张量
        output = torch.zeros(2, ).to(device)
        if rank == 0:
            input_list = [
                torch.tensor([1., 2.]).to(device), 
                torch.tensor([3., 4.]).to(device),
            ]
        elif rank == 1:
            input_list = [
                torch.tensor([5., 6.]).to(device), 
                torch.tensor([7., 8.]).to(device),
            ]

        dist.reduce_scatter(output, input_list, group=nccl_group, op=dist.ReduceOp.AVG)
        flush_print(f"reduce_scatter op {rank}: {output}")
        dist.barrier(group=nccl_group)
        # endregion

        # region all_to_all
        if rank == 0:
            input_tensor_list = [
                torch.tensor([1., 2., 3., 4.]).to(device), 
                torch.tensor([5., 6. ]).to(device)
            ]
            output_tensor_list = [
                torch.zeros(4).to(device), 
                torch.zeros(4).to(device)
            ]
        elif rank == 1:
            input_tensor_list = [
                torch.tensor([7., 8., 9., 10.]).to(device), 
                torch.tensor([11., 12.]).to(device)
            ]
            output_tensor_list = [
                torch.zeros(2).to(device), 
                torch.zeros(2).to(device)
            ]
        dist.all_to_all(output_tensor_list, input_tensor_list, group=nccl_group)
        flush_print(f"all_to_all op {rank}: {output_tensor_list}")
        dist.barrier(group=nccl_group)
        # endregion 

    dist.barrier()

    """
    send 和 recv 互为反向运算
    broadcast 和 reduce 互为反向运算
    gather 和 scatter 互为反向运算
    """


def check_reduce_scatter(rank: int, world_size: int):

    dist.init_process_group(
        backend="nccl", init_method="tcp://127.0.0.1:5679", 
        world_size=world_size, rank=rank, 
    )

    device = torch.device(f"cuda:{rank}")

    input_list = [torch.randn(5).to(device) for _ in range(world_size)]
    output = torch.zeros(5).to(device)

    dist.reduce_scatter(output, input_list, op=dist.ReduceOp.SUM)

    flush_print(f"rank {rank}: {output}")

    for dst, tensor in enumerate(input_list):
        tensor = tensor.clone()
        dist.reduce(tensor, dst=dst, op=dist.ReduceOp.SUM)
        if dist.get_rank() == dst:
            output = tensor 

    flush_print(f"rank {rank}: {output}")


def check_all_to_all(rank: int, world_size: int):

    dist.init_process_group(
        backend="nccl", init_method="tcp://127.0.0.1:5679", 
        world_size=world_size, rank=rank, 
    )

    device = torch.device(f"cuda:{rank}")

    input_tensor_list = [torch.randn(5).to(device) for _ in range(world_size)]
    output_tensor_list = [torch.zeros(5).to(device) for _ in range(world_size)]

    dist.all_to_all(output_tensor_list, input_tensor_list)

    flush_print(f"rank {rank}: {output_tensor_list}")

    for dst, tensor in enumerate(input_tensor_list):
        tensor = tensor.clone()

        if dist.get_rank() == dst:
            gather_list = [torch.zeros(5).to(device) for _ in range(world_size)]
            dist.gather(tensor, gather_list, dst=dst)
            output_tensor_list = gather_list
        else:
            dist.gather(tensor, None, dst=dst)

    flush_print(f"rank {rank}: {output_tensor_list}")


def check_grad(rank: int, world_size: int):

    dist.init_process_group(
        backend="nccl", init_method="tcp://127.0.0.1:5679", 
        world_size=world_size, rank=rank 
    )

    device = torch.device(f"cuda:{rank}")

    tensor = torch.randn(2, requires_grad=True).to(device)
    tensor_list = [torch.zeros(2, requires_grad=True).to(device) for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_list, tensor)
    flush_print(tensor_list)  # 有 grad_fn 就是有计算图

    if rank == 0:
        input = torch.randn(3, requires_grad=True).to(device)
        dist.send(input, dst=1)
        flush_print(f"rank {rank}: {input}")
    elif rank == 1:
        input = torch.zeros(3, requires_grad=True).to(device)
        dist.recv(input, src=0)
        flush_print(f"rank {rank}: {input}")


if __name__ == "__main__":

    # WORLD_SIZE = 3

    # processes: list[mp.Process] = []

    # for i in range(WORLD_SIZE):
    #     processes.append(
    #         mp.Process(target=run_worker, args=(i, WORLD_SIZE))
    #     )

    # for process in processes:
    #     process.start()

    # for process in processes:
    #     process.join()

    WORLD_SIZE = 2

    mp.spawn(check_grad, args=(WORLD_SIZE, ), nprocs=WORLD_SIZE, join=True)
