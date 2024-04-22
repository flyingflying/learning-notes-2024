
"""
pipeline model parallelism 示例代码

reference: https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html 

原理: PyTorch launches CUDA operations asynchronously 但是在两张 3090 上测试不出来教程中的效果

RPC Version: https://pytorch.org/tutorials/intermediate/dist_pipeline_parallel_tutorial.html 
"""

import torch 
from torch import nn, Tensor

BATCH_SIZE, HIDDEN_SIZE, NUM_LAYERS = 1000, 1000, 10


def cdiv(x: int, y: int) -> int:
    return (x + y - 1) // y


def init_model() -> nn.Module:
    layers = []

    for _ in range(NUM_LAYERS):
        layers.append(nn.Linear(
            in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE * 2, 
        ))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(
            in_features=HIDDEN_SIZE * 2, out_features=HIDDEN_SIZE
        ))

    return nn.Sequential(*layers).eval()


part1_model = init_model().to("cuda:0")

part2_model = init_model().to("cuda:1")


def native_mp(input: Tensor):
    input_device = input.device 
    output: Tensor = input

    output = part1_model(output.to("cuda:0"))
    output = part2_model(output.to("cuda:1"))

    return output.to(input_device)


def native_split_mp(input: Tensor) -> Tensor:
    input_device = input.device 
    input = input.to("cuda:0")

    split_size = cdiv(input.size(0), 2)
    part_inputs = torch.split(input, split_size)
    part_outputs = []

    for part_input in part_inputs:
        part1_output: Tensor = part1_model(part_input)
        part2_output: Tensor = part2_model(part1_output.to("cuda:1"))

        part_outputs.append(part2_output)

    return torch.concat(part_outputs).to(input_device)


def pipeline_mp(input: Tensor, split_size: int = None) -> Tensor:
    input_device = input.device 
    input = input.to("cuda:0")

    if split_size is None:
        split_size = cdiv(input.size(0), 2)
    part_inputs = torch.split(input, split_size)
    part_outputs = []

    part1_output: Tensor = part1_model(part_inputs[0])

    for part_input in part_inputs[1:]:
        # part1_output_temp = part1_model(part_input)

        part2_output: Tensor = part2_model(part1_output.to("cuda:1"))
        part_outputs.append(part2_output)

        part1_output = part1_model(part_input)
        # part1_output = part1_output_temp

    part2_output: Tensor = part2_model(part1_output.to("cuda:1"))
    part_outputs.append(part2_output)

    return torch.concat(part_outputs).to(input_device)


def simple_train(func, input):
    from torch import optim

    params = list(part1_model.parameters()) + list(part2_model.parameters())

    optimizer = optim.SGD(params, lr=1e-3)

    for _ in range(100):
        func(input).sum().backward()

        optimizer.step()

        optimizer.zero_grad()


if __name__ == "__main__":

    import timeit
    import numpy as np 

    test_input = torch.randn(BATCH_SIZE, HIDDEN_SIZE)

    print(
        (native_mp(test_input) - pipeline_mp(test_input, 100)).abs().max().cpu().item()
    )

    # pp_run_times = timeit.repeat(
    #     stmt="native_mp(test_input)", number=1, repeat=1000, globals=globals()
    # )
    # print(np.median(pp_run_times))

    # pp_run_times = timeit.repeat(
    #     stmt="pipeline_mp(test_input, 100)", number=1, repeat=1000, globals=globals()
    # )
    # print(np.median(pp_run_times))

    pp_run_times = timeit.repeat(
        stmt="simple_train(native_mp, test_input)", number=1, repeat=10, globals=globals()
    )
    print(np.median(pp_run_times))

    pp_run_times = timeit.repeat(
        stmt="simple_train(pipeline_mp, test_input)", number=1, repeat=10, globals=globals()
    )
    print(np.median(pp_run_times))
