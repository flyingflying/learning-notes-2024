
# %%

import torch 
from torch import nn, Tensor 

# %% 梯度累积 示例代码

BATCH_SIZE, NUM_FEATURES, HIDDEN_SIZE = 32, 8, 4
GRADIENT_ACCUMULATION_STEP = 2

assert BATCH_SIZE % GRADIENT_ACCUMULATION_STEP == 0

input = torch.randn(BATCH_SIZE, NUM_FEATURES)
target = torch.randn(BATCH_SIZE, 1)

torch.manual_seed(0)
model_0 = nn.Sequential(
    nn.Linear(in_features=NUM_FEATURES, out_features=HIDDEN_SIZE), 
    nn.Sigmoid(), 
    nn.LayerNorm(normalized_shape=(HIDDEN_SIZE, )), 
    # nn.BatchNorm1d(num_features=HIDDEN_SIZE), 
    nn.Linear(in_features=HIDDEN_SIZE, out_features=1)
)

torch.manual_seed(0)
model_1 = nn.Sequential(
    nn.Linear(in_features=NUM_FEATURES, out_features=HIDDEN_SIZE), 
    nn.Sigmoid(), 
    nn.LayerNorm(normalized_shape=(HIDDEN_SIZE, )), 
    # nn.BatchNorm1d(num_features=HIDDEN_SIZE), 
    nn.Linear(in_features=HIDDEN_SIZE, out_features=1)
)

torch.nn.functional.mse_loss(
    model_0(input), target
).backward()

print(model_0[0].weight.grad)

part_inputs = torch.split(input, BATCH_SIZE // GRADIENT_ACCUMULATION_STEP, dim=0)
part_targets = torch.split(target, BATCH_SIZE // GRADIENT_ACCUMULATION_STEP, dim=0)

for part_input, part_target in zip(part_inputs, part_targets):

    torch.nn.functional.mse_loss(
        model_1(part_input), part_target
    ).div(GRADIENT_ACCUMULATION_STEP).backward()

print(model_1[0].weight.grad)

for param1, param2 in zip(model_0.parameters(), model_1.parameters()):
    print((param1.grad - param2.grad).abs().max())


"""
只要不涉及到 BatchNorm, 梯度累积的问题就不大

BatchNorm 的主要问题: running_mean 和 running_var 是在前向传播时就更新的
"""
