
import torch 
from torch import nn, Tensor

MAIN_DEVICE_ID = 0
BATCH_SIZE, NUM_FEATURES, HIDDEN_SIZE = 32, 100, 10

main_device = f"cuda:{MAIN_DEVICE_ID}"

model = nn.Sequential(  # 简单的回归任务
    nn.Linear(NUM_FEATURES, HIDDEN_SIZE),
    nn.ReLU(),
    nn.Linear(HIDDEN_SIZE, 1),
    nn.Flatten(start_dim=0, end_dim=-1)
).to(main_device)

# region DP 部分的代码
model = nn.DataParallel(
    model,
    device_ids=[0, 1],  # 可以使用的 GPU id, 可以配合 `CUDA_VISIBLE_DEVICES` 使用
    output_device=MAIN_DEVICE_ID,  # output 张量的 GPU id, 需要和 target 张量的 GPU id 保持一致
    dim=0  # batch 对应的维度, 在此维度上进行拆分
)
# endregion 

optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

for _ in range(3):
    optimizer.zero_grad()

    input = torch.randn(BATCH_SIZE, NUM_FEATURES).to(main_device)
    target = torch.randn(BATCH_SIZE).to(main_device)

    # 前向传播, 计算 loss
    output: Tensor = model(input)
    loss = torch.nn.functional.mse_loss(output, target)

    # 反向传播, 计算 梯度
    loss.backward()

    # 更新参数
    optimizer.step()

# region DP 部分的代码
model = model.module
# endregion 

model = model.cpu().eval()

# 保存模型
torch.save(model.state_dict(), "demo_dp.bin")

# 加载模型
model.load_state_dict(torch.load("demo_dp.bin"), strict=True)
