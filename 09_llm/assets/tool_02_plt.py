
# %%

import torch 
from torch import Tensor 
import matplotlib.pyplot as plt 

plt.rcParams['font.family'] = 'Microsoft YaHei'

# %%

def unit_step_func(x: Tensor):
    idx = x > 0
    x = x.clone()
    x[idx] = 1
    x[~idx] = 0
    return x 


def sigmoid_derivative(x):
    return torch.sigmoid(x) * (1 - torch.sigmoid(x))

# %%

def normal_dist_cdf(x: Tensor):
    # 均值为 0, 方差为 1 的标准正态分布
    dist = torch.distributions.Normal(
        loc=torch.tensor([0.]), scale=torch.tensor([1.])
    )
    return dist.cdf(x)

def normal_dist_pdf(x: Tensor):
    dist = torch.distributions.Normal(
        loc=torch.tensor([0.]), scale=torch.tensor([1.])
    )
    return torch.exp(dist.log_prob(x))

# %%

with torch.no_grad():
    plt.figure(figsize=(11, 8))
    x_points = torch.linspace(start=-8, end=8, steps=10000)

    plt.subplot(2, 2, 1)
    plt.title("单位阶跃函数", fontdict={'family': 'Microsoft YaHei'})
    y_points = unit_step_func(x_points)
    plt.plot(x_points.numpy(), y_points.numpy())

    plt.subplot(2, 2, 2)
    plt.title("Sigmoid")
    y_points = torch.sigmoid(x_points)
    plt.plot(x_points.numpy(), y_points.numpy())

    plt.subplot(2, 2, 3)
    plt.title("Rectified Linear Unit")
    y_points = torch.relu(x_points)
    plt.plot(x_points.numpy(), y_points.numpy())

    plt.subplot(2, 2, 4)
    plt.title("Sigmoid Linear Unit")
    y_points = torch.nn.functional.silu(x_points)
    plt.plot(x_points.numpy(), y_points.numpy())

    plt.show()

# %%

with torch.no_grad():
    plt.figure(figsize=(11, 8))
    x_points = torch.linspace(start=-8, end=8, steps=10000)

    plt.subplot(2, 2, 1)
    y_points = torch.sigmoid(x_points)
    plt.plot(x_points, y_points, label="Sigmoid 函数")
    y_points = normal_dist_cdf(x_points)
    plt.plot(x_points, y_points, label="标准正态分布 CDF")
    plt.legend(loc='upper left')

    plt.subplot(2, 2, 2)
    y_points = sigmoid_derivative(x_points)
    plt.plot(x_points, y_points, label="Sigmoid 导函数")
    y_points = normal_dist_pdf(x_points)
    plt.plot(x_points, y_points, label="标准正态分布 PDF")
    plt.legend(loc='upper left')

    plt.subplot(2, 2, 3)
    y_points = torch.nn.functional.silu(x_points)
    plt.plot(x_points, y_points, label="SiLU 函数")
    y_points = torch.nn.functional.gelu(x_points)
    plt.plot(x_points, y_points, label="GELU 函数")
    plt.legend(loc='upper left')

# %%

def relu_gradient(x: Tensor):
    x = x.clone().requires_grad_(True)
    torch.nn.functional.relu(x).sum().backward()
    return x.grad.detach()

def gelu_gradient(x: Tensor):
    x = x.clone().requires_grad_(True)
    torch.nn.functional.gelu(x).sum().backward()
    return x.grad.detach()

def silu_gradient(x: Tensor):
    x = x.clone().requires_grad_(True)
    torch.nn.functional.silu(x).sum().backward()
    return x.grad.detach()

x_points = torch.linspace(-8, 8, 10000)

y_points = relu_gradient(x_points)
plt.plot(x_points, y_points, label="ReLU 导函数")

y_points = gelu_gradient(x_points)
plt.plot(x_points, y_points, label="GELU 导函数")

y_points = silu_gradient(x_points)
plt.plot(x_points, y_points, label="SiLU 导函数")

plt.legend()

# %%

@torch.no_grad()
def mlp_func(x: Tensor, act_func, up_bias=True, down_bias=True):
    up_proj = torch.nn.Linear(1, 128, bias=up_bias)
    down_proj = torch.nn.Linear(128, 1, bias=down_bias)

    x = x.unsqueeze(-1)
    x = up_proj(x)
    x = act_func(x)
    x = down_proj(x)
    x = x.squeeze()

    return x 

@torch.no_grad()
def glu_mlp_func(x: Tensor, act_func, up_bias=True, down_bias=True, gate_bias=True):
    up_proj = torch.nn.Linear(1, 128, bias=up_bias)
    down_proj = torch.nn.Linear(128, 1, bias=down_bias)
    gate_proj = torch.nn.Linear(1, 128, bias=gate_bias)

    x = x.unsqueeze(-1)
    y: Tensor = up_proj(x)
    y = act_func(gate_proj(x)) * y
    y = down_proj(y)
    y = y.squeeze()

    return y 

x_points = torch.linspace(-8, 8, 10000)
# y_points = mlp_func(x_points, act_func=lambda x: x)
y_points = mlp_func(x_points, act_func=torch.nn.functional.relu, up_bias=False)
# y_points = glu_mlp_func(x_points, act_func=torch.relu, gate_bias=False, up_bias=False, down_bias=False)

plt.plot(x_points, y_points)

# %%
