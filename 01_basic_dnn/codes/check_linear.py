
# %%

import torch 
from torch import nn, Tensor 
import matplotlib.pyplot as plt 


class UnitStep(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input):
        return torch.heaviside(input, values=torch.zeros(1).to(input.device))


# %%

model = nn.Sequential(
    nn.Linear(1, 5, bias=True), UnitStep(), nn.Linear(5, 1, bias=True)
)

nn.init.uniform_(model[0].weight, 0, 2)
nn.init.uniform_(model[0].bias, -2, 2)
nn.init.uniform_(model[2].weight, 0, 2)
nn.init.uniform_(model[2].bias, -2, 2)

input = torch.linspace(-20, 20, 10000).float().unsqueeze(-1)
with torch.no_grad():
    output = model(input)

plt.plot(
    input.squeeze().cpu().detach().numpy(),
    output.squeeze().cpu().detach().numpy()
)

# %%

model = nn.Sequential(
    nn.Linear(1, 100), UnitStep(),
    nn.Linear(100, 100), UnitStep(),
    nn.Linear(100, 1)
)

for idx in range(0, len(model) - 1, 2):
    nn.init.uniform_(model[idx].weight, 0, 2)
    nn.init.uniform_(model[idx].bias, -2, 2)

input = torch.linspace(-200, 200, 10000).float().unsqueeze(-1)
with torch.no_grad():
    output = model(input)

plt.plot(
    input.squeeze().cpu().detach().numpy(),
    output.squeeze().cpu().detach().numpy()
)

# %%

plt.rcParams["mathtext.fontset"] = "cm"

input = torch.linspace(-2, 2, 10000).float().unsqueeze(-1)

plt.figure(figsize=(12, 4))
plt.suptitle("图八: ReLU 激活函数示意图", fontdict={"family": "SimSun",})

# ## part1
plt.subplot(1, 3, 1)
plt.title("ReLU 函数图像", fontdict={"family": "SimSun",})
plt.plot(
    input.squeeze().numpy(), torch.relu(input).squeeze().numpy()
)
plt.xlabel("$z$", fontsize=15)
plt.ylabel("$a$", fontsize=15, rotation=0)

# ## part2
plt.subplot(1, 3, 2)
plt.title("拟合函数图像 (10 个神经元)", fontdict={"family": "SimSun",})

model = nn.Sequential(
    nn.Linear(1, 10), nn.ReLU(),
    nn.Linear(10, 1)
)
with torch.no_grad():
    output = model(input)

plt.plot(
    input.squeeze().cpu().detach().numpy(),
    output.squeeze().cpu().detach().numpy()
)

plt.xlabel("$x$", fontsize=15)
plt.ylabel("$\hat{y}$", fontsize=15, rotation=0)

# ## part3
plt.subplot(1, 3, 3)
plt.title("拟合函数图像 (100 个神经元)", fontdict={"family": "SimSun",})

model = nn.Sequential(
    nn.Linear(1, 100), nn.ReLU(),
    nn.Linear(100, 1)
)
with torch.no_grad():
    output = model(input)

plt.plot(
    input.squeeze().cpu().detach().numpy(),
    output.squeeze().cpu().detach().numpy()
)

plt.xlabel("$x$", fontsize=15)
plt.ylabel("$\hat{y}$", fontsize=15, rotation=0)

plt.tight_layout()

# %%

model = nn.Sequential(
    nn.Linear(1, 5), nn.Sigmoid(),
    nn.Linear(5, 5), nn.Sigmoid(),
    nn.Linear(5, 1)
)

input = torch.linspace(-1, 1, 10000).float().unsqueeze(-1)
with torch.no_grad():
    output = model(input)

plt.plot(
    input.squeeze().cpu().detach().numpy(),
    output.squeeze().cpu().detach().numpy()
)

# %%

model = nn.Sequential(
    nn.Linear(1, 500), nn.Sigmoid(),
    # nn.Linear(500, 500), nn.Sigmoid(),
    nn.Linear(500, 1)
)

# nn.init.uniform_(model[0].weight, 0, 2)
# nn.init.uniform_(model[0].bias, -2, 2)
# nn.init.uniform_(model[2].weight, 0, 2)
# nn.init.uniform_(model[2].bias, -2, 2)

input = torch.linspace(-50, 50, 10000).float().unsqueeze(-1)
with torch.no_grad():
    output = model(input)

plt.plot(
    input.squeeze().cpu().detach().numpy(),
    output.squeeze().cpu().detach().numpy()
)

# %%

import warnings
import numpy as np 
from numpy import ndarray

from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning


# %%

features = np.random.randn(1000, 1)
# targets = np.cos(np.power(features, 6)).flatten()
targets = 0.2 + 0.4 * np.square(features) 
targets = targets + 0.3 * features * np.sin(15 * features) 
targets = targets + 0.05 * np.cos(50 * features)
targets = targets.flatten()

plt.scatter(features.flatten(), targets, s=1)

# %%

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    predictor = MLPRegressor(
        hidden_layer_sizes=(10, ), 
        activation="relu",
        solver="lbfgs", 
        max_iter=1000
    )

    predictor.fit(features, targets)

plt.scatter(features.flatten(), targets, s=1)

inputs = np.expand_dims(np.linspace(-3.5, 3.5, 1000), -1)
outputs = predictor.predict(inputs)

plt.plot(inputs.flatten(), outputs, color="red")
    
total_params = 0

for weight, bias in zip(predictor.coefs_, predictor.intercepts_):
    total_params += weight.size
    total_params += bias.size
    
print(f"总共有 {total_params} 个参数")

# %%

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    predictor = MLPRegressor(
        hidden_layer_sizes=(10, 10, 10, 10), 
        activation="relu",
        solver="lbfgs", 
        max_iter=10000
    )

    predictor.fit(features, targets)

plt.scatter(features.flatten(), targets, s=1)

inputs = np.expand_dims(np.linspace(-3.5, 3.5, 1000), -1)
outputs = predictor.predict(inputs)

plt.plot(inputs.flatten(), outputs, color="red")

total_params = 0

for weight, bias in zip(predictor.coefs_, predictor.intercepts_):
    total_params += weight.size
    total_params += bias.size
    
print(f"总共有 {total_params} 个参数")

# %%

outputs_ = inputs


def relu(a: ndarray):
    return np.maximum(a, 0)


for weight, bias in zip(predictor.coefs_[:-1], predictor.intercepts_[:-1]):
    outputs_ = relu(outputs_ @ weight + bias)

outputs_ = outputs_ @ predictor.coefs_[-1] + predictor.intercepts_[-1]

outputs_ = outputs_.flatten()

print(np.allclose(outputs_, outputs))

print(np.alltrue(
    np.abs(outputs - outputs_) < 1e-10
))

# %%

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    predictor = MLPRegressor(
        hidden_layer_sizes=(100), 
        activation="relu",
        solver="lbfgs", 
        max_iter=10000
    )

    predictor.fit(features, targets)

plt.scatter(
    features.flatten(), targets, s=1
)

inputs = np.expand_dims(np.linspace(-3.5, 3.5, 1000), -1)
outputs = predictor.predict(inputs)

plt.plot(inputs.flatten(), outputs, color="red")

total_params = 0

for weight, bias in zip(predictor.coefs_, predictor.intercepts_):
    total_params += weight.size
    total_params += bias.size
    
print(f"总共有 {total_params} 个参数")

# %%
