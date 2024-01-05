
# %%

import numpy as np 
from numpy import ndarray
import matplotlib.pyplot as plt 


def unit_step(a: ndarray):
    return np.heaviside(a, 0)


def sigmoid(a: ndarray):
    return 1. / (1. + np.exp(-a))


def sigmoid_grad(a: ndarray):
    output = sigmoid(a)
    return output * (1 - output)


x = np.linspace(-10, 10, 1000)

plt.rcParams["mathtext.fontset"] = "stix"

plt.figure(figsize=(12, 4))
plt.suptitle("图三: $\mathrm{sigmoid}$ 函数相关图像", fontdict={"family": "SimSun",})
plt.subplot(1, 3, 1)
plt.title("unit step function", fontdict={"family": "Times New Roman"})
plt.plot(x, unit_step(x))

plt.subplot(1, 3, 2)
plt.title("sigmoid function", fontdict={"family": "Times New Roman"})
plt.plot(x, sigmoid(x))

plt.subplot(1, 3, 3)
plt.title("derivative of sigmoid function", fontdict={"family": "Times New Roman"})
plt.plot(x, sigmoid_grad(x))

plt.tight_layout()
# %%

import torch 

model = torch.nn.Linear(5, 1)
loss_fn = torch.nn.BCEWithLogitsLoss()

x = torch.randn(5)
z = model(x)

loss_fn(z, torch.tensor([1., ])).backward()

print(model.weight.grad)
print((torch.sigmoid(z) - 1) * x)
print(model.bias.grad)
print((torch.sigmoid(z) - 1))


# %%

import torch 

model = torch.nn.Linear(5, 1)
loss_fn = torch.nn.BCEWithLogitsLoss()

x = torch.randn(3, 5)
z = model(x)
target = torch.tensor([1., 0., 1.]).unsqueeze(-1)

loss_fn(z, target).backward()

print(model.weight.grad)
print(torch.mean((torch.sigmoid(z) - target) * x, dim=0))
print(model.bias.grad)
print(torch.mean((torch.sigmoid(z) - target), dim=0))

# %%

# latex support

import os

os.environ["path"] = r"D:\Software-2024\texlive\2023\bin\windows;" + \
    "D:\Software-2024\texlive\2023\texmf-dist;" + os.environ["path"]

# import matplotlib
import matplotlib.pyplot as plt 

# matplotlib.use("pgf")

# plt.rcParams.update({
#     'text.usetex': True,
#     'text.latex.preamble': r'\usepackage{CJK}',
# })

# %%

import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams["mathtext.fontset"] = "cm"
figure = plt.figure(figsize=(8, 10))
plt.suptitle("图一: 感知机原理图", fontdict={"family": "SimSun", "size": 14})

# ## 一维特征空间

axes = figure.add_subplot(3, 2, 1)
pos_examples = np.random.uniform(low=-1, high=1, size=(10, ))
neg_examples = np.random.uniform(low=3, high=5, size=(10, ))
axes.scatter(
    pos_examples, np.zeros_like(pos_examples), color="blue"
)
axes.scatter(
    neg_examples, np.zeros_like(neg_examples), color="orange"
)
axes.scatter(
    2, 0, color="red"
)
axes.set_title("一维特征空间", fontdict={"family": "SimSun"})
axes.set_xlabel("$x_1$", fontsize=14)
axes.set_yticks([])

# ## 一维特征函数图像

axes = figure.add_subplot(3, 2, 2)
axes.plot(np.linspace(-1, 2, 100), np.zeros(100), color="blue")
axes.plot(np.linspace(2, 5, 100), np.ones(100), color="orange")
axes.plot(np.ones(100) * 2, np.linspace(0, 1, 100), color="red")
axes.set_title("一维特征函数图像", fontdict={"family": "SimSun"})
axes.set_xlabel("$x_1$", fontsize=14)
axes.set_ylabel("$\hat{y}$", rotation=0, fontsize=14)

# ## 二维特征空间

axes = figure.add_subplot(3, 2, 3)
x = y = np.linspace(-3, 3, 100)
axes.plot(x, -y, color="red")
# x = np.random.uniform(low=-2, high=0, size=(10, ))
# y = x - np.random.rand(10)
# axes.scatter(x, y, color="blue")
# x = np.random.uniform(low=0, high=2, size=(10, ))
# y = x + np.random.rand(10)
# axes.scatter(x, y, color="orange")

x1 = np.random.uniform(low=-2.5, high=2.5, size=(100, ))
x2 = np.random.uniform(low=-2.5, high=2.5, size=(100, ))
pos_idx = x1 + x2 > 0.5
neg_idx = x1 + x2 < -0.5
axes.scatter(x1[pos_idx], x2[pos_idx], color="orange")
axes.scatter(x1[neg_idx], x2[neg_idx], color="blue")
axes.set_title("二维特征空间", fontdict={"family": "SimSun"})
axes.set_xlabel("$x_1$", fontsize=14)
axes.set_ylabel("$x_2$", rotation=0, fontsize=14)

# ## 二维特征函数图像

colors = [
    (0, 'blue'), (1, 'orange')
]

cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)

axes = figure.add_subplot(3, 2, 4, projection="3d")
xx, yy = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))
zz = xx - yy
zz[zz > 0] = 1
zz[zz < 0] = 0.
axes.plot_surface(xx, yy, zz, alpha=0.3, cmap=cmap)
xx, zz = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(0, 1, 100))
yy = xx
axes.plot_surface(xx, yy, zz, alpha=0.4, color='red')
axes.set_title("二维特征函数图像", fontdict={"family": "SimSun"})

axes.set_xlabel("$x_1$", fontsize=14)
axes.set_ylabel("$x_2$", fontsize=14)

axes.zaxis.set_rotate_label(False)
axes.set_zlabel("$\hat{y}$", fontsize=14)
axes.set_zticks([0.0, 0.5, 1.0])
# axes.set_box_aspect(aspect=None, zoom=0.8)

# ## 三维特征空间

axes = figure.add_subplot(3, 2, 5, projection="3d")
# x = np.random.uniform(low=-2, high=2, size=(10, ))
# y = np.random.uniform(low=-2, high=2, size=(10, ))
# z = x + y + np.random.uniform(low=2, high=3, size=(10, ))
# axes.scatter(x, y, z, color="blue")
# x = np.random.uniform(low=-2, high=2, size=(10, ))
# y = np.random.uniform(low=-2, high=2, size=(10, ))
# z = x + y - np.random.uniform(low=2, high=3, size=(10, ))
# axes.scatter(x, y, z, color="orange")
x = np.random.uniform(low=-2, high=-1, size=(10, ))
y = np.random.uniform(low=-2, high=2, size=(10, ))
z = np.random.uniform(low=-2, high=2, size=(10, ))
axes.scatter(x, y, z, color="blue")
x = np.random.uniform(low=1, high=2, size=(10, ))
y = np.random.uniform(low=-2, high=2, size=(10, ))
z = np.random.uniform(low=-2, high=2, size=(10, ))
axes.scatter(x, y, z, color="orange")
yy, zz = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))
xx = np.zeros_like(yy)
axes.plot_surface(xx, yy, zz, color="red", alpha=0.3, )
axes.set_title("三维特征空间", fontdict={"family": "SimSun"})
axes.set_xlabel("$x_1$", fontsize=14)
axes.set_ylabel("$x_2$", fontsize=14)
axes.zaxis.set_rotate_label(False)
axes.set_zlabel("$x_3$", fontsize=14)

plt.tight_layout()

# %%

import numpy as np 
from numpy import ndarray
import matplotlib.pyplot as plt 


def unit_step(a: ndarray):
    return np.heaviside(a, 0)


x = np.linspace(-10, 10, 1000)

plt.rcParams["mathtext.fontset"] = "cm"

plt.figure(figsize=(8, 4))
plt.suptitle('图七: "单步" 阶跃函数示意图', fontdict={"family": "SimSun",})

plt.subplot(1, 2, 1)
plt.title('"single" step function ($w > 0$)', fontdict={"family": "Times New Roman"})
plt.plot(x, unit_step(3 * x))
plt.xticks([0, ], [r"- $\frac{b^h}{w^h}$"], fontsize=15)
plt.xlabel("$x$", loc="right", fontsize=15, labelpad=-20)
plt.ylabel("$a$", loc="top", rotation=0, fontsize=15, labelpad=-20)

plt.subplot(1, 2, 2)
plt.title('"single" step function ($w < 0$)', fontdict={"family": "Times New Roman"})
plt.plot(x, unit_step(-3 * x))
plt.xticks([0, ], [r"- $\frac{b^h}{w^h}$"], fontsize=15)
plt.xlabel("$x$", loc="right", fontsize=15, labelpad=-20)
plt.ylabel("$a$", loc="top", rotation=0, fontsize=15, labelpad=-20)

plt.tight_layout()

# %%

import numpy as np 
from numpy import ndarray
import matplotlib.pyplot as plt 


def relu(array: ndarray) -> ndarray:
    return np.maximum(array, 0)


x = np.expand_dims(np.linspace(-1, 1, 1000), axis=1)  # [1000, 1]
w_h = np.random.randn(1, 10)
b_h = np.random.randn(10)
z = np.matmul(x, w_h) + b_h
a = relu(z)
print(a)
w_o = np.random.randn(10, 1)
y_hat = np.matmul(a, w_o)

plt.plot(x, y_hat)


# %%
