
"""
Implement Paper "Visualizing and Understanding Convolutional Networks"
based on project: https://github.com/huybery/VisualizingCNN 
"""

# %% 设置环境变量 以及 加载模型

import os

os.environ["TORCH_HOME"] = "D:\\DatasetModel\\torch"
os.chdir("../")

# %%

import json 

import torch 
from torch import Tensor, nn 

import numpy as np 
from matplotlib import pyplot as plt 


def is_same_tensor(t1: Tensor, t2: Tensor, eps: float = 1e-8) -> bool:
    return t1.shape == t2.shape and (t1 - t2).abs().max().item() < eps


def load_vgg16():
    from torchvision import models
    from torchvision.models import VGG16_Weights

    return models.vgg16(weights=VGG16_Weights.DEFAULT).eval()


def load_alexnet():
    from torchvision import models
    from torchvision.models import AlexNet_Weights

    return models.alexnet(weights=AlexNet_Weights.DEFAULT).eval()


vgg16_model = load_vgg16()
alexnet_model = load_alexnet()
img_path = "assets/cat.jpg"

# %% 加载图片预处理


def load_image(img_path: str) -> Tensor:

    from PIL import Image 
    from torchvision import transforms

    if not os.path.isfile(img_path):
        raise ValueError 

    img = Image.open(img_path)

    if img.mode == "RGBA":
        img = img.convert("RGB")

    transform = transforms.Compose([
        # resize 成标准大小, 返回的依旧是 PIL 对象
        transforms.Resize(size=(224, 224)), 
        # 将 PIL 对象转换为 Tensor 对象, 并除以 255 (千万不要用 PILToTensor)
        transforms.ToTensor(),
        # z-score 标准化
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return transform(img)


load_image(img_path).shape 

# %% 进行推理


@torch.no_grad()
def inference(img_path: str, model: nn.Module) -> str:
    img = load_image(img_path)
    img = img.unsqueeze(0)

    with open("assets/imagenet_class_index.json", "r", encoding="utf-8") as reader:
        id_to_label = {
            int(key): value[1] for key, value in json.load(reader).items()
        }

    logits = model(img).detach().cpu()
    id_ = torch.argmax(logits, dim=1)[0].item()
    return id_to_label[id_]


# tabby
print("alexnet result:", inference(img_path, alexnet_model))
print("vgg16 result:", inference(img_path, vgg16_model))


# %% 第一个卷积层 kernel 可视化


def visualize_kernels(conv_layer: nn.Conv2d):
    from torchvision.utils import make_grid
    assert conv_layer.in_channels == 3

    with torch.no_grad():
        pics = make_grid(
            conv_layer.weight, 
            nrow=int(np.sqrt(conv_layer.out_channels)), 
            normalize=True, scale_each=True,
            padding=1
        )

    plt.imshow(pics.permute(1, 2, 0).detach().numpy())
    plt.axis("off")


plt.subplot(1, 2, 1)
visualize_kernels(alexnet_model.features[0])

plt.subplot(1, 2, 2)
visualize_kernels(vgg16_model.features[0])

# %% feature map 可视化


def visualize_feature_maps(feature_maps: Tensor):
    from torchvision.utils import make_grid
    assert feature_maps.ndim == 3  # [num_fm, h_fm, w_fm]

    with torch.no_grad():
        pics = make_grid(
            feature_maps.unsqueeze(1), 
            nrow=int(np.sqrt(feature_maps.size(0))), 
            normalize=True, scale_each=True,
            padding=1,
        )

    # 对于单通道的图片, make_grid 函数会转成三通道的图片, 并且三个通过颜色是一样的
    # print(pics.shape, torch.all(pics[0] == pics[1]), torch.all(pics[0] == pics[2]))
    plt.imshow(pics[0].detach().numpy())
    plt.axis("off")


plt.figure(figsize=(20, 20))
plt.subplot(1, 2, 1)
visualize_feature_maps(
    alexnet_model.features[0](load_image(img_path))
)

plt.subplot(1, 2, 2)
visualize_feature_maps(
    vgg16_model.features[0](load_image(img_path))
)

# %% CNN 可视化


@torch.no_grad()
def prepare_for_vis(img: Tensor, is_picture: bool = True, do_sharpen: bool = False):

    img = img.permute(1, 2, 0).double()

    if is_picture:  # 如果是真实的图片, 用 mean 和 std 标准化
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        img = (img * std) + mean
    else:  # 如果不是真实的图片, 用 min-max 标准化
        img = (img - img.min()) / (img.max() - img.min())

    img = (img * 255).to(torch.uint8).detach().numpy()

    if do_sharpen:  # 让轮廓更加清晰一些
        from PIL import Image, ImageFilter
        img_ = Image.fromarray(img, mode="RGB")
        img_.filter(ImageFilter.SHARPEN)
        img = np.array(img)

    return img


def gen_output_img(o_img: Tensor, r_img: Tensor):
    from torchvision.utils import make_grid

    # step1: 裁剪
    h_min, h_max, w_min, w_max = r_img.size(1), 0, r_img.size(2), 0
    for c in range(3):
        h_range, w_range = r_img[c].nonzero(as_tuple=True)
        h_min = h_range.min().item() if h_range.min() < h_min else h_min
        h_max = h_range.max().item() if h_range.max() > h_max else h_max
        w_min = w_range.min().item() if w_range.min() < w_min else w_min
        w_max = w_range.max().item() if w_range.max() > w_max else w_max
    o_img = o_img[:, h_min:h_max+1, w_min:w_max+1, ]
    r_img = r_img[:, h_min:h_max+1, w_min:w_max+1, ]

    # step2: 反标准化
    o_img = o_img.permute(1, 2, 0)
    o_img = o_img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
    o_img = o_img.permute(2, 0, 1)

    r_img = (r_img - r_img.min()) / (r_img.max() - r_img.min())

    # step3: 合成一张图片
    combined_img = make_grid([o_img, r_img], nrow=2, padding=4)
    combined_img = combined_img.permute(1, 2, 0)
    return combined_img.detach().numpy()


@torch.no_grad()
def cal_vis_data(img: Tensor, submodel: nn.Sequential, fm_index: int = None) -> tuple[Tensor, int]:

    # 在 Python 中, append 和 pop() 的组合就是 stack (FILO)
    # append 和 pop(0) 的组合就是 queue (FIFO)
    indices_stack = []

    # ## step1: 前向计算
    assert img.ndim == 3 and img.size(0) == 3  # [3, h_img, w_img]
    output = img 
    for layer in submodel:
        if isinstance(layer, nn.Conv2d):
            assert layer.dilation == (1, 1)
            output = layer.forward(output)
        elif isinstance(layer, nn.ReLU):
            layer.inplace = False
            output = layer.forward(output)
        elif isinstance(layer, nn.MaxPool2d):
            layer.return_indices = True
            output, indices = layer.forward(output)
            indices_stack.append(indices)
            layer.return_indices = False
        else:
            raise NotImplementedError
    
    # ## step2: 构建逆向的输入
    if fm_index is None:
        fm_index = torch.argmax(
            torch.max(output.flatten(-2, -1), axis=-1).values
        ).item()

    max_pos = np.unravel_index(  # torch 中居然没有 unravel_index 函数
        indices=torch.argmax(output[fm_index]).detach().numpy(),
        shape=output[fm_index].shape
    )
    r_output = torch.zeros_like(output)
    r_output[fm_index][max_pos] = 1.

    # ## step3: 反向计算
    for layer in reversed(submodel):
        if isinstance(layer, nn.Conv2d):
            r_layer = nn.ConvTranspose2d(
                in_channels=layer.out_channels, out_channels=layer.in_channels, 
                kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding,
                bias=False
            )
            r_layer.weight.data.copy_(layer.weight.data)
            r_output = r_layer.forward(r_output)

        elif isinstance(layer, nn.ReLU):
            r_layer = nn.ReLU()
            r_output = r_layer.forward(r_output)

        elif isinstance(layer, nn.MaxPool2d):
            r_layer = nn.MaxUnpool2d(
                kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding
            )
            r_output = r_layer.forward(r_output, indices=indices_stack.pop())

    return r_output, fm_index


def alexnet_visualization():
    conv_indices = []
    for idx, layer in enumerate(alexnet_model.features):
        if isinstance(layer, nn.Conv2d):
            conv_indices.append(idx)

    img = load_image(img_path)

    n_total = len(conv_indices) + 1
    n_rows = int(np.sqrt(n_total))
    n_cols = int(np.ceil(n_total / n_rows))
    plt.figure(figsize=(n_cols * 4, n_rows * 4))

    plt.subplot(n_rows, n_cols, 1)
    plt.title("original picture")
    plt.imshow(prepare_for_vis(img, is_picture=True))
    plt.axis("off")

    for plot_index, conv_index in enumerate(conv_indices, start=2):
        submodel = alexnet_model.features[:conv_index+1]
        r_output, fm_index = cal_vis_data(img, submodel)

        plt.subplot(n_rows, n_cols, plot_index)
        plt.title(f"No. {conv_index} layer, No. {fm_index} feature map")

        plt.imshow(gen_output_img(img, r_output))
        # plt.imshow(prepare_for_vis(r_output, is_picture=False, do_sharpen=True))

        plt.axis("off")

    plt.show()

alexnet_visualization()

# %%


def vgg16_visualization():
    conv_indices = []
    for idx, layer in enumerate(vgg16_model.features):
        if isinstance(layer, nn.Conv2d):
            conv_indices.append(idx)

    img = load_image(img_path)

    n_total = len(conv_indices) + 1
    n_rows = int(np.sqrt(n_total))
    n_cols = int(np.ceil(n_total / n_rows))
    plt.figure(figsize=(n_cols * 4, n_rows * 4))

    plt.subplot(n_rows, n_cols, 1)
    plt.title("original picture")
    plt.imshow(prepare_for_vis(img, is_picture=True))
    plt.axis("off")

    for plot_index, conv_index in enumerate(conv_indices, start=2):
        submodel = vgg16_model.features[:conv_index+1]
        r_output, fm_index = cal_vis_data(img, submodel)

        plt.subplot(n_rows, n_cols, plot_index)
        plt.title(f"No. {conv_index} layer, No. {fm_index} feature map")
        plt.imshow(gen_output_img(img, r_output))
        plt.axis("off")

    plt.show()


vgg16_visualization()

# %%


from torch.autograd import Function


class RevisedReLU(Function):
    @staticmethod
    def forward(ctx, input):
        return torch.relu(input)
    
    @staticmethod
    def backward(ctx, grad_output):
        return torch.relu(grad_output)


def cal_vis_data_grad_ver(img: Tensor, submodel: nn.Sequential, fm_index: int = None) -> tuple[Tensor, int]:
    for param in submodel.parameters():
        param.requires_grad = False

    assert img.ndim == 3 and img.size(0) == 3
    output = img = nn.Parameter(img)

    for layer in submodel:
        if isinstance(layer, nn.ReLU):
            output = RevisedReLU.apply(output)
        else:
            output = layer(output)

    if fm_index is None:
        fm_index = torch.argmax(
            torch.max(output.flatten(-2, -1), axis=-1).values
        ).item()

    output[fm_index].max().backward()

    return img.grad.detach(), fm_index


def check_grad_ver():
    img = load_image(img_path)

    submodel = vgg16_model.features[:15]
    result1 = cal_vis_data(img, submodel)[0]
    result2 = cal_vis_data_grad_ver(img, submodel)[0]

    print(is_same_tensor(result1, result2, eps=1e-4))


check_grad_ver()

# %% cache


# def check_conv_tconv():
#     conv = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3, bias=False)
#     tconv = nn.ConvTranspose2d(in_channels=10, out_channels=5, kernel_size=3, bias=False)

#     assert conv.weight.shape == tconv.weight.shape 
#     tconv.weight.data.copy_(conv.weight.data)

#     input = nn.Parameter(torch.randn(5, 56, 56))
#     output_grad = torch.randn(10, 54, 54)

#     """
#     在没有 bias 的情况下, Conv2d 对 input 的反向过程 和 ConvTranspose2d 是一致的!
#     """

#     conv(input).backward(output_grad)
#     grad_ver = input.grad 
#     tconv_ver = tconv(output_grad)
#     print(is_same_tensor(grad_ver, tconv_ver, eps=1e-4))


# check_conv_tconv()


# def zero_grad(tensors: list[Tensor]) -> None:
#     # from torch.optim import SGD
#     # optimizer = SGD(tensors, lr=1)
#     # optimizer.zero_grad()
#     for tensor in tensors:
#         tensor.grad = None


# def check_zero_grad():
#     tensors = list(vgg16_model.parameters())

#     input = torch.randn(1, 3, 224, 224)
#     output_grad = torch.randn(1, 1000)

#     # first
#     output1 = vgg16_model(input)
#     output1.backward(output_grad)
#     grad1 = tensors[0].grad.detach()
#     zero_grad(tensors)

#     # second
#     output2 = vgg16_model(input)
#     output2.backward(output_grad)
#     grad2 = tensors[0].grad.detach()
#     zero_grad(tensors)

#     print(torch.allclose(grad1, grad2))


# check_zero_grad()
