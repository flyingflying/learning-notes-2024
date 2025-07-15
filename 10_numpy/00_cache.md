
# NumPy 杂项

## 1. 等差数列 和 等比数列

`np.linspace(start, stop, num)` 则是将 $[\mathrm{start, stop}]$ 区间等分成 $\mathrm{num} - 1$ 份, 最终将 $\mathrm{num}$ 个端点值组合成 一维数组 返回。另一种理解方式是: 返回的一维数组是 首项 为 $\mathrm{start}$, 公差 为 $\mathrm{\frac{stop-start}{num-1}}$, 项数 为 $\mathrm{num}$ 的等差数列 (arithmetic progression)。

`np.logspace(start, stop, num, base)` 返回的一维数组在 [对数尺度](https://en.wikipedia.org/wiki/Logarithmic_scale) 下是等距的, 即返回的一维数组经过对数运算后是等距的。你可以使用 `np.diff(np.log10(start, stop, num))` 进行验证。实现方式也很简单: `np.power(base, np.linspace(start, stop, num))`。也就是说, 最终返回的一维数组所有的 item 都是正数。

`np.geomspace(start, stop, num)` 返回的一维数组是 首项 为 $\mathrm{start}$, 公比 为 $\mathrm{\sqrt[num-1]{\frac{stop}{start}}}$, 项数 为 $\mathrm{num}$ 的等比数列 (geometric progression)。

等比数列在 对数尺度 下就是 等差数列。假设等比数列的首项为 $a$, 公比为 $r$, 那么通项公式是 $a_n = a \cdot r^{n-1}$。现在转换到 对数尺度 下, 我们可以进行如下的计算:

$$
\begin{align*}
    \log(a_n) &= \log(a \cdot r^{n-1}) \\
    &= \log(a) + \log(r^{n-1}) \\
    &= \log(a) + (n - 1) \cdot \log(r)
\end{align*}
$$

此时获得的公式正好是等差数列的通项。那么, 我们可以借助 `np.linspace` 来实现 `np.geomspace` 函数, 方式如下:

```python
def demo_geomspace(start, stop, num):
    # step1: 转换到 对数尺度 下
    start = np.log10(start)
    stop = np.log10(stop)

    # step2: 在 对数尺度 下创建等差数列
    sequence = np.linspace(start, stop, num)

    # step3: 转换回原本的尺度
    sequence = np.power(10., sequence)
    return sequence
```

在上面的代码中, 第二步和第三步可以合并成 `np.logspace`。`np.geomspace` 的实现方式和上述代码差不多, 额外添加了 符号 的判断。从这里可以看出, `np.geomspace` 不适用于 公差 是负数的情况, 即 $\mathrm{start}$ 和 $\mathrm{stop}$ 入参有一个是负数的情况。

对数尺度 非常重要: 人类在设计 度量 时基本使用的都是 正数; 自然界很多东西的增长也呈现指数特性 (即相对于上一个状态增量不变)。这些条件完美契合了 对数运算 的要求。也正是因为此, 在 实数乘方 中, 指数位 比 底数位 重要得多。在没有合适 底数位 的情况下, 自然对数 $e$ 是一个很好地选择。
