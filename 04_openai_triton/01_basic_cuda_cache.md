
# 计算机基础知识

[TOC]

## 硬件

学过 计算机组成原理 的应该知道, [CPU](https://en.wikipedia.org/wiki/Central_processing_unit) 主要由 [控制单元](https://en.wikipedia.org/wiki/Control_unit) 和 [运算单元](https://en.wikipedia.org/wiki/Arithmetic_logic_unit) 构成。**控制单元** (CU) 的主要功能有: (1) 读取、分析 和 执行指令; (2) 控制程序的 输入 和 输出; (3) 处理异常情况。如果执行的指令中包含 逻辑运算 或者 算术运算, 那么 CU 会调用 **运算单元** (ALU) 完成。

在 CPU 中, 除了 CU 和 ALU 外, 还有 存储单元, 我们一般称之为 [寄存器](https://en.wikipedia.org/wiki/Processor_register) (register)。只有当数据在 寄存器 中时, ALU 才能进行运算; 如果不在, 我们就需要从 **内存** (main memory) 中读取数据。然而, 从 内存 中读取数据的过程是非常缓慢的, 于是, 工程师们就在 CPU 中设置了 "多级缓存"。在读取数据的过程中, CU 会先从 第一级缓存 (L1 cache) 中读取, 如果没有, 再到 第二级缓存 (L2 cache) 中读取, 如果还没有, 再到 内存 中读取。我们一般用 [静态随机访问存储器](https://en.wikipedia.org/wiki/Static_random-access_memory) (SRAM) 来构建 CPU 中的 "多级缓存", 用 [动态随机访问存储器](https://en.wikipedia.org/wiki/Dynamic_random-access_memory) (DRAM) 来构建 **内存** (main memory)。因此, 内存有时也被称为 DRAM。

CPU 在不同的语境下, 含义是不相同的。在日常生活中, 我们会将上面所说的 CPU 称为 **CPU 核** (core)。一般情况下, 一个 **CPU 核** 有一套完成的 CU, ALU, 寄存器 和 L1 cache, 而 CPU 则是由多个 **CPU 核** 构成的, 这种 CPU 也被称为 **多核 CPU** (multicore CPU)。

线程 也是一个在不同语境下含义不相同的概念。操作系统 中的 "线程" 和 CPU 中的 "线程" 概念完全不同! 在操作系统中, "线程" 和 "进程" 的概念是相关的, "进程" 有独立的资源分配, 而 "线程" 没有, 这里就不详细介绍了。而在 CPU 中, "线程" 表示的是工艺的改进: 一个 **CPU 核** 有两个线程 的意思是 一个 **CPU 核** 可以 "模拟" 两个 **CPU 核**。从本质上来说, 就是改进 CPU 的调度策略。在一些科普视频中, 将 "CPU" 类比成 "工厂", "CPU 核" 类比成 "工人", "CPU 线程" 类比成 "传送带", 我觉得非常形象。有时, 我们将 "CPU 核" 称为 "物理核" 或者 "内核", "CPU 线程" 称为 "逻辑核" 或者 "逻辑处理器"。比方说, 对于一个 "八核十六线程" 的 CPU, 我们可以说其有 8 个 "物理核", 16 个 "逻辑核"。

## 堆栈

**堆** (heap) 和 **栈** (stack, 和 "站" 同音) 这两个概念在 **数据结构** 和 **程序内存** 中的含义是不相同的, 千万不能弄混淆。

在 **数据结构** 中, **栈** (stack) 和 **队列** (queue) 两者之间是一组概念:

+ **队列** (queue) 是 FIFO (First In First Out) 的, 即先进入容器的元素先出来。在 Python 中, `list` 的 `append(item)` 和 `pop()` 方法就可以模拟 queue 数据结构。
+ **栈** (stack) 是 FILO (First In Last Out) 的, 即先进入容器的元素后出来。在 Python 中, `list` 的 `append(item)` 和 `pop(0)` 方法就可以模拟 stack 数据结构。

在 **数据结构** 中, **堆** 是一种特殊的 **完全二叉树**, 主要用于 [堆排序](https://en.wikipedia.org/wiki/Heapsort) 中。其含义如下:

+ **二叉树** 指的是每一个结点只有两个子结点的树
+ **满二叉树** 指的是除了 叶子结点 外其它所有的 结点 都有两个子结点
+ **完全二叉树** 指的是除最后一层 结点 外，其他层的 结点 都必须要有两个子结点，并且最后一层的结点需要左排列
+ **堆** 则是在 **完全二叉树** 的基础上, 要求所有 父结点 的值 总是 大于 (小于) 或者等于 两个子结点的值 (前提是不同结点存储的值是 可比较的)

更多的内容可以参考 堆排序, 这里就不详细介绍了。额外说明一点: **节点** 和 **结点** 的区别:

+ **节点** 一般指 直的东西 上的点, 比方说 竹子的竹节, 人体四肢的关节
+ **结点** 一般指 交叉点, 比方说 结绳记事
+ **节点** 现在常用于表示有实体的东西, 比方说 计算机, 交换器 等等, 因此常用于 **计算机网络** 中, 比方说 网络节点
+ **结点** 则常用于 数据结构 和 算法 中, 树结构 和 图结构 中的 "node" 一般翻译成 "结点"

在 **程序内存** 中, 栈 (stack) 是由程序自动分配和释放的, 主要用于存放函数的参数值, 局部变量等等内容。在部分操作系统上, 程序员可以用 `alloca` 函数主动分配, 但是不推荐使用。

在 **程序内存** 中, 堆 (heap) 是由程序员用 `malloc` 函数主动分配的内存。内存释放有两种方式: (1) 由程序员用 `free` 函数主动释放; (2) 程序结束时自动释放。

一般情况下, 栈区域 对应 CPU 的 L1 cache 或者 L2 cache 区域, 堆区域 对应 电脑的主内存区域。而 L1 cache 和 L2 cache 一般是用 SRAM 构建的, 主内存区域一般是用 DRAM 构建的, 因此有时我们会说, 栈区域 对应 SRAM, 堆区域 对应 DRAM。但是, 这不绝对!

栈区域 和 栈数据结构 的思想是相似的, 都是采用 FILO 的方式。举例来说, 假设我们在 `main` 函数中调用了 `add` 函数, 此时肯定是先申明 `main` 函数变量的内存, 再申明 `add` 函数变量的内存。由于 `add` 函数是先调用完成的, 那么肯定是先释放 `add` 函数的内存, 再释放 `main` 函数的内存。这样恰好符合 FILO 的思想。

在栈数据结构中, 我们将第一个入栈的元素位置称为 **栈底**, 最后一个入栈的元素位置称为 **栈顶**。也就是说, **栈底** 是一个固定的位置, 不会随着元素的增加或者减少而改变。而 **栈顶** 的位置是一直在变动的。在 Python 中, 如果我们用 `list` 的 `append(item)` 和 `pop()` 方法来模拟栈数据结构, 那么 **栈底** 始终是数组的第一个位置, 对应索引值是 `0`。当有元素入栈时, **栈顶** 的索引值增加; 当有元素出栈时, **栈顶** 的索引值减小。

如果我们将 "栈区域" 看成是一个 "数组", 那么 "变量地址" 就对应 "数组索引"。在程序内存中, 我们不会将 "数组" 的第一个位置作为 **栈底**, 而是将 "数组" 的最后一个位置作为 **栈底**。这也就意味着, 当有变量入栈时, **栈顶** 的 "索引值" 减小; 当有变量出栈时, **栈顶** 的 "索引值" 增加。因此, 在 C 或者 C++ 中, 后申明的变量地址 会小于 先申明的变量地址。

在一些文章中, 会将一个程序占用的内存分成五部分: 代码区, 常量区, 全局区, 堆区域 和 栈区域。代码区的内存地址最小, 常量区其次, 全局区再次, 然后就是 堆区域。堆区域由 `malloc` 分配内存, 后申请的地址大于先申请的地址。而 栈区域 则是从内存地址最大处开始分配, 后申请的地址小于先申请的地址。堆区域 和 栈区域 的内存增长是 "相向而行" 的, 如果相遇, 程序内存就会报错, 无法继续执行。当然, 这也不绝对!

如果在 Java 中, 创建的对象一般放在 堆区域 中, 而函数调用的信息放在 栈区域 中。因此, 如果你使用 递归 导致了过多的函数调用, 栈区域 内存不够, 会报 `StackOverflowError`。如果创建过多的 对象, 导致 堆区域 内存不够, 会报 `OutOfMemoryError`。

## 编译

**编译器** 的本质就是 **程序**, 这个 **程序** 的功能就是将我们写的代码转换成另一种形式的代码。举例来说, gcc 编译器就是将 C 语言代码 转换成 机器码。学习过 计算机组成原理 应该知道, CPU 中的 控制单元 (CU) 可以读取、分析和执行的就是 机器码。我们将这个有 机器码 组成的文件称为 **程序**。从这里可以看出, **编译器** 的本质就是一堆 机器码 而已。

一般情况下, 在相似配置的计算机和操作系统中, 程序 都是可以执行的, 不需要重新编译。编译程序是一件非常繁琐的事情, 很容易失败, 因此很多软件在发布时, 都已经事先编译完成了, 安装过程更多的是解压资源, 配置目录等等。

从上面可以看出, 程序 一旦编译完成了, 就和 编译器 无关了, 只和 计算机的硬件 以及 操作系统 相关。我们甚至可以写一个编译器, 其只能在 Windows 系统下运行, 生成的 程序 只能在 Linux 系统上运行。当然, 这样的难度是非常高的。

此时, 你可能会产生疑问, gcc 这个 程序 是怎么生成的呢? 答案是 高版本的 gcc 程序用 低版本的 gcc 程序生成。举例来说:

假设现在 Linux 系统从 500 升级到 600, 假设现在有 gcc 1000 编译器, 可以在 Linux 500 上面运行, 不能在 Linux 600 上运行, 现在需要开发一个新的 gcc 版本, 适配 Linux 600。应该怎么办呢?

答案是: 修改 gcc 1000 的源代码, 适配 Linux 600, 然后在 Linux 500 上编译, 确保生成的 新编译器 程序可以在 Linux 600 上运行。最后, 我们将新的 gcc 编译器命名为 gcc 1200。

上述过程有一个专门的名称: [自举](https://en.wikipedia.org/wiki/Bootstrapping_(compilers)) (bootstrapping)。这对于一个编程语言来说, 非常重要。

此时, 你可能有新的问题了: 第一个 gcc 程序是怎么诞生的呢? 答案是用其它 编译器 编译生成的, 比方说 TinyCC。那么, 你可能又要问了, 第一个 编译器程序 是怎么诞生的呢? 答案是 程序员 手写机器码 (手动编译)。从这里可以看出, 计算机的发展真的很不容易, 是无数人智慧的结晶。网上有人列出从零编译 gcc 的过程: "[gcc是怎么写出来的](https://www.zhihu.com/question/280089090/answer/2721015687)", 有兴趣的可以看看。

换一个角度来说, 编译器 就是 工具。对于使用者来说, 其是用什么语言, 什么方式实现的并不重要, 重要的是能在 目标机器 上运行即可。

gcc 从第一版开始就是用 C / C++ 实现的, 一直延续至今。除此之外, rust 的情况也是相似的。第一个 rust 编译器是用 OCaml 实现的, 之后都是用 rust 旧版本 不断地去编译 新版本。bootstrap 的英文解释是: the technique of starting with existing resources to create something more complex and effective. 正好符合这里的语境!

C / C++ / rust 这些语言编译后生成的是 机器码, 它们的运行只依赖于 硬件设备 和 操作系统。除此之外, 还有一类语言, 它们编译后生成的不是 机器码, 而是自定义的其它代码, 它们的运行依赖的也不是操作系统, 而是其它程序, 最经典的就是 Java 语言了。

对于 Java 来说, 其编译过程使用的是 javac 工具, 其作用是将 **Java 代码** 转换成 **Java 字节码**。操作系统 是没有办法识别 Java 字节码的, 在启动时, 先运行 JVM (Java Virtual Machine) 程序, 再由 JVM 程序 读取, 识别 和 执行 Java 字节码中的指令。JVM 相当于代替了 操作系统 的工作。这样的好处是可以 跨平台移植: 对于不同的操作系统, 只需要编译一个 JVM 即可, 不需要修改代码。

换言之, 运行 Java 程序必须先安装 JVM, 那么问题来了, javac 工具可以 "自举" 实现吗? 答案是可以的。假设最初的 Java 版本号是 Java 1.0, 其内部的 javac 和 JVM 都是用 C 语言开发的, 并且已经编译好, 发布成功了, 我们现在要开发 Java 2.0, 流程如下:

1. 开发 JVM 2.0, 确定好 Java 2.0 的运行环境
2. 用 Java 1.0 编写 javac 2.0 的代码
3. 用 javac 1.0 编译代码, 生成 Java 字节码文件
4. 用 JVM 1.0 运行字节码文件, 生成 javac 2.0
5. 检测 javac 2.0 能否在 JVM 2.0 上正常运作

需要注意的是, 很多人 (包括我), 会陷入 "javac 1.0 编译后的代码只能在 jvm 1.0 上运行" 的陷阱中, 认为 javac 没有办法 "自举"。这是对编译器的功能没有理解透彻。javac 1.0 编译后的代码确实只能在 JVM 1.0 上运行, 但是其运行生成的 javac 2.0 能在 JVM 2.0 上运行即可。因此, 如果你想自行 build Java JDK, 需要电脑中先有一个 JVM!

"编译器" 也是程序, 也有 "代码", "编译" 和 "运行" 阶段。新的 "编译器" 是 "运行" 生成的, 不是 "编译" 生成的, 不要将概念弄混淆了。

"编译" 的本质是将代码从一种形式转换成另一种形式。那么, 有没有不用 "编译", 用户写的代码可以直接运行的语言呢? 有, 最经典的就是 Python。

学过编译原理应该知道, 机器码一般不是一步就生成的, 而是分步生成的。对于 C/C++ 而言, 是先生成 汇编语言, 再从汇编语言生成机器码。汇编语言 就是 中间语言。我们一般将从 C/C++ 生成 汇编语言 的过程 称为 **前端**; 从 汇编语言 生成 机器码 的过程称为 **后端**。

C/C++ 有多种编译器, 比方说: msvc, gcc, clang, llvm 等等。目前, rust 语言的前端就是用 rust 实现的, 而后端则是直接用的 llvm。

## 指针

在计算机中, 我们可以将 **内存** 看成是一个 **字节数组**, 那么 **内存地址** 对应 **字节数组的索引**。

在 C 语言中, 基本的数据类型包括: `char`, `short`, `int`, `long`, `float`, `double` 等等, 它们占用 1-8 个字节, 我们也可以将其理解为 **字节数组**。机器 是不知道 内存 中存储的数据是 什么类型 的, 只知道按照指令要求去执行。

在 C 语言中, **指针** 的本质是 **内存地址**, 或者说 **字节数组的索引**。

一般情况下, C 代码 `int a = 100;` 对应的 汇编代码 是: `mov DWORD PTR [rbp-4], 100`。其含义如下:

+ `mov dest source` 表示将 `source` 值赋给 `dest`
+ 在进入 C 语言函数时, 会初始化一个 `rbp` 寄存器, 表示当前栈区域 "栈顶"
+ `DWORD PTR [rbp-4]` 表示的就是变量 `a`, 其内存地址是 `rbp-4`
+ 整个语句的含义是将内存 `rbp-4` 到 `rbp` 的位置赋值为 100 (采用 "补码" 的形式)

在这之后, 变量 `a` 都会以 `DWORD PTR [rbp-4]` 的形式出现。比方说, `a = a + 1;` 会被解析成 `add DWORD PTR [rbp-4], 1`。

再比方说:

```C
int a;
int *a_ptr = &a;
*a_ptr = 200;
```

这段代码编译成 汇编 是:

```asm
; `int *a_ptr = &a;` 对应的汇编代码:
lea     rax, [rbp-12]
mov     QWORD PTR [rbp-8], rax

; `*a_ptr = 200` 对应的汇编代码:
mov     rax, QWORD PTR [rbp-8]
mov     DWORD PTR [rax], 200
```

前两句的含义如下:

+ `int a;` 仅仅是申明, 不对应任何的汇编代码
+ `lea rax, [rbp-12]` 表示将 `rbp-12` 的值赋给 `rax` 寄存器中
+ `lea` 指令和 `mov` 指令相似, 只是在加载地址时更加高效
+ `QWORD PTR [rbp-8]` 表示的就是 `a_ptr` 变量
+ `mov QWORD PTR [rbp-8], rax` 表示将 `rax` 寄存器的值存储到内存中
+ 内存从 `rbp-8` 到 `rbp` 的区域内存储的是变量 `a` 的地址, 也就是 `rbp-12`

如果看懂了, 后两句应该也不难理解, 甚至感觉可以优化上述代码了 (第三句代码不需要)。

从上面可以看出, 在汇编语言中, 对于 内存 的操作都是通过 地址 完成的, 地址 存储在 寄存器 中, 思路非常清晰。最后附上在线 C 语言转汇编的网站: [Compiler Explorer](https://godbolt.org/)。

在 C 语言中, 数组 的本质就是 指针! 多维数组会转换成一维数组来处理! `&a` 表示取地址, `*a_ptr` 表示取值! 至此, 你应该对 C 语言的指针有明确的认识了。

在 C 语言中, 我们为什么要指定 指针 的类型呢? 答案是用于 **加减运算**。比方说, 如果 `a_ptr` 是一个 `int` 类型的指针, 那么 `a_ptr + 1` 实际上对应 内存地址 加 4; 如果 `a_ptr` 是一个 `double` 类型的指针, 那么 `a_ptr + 1` 实际上对应 内存地址 加 8; 如果 `a_ptr` 是 `void` 类型的指针, 那么 `a_ptr + 1` 实际上对应 内存地址 加 1。

额外提醒一下, 如果你想申明两个 `int` 类型的指针, 正确写法是 `int *a, *b;`。如果你写成 `int * a, b;`, 那么变量 `a` 是指针, 变量 `b` 是 `int` 类型的变量。在申明指针变量时, `*` 号是跟着变量名走的, 这一点巨坑!

至此, 指针的概念就说明清楚了。至于多级指针, 也就是 指针 的 指针, 和上面一样, 又是一个 套娃 的概念。需要注意的是, 二维数组 不是 二级指针, 还是 一级指针, 解析时会将其作为 一维数组 处理。多维数组 又是一个 套娃 的概念, 真的心累!

## C 语言吐槽

网上看到很多吐槽 Python 缩进等等的内容, 这里, 我来好好地吐槽一下 C 语言。

1. 没有 namespace, include 一个库后, 库中所有的 函数, 变量 都是全局的, 不熟悉 C 语言库的估计都能被坑死。解决方案: 直接使用 C++ 封装的库, 比方说 `cstdio`, 而不要用 `stdio.h`, 这样就可以写 `std::printf` 这样的语句了, 但还是很坑, 代码的可读性低。
2. 调用函数时, 不能加上函数参数的名称。所谓的 "代码可读性", 很多时候是靠变量名称实现的。调用函数时, 没有参数名称, 鬼知道什么是什么啊。目前, 只知道 Python 在这方面做的比较好, 其它的不知道。(即使是编译, 感觉实现起来也没有那么困难吧)。
3. 标准库函数名起的 "乱七八糟", 即不符合 驼峰命名法, 也不符合 下划线命名法, 全 TMD 的都是简写, 我服。`malloc` 变成 `memory_allocate` 或者 `memoryAllocate` 多好啊。还有, 读取文件时, `fgets`, `gets`, `fgetc`, `getc`, `getchar` 这几个函数各不相同, 你是认真的吗!
4. 语法 乱七八糟 的, 主要体现在:
   1. 不需要 "指针" 的概念, 万物皆是 "指针" 即可, 反正不是 "指针" 的你最后也要解析成 "指针", 何必呢
   2. 申明变量的方式真的不能接受, 数据就应该按照 `int[5] a;` 的方式来申明, 为什么要写成 `int a[5];` 啊, 这不是纯坑人吗! 还有 "指针", 在 `int * a, b;` 中, `b` 居然是 `int` 类型的, 星号是跟着变量走的, 我服。在类型强转时, 星号又变成跟着类型走的了, 无语。这里, Java 就做了很大的改进: `int[] a = new int[5];` 非常易懂 (但不简洁)。
   3. 同一个符号表示过多的含义, 比方说, 星号 可以表示 乘法, 指针, 取值 三个含义, 你是深怕考研老师不知道怎么出题是吗。还有, 圆括号 可以表示 函数调用, for/while/if 等语句的条件, 类型强转 等等, 考研老师太爱你们了。顺便吐槽一下 C++, 创建对象的方式居然是: `objectType objectName(param1, param2);`。你看看除了你们, 谁家创建对象时参数是跟着 "objectName" 的, 当然应该跟着 "objectType" 啊, 你这样还有啥可读性。然后匿名对象时又变成跟着 "objectName" 走, 活生生多出个 "匿名对象" 的概念。

## 并行 与 并发

在高性能计算中, 有两种策略: [并发计算](https://en.wikipedia.org/wiki/Concurrent_computing) (concurrent computing) 和 [并行计算](https://en.wikipedia.org/wiki/Parallel_computing) (parallel computing)。

所谓 **并发**, 指的是同一个时间点有多个任务需要处理, 但是只有一个工人, 应该按照什么样的方式在最短的时间内完成这些任务。一般情况下, 我们会用一个 数组/序列/列表 来记录 多个任务。这是一个 **调度系统** 的问题, 常用的方式有: 先来先处理 (queue), 后来先处理 (stack), 执行时间短的先处理 等等。在 node.js 或者 python 中, 我们会接触到 "异步编程" 这一概念, 其就是 并发编程, 主要用于解决 IO 密集型的任务, 而 计算密集型 的任务就束手无策了。

怎么 解决 计算密集型 的问题呢? 答案是 **并行** 计算。所谓 **并行**, 也是同一个时间点有多个任务需要处理, 不过现在有多个工人, 而非单个工人, 怎么在最短的时间内完成这些任务。这好像也是一个 **调度系统** 的问题, 但是 **并行** 更加强调 工人的数量。如果 工人的数量 越多, 那么我们说机器的 并行能力 越高。

## 未提及引用

+ [一文读懂堆与栈的区别](https://dablelv.blog.csdn.net/article/details/80849966)
+ [短篇知识点（一）：节点与结点的区别](https://blog.csdn.net/weixin_42269028/article/details/125970498)