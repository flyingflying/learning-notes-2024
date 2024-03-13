
// 准备: source cuda_bash.sh
// 编译: nvcc 02_hello_thread.cu -o bin/hello_thread.exe
// 运行: ./bin/hello_thread.exe

#include <stdio.h>
#include <limits.h>

// C 语言中输出变量类型是一件很麻烦的事情, 目前找到最好的方式是用 "重载" 间接实现
// reference: https://stackoverflow.com/a/28253910
__host__ __device__ const char * type_name(int) {return "int";}
__host__ __device__ const char * type_name(unsigned int) {return "unsigned int";}
__host__ __device__ const char * type_name(long) {return "long";}
__host__ __device__ const char * type_name(unsigned long) {return "unsigned long";}

__global__ void hello_thread() {

    // 核函数中有 内建变量 (build-in variable) blockIdx, threadIdx, gridDim 和 blockDim
    // 注意, 核函数之外是没有这些变量的!
    const unsigned int block_index = blockIdx.x;
    const unsigned int thread_index = threadIdx.x;

    const unsigned int global_index = thread_index + block_index * blockDim.x;

    printf(
        "hello, cuda world from No. %d block with No. %d thread, and the global index is %d!\n",
        block_index, thread_index, global_index
    );

    printf("%s\n", type_name(blockIdx.x)); // unsigned int !!!

}


__global__ void hello_multidim_thread() {

    /* **************************************************************************************************************
        1. 对于一个二维数组来说, 我们用 [y, x] 表示某一个元素的坐标值, 用 y_size 和 x_size 分别表示 y 维和 x 维的元素个数, 
        如果将这个二维数组 "平铺" 成一维数组, 那么坐标值是 x + y * x_size
        2. 对于一个三维数组来说, 我们用 [z, y, x] 表示某一个元素的坐标, 用 z_size, y_size 和 x_size 分别表示 z 维, y 维
        和 x 维的元素个数, 如果将这个三维数组 "平铺" 成一维数组, 那么坐标值是 x + y * x_size + z * y_size * x_size
    ************************************************************************************************************** */
    const unsigned int block_index = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.y * gridDim.x;
    const unsigned int thread_index = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;

    // block 和 grid 之间是 "嵌套关系", 我们可以认为是 二维数组
    const unsigned int global_index = thread_index + block_index * (blockDim.x * blockDim.y * blockDim.z);

    printf(
        "hello, cuda word from (%d, %d, %d) block with (%d, %d, %d) thread, and the global index is %d!\n", 
        blockIdx.z, blockIdx.y, blockIdx.x, threadIdx.z, threadIdx.y, threadIdx.x, global_index
    );

}


int main(void) {
    hello_thread<<<2, 4>>>();  // 申明 2 个 block, 每一个 block 有 4 个 thread

    // # test case1: 每一个 block 中 thread 数最多为 1024
    // hello_thread<<<2, 1024>>>();
    // # test case2: 如果超过 1024, 编译可以通过, 执行时直接中断, 没有任何提示 (不愧是 C++)
    // hello_thread_kernel<<<2, 1025>>>();

    // # test case3: block 数最多为 INT_MAX (INT_MAX 比 21 亿略大一些)
    // hello_thread<<<INT_MAX, 1>>>();
    // # test case4: 如果超过 INT_MAX, 和上面一样, 编译可以通过, 执行时直接中断, 没有任何提示
    // hello_thread<<<(long) INT_MAX + 1, 1>>>();

    cudaDeviceSynchronize();

    hello_multidim_thread<<<dim3 (2, 2), dim3 (3, 3)>>>();  // dim3 (2, 2) 采用的是 C++ 匿名对象的形式
    // hello_multidim_thread<<<dim3 (2, 2, 2), dim3 (3, 3, 3)>>>();
    cudaDeviceSynchronize();

    printf("%s\n", type_name(INT_MAX));

    return 0;
}

// 更多的编译参数: nvcc 02_hello_thread.cu -o bin/hello_thread.exe -arch=compute_89 -code=sm_89
// GeForce RTX 4090 支持的最大 `计算能力` 是 89
// reference: https://developer.nvidia.com/cuda-gpus
// nvcc 02_hello_thread.cu -o bin/hello_thread.exe -gencode arch=compute_89,code=sm_89 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_70,code=sm_70

// 生成 PTX 文件的指令: nvcc 02_hello_thread.cu -ptx
// 生成 cubin 文件的指令: nvcc 02_hello_thread.cu -cubin
