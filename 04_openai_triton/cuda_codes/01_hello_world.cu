
// 准备: source cuda_bash.sh
// 编译: nvcc -o bin/hello_world.exe 01_hello_world.cu
// 运行: ./bin/hello_world.exe

#include <stdio.h>

__global__ void hello_world_kernel() {
    // 核函数必须用 `__global__` 限定词修饰, 返回是 void
    // 只能用 printf 输出, 不能用 C++ 中的 std::cout 输出
    printf("hello, cuda world!\n");
}


int main(void) {
    // 调用 hello_world_kernel 函数
    hello_world_kernel<<<1, 1>>>();
    // 同步 (cuda 编程是 主机 和 设备 之间的异步编程)
    cudaDeviceSynchronize();

    return 0;
}
