
/* **********************************************************************************************
    CUDA 对应的 CPU 代码不应该是用 for 循环实现的代码, 而应该是用 多线程 实现的代码。
    C++ 在 windows 下需要调用 win32 API 实现, 在 linux 下需要使用 POSIX 标准, 使用 pthread 实现。
    在 C++ 11 标准后, 有 std::thread, 可以跨平台使用, 这里就用这种 API 实现。
    编译指令:
        gcc 03_array_add.cpp -o bin/array_add_cpu.exe -std=c++11 -lstdc++
        g++ 03_array_add.cpp -o bin/array_add_cpu.exe -std=c++11
    注意:
        gcc 一般是用于编译 C 代码, 如果编译 CPP 代码, 不会自动链接 STL 库
        g++ 则将所有的代码作为 CPP 代码编译, 并且会自动链接 STL 库
    运行:
        ./bin/array_add_cpu.exe
    References:
        1. https://immortalqx.github.io/2021/12/04/cpp-notes-3/ 
        2. https://immortalqx.github.io/2021/12/05/cpp-notes-4/ 
        3. https://www.runoob.com/w3cnote/cpp-std-thread.html 
        4. https://www.cnblogs.com/oxspirt/p/6847438.html 
********************************************************************************************** */

#include <cstdio>
#include <thread>
#include <cstdlib>
#include <cstring>


void add_func(float *arr_A, float *arr_B, float *arr_C, int index, const int arrSize) {
    if (index < arrSize) {
        arr_C[index] = arr_A[index] + arr_B[index];
    }
}


int main(void) {

    const int arraySize = 512;
    size_t arrayMemorySize = arraySize * sizeof(float);

    // 申明三个数组
    float *arrPtr_A = (float *) std::malloc(arrayMemorySize);
    float *arrPtr_B = (float *) std::malloc(arrayMemorySize);
    float *arrPtr_C = (float *) std::malloc(arrayMemorySize);

    // 判断三个数组是否申明成功
    if (arrPtr_A == NULL || arrPtr_B == NULL || arrPtr_C == NULL) {
        std::printf("Failed to allocate memory!");
        std::free(arrPtr_A); std::free(arrPtr_B); std::free(arrPtr_C);
        exit(-1);
    }

    // 初始化三个数组的值为 0.0
    std::memset(arrPtr_A, 0.0f, arrayMemorySize);
    std::memset(arrPtr_B, 0.0f, arrayMemorySize);
    std::memset(arrPtr_C, 0.0f, arrayMemorySize);

    // 随机初始化数组 A 和 B
    std::srand(666);
    for (int i = 0; i < arraySize; i++) {
        arrPtr_A[i] = (float)(std::rand() & 0xFF) / 10.0f;
        arrPtr_B[i] = (float)(std::rand() & 0xFF) / 10.0f;
    }

    // 多线程执行
    std::thread threads[arraySize];
	for (int i = 0; i < arraySize; i++) {
		threads[i] = std::thread(add_func, arrPtr_A, arrPtr_B, arrPtr_C, i, arraySize);
	}

    // 等待线程执行完毕
    for (int i = 0; i < arraySize; i++) {
        threads[i].join();
    }

    // 输出
    const int rowNum = 8;
    for (int i = 0; i < arraySize; i++) {
        std::printf("%05.2f + %05.2f = %05.2f\t", arrPtr_A[i], arrPtr_B[i], arrPtr_C[i]);
        if (i % rowNum == rowNum - 1) {
            std::printf("\n");
        }
    }

	return 0;
}
