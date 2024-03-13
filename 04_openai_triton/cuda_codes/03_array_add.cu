
#include <stdio.h>
#include <stdarg.h> // https://www.runoob.com/cprogramming/c-standard-library-stdarg-h.html 

// region: 简单的日志输出函数
const char *RESET_COLOR = "\033[0m";
const char *RED_COLOR = "\033[0m\033[1;31m";
const char *CYAN_COLOR = "\033[0m\033[1;36m";

void logInfo(const char *format, ...) {
    printf("%sINFO: %s", CYAN_COLOR, RESET_COLOR);

    va_list args;
    va_start(args, format); // 初始化 args
    vprintf(format, args);
    va_end(args);  // 释放 args

    printf("\n");
}

void logError(const char *format, ...) {
    printf("%sError: %s", RED_COLOR, RESET_COLOR);

    va_list args;
    va_start(args, format); // 初始化 args
    vprintf(format, args);
    va_end(args);  // 释放 args

    printf("\n");
}
// endregion 


// __global__ void add_kernel(float array_A[], float array_B[], float array_C[], const int arraySize) {
__global__ void add_kernel(float *array_A, float *array_B, float *array_C, const int arraySize) {
    const int thread_index = threadIdx.x;
    const int block_index = blockIdx.x;
    const int index = thread_index + block_index * blockDim.x;

    if (index < arraySize) {
        array_C[index] = array_A[index] + array_B[index];
        // 下面两种写法也是可以的:
        // *(array_C + index) = *(array_A + index) + *(array_B + index);
        // index[array_C] = index[array_A] + index[array_B];
    }
}


int main(void) {

    // # part1: 检测计算机中 GPU 的数量 (重点: cudaGetDeviceCount)
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess || deviceCount == 0) {
        logError("No CUDA computable GPU found!");
        exit(-1);
    }
    logInfo("The number of GPUs is %d.", deviceCount);

    // # part2: 设置执行的 GPU 设备 (重点: cudaSetDevice)
    for(int i = 0; i < deviceCount; i++) {
        error = cudaSetDevice(i);

        if (error == cudaSuccess) {
            logInfo("set No. %d GPU for computing.", i);
            break;
        }

        if (i == deviceCount - 1) {
            logError("No GPU can be used for computing!");
            exit(-1);
        }
    }

    // 内存管理: cudaMalloc (malloc); cudaMemcpy (memcpy); cudaMemset (memset); cudaFree (free)

    const int arraySize = 512;
    size_t arrayMemorySize = arraySize * sizeof(float);

    // # part3: 分配主机内存, 并初始化 (重点: malloc, memset, free)
    float *arrayHostPtr_A, *arrayHostPtr_B, *arrayHostPtr_C = NULL;
    // 初始化一块 arrayMemorySize 大小的内存, 返回 void *, 强转成 float *, 赋值给 arrayHostPtr_A 
    arrayHostPtr_A = (float *) malloc(arrayMemorySize);
    arrayHostPtr_B = (float *) malloc(arrayMemorySize);
    arrayHostPtr_C = (float *) malloc(arrayMemorySize);

    // 如果 arrayHostPtr_A, arrayHostPtr_B 或者 arrayHostPtr_C 中有空值, 表示初始化失败, 进行异常处理
    if (arrayHostPtr_A == NULL || arrayHostPtr_B == NULL || arrayHostPtr_C == NULL) {
        logError("Failed to allocate host memory!");
        free(arrayHostPtr_A); free(arrayHostPtr_B); free(arrayHostPtr_C);
        exit(-1);
    }

    // 将三个数组初始化为 0.0f
    memset(arrayHostPtr_A, 0.0f, arrayMemorySize);
    memset(arrayHostPtr_B, 0.0f, arrayMemorySize);
    memset(arrayHostPtr_C, 0.0f, arrayMemorySize);

    // # part4: 分配设备内存, 并初始化 (重点: cudaMalloc, cudaMemset, cudaFree)
    float *arrayDevicePtr_A, *arrayDevicePtr_B, *arrayDevicePtr_C = NULL;
    // 在 GPU 中初始化一块大小是 arrayMemorySize 的内存, 将其 "地址的地址" 赋值给 &arrayDevicePtr_A, 返回任务是否成功
    // 注意 malloc 和 cudaMalloc 之间的区别
    if (cudaMalloc((void **) &arrayDevicePtr_A, arrayMemorySize) != cudaSuccess || arrayDevicePtr_A == NULL) {
        logError("Failed to allocate device memory!");
        free(arrayHostPtr_A); free(arrayHostPtr_B); free(arrayHostPtr_C);
        exit(-1);
    }
    if (cudaMalloc((void **) &arrayDevicePtr_B, arrayMemorySize) != cudaSuccess || arrayDevicePtr_B == NULL) {
        logError("Failed to allocate device memory!");
        free(arrayHostPtr_A); free(arrayHostPtr_B); free(arrayHostPtr_C);
        cudaFree(arrayDevicePtr_A);
        exit(-1);
    }
    if (cudaMalloc((void **) &arrayDevicePtr_C, arrayMemorySize) != cudaSuccess || arrayDevicePtr_C == NULL) {
        logError("Failed to allocate device memory!");
        free(arrayHostPtr_A); free(arrayHostPtr_B); free(arrayHostPtr_C);
        cudaFree(arrayDevicePtr_A); cudaFree(arrayDevicePtr_B);
        exit(-1);
    }

    cudaMemset(arrayDevicePtr_A, 0.0f, arrayMemorySize); // memset 和 cudaMemset 之间没有区别
    cudaMemset(arrayDevicePtr_B, 0.0f, arrayMemorySize);
    cudaMemset(arrayDevicePtr_C, 0.0f, arrayMemorySize);

    // # part5: 随机初始化主机上的数组
    // srand: 设置随机种子
    srand(666);
    for (int i = 0; i < arraySize; i++) {
        // rand() 函数生成 0 到 RAND_MAX 之间的一个随机整数, 和 0xFF (255) 之间进行 "按位与" 运算, 保证整数小于 255
        // 然后强转成 float 类型, 再除以 10, 最后的值一定是 0 到 25.5 之间的浮点数!
        arrayHostPtr_A[i] = (float)(rand() & 0xFF) / 10.0f;
        arrayHostPtr_B[i] = (float)(rand() & 0xFF) / 10.0f;
    }

    // part6: 将 host 上的数据拷贝到 GPU 中 (重点: cudaMemcpy)
    cudaMemcpy(arrayDevicePtr_A, arrayHostPtr_A, arrayMemorySize, cudaMemcpyHostToDevice);
    cudaMemcpy(arrayDevicePtr_B, arrayHostPtr_B, arrayMemorySize, cudaMemcpyHostToDevice);

    // part7: 在 GPU 中进行计算
    const int threadPerBlock = 32;
    dim3 block_size(threadPerBlock);
    dim3 grid_size(arraySize / threadPerBlock);
    add_kernel<<<grid_size, block_size>>>(arrayDevicePtr_A, arrayDevicePtr_B, arrayDevicePtr_C, arraySize);
    cudaDeviceSynchronize();

    // part8: 将 GPU 中的数据拷贝到 host 上 (重点: cudaMemcpy)
    cudaMemcpy(arrayHostPtr_C, arrayDevicePtr_C, arrayMemorySize, cudaMemcpyDeviceToHost);

    // part9: 输出结果
    const int rowNum = 8;
    for (int i = 0; i < arraySize; i++) {
        printf("%05.2f + %05.2f = %05.2f\t", arrayHostPtr_A[i], arrayHostPtr_B[i], arrayHostPtr_C[i]);
        if (i % rowNum == rowNum - 1) {
            printf("\n");
        }
    }

    // part10: 释放内存
    free(arrayHostPtr_A); free(arrayHostPtr_B); free(arrayHostPtr_C);
    cudaFree(arrayDevicePtr_A); cudaFree(arrayDevicePtr_B); cudaFree(arrayDevicePtr_C);

    return 0;
}

