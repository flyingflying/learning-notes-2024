
/* 矩阵乘法 */

#include <stdio.h>

cudaError_t errorCheck(cudaError_t error_code, const char* file_name, int line_number) {
    // C 语言独特的 debug 方式
    if (error_code != cudaSuccess) {
        printf("%sCUDA error%s occured in file %s with # %d line.\n", "\033[0m\033[1;31m", "\033[0m", file_name, line_number);
        printf("\tError Code: %d, Error Name: %s\n", error_code, cudaGetErrorName(error_code));
        printf("\tError Description: %s\n", cudaGetErrorString(error_code));
        exit(-1);
    }
    return error_code;
}

#define ErrorCheck(ret_value) errorCheck(ret_value, __FILE__, __LINE__)  // 使用 宏 (macro) 减少代码量

#define BLOCK_SIZE 16  // block 的大小决定了 共享内存 的大小, 而 共享内存 的大小必须在编码阶段就确定, 因此必须事先定义好

typedef struct {
    // 本程序使用 row major 矩阵, 需要注意的是, cuda 二维线程模型是 column major 的
    unsigned int num_rows;
    unsigned int num_cols;
    unsigned int stride;  // 矩阵的步长, 就是 行向量 元素个数! 可以用 num_cols 替代, 但是 submatrix 时会有问题!
    float *array_ptr;
} Matrix;

__host__ __device__ size_t matrix_mem_size(Matrix mat) {
    // 计算矩阵 mat 占用的空间
    return mat.num_cols * mat.num_rows * sizeof(float);
}

__host__ __device__ int ceil_div(unsigned int num1, unsigned int num2) {
    // int 除法默认是 floor 除法, 这里实现的快速 ceil 除法
    // reference: https://stackoverflow.com/questions/2745074/fast-ceiling-of-an-integer-division-in-c-c 
    if (num2 == 0) {
        printf("division by zero.");
    }
    return (num1 + num2 - 1) / num2;
}

__host__ __device__ float get_matrix_value(Matrix mat, int r_index, int c_index) {
    // 获取矩阵中的元素值, 注意 面向过程 的写法, 以及 二维数组 和 一维数组 之间的转换
    if (r_index < mat.num_rows && c_index < mat.num_cols){
        return mat.array_ptr[r_index * mat.stride + c_index];
    }
    return 0.0f;
}

__host__ __device__ void set_matrix_value(Matrix mat, int r_index, int c_index, float value) {
    // 设置矩阵中的元素值, 注意 面向过程 的写法, 以及 二维数组 和 一维数组 之间的转换
    if (r_index < mat.num_rows && c_index < mat.num_cols){
        mat.array_ptr[r_index * mat.stride + c_index] = value;
    }
}

Matrix get_random_matrix(unsigned int num_rows, unsigned int num_cols) {
    // 随机初始化一个 [num_rows, num_cols] 大小的矩阵
    Matrix mat;
    mat.num_rows = num_rows;
    mat.num_cols = mat.stride = num_cols;
    mat.array_ptr = (float *) malloc(num_rows * num_cols * sizeof(float));

    for (int r_index = 0; r_index < num_rows; r_index++) {
        for (int c_index = 0; c_index < num_cols; c_index++) {
            // rand() 函数返回 [​0​, RAND_MAX] 之间的伪随机整数 (int),
            // 和 0xFF (255) 之间进行 "按位与" 运算, 保证随机数的范围在 [0, 255] 之间
            // 最后强转成 float 类型, 再除以 255, 保证得到的随机数一定是 [0.0, 1.0] 之间
            // reference: https://en.cppreference.com/w/c/numeric/random/rand 
            float value = (float)(rand() & 0xFF) / 255.0f;
            set_matrix_value(mat, r_index, c_index, value);
        }
    }

    return mat;
}

void print_matrix(FILE *out_stream, Matrix mat) {
    // 输出 矩阵, 和 PyTorch 中的格式兼容
    fprintf(out_stream, "tensor([\n");

    for (int r_index = 0; r_index < mat.num_rows; r_index++) {
        fprintf(out_stream, "\t[");

        for (int c_index = 0; c_index < mat.num_cols; c_index++) {
            float value = get_matrix_value(mat, r_index, c_index);
            fprintf(out_stream, "%0.10f, ", value);
        }

        fprintf(out_stream, "],\n");
    }

    fprintf(out_stream, "])\n");
}

void print_matrix(Matrix mat) { print_matrix(stdout, mat); }

void gen_check_pycode(FILE *out_stream, Matrix mat1, Matrix mat2, Matrix result_mat) {

    fprintf(out_stream, "\nmat1 = ");
    print_matrix(out_stream, mat1);
    fprintf(out_stream, "\nmat2 = ");
    print_matrix(out_stream, mat2);
    fprintf(out_stream, "\nresult_mat = ");
    print_matrix(out_stream, result_mat);
    fprintf(out_stream, "\ngold_mat = mat1 @ mat2\n");
    fprintf(out_stream, "\nprint((result_mat - gold_mat).abs().max())\n");

}

__global__ void matmul_kernel(Matrix left_mat, Matrix right_mat, Matrix result_mat) {
    // 线程索引是 column major 的
    int r_index = blockIdx.y * blockDim.y + threadIdx.y;
    int c_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (r_index < result_mat.num_rows && c_index < result_mat.num_cols) {
        float dot_result = 0.0f;
        int vector_size = left_mat.num_cols;  // 也可以是 right_mat.num_rows

        for (int vec_idx = 0; vec_idx < vector_size; vec_idx++) {
            dot_result += get_matrix_value(left_mat, r_index, vec_idx) * get_matrix_value(right_mat, vec_idx, c_index);
        }

        set_matrix_value(result_mat, r_index, c_index, dot_result);

    }
}

__global__ void matmul_kernel_v2(Matrix left_mat, Matrix right_mat, Matrix result_mat) {

    unsigned int block_row_index = blockIdx.y;
    unsigned int block_col_index = blockIdx.x;

    unsigned int vec_size = left_mat.num_cols;  // 也可以是 right_mat.num_rows
    unsigned int r_index = threadIdx.y;
    unsigned int c_index = threadIdx.x;

    float dot_result = 0.0f;

    left_mat.array_ptr += block_row_index * left_mat.num_cols * BLOCK_SIZE;
    right_mat.array_ptr += block_col_index * BLOCK_SIZE;

    for (int sub_idx = 0; sub_idx < ceil_div(vec_size, BLOCK_SIZE); sub_idx++) {

        __shared__ float left_submat[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float right_submat[BLOCK_SIZE][BLOCK_SIZE];

        left_submat[r_index][c_index] = get_matrix_value(left_mat, r_index, c_index + sub_idx * BLOCK_SIZE);
        right_submat[r_index][c_index] = get_matrix_value(right_mat, r_index + sub_idx * BLOCK_SIZE, c_index);
        __syncthreads();

        for (int subvec_idx = 0; subvec_idx < BLOCK_SIZE; subvec_idx++) {
            dot_result += left_submat[r_index][subvec_idx] * right_submat[subvec_idx][c_index];
        }
        __syncthreads();

    }
    set_matrix_value(result_mat, r_index + block_row_index * BLOCK_SIZE, c_index + block_col_index * BLOCK_SIZE, dot_result);
}

Matrix matmul(Matrix mat1, Matrix mat2, int version) {
    if (mat1.num_cols != mat2.num_rows) {
        printf("mat1 and mat2 shapes cannot be multiplied");
        exit(-1);
    }

    // step1: 初始化 output 结构体
    Matrix output;
    output.num_rows = mat1.num_rows;
    output.num_cols = output.stride = mat2.num_cols;
    cudaMalloc(&output.array_ptr, matrix_mem_size(output));

    // step2: 将 mat1 移动到 GPU 上
    float *mat1_cpu_ptr = mat1.array_ptr;
    cudaMalloc(&mat1.array_ptr, matrix_mem_size(mat1));
    cudaMemcpy(mat1.array_ptr, mat1_cpu_ptr, matrix_mem_size(mat1), cudaMemcpyHostToDevice);  // dest, source

    // step3: 将 mat2 移动到 GPU 上
    float *mat2_cpu_ptr = mat2.array_ptr;
    cudaMalloc(&mat2.array_ptr, matrix_mem_size(mat2));
    cudaMemcpy(mat2.array_ptr, mat2_cpu_ptr, matrix_mem_size(mat2), cudaMemcpyHostToDevice);

    // step4: 启动多线程
    // 使用二维线程块, 每一个线程块有 16 * 16 = 256 个线程, 一个线程块有 256 / 32 = 8 个线程束
    // 在这里, 每一个线程负责一个向量点乘的运算
    dim3 block_size(16, 16);
    // 使用二维 grid, 需要注意的是, 线程二维模型是 column major 的, 不是 row major 的
    dim3 grid_size(ceil_div(output.num_cols, block_size.x), ceil_div(output.num_rows, block_size.y));

    if (version == 0) {
        matmul_kernel<<<grid_size, block_size>>>(mat1, mat2, output);
    } else {
        matmul_kernel_v2<<<grid_size, block_size>>>(mat1, mat2, output);
    }

    ErrorCheck(cudaDeviceSynchronize());
    ErrorCheck(cudaGetLastError());

    // step5: 移动 output 到 CPU 上
    float *output_gpu_ptr = output.array_ptr;
    output.array_ptr = (float *) malloc(matrix_mem_size(output));
    cudaMemcpy(output.array_ptr, output_gpu_ptr, matrix_mem_size(output), cudaMemcpyDeviceToHost);

    // step6: 释放 GPU 内存
    cudaFree(mat1.array_ptr);
    cudaFree(mat2.array_ptr);
    cudaFree(output_gpu_ptr);

    return output;

}

int main(void) {

    clock_t start, end;
    Matrix mat_a, mat_b, mat_c;

    FILE *writer = fopen("check_matmul.py", "w");
    fprintf(writer, "\nfrom torch import tensor\n");

    // check case1
    mat_a = get_random_matrix(16, 16);
    mat_b = get_random_matrix(16, 16);

    mat_c = matmul(mat_a, mat_b, 0);
    gen_check_pycode(writer, mat_a, mat_b, mat_c);

    mat_c = matmul(mat_a, mat_b, 1);
    gen_check_pycode(writer, mat_a, mat_b, mat_c);

    // check case2
    mat_a = get_random_matrix(14, 8);
    mat_b = get_random_matrix(8, 12);

    mat_c = matmul(mat_a, mat_b, 0);
    gen_check_pycode(writer, mat_a, mat_b, mat_c);

    mat_c = matmul(mat_a, mat_b, 1);
    gen_check_pycode(writer, mat_a, mat_b, mat_c);

    // check case3
    mat_a = get_random_matrix(20, 20);
    mat_b = get_random_matrix(20, 20);

    mat_c = matmul(mat_a, mat_b, 0);
    gen_check_pycode(writer, mat_a, mat_b, mat_c);

    mat_c = matmul(mat_a, mat_b, 1);
    gen_check_pycode(writer, mat_a, mat_b, mat_c);

    // check case4
    mat_a = get_random_matrix(99, 100);
    mat_b = get_random_matrix(100, 101);

    mat_c = matmul(mat_a, mat_b, 0);
    gen_check_pycode(writer, mat_a, mat_b, mat_c);

    mat_c = matmul(mat_a, mat_b, 1);
    gen_check_pycode(writer, mat_a, mat_b, mat_c);

    fclose(writer);
    system("python3 check_matmul.py");

    // check time
    mat_a = get_random_matrix(8192, 8192);
    mat_b = get_random_matrix(8192, 8192);

    start = clock();
    mat_c = matmul(mat_a, mat_b, 0);
    end = clock();
    printf("Basic version time consuming: %0.4f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    // 4090 测试, 最快 0.38 秒

    start = clock();
    mat_c = matmul(mat_a, mat_b, 1);
    end = clock();
    printf("Shared memory version time consuming: %0.4f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);
    // 4090 测试, 最快 0.34 秒

    return 0;
}
