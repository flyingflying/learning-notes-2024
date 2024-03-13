
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void outputCharArray(char *array, int arraySize, int rowNum) {
    printf("\n");
    for (int i = 0; i < arraySize; i++) {
        printf("%04d ", array[i]);
        if (i % rowNum == rowNum - 1) {
            printf("\n");
        }
    }
    printf("\n");
}


int readFileByFgets(const char* fileName) {
    // 申明一个 buffer 用于存储数据
    int bufferSize = 100;
    char *buffer = (char *) malloc(bufferSize * sizeof(char));
    // 将 buffer 数组全部初始化为 `空字符`
    // 在 ASCII 中, '\0' 表示 `空字符`, 表示无需输出任何东西, 在 C 语言中, 被用作 字符串 的结尾
    // reference: https://zh.wikipedia.org/wiki/ASCII
    memset(buffer, '\0', bufferSize);

    // 在 C 语言中, 只有 指针 的值可以是 NULL, 其它的都不可以, 因此 NULL 应该翻译成 `空指针`
    // 在 64 位的系统中, 指针应该是一个 `unsigned long` 类型的整数
    // `空指针` NULL 对应值 0, 一般是用不到 0 地址 (操作系统保留) 
    FILE *reader = fopen(fileName, "r");

    if (reader == NULL) {
        return -1;
    }

    while (1) {
        /* ***********************************************************
        fgets 函数的申明如下: char *fgets(char *str, int n, FILE *stream)
        其从 stream 中读取字符串, 存储到 str 指针对应的位置, 返回 str 指针或者 NULL。
        1. 如果读取到 换行符, 或者读取到文件尾, 或者读取到 n - 1 个字符, 就不会再读取字符了;
        2. 如果 stream 中没有字符可以读取, 此时不会返回 str 指针, 而是返回 NULL;
        3. 返回时, 会在字符串最后添加 空字符 "\0", 用于标识字符串的结束;
        注意:
        1. fgets 和 python 中的 readline 差不多, 在文件中没有字符可以读取时, readline 返回 空字符串
        2. fgets 和 EOF 是无关的
        reference: https://www.runoob.com/cprogramming/c-function-fgets.html 
        *********************************************************** */
        // if (fgets(buffer, bufferSize, reader) == NULL) {
        if (fgets(buffer, bufferSize, reader) != buffer) {
            break;
        }

        printf("%s", buffer); fflush(stdout);

        // puts(buffer);  // 等价于 printf("%s\n", buffer);
    }

    fclose(reader);

    return 0;

}


int readFileByFgetc(const char* fileName) {
    int byte;
    FILE *reader = fopen(fileName, "r");

    if (reader == NULL) {
        return -1;
    }

    while (1) {
        /* **********************************************************
        fgetc 的函数申明是: int fgetc(FILE *stream)
        这是一个很神奇的函数, 其从 stream 中读取一个 字节, 
        然后将这个 字节 当作 unsigned int8 类型, 再强制转换成 int 类型返回!
        对, 你没看错, 是强制转换成 int 类型! 如果 stream 中没有可读取的 字节, 则返回 -1 (EOF)。
        因此, 返回值的取值范围在 [-1, 255] 之间!
        注意: 
            1. getc 和 fgetc 差不多, 只是实现方式不同
            2. gets 和 fgets 则不同, 前者是从 stdin 中读取, 后者是从文件流中读取
            3. gets 读取没有长度限制 (大坑), 在 C11 中新增 gets_s, 添加了长度限制
            4. getchar 则是从 stdin 中读取单个字符
            5. 不愧是 C 语言, 这些函数设计的都是什么玩意啊, 变量名几乎都是简写, 还 TMD 的没有 namespace
        reference: https://www.runoob.com/cprogramming/c-function-fgetc.html 
        ********************************************************** */
        byte = fgetc(reader);

        // if (feof(reader)) {
        if (byte == EOF) {
            break;
        }

        // 测试了一下, C 语言本身是不支持解析 unicode 的, 能输出 unicode 是因为 shell 支持
        if (byte > 255) {
            return -2;
        }

        printf("%c", byte);

    }

    printf("\n");
    return 0;

}

void main(void) {
    exit(readFileByFgetc("03_array_add.cu"));
}
