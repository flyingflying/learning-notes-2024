#include <iostream>   // std::cout, std::endl
#include <string>     // std::string
#include <vector>     // std::vector
#include <utility>    // std::move
#include <algorithm>  // std::copy
#include <stdexcept>
#include <typeinfo>
#include <functional>

struct Buffer {
    std::string name = "unnamed";
    int   size = 0;
    char *data = nullptr;

    // 默认构造函数
    Buffer() = default;

    // 普通构造函数
    Buffer(int size_, std::string name_) {
        if (size_ <  0) {throw std::invalid_argument("Buffer size cannot be negative");}  // 异常处理

        name = name_;  // RAII 型对象 (拷贝赋值)
        size = size_;  // 普通成员变量
        if (size_ > 0) {data = new char[size];}  // 堆区域数据
    }

    // 析构函数
    ~Buffer() {delete[] this->data;}  // delete nullptr 是允许的

    // 拷贝构造
    Buffer(const Buffer& other) {
        std::cout << "调用 '拷贝构造' 函数" << std::endl;
        name = other.name;  // RAII 型对象: 使用其自身的 "拷贝赋值" 函数
        size = other.size;  // 普通成员变量: 直接赋值
        if (size > 0) {     // 堆区域数据: std::copy 拷贝数据
            data = new char[other.size];
            std::copy(other.data, other.data + size, data);
        }
    }

    // 移动构造
    // 建议标记 noexcept，否则 std::vector 在重新分配时可能改用拷贝构造以保证强异常安全
    Buffer(Buffer&& other) noexcept {
        std::cout << "调用 '移动构造' 函数" << std::endl;
        name = std::move(other.name);             // RAII 型对象: 使用其自身的 "移动赋值" 函数
        size = other.size; other.size = 0;        // 普通成员变量: 直接赋值即可
        data = other.data; other.data = nullptr;  // 堆区域数据: 转交数据, 源指针置空
    }

    // 拷贝赋值
    Buffer& operator=(const Buffer& other) {
        std::cout << "调用 '拷贝赋值' 函数" << std::endl;
        if (this == &other) {return *this;}

        // 构造临时资源: 防止在 data 拷贝的过程中触发异常, 这样异常抛出后 name 值修改了 data 值没修改
        std::string tmp_name = other.name;       // RAII 型对象: 使用其自身的 "拷贝构造" 函数
        char* tmp_data = nullptr;                // 堆区域数据: std::copy 拷贝数据
        if (other.size > 0) {
            tmp_data = new char[other.size];
            std::copy(other.data, other.data + other.size, tmp_data);
        }

        delete[] data; data = tmp_data;
        size = other.size;                       // 普通成员变量: 直接赋值
        name = std::move(tmp_name);
        return *this;
    }

    // 移动赋值
    Buffer& operator=(Buffer&& other) noexcept {
        std::cout << "调用 '移动赋值' 函数" << std::endl;
        if (this == &other) {return *this;}

        name = std::move(other.name);       // RAII 型对象: 使用其自身的 "移动赋值" 函数
        size = other.size; other.size = 0;  // 普通成员变量: 直接赋值
        delete[] data; data = other.data; other.data = nullptr;  // 堆区域数据: 释放自身旧资源，窃取 other 资源，源对象置空
        return *this;
    }
};

void func(Buffer&& b) {
    std::cout << b.name << std::endl;
}

int main() {

    Buffer buffer1(10, "node1");
    Buffer buffer2(20, "node2");
    Buffer& r1 = buffer1;  // 左值引用
    r1 = buffer2;  // 修改值, 不是重新绑定, 触发拷贝构造函数

    Buffer buffer3(10, "node1");
    Buffer buffer4(20, "node2");
    std::reference_wrapper<Buffer> r2 = std::ref(buffer3);  // 引用封装
    r2 = buffer4;  // 重新绑定, 不是修改值
    r2.get() = buffer3;  // 修改值, 不是重新绑定 [指针方式: *p = buffer3]
    std::cout << r2.get().name << std::endl;  // [指针方式: p->name]

    Buffer buffer5(10, "node1");
    Buffer buffer6(20, "node2");
    Buffer *p = &buffer5;
    *p = buffer6;
}
