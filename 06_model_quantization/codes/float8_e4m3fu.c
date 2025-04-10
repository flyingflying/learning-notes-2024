
#include <stdint.h>
#include <limits.h>
#include <stdio.h>

// copied from c10/util/floating_point_utils.h 
float fp32_from_bits(uint32_t w) {

    // union: 同一份数据, 不同类型
    // 注意: 这里 fp32 变量的类型是 union {uint32_t as_bits; float as_value;}
    union {
        uint32_t as_bits;
        float as_value;
    } fp32;

    fp32.as_bits = w;
    return fp32.as_value;
}

// copied from c10/util/Float8_e4m3fn.h
float fp8e4m3fn_to_fp32_value(uint8_t input) {
    // 将 fp8e4m3fn 类型转换成 fp32 类型。注意: fp8 类型是以 uint8 形式存在的。
    // fp8 e4m3 fn: 4 位阶码 3 位尾码, 无 inf 编码, 阶码取值范围 [-7, 8]
    // 非规约数、零值 和 规约数 的编码方式跟 IEEE 754 是一致的
    // 特殊值的编码方式不同, 当阶码等于 8 时, 尾数值全是 1 对应 nan, 否则按照 规约数 的方式进行编码
    // reference: https://onnx.ai/onnx/technical/float8.html 

    // w: S EEEE MMM 0000 0000 0000 0000 0000 0000
    const uint32_t w = (uint32_t)input << 24;

    // nonsign: 0 EEEE MMM 0000 0000 0000 0000 0000 0000
    // UINT32_C 是 stdint.h 中定义的宏, 表示 unsigned 常量
    const uint32_t nonsign = w & UINT32_C(0x7FFFFFFF);

    // __builtin_clz 是 gcc 编译器的 builtin 函数, 其含义是: counts the leading number of zeros of the integer
    // 如果 input 是 规约数 (normalized), 第 30 - 27 位中一定有一位是 1, 此时 renorm_shift 的值一定在 1 - 4 之间。
    // 如果 input 是 非规约数 (denormalized), renorm_shift 的值是 5, 6 或者 7 (不考虑 input 是零值的情况)
    uint32_t renorm_shift = __builtin_clz(nonsign);

    // 如果 input 是 规约数, renorm_shift 的值是 0。
    // 如果 input 是 非规约数, renorm_shift 的值是 1, 2 或者 3
    renorm_shift = renorm_shift > 4 ? renorm_shift - 4 : 0;

    // (1) 如果 input 是 规约数, 那么下面的过程可以简化为 (nonsign) >> 4 + (0x78 << 23)
    // 即先 右移 4 位, 让 阶码 从四位变成八位, 然后给阶码值加上 0x78 (120 = 127 - 7) 即可
    // (2) 如果 input 是 非规约数, nonsign << renorm_shift 后阶码的编码值都是 0001
    // renorm_shift 值 1, 2, 3 分别对应的阶码值是 -7, -8 和 -9, 指数偏移 分别是 8, 9 和 10
    // 那么在转换成 float32 时, 需要给阶码值分别加上 119, 118 和 117。整个过程非常巧妙。
    uint32_t result = (nonsign << renorm_shift) >> 4;
    result += (0x78 - renorm_shift) << 23;

    // (1) 当 input 是 nan 时, 指数位 和 尾数位 都是 1, nonsign = 0x7F000000, nonsign + 0x01000000 = 0x80000000
    // 此时 (int32_t)(nonsign + 0x01000000) >> 8 = 0xff800000, 最终 inf_nan_mask 恰好是 0x7F800000
    // (2) 当 input 不是 nan 时, nonsign 取值范围是 [0x00000000, 0x7E000000], nonsign + 0x01000000 取值范围 [0x01000000, 0x7F000000]
    // 此时 (int32_t)(nonsign + 0x01000000) >> 8 取值范围是 [0x00010000, 0x007F0000], 最终 inf_nan_mask 恰好是 0x00000000
    const int32_t inf_nan_mask = ((int32_t)(nonsign + 0x01000000) >> 8) & INT32_C(0x7F800000);
    result |= inf_nan_mask;  // 当 input 是 nan 时, float32 的 8 个指数位强制赋值为 1

    // 如果 nonsign 值为 0, 那么 result 的值也应该为 0
    // 当 nonsign = 0 时, zero_mask = 0xffffffff, ~zero_mask = 0x00000000
    // 当 nonsign != 0 时, zero_mask = 0x00000000, ~zero_mask = 0xffffffff
    // 这里利用了 带符号右移 的特性: 当首位为 0 时, 右移高位补零; 当首位为 1 时, 右移高位补一
    const int32_t zero_mask = (int32_t)(nonsign - 1) >> 31;
    result &= ~zero_mask;  // 当 input 是 0 时, float32 的所有位都强制赋值为 0

    // 给 result 添加 sign 符号
    // sign: S 0000 000 0000 0000 0000 0000 0000 0000
    const uint32_t sign = w & UINT32_C(0x80000000);
    result |= sign;

    return fp32_from_bits(result);
}


void main() {
    for (int i = 0; i <= 255; i++) {
        printf("%03d %15.10f\n", i, fp8e4m3fn_to_fp32_value((uint8_t)i));
    }
}
