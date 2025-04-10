
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <climits>
#include <cstring>  // std::memcpy

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

float fp32_from_bits_v2(uint32_t w) {
    // 使用 指针类型强转 的方式实现
    // C 语言有一点非常的无语, 那就是 同一个符号 表示多个意思
    // * 至少有三个含义: 乘法, 指针定义, 指针变量取值
    return *(float *) &w;
}


float fp32_from_bits_v3(uint32_t w) {
    // 第三种方式, 使用 memcpy 函数
    float f = 0.0f;
    std::memcpy(&f, &w, sizeof(w));
    return f;
}

// copied from c10/util/floating_point_utils.h
uint32_t fp32_to_bits(float f) {
    union {
        float as_value;
        uint32_t as_bits;
    } fp32;
    
    fp32.as_value = f;
    return fp32.as_bits;
}

uint32_t fp32_to_bits_v2(float f) {
    // 使用 指针类型强转 的方式实现
    return *(uint32_t *) &f;
}

uint32_t fp32_to_bits_v3(float f) {
    // 第三种方式, 使用 memcpy 函数
    uint32_t w = 0U;
    std::memcpy(&w, &f, sizeof(f));
    return w;
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

// copied from c10/util/Float8_e4m3fn.h
uint8_t fp8e4m3fn_from_fp32_value(float input) {
    /*
     * Binary representation of 480.0f, which is the first value
     * not representable in fp8e4m3fn range:
     * 0 1111 111 - fp8e4m3fn
     * 0 10000111 11100000000000000000000 - fp32
     */
    // constexpr 常量表达式, 在 编译时 就确定值了
    // fp8_max: 0 1000 0111 1110 0000 0000 0000 0000 000 = 480.f
    constexpr uint32_t fp8_max = UINT32_C(0x43f) << 20;

    // f_bits: S EEEE EEEE MMMM MMMM MMMM MMMM MMMM MMM
    uint32_t f_bits = fp32_to_bits(input);
  
    uint8_t result = 0u;

    // sign: S 0000 0000 0000 0000 0000 0000 0000 000
    const uint32_t sign = f_bits & UINT32_C(0x80000000);
    // f_bifs: 0 EEEE EEEE MMMM MMMM MMMM MMMM MMMM MMM
    f_bits ^= sign;
  
    if (f_bits >= fp8_max) {
        // 如果 f_bits 比 480.0f 大, 那么 result 就是 nan
        result = 0x7f;
    } else if (f_bits < (UINT32_C(121) << 23)) {
        // fp8e3m4fn 的最小 规约数 是 2^(-6)
        // 当 abs(input) 比 2^(-6) 小时
        constexpr uint32_t denorm_mask = UINT32_C(141) << 23;  // 2^(14)

        // 巧妙地利用 浮点数加法 完成 非规约数 的转换
        // f_bits: 0 1000 1101 0000 0000 0000 0000 0000 MMM
        // 如果 abs(input) 小于 2^(-9), 在和 2^(14) 相加后一定等于 2^(14)
        // 如果 abs(input) 取值在 2^(-7) 到 2^(-9) 之间, 和 2^(14) 相加后只有最后三个尾码有值, 其它尾码值都是 0
        f_bits = fp32_to_bits(fp32_from_bits(f_bits) + fp32_from_bits(denorm_mask));

        // 在 uint32_t 转 uint8_t 时, 只会保留最低的 8 位
        // f_bits - denorm_mask: 0 0000 0000 0000 0000 0000 0000 0000 MMM
        // result: 0 0000 MMM
        result = static_cast<uint8_t>(f_bits - denorm_mask);
    } else {
        // 当 abs(input) 值在 2^(-6) 到 480.0f 之间时, 转换后的数字是 规约数

        uint8_t mant_odd = (f_bits >> 20) & 1;  // 取第 21 位数字

        // float32 的 指数偏移 是 127, float8e4m3fn 的 指数偏移 是 7, 两者相差 120, 因此需要修正
        f_bits += (uint32_t)(7 - 127) << 23;

        // rounding (四舍五入), 不太明白其原理
        f_bits += 0x7FFFF;
        f_bits += mant_odd;

        // result: ? EEEE MMM
        result = static_cast<uint8_t>(f_bits >> 20);
    }
  
    result |= static_cast<uint8_t>(sign >> 24);
    return result;
}


int main() {
    // test fp8_e4m3_fn
    for (int i = 0; i <= 255; i++) {
        float value = fp8e4m3fn_to_fp32_value((uint8_t)i);
        printf("%03d %15.10f\n", i, value);
        if (i != fp8e4m3fn_from_fp32_value(value)) {
            return -1;
        }
    }

    return 0;
}

// copied from c10/util/Half.h
float fp16_ieee_to_fp32_value(uint16_t h) {
    // FP16 是 5 阶码 10 尾数, FP32 是 8 阶码 23 尾数
    // 在 C 语言中, 以 0x 开头的是 16 进制, 0 开头的是 8 进制数字
    // UINT32_C 数字是 UINT32 常量
    // const 常量, 在初始化完成后就不会改变了, 可以在 运行时 确定值
    // constexpr 常量表达式, 在 编译时 就确定值了

    // w: SEEE EEMM MMMM MMMM 0000 0000 0000 0000
    const uint32_t w = (uint32_t)h << 16;
    // sign: S000 0000 0000 0000 0000 0000 0000 0000
    const uint32_t sign = w & UINT32_C(0x80000000);
    // two_w: EEEE EMMM MMMM MMM0 0000 0000 0000 0000
    const uint32_t two_w = w + w;  // 等价于 w << 1
    // exp_offset: 0111 0000 0000 0000 0000 0000 0000 0000
    constexpr uint32_t exp_offset = UINT32_C(0xE0) << 23;
    // scale_bits: 0000 0111 1000 0000 0000 0000 0000 0000
    constexpr uint32_t scale_bits = (uint32_t)15 << 23;
    /* *****************************************************************************************************
    (two_w >> 4) + exp_offset: 0111 EEEE EMMM MMMM MMM0 0000 0000 0000
    我们设 w_hat = fp32_from_bits((two_w >> 4) + exp_offset), w 的阶码值是 EV

    原本 w 的 阶码编码 是 000E EEEE, 对应 EV + 15, 现在 w_hat 的阶码编码 改成 111E EEEE, 对应 EV + 15 + 224
    我们期望的 阶码编码 是 EV + 127, 所以需要将 w_hat 的 阶码编码 减小 224 + 15 - 127 = 112
    这里是通过 浮点数乘法 来实现, 即 w_hat 乘以 2^{-112} 即可
    2^{-112} 阶码 为 -112, 尾数为 0, 此时 阶码编码 是 -112 + 127 = 15, 对应 0000 1111, 也就是 scale_bits

    这样做的好处是不用自己处理 特殊值。如果 w 是特殊值 (nan / inf), 那么其阶码是 16 (nan / inf), 对应的阶码编码是 0001 1111。
    此时, w_hat 的阶码编码是 1111 1111, 属于 fp32 的特殊值, 那么和 fp32_from_bits(scale_bits) 相乘时会自动处理!
    ***************************************************************************************************** */
    const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * fp32_from_bits(scale_bits);

    // magic_mask: 0011 1111 0000 0000 0000 0000 0000 0000 
    constexpr uint32_t magic_mask = UINT32_C(126) << 23;
    // magic_bias: 0011 1111 0000 0000 0000 0000 0000 0000
    constexpr float magic_bias = 0.5f;
    /* *****************************************************************************************************
    处理阶码为 0 的情况。对于浮点数来说, 阶码为 0, 要不是 规约数, 要不是 0
    此时, two_w: 0000 0MMM MMMM MMM0 0000 0000 0000 0000, 
    (two_w >> 17) | magic_mask: 0011 1111 0000 0000 0000 00MM MMMM MMMM

    所有的 fp16 非规约数转换成 fp32 形式时, 应该都是规约数
    所有的 fp16 非规约数可以表示为 mantissa * 2^{-24}, 注意用这种形式表示数时, 应该在高位补零
    所有的 fp32 规约数可以表示为 (1 + mantissa * 2^{-23}) * 2^{exponent - 127}
    当 exponent=126 时, fp32 规约数形式为 0.5 + mantissa * 2^{-24}
    ***************************************************************************************************** */
    const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

  /*
   * - Choose either results of conversion of input as a normalized number, or
   * as a denormalized number, depending on the input exponent. The variable
   * two_w contains input exponent in bits 27-31, therefore if its smaller than
   * 2**27, the input is either a denormal number, or zero.
   * - Combine the result of conversion of exponent and mantissa with the sign
   * of the input number.
   */
  constexpr uint32_t denormalized_cutoff = UINT32_C(1) << 27;
  const uint32_t result = sign |
      (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value)
                                   : fp32_to_bits(normalized_value));
  return fp32_from_bits(result);
}
