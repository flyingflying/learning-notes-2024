
import math 
import torch 


def advance_bin(value: int, num_bits: int) -> str:
    res = bin(value)[2:]
    res = '0' * (num_bits - len(res)) + res 
    return res 


def get_bit(value: int, pos: int) -> int:
    return (value >> (pos - 1)) & 0x01


def show_float8_e4m3fn():
    # e4m3fn: 4 位阶码 3 位尾码, 无 inf
    # 阶码取值范围 [-7, 8], 指数偏移 7

    def get_mantissa_value(m_byte):
        return get_bit(m_byte, 3) * math.pow(2, -1) + get_bit(m_byte, 2) * math.pow(2, -2) + get_bit(m_byte, 1) * math.pow(2, -3)

    def get_sign_value(s_byte):
        return 1 if s_byte == 0 else -1

    values = []

    for sign_byte in range(2):
        for exponent in range(-7, 9):

            if exponent == -7:  # 非规约数
                integer_part = 0.0
                exponent += 1
            else:  # 规约数
                integer_part = 1.0

            for mantissa_byte in range(8):
                value = get_sign_value(sign_byte) * (integer_part + get_mantissa_value(mantissa_byte)) * 2 ** exponent
                encode_str = " ".join([advance_bin(sign_byte, 1), advance_bin(exponent + 7, 4), advance_bin(mantissa_byte, 3)])

                # 处理特殊值
                if encode_str[2:] == '1111 111':
                    value = float("nan")

                true_value = torch.tensor(value, dtype=torch.float8_e4m3fn).item() 
                print(f"{len(values):03d}", encode_str, f"{value:15.10f}", math.isnan(value) or value == true_value, sep="\t\t")

                values.append(value)
    
    tensor = torch.tensor(values, dtype=torch.float8_e4m3fn)
    unique_values = set(tensor.tolist())
    # 255: 负零 和 真零 会判定为 相等; nan 之间会判定为不相等
    print(f"一共有 {len(unique_values)} 个不同值。")
    
    return values


def show_float8_e5m2():
    # e5m2: 5 位阶码 2 位尾码, IEEE 754 标准
    # 阶码取值范围 [-15, 16], 指数偏移 15

    def get_mantissa_value(m_byte):
        return get_bit(m_byte, 2) * math.pow(2, -1) + get_bit(m_byte, 1) * math.pow(2, -2)

    def get_sign_value(s_byte):
        return 1 if s_byte == 0 else -1

    values = []

    for sign_byte in range(2):
        for exponent in range(-15, 16):

            if exponent == -15:  # 非规约数
                integer_part = 0.0
                exponent += 1
            else:  # 规约数
                integer_part = 1.0

            for mantissa_byte in range(4):
                value = get_sign_value(sign_byte) * (integer_part + get_mantissa_value(mantissa_byte)) * 2 ** exponent
                encode_str = " ".join([advance_bin(sign_byte, 1), advance_bin(exponent + 15, 5), advance_bin(mantissa_byte, 2)])

                true_value = torch.tensor(value, dtype=torch.float8_e5m2).item() 
                print(f"{len(values):03d}", encode_str, f"{value:15.10f}", value == true_value, sep="\t\t")

                values.append(value)
        
        exponent = 16  # 特殊值
        for mantissa_byte in range(4):
            if mantissa_byte == 0:  # inf
                value = get_sign_value(sign_byte) * float("inf")
                encode_str = " ".join([advance_bin(sign_byte, 1), advance_bin(exponent + 15, 5), advance_bin(mantissa_byte, 2)])

                true_value = torch.tensor(value, dtype=torch.float8_e5m2).item() 
                print(f"{len(values):03d}", encode_str, f"{value:15.10f}", value == true_value, sep="\t\t")

                values.append(value)
            else:  # nan
                value = get_sign_value(sign_byte) * float("nan")
                encode_str = " ".join([advance_bin(sign_byte, 1), advance_bin(exponent + 15, 5), advance_bin(mantissa_byte, 2)])

                print(f"{len(values):03d}", encode_str, f"{value:15.10f}", math.isnan(value), sep="\t\t")

                values.append(value)
    
    tensor = torch.tensor(values, dtype=torch.float8_e5m2)
    unique_values = set(tensor.tolist())
    # 255: 负零 和 真零 会判定为 相等; nan 之间会判定为不相等
    print(f"一共有 {len(unique_values)} 个不同值。")
    
    return values


if __name__ == "__main__":
    show_float8_e5m2()
