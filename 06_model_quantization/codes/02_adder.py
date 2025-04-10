
INT32_MAX = 2 ** 31 - 1
INT32_MIN = -2 ** 31
UINT32_MAX = 2 ** 32 - 1
UINT32_MIN = 0


def to_uint32(v: int):
    assert isinstance(v, int) and INT32_MIN <= v <= INT32_MAX

    if v < 0:
        return v + UINT32_MAX + 1
    return v 
    # return ctypes.c_uint32(v).value 


def from_uint32(v: int):
    assert isinstance(v, int) and UINT32_MIN <= v <= UINT32_MAX
    if v > INT32_MAX:
        return v - UINT32_MAX - 1
    return v 


def to_bitarray(v: int) -> tuple[int, ...]:
    assert isinstance(v, int) and INT32_MIN <= v <= INT32_MAX

    v = to_uint32(v)  # python 不支持 无符号 右移, 不需要先转换成 uint32
    bitarray = []

    for _ in range(32):
        bitarray.insert(0, v & 0x1)
        v = v >> 1

    return tuple(bitarray)


def from_bitarray(bitarray: tuple[int, ...]) -> int:
    v = bitarray[0]

    for bit in bitarray[1:]:
        assert isinstance(bit, int) and (bit == 0 or bit == 1)
        v = (v << 1) | bit
    
    return from_uint32(v)


def half_adder(bit_a: int, bit_b: int) -> tuple[int, int]:
    bit_carry = bit_a & bit_b
    bit_sum = bit_a ^ bit_b
    return bit_carry, bit_sum


def old_full_adder(bit_a: int, bit_b: int, bit_carry_in: int) -> tuple[int, int]:
    bit_carry1, bit_sum = half_adder(bit_a, bit_b)
    bit_carry2, bit_sum = half_adder(bit_carry_in, bit_sum)
    bit_carry3 = bit_carry1 | bit_carry2
    return bit_carry3, bit_sum


def full_adder(bit_a: int, bit_b: int, bit_carry_in: int) -> tuple[int, int]:
    bit_sum = bit_a ^ bit_b ^ bit_carry_in
    # bit_carry_out = (bit_a | bit_b) & (bit_b | bit_carry_in) & (bit_a | bit_carry_in)
    bit_carry_out = (bit_a & bit_b) | ((bit_a ^ bit_b) & bit_carry_in)
    return bit_carry_out, bit_sum


def check_full_adder():
    for c in (0, 1):
        for b in (0, 1):
            for a in (0, 1):
                print(a, b, c, full_adder(a, b, c))


def int32_adder(v1: int, v2: int):
    # https://zh.wikipedia.org/wiki/加法器 
    bitarray_v1 = to_bitarray(v1)
    bitarray_v2 = to_bitarray(v2)

    bit_carry = 0
    bit_array = []
    for bit_v1, bit_v2 in zip(reversed(bitarray_v1), reversed(bitarray_v2)):
        bit_carry, bit_sum = full_adder(bit_v1, bit_v2, bit_carry)
        bit_array.insert(0, bit_sum)
    
    return from_bitarray(bit_array)


if __name__  == "__main__":
    print(int32_adder(118, 2345) == 118 + 2345)
    print(int32_adder(-234, -345) == -234 - 345)
    print(int32_adder(-567, 789) == -567 + 789)
