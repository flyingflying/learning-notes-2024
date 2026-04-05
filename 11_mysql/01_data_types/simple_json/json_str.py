
# %%

import re 
from json import JSONDecodeError

__all__ = ["encode_string", "encode_string_ascii", "decode_string"]

# %%

ESCAPE_DICT = {  # key 都是单字符, value 都是双字符
    '\\': '\\\\',  # 反斜杠 (92) -> 反斜杠 + 反斜杠
    '"' : '\\"',   # 双引号 (34) -> 反斜杠 + 双引号
    '\b': '\\b',   # 退格符 (08) -> 反斜杠 + b
    '\t': '\\t',   # 制表符 (09) -> 反斜杠 + t
    '\n': '\\n',   # 换行符 (10) -> 反斜杠 + n
    '\f': '\\f',   # 分页符 (12) -> 反斜杠 + f
    '\r': '\\r',   # 回车符 (13) -> 反斜杠 + r    
}

BACKSLASH = {  # 用于反转义
    '\\': '\\', 
    '"': '"', 
    '/': '/',  # 其它的 编码库会编码 正斜杠 也进行转义
    'b': '\b', 
    't': '\t', 
    'n': '\n', 
    'f': '\f', 
    'r': '\r', 
}

# %%

def _repl_func(match: re.Match) -> str:
    char = match.group(0)
    try:
        return ESCAPE_DICT[char]
    except KeyError:
        pass 

    cp = ord(char)  # code point
    if cp < 0x10000:  # cp < 65536
        return f"\\u{cp:04x}"  # \u + 四位十六进制数

    # 非 BMP 平面内的字符, 采取 双转义字符 编码 (借鉴 UTF-16 编码的思想)
    cp -= 0x10000  # cp -= 65536: 此时 cp 最多 20 位

    # 保留 cp 的高 10 位, 然后将高 6 位赋值为 `110111`
    s1 = 0xd800 | ((cp >> 10) & 0x3ff)
    # 保留 n 的低 10 位, 然后将高 6 位赋值为 `110110`
    s2 = 0xdc00 | (cp & 0x3ff)
    return f"\\u{s1:04x}\\u{s2:04x}"


def encode_string(str_: str) -> str:
    # 寻找 str_ 中的所有转义字符, 调用 _repl_func 进行替换
    # 这里仅仅匹配 码值 为 0 - 31 之间的控制字符, 以及 反斜杠 和 双引号
    return '"' + re.sub(r'[\x00-\x1f\\"\b\f\n\r\t]', _repl_func, str_) + '"'


def encode_string_ascii(str_: str) -> str:
    # [^\ -~]: 除了第 32 (空格) 到第 126 (~) 之间的可打印字符, 其它都匹配
    return '"' + re.sub(r'([\\"]|[^\ -~])', _repl_func, str_) + '"'

# %%

def _decode_uXXXX(str_: str, pos: int):
    esc = re.compile(r'[0-9A-Fa-f]{4}').match(str_, pos)
    if esc is not None:
        try:
            return int(esc.group(), 16)
        except ValueError:
            pass
    msg = "Invalid \\uXXXX escape"
    raise JSONDecodeError(msg, str_, pos)


def decode_string(json_str: str, cur_pos: int, strict: bool = True) -> tuple[str, int]:
    chunks = []
    begin = cur_pos - 1  # 字符串的起始位置, 用于异常输出的

    while True:
        # 非贪婪模式匹配字符, 直到遇到 双引号, 反斜杠 和 控制字符
        chunk = re.compile(r'(.*?)(["\\\x00-\x1f])', re.DOTALL).match(json_str, cur_pos)
        if chunk is None:
            msg = "Unterminated string starting at"
            raise JSONDecodeError(msg, json_str, begin)

        cur_pos = chunk.end()
        content, terminator = chunk.groups()
        chunks.append(content)  # 如果 '.*?' 没有匹配到内容, 返回 空串, 不会返回 None

        if terminator == '"':  # 字符串结束
            break

        if terminator != '\\':  # 控制字符
            if strict:  # 严格按照 JSON 标准
                msg = f"Invalid control character {repr(terminator)} at"
                raise JSONDecodeError(msg, json_str, cur_pos)
            else:
                chunks.append(terminator)
                continue

        # 转义字符的情况
        try:
            esc = json_str[cur_pos]
        except IndexError:
            msg = "Unterminated string starting at"
            raise JSONDecodeError(msg, json_str, begin) from None

        if esc != 'u':
            try:
                chunks.append(BACKSLASH[esc])
            except KeyError:
                msg = f"Invalid \\escape: {repr(esc)}"
                raise JSONDecodeError(msg, json_str, cur_pos) from None
            cur_pos += 1
            continue

        uni = _decode_uXXXX(json_str, cur_pos + 1)
        cur_pos += 5
        if 0xd800 <= uni <= 0xdbff and json_str[cur_pos:cur_pos + 2] == '\\u':  # 双转义字符
            uni2 = _decode_uXXXX(json_str, cur_pos + 1)
            if 0xdc00 <= uni2 <= 0xdfff:
                uni = 0x10000 + (((uni - 0xd800) << 10) | (uni2 - 0xdc00))
                cur_pos += 6
        chunks.append(chr(uni))

    return ''.join(chunks), cur_pos


# %%

if __name__ == '__main__':
    import json 

    test_case = 'abc\n"\\abc我' + chr(0)

    print(encode_string(test_case))
    print(encode_string(test_case) == json.dumps(test_case, ensure_ascii=False))

    print(encode_string_ascii(test_case))
    print(encode_string_ascii(test_case) == json.dumps(test_case, ensure_ascii=True))

    print(decode_string(encode_string(test_case), 1))
    print(decode_string(encode_string_ascii(test_case), 1)) 

    print(decode_string(encode_string(test_case), 1)[0] == test_case)
    print(decode_string(encode_string_ascii(test_case), 1)[0] == test_case) 

# %%
