
# %%

import re 
from typing import Callable, Any, Union
from json import JSONDecodeError 
from json_str import decode_string

# %%

JSON_TYPE = Union[int, float, str, bool, None, list, dict]

# %%


def erase_whitespace(json_str: str, cur_pos: int = 0) -> int:
    return re.compile(r'[ \t\n\r]*').match(json_str, cur_pos).end()


def convert_kv_pairs(
        pairs: list[tuple[str, Any]], 
        object_hook: Callable[[dict[str, Any], ], Any] = None, 
        object_pairs_hook: Callable[[list[tuple[str, Any]], ], Any] = None, 
    ) -> dict:
    
    if object_pairs_hook is not None:  # highest priority
        return object_pairs_hook(pairs)
    
    if object_hook is not None:
        return object_hook(dict(pairs))
    
    return dict(pairs)


def scan_once(
        json_str: str, cur_pos: int, strict: bool, 
        parse_int, parse_float, parse_constant, object_hook, object_pairs_hook,
    ) -> JSON_TYPE:
    try:
        nextchar = json_str[cur_pos]
    except IndexError:
        raise JSONDecodeError("Expecting value", json_str, cur_pos) from None
    
    if nextchar == '"':
        return decode_string(json_str, cur_pos + 1, strict)
    if nextchar == '{':
        return parse_json_object(json_str, cur_pos + 1, strict, parse_int, parse_float, parse_constant, object_hook, object_pairs_hook)
    if nextchar == '[':
        return parse_json_array(json_str, cur_pos + 1, strict, parse_int, parse_float, parse_constant, object_hook, object_pairs_hook)

    # 解析数字
    # "(?:0|[1-9][0-9]*)": 整数部分: 非捕获组 匹配单个零 或者 `[1-9][0-9]*`, 不允许 整数前缀零
    # "(\.[0-9]+)?": 小数部分, 允许 小数后缀零
    # "[eE][-+]?[0-9]+": 指数部分, 指数位只能是 整数
    match = re.compile(r'(-?(?:0|[1-9][0-9]*))(\.[0-9]+)?([eE][-+]?[0-9]+)?').match(json_str, cur_pos)
    if match is not None:
        integer, frac, exp = match.groups()
        if frac is None and exp is None:
            result = parse_int(integer)
        else:
            result = parse_float(match.group())
        return result, match.end()

    # 解析常量
    if nextchar == "n" and json_str[cur_pos:cur_pos+4] == "null":
        return None, cur_pos + 4
    if nextchar == "t" and json_str[cur_pos:cur_pos+4] == "true":
        return True, cur_pos + 4
    if nextchar == "f" and json_str[cur_pos:cur_pos+5] == "false":
        return False, cur_pos + 5 
    if nextchar == "N" and json_str[cur_pos:cur_pos+3] == "NaN":
        return parse_constant("NaN"), cur_pos + 3
    if nextchar == "I" and json_str[cur_pos:cur_pos+8] == "Infinity":
        return parse_constant("Infinity"), cur_pos + 8
    if nextchar == "-" and json_str[cur_pos:cur_pos+9] == "-Infinity":
        return parse_constant("-Infinity"), cur_pos + 9
    
    raise JSONDecodeError("Expecting value", json_str, cur_pos)

# %%

def parse_json_object(
        json_str: str, cur_pos: int, strict: bool, 
        parse_int, parse_float, parse_constant, object_hook, object_pairs_hook,
    ) -> tuple[dict, int]:

    pairs = []
    # json object 样式: {"key1": value1, "key2": value2}
    # 左大括号 在 scan_once 中已经被消费, 那么第一个非空白字符应该是 双引号 或者 右大括号

    def _get_nextchar(excepted, msg):
        nonlocal json_str, cur_pos
        _nextchar = json_str[cur_pos:cur_pos+1]  # 不需要用 try except 封装
        if _nextchar not in excepted:
            raise JSONDecodeError(msg, json_str, cur_pos - 1)
        cur_pos += 1
        return _nextchar

    cur_pos = erase_whitespace(json_str, cur_pos)  # 清除双引号前的空白字符
    nextchar = _get_nextchar('"}', "Expecting property name enclosed in double quotes")
    if nextchar == '}':  # 空 object
        return convert_kv_pairs(pairs, object_hook, object_pairs_hook), cur_pos

    while True:
        key, cur_pos = decode_string(json_str, cur_pos, strict)  # 获取 key (key 限定只能是字符串)

        cur_pos = erase_whitespace(json_str, cur_pos)  # 消费冒号前的空白字符
        _get_nextchar(':', "Expecting ':' delimiter")  # 消费冒号
        cur_pos = erase_whitespace(json_str, cur_pos)  # 消费冒号后的空白字符

        value, cur_pos = scan_once(json_str, cur_pos, strict, parse_int, parse_float, parse_constant, object_hook, object_pairs_hook)  # 获取 value
        pairs.append((key, value))

        cur_pos = erase_whitespace(json_str, cur_pos)  # 消费逗号前的空白字符
        nextchar = _get_nextchar(',}', "Expecting ',' delimiter")  # 消费逗号
        if nextchar == '}':
            return convert_kv_pairs(pairs, object_hook, object_pairs_hook), cur_pos 
        cur_pos = erase_whitespace(json_str, cur_pos)  # 消费逗号后的空白字符

        _get_nextchar('"', "Expecting property name enclosed in double quotes")  # 消费双引号

# %%

def parse_json_array(
        json_str: str, cur_pos: int, strict: bool, 
        parse_int, parse_float, parse_constant, object_hook, object_pairs_hook
    ) -> tuple[list, int]:

    # json array 样式: [value1, value2]
    # 左中括号 在 scan_once 中已经被消费, 那么第一个非空白字符应该是 value1 或者 右中括号

    values = []

    # 判断是否空列表
    cur_pos = erase_whitespace(json_str, cur_pos)  # 清除左中括号之后的空白字符
    nextchar = json_str[cur_pos:cur_pos+1]
    if nextchar == ']':
        return values, cur_pos + 1

    while True:
        # 获取元素值
        value, cur_pos = scan_once(json_str, cur_pos, strict, parse_int, parse_float, parse_constant, object_hook, object_pairs_hook)
        values.append(value)

        # 消费逗号
        cur_pos = erase_whitespace(json_str, cur_pos)  # 清除逗号前的空白字符
        nextchar = json_str[cur_pos:cur_pos+1]
        cur_pos += 1
        if nextchar == ']':  # 特殊情况
            return values, cur_pos

        if nextchar != ',':
            raise JSONDecodeError("Expecting ',' delimiter", json_str, cur_pos - 1)
        cur_pos = erase_whitespace(json_str, cur_pos)  # 清除逗号后的空白字符

# %%

def json_loads(
        json_str: str, 
        object_hook: Callable[[dict[str, Any], ], Any] = None,  # 转换 json object 
        parse_float: Callable[[str, ], Any] = None,     # 解析浮点数
        parse_int  : Callable[[str, ], Any] = None,     # 解析整数
        parse_constant: Callable[[str, ], Any] = None,  # 解析 inf, -inf 和 nan 三个特殊浮点数
        strict: bool = True,                            # 字符串中的控制字符是否进行转义字符
        object_pairs_hook: Callable[[list[tuple[str, Any]], ], Any] = None,  # 转换 json object 
    ) -> JSON_TYPE:

    # 如果想将数字解析成 Decimal 类, 方式如下:
    # from decimal import Decimal
    # CONSTANT = {"-Infinity": Decimal(float("-inf")), "Infinity": Decimal(float("inf")), "NaN": Decimal(float("nan"))}
    # json.loads('[1, 2.3, NaN]', parse_int=Decimal, parse_float=Decimal, parse_constant=CONSTANT.__getitem__)

    parse_float = parse_float or float
    parse_int   = parse_int   or int

    if parse_constant is None:
        parse_constant = {
            '-Infinity': float("-inf"),
            'Infinity' : float("inf"),
            'NaN': float("nan"),
        }.__getitem__

    cur_pos = erase_whitespace(json_str, 0)
    obj, cur_pos = scan_once(json_str, cur_pos, strict, parse_int, parse_float, parse_constant, object_hook, object_pairs_hook)
    cur_pos = erase_whitespace(json_str, cur_pos)

    if cur_pos != len(json_str):
        raise JSONDecodeError("Extra data", json_str, cur_pos)
    
    return obj 



# %%

if __name__ == '__main__':
    import json 

    demo_obj = ["因合而肥", 1, float("nan"), float("inf"), None, {"1": "haha", None: None, }, {}, True, False, []]
    demo_str = json.dumps(demo_obj)

    print(demo_str)
    print(json_loads(demo_str))

# %%
