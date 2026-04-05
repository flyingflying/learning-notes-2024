
# %%

from typing import *
from json_str import encode_string, encode_string_ascii

JSON_TYPE = Union[int, float, str, bool, None, list, dict]

# %%

class JSONEncoder:
    def __init__(
            self, *,
            skipkeys: bool = False,  
            ensure_ascii: bool = True, 
            check_circular: bool = True,
            allow_nan: bool = True, 
            indent: int | str | None = None, 
            separators: Tuple[str, str] = None,
            default = None, 
            sort_keys: bool = False
        ):

        self.skipkeys = skipkeys
        if ensure_ascii:
            self.encode_str = encode_string_ascii
        else:
            self.encode_str = encode_string
        if check_circular:
            self.markers = set()
        else:
            self.markers = None 
        self.allow_nan = allow_nan
        if indent is not None and isinstance(indent, int):
            self.indent = " " * indent 
        else:
            self.indent = indent
        if separators is not None:
            self.item_separator, self.key_separator = separators
        else:
            self.item_separator = ', '
            self.key_separator = ': '
        if default is not None:
            self.default = default 
        self.sort_keys = sort_keys

    def encode(self, obj: JSON_TYPE) -> str:  # 入口函数
        return "".join(self.iter_encode_obj(obj))
        # return self.encode_obj(obj)

    def iter_encode_obj(self, obj: JSON_TYPE, current_indent_level: int = 0) -> Iterable[str]:
        if isinstance(obj, (int, float, bool)) or obj is None:
            yield self.encode_sobj(obj)
        elif isinstance(obj, str):
            yield self.encode_str(obj)
        elif isinstance(obj, (list, tuple)):
            yield from self.iter_encode_list(obj, current_indent_level)
        elif isinstance(obj, dict):
            yield from self.iter_encode_dict(obj, current_indent_level)
        else:
            self.mark_obj(obj)
            tobj = self.default(obj)
            # TODO: 个人认为, 这里的 tobj 应该加上 JSON_TYPE 类型限制, 不然还是有 无限递归 的可能
            yield from self.iter_encode_obj(tobj, current_indent_level)
            self.unmark_obj(obj)

    def encode_obj(self, obj: JSON_TYPE, current_indent_level: int = 0) -> str:
        buffer = []

        if isinstance(obj, (int, float, bool)) or obj is None:
            buffer.append(self.encode_sobj(obj))
        elif isinstance(obj, str):
            buffer.append(self.encode_str(obj))
        elif isinstance(obj, (list, tuple)):
            buffer.append(self.encode_list(obj, current_indent_level))
        elif isinstance(obj, dict):
            buffer.append(self.encode_dict(obj, current_indent_level))
        else:
            self.mark_obj(obj)
            tobj = self.default(obj)
            buffer.append(self.encode_obj(tobj, current_indent_level))
            self.unmark_obj(obj)
        
        return "".join(buffer)

    def iter_encode_list(self, obj: list, current_indent_level: int) -> Iterable[str]:
        # 处理空数组
        if len(obj) == 0:
            yield '[]'
            return

        self.mark_obj(obj)

        # 抛出第一个元素
        yield '['
        if self.indent is not None:
            current_indent_level += 1
            yield '\n' + self.indent * current_indent_level
        yield from self.iter_encode_obj(obj[0], current_indent_level)

        for value in obj[1:]:
            yield self.item_separator
            if self.indent is not None:
                yield '\n' + self.indent * current_indent_level
            yield from self.iter_encode_obj(value, current_indent_level)
        
        if self.indent is not None:
            yield '\n' + self.indent * (current_indent_level - 1)
        yield ']'

        self.unmark_obj(obj)

    def encode_list(self, obj: list, current_indent_level: int) -> str:
        # 处理空数组
        if len(obj) == 0:
            return '[]'

        self.mark_obj(obj)
        buffer = []

        # 抛出第一个元素
        buffer.append('[')
        is_first = True
        current_indent_level += 1

        for value in obj:
            if is_first:
                is_first = False
            else:
                buffer.append(self.item_separator)

            if self.indent is not None:
                buffer.append('\n' + self.indent * current_indent_level)
            buffer.append(self.iter_encode_obj(value, current_indent_level))

        if self.indent is not None:
            buffer.append('\n' + self.indent * (current_indent_level - 1))
        buffer.append(']')

        self.unmark_obj(obj)
        return "".join(buffer)

    def iter_encode_dict(self, obj: dict, current_indent_level: int) -> Iterable[str]:
        if len(obj) == 0:
            yield "{}"
            return 
        
        self.mark_obj(obj)

        yield '{'
        is_first = True
        current_indent_level += 1

        if self.sort_keys:
            items = sorted(obj.items())  # dict 的 key 值唯一, 因此排序 key 函数不用设置成 operator.itemgetter(1)
        else:
            items = obj.items()  # 插入序

        for key, value in items:
            if not isinstance(key, str):            
                # JSON 中要求 key 必须是字符串, Python 中 dict 的 key 值只要是 Hashable 即可。因此这里进行了扩展: 
                #       (1) 如果 key 值是 int, float, bool 或者 None 对象, 则将他们转换成字符串
                #       (2) 如果不在上述类型中, 当 skipkeys 入参为 True, 则跳过该 KV 元素, 否则则报错
                if isinstance(key, (int, float, bool)) or key is None:
                    key = self.encode_sobj(key)
                elif self.skipkeys:
                    continue
                else:
                    raise TypeError(f'keys must be str, int, float, bool or None, not {key.__class__.__name__}')

            if is_first:
                is_first = False
            else:
                yield self.item_separator

            if self.indent is not None:
                yield '\n' + self.indent * current_indent_level

            yield self.encode_str(key)
            yield self.key_separator
            yield from self.iter_encode_obj(value, current_indent_level)
        
        current_indent_level -= 1
        if self.indent is not None:
            yield '\n' + self.indent * current_indent_level
        yield '}'

        self.unmark_obj(obj)

    def encode_dict(self, obj: dict, current_indent_level: int) -> str:
        if len(obj) == 0:
            return "{}"

        buffer = []
        self.mark_obj(obj)

        buffer.append('{')
        is_first = True
        current_indent_level += 1

        if self.sort_keys:
            items = sorted(obj.items())  # dict 的 key 值唯一, 因此排序 key 函数不用设置成 operator.itemgetter(1)
        else:
            items = obj.items()  # 插入序

        for key, value in items:
            if not isinstance(key, str):
                if isinstance(key, (int, float, bool)) or key is None:
                    key = self.encode_sobj(key)
                elif self.skipkeys:
                    continue
                else:
                    raise TypeError(f'keys must be str, int, float, bool or None, not {key.__class__.__name__}')

            if is_first:
                is_first = False
            else:
                buffer.append(self.item_separator)

            if self.indent is not None:
                buffer.append('\n' + self.indent * current_indent_level)

            buffer.append(self.encode_str(key))
            buffer.append(self.key_separator)
            buffer.append(self.encode_obj(value, current_indent_level))

        current_indent_level -= 1
        if self.indent is not None:
            buffer.append('\n' + self.indent * current_indent_level)
        buffer.append('}')

        self.unmark_obj(obj)
        return "".join(buffer)

    def mark_obj(self, obj):
        if self.markers is None:
            return 
        obj_id = id(obj)
        if obj_id in self.markers:
            raise ValueError("Circular reference detected")
        self.markers.add(obj_id)
    
    def unmark_obj(self, obj):
        if self.markers is not None:
            self.markers.remove(id(obj))

    def encode_sobj(self, obj: int | float | bool | None) -> str:
        # 编码非集合的对象
        if obj is None:
            return "null"
        if obj is True:
            return 'true'
        if obj is False:
            return 'false'
        if isinstance(obj, int):
            return repr(obj)
        if isinstance(obj, float):
            return self.encode_float(obj)
        raise NotImplementedError

    def encode_float(self, obj: float) -> str:
        if obj != obj:
            text = 'NaN'
        elif obj == float("inf"):
            text = 'Infinity'
        elif obj == float("-inf"):
            text = '-Infinity'
        else:
            return repr(obj)

        if self.allow_nan:
            return text

        raise ValueError("Out of range float values are not JSON compliant: " + repr(obj))
    
    def default(self, obj: object) -> object:
        raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

# %%

if __name__ == "__main__":

    import json

    encoder = JSONEncoder(indent=4)

    print(encoder.encode([1, 2, 3, 4, [6, 7, 8.8, float("nan"), [float("inf")]], [8, 9]]))
