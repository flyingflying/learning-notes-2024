#!/home/lqxu/miniconda3/bin/python

import re
import sys
import argparse 


def get_running_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--type", choices=["mapper", "reducer"], required=True, help="程序运行的类型")
    parser.add_argument("--input_sep", type=str, default="\t", required=False, help="输入记录分隔符")
    parser.add_argument("--output_sep", type=str, default="\t", required=False, help="输出记录分隔符")
    
    return parser.parse_args()


def in_records_iterator(in_sep: str, debug: bool):
    in_records = iter(sys.stdin)

    for in_record in in_records:
        in_record = in_record.strip("\n")
        if not in_record:
            continue

        if debug:
            sys.stderr.write(f"输入的内容是: {in_record}\n")

        in_key, in_value = in_record.split(in_sep, 1)
        yield in_key, in_value


def write_out_record(out_key: any, out_value: any, out_sep: str, debug: bool):
    out_record = f"{out_key}{out_sep}{out_value}"

    if debug:
        sys.stderr.write(f"输出的内容是: {out_record}\n")

    sys.stdout.write(f"{out_record}\n")


def main_mapper(in_sep: str = "\t", out_sep: str = "\t", debug: bool = False):
    tokenize = re.compile(r"\s+").split 

    for offset, sentence in in_records_iterator(in_sep, debug):
        offset = int(offset)
        words = tokenize(sentence.strip())

        for word in words:
            # 检测是按 key 还是 key + value 的方式进行排序, 答案是 key 
            # import random
            # write_out_record(out_key=word, out_value=random.randint(10, 20), out_sep=out_sep, debug=debug)
            write_out_record(out_key=word, out_value=1, out_sep=out_sep, debug=debug)


def main_reducer(in_sep: str = "\t", out_sep: str = "\t", debug: bool = False):
    cur_word, cur_count = None, None

    for word, count in in_records_iterator(in_sep, debug):
        count = int(count)

        if word == cur_word:
            cur_count += count 
            continue

        if cur_word is not None:  # 初始化的时候, cur_word 是 None, 需要过滤掉
            write_out_record(out_key=cur_word, out_value=cur_count, out_sep=out_sep, debug=debug)

        cur_word = word
        cur_count = count 

    if cur_word is not None:
        write_out_record(out_key=cur_word, out_value=cur_count, out_sep=out_sep, debug=debug)


if __name__ == "__main__":

    """
    如果你在网上看到使用 Python 进行 MapReduce 编程的文章, 比方说 [博客](https://www.cnblogs.com/kaituorensheng/p/3826114.html), 
    那都是用 Hadoop 官方提供的 [streaming](https://hadoop.apache.org/docs/current/hadoop-streaming/HadoopStreaming.html) 工具实现的。
    Hadoop Streaming 将 map 运算和 reduce 运算转换成两个可执行的程序, 这个程序可以使用任意语言 (shell, C/C++, Python, etc.) 实现, 我们将这两个程序称为 mapper 程序和 reducer 程序。
    """

    args = get_running_arguments()

    if args.type == "mapper":
        main_mapper(args.input_sep, args.output_sep, debug=True)
    else:
        main_reducer(args.input_sep, args.output_sep, debug=True)
