#!/bin/bash

INPUT_DIR_PATH="./inputs/corpus/"
OUTPUT_DIR_PATH="./outputs/stream_demo/"
# bash 中是不支持转义字符的, 这需要程序自行处理。如果指令涉及到 空白字符, 就需要添加双引号
MAP_INPUT_SEP="	"
MAP_OUTPUT_SEP="	"
REDUCE_INPUT_SEP=","
REDUCE_OUTPUT_SEP=","

if [ -d $OUTPUT_DIR_PATH ]; then
    echo "输出文件夹 $OUTPUT_DIR_PATH 存在, 删除"
    rm -rf $OUTPUT_DIR_PATH
fi

# MapTask/ReduceTask 执行过程如下:
#   1. 读取 split/partition 数据, 然后将所有的 "记录" 按照 `key + 分隔符 + value + 换行符` 输出到 文件 f 中
#   2. 启动 mapper/reducer 程序, 并将程序的输入流 (stdin) 重定向到 f 文件
#   3. 等待 mapper/reducer 程序执行完成
#   3. 按行读取将 mapper/reducer 程序的输出流, 根据 分隔符 进行切分, 获取到输出 key & value
#   4. 后续按照 MapReduce 框架正常的流程走

# MapTask 和 ReduceTask 的执行流程是差不多的, 区别在于输入给 reducer 程序的 "记录" 已经按照 key 排序好了。
# 和 Java API 相比, 我们在 reducer 程序中要自己根据 key 进行分组, 其它的没有太多区别。

# 需要注意的是, Hadoop Streaming 使用旧版 MapReduce 组件接口开发的。因此, 在配置中指定 Java 类时, 一定要是 org.apache.hadoop.mapred 包下面的。

/home/huser/hadoop/bin/mapred streaming \
    -D mapreduce.framework.name=local \
    -D mapreduce.job.name=word_count \
    -D stream.map.input.ignoreKey=false \
    -D "stream.map.input.field.separator=$MAP_INPUT_SEP" \
    -D "stream.map.output.field.separator=$MAP_INPUT_SEP" \
    -D "stream.reduce.output.field.separator=$REDUCE_INPUT_SEP" \
    -D "stream.reduce.input.field.separator=$REDUCE_OUTPUT_SEP" \
    -D mapreduce.job.reduces=2 \
    -fs file:/// \
    -files ./stream_demo/word_count.py \
    -input $INPUT_DIR_PATH \
    -inputformat org.apache.hadoop.mapred.TextInputFormat \
    -output $OUTPUT_DIR_PATH \
    -partitioner org.apache.hadoop.mapred.lib.HashPartitioner \
    -mapper "./stream_demo/word_count.py --type mapper --input_sep \"$MAP_INPUT_SEP\" --output_sep \"$MAP_OUTPUT_SEP\"" \
    -reducer "./stream_demo/word_count.py --type reducer --input_sep \"$REDUCE_INPUT_SEP\" --output_sep \"$REDUCE_OUTPUT_SEP\"" \
    -combiner "./stream_demo/word_count.py --type reducer --input_sep \"$REDUCE_INPUT_SEP\" --output_sep \"$REDUCE_OUTPUT_SEP\""
