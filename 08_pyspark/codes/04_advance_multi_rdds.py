
import itertools
from typing import *

from pyspark.rdd import (
    T, K, V, U, # typing
    RDD, portable_hash,
)


def intersection(self: "RDD[T]", other: "RDD[T]", numPartitions: int = None, partitionFunc = portable_hash) -> "RDD[T]":
    # 取 self 和 other 的交集, 类似于 self & other
    pair_rdd1 =  self.map(lambda v: (v, 1))
    pair_rdd2 = other.map(lambda v: (v, 2))
    pair_rdd = pair_rdd1.union(pair_rdd2)
    # 按位运算: 任何数字和其本身 & 运算等于其本身, 同时 1 & 2 = 0
    pair_rdd = pair_rdd.reduceByKey(lambda f1, f2: f1 & f2, numPartitions, partitionFunc)
    # 此时的 pair_rdd 的 value 含义: 0 表示 key 值出现两个 RDD 中
    # 1 表示 key 只在 RDD1 中出现, 2 表示 key 只在 RDD2 中出现
    pair_rdd = pair_rdd.filter(lambda kv: kv[1] == 0)
    return pair_rdd.keys()


def subtract_(self: "RDD[T]", other: "RDD[T]", numPartitions: int = None, partitionFunc = portable_hash) -> "RDD[T]":
    # 取 self 和 other 的差集, 类似于 self - other (这种做法会对结果集进行去重)
    pair_rdd1 = self.map(lambda v: (v, 1))
    pair_rdd2 = other.map(lambda v: (v, 0))
    pair_rdd = pair_rdd1.union(pair_rdd2)
    # 按位运算: 任何数字和其本身 & 运算等于其本身, 同时 1 & 0 = 0
    pair_rdd = pair_rdd.reduceByKey(lambda f1, f2: f1 & f2, numPartitions, partitionFunc)
    pair_rdd = pair_rdd.filter(lambda kv: kv[1] == 1)
    rdd = pair_rdd.keys()
    return rdd


def subtract(self: "RDD[T]", other: "RDD[T]", numPartitions: int = None, partitionFunc = portable_hash) -> "RDD[T]":
    # 取 self 和 other 的差集, 类似于 self - other (和 PySpark 中保持一致, 不会对结果集进行去重)
    pair_rdd1 = self.map(lambda v: (v, 1))
    pair_rdd2 = other.map(lambda v: (v, 0))
    pair_rdd = pair_rdd1.union(pair_rdd2)

    pair_rdd = pair_rdd.aggregateByKey(
        zeroValue=(0, 1),  # (count, flag)
        seqFunc=lambda c, f: (c[0] + f, c[1] & f), 
        combFunc=lambda c1, c2: (c1[0] + c2[0], c1[1] & c2[1]),
        numPartitions=numPartitions, partitionFunc=partitionFunc
    )

    def map_func(element):
        key, (count, flag) = element
        if flag == 0:
            return 
        yield from itertools.repeat(key, count)

    rdd = pair_rdd.flatMap(map_func)
    return rdd 


def standard_join(
    self: "RDD[Tuple[K, V]]", other: "RDD[Tuple[K, U]]", 
    mode: str = "inner", numPartitions: int = None, partitionFunc = portable_hash,
) -> "RDD[Tuple[K, Tuple[V, U]]]":

    pair_rdd1 =  self.mapValues(lambda value: (1, value))
    pair_rdd2 = other.mapValues(lambda value: (2, value))
    pair_rdd = pair_rdd1.union(pair_rdd2)
    # join 运算的一个弊端, 只有 分组, 没有 聚合 过程
    pair_rdd = pair_rdd.groupByKey(numPartitions=numPartitions, partitionFunc=partitionFunc)

    def map_func(iterator):
        buffer1, buffer2 = [], []
        for src, value in iterator:
            if src == 1:
                buffer1.append(value)
                continue
            buffer2.append(value)

        if mode in ("left", "full") and len(buffer1) == 0:
            buffer1.append(None)
        elif mode in ("right", "full") and len(buffer2) == 0:
            buffer2.append(None)

        yield from itertools.product(buffer1, buffer2)

    pair_rdd = pair_rdd.flatMapValues(map_func)
    return pair_rdd 
