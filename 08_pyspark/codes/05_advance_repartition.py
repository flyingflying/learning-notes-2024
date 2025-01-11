
# %%

import sys
import math 
import bisect
import functools 
from typing import *
from collections import defaultdict

from pyspark.rdd import RDD, T, U, portable_hash
from pyspark.traceback_utils import SCCallSiteSync
from pyspark.shuffle import get_used_memory, ExternalSorter
from pyspark.serializers import pack_long, BatchedSerializer


# %% 


def custom_repartition(
        self: RDD[Tuple[int, T]], 
        numPartitions: Optional[int] = None, 
        # partitionFunc: Callable[[T], int] = portable_hash
    ) -> RDD[T]:
    """
    在 PySpark 中, 有两种 "重新分区" 的方式:
        1. repartition: 对 RDD 中的元素进行重新分区, 让每一个分区内的元素尽可能地相等;
        2. partitionBy: 根据 PairRDD 的 key 进行重新分区。

    这里存在一个问题, 那就是 repartition 不能自定义分区方式, 
        这样会造成 sortBy, treeAggregate 以及 treeReduce 等算子在实现的时候需要构建 PairRDD,
        从而增加 shuffle 过程的数据传输。此方法就是为了解决这个问题而存在的。
    """

    outputSerializer = self.ctx._unbatched_serializer  # 默认为 CPickleSerializer
    limit = self._memory_limit() / 2  # 参数: spark.python.worker.memory, 默认为 512MB

    def add_shuffle_key(iterator: Iterable[T]) -> Iterable[bytes]:

        buckets = defaultdict(list)
        c, batch = 0, 1000

        # for element in iterator:
            # p_idx = partitionFunc(element) % numPartitions
        for p_idx, element in iterator:
            buckets[p_idx].append(element)
            c += 1

            # 每 1000 次检查一下内存使用情况
            # 当内存使用超过限制或者或者 buckets 中元素个数超过 batch 时触发
            # and 优先级是大于 or 的, 因此括号可以不需要
            if (c % 1000 == 0 and get_used_memory() > limit) or c > batch:
                n, size = len(buckets), 0  # 用于计算平均字节数 (avg)
                for split in list(buckets.keys()):
                    d = outputSerializer.dumps(buckets[split])
                    size += len(d)
                    del buckets[split]

                    yield pack_long(split)
                    yield d

                avg = int(size / n) >> 20  # Bytes ==> MB (除以 1024 ** 2)
                # 根据 avg 的值调整 batch 的值, 让 avg 的值在 1MB - 10MB 之间
                if avg < 1:
                    batch = min(sys.maxsize, batch * 1.5)  # batch 增加 50%
                elif avg > 10:
                    batch = max(int(batch / 1.5), 1)  # batch 减少 50%
                c = 0

        for split, items in buckets.items():
            yield pack_long(split)
            yield outputSerializer.dumps(items)

    keyed = self.mapPartitions(add_shuffle_key)

    # 调用 Java/Scala 侧函数
    assert self.ctx._jvm is not None
    keyed._bypass_serializer = True
    with SCCallSiteSync(self.context):
        pairRDD = self.ctx._jvm.PairwiseRDD(keyed._jrdd.rdd()).asJavaPairRDD()
        jpartitioner = self.ctx._jvm.PythonPartitioner(numPartitions, id(pairRDD))
    jrdd = self.ctx._jvm.PythonRDD.valueOfPair(pairRDD.partitionBy(jpartitioner))
    rdd: "RDD[T]" = RDD(jrdd, self.ctx, BatchedSerializer(outputSerializer))

    return rdd


# %%


def sortBy(
        self: "RDD[T]", ascending: Optional[bool] = True, 
        numPartitions: Optional[int] = None, keyfunc: Callable[[Any], Any] = None
    ) -> "RDD[T]":

    if numPartitions is None:
        numPartitions = self._defaultReducePartitions()

    memory = self._memory_limit()
    serializer = self._jrdd_deserializer

    def sortPartition(iterator: Iterable[T]) -> Iterable[T]:
        sort = ExternalSorter(memory * 0.9, serializer).sorted
        yield from sort(iterator, key=keyfunc, reverse=(not ascending))

    if numPartitions == 1:
        # 这里直接将当前 stage 的并行度调整成 1 了, 非常危险的操作
        self = self.coalesce(1)
        return self.mapPartitions(sortPartition, True)

    # 获取元素的个数
    rddSize = self.count()  # +1 Job
    if rddSize == 0:  # emptyRDD
        return self

    # 构建 range partitioner
    maxSampleSize = numPartitions * 20.0  # constant from Spark's RangePartitioner
    fraction = min(maxSampleSize / max(rddSize, 1), 1.0)
    samples = self.sample(False, fraction, 1).collect()  # +1 Job
    samples = sorted(samples, key=keyfunc)

    # 如果 numPartitions 的值为 3, 最终采样出来 60 个 key, 那么在排序后, 将第 20 个 key 和第 40 个 key 作为分界线。
    # 当元素的 key 小于第 20 个 key, 新的分区索引为 0; 
    # 当元素的 key 在第 20 至第 40 个 key 之间, 新的分区索引为 1; 
    # 当元素的 key 大于第 40 个 key, 新的分区索引为 2。
    bounds = [
        samples[int(len(samples) * (i + 1) / numPartitions)]
        for i in range(0, numPartitions - 1)
    ]

    def rangePartitioner(k: T) -> int:
        if keyfunc is None:
            p = bisect.bisect_left(bounds, k)
        else:
            # bisect.bisect_left 中的 key 参数仅仅作用于 a 集合, 不会作用于 x 元素
            # 这里是 sortByKey 算子里面有 bug, 应该加上 key 参数
            p = bisect.bisect_left(bounds, keyfunc(k), key=keyfunc)

        if ascending:
            return p
        else:
            return numPartitions - 1 - p

    self = self.keyBy(rangePartitioner)
    self = custom_repartition(self, numPartitions)
    return self.mapPartitions(sortPartition, False)

    # return custom_repartition(self, numPartitions, rangePartitioner).mapPartitions(sortPartition, False)


# %%


def treeAggregate(
    self: "RDD[T]",
    zeroValue: U,
    seqOp: Callable[[U, T], U],
    combOp: Callable[[U, U], U],
    depth: int = 2,
) -> U:

    if depth < 1:
        raise ValueError("Depth cannot be smaller than 1 but got %d." % depth)

    if self.getNumPartitions() == 0:
        return zeroValue

    def aggregatePartition(it):
        yield functools.reduce(seqOp, it, zeroValue)
    
    def combinePartition(it):
        yield functools.reduce(combOp, it, zeroValue)

    # 此时, RDD 的一个分区内只有一个元素, 因此不用 custom_repartition 也是可以的
    partiallyAggregated = self.mapPartitions(aggregatePartition)
    numPartitions = partiallyAggregated.getNumPartitions()
    scale = pow(numPartitions, 1.0 / depth)  # 开 depth 次方
    scale = max(int(math.ceil(scale)), 2)

    while numPartitions > scale + numPartitions / scale:  # depth 是建议的 Stage 数
        numPartitions /= scale
        curNumPartitions = int(numPartitions)

        def mapPartition(i: int, iterator: Iterable[U]) -> Iterable[Tuple[int, U]]:
            for obj in iterator:
                yield (i % curNumPartitions, obj)

        partiallyAggregated = partiallyAggregated.mapPartitionsWithIndex(mapPartition)
        partiallyAggregated = custom_repartition(partiallyAggregated, curNumPartitions)
        partiallyAggregated = partiallyAggregated.mapPartitions(combinePartition)

    return list(combinePartition(partiallyAggregated.collect()))[0]

# %%
