
# %%

import operator
import itertools
from typing import *
from pyspark.rdd import RDD
from pyspark import SparkContext, SparkConf

conf = SparkConf()
conf.setMaster("local[*]")
conf.setAppName("test_rdd")
conf.set("spark.ui.enabled", "True")
sc = SparkContext(conf=conf)

# %%


def median_runjob_version(rdd: RDD[int | float]) -> float:
    """ 
    使用 runJob 实现的版本, 最终会触发 4 次 Job。
    当 rdd 中的元素个数为奇数时, 取正中间一个数; 当 rdd 中的元素个数为偶数时, 取正中间两个数的平均数。
    """

    num_partitions = rdd.getNumPartitions()
    # sortByKey 在构建 DAG 时会触发两次 Job
    sorted_rdd: RDD = rdd.keyBy(lambda x: x).sortByKey(True, num_partitions)  # +2 Jobs

    # 获取排序后每一个分区的元素个数, 由于排序使用的是 rangePartitioner, 这个步骤无论如何都没法省略
    part_counts = sorted_rdd.mapPartitions(
        lambda it: [sum(1 for _ in it), ]  # mapPartitions 算子返回的必须是可迭代对象
    ).collect()  # +1 Job

    total_count = sum(part_counts)
    target_idx = total_count // 2 - 1
    bounds = itertools.accumulate(part_counts, operator.add, initial=0)

    # 寻找 target_idx 所在的分区索引
    for p_idx, (lb, ub) in enumerate(itertools.pairwise(bounds)):
        if lb <= target_idx < ub:
            break 

    if total_count % 2 == 1 or target_idx != ub - 1:

        # 只需要运行一个分区的情况
        def map_func(iterator):
            for idx, (k, _) in enumerate(iterator, lb):
                if idx == target_idx:
                    yield k 
                    break 

            # 当 total_count 是偶数时, 需要多取一个数字
            if total_count % 2 == 0:
                yield next(iterator)[0]

        results = rdd.context.runJob(sorted_rdd, partitionFunc=map_func, partitions=[p_idx, ])

    else:
        # 当 total_count 为偶数时, 且 target_idx 正好是当前分区的最后一个元素, 那么我们就要从两个分区中取元素。
        # 由于 runJob 中调用的是 mapPartitions 算子, 而不是 mapPartitionsWithIndex 算子, 
        # 那么我们只能从两个分区中取首尾元素, 汇聚到 Driver 端后再取中间的元素。

        def map_func(iterator):
            yield next(iterator)[0]
            for k, _ in iterator:
                pass 
            yield k 

        # 获取下一个分区索引值, 排除中间分区元素个数为 0 的情况
        for next_p_idx in range(p_idx+1, num_partitions):
            if part_counts[next_p_idx] != 0:
                break 

        results = rdd.context.runJob(sorted_rdd, partitionFunc=map_func, partitions=[p_idx, next_p_idx])
        results = results[1:3]
    
    return sum(results) / len(results)


def check_median_runjob_version():
    import numpy as np 
    from random import Random

    # 测试运行两个分区的情况
    random = Random(0)
    c1 = list(range(50))
    random.shuffle(c1)
    result = median_runjob_version(sc.parallelize(c1, numSlices=2)) 
    gold_result = np.median(c1)
    print(result, gold_result)

# %%
