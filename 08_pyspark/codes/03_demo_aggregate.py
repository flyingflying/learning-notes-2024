
# %%

import heapq
import functools
from typing import *
from pyspark.rdd import portable_hash, RDD, Partitioner
from pyspark import SparkContext, SparkConf

conf = SparkConf()
conf.setMaster("local[*]")
conf.setAppName("test_rdd")
conf.set("spark.ui.enabled", "True")
sc = SparkContext(conf=conf)

# %%


def demo_combineByKey(
        pair_rdd: RDD, createCombiner, mergeValue, mergeCombiners, 
        numPartitions: int = None, partitionFunc = portable_hash
    ):

    """ combineByKey 没有 "溢写" 的版本 """

    def map_func_before_shuffle(iterator):
        result = dict()
        for k, v in iterator:
            if k in result:
                result[k] = mergeValue(result[k], v)
            else:
                result[k] = createCombiner(v)
        yield from result.items()
    
    def map_func_after_shuffle(iterator):
        result = dict()
        for k, v in iterator:
            if k in result:
                result[k] = mergeCombiners(result[k], v)
            else:
                result[k] = v 
        yield from result.items()

    pair_rdd = pair_rdd.mapPartitions(map_func_before_shuffle, preservesPartitioning=True)

    if pair_rdd.partitioner is not None:  # pair_rdd 已分区
        if numPartitions is None:
            numPartitions = pair_rdd._defaultReducePartitions()
        partitioner = Partitioner(numPartitions, partitionFunc)
        if pair_rdd.partitioner == partitioner:  # 分区数 和 分区函数 一致
            return pair_rdd

    pair_rdd = pair_rdd.partitionBy(numPartitions, partitionFunc)
    pair_rdd = pair_rdd.mapPartitions(map_func_after_shuffle, preservesPartitioning=True)
    return pair_rdd 


def check_demo_combineByKey(check_shuffle: bool = False):
    import random 
    import string
    import operator

    N_ELEMENTS = 10_0000
    keys = [random.choice(string.ascii_lowercase) for _ in range(N_ELEMENTS)]
    values = [random.random() * 3 for _ in range(N_ELEMENTS)]
    pair_rdd = sc.parallelize(zip(keys, values))

    if check_shuffle:
        pair_rdd.partitionBy(pair_rdd.getNumPartitions())

    result = demo_combineByKey(pair_rdd, lambda x: x, operator.add, operator.add).collectAsMap()
    gold_result = pair_rdd.combineByKey(lambda x: x, operator.add, operator.add).collectAsMap()

    assert result == gold_result
    print(result)
    print(gold_result)


# %%

def demo_spill():
    import operator
    from itertools import chain
    from pyspark.shuffle import ExternalMerger, Aggregator

    aggregator = Aggregator(lambda x: x, operator.add, operator.add)

    keys = ['a', 'b', 'a', 'b']
    values = [1, 2, 3, 4]
    merger = ExternalMerger(aggregator=aggregator, memory_limit=1, batch=1)
    merger.mergeValues(zip(keys, values))  # 溢写的同时会调用 mergeCombiners
    result = list(merger.items())
    print(result)

    merger = ExternalMerger(aggregator=aggregator, memory_limit=1, batch=1)
    merger.mergeCombiners(chain.from_iterable([result, ] * 3))
    result = list(merger.items())
    print(result)

# %%


def demo_group_variance():
    import random 
    import string 
    from pyspark.statcounter import StatCounter

    N_ELEMENTS = 1000

    keys = [random.choice(string.ascii_lowercase) for _ in range(N_ELEMENTS)]
    values = [random.random() * 3 for _ in range(N_ELEMENTS)]

    pair_rdd = sc.parallelize(zip(keys, values))
    pair_rdd = pair_rdd.combineByKey(
        createCombiner=lambda v: StatCounter([v, ]),
        mergeValue=lambda c, v: c.merge(v),
        mergeCombiners=lambda c1, c2: c1.mergeStats(c2)
    ).mapValues(lambda c: c.variance())

    print(pair_rdd.collect())
    
# %%

def demo_reduceByKeyLocally(self: RDD, func: Callable) -> Dict:

    def map_func(iterator):
        # 标准的 dict 分组聚合运算
        result = dict()
        for key, value in iterator:
            if key in result:
                result[key] = func(result[key], value)
            else:
                result[key] = value 
        yield result 
    
    results = self.mapPartitions(map_func).collect()

    def reduce_func(result1: Dict, result2: Dict):
        # reduce 运算的二元函数: 第一个参数是 combiner, 第二个参数 value
        # 这里将 result2 的结果并入到 result1 中, 然后返回 result1
        for key, value in result2.items():
            if key in result1:
                result1[key] = func(result1[key], value)
            else:
                result1[key] = value
        return result1 
    
    return functools.reduce(reduce_func, results, dict())

# %%

def check_demo_reduceByKeyLocally():
    import random
    import string
    import operator 

    num_elements = 10_0000
    keys = [random.choice(string.ascii_lowercase) for _ in range(num_elements)]
    values = [random.random() for _ in range(num_elements)]
    rdd = sc.parallelize(zip(keys, values))

    result = demo_reduceByKeyLocally(rdd, operator.add)
    gold_result = rdd.reduceByKeyLocally(operator.add)
    assert sorted(gold_result.items()) == sorted(result.items())
    print(sorted(result.items()))

# %%

def demo_reduceByKeyLocally(self: RDD, func: Callable) -> Dict:

    def sequence_func(result: Dict, element: Tuple):
        # 标准的 dict 分组聚合运算
        key, value = element
        if key in result:
            result[key] = func(result[key], value)
        else:
            result[key] = value 
        return result 

    def combine_func(result1: Dict, result2: Dict):
        # reduce 运算的二元函数: 第一个参数是 combiner, 第二个参数 value
        # 这里将 result2 的结果并入到 result1 中, 然后返回 result1
        for key, value in result2.items():
            if key in result1:
                result1[key] = func(result1[key], value)
            else:
                result1[key] = value
        return result1 
    
    return self.aggregate(zeroValue=dict(), seqOp=sequence_func, combOp=combine_func)

# %%

def demo_top(self: RDD, num: int, key: Callable = None) -> List:

    if key is None:
        def key(element):  # identity func
            return element
    
    def sequence_func(result: List, element):
        if len(result) < num:
            result.append(element)
            if len(result) == num:
                result.sort(key=key, reverse=True)

        elif key(element) > key(result[-1]):
            result[-1] = element
            result.sort(key=key, reverse=True)

        return result 
    
    def combine_func(result1: List, result2: List):
        # return sorted(result1 + result2, key=key, reverse=True)[:num]
        return heapq.nlargest(num, result1 + result2, key=key)
    
    return self.aggregate(zeroValue=list(), seqOp=sequence_func, combOp=combine_func)


# %%

def demo_group_top(self: RDD, num: int, key: Callable = None) -> RDD:

    if key is None:
        def key(element):  # identity func
            return element

    def create_combiner(value) -> List:
        return [value, ]
    
    def merge_value(combiner: List, value):
        if len(combiner) < num:
            combiner.append(value)
            if len(combiner) == num:
                combiner.sort(key=key, reverse=True)

        elif key(value) > key(combiner[-1]):
            combiner[-1] = value
            combiner.sort(key=key, reverse=True)

        return combiner

    def merge_combiners(combiner1: List, combiner2: List):
        return heapq.nlargest(num, combiner1 + combiner2, key=key)  

    return self.combineByKey(
        createCombiner=create_combiner, 
        mergeValue=merge_value, 
        mergeCombiners=merge_combiners
    )

# %%

def check_demo_group_top():
    import random
    import string

    num_elements = 100_0000
    keys = [random.choice(string.ascii_lowercase) for _ in range(num_elements)]
    values = [random.random() for _ in range(num_elements)]
    rdd = sc.parallelize(zip(keys, values))

    result = demo_group_top(rdd, 5).collectAsMap()
    for item in sorted(result.items()):
        print(item)

# %%

def demo_group_top(self: RDD, num: int, key: Callable = None) -> RDD:
    return self.groupByKey().mapValues(lambda it: heapq.nlargest(num, it, key=key))

# %%
