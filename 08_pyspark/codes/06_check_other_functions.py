
# %%

from typing import *
from pyspark.rdd import portable_hash, RDD, Partitioner
from pyspark import SparkContext, SparkConf

conf = SparkConf()
conf.setMaster("local[*]")
conf.setAppName("test_rdd")
# conf.set("spark.cleaner.referenceTracking.cleanCheckpoints", "True")
conf.set("spark.ui.enabled", "True")
sc = SparkContext(conf=conf)

# %% 测试非局部变量

def count(rdd: RDD, debug: bool = False):
    num = 0

    def map_func(iterator):
        nonlocal num 
        for _ in iterator:
            num += 1
        yield num  

    if debug:
        def map_func_v2(iterator):
            nonlocal num 
            for _ in iterator:
                num += 1
            yield num  
        counts = rdd.mapPartitions(map_func).mapPartitions(map_func_v2).collect()
        return counts

    counts = rdd.mapPartitions(map_func).collect()
    return sum(counts)



# %% 测试累加器

def count(rdd: RDD):
    num = sc.accumulator(0)

    def map_func(_):
        nonlocal num 
        num += 1

    rdd.foreach(map_func)
    return num.value

# %% 测试多线程并发

sc.setJobGroup

from pyspark import InheritableThread, inheritable_thread_target

@inheritable_thread_target
def test_thread():
    rdd = sc.parallelize([1, 2, 3, 4, 5, 6, 7], 3).map(lambda n: n * 2)
    print(rdd.collect())

threads: list[InheritableThread] = []

for _ in range(5):
    threads.append(InheritableThread(target=test_thread))

for thread in threads:
    thread.start()

for thread in threads:
    thread.join()

# %% 测试 checkpoint

sc.setCheckpointDir("/home/lqxu/tmp-data/")  # 数据需要手动删除
rdd = sc.range(100)
rdd.checkpoint()
rdd.count()
print(rdd.getCheckpointFile())

# %% 测试 profiler

# reference: https://data-flair.training/blogs/pyspark-profiler/ 

from typing import *
from pyspark.rdd import RDD
from pyspark import SparkContext, SparkConf
# from pyspark.profiler import MemoryProfiler

conf = SparkConf()
conf.setMaster("local[*]")
conf.setAppName("test_rdd")
conf.set("spark.ui.enabled", "True")
conf.set("spark.python.profile", "true")  # important
sc = SparkContext(conf=conf)

sc.range(5, 100_0000, 2).collect()

sc.show_profiles()

# %% 
