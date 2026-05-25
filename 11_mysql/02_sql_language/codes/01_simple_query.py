
# 简单查询 代码对应

"""
select 
    emp_no as eid,
    concat_ws(" ", first_name, last_name) as name,
    case gender when 'M' then '男' when 'F' then '女' else '未知' end as gender,
    datediff(current_date(), birth_date) div 365.25 as age,
    hire_date
from 
    employees
where 
    hire_date between '1990-01-01' and '1990-12-31'
order by 
    age desc, emp_no asc
limit 
    10 offset 50;
"""

# %% 普通示例代码

from datetime import datetime
from operator import itemgetter
from dataclasses import dataclass

@dataclass
class Employee:
    emp_no: int 
    birth_date: str 
    first_name: str 
    last_name: str 
    gender: str 
    hire_date: str 

    def __post_init__(self):
        self.emp_no = int(self.emp_no)
        # self.birth_date = datetime.strptime(self.birth_date, '%Y-%m-%d')
        # self.hire_date = datetime.strptime(self.hire_date, '%Y-%m-%d')

with open(r"D:\softwares\docker\shares\learn_hive\main_shares\employees\employees\00000.csv", "r", encoding="utf-8") as reader:
    employees = [Employee(*line.strip().split(",")) for line in reader]

results = []

for employee in employees:  # from 从句
    if '1990-01-01' <= employee.hire_date <= '1990-12-31':  # where 从句
        results.append({  # select 从句
            "eid": employee.emp_no,
            "name": employee.first_name + " " + employee.last_name, 
            "gender": "男" if employee.gender == "M" else "女",
            "age": (datetime.today() - datetime.strptime(employee.birth_date, '%Y-%m-%d')).days // 365.25,
            "hire_date": employee.hire_date, 
        })

# sort by 从句
# Python 内部采用的是 稳定排序, 多字段排序可以拆分成多次排序, 从后往前进行
results = sorted(results, key=itemgetter("eid"), reverse=False)
results = sorted(results, key=itemgetter("age"), reverse=True)

# limit 从句
results = results[50:60]

# 输出
for r in results:
    print(r)

# %% Pandas 示例代码

import pandas as pd 

# 1. from 从句
df = pd.read_csv(
    r'D:\softwares\docker\shares\learn_hive\main_shares\employees\employees\00000.csv',
    names=['emp_no', 'birth_date', 'first_name', 'last_name', 'gender', 'hire_date']
)

# 2. where 从句
df = df[ (df['hire_date'] >= '1990-01-01') & (df['hire_date'] <= '1990-12-31') ]

# 3. select 从句
df = pd.DataFrame({
    "eid": df["emp_no"],
    "name": df["first_name"].str.cat(df["last_name"], sep=" "), 
    "gender": df["gender"].apply(lambda x: '男' if x == 'M' else '女'),  
    "age": (pd.Timestamp.today() - pd.to_datetime(df["birth_date"])).dt.days // 365.25,  
    "hire_date": df["hire_date"]
})

# 4. order by 从句
df = df.sort_values(by=["age", "eid"], ascending=[False, True])

# 5. limit 从句
df = df.iloc[50:60]

# 输出
df 

# %% PySpark RDD 示例代码

# 本地启动
from pyspark import SparkContext
from pyspark.sql import SparkSession

spark: SparkSession = SparkSession.builder.appName("learn-spark").master("local[*]").getOrCreate()
sc: SparkContext = spark._sc 

# docker 启动
# docker run -p 4040:4040 -v D:\softwares\docker\shares\learn_hive\main_shares:/shares -it spark:4.1.1-scala2.13-java21-python3-ubuntu /opt/spark/bin/pyspark
from datetime import datetime
from functools import cmp_to_key
from pyspark.sql import Row
# from pyspark.sql.types import StructType, StringType, StructField, Row, IntegerType

# 1. from 从句
def preprocess_line(line: str) -> Row:
    entries = line.split(",")
    row = Row(
        emp_no=int(entries[0]), birth_date=entries[1], first_name=entries[2], 
        last_name=entries[3], gender=entries[4], hire_date=entries[5]
    )
    return row 

rdd = sc.textFile("/shares/employees/employees").map(preprocess_line)

# col_names = ['birth_date', 'first_name', 'last_name', 'gender', 'hire_date']
# schema = [StructField("emp_no", IntegerType(), nullable=True)]
# schema.extend([StructField(col_name, StringType(), nullable=True) for col_name in col_names])
# rdd = spark.read.csv("/shares/employees/employees", schema=StructType(schema)).rdd

# where 从句
rdd = rdd.filter(lambda row: '1990-01-01' <= row.hire_date <= '1990-12-31')

# select 从句
def map_func(row: Row) -> Row:
    return Row(
        eid=row.emp_no, 
        name=row.first_name + " " + row.last_name,
        gender="男" if row.gender == "M" else "女",
        age=(datetime.today() - datetime.strptime(row.birth_date, '%Y-%m-%d')).days // 365.25, 
        hire_date=row.hire_date, 
    )

rdd = rdd.map(map_func)

# order by 从句
def compare_to(row1: Row, row2: Row) -> int:
    # < 0: row1 < row2; = 0: row1 = row2; > 0: row1 > row2
    if row1.age != row2.age:
        return -1 if row1.age > row2.age else 1  # 降序排列
    
    if row1.eid != row2.eid:
        return -1 if row1.eid < row2.eid else 1              # 升序排列
    
    return 0

# 多字段排序可以分成两阶段排序: 首先用 sortBy 方法进行预排序, 然后再每一个分区内部进行精准排序
rdd = rdd.sortBy(keyfunc=lambda row: row.age, ascending=False)  # 相同 age 一定在同一分区内
rdd = rdd.mapPartitions(lambda rows: iter(sorted(rows, key=cmp_to_key(compare_to))))

# limit 从句
results = rdd.take(60)[-10:]  # take: 行动算子

# 输出
for r in results:
    print(r)
