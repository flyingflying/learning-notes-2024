
"""
select
    employees.*, 
    titles.title
from 
    employees.employees  
    inner join employees.titles on employees.emp_no = titles.emp_no
where 
    titles.to_date = '9999-01-01';
"""

# %%

import pandas as pd 

# from 从句
edf = pd.read_csv(
    r'D:\softwares\docker\shares\learn_hive\main_shares\employees\employees\00000.csv',
    names=['emp_no', 'birth_date', 'first_name', 'last_name', 'gender', 'hire_date']
)

tdf = pd.read_csv(
    r'D:\softwares\docker\shares\learn_hive\main_shares\employees\titles\00000.csv',
    names=["emp_no", "title", "from_date", "to_date", ]
)

# where 从句
tdf = tdf[tdf["to_date"] == "9999-01-01"]

# from -> join 从句
df = edf.set_index("emp_no").join(tdf.set_index("emp_no"), how="inner", validate="1:1")  # pandas 的 join 只能处理相等情况

# select 从句
df = df.reset_index(drop=False)[["emp_no", "first_name", "last_name", "gender", "hire_date", "title"]]

# 输出
df

# %%

from pyspark import SparkContext
from pyspark.sql import SparkSession

spark: SparkSession = SparkSession.builder.appName("learn-spark").master("local[*]").getOrCreate()
sc: SparkContext = spark._sc 

from pyspark.sql import Row
from operator import itemgetter

# from 从句
def preprocess_line(line: str) -> Row:
    entries = line.split(",")
    row = Row(
        emp_no=int(entries[0]), birth_date=entries[1], first_name=entries[2], 
        last_name=entries[3], gender=entries[4], hire_date=entries[5]
    )
    return row 

erdd = sc.textFile("/shares/employees/employees").map(preprocess_line)

def preprocess_line(line: str) -> Row:
    entries = line.split(",")
    row = Row(
        emp_no=int(entries[0]), title=entries[1], from_date=entries[2], to_date=entries[3]
    )
    return row 

trdd = sc.textFile("/shares/employees/titles").map(preprocess_line)

# where 从句
trdd = trdd.filter(lambda row: row["to_date"] == "9999-01-01")

# from 从句
erdd = erdd.keyBy(itemgetter("emp_no"))
trdd = trdd.keyBy(itemgetter("emp_no"))
rdd = erdd.join(trdd)

# select 从句
def map_func(rows: tuple[Row, Row]) -> Row:
    erow, trow = rows 
    return Row(
        emp_no=erow["emp_no"], birth_date=erow["birth_date"], first_name=erow["first_name"],
        last_name=erow["last_name"], gender=erow["gender"], hire_date=erow["hire_date"],
        title=trow["title"]
    )

rdd = rdd.values().map(map_func)

# 输出
for r in rdd.take(100):
    print(r)

# %%

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

@dataclass
class Title:
    emp_no: int 
    title: str 
    from_date: str 
    to_date: str 

    def __post_init__(self):
        self.emp_no = int(self.emp_no)
    
with open(r"D:\softwares\docker\shares\learn_hive\main_shares\employees\employees\00000.csv", "r", encoding="utf-8") as reader:
    employees = [Employee(*line.strip().split(",")) for line in reader]

with open(r"D:\softwares\docker\shares\learn_hive\main_shares\employees\titles\00000.csv", "r", encoding="utf-8") as reader:
    titles = [Title(*line.strip().split(",")) for line in reader]

# %%

# where 从句 (谓词下推)
titles = [title for title in titles if title.to_date == '9999-01-01']
results = []

for employee in employees:  # from 从句
    for title in titles:    # inner join 从句
        if employee.emp_no == title.emp_no:  # on 从句
            # select 从句
            result = employee.__dict__.copy()
            result["title"] = title.title
            results.append(result)

# 输出
for r in results:
    print(r)

# %%
