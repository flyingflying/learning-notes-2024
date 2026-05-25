
"""
select 
    dept_name,
    count(1) as num_employees,
    avg(salary) as avg_salary
from 
    employees.employees_plus
where 
    gender = 'M'
group by 
    dept_name 
having 
    num_employees > 9000
order by 
    avg_salary desc
limit 
    10 offset 0;
"""

# %% 普通示例代码

from datetime import datetime
from operator import itemgetter
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class EmployeePlus:
    emp_no: int 
    birth_date: str 
    first_name: str 
    last_name: str 
    gender: str 
    hire_date: str 
    dept_name: str 
    salary: int 
    title: str 

    def __post_init__(self):
        self.emp_no = int(self.emp_no)
        self.birth_date = datetime.strptime(self.birth_date, '%Y-%m-%d')
        self.hire_date = datetime.strptime(self.hire_date, '%Y-%m-%d')
        self.salary = int(self.salary)

with open(r"D:\softwares\docker\shares\learn_hive\main_shares\employees\employees_plus\00002.csv", "r", encoding="utf-8") as reader:
    employees = [EmployeePlus(*line.strip().split(",")) for line in reader]

pre_results = defaultdict(lambda: defaultdict(int))

for employee in employees:  # from 从句
    if employee.gender == 'M':  # where 从句
        # group by 从句 + select 预处理
        pre_result = pre_results[employee.dept_name]
        pre_result["num_employees"] += 1
        pre_result["total_salary"] += employee.salary

results = []

for dept_name, pre_result in pre_results.items():
    if pre_result["num_employees"] > 9000:  # having 从句
        results.append({  # select 后处理
            "dept_name": dept_name,
            "num_employees": pre_result["num_employees"],
            "avg_salary": pre_result["total_salary"] / pre_result["num_employees"]
        })

results = sorted(results, key=itemgetter("avg_salary"), reverse=True)  # sort by 从句

results = results[0:10]  # limit 从句

# 输出
for r in results:
    print(r)

# %% Pandas 示例代码

import pandas as pd 

# 1. from 从句
df = pd.read_csv(
    r'D:\softwares\docker\shares\learn_hive\main_shares\employees\employees_plus\00002.csv',
    names=['emp_no', 'birth_date', 'first_name', 'last_name', 'gender', 'hire_date', 'dept_name', 'salary', 'title']
)

# 2. where 从句
df = df[ df["gender"] == "M" ]

# 3. group by 从句
gb = df.groupby(["dept_name"])

# 4. select 从句
df = pd.DataFrame({
    "num_employees": gb["emp_no"].count(),
    "avg_salary": gb["salary"].mean(), 
}).reset_index(drop=False)

# 5. having 从句
df = df[ df["num_employees"] > 9000 ]

# 6. order by 从句
df.sort_values(["avg_salary"], ascending=False, inplace=True)

# 7. limit 从句
df = df.iloc[0:10]

# 8. 输出
df 

# %% PySpark RDD 示例代码

# 本地启动
from pyspark import SparkContext
from pyspark.sql import SparkSession

spark: SparkSession = SparkSession.builder.appName("learn-spark").master("local[*]").getOrCreate()
sc: SparkContext = spark._sc 

# docker 启动
# docker run -p 4040:4040 -v D:\softwares\docker\shares\learn_hive\main_shares:/shares -it spark:4.1.1-scala2.13-java21-python3-ubuntu /opt/spark/bin/pyspark
from pyspark.sql import Row
from operator import itemgetter

# 1. from 从句
def preprocess_line(line: str) -> Row:
    entries = line.split(",")
    row = Row(
        emp_no=int(entries[0]), birth_date=entries[1], first_name=entries[2], 
        last_name=entries[3], gender=entries[4], hire_date=entries[5],
        dept_name=entries[6], salary=int(entries[7]), title=entries[8]
    )
    return row 

rdd = sc.textFile("/shares/employees/employees_plus").map(preprocess_line)

# 2. where 从句
rdd = rdd.filter(lambda row: row.gender == "M")

# 3. group by + select 从句
rdd = rdd.map(lambda row: (row.dept_name, row))
rdd = rdd.combineByKey(
    createCombiner=lambda e: Row(num_employees=1, total_salary=e.salary),
    mergeValue=lambda c, e: Row(num_employees=c.num_employees + 1, total_salary=c.total_salary + e.salary),
    mergeCombiners=lambda c1, c2: Row(num_employees=c1.num_employees + c2.num_employees, total_salary=c1.total_salary + c2.total_salary),
)
rdd = rdd.map(lambda kv: Row(dept_name=kv[0], num_employees=kv[1].num_employees, avg_salary=kv[1].total_salary / kv[1].num_employees))

# 4. having 从句
rdd = rdd.filter(lambda row: row.num_employees > 9000)

# 5. order by 从句
rdd.persist()
rdd = rdd.sortBy(itemgetter("avg_salary"), ascending=False)

# 6. limit 从句
rdd = rdd.take(10)[0:10]

# 7. 输出
for r in results:
    print(r)
