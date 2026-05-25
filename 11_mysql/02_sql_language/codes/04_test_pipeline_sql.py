
# %% PySpark RDD 示例代码

# 本地启动
from pyspark import SparkContext
from pyspark.sql import SparkSession

spark: SparkSession = SparkSession.builder.appName("learn-spark").master("local[*]").getOrCreate()
sc: SparkContext = spark._sc 

# docker 启动
# docker run -p 4040:4040 -v D:\softwares\docker\shares\learn_hive\main_shares:/shares -it spark:4.1.1-scala2.13-java21-python3-ubuntu /opt/spark/bin/pyspark

# 执行 SQL 语句
spark.sql("select 1").show()

# 
df = spark.read.csv(
    r'/shares/employees/dept_emp', 
    header=False, inferSchema=True
).toDF('emp_no', 'dept_no', 'from_date', 'to_date')

df.createOrReplaceTempView("dept_emp")

spark.sql("""\
select 
    num, count(1)
from (
    select emp_no, count(1) as num
    from dept_emp 
    group by emp_no    
) t
group by num;
""").show()

spark.sql("""\
from dept_emp
|> aggregate count(1) as n_count group by emp_no
|> aggregate count(1) as custdist group by n_count
|> order by custdist desc, n_count desc;
""").show()

# %%

df_ep = spark.read.csv(
    r'/shares/employees/employees_plus', 
    header=False, inferSchema=True
).toDF('emp_no', 'birth_date', 'first_name', 'last_name', 'gender', 'hire_date', 'dept_name', 'salary', 'title')

df_ep.createOrReplaceTempView("employees_plus")

df_d = spark.read.csv(
    r'/shares/employees/departments', 
    header=False, inferSchema=True
).toDF('dept_no', 'dept_name')

df_d.createOrReplaceTempView("departments")

spark.sql("""\
from departments d
|> select
    d.dept_no,
    d.dept_name,
    (
        from employees_plus ep
        |> where ep.dept_name = d.dept_name
        |> aggregate max(ep.salary)
    ) as max_salary
|> order by d.dept_no;
""").show()
