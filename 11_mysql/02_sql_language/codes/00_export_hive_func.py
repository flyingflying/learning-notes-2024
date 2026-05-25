
# %%

# pip install pyhive, thrift, thrift_sasl -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple 

# pip install pandas, openpyxl -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple 

# %%

from pyhive import hive 

conn = hive.Connection(host="127.0.0.1", port=10000, database="employees")
cursor = conn.cursor()

cursor.execute("show functions")
funcs = [row[0] for row in cursor.fetchall()]

descriptions = []
for func in funcs:
    try:
        cursor.execute(f"describe function extended '{func}'")
        results = [row[0] for row in cursor.fetchall()]
        results = ["" if result is None else result for result in results]
        descriptions.append("\n".join(results))
    except Exception:
        print(func)
        break

cursor.close()
conn.close()

# %%

import pandas as pd 

df = pd.DataFrame({"func_name": funcs, "func_desc": descriptions})

df.to_excel("hive_funcs.xlsx")
