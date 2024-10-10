
# 单机版 Hadoop 部署

[TOC]

本文档是在 WSL 虚拟机中的 Ubuntu 系统下配置单机版单节点 Hadoop, 主要用于测试一些功能。

## 一、创建用户

对于大数据架构来说, 一般不会用 `root` 用户启动程序, 而是以 普通用户 启动程序。我们这里创建一个 `huser` 用户, 并配置 sudo 权限。具体的操作如下:

```shell
# ## 以 root 身份执行:
# 1. 创建新用户 huser
adduser huser
# 2. 将 huser 用户添加到 sudo 用户组中 (不建议直接修改 /etc/sudoers)
usermod -aG sudo huser
```

后续的操作都是在 `huser` 用户下执行的, 所有的文件都放在 `/home/huser` 下面。

## 二、设置 SSH

对于大数据平台来说, 不同计算机节点之间的免密 SSH 登录是基础要求。这里, 我们需要配置 localhost 自己到自己的免密登录 (虽然很奇怪, 但是是必须的)。

首先, 我们要确保 SSH 服务开启了。在 WSL 中使用 Ubuntu 属于直连, 不属于 SSH 登录。我们可以用 `ssh huser@localhost` 测试系统中的 SSH 是否开启。如果没有, 方式如下:

```shell
# ## reference: https://blog.csdn.net/q4616756/article/details/131842814 
# 1. 重装 openssh-server
sudo apt-get remove openssh-server
sudo apt-get install openssh-server

# 2. 在 /etc/ssh/sshd_config 文件中添加:
PermitRootLogin Yes
PasswordAuthentication Yes

# 3. 在 /etc/hosts.allow 文件中添加
sshd:ALL

# 4. 启动 ssh 服务
sudo service ssh --full-restart

# 5. 设置开机自启动
sudo systemctl enable ssh

# 5. 测试
ssh huser@localhost
```

免密登录的配置如下:

```shell
# 1. 生成密钥, 输入下面的指令, 然后连续敲三次回车即可
ssh-keygen -t rsa -b 4096

# 2. 拷贝密钥 (需要输入密码)
ssh-copy-id huser@localhost
```

## 三、安装 JDK

`apt` 安装方式如下:

```shell
# ## 1. 安装
sudo apt install openjdk-11-jdk

# ## 2. 验证是否安装成功
java --version
javac --version
which java

# ## 3. 编辑 ~/.bashrc 或者 /etc/bash.bashrc 文件, 添加下面的内容
# 注意: 千万不要编辑 /etc/profile 文件, 这样是无效的
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
```

自行安装的方式如下:

```shell
# ## 1. 从清华源下载安装 OpenJDK 安装包
# 清华源官网: https://mirrors.tuna.tsinghua.edu.cn/Adoptium/
# 这里使用的是 JDK 11, 不是 JDK 8, 一般情况下都选择 x64
wget -c https://mirrors.tuna.tsinghua.edu.cn/Adoptium/11/jdk/x64/linux/OpenJDK11U-jdk_x64_linux_hotspot_11.0.24_8.tar.gz -P downloads

# ## 2. 解压缩安装包
mkdir -p softwares
tar -zxvf downloads/OpenJDK11U-jdk_x64_linux_hotspot_11.0.24_8.tar.gz -C softwares

# ## 3. 创建软连接
ln -s softwares/jdk-11.0.24+8/ openjdk

# ## 4. 编辑 ~/.bashrc 或者 /etc/bash.bashrc 文件, 添加下面的内容
# 注意: 千万不要编辑 /etc/profile 文件, 这样是无效的
export JAVA_HOME=/home/huser/openjdk
export PATH=$PATH:$JAVA_HOME/bin

# ## 5. 验证安装是否成功
# 重新登陆 bash, 检测 java 和 javac 的版本号
java --version
javac --version
```

## 四、安装 Hadoop

```shell
# ## 1. 下载 hadoop 安装包
# 1.1 官网下载 (大陆估计是下载不下来)
wget -c https://dlcdn.apache.org/hadoop/common/hadoop-3.4.0/hadoop-3.4.0.tar.gz -P downloads
# 1.2 清华镜像源下载
# 清华源官网: https://mirrors.tuna.tsinghua.edu.cn/apache/hadoop/ 
wget -c https://mirrors.tuna.tsinghua.edu.cn/apache/hadoop/common/hadoop-3.4.0/hadoop-3.4.0.tar.gz -P downloads

# ## 2: 解压缩 hadoop 安装包
mkdir -p softwares
tar -zxvf downloads/hadoop-3.4.0.tar.gz -C softwares

# ## step3: 构建软连接
ln -s softwares/hadoop-3.4.0 hadoop

# ## step4: 添加 hadoop 运行环境
# 在 ~/.bashrc 中添加下面的内容
export HADOOP_HOME=/home/huser/hadoop
export PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin
```

在 hadoop 文件夹中, `etc` 文件夹表示的是配置文件夹, 其内部的主要文件有:

+ `workers`
+ `hadoop-env.sh`
+ `mapred-env.sh`
+ `yarn-env.sh`
+ `core-site.xml`
+ `hdfs-site.xml`
+ `mapred-site.xml`
+ `yarn-site.xml`

## 五、HDFS 部署

### 4.1 workers

`workers` 文件配置集群中的 "从节点", 这里的 "主节点" 和 "从节点" 都是 localhost, 我们将文件内容改成:

```shell
localhost
```

### 4.2 环境变量 配置

所有文件名以 `env` 结尾的都是 "环境变量" 配置文件。按照文档中的说法, `hadoop-env.sh` 配置对所有组件都生效, HDFS 应该有专属的 `hdfs-env.sh` 文件。但是, 我在 `etc` 文件夹下面没有找到, 那么就配置在 `hadoop-env.sh` 中吧, 问题不大。

```shell
# 编辑 ./etc/hadoop/hadoop-env.sh, 添加以下的内容:
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export HADOOP_HOME=/home/huser/hadoop
export HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop
export HADOOP_LOG_DIR=$HADOOP_HOME/logs
```

### 4.3 site 配置

所有文件名以 `site` 结尾的都是运行参数配置。这里我们需要修改两个文件: `core-site.xml` 和 `hdfs-site.xml`, 前者是 Hadoop 运行的整体配置文件, 后者是 HDFS 的运行的整体配置文件。

将 [core-site.xml](../etc/core-site.xml) 和 [hdfs-site.xml](../etc/hdfs-site.xml) 拷贝至 `./etc/hadoop/` 目录夹下面即可。

### 4.4 启动 HDFS 组件

```shell
# ## 1. 初始化 namenode
hdfs namenode -format

# ## 2. 启动
# 2.1 全启动
start-dfs.sh
# 2.2 单个启动
hdfs --daemon start namenode
hdfs --daemon start datanode
hdfs --daemon start secondarynamenode
# 一般首次运行都会使用 "全启动" 的方式, 后续添加节点才会使用 "单个启动" 的方式

# ## 3. 查看进程状态
# 3.1 查看所有的 Java 进程
jps
# 3.2 使用 hdfs 指令查看
hdfs --daemon status namenode
hdfs --daemon status datanode
hdfs --daemon status secondarynamenode

# ## 4. 关闭
# 4.1 全关闭
stop-dfs.sh
# 4.2 单个进程关闭
hdfs --daemon stop namenode
hdfs --daemon stop datanode
hdfs --daemon stop secondarynode
```

### 4.5 HDFS shell

上传和下载文件:

```shell
# ## 1. 上传文件
# 1.1 指令模板: hdfs dfs -put [-f] [local-path] [hdfs-path]
# 1.2 参数含义: -f 覆盖目标文件
# 1.3 示例:
truncate -s 256MB test-file  # 创建一个 256MB 的文件
hdfs dfs -put file:///home/huser/test-file hdfs://localhost:8020/  # 完整版写法
hdfs dfs -put test-file /  # 简略版写法

# ## 2. 下载文件
# 2.1 指令模板: hdfs dfs -get [-f] [hdfs-path] [local-path]
# 2.2 参数含义: -f 覆盖目标文件
# 2.3 示例:
hdfs dfs -get hdfs://localhost:8020/test-file file:///home/huser/test-file2  # 完整版写法
hdfs dfs -get /test-file test-file2  # 简略版写法

# ## 3. 文件存储个数
# 3.1 上传文件时, 指定文件的存储个数
hdfs fs -D dfs.replication=2 -put test-file /
# 3.2 修改文件的存储个数 
hdfs dfs -setrep 2 /test-file
# 3.3 查看文件状态: hdfs fsck [hdfs-path]
hdfs fsck /test-file
```

文件操作指令 和 Linux Shell 很相似, 有: `ls`, `cat`, `cp`, `mv`, `rm`, `mkdir`, `tail` 等等。

需要注意的是, HDFS 中的文件是不支持修改的, 因此没有类似 `vim` 的操作。但是可以在文件后面追加, 追加指令是 `appendToFile`, 使用方式如下:

```shell
# ## 指令模板: hdfs dfs -appendToFile [local-src-file] [hdfs-dst-file]

# 在本地创建两个文件
echo test-content-1 > test-file1.txt
echo test-content-2 > test-file2.txt

# 上传 test-file1.txt 文件, 追加 test-file2.txt 文件
hdfs dfs -put -f test-file1.txt /test-file.txt
hdfs dfs -appendToFile test-file2.txt /test-file.txt

# 查看文件
hdfs dfs -cat /test-file.txt
hdfs dfs -tail -f /test-file.txt
```

### 4.6 Web UI

我们可以在 `http://localhost:9870` 使用 web 界面进行管理。

## 五、YARN 部署

### 5.1 YARN 启动

```shell
# 全启动
start-yarn.sh

# 单个进程启动
# 模板: yarn --daemon [start | stop | status] [resourcemanager | nodemanager | proxyserver]
yarn --daemon start resourcemanager
yarn --daemon start nodemanager
yarn --daemon start proxyserver

# 全关闭
stop-yarn.sh

# 单个进程关闭
yarn --daemon stop resourcemanager
yarn --daemon stop nodemanager
yarn --daemon stop proxyserver
```

### 5.2 运行 MapReduce 程序

```shell
# 启动 MapReduce 的历史服务器
# 模板: mapred --daemon [start | stop | status] historyserver
mapred --daemon start historyserver

# 关闭历史服务器
mapred --daemon stop historyserver

# 运行 MapReduce 示例代码
hadoop jar $HADOOP_HOME/share/hadoop/mapreduce/hadoop-mapreduce-examples-3.4.0.jar pi 3 1000
```

### 5.3 WEB 页面查看信息

+ YARN Resource Manager 管理页面: `http://localhost:8088/`
+ YARN Node Manager 管理页面: `http://localhost:8042/`
+ MapReduce 历史服务器页面: `http://localhost:19888/`
+ HDFS NameNode 管理页面: `http://localhost:9870`
+ HDFS SecondaryNameNode 页面: `http://localhost:19888`

## 六、Hadoop 工具使用

### 6.1 Hadoop Archive

将 HDFS 集群中的多个文件 "打包" 成一个文件:

```shell
# format: hadoop archive -archiveName <file_name> -p <root-dir> [-r <replication>] <src>* <dest>
# 指令名称: hadoop archive 这是一个 MapReduce 程序, 需要先启动 YARN 集群
# 参数: -archiveName "打包" 后的文件名
# 参数: -p 所有 src 路径的根目录
# 参数: -r "打包" 后的文件在 HDFS 中存储几个副本
hadoop archive -archiveName test_dir.har -p /test-dir -r 1 test_dir_1 test_dir_2 /
# 将 /test-dir/test_dir_1 和 /test-dir/test_dir_2 路径下面的内容 "打包" 到 test_dir.har 文件中, 并存储 1 份
```
