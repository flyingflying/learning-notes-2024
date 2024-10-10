
# MapReduce

## Local 模式运行

在 main 函数中添加下面的代码, 就可以直接运行程序了, 不需要通过 `hadoop jar` 指令。代码如下:

```java
Configuration conf = new Configuration();

// 设置本地运行的参数
conf.set("mapreduce.framework.name", "local");
conf.set("fs.defaultFS", "file:///");
```

在 VSCode 中安装 "Extension Pack for Java" 组件, 然后将 `$HADOOP_HOME/share` 文件夹中的所有 `jar` 包作为依赖, 方式如下:

在 `.vscode/settings.json` 中添加下面的内容:

```json
{
    "java.project.referencedLibraries": [
        "lib/**/*.jar",
        "/home/huser/hadoop/share/hadoop/**/*.jar"
    ],
    "java.jdt.ls.vmargs": "-XX:+UseParallelGC -XX:GCTimeRatio=4 -XX:AdaptiveSizePolicyWeight=90 -Dsun.zip.disableMemoryMapping=true -Xmx2G -Xms100m -Xlog:disable"
}
```

其中, `java.project.referencedLibraries` 参数配置依赖包的路径, `lib/**/*.jar` 是 Java 基础库, `/home/huser/hadoop/share/hadoop/**/*.jar` 是 hadoop 依赖库。

`java.jdt.ls.vmargs` 是用来解除限制的, 不然插件会报解析错误, 没有办法进行代码提示。

```shell
javac @argfile mr_demo/*.java
java @argfile mr_demo.WordCount
rm -rf mr_demo/*.class
```
