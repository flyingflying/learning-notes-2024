package mr_demo;

import java.io.IOException;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.partition.HashPartitioner;

// 可以从 ExampleDriver 中找到 MapReduce 编程模型的示例代码
// import org.apache.hadoop.examples.ExampleDriver;

public class WordCount {
    public static class TokenizerMapper extends Mapper<LongWritable, Text, Text, IntWritable>{

        private final static IntWritable one = new IntWritable(1);
        private Text word = new Text();

        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {

            StringTokenizer itr = new StringTokenizer(value.toString());
            while (itr.hasMoreTokens()) {
                word.set(itr.nextToken());
                context.write(word, one);
            }
        }

    }

    public static class IntSumReducer extends Reducer<Text, IntWritable, Text, IntWritable> {

        private IntWritable result = new IntWritable();

        @Override
        public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val: values) {
                sum += val.get();
            }
            result.set(sum);
            context.write(key, result);
        }
    }

    public static void main(String[] args) throws Exception {

        Job job = Job.getInstance(new Configuration(), "word count");
        job.setJarByClass(WordCount.class);

        /* 设置本地运行, 主要用于测试 */
        // 使用 local 模式启动, 而不是 yarn 模式启动
        job.getConfiguration().set("mapreduce.framework.name", "local");
        // 使用本地文件系统, 而不是 HDFS 文件系统
        job.getConfiguration().set("fs.defaultFS", "file:///");

        /* 设置文件输入类 */
        job.setInputFormatClass(TextInputFormat.class);

        /* 设置输入文件夹 */
        Path inputFilePath = new Path("./inputs/random_corpus/");
        FileInputFormat.addInputPath(job, inputFilePath);

        /* split (MapTask) 数量设置 */
        /* 在源码中, splitSize 的计算方式是: Math.max(minSize, Math.min(maxSize, blockSize)) */
        // 如果输入文件的 block_size 是 128 MB, 一个 block 对应两个 split, 设置方式如下:
        // FileInputFormat.setMaxInputSplitSize(job, 64 * 1024 * 1024);
        // 如果输入文件的 block_size 是 128 MB, 两个 block 对应一个 split, 设置方式如下:
        // FileInputFormat.setMinInputSplitSize(job, 256 * 1024 * 1024);

        /* 设置 Mapper 类 */
        job.setMapperClass(TokenizerMapper.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(IntWritable.class);

        /* 设置 Combiner 类 */
        job.setCombinerClass(IntSumReducer.class);

        /* 设置 Partitioner 类 */
        job.setPartitionerClass(HashPartitioner.class);

        /* 设置 partition (ReduceTask) 的数量, 默认值为 1 */
        job.setNumReduceTasks(2);

        /* 设置 Reducer 类 */
        job.setReducerClass(IntSumReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        /* 输出相关设置 */
        Path outputDirPath = new Path("./outputs/word_count/");
        // 如果输出文件夹存在, 则删除 (否则程序会报错)
        FileSystem fs = outputDirPath.getFileSystem(job.getConfiguration());
        if (fs.exists(outputDirPath)) {
            fs.delete(outputDirPath, true);
        }
        FileOutputFormat.setOutputPath(job, outputDirPath);
        // 设置输出的分隔符, 默认为 '\t'
        job.getConfiguration().set("mapred.textoutputformat.separator", "\t");


        /* 设置 Map 和 Reduce 运算的相关参数 */
        // MapTask 执行的内存限制, 默认为 1 GB
        job.getConfiguration().set("mapreduce.map.memory.mb", "1024");
        // MapTask 执行的 vcore 数量, 默认为 1, 需要 YARN 集群开启相关设置才有效
        job.getConfiguration().set("mapreduce.map.cpu.vcores", "1");
        // ReduceTask 执行的内存限制, 默认为 1 GB
        job.getConfiguration().set("mapreduce.reduce.memory.mb", "1024");
        // ReduceTask 执行的 vcore 数量, 默认为 1, 需要 YARN 集群开启相关设置才有效
        job.getConfiguration().set("mapreduce.reduce.cpu.vcores", "1");
        // 圆形缓冲区的大小, 默认 100 MB
        job.getConfiguration().set("mapreduce.task.io.sort.mb", "100");
        // 溢写触发阈值, 默认 80%
        job.getConfiguration().set("mapreduce.map.sort.spill.percent", "0.80");
        // MapTask 最多的 "溢写" 文件数, 超过了则 MapTask 执行失败, 默认为 -1, 表示无限制
        job.getConfiguration().set("mapreduce.task.spill.files.count.limit", "-1");
        // "溢写" 文件达到多少时, 会进行 combine 运算, 默认为 3
        job.getConfiguration().set("mapreduce.map.combine.minspills", "3");
        // 一次合并最多和几个文件, 默认值为 10
        job.getConfiguration().set("mapreduce.task.io.sort.factor", "10");
        // ReduceTask 最多同时抓取的文件数, 默认值为 5
        job.getConfiguration().set("mapreduce.reduce.shuffle.parallelcopies", "5");
        // 设置排序比较器
        job.setSortComparatorClass(Text.Comparator.class);
        // 设置分组比较器
        job.setGroupingComparatorClass(Text.Comparator.class);

        // 提交作业, 等待运行结束
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }

}
