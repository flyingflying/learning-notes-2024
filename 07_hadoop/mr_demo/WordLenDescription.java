package mr_demo;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Iterator;
import java.util.StringTokenizer;
import java.util.TreeMap;
import java.util.Map.Entry;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.TaskCounter;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

/**
 * 输出语料库中词语长度的统计量, 包括 平均数, 中位数, 众数, 方差, 标准差, 极值 等等。
 * 实现参考 {@link org.apache.hadoop.examples.WordMedian}
 * 使用 MapReduce 框架统计词语长度频数, 然后读取 HDFS 文件进行进一步的计算。
 */
public class WordLenDescription {

    private static Job job = null;
    private static Path inputDirPath = new Path("./inputs/random_corpus/");
    private static Path outputDirPath = new Path("./outputs/word_length_count/");

    public static class WordLenCountMapper extends Mapper<Object, Text, IntWritable, IntWritable> {

        private IntWritable length = new IntWritable();
        private final static IntWritable ONE = new IntWritable(1);

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            /* 分词, 按照词语长度输出频数 (1) */
            StringTokenizer itr = new StringTokenizer(value.toString());
            while (itr.hasMoreTokens()) {
                String string = itr.nextToken();
                this.length.set(string.length());
                context.write(this.length, ONE);
            }
        }
    }

    public static class WordLenCountReducer extends Reducer<IntWritable, IntWritable, IntWritable, IntWritable> {

        private IntWritable val = new IntWritable();

        public void reduce(IntWritable key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            /* 将不同长度的频数相加 */
            int sum = 0;
            for (IntWritable value : values) {
                sum += value.get();
            }
            val.set(sum);
            context.write(key, val);
        }
    }

    private static void describe() throws Exception {

        // 判定输出文件是否合法
        Configuration conf = job.getConfiguration();
        FileSystem fs = FileSystem.get(conf);

        if (!fs.exists(outputDirPath)) {
            throw new IOException("MapReduce 作业未执行完成");
        } else if(!fs.exists(new Path(outputDirPath, "_SUCCESS"))) {
            throw new IOException("MapReduce 作业执行失败");
        } else if(fs.exists(new Path(outputDirPath, "part-r-00001"))) {
            throw new IOException("MapReduce 作业只能有一个 ReduceTask");
        }
        Path outputFilePath = new Path(outputDirPath, "part-r-00000");
        if (!fs.exists(outputFilePath)){
            throw new IOException("MapReduce 作业输出文件未找到");
        }

        // 读取文件, 构建长度词频字典
        BufferedReader br = null;
        TreeMap<Integer, Long> lengthCountDict = new TreeMap<>();

        try {
            br = new BufferedReader(new InputStreamReader(fs.open(outputFilePath), StandardCharsets.UTF_8));
            String line = null;

            while (true) {
                line = br.readLine();
                if (line == null) {
                    break;
                }
                String[] tokens = line.strip().split("\t");
                if (tokens.length != 2) {
                    continue;
                }
                lengthCountDict.put(
                    Integer.parseInt(tokens[0]), 
                    Long.parseLong(tokens[1])
                );
            }
        } finally {
            if (br != null) {
                br.close();
            }
        }

        // 从 MapReduce 框架的 计数器 (Counter) 获取总词数
        // reference: https://segmentfault.com/a/1190000040898528 
        long totalNumWords2 = job.getCounters()
            .getGroup(TaskCounter.class.getCanonicalName())
            .findCounter("MAP_OUTPUT_RECORDS", "Map output records").getValue();
        
        long totalNumWords = lengthCountDict.values().stream().mapToLong(s -> s).sum();

        if (totalNumWords != totalNumWords2) {
            throw new InterruptedException("逻辑错误");
        }
        System.out.println(lengthCountDict);

        // 求平均数
        long totalLength = lengthCountDict.entrySet().stream().mapToLong(entry -> entry.getKey() * entry.getValue()).sum();
        double meanLength = (double) totalLength / (double) totalNumWords;
        System.out.println("The average number of words' length is " + String.format("%.3f", meanLength));

        // 求中位数
        double halfNumWords = totalNumWords / 2.0;
        double accNumWords = 0.0;
        double medianLength = 0.0;
        Iterator<Entry<Integer, Long>> iterator = lengthCountDict.entrySet().iterator();
        while (iterator.hasNext()) {
            Entry<Integer, Long> entry = iterator.next();
            accNumWords += entry.getValue();
            if (accNumWords > halfNumWords) {
                medianLength = entry.getKey();
                break;
            }
            if (accNumWords == halfNumWords) {
                Entry<Integer, Long> nextEntry = iterator.next();
                medianLength = (entry.getKey() + nextEntry.getKey()) / 2.0;
            }
        }
        System.out.println("The median number of words' length is " + String.format("%.3f", medianLength));

        // 求众数
        long maxLength = lengthCountDict.values().stream().mapToLong(s -> s).max().orElse(-1);
        List<Integer> modeLength = new ArrayList<>();
        for(Entry<Integer, Long> entry: lengthCountDict.entrySet()) {
            if (entry.getValue() == maxLength) {
                modeLength.add(entry.getKey());
            }
        }
        System.out.println("The mode numbers of word's length are " + modeLength);

        // 求方差
        double sumDeviationSquare = lengthCountDict.entrySet().stream().mapToDouble(
            entry -> Math.pow(entry.getKey() - meanLength, 2) * entry.getValue()
        ).sum();
        double variance = sumDeviationSquare / totalNumWords;
        System.out.println("The variance number of words' length is " + String.format("%.3f", variance));

        // 求标准差
        double stdVariance = Math.sqrt(variance);
        System.out.println("The standard variance number of words' length is " + String.format("%.3f", stdVariance));

        // 求极差
        double range = lengthCountDict.lastKey() - lengthCountDict.firstKey();
        System.out.println("The range number of words' length is " + range);
    }

    public static void main(String[] args) throws Exception {
        job = Job.getInstance(new Configuration(), "word length description");

        job.getConfiguration().set("mapreduce.framework.name", "local");
        job.getConfiguration().set("fs.defaultFS", "file:///");

        job.setJarByClass(WordLenDescription.class);
        job.setMapperClass(WordLenCountMapper.class);
        job.setCombinerClass(WordLenCountReducer.class);
        job.setReducerClass(WordLenCountReducer.class);
        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(IntWritable.class);
        job.setNumReduceTasks(1);

        // 设置输入文件夹
        FileInputFormat.addInputPath(job, inputDirPath);

        // 设置输出文件夹
        FileSystem fs = outputDirPath.getFileSystem(job.getConfiguration());
        if (fs.exists(outputDirPath)) {
            fs.delete(outputDirPath, true);
        }
        FileOutputFormat.setOutputPath(job, outputDirPath);

        // 执行 MapReduce 任务
        if (!job.waitForCompletion(false)){System.exit(-1);}

        describe();
    }
}
