package mr_demo;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

/**
 * 输出语料库中词语长度的统计量, 只实现了 平均数, 和 标准差。
 * 实现参考 {@link org.apache.hadoop.examples.WordMean} 和 {@link org.apache.hadoop.examples.WordStandardDeviation}
 * 实现公式: 
 *      mean = totalLengths / totalNumbers
 *      variance = E(X^2) - E(X)^2
 * 使用 MapReduce 框架计算 totalLengths, totalNumbers 和 totalSquareLengths, 然后读取文件进行计算
 */
public class WordLenDescriptionV2 {
    private final static Text LENGTH = new Text("length");  // 用于统计词语长度之和 (key)
    private final static Text SQUARE = new Text("square");  // 用于统计词语长度的平方之和 (key)
    private final static Text COUNT = new Text("count");    // 用于统计语料库中词语的总频数 (key)
    private final static LongWritable ONE = new LongWritable(1);  // 用于统计语料库中词语的总频数 (value)

    private static Job job = null;
    private final static Path inputDirPath = new Path("./inputs/random_corpus/");
    private final static Path outputDirPath = new Path("./outputs/word_length_description_v2/");

    public static class WordLenMapper extends Mapper<Object, Text, Text, LongWritable> {

        private LongWritable wordLen = new LongWritable();    // 用于统计词语长度之和 (value)
        private LongWritable wordLenSq = new LongWritable();  // 用于统计词语长度的平方之和 (value)

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            StringTokenizer itr = new StringTokenizer(value.toString());
            while (itr.hasMoreTokens()) {
                String string = itr.nextToken();
                this.wordLen.set(string.length());
                this.wordLenSq.set((long) Math.pow(string.length(), 2.0));  // 整数的平方, 一定还是整数
                context.write(LENGTH, this.wordLen);    // 词语长度
                context.write(SQUARE, this.wordLenSq);  // 词语长度平方
                context.write(COUNT, ONE);              // 词语长度频数
            }
        }
    }

    public static class WordLenReducer extends Reducer<Text, LongWritable, Text, LongWritable> {

        private LongWritable val = new LongWritable();

        public void reduce(Text key, Iterable<LongWritable> values, Context context) throws IOException, InterruptedException {
            int sum = 0;
            for (LongWritable value: values) {
                sum += value.get();
            }
            this.val.set(sum);
            context.write(key, this.val);
        }
    }

    private static void describe() throws Exception {

        if (job == null) {
            throw new InterruptedException("Please run job first!");
        }
        FileSystem fs = outputDirPath.getFileSystem(job.getConfiguration());
        Path file = new Path(outputDirPath, "part-r-00000");

        if (!fs.exists(file)){
            throw new IOException("Output not found!");
        }

        double stddev = 0;
        long count = 0, length = 0, square = 0;
        BufferedReader br = null;
        try {
            br = new BufferedReader(new InputStreamReader(fs.open(file), StandardCharsets.UTF_8));
            String line;
            while ((line = br.readLine()) != null) {
                StringTokenizer st = new StringTokenizer(line);
                String type = st.nextToken();  // grab type

                // differentiate
                if (type.equals(COUNT.toString())) {
                    String countLit = st.nextToken();
                    count = Long.parseLong(countLit);
                } else if (type.equals(LENGTH.toString())) {
                    String lengthLit = st.nextToken();
                    length = Long.parseLong(lengthLit);
                } else if (type.equals(SQUARE.toString())) {
                    String squareLit = st.nextToken();
                    square = Long.parseLong(squareLit);
                }
            }
        } finally {
            if (br != null) {
                br.close();
            }
        }

        double mean = (((double) length) / ((double) count));
        System.out.println("The mean of words' length in corpus is " + String.format("%.3f", mean));

        mean = Math.pow(mean, 2.0);
        double term = (((double) square / ((double) count)));
        stddev = Math.sqrt((term - mean));

        System.out.println("The standard deviation of words' length in corpus is " + String.format("%.3f", stddev));
    }

    public static void main(String[] args) throws Exception {

        job = Job.getInstance(new Configuration(), "word stddev");

        job.getConfiguration().set("mapreduce.framework.name", "local");
        job.getConfiguration().set("fs.defaultFS", "file:///");

        job.setJarByClass(WordLenDescriptionV2.class);
        job.setMapperClass(WordLenMapper.class);
        job.setCombinerClass(WordLenReducer.class);
        job.setReducerClass(WordLenReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(LongWritable.class);
        job.setNumReduceTasks(1);

        // 设置输入文件夹
        FileInputFormat.addInputPath(job, inputDirPath);

        // 设置输出文件夹
        FileSystem fs = outputDirPath.getFileSystem(job.getConfiguration());
        if (fs.exists(outputDirPath)) {
            fs.delete(outputDirPath, true);
        }
        FileOutputFormat.setOutputPath(job, outputDirPath);

        if (!job.waitForCompletion(true)){
            System.exit(-1);
        }

        describe();
  }
}
