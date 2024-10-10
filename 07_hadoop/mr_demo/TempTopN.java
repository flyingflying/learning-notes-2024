package mr_demo;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Iterator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

/**
 * 找出每个月温度最高的两天, 利用 MapReduce 中的 排序 机制实现的。
 * 
 * Reference: <a href="https://www.bilibili.com/video/BV1Kd4y1r7nQ?p=100">视频</a>
 */
public class TempTopN {

    // 任务描述: 找出每个月温度最高的两天
    public static void genInputFile(String inputFilePath) throws IOException{
        final int[] daysPerMonthArray = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
        int daysPerMonth, recordsPerDay = 0;

        // 文件对象
        File file = new File(inputFilePath + "temp_records.txt");
        if( !file.exists() ) {
            file.createNewFile();
        }
        // 文件写对象
        FileWriter writer = new FileWriter(file, false);

        // 遍历 1978 - 2024 所有的年份
        for(int curYear = 1978; curYear <= 2024; curYear++) {

            // 假设: 在 1978 年, 某地有 10 个气象监测站, 每个气象检测站每天会产生 1 条数据
            // 随后每一年气象监测站会增加 10 个, 逐年递增, 也就是说每天会多产生 10 条数据记录
            recordsPerDay += 10;

            // 遍历每一个月
            for(int curMonth = 1; curMonth <= 12; curMonth++) {

                // 计算一个月的天数
                daysPerMonth = daysPerMonthArray[curMonth - 1];
                if(curMonth == 2 && curYear % 4 == 0){
                    daysPerMonth = 29;
                }

                // 遍历每一天
                for(int curDay = 1; curDay <= daysPerMonth; curDay++) {

                    // 遍历生成每一个记录
                    // 一条记录的格式: '1978-01-01	015.35364' 日期和年份之间用 制表符 \t 分隔
                    for(int curRecord = 1; curRecord <= recordsPerDay; curRecord++) {
                        // 温度是随机生成的, 认为在 -20 到 40 之间
                        // double tempValue = Math.random() * (maxValue - minValue) + minValue;
                        double tempValue = Math.random() * 60 - 20;
                        // '%09.5f' 的含义: 
                        //     (1) 精确到小数点后五位; 
                        //     (2) 数字一共占九位, 小数点算一位, 小数点后确定是五位, 那么小数点之前有三位;
                        //     (3) 负号占一位, 也就是说: 如果有负号, 整数部分是两位, 如果没有负号, 整数部分是三位。
                        writer.write(String.format("%04d-%02d-%02d\t%09.5f\n", curYear, curMonth, curDay, tempValue));
                    }
                }
            }
        }

        writer.close();
    }

    public static class Temp implements WritableComparable<Temp>{
        private int year;
        private int month;
        private int day;
        private double tempValue;

        // public Temp(int year, int month, int day, double tempValue) {
        //     // 构造函数, 主要用于测试
        //     this.year = year;
        //     this.month = month;
        //     this.day = day;
        //     this.tempValue = tempValue;
        // }

        @Override
        public int compareTo(Temp other) {
            // 比较接口: 返回 sign(this - other)
            // 排序比较器 和 分组比较器 的规则不相同, 不应该在这里实现
            throw new UnsupportedOperationException();
        }

        @Override
        public void write(DataOutput out) throws IOException {
            // 序列化操作
            out.writeInt(this.year);
            out.writeInt(this.month);
            out.writeInt(this.day);
            out.writeDouble(this.tempValue);
        }

        @Override
        public void readFields(DataInput in) throws IOException {
            // 反序列化操作
            this.year = in.readInt();
            this.month = in.readInt();
            this.day = in.readInt();
            this.tempValue = in.readDouble();
        }

        public void setYear(int year) {this.year = year;}
        public void setMonth(int month) {this.month = month;}
        public void setDay(int day) {this.day = day;}
        public void setTempValue(double tempValue) {this.tempValue = tempValue;}
        public int getYear() {return year;}
        public int getMonth() {return month;}
        public int getDay() {return day;}
        public double getTempValue() {return tempValue;}

        @Override
        public String toString() {
            return String.format(
                "Temp: Date=%04d-%02d-%02d, Value=%09.5f", 
                this.year, this.month, this.day, this.tempValue
            );
        }
    }

    public static class TempSortComparator extends WritableComparator{
        public TempSortComparator() {
            super(Temp.class, true);
        }

        @Override
        @SuppressWarnings("rawtypes")
        public int compare(WritableComparable a, WritableComparable b) {
            // 排序比较器: 根据 年 和 月 从小到大 (顺序) 排序, 根据 温度 从大到小 (倒序) 排序
            Temp ta = (Temp) a;
            Temp tb = (Temp) b;
            int result = Integer.compare(ta.year, tb.year);
            if (result == 0) {result = Integer.compare(ta.month, tb.month);}
            if (result == 0) {result = Double.compare(tb.tempValue, ta.tempValue);}
            return result;
        }
    }

    public static class TempGroupingComparator extends WritableComparator{
        public TempGroupingComparator() {
            super(Temp.class, true);
        }

        @Override
        @SuppressWarnings("rawtypes")
        public int compare(WritableComparable a, WritableComparable b) {
            // 分组比较器: 根据 年 和 月 从小到大 (顺序) 排序
            Temp ta = (Temp) a;
            Temp tb = (Temp) b;
            int result = Integer.compare(ta.year, tb.year);
            if (result == 0) {result = Integer.compare(ta.month, tb.month);}
            return result;
        }
    }

    public static class TempPartitioner extends Partitioner<Temp, Text>{

        @Override
        public int getPartition(Temp key, Text value, int numPartitions) {
            // 数据是逐年递增的, 那么按照 年 进行划分就不合适了, 会产生数据倾斜
            // 可以按照 月份 划分, 虽然每个月会相差几天, 但是问题不大
            return key.getMonth() % numPartitions;  // 分区器: 根据月份进行分区
        }

    }

    public static class TempMapper extends Mapper<LongWritable, Text, Temp, Text> {
        private Temp tempOut = new Temp();
        private SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd");

        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString().strip();
            String[] parts = line.split("\t");
            if (parts.length != 2) {
                return;  // 解析错误直接跳过
            }

            // 处理时间
            try {
                Calendar calendar = Calendar.getInstance();
                calendar.setTime(this.sdf.parse(parts[0]));
                tempOut.setYear(calendar.get(Calendar.YEAR));
                tempOut.setMonth(calendar.get(Calendar.MONTH) + 1);
                tempOut.setDay(calendar.get(Calendar.DAY_OF_MONTH));
            } catch (ParseException e) {
                return;  // 解析错误直接跳过
            }

            // 处理温度
            this.tempOut.setTempValue(Double.parseDouble(parts[1]));

            context.write(tempOut, value);
        }
    }

    public static class TempReducer extends Reducer<Temp, Text, Text, NullWritable>{
        @Override
        protected void reduce(Temp key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            Iterator<Text> iter = values.iterator();
            int maxDay = -1;

            if (iter.hasNext()) {
                Text value = iter.next();
                maxDay = key.getDay();
                context.write(value, NullWritable.get());
            }

            // iter 迭代器虽然是 values 迭代器, 但是在调用 next 方法时, 会改变 key 值
            // 原理很简单, 我们在迭代 value 时, 调用 key 对象的 set 方法。也就是说, 对于 key 而言, 对象没有改变, 但是对象内的属性值改变了。
            while(iter.hasNext()) {
                Text value = iter.next();
                if (maxDay != key.getDay()) {
                    context.write(value, NullWritable.get());
                    break;
                }
            }
        }
    }

    public static class TempCombiner extends Reducer<Temp, Text, Temp, Text> {
        @Override
        protected void reduce(Temp key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            Iterator<Text> iter = values.iterator();
            int maxDay = -1;

            if (iter.hasNext()) {
                Text value = iter.next();
                maxDay = key.getDay();
                context.write(key, value);
            }

            while(iter.hasNext()) {
                Text value = iter.next();
                if (maxDay != key.getDay()) {
                    context.write(key, value);
                    break;
                }
            }
        }
    }

    public static void main(String[] args) throws Exception{
        final String inputDirPath = "./inputs/temperature/";
        final String outputDirPath = "./outputs/temperature/";

        Configuration conf = new Configuration(true);
        Job job = Job.getInstance(conf, "Temperature");
        job.setJarByClass(TempTopN.class);

        // 生成输入数据
        File inFile = new File(inputDirPath);
        if( !inFile.exists() ) {
            inFile.mkdirs();
            genInputFile(inputDirPath);
        }

        // 删除输出文件夹
        File outFile = new File(outputDirPath);
        if(outFile.exists()) {
            for (File file : outFile.listFiles()) {
                file.delete();
            }
            outFile.delete();
        }

        // 设置本地运行的参数
        job.getConfiguration().set("mapreduce.framework.name", "local");  // 使用 local 模式启动, 而不是 yarn 模式启动
        job.getConfiguration().set("fs.defaultFS", "file:///");  // 使用本地文件系统, 而不是 HDFS 文件系统

        // 设置输入输出路径
        FileInputFormat.addInputPath(job, new Path(inputDirPath));
        FileOutputFormat.setOutputPath(job, new Path(outputDirPath));

        // 设置 MapTask 参数
        job.setMapperClass(TempMapper.class);
        job.setMapOutputKeyClass(Temp.class);
        job.setMapOutputValueClass(Text.class);

        // 设置 ReduceTask 参数
        job.setReducerClass(TempReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(NullWritable.class);
        job.setNumReduceTasks(6);

        // 其它设置
        job.setSortComparatorClass(TempSortComparator.class);
        job.setGroupingComparatorClass(TempGroupingComparator.class);
        job.setPartitionerClass(TempPartitioner.class);
        job.setCombinerClass(TempCombiner.class);

        // 运行
        job.waitForCompletion(true);

    }
}
