package mr_demo;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.util.HashMap;

import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

/**
 * 接 {@link JoinReduce} 作业的需求。
 * 在实际的生产过程中, 订单表 的记录量可能会非常大, 一个产品有上亿个订单。相对应的, 产品表就是一个 字典表, 非常地小, 假设有几千个产品。
 * 此时, 如果使用 JoinReduce, 一个 ReduceTask 需要处理上亿条 key & value 记录。
 * 同时, 这里不能使用 Combiner 减轻 ReduceTask 的任务量, 因为 Reduce 阶段输出的 "记录" 数量和 订单表 是一致的。
 * 那么, 应该怎么办呢? 我们可以去掉 reduce 阶段, 在 map 阶段直接读取 产品表 (小表) 的数据, 然后直接进行 join 即可。
 * 这种方式适用于 大型表格 关联 小型表格 (字典表) 的情况。和 JoinReduce 的区别在于, 没有进行排序。
 * 
 * Reference: <a href="https://www.bilibili.com/video/BV1Qp4y1n7EN?p=118">视频</a> 
 */
public class JoinMap {

    public static class MJoinMapper extends Mapper<Object, Text, NullWritable, Text> {

        private HashMap<String, String> productHashMap = new HashMap<>();
        private Text outputValue = new Text(); 
    
        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            // 获取文件路径
            URI[] cacheFiles = context.getCacheFiles();
            if (cacheFiles.length != 1) {
                throw new InterruptedException("需要一个缓存的字典表文件");
            }
            Path productFile = new Path(cacheFiles[0]);

            // 获取文件系统
            FileSystem fs = productFile.getFileSystem(context.getConfiguration());
            if (!fs.exists(productFile)) {
                throw new InterruptedException("字典表文件不存在");
            }

            // 获取文件输入流
            BufferedReader reader = null;
            String line = null;

            try{
                reader = new BufferedReader(new InputStreamReader(fs.open(productFile), "UTF-8"));
                while(true){
                    line = reader.readLine();
                    if (line == null) {
                        break;
                    }

                    String[] tokens = line.split("\\s+");
                    if (tokens.length != 2) {  // 过滤 bad line
                        continue;
                    }

                    // 现在输入的样式是: 01 西湖龙井
                    productHashMap.put(tokens[0], tokens[1]);
                }
            } finally {
                if (reader != null) {
                    reader.close();
                }
            }
        }

        @Override
        protected void map(Object key, Text value, Context context)throws IOException, InterruptedException {
            // 现在的输入样式是: 1001,01,1
            String[] tokens = value.toString().split(",");
            if(tokens.length != 3) {  // 如果存在 bad line, 直接忽略
                return;
            }

            // 赋值输出 key & value
            tokens[1] = productHashMap.get(tokens[1]);  // 将 productID 替换成 productName
            if (tokens[1] == null) {
                return;
            }
            outputValue.set(StringUtils.join(tokens, ","));
            
            // 输出 key & value
            context.write(NullWritable.get(), this.outputValue);
            return;
        }
    }

    public static void main(String[] args) throws Exception{
        final Path outputDirPath = new Path("./outputs/join_demo/");

        Job job = Job.getInstance(new Configuration(true), "reduce join");

        job.setJarByClass(JoinMap.class);

        // 删除输出文件夹
        FileSystem fs = outputDirPath.getFileSystem(job.getConfiguration());
        if (fs.exists(outputDirPath)) {
            fs.delete(outputDirPath, true);
        }

        // 设置本地运行的参数
        job.getConfiguration().set("mapreduce.framework.name", "local");  // 使用 local 模式启动, 而不是 yarn 模式启动
        job.getConfiguration().set("fs.defaultFS", "file:///");  // 使用本地文件系统, 而不是 HDFS 文件系统

        // 设置输入输出路径
        job.addCacheFile(new URI("./inputs/join_demo/product.csv"));  // 字典表 (小型数据表)
        FileInputFormat.addInputPath(job, new Path("./inputs/join_demo/order.csv"));  // 大型数据表
        FileOutputFormat.setOutputPath(job, outputDirPath);

        // 设置 MapTask 参数
        job.setMapperClass(MJoinMapper.class);
        job.setNumReduceTasks(0);
        job.setOutputKeyClass(NullWritable.class);
        job.setOutputValueClass(Text.class);

        // 运行
        job.waitForCompletion(true);

    }
}
