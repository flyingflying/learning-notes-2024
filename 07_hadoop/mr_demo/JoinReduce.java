package mr_demo;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

/**
 * 假设现在有两个 CSV 文件:
 *      1. 订单表, 有三个字段: 订单 ID (orderID), 产品 ID (productID) 和 购买数量 (amount)。
 *      2. 产品表, 有两个字段: 产品 ID (productID) 和 产品名称 (productName)
 * 
 * 我们现在要将两张表 JOIN 在一起, 用 产品名称 替换 订单表 中的 产品 ID, 使用 SQL 语句如下:
 * SELECT 
 *      order_table.order_id,
 *      product_table.product_name,
 *      order_table.amount
 * FROM 
 *      order_table 
 *      LEFT JOIN product_table ON order_table.product_id = product_table.product_id;
 * 
 * 那么用 MapReduce 框架如何实现呢?
 *      1. 首先, 这里有两个文件。在 map 运算中, 我们需要根据 split 对应的文件名分开处理。
 *      2. map 运算的输出 key 是 productID, value 是一个自定义的类, 包含两张表所有的字段
 *      3. shuffle 阶段会自动帮助我们将相同 key 的数据聚合在一起
 *      4. reduce 运算遍历 values, 进行配对即可
 * 
 * Reference: <a href="https://www.bilibili.com/video/BV1Qp4y1n7EN?p=113">视频</a>
 * 
 * 后续: {@link JoinMap}
 */
public class JoinReduce {

    // 这里的自定义类作为 value 的类型, 不是 key 的类型, 不需要实现 WritableComparable 接口, 只要实现 Writable 接口即可
    static class TableFields implements Writable {
        private String orderID = "";
        private String amount = "";
        private String productName = "";
        private String flag = "";  // 标识是 order 记录还是 product 记录

        public String getOrderID() {return this.orderID;}
        public String getAmount() {return this.amount;}
        public String getProductName() {return this.productName;}
        public String getFlag() {return this.flag;}

        public void setOrderID(String orderID) {this.orderID = orderID;}
        public void setAmount(String amount) {this.amount = amount;}
        public void setProductName(String productName) {this.productName = productName;}
        public void setFlag(String flag) {this.flag = flag;}

        public TableFields() {}  // 空构造方法

        @Override
        public void write(DataOutput out) throws IOException {
            out.writeUTF(this.flag);
            out.writeUTF(this.orderID);
            out.writeUTF(this.amount);
            out.writeUTF(this.productName);
        }

        @Override
        public void readFields(DataInput in) throws IOException {
            this.flag = in.readUTF();
            this.orderID = in.readUTF();
            this.amount = in.readUTF();
            this.productName = in.readUTF();
        }

        public String toString() {return this.orderID + "," + this.productName + "," + this.amount;}
    }

    public static class RJoinMapper extends Mapper<Object, Text, Text, TableFields> {

        private Text outputKey = new Text();
        private TableFields tableFields = new TableFields();
    
        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            FileSplit split = (FileSplit) context.getInputSplit();
            String fileName = split.getPath().getName();
            if (fileName.contains("order")) {
                tableFields.setFlag("order");
                return;
            }
            
            if (fileName.contains("product")) {
                tableFields.setFlag("product");
                return;
            }
        }

        @Override
        protected void map(Object key, Text value, Context context)throws IOException, InterruptedException {
            if ("order".equals(tableFields.getFlag())) {
                // 现在的输入样式是: 1001,01,1
                String[] tokens = value.toString().split(",");
                if(tokens.length != 3) {  // 如果存在 bad line, 直接忽略
                    return;
                }

                // 赋值输出 key & value
                this.tableFields.setOrderID(tokens[0]);
                this.outputKey.set(tokens[1]);
                this.tableFields.setAmount(tokens[2]);
                
                // 输出 key & value
                context.write(this.outputKey, this.tableFields);
                return;
            }

            if ("product".equals(tableFields.getFlag())) {
                // 现在输入的样式是: 01 西湖龙井
                String[] tokens = value.toString().split("\\s+");
                if(tokens.length != 2) {  // 如果存在 bad line, 直接忽略
                    return;
                }

                // 赋值输出 key & value
                this.outputKey.set(tokens[0]);
                this.tableFields.setProductName(tokens[1]);

                // 输出 key & value
                context.write(this.outputKey, this.tableFields);
                return;
            }
        }
    }

    public static class RJoinReducer extends Reducer<Text, TableFields, NullWritable, TableFields> {
        @Override
        protected void reduce(Text key, Iterable<TableFields> values, Context context) throws IOException, InterruptedException {
            
            List<TableFields> results = new ArrayList<>();
            String productName = null;

            for (TableFields value: values) {
                if ("order".equals(value.getFlag())) {
                    // 注意, 这里一定要 new 一个对象
                    TableFields tableFields = new TableFields();
                    tableFields.setOrderID(value.getOrderID());
                    tableFields.setAmount(value.getAmount());
                    results.add(tableFields);
                    continue;
                }

                if ("product".equals(value.getFlag())) {
                    // 我们假设 productID 是唯一的, 只有一个
                    if (!(productName == null)) {
                        throw new InterruptedException("productID 在产品表中不唯一");
                    }
                    productName = value.getProductName();
                    continue;
                }
            }

            for (TableFields result: results) {
                result.setProductName(productName);
                context.write(NullWritable.get(), result);
            }
        }
    }

    public static void main(String[] args) throws Exception{
        final Path inputDirPath = new Path("./inputs/join_demo/");
        final Path outputDirPath = new Path("./outputs/join_demo/");

        Job job = Job.getInstance(new Configuration(true), "reduce join");

        job.setJarByClass(JoinReduce.class);

        // 删除输出文件夹
        FileSystem fs = outputDirPath.getFileSystem(job.getConfiguration());
        if (fs.exists(outputDirPath)) {
            fs.delete(outputDirPath, true);
        }

        // 设置本地运行的参数
        job.getConfiguration().set("mapreduce.framework.name", "local");  // 使用 local 模式启动, 而不是 yarn 模式启动
        job.getConfiguration().set("fs.defaultFS", "file:///");  // 使用本地文件系统, 而不是 HDFS 文件系统

        // 设置输入输出路径
        FileInputFormat.addInputPath(job, inputDirPath);
        FileOutputFormat.setOutputPath(job, outputDirPath);

        // 设置 MapTask 参数
        job.setMapperClass(RJoinMapper.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(TableFields.class);

        // 设置 ReduceTask 参数
        job.setReducerClass(RJoinReducer.class);
        job.setOutputKeyClass(NullWritable.class);
        job.setOutputValueClass(TableFields.class);
        job.setNumReduceTasks(1);

        // 运行
        job.waitForCompletion(true);

    }
}
