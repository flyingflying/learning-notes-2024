
package mr_demo;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Random;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

/**
 * 语料库生成的 MapReduce 作业。
 * 参考 {@link org.apache.hadoop.examples.RandomWriter} 和 {@link org.apache.hadoop.examples.RandomTextWriter} 两个类实现的。
 * 可以作为 {@link WordCount} 作业的输入。
 */
public class CorpusInputGeneration{
   
    public static final String MAX_VALUE = "mapreduce.corpusinputgeneration.maxwordsvalue";
    public static final String MIN_VALUE = "mapreduce.corpusinputgeneration.minwordsvalue";
    public static final String MIN_KEY = "mapreduce.corpusinputgeneration.minwordskey";
    public static final String MAX_KEY = "mapreduce.corpusinputgeneration.maxwordskey";

    public static final String NUM_OUTPUT_FILES = "mapreduce.corpusinputgeneration.numoutputfiles";
    public static final String BYTES_PER_FILE = "mapreduce.corpusinputgeneration.bytesperfile";

    enum Counters {RECORDS_WRITTEN, BYTES_WRITTEN}
 
    static class RandomTextMapper extends Mapper<NullWritable, NullWritable, Text, Text> {

        private long numBytesToWrite;  // 一个输出文件的字节数 (默认 1 GB)
        private int minWordsInKey;     // key 的最小词数 (默认 5)
        private int wordsInKeyRange;   // key 的词数浮动范围 (默认 5)
        private int minWordsInValue;   // value 的最小词数 (默认 10)
        private int wordsInValueRange; // value 的词数浮动范围 (默认 90)

        private Random random = new Random();
        private final String space = " ";

        public void setup(Context context) {
            // 读取配置文件, 设置参数值
            Configuration conf = context.getConfiguration();
            this.numBytesToWrite = conf.getLong(BYTES_PER_FILE, 1 * 1024 * 1024 * 1024);
            this.minWordsInKey = conf.getInt(MIN_KEY, 5);
            this.wordsInKeyRange = (conf.getInt(MAX_KEY, 10) - minWordsInKey);
            this.minWordsInValue = conf.getInt(MIN_VALUE, 10);
            this.wordsInValueRange = (conf.getInt(MAX_VALUE, 100) - minWordsInValue);
        }

        private Text generateSentence(int numWords) {
            // 从 words 列表中随机取一个 word 出来, 然后用空格拼接起来
            StringBuffer sentence = new StringBuffer();
            for (int i=0; i < numWords; ++i) {
                sentence.append(words[random.nextInt(words.length)]);
                sentence.append(space);
            }
            return new Text(sentence.toString());
        }

        public void map(NullWritable key, NullWritable value, Context context) throws IOException,InterruptedException {
            // 根据 InputFormat 的设定, 一个 MapTask 只有一条 "记录", 因此我们在这里要生成整个文件
            int itemCount = 0;

            while (this.numBytesToWrite > 0) {
                // 生成 key
                int numWordsKey = minWordsInKey + (wordsInKeyRange != 0 ? random.nextInt(wordsInKeyRange) : 0);
                Text keyWords = generateSentence(numWordsKey);
                // 生成 value
                int numWordsValue = minWordsInValue + (wordsInValueRange != 0 ? random.nextInt(wordsInValueRange) : 0);
                Text valueWords = generateSentence(numWordsValue);
                // 输出 key 和 value 
                context.write(keyWords, valueWords);

                // 更新状态
                // 注意: 这里生成的文件大小会略大于 BYTES_PER_MAP
                this.numBytesToWrite -= (keyWords.getLength() + valueWords.getLength() + 1);
                context.getCounter(Counters.BYTES_WRITTEN).increment(keyWords.getLength() + valueWords.getLength());
                context.getCounter(Counters.RECORDS_WRITTEN).increment(1);
                if (++itemCount % 200 == 0) {
                    context.setStatus("wrote record " + itemCount + ". " + numBytesToWrite + " bytes left.");
                }
            }
            context.setStatus("done with " + itemCount + " records.");
        }
    }

    // 自定义 RecordReader 类读取数据
    static class DummyRecordReader extends RecordReader<NullWritable, NullWritable> {
        private boolean isCalled;
        private NullWritable key;
        private NullWritable value;

        public DummyRecordReader() {this.isCalled = false;}

        /**
         * 整体逻辑: 框架调用 nextKeyValue() 方法: 
         *      如果有 "记录", 更新 this.key 和 this.value 的值, 返回 true。框架根据 getCurrentKey() 和 getCurrentValue() 获取 key 和 value 的内容。
         *      如果没有 "记录", 返回 false。框架认为数据读取完成。
         */
        public boolean nextKeyValue() {
            if (!this.isCalled) {
                this.isCalled = true;
                this.key = NullWritable.get();
                this.value = NullWritable.get();
                return true;
            }
            return false;
        }
        public NullWritable getCurrentKey() {return this.key;}
        public NullWritable getCurrentValue() {return this.value;}

        public void initialize(InputSplit split, TaskAttemptContext context) throws IOException, InterruptedException {}
        public void close() {}
        public float getProgress() {
            if(this.isCalled) return 0.0f; 
            else return 1.0f;
        }
    }

    // 自定义 InputFormat 类
    static class DummyInputFormat extends InputFormat<NullWritable, NullWritable> {

        public List<InputSplit> getSplits(JobContext job) throws IOException {
            List<InputSplit> splits = new ArrayList<InputSplit>();

            // 一个 split 对应一个 MapTask, 一个 MapTask 对应一个输出文件
            int numSplits = job.getConfiguration().getInt(NUM_OUTPUT_FILES, 1);

            // 这里的路径是无所谓的, 就随便取输出路径了
            Path outputDirPath = FileOutputFormat.getOutputPath(job);
            for(int i=0; i < numSplits; ++i) {
                Path dummyPath = new Path(outputDirPath, "dummy-split-" + i);
                splits.add(new FileSplit(dummyPath, 0,  1, (String[]) null));
            }
    
            return splits;
        }

        public RecordReader<NullWritable, NullWritable> createRecordReader(InputSplit split, TaskAttemptContext context) throws IOException, InterruptedException {
            return new DummyRecordReader();
        }
    }

    public static void main(String[] args) throws Exception {    

        Job job = Job.getInstance(new Configuration(), "corpus input generation");

        job.setJarByClass(CorpusInputGeneration.class);
        job.getConfiguration().set("mapreduce.framework.name", "local");
        job.getConfiguration().set("fs.defaultFS", "file:///");

        job.setInputFormatClass(DummyInputFormat.class);
        job.setMapperClass(RandomTextMapper.class);
        job.setNumReduceTasks(0);
        job.setOutputFormatClass(TextOutputFormat.class);
        job.getConfiguration().set("mapred.textoutputformat.separator", "");
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        Path outputDirPath = new Path("./inputs/random_corpus");
        FileSystem fs = outputDirPath.getFileSystem(job.getConfiguration());
        if (fs.exists(outputDirPath)) {
            fs.delete(outputDirPath, true);
        }
        FileOutputFormat.setOutputPath(job, new Path("./inputs/random_corpus"));

        job.getConfiguration().setInt(BYTES_PER_FILE, 16384);
        job.getConfiguration().setInt(NUM_OUTPUT_FILES, 1);

        Date startTime = new Date();
        boolean successFlag = job.waitForCompletion(true);
        Date endTime = new Date();
        System.out.println("The job took " + (endTime.getTime() - startTime.getTime()) /1000 + " seconds.");
     
        System.exit(successFlag? 0 : -1);
   }

    // 随机的 1000 个词语
    private static String[] words = {
        "diurnalness", "Homoiousian", "spiranthic", "tetragynian", "silverhead", "ungreat", "lithograph", "exploiter", "physiologian", "by",
        "hellbender", "Filipendula", "undeterring", "antiscolic", "pentagamist", "hypoid", "cacuminal", "sertularian", "schoolmasterism", "nonuple",
        "gallybeggar", "phytonic", "swearingly", "nebular", "Confervales", "thermochemically", "characinoid", "cocksuredom", "fallacious", "feasibleness",
        "debromination", "playfellowship", "tramplike", "testa", "participatingly", "unaccessible", "bromate", "experientialist", "roughcast", "docimastical",
        "choralcelo", "blightbird", "peptonate", "sombreroed", "unschematized", "antiabolitionist", "besagne", "mastication", "bromic", "sviatonosite",
        "cattimandoo", "metaphrastical", "endotheliomyoma", "hysterolysis", "unfulminated", "Hester", "oblongly", "blurredness", "authorling", "chasmy",
        "Scorpaenidae", "toxihaemia", "Dictograph", "Quakerishly", "deaf", "timbermonger", "strammel", "Thraupidae", "seditious", "plerome",
        "Arneb", "eristically", "serpentinic", "glaumrie", "socioromantic", "apocalypst", "tartrous", "Bassaris", "angiolymphoma", "horsefly",
        "kenno", "astronomize", "euphemious", "arsenide", "untongued", "parabolicness", "uvanite", "helpless", "gemmeous", "stormy",
        "templar", "erythrodextrin", "comism", "interfraternal", "preparative", "parastas", "frontoorbital", "Ophiosaurus", "diopside", "serosanguineous",
        "ununiformly", "karyological", "collegian", "allotropic", "depravity", "amylogenesis", "reformatory", "epidymides", "pleurotropous", "trillium",
        "dastardliness", "coadvice", "embryotic", "benthonic", "pomiferous", "figureheadship", "Megaluridae", "Harpa", "frenal", "commotion",
        "abthainry", "cobeliever", "manilla", "spiciferous", "nativeness", "obispo", "monilioid", "biopsic", "valvula", "enterostomy",
        "planosubulate", "pterostigma", "lifter", "triradiated", "venialness", "tum", "archistome", "tautness", "unswanlike", "antivenin",
        "Lentibulariaceae", "Triphora", "angiopathy", "anta", "Dawsonia", "becomma", "Yannigan", "winterproof", "antalgol", "harr",
        "underogating", "ineunt", "cornberry", "flippantness", "scyphostoma", "approbation", "Ghent", "Macraucheniidae", "scabbiness", "unanatomized",
        "photoelasticity", "eurythermal", "enation", "prepavement", "flushgate", "subsequentially", "Edo", "antihero", "Isokontae", "unforkedness",
        "porriginous", "daytime", "nonexecutive", "trisilicic", "morphiomania", "paranephros", "botchedly", "impugnation", "Dodecatheon", "obolus",
        "unburnt", "provedore", "Aktistetae", "superindifference", "Alethea", "Joachimite", "cyanophilous", "chorograph", "brooky", "figured",
        "periclitation", "quintette", "hondo", "ornithodelphous", "unefficient", "pondside", "bogydom", "laurinoxylon", "Shiah", "unharmed",
        "cartful", "noncrystallized", "abusiveness", "cromlech", "japanned", "rizzomed", "underskin", "adscendent", "allectory", "gelatinousness",
        "volcano", "uncompromisingly", "cubit", "idiotize", "unfurbelowed", "undinted", "magnetooptics", "Savitar", "diwata", "ramosopalmate",
        "Pishquow", "tomorn", "apopenptic", "Haversian", "Hysterocarpus", "ten", "outhue", "Bertat", "mechanist", "asparaginic",
        "velaric", "tonsure", "bubble", "Pyrales", "regardful", "glyphography", "calabazilla", "shellworker", "stradametrical", "havoc",
        "theologicopolitical", "sawdust", "diatomaceous", "jajman", "temporomastoid", "Serrifera", "Ochnaceae", "aspersor", "trailmaking", "Bishareen",
        "digitule", "octogynous", "epididymitis", "smokefarthings", "bacillite", "overcrown", "mangonism", "sirrah", "undecorated", "psychofugal",
        "bismuthiferous", "rechar", "Lemuridae", "frameable", "thiodiazole", "Scanic", "sportswomanship", "interruptedness", "admissory", "osteopaedion",
        "tingly", "tomorrowness", "ethnocracy", "trabecular", "vitally", "fossilism", "adz", "metopon", "prefatorial", "expiscate",
        "diathermacy", "chronist", "nigh", "generalizable", "hysterogen", "aurothiosulphuric", "whitlowwort", "downthrust", "Protestantize", "monander",
        "Itea", "chronographic", "silicize", "Dunlop", "eer", "componental", "spot", "pamphlet", "antineuritic", "paradisean",
        "interruptor", "debellator", "overcultured", "Florissant", "hyocholic", "pneumatotherapy", "tailoress", "rave", "unpeople", "Sebastian",
        "thermanesthesia", "Coniferae", "swacking", "posterishness", "ethmopalatal", "whittle", "analgize", "scabbardless", "naught", "symbiogenetically",
        "trip", "parodist", "columniform", "trunnel", "yawler", "goodwill", "pseudohalogen", "swangy", "cervisial", "mediateness",
        "genii", "imprescribable", "pony", "consumptional", "carposporangial", "poleax", "bestill", "subfebrile", "sapphiric", "arrowworm",
        "qualminess", "ultraobscure", "thorite", "Fouquieria", "Bermudian", "prescriber", "elemicin", "warlike", "semiangle", "rotular",
        "misthread", "returnability", "seraphism", "precostal", "quarried", "Babylonism", "sangaree", "seelful", "placatory", "pachydermous",
        "bozal", "galbulus", "spermaphyte", "cumbrousness", "pope", "signifier", "Endomycetaceae", "shallowish", "sequacity", "periarthritis",
        "bathysphere", "pentosuria", "Dadaism", "spookdom", "Consolamentum", "afterpressure", "mutter", "louse", "ovoviviparous", "corbel",
        "metastoma", "biventer", "Hydrangea", "hogmace", "seizing", "nonsuppressed", "oratorize", "uncarefully", "benzothiofuran", "penult",
        "balanocele", "macropterous", "dishpan", "marten", "absvolt", "jirble", "parmelioid", "airfreighter", "acocotl", "archesporial",
        "hypoplastral", "preoral", "quailberry", "cinque", "terrestrially", "stroking", "limpet", "moodishness", "canicule", "archididascalian",
        "pompiloid", "overstaid", "introducer", "Italical", "Christianopaganism", "prescriptible", "subofficer", "danseuse", "cloy", "saguran",
        "frictionlessly", "deindividualization", "Bulanda", "ventricous", "subfoliar", "basto", "scapuloradial", "suspend", "stiffish", "Sphenodontidae",
        "eternal", "verbid", "mammonish", "upcushion", "barkometer", "concretion", "preagitate", "incomprehensible", "tristich", "visceral",
        "hemimelus", "patroller", "stentorophonic", "pinulus", "kerykeion", "brutism", "monstership", "merciful", "overinstruct", "defensibly",
        "bettermost", "splenauxe", "Mormyrus", "unreprimanded", "taver", "ell", "proacquittal", "infestation", "overwoven", "Lincolnlike",
        "chacona", "Tamil", "classificational", "lebensraum", "reeveland", "intuition", "Whilkut", "focaloid", "Eleusinian", "micromembrane",
        "byroad", "nonrepetition", "bacterioblast", "brag", "ribaldrous", "phytoma", "counteralliance", "pelvimetry", "pelf", "relaster",
        "thermoresistant", "aneurism", "molossic", "euphonym", "upswell", "ladhood", "phallaceous", "inertly", "gunshop", "stereotypography",
        "laryngic", "refasten", "twinling", "oflete", "hepatorrhaphy", "electrotechnics", "cockal", "guitarist", "topsail", "Cimmerianism",
        "larklike", "Llandovery", "pyrocatechol", "immatchable", "chooser", "metrocratic", "craglike", "quadrennial", "nonpoisonous", "undercolored",
        "knob", "ultratense", "balladmonger", "slait", "sialadenitis", "bucketer", "magnificently", "unstipulated", "unscourged", "unsupercilious",
        "packsack", "pansophism", "soorkee", "percent", "subirrigate", "champer", "metapolitics", "spherulitic", "involatile", "metaphonical",
        "stachyuraceous", "speckedness", "bespin", "proboscidiform", "gul", "squit", "yeelaman", "peristeropode", "opacousness", "shibuichi",
        "retinize", "yote", "misexposition", "devilwise", "pumpkinification", "vinny", "bonze", "glossing", "decardinalize", "transcortical",
        "serphoid", "deepmost", "guanajuatite", "wemless", "arval", "lammy", "Effie", "Saponaria", "tetrahedral", "prolificy",
        "excerpt", "dunkadoo", "Spencerism", "insatiately", "Gilaki", "oratorship", "arduousness", "unbashfulness", "Pithecolobium", "unisexuality",
        "veterinarian", "detractive", "liquidity", "acidophile", "proauction", "sural", "totaquina", "Vichyite", "uninhabitedness", "allegedly",
        "Gothish", "manny", "Inger", "flutist", "ticktick", "Ludgatian", "homotransplant", "orthopedical", "diminutively", "monogoneutic",
        "Kenipsim", "sarcologist", "drome", "stronghearted", "Fameuse", "Swaziland", "alen", "chilblain", "beatable", "agglomeratic",
        "constitutor", "tendomucoid", "porencephalous", "arteriasis", "boser", "tantivy", "rede", "lineamental", "uncontradictableness", "homeotypical",
        "masa", "folious", "dosseret", "neurodegenerative", "subtransverse", "Chiasmodontidae", "palaeotheriodont", "unstressedly", "chalcites", "piquantness",
        "lampyrine", "Aplacentalia", "projecting", "elastivity", "isopelletierin", "bladderwort", "strander", "almud", "iniquitously", "theologal",
        "bugre", "chargeably", "imperceptivity", "meriquinoidal", "mesophyte", "divinator", "perfunctory", "counterappellant", "synovial", "charioteer",
        "crystallographical", "comprovincial", "infrastapedial", "pleasurehood", "inventurous", "ultrasystematic", "subangulated", "supraoesophageal", "Vaishnavism", "transude",
        "chrysochrous", "ungrave", "reconciliable", "uninterpleaded", "erlking", "wherefrom", "aprosopia", "antiadiaphorist", "metoxazine", "incalculable",
        "umbellic", "predebit", "foursquare", "unimmortal", "nonmanufacture", "slangy", "predisputant", "familist", "preaffiliate", "friarhood",
        "corelysis", "zoonitic", "halloo", "paunchy", "neuromimesis", "aconitine", "hackneyed", "unfeeble", "cubby", "autoschediastical",
        "naprapath", "lyrebird", "inexistency", "leucophoenicite", "ferrogoslarite", "reperuse", "uncombable", "tambo", "propodiale", "diplomatize",
        "Russifier", "clanned", "corona", "michigan", "nonutilitarian", "transcorporeal", "bought", "Cercosporella", "stapedius", "glandularly",
        "pictorially", "weism", "disilane", "rainproof", "Caphtor", "scrubbed", "oinomancy", "pseudoxanthine", "nonlustrous", "redesertion",
        "Oryzorictinae", "gala", "Mycogone", "reappreciate", "cyanoguanidine", "seeingness", "breadwinner", "noreast", "furacious", "epauliere",
        "omniscribent", "Passiflorales", "uninductive", "inductivity", "Orbitolina", "Semecarpus", "migrainoid", "steprelationship", "phlogisticate", "mesymnion",
        "sloped", "edificator", "beneficent", "culm", "paleornithology", "unurban", "throbless", "amplexifoliate", "sesquiquintile", "sapience",
        "astucious", "dithery", "boor", "ambitus", "scotching", "uloid", "uncompromisingness", "hoove", "waird", "marshiness",
        "Jerusalem", "mericarp", "unevoked", "benzoperoxide", "outguess", "pyxie", "hymnic", "euphemize", "mendacity", "erythremia",
        "rosaniline", "unchatteled", "lienteria", "Bushongo", "dialoguer", "unrepealably", "rivethead", "antideflation", "vinegarish", "manganosiderite",
        "doubtingness", "ovopyriform", "Cephalodiscus", "Muscicapa", "Animalivora", "angina", "planispheric", "ipomoein", "cuproiodargyrite", "sandbox",
        "scrat", "Munnopsidae", "shola", "pentafid", "overstudiousness", "times", "nonprofession", "appetible", "valvulotomy", "goladar",
        "uniarticular", "oxyterpene", "unlapsing", "omega", "trophonema", "seminonflammable", "circumzenithal", "starer", "depthwise", "liberatress",
        "unleavened", "unrevolting", "groundneedle", "topline", "wandoo", "umangite", "ordinant", "unachievable", "oversand", "snare",
        "avengeful", "unexplicit", "mustafina", "sonable", "rehabilitative", "eulogization", "papery", "technopsychology", "impressor", "cresylite",
        "entame", "transudatory", "scotale", "pachydermatoid", "imaginary", "yeat", "slipped", "stewardship", "adatom", "cockstone",
        "skyshine", "heavenful", "comparability", "exprobratory", "dermorhynchous", "parquet", "cretaceous", "vesperal", "raphis", "undangered",
        "Glecoma", "engrain", "counteractively", "Zuludom", "orchiocatabasis", "Auriculariales", "warriorwise", "extraorganismal", "overbuilt", "alveolite",
        "tetchy", "terrificness", "widdle", "unpremonished", "rebilling", "sequestrum", "equiconvex", "heliocentricism", "catabaptist", "okonite",
        "propheticism", "helminthagogic", "calycular", "giantly", "wingable", "golem", "unprovided", "commandingness", "greave", "haply",
        "doina", "depressingly", "subdentate", "impairment", "decidable", "neurotrophic", "unpredict", "bicorporeal", "pendulant", "flatman",
        "intrabred", "toplike", "Prosobranchiata", "farrantly", "toxoplasmosis", "gorilloid", "dipsomaniacal", "aquiline", "atlantite", "ascitic",
        "perculsive", "prospectiveness", "saponaceous", "centrifugalization", "dinical", "infravaginal", "beadroll", "affaite", "Helvidian", "tickleproof",
        "abstractionism", "enhedge", "outwealth", "overcontribute", "coldfinch", "gymnastic", "Pincian", "Munychian", "codisjunct", "quad",
        "coracomandibular", "phoenicochroite", "amender", "selectivity", "putative", "semantician", "lophotrichic", "Spatangoidea", "saccharogenic", "inferent",
        "Triconodonta", "arrendation", "sheepskin", "taurocolla", "bunghole", "Machiavel", "triakistetrahedral", "dehairer", "prezygapophysial", "cylindric",
        "pneumonalgia", "sleigher", "emir", "Socraticism", "licitness", "massedly", "instructiveness", "sturdied", "redecrease", "starosta",
        "evictor", "orgiastic", "squdge", "meloplasty", "Tsonecan", "repealableness", "swoony", "myesthesia", "molecule", "autobiographist",
        "reciprocation", "refective", "unobservantness", "tricae", "ungouged", "floatability", "Mesua", "fetlocked", "chordacentrum", "sedentariness",
        "various", "laubanite", "nectopod", "zenick", "sequentially", "analgic", "biodynamics", "posttraumatic", "nummi", "pyroacetic",
        "bot", "redescend", "dispermy", "undiffusive", "circular", "trillion", "Uraniidae", "ploration", "discipular", "potentness",
        "sud", "Hu", "Eryon", "plugger", "subdrainage", "jharal", "abscission", "supermarket", "countergabion", "glacierist",
        "lithotresis", "minniebush", "zanyism", "eucalypteol", "sterilely", "unrealize", "unpatched", "hypochondriacism", "critically", "cheesecutter",
    };
}
 
