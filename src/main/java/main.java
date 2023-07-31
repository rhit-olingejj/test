import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.storage.StorageLevel;
import org.apache.spark.util.DoubleAccumulator;
import scala.Option;
import scala.Tuple2;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Stream;

public class main {
    //split into targets and not
    private static SparkConf conf;
    private static final int MAX_PARTICLES = 10;
    private static final String training_file = "phplE7q6h.arff";
//    Iris/Training.txt 4
//ArtificalDataset1/Data.txt 2
//    Wine/Training.txt 13
    private static final String testing_file = "phplE7q6h.arff";
    //    Iris/Test.txt
//    ArtificalDataset1/test.txt
    //    Wine/Testing.txt

    private static final int MAX_ITERATION = 100;
    private static SparkContext SparkCon;
    private static JavaSparkContext sc;
    private static ArrayList<Point> swarm = new ArrayList<Point>();
    public static HashMap<String, Double> globalMax = new HashMap<String, Double>();

    public static void main(String[] args) throws IOException {
        FileWriter fw = new FileWriter(new File("output.txt"));

        // setting the configuration for Spark Driver Program
        Logger.getLogger("org").setLevel(Level.ERROR);
        conf = new SparkConf().setAppName("SparkFire").setMaster("spark://spark.cs.ndsu.edu:7077");
        conf.set("spark.executor.instances", "2");

        // conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer");
        // 5 cores on each workers - Local Mode
        //conf.set("spark.executor.cores", "5");


        SparkCon = new SparkContext(conf);
        sc = new JavaSparkContext(SparkCon);
        sc.setCheckpointDir("checkpoint");

        //Read Data File
        JavaRDD<String> file = sc.textFile(training_file, 2);

        //create PairRDD Dataset from file <Features,Class Label>
        JavaPairRDD<String, String> Dataset_preprocessing = file.mapToPair(line -> new Tuple2<>
                (
                        line.substring(line.lastIndexOf(',') + 1),
                        line.substring(0, line.lastIndexOf(','))
                )
        );

        //Extract the Classes labels from Dataset
        List<String> class_label = Dataset_preprocessing.map(v -> v._1.trim()).distinct().collect();

        JavaPairRDD<String, double[]> dataset;
        //Normalized DataSet
        dataset = Dataset_preprocessing.mapValues(v -> Stream.of(v.split(",")).mapToDouble(Double::parseDouble).toArray());

        dataset.persist(StorageLevel.MEMORY_ONLY_SER());
        dataset.setName("persist Dataset");

        MapAccumulator accumlater = new MapAccumulator();
        SparkCon.register(accumlater);
        long count = dataset.count();
        long StartTime = System.currentTimeMillis();
        long iteration = 0;
        //iterate for every class
        //initilize swarm of points
        for(int i = 0; i<MAX_PARTICLES;i++){
            for(String s : class_label){
                double[] temp = new double[14];
                for (int j = 0; j < temp.length; j++) {
                    temp[j]=Math.random()*10;
                }
                swarm.add(new Point(s,temp,i));
            }
        }
        while (iteration <= MAX_ITERATION) {
            Broadcast<List<Point>> BroadcastGroupOfParticles = sc.broadcast(swarm);

            dataset.foreach(e -> {
                for (Point p : BroadcastGroupOfParticles.getValue()) {
                    if (e._1.trim().equals(p.className)) {
                        double distance = calcFitness(p,e);
                        accumlater.add(new Tuple2<>(p.toString(), distance));

                    }
                }
            });
                for(int i =0;i< swarm.size();i++){
                    for(int j=0;j<swarm.size();j++){
                        if(swarm.get(i).className.equals(swarm.get(j).className) && accumlater.value().get(swarm.get(i).toString())/count<calcBrightness(swarm.get(i),swarm.get(j), accumlater.value(),count)){
                           //movement
                            for(int k = 0;k<swarm.get(i).vals.length; k ++){
                                swarm.get(i).vals[k]+=moveDistanceX(swarm.get(i),swarm.get(j),k, accumlater.value(), count);
                            }
                        }
                    }
                }
            int q = 0;
                double total = 0;
            for(String s : accumlater.value().keySet()){
                fw.write(String.format("%s : %f\n",s,accumlater.value().get(s)));
                total+=accumlater.value().get(s);
                fw.write(String.format("%s : %s\n", swarm.get(q).toString(), Arrays.toString(swarm.get(q).vals)));
                q++;
            }

            accumlater.reset();
            BroadcastGroupOfParticles.unpersist(true);
            BroadcastGroupOfParticles.destroy();
            ++iteration;
        }
        HashMap<String,double[]> classifer = new HashMap<>();
        for(Point g: swarm)
        {
            classifer.put(g.className, g.vals);
        }
        evaluate(classifer, fw);
        long EndTime = System.currentTimeMillis();
        fw.write(String.format("TOTAL RUNTIME: %d\n",EndTime-StartTime));
        fw.flush();
        fw.close();

    }
    public static double evaluate(HashMap<String, double []> classifier, FileWriter fw) throws IOException {
        String result="";
        JavaRDD<String> file = sc.textFile(testing_file,6);
        JavaPairRDD<String,String> Dataset_preprocessing=file.mapToPair(line->new Tuple2<>
                (
                        line.substring(line.lastIndexOf(',')+1),
                        line.substring(0, line.lastIndexOf(','))
                )
        );

        JavaPairRDD<String, double[]> dataset;
        dataset = Dataset_preprocessing
                    .mapValues(v -> Stream.of(v.split(",")).mapToDouble(Double::parseDouble).toArray());

        Broadcast<HashMap<String,double []>> Broadcast_classifer = sc.broadcast(classifier);
        DoubleAccumulator MissClassification = new DoubleAccumulator();
        MissClassification.register(SparkCon, Option.apply("fitnees_value"), false);
        DoubleAccumulator NumberOfInstance = new DoubleAccumulator();
        NumberOfInstance.register(SparkCon, Option.apply("fitnees_value"), false);


        dataset.foreach(e-> {
            NumberOfInstance.add(1);
            double min=0;
            String Predicate_class="";
            for (Map.Entry<String, double[]> c : Broadcast_classifer.getValue().entrySet())
            {
                double distance = calcEuclidDistance(e._2,c.getValue());
                if (Predicate_class.isEmpty()) { min=distance; Predicate_class=c.getKey();}
                else if(distance<min) {min=distance; Predicate_class=c.getKey();}
            }
            if (!(Predicate_class.trim().equals(e._1.trim()))) {MissClassification.add(1);}
        });


        fw.write("--------------- Final Results ---------------------\n");
        result+=System.lineSeparator()+"--------------- Final Results ---------------------";
        fw.write("Number of incorrectly classified instances : "+MissClassification.value() + "\n");
        result+=System.lineSeparator()+("Number of incorrectly classified instances : "+MissClassification.value());
        fw.write("Accuarcy :"+(100.0-(MissClassification.value()/NumberOfInstance.value())*100.0) + "\n");
        result+=System.lineSeparator()+("Accuarcy :"+(100.0-(MissClassification.value()/NumberOfInstance.value())*100.0));
        fw.write("Miss-classification rate : "+MissClassification.value()/NumberOfInstance.value() + "\n");
        result+=System.lineSeparator()+("Miss-classification rate : "+MissClassification.value()/NumberOfInstance.value());

        Broadcast_classifer.unpersist(true);
        Broadcast_classifer.destroy();
        NumberOfInstance.reset();
        NumberOfInstance.reset();
    return 0;
    }

    public static int randomWalk() {
        return (int) ((Math.random() * 2) - 1);
    }

    public static double moveDistanceX(Point p1, Point p2, int dim, HashMap<String,Double> bright, long count) {
        return ((bright.get(p1.toString())/count)*Math.exp(-calcEuclidDistance(p1.vals, p2.vals)) * (p2.vals[dim] - p1.vals[dim]) + Math.random());
    }


    public static double calcFitness(Point p, Tuple2<String, double[]> e) throws Exception {
        double ret = 0;
            if(p.className.equals(e._1.trim())){
                ret+= calcEuclidDistance(p.vals, e._2);
            }

        return ret;
    }

    public static double calcBrightness(Point p1, Point p2, HashMap<String, Double> fit, long count) {
        return (fit.get(p2.toString())/count) / Math.pow(calcEuclidDistance(p1.vals, p2.vals), 2);
    }

    public static double calcEuclidDistance(double[] a, double[] b) {
        double temp = 0;
        for(int i =0; i<b.length;i++){
            temp +=Math.pow((a[i] - b[i]), 2);
        }
        return Math.sqrt(temp);
    }
}