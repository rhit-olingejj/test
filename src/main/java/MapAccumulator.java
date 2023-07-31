
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.spark.util.AccumulatorV2;

import scala.Serializable;
import scala.Tuple2;

public class MapAccumulator extends AccumulatorV2<Tuple2<String,Double>,
		HashMap<String, Double>> implements Serializable {
	private HashMap<String, Double> accumlator=null;
	public MapAccumulator() {
		accumlator = new HashMap<String, Double>();
	}
	public MapAccumulator(HashMap<String, Double> a) {
		if (a.size()!=0)
			accumlator = new HashMap<String,Double>(a);
	}
	@Override
	public void add(Tuple2<String, Double> e) {
		if (accumlator.containsKey(e._1))
		{

			double a=accumlator.get(e._1);
			accumlator.put(e._1,a+ e._2);
		}
		else
			accumlator.put(e._1,e._2);

	}
	@Override
	public boolean isZero() {

		return accumlator.size()==0 ?true : false;
	}

	@Override
	public AccumulatorV2<Tuple2<String, Double>, HashMap<String, Double>> copy() {

		return new MapAccumulator(value());
	}
	@Override
	public void merge(AccumulatorV2<Tuple2<String, Double>, HashMap<String, Double>> other) {


		HashMap<String, Double> a =value();
		for (Map.Entry<String,Double> e : other.value().entrySet())
			if (a.containsKey(e.getKey()))
			{
				double other_value= e.getValue();
				double current_value = a.get(e.getKey());
				a.put(e.getKey(), other_value+current_value);
			}
			else {
				a.put(e.getKey(), e.getValue());
			}
	}
	@Override
	public void reset() {
		accumlator= new HashMap<String,Double>();

	}
	@Override
	public HashMap<String, Double> value() {
		// TODO Auto-generated method stub
		return accumlator;
	}


}