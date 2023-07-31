import java.io.Serializable;

public class Point implements Comparable<Point>, Serializable {
    public String className = "";
    public double[] vals;
    public double fit = -1;
    public int id = -1;

    public Point(String className, double[] vals, int id) {
        this.className = className;
        this.vals = vals;
        this.id = id;
    }

    @Override
    public int compareTo(Point o) {
        if(this.fit<o.fit) return 1;
        if(this.fit>o.fit) return -1;
        return 0;
    }

    @Override
    public String toString(){
        return String.format("%s %d", this.className, this.id);
    }
}
