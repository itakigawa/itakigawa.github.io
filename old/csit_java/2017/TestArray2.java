public class TestArray2 {
    public static double getMean(double[] array){
        double ret=0.0;
        for(double value : array){ ret += value; }
        return ret/array.length;
    }
    public static double[] incArray(double[] array){
	double[] ret = new double[array.length];
        for(int i=0; i < array.length; i++){
            ret[i] = array[i] + 1;
        }
	return ret;
    }
    public static void main(String[] args) {
        double[] a1 = {1.012, -2.599, 3.421};
        System.out.printf("mean: %1.3f\n",getMean(a1));
        double[] a2 = incArray(a1);
        for (double value : a2) {
            System.out.printf("%1.3f ", value);
        }
        System.out.println();
        System.out.printf("mean: %1.3f\n",getMean(a2));
    }
}
