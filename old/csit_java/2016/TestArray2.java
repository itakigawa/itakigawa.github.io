public class TestArray2 {
    public static double getMean(double[] array){
        double ret=0.0;
        for(double value : array){ ret += value; }
        return ret/array.length;
    }
    public static void incArray(double[] array){
        for(int i=0; i < array.length; i++){
            array[i] += 1;
        }
    }
    public static void main(String[] args) {
        double[] array = {1.012, -2.599, 3.421};
        System.out.printf("mean: %1.2f\n",getMean(array));
        incArray(array);
        for (double value : array) {
            System.out.printf("%1.2f ", value);
        }
        System.out.println();
        System.out.printf("mean: %1.2f\n",getMean(array));
    }
}
