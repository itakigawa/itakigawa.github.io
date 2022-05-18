public class TestArray {
    public static void main(String[] args) {

        double[] array1 = {1.012, -2.599, 3.421};

        int[] array2 = new int[6];
        array2[0] = 3;  array2[1] = 4;  array2[2] = 5; 
        array2[3] = 6;  array2[4] = 7;  array2[5] = 8;

        // for文 with println
        for (int i=0; i < array1.length; i++) {
            System.out.println(array1[i]);
        }

        // 多次元配列
        int[][] array3;
        array3 = new int[2][3];

        for (int i=0; i < array3.length; i++) {
            for (int j=0; j < array3[0].length; j++) {
                array3[i][j] = array2[i+j];
            }
        }
        // 拡張for文 with printf
        for (double value : array1) {
            System.out.printf("array1 %1.2f\n", value);
        }
        for (int value : array2) {
            System.out.printf("array2 %03d\n", value);
        }
        // Javaでは多次元配列は「配列の配列」
        for (int[] row : array3) {
            for (int value : row) {
                System.out.print(value+" ");
            }
            System.out.print("\n");
        }
    }
}
