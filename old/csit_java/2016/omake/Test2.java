public class Test2 {
    public static void main(String[] s){
        // 単体のインスタンス
        C obj = new C();

        obj.a.b = true;
        obj.a.i = 11;
        obj.a.h = Hand.ROCK;
        
        obj.a.print();
        obj.b.print();

        // インスタンスを配列に
        // 注意: Javaの配列のnewの構文
        //       クラス名[] 変数名 = new クラス名[配列の要素数]
        C[] setC = new C[3];

        for(int i=0; i<setC.length; i++){
            setC[i] = new C(); // 各々の要素にインスタンスを作って代入
        }

        for(int i=0; i<setC.length; i++){
            setC[i].b.i = 3*i;
            setC[i].b.j = 4*i;
        }

        for(int i=0; i<setC.length; i++){
            System.out.println("Array element "+i+"th");
            setC[i].b.print();
        }
        
    }
}
