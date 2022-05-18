public class Test4 {
    public static void main(String[] s){
        // objのprivate変数に直接アクセスできない/しないが
        // 部品objを使って機能が実現できていることに注意
        
        E obj; 

        obj = new E();  
        
        for(int i=0; i<7; i++) obj.call();
        obj.print();

        obj.lost("Change mode");

        for(int i=0; i<3; i++) obj.call();
        obj.print();
    }
}
