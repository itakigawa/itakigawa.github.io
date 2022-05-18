public class Test1 {
    public static void main(String[] s){
        B x = new B();

        B y = new B();
        y.b = false;
        y.i = 20;
        
        x.print();
        y.print();
    }
}
