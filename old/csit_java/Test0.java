public class Test0 {
    public static void main(String[] s){
        A x = new A();
        x.b = true;
        x.i = 30;
        x.h = Hand.ROCK;

        A y = new A();
        y.b = false;
        y.i = 40;
        y.h = Hand.PAPER;

        x.print();
        y.print();
    }
}
