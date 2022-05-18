public class Test3 {
    public static void main(String[] s){
        A a;
        B b1, b2;

        a = new A();
        b1 = new B();
        b2 = new B();
        b2.i = 99;
        b2.h = Hand.ROCK;

        D obj = new D();

        obj.set(a,b1);
        obj.run();

        obj.set(a,b2);
        obj.run();
    }
}
