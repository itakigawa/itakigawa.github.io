public class D {
    A a;
    B b;
    void set(A a, B b){
        this.a = a;
        this.b = b;
    }
    void run(){
        this.a.b = false;
        this.a.i = 1;
        this.a.h = Hand.PAPER;

        this.a.print();
        this.b.print();
    }
}
