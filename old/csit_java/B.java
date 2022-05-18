public class B {
    boolean b;
    int i, j;
    Hand h;
    public B(){ // コンストラクタ
        this.b = true;
        this.i = 10;
        this.j = 20;
        this.h = Hand.SCISSORS;
    }
    public void print(){
        System.out.println(b);
        System.out.println(i);
        System.out.println(h);  
    }
}
