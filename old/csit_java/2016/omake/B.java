public class B {
    boolean b;
    int i, j;
    Hand h;
    public B(){ // コンストラクタ(new時にのみ呼ばれる)
        this.b = true;
        this.i = 10;
        this.j = 20;
        this.h = Hand.SCISSORS;
    }
    public void print(){
        System.out.println(b+" "+i+" "+j+" "+h);
    }
}
