public class C {
    A a;
    B b;
    public C(){ // コンストラクタ(new時にのみ呼ばれる)
        this.a = new A();
        this.b = new B();
    }
}
