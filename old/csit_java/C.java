public class C {
    // 変数を「カプセル化」
    private boolean flag;
    private int cnt1, cnt2;
    // コンストラクタ
    public C(){
        this.flag = true;
        this.cnt1 = 0;
        this.cnt2 = 0;
    }
    // メソッド
    public void call(){
        System.out.println("call");
        if(flag) this.cnt1++;
        else this.cnt2++;
    }
    public void lost(String msg){
        System.out.println(msg);
        this.flag = false;
    }
    public void print(){
        System.out.println(this.cnt1);
        System.out.println(this.cnt2);
    }
}
            
