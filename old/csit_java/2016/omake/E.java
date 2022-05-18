public class E {
    // 変数を「カプセル化」
    private boolean flag;
    private int counter1, counter2;
    // コンストラクタ
    public E(){
        this.flag = true;
        this.counter1 = 0;
        this.counter2 = 0;      
    }
    // メソッド
    public void call(){
        System.out.println("call");
        if(flag)
            this.counter1++;
        else
            this.counter2++;
    }
    public void lost(String msg){
        System.out.println(msg);
        this.flag = false;
    }
    public void print(){
        System.out.println("counter 1:"+counter1+" 2:"+counter2);
    }
}
