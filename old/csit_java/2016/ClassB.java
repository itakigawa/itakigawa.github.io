public class ClassB extends ClassA {
    private int value;
    public ClassB(String message,int value){
        super(message);
        this.value = value;
    }
    public void printMessage(){
        super.printMessage();
        System.out.println("Value:"+this.value);
    }
    public static void main(String[] args){
        ClassA a = new ClassA("ClassA instance");
        ClassB b = new ClassB("ClassB instance",100);
        a.printMessage();
        b.printMessage();
    }
}
