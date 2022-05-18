public class ClassA {
    private String message;
    public ClassA(String message){
        System.out.println("ClassA-Constructor:"+message);
        this.message = message;
    }
    public void printMessage(){
        System.out.println("Message:"+this.message);
    }
}
