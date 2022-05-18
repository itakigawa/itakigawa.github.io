public class RandomJankenPlayer {
    private String name;
    public String getName(){
        return this.name;
    }
    public void setName(String name){
        this.name = setName(name);
    }
    public RandomJankenPlayer(){}
    public RandomJankenPlayer(String name){
        this.name = name;
    }
    ...
    // 他のメソッドではprivate変数にはアクセサ経由にしておく
    public void report(){
        System.out.println(getName()+" "+getNWin()+" win, "+getNLose()+" lose, "+getNDraw()+" draw");
    }
    ...
}

public class InteractiveJankenPlayer extends RandomJankenPlayer {
    public InteractiveJankenPlayer(){
        System.out.print("Your Name? ");
        String scanner = new Scanner(System.in);
        String name = scanner.nextLine();
        this.setName(name);
        scanner.close();
    }
    ...
}
