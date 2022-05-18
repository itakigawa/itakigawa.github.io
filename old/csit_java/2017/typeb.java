public class JankenPlayerTypeB extends JankenPlayer {
    public JankenPlayerTypeB(String name){
        super(name);
    }
    public Hand showHand(){
        return Hand.ROCK;
    }
}
