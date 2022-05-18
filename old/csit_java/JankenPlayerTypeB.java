public class JankenPlayerTypeB extends RandomJankenPlayer {
    public JankenPlayerTypeB(String name){
        super(name);
    }
    public Hand showHand(){
        return Hand.ROCK;
    }
    public static void main(String[] args){
        RandomJankenPlayer player1 = new RandomJankenPlayer("Suzuki");
        RandomJankenPlayer player2 = new JankenPlayerTypeB("Yamamoto");
        Hand hand1, hand2;
        for(int i=0; i<10; i++){
            hand1 = player1.showHand();
            hand2 = player2.showHand();
            System.out.println(i+")"+
                               "["+player1.getName()+"]"+hand1+" vs "+
                               "["+player2.getName()+"]"+hand2);
        }
    }
}
