public class RandomJankenPlayer {
    public String name;
    public RandomJankenPlayer(String name){
        this.name = name;
    }
    public Hand showHand(){
        Hand play;
        double rnd = Math.random();
        if(rnd < 1.0/3.0){
            play = Hand.ROCK;
        }else if (rnd < 2.0/3.0){
            play = Hand.PAPER;
        }else{
            play = Hand.SCISSORS;
        }
        return play;
    }
    // main
    public static void main(String[] args){
        RandomJankenPlayer player1 = new RandomJankenPlayer("Yamada");
        RandomJankenPlayer player2 = new RandomJankenPlayer("Suzuki");
        Hand hand1, hand2;
        for(int i=0; i<10; i++){
            hand1 = player1.showHand();
            hand2 = player2.showHand();
            System.out.println(i+")"+
                               "["+player1.name+"]"+hand1+" vs "+
                               "["+player2.name+"]"+hand2);
        }
    }
}
