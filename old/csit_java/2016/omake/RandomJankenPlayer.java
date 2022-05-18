public class RandomJankenPlayer {
    private boolean isLose;
    public RandomJankenPlayer(){
        this.isLose = false;
    }
    public void lost(){
        this.isLose = true;
    }
    public Hand showHand(){
        Hand play;
        if(this.isLose){
            System.out.print("ROCK-Mode: ");
            play = Hand.ROCK;
        }else{
            double rnd = Math.random();
            if(rnd < 1.0/3.0){
                play = Hand.ROCK;
            }else if (rnd < 2.0/3.0){
                play = Hand.PAPER;
            }else{
                play = Hand.SCISSORS;
            }
        }
        return play;
    }
}
