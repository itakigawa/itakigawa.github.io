import java.util.Scanner;
import java.util.HashMap;

public class Judge2 {
    private String name;
    private JankenPlayer[] players;
    private Hand[] hands;
    public Judge2(String name){
        this.name = name;
    }
    public String getName(){
        return this.name;
    }
    public void setPlayers(JankenPlayer[] players){
        this.players = players;
    }
    private void notifyAll(Hand handWin, Hand handLose){
        for(int i=0; i<players.length; i++){
            if(hands[i]==handWin){
                players[i].notify(Result.WIN);
            }else if(hands[i]==handLose){
                players[i].notify(Result.LOSE);
            }
        }
    }
    public void play(){
        hands = new Hand[players.length];
        int countRock=0, countScissors=0, countPaper=0;
        for(int i=0; i<players.length; i++){
            hands[i] = players[i].showHand();
            if(hands[i]==Hand.ROCK){ 
                countRock += 1;
            }else if(hands[i]==Hand.SCISSORS){ 
                countScissors += 1;
            }else if(hands[i]==Hand.PAPER){ 
                countPaper += 1;
            }
            System.out.println(players[i].getName()+" "+hands[i]);
        }
	if(countRock==players.length ||
	   countScissors==players.length ||
	   countPaper==players.length ||
	   countRock*countScissors*countPaper!=0){ // Draw
            for(int i=0; i<players.length; i++){
                players[i].notify(Result.DRAW);
            }
	}else if (countRock==0){     // Scissors Win
            notifyAll(Hand.SCISSORS,Hand.PAPER);
        }else if(countScissors==0){  // Paper Win
            notifyAll(Hand.PAPER,Hand.ROCK);
        }else if(countPaper==0){     // Rock Win
            notifyAll(Hand.ROCK,Hand.SCISSORS);
        }else{
	    System.err.println("Please email to takigawa if you see this message.");
        }
        System.out.println("R:"+countRock+" S:"+countScissors+" P:"+countPaper);
        System.out.println();
    }
    public static void main(String[] args) {
        try{
            int num = Integer.parseInt(args[0]);
            JankenPlayer[] players = new JankenPlayer[3];
            players[0] = new RandomJankenPlayer("Yamada");
            players[1] = new JankenPlayerTypeA("Suzuki");
            players[2] = new JankenPlayerTypeB("Tanaka");
            Judge2 judge = new Judge2("Sato");
            judge.setPlayers(players);
            for(int i=0; i<num; i++){
                judge.play();
            }
            for(int j=0; j<players.length; j++){
                players[j].report();
            }
        }catch(Exception e){
            System.out.println("this requires an integer argument.");
            e.printStackTrace();
        }
    }
}
