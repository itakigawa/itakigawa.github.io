public class Judge {
    private String name;
    private RandomJankenPlayer player1, player2;
    public Judge(String _name){
        name = _name;
    }
    public void setPlayers(RandomJankenPlayer _player1,
                           RandomJankenPlayer _player2){
        // ここを埋める
    }
    public void play(int n){
        Hand hand1, hand2;
        int win1=0, win2=0;
        int lose1=0, lose2=0;
        int draw1=0, draw2=0;

        // ここを埋める

        System.out.println("Player1 : "+player1.name);
        System.out.println("Player2 : "+player2.name);
        System.out.println("Judge   : "+name);
        System.out.println();
        System.out.println("Results: "+n+" games");
        System.out.println(player1.name+" "+win1+" win, "+lose1+" lose, "+draw1+" draw");
        System.out.println(player2.name+" "+win2+" win, "+lose2+" lose, "+draw2+" draw");
    }
    public static void main(String[] args) {
        try{
            int num = Integer.parseInt(args[0]);
            RandomJankenPlayer player1 = new RandomJankenPlayer("Yamada");
            RandomJankenPlayer player2 = new RandomJankenPlayer("Suzuki");
            Judge judge = new Judge("Sato");
            judge.setPlayers(player1,player2);
            judge.play(num);
        }catch(Exception e){
            System.out.println("this requires an integer argument.");
        }
    }
}
