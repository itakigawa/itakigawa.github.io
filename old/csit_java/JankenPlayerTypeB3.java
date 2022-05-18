public static void main(String[] args) {
    try{
        int num = Integer.parseInt(args[0]);
        RandomJankenPlayer player1 = new RandomJankenPlayer("Yamada");
        RandomJankenPlayer player2 = new JankenPlayerTypeB("Yamamoto");
        Judge judge = new Judge("Sato");
        judge.setPlayers(player1,player2);
        judge.play(num);
    }catch(Exception e){
        System.out.println("this requires an integer argument.");
    }
}
