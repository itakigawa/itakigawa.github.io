public class Test5 {
    public static void main(String[] s){
        RandomJankenPlayer p1 = new RandomJankenPlayer(); // 「負けグー」
        RandomJankenPlayer p2 = new RandomJankenPlayer(); // 普通のランダム

        Hand h1, h2;

        for(int i=0; i<10; i++){
            h1 = p1.showHand();
            h2 = p2.showHand();
            System.out.println(h1+" vs "+h2);
            if( (h1==Hand.ROCK && h2==Hand.PAPER) ||
                (h1==Hand.PAPER && h2==Hand.SCISSORS) ||
                (h1==Hand.SCISSORS && h2==Hand.ROCK) ){
                p1.lost(); // p1に負けたことを通知
            }
        }
    }
}
