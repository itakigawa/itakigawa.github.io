public class Test2 {
    public static void main(String[] s){
        C[] obj = new C[2];

        obj[0] = new C();
        obj[1] = new C();

        for(int i=0; i<7; i++)
            obj[0].call();

        obj[0].print();
        obj[0].lost("changed");

        for(int i=0; i<3; i++)
            obj[0].call();

        obj[0].print();
        obj[1].print();
    }
}
