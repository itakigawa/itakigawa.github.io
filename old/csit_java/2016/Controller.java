public class Controller {
    public static void main(String[] args) {
        YesMan person1 = new YesMan();
        NoMan  person2 = new NoMan();
        
        String query1 = "Are you hungry?";
        String query2 = "Are you stupid?";

        System.out.println("Query:"+query1);
        person1.query(query1);
        person2.query(query1);

        System.out.println("Query:"+query2);
        person1.query(query2);
        person2.query(query2);
    }
}
