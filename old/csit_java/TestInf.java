public class TestInf implements NameAvailable {
    private String name;
    public String getName(){
        return this.name;
    }
    public TestInf(String name){
        this.name = name;
    }
    public static void main(String[] args) {
        TestInf t = new TestInf("hoge");
        System.out.println(t.getName());
    }
}
