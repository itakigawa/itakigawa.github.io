public class TestCalendar1 {
    public static void main(String[] args) {
        java.util.Calendar c = java.util.Calendar.getInstance();
        java.text.SimpleDateFormat sdf = new java.text.SimpleDateFormat("yyyy/MM/dd (E)");
        System.out.println(sdf.format(c.getTime()));
    }
}
