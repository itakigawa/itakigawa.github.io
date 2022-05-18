import java.util.HashMap;

public class TestHashMap {
    public static void main(String[] args) {
        HashMap<String, String> weekdays = new HashMap<String, String>();
        
        weekdays.put("Monday", "月曜日");
        weekdays.put("Tuesday", "火曜日");
        weekdays.put("Wednesday", "水曜日");
        weekdays.put("Thursday", "木曜日");
        weekdays.put("Friday", "金曜日");
        weekdays.put("Saturday", "土曜日");
        weekdays.put("Sunday", "日曜日");

        System.out.println(weekdays.get("Thursday"));
        System.out.println(weekdays.get("Sunday"));

        weekdays.remove("Wednesday");

        for(String key : weekdays.keySet()){
            System.out.println("key:"+key+"->value:"+weekdays.get(key));
        }
    }
}
