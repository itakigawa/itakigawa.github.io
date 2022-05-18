// 要Java SE 8
import java.util.Calendar;
import java.text.SimpleDateFormat;
import java.time.LocalDate;
import java.time.Period;
import java.time.Month;
import java.time.temporal.ChronoUnit;

public class TestCalendar2 {
    public static void main(String[] args) {
	// (1) 今日の日付をきまった書式で出力
        Calendar c = Calendar.getInstance();
        SimpleDateFormat sdf = new SimpleDateFormat("yyyy/MM/dd (E)");
        System.out.println(sdf.format(c.getTime()));

	// (2) 今日、私が何歳何ヶ月何日か + 生まれてから何日生きたか を出力
        LocalDate today = LocalDate.now();
        LocalDate birthday = LocalDate.of(1977, 3, 5); // 1977年3月5日(瀧川の誕生日)
        Period p = Period.between(birthday, today);
        long p2 = ChronoUnit.DAYS.between(birthday, today);
        System.out.println("You are " + p.getYears() + " years, "
                           + p.getMonths() + " months, and " 
                           + p.getDays() +  " days old. (" 
                           + p2 + " days total)\n"+
                           + p2 * 24 * 60 * 60 + " seconds total.");
    }
}
