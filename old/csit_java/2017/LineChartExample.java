import static com.googlecode.charts4j.Color.*;
import com.googlecode.charts4j.*;
import java.util.Arrays;

public class LineChartExample {
    public static void main(String[] args) {
        Plot plot = Plots.newPlot(Data.newData(0, 66.6, 33.3, 100));
        LineChart chart = GCharts.newLineChart(plot);
        chart.addHorizontalRangeMarker(33.3, 66.6, LIGHTBLUE);
        chart.setGrid(33.3, 33.3, 3, 3);
        chart.addXAxisLabels(AxisLabelsFactory.newAxisLabels(Arrays.asList("Peak","Valley"), Arrays.asList(33.3,66.6)));
        chart.addYAxisLabels(AxisLabelsFactory.newNumericAxisLabels(0,33.3,66.6,100));
        String url = chart.toURLString();
        System.out.println(url);
    }
}

