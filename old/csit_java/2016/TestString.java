public class TestString {
    public static void main(String[] args) {

        System.out.println(String.valueOf(-3.45));

        String str1 = "テストです";
        String str2 = "this test";
        String str3 = "this TEST";
        
        System.out.println(str1);
        System.out.println(str1.charAt(3));
        System.out.println(str1.length());
        System.out.println(str2.charAt(5));
        System.out.println(str3.charAt(5));
        System.out.println(str2.toUpperCase());

        System.out.println(str3.indexOf("EST"));
        System.out.println(str3.indexOf("abc"));

        if(str2.equals(str3)){
            System.out.println("case-A str2:"+str2+" str3:"+str3);
        }else if(str2.equalsIgnoreCase(str3)){
            System.out.println("case-B str2:"+str2+" str3:"+str3);
        }else{
            System.out.println("case-C str2:"+str2+" str3:"+str3);
        }

        // 数字を表す文字列の型への変換
        String str4 = "12345";
        String str5 = "-1.25";

        int i = Integer.parseInt(str4);
        double d = Double.parseDouble(str5);
        System.out.printf("int %d, double %f\n",i,d);

        // 文字列配列の分割
        String str = "one,two,,three,four,,five";
        String[] tokens = str.split(",");
        for(String s : tokens){
            System.out.println(s);
        }
    }
}
