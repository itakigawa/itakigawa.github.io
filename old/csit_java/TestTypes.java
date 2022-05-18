public class TestTypes {
    public static void main(String[] args) {
	// 明示的な型-クラス変換
	int int1 = 15;
	Integer int2 = Integer.valueOf(int1);
	int int3 = int2.intValue();
	// オートボクシング
	int int4 = 16;
	Integer int5 = int4;
	int int6 = int5;
    }
}
