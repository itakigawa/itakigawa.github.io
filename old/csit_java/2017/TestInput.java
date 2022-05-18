import java.util.Scanner;
public class TestInput {
    public static void main(String[] args) {
        System.out.println("Input Your Name:");
        Scanner scanner = new Scanner(System.in);
        String name = scanner.nextLine();
        System.out.println("Input Your Age:");
        int age = scanner.nextInt();
        System.out.printf("Your Name=%s (Age=%d)\n",name,age);
        scanner.close();
    }
}
