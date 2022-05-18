import java.io.*;
import java.util.Scanner;

public class TestFileIO1 {
    public static void main(String[] args) {
        try{
            Scanner scanner = new Scanner(new File("file.txt"));
            while (scanner.hasNextLine()) {
                String line = scanner.nextLine();
                System.out.println("READ: "+line);
            }
            scanner.close();
        }catch(FileNotFoundException e){
            System.err.println("ERROR");
        }
    }
}
