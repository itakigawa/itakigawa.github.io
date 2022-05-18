import java.io.*;
import java.nio.file.*;

public class TestFileIO2 {
    public static void main(String[] args) {
        try{
            Path src = Paths.get("file.txt");
            BufferedReader br = Files.newBufferedReader(src);
            String line;
            while((line = br.readLine()) != null){
                System.out.println("READ: "+line);
            }
        }catch(IOException e){
            System.out.println("ERROR");
        }
    }
}
