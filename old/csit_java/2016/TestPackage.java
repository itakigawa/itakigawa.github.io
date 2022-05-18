package org.mypackage.test;
public class TestPackage {
    public static void main(String[] args) {
        String str = "AA,BBB,C,DD,EEE,F,GGGGG";
        String[] strArray = str.split(",");
        for(String s : strArray){
            System.out.println(s);
        }
    }
}
