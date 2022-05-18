#include<stdio.h>
int add(int x, int y){
  return x+y;
}
double add2(double x, double y){
  return x+y;
}
int main(){
  printf("%d\n",add(12,29));
  printf("%f\n",add2(12.80,29.12));
}
