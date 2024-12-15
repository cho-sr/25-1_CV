#include <stdio.h>
int main()
{
  float i =1.0, j=-1.75, *a;
  a = &i;
  printf("%p, %p\n",&i,a);
  printf("%f\n",*a);
  a = &j;
  printf("%p, %p\n",&j,a);
  printf("%f\n",*a);
  
}

