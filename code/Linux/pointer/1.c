#include <stdio.h>

int main()
{
    int i = 10, j = 20, *a;
    a = &i;
    printf("i address : %p, a : %p\n",&i,a);
    printf("*a : %d\n",*a);
    a = &j;
    printf("j address : %p, a : %p\n",&j,a);
    printf("*a : %d\n",*a);
    
}