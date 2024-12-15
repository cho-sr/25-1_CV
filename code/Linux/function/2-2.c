#include <stdio.h>

void swap(int *,int *);

int main()
{
    int x=15,y=37;
    printf("%d %d\n",x, y);
    swap(&x,&y);
    printf("%d %d",x,y);
}

void swap(int * x, int * y)
{
    int temp;
    temp = *x;
    *x = *y;
    *y = temp;
}