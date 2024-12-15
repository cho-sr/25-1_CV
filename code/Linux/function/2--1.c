#include <stdio.h>
void add_xy(int,int,int);
void add_xy_p(int,int,int*);

int main()
{
    int x,y,total;
    x = 10;
    y =5 ;
    total = 0;
    printf("total in main = %d\n\n", total);
    add_xy(x,y,total);
    printf("total in main = %d\n\n", total);
    add_xy_p(x,y,&total);
    printf("total in main = %d\n\n", total);
}

void add_xy(int x, int y, int total)
{
    total = x + y;
    printf("total in add_xy = %d\n\n", total);

}

void add_xy_p(int x, int y, int * total)
{
    *total = x + y;
        printf("total in add_xy_p = %d\n\n", total);

}