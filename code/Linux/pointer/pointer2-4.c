#include <stdio.h>
#define ROW 3
#define COL 3
int main()
{
    float x[ROW][COL] = {1.1,2.2,3.3,4.4,5.5,6.6,7.7,8.8,9.9};
    float y[ROW][COL] = {9.1,8.2,7.3,6.4,5.5,4.6,3.7,2.8,1.9};
    float z[ROW][COL], *x1, *y1, *z1;
    int i,j;

    x1 = &x[0][0];
    y1 = (float *)y;
    z1 = (float *)z;
    for (i=0; i<ROW*COL;i++,x1++,y1++,z1++)
        *z1 = *x1 + *y1;
    z1 = (float *)z;
    for(i=0; i<ROW; i++){
        for (j=0;j<COL;j++)
            printf("%4.1f\n", *z1++);
        printf("\n");
    }
}