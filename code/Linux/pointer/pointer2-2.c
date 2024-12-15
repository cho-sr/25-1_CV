#include <stdio.h>
int main(){
    float x[10] = {4.5,7.7,8.9,3.8,2.9,5.8,9.3,8.4,7.3,6.6};
    float sum = 0.,*xp;
    int i;

    xp = &x[0];
    for (i=0;i<10;i++,xp++){
        printf("%p: %4.1f\n",xp,*xp);
        sum += *xp;
    }
    printf("sum = %f\n",sum);
}