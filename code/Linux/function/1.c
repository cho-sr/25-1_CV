#include <stdio.h>

int divisor(int);

int main()
{
    int i;
    for(i=1; i<=20; i++){
        printf("%d = %d\n",i,divisor(i));
    }
}

int divisor(int n)
{
    int count =0,i;
    for (i=1; i<=n; i++){
        if(n%i == 0){
            count++;
        }
    }
    return count;
}