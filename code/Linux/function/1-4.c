#include <stdio.h>
int exact(int);

int friendly(int, int);

int main()
{
    int i,j;
    for (i=1; i<1000; i++){
        for (j = i + 1; j < 1000; j++)
        {
            if(friendly(i,j)){
                printf("%d %d = friendly\n",i,j);
            }
        }
        
    }
    return 0;
}

int exact(int a)
{
    int i,sum =0;
    for (i=1; i <= a/2; i++){
        if (a % i == 0){
            sum += i;
        }
    }
    return sum;
}

int friendly(int a, int b)
{
    int sum1, sum2;
    sum1 = exact(a);
    sum2 = exact(b);
    if ( sum1 == b && sum2 == a){
        return 1;
    }
    else{
        return 0;
    }
}