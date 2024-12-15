#include <stdio.h>

int sum_series(int);

int main()
{
    int x;
    printf("정수를 입력하시오 : ");
    scanf("%d",&x);
    printf("%d\n",sum_series(x));
}

int sum_series(int x){
    if(x==0)
        return 0;
    return (sum_series(x-1)+x);
}