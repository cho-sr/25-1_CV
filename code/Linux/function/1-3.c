#include <stdio.h>

int gcd(int, int);

int main()
{
    int i,j;
    printf("input : ");
    scanf("%d %d",&i,&j);
    printf("최대공약수는 = %d\n",gcd(i,j));
    printf("48,56 = %d\n",gcd(48,56));
}

int gcd(int a,int b){
    int z =1;
    while (z != 0)
    {
        z = a%b;
        a = b;
        b = z;
    }
    return a;
}