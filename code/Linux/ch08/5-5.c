#include <stdio.h>
void dec2bin(int);

int main()
{
    int data;
    printf("정수 입력: ");
    scanf("%d",&data);
    dec2bin(data);
}

void dec2bin(int x)
{
    if( !x)
        return;
    dec2bin(x/2);
    printf("%d\n",x%2);
}