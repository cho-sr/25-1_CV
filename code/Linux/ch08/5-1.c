#include <stdio.h>
void print_series(int);

int main(){
    int x;
    printf("정수를 입력하시오 : ");
    scanf("%d",&x);
    print_series(x);
}

void print_series(int x){
    if(x==0)
        return;
    print_series(x-1);
    printf("%d ",x);
}