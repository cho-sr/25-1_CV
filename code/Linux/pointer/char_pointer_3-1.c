#include <stdio.h>
int main(){
    char x[] = "string", y[] = "test",*b;
    b = x;
    printf("%p, %p, ", &x,b);
    printf("%s\n",b);
    b = y;
    printf("%p, %p, ", &y,b);
    printf("%s\n", b);
}