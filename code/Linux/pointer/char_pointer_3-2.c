#include <stdio.h>
int main()
{
    char * str = "C language Programming",*s;
    s = str; //str을 그냥 써도 되지만 포인터가 변하면서 처음 위치를 모르게 될 수도 있다.
    while(*s)
    printf("%s\n",s++);
}
//str[]= "C language Programming" 는 *str = "C language Programming" 와 다름
// str[]는 포인터 상수가 되고, *str은 포인터 변수가 된다.