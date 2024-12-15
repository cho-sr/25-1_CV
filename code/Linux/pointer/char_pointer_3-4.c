#include <stdio.h>
int main()
{
    char str[30] = "C programming language";
    char target[100], *s, *t;
    s = str;
    t = target;
    while(*s)
    *t++ = *s++;
    *t = '\0';
    printf("str: %s\n", str);
    printf("target: %s\n", target);
}