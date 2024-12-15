#include <stdio.h>
#include <string.h>
int main()
{
    char str[30] = "C programming language";
    char target[100],i;
    for (i=0; i<strlen(str);i++)
    *(target+i) = *(str+i);//*(str+i) = str[i]
    *(target+i) = '\0';
    printf("str: %s\n",str);
    printf("target: %s\n",target);
}