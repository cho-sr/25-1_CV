#include <string.h>
#include <stdio.h>
#include <stdlib.h>
int main()
{
    char *str,i,len,temp;
    str = (char *)malloc(100);
    strcpy(str, "This is example");
    printf("%s\n",str);

    len = strlen(str);
    for (i=0; i<len/2;i++){
        temp = *(str+i);
        *(str+i) = *(str+len-i-1);
    }
    printf("%s\n",str);
    free(str);
}