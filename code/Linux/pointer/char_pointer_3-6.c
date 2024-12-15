#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
int main()
{
    char * str,temp,in_str = 0;
    int i,len;
    str = (char *)malloc(400);
    strcpy(str, "Char string_name[size];");
    len = strlen(str);
    printf("%s\n",str);
    for (i=0;i<len;i++){
        if(!in_str && isalnum(*(str+i))){
            if(islower(*(str+i)))
                *(str+i) = toupper(*(str+i));
            in_str = 1;
        }
        else if (isspace(*(str+i)) || ispunct(*(str+i)))
            in_str = 0;
    }
    printf("%s\n",str);
    free(str);
}