#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>

void StrUprFirstChar(char *);

int main()
{
    char *str;
    str =(char *)malloc(400);
    strcpy(str,"Char starting_name[size];");
    printf("%s\n",str);
    StrUprFirstChar(str);
    printf("%s\n",str);
    free(str);
}
void StrUprFirstChar(char *s){
    int i,in_str = 0;
    while(*s){
        if (!in_str && isalnum(*s)){
            if(islower(*s))
                *s = toupper(*s);
            in_str = 1;
        }
        else if (isspace(*s) || ispunct(*s))
            in_str = 0;
        s++;
    }
}