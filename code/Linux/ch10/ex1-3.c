#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void)
{
    FILE *fp;
    char buf[256], t[5], info[5][100], *cp;
    int i;
    fp = fopen("books.txt","r");
    if (fp == NULL){
        printf("No File\n");
        return 0;
    }
    while (1)
    {
        fgets(buf, 256, fp);
        if (feof(fp)){
            break;
        }
        printf("%s\n",buf);
        cp = buf;
        for (i=0; i<5; i++){
            sscanf(cp,"%[^,]",info[i]);
            cp = strstr(cp, ", ")+2;
            printf("%s \n",info[i]);
        }
    }
    fclose(fp);
}