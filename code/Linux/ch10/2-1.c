#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

int main()
{
    int dan, cnt, col;
    FILE *fp;
    fp = fopen("gugudna.txt", "w");
    assert(fp);
    for (dan=2; dan<20; dan+=4){
        for(col=dan; col<dan+4;col++){
            fprintf(fp,"%7d  dan%7s",col,"");
        }
        fprintf(fp,"\n");
        for (cnt = 1; cnt < 20; cnt++)
        {
            for (col = dan; col < dan+4; col++)
            {
                fprintf(fp,"%3d * %3d = %3d ",col,cnt,col*cnt);
            }
            fprintf(fp,"\n");
        }
        fprintf(fp,"\n");
    }
    fclose(fp); 
}