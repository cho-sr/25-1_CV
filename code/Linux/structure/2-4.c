#include <stdio.h>
#include <string.h>

typedef struct {
    char name[20]; 
    int korean,english;
} stud;

char *search( stud *aa, char *name){
    int i;
    if (strstr(aa->name,name)){
        return aa->name;
    }
    else{
        return NULL;
    }
}

int main()
{
    stud student[3] = { {"HongGilDong",95,83},{"ParkGiDong",90,78},
    {"KimGiYong",79,77}};
    char *name = "Dong",i;
    char *result;

    for (i=0; i<3;i++)
    {
        result = search(&student[i],name);
        if(result){
            printf("%s\n",result);
        }
    }
}