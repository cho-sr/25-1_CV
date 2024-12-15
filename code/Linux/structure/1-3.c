#include <stdio.h>
#include <string.h>

struct member 
{
    char name[20];
    char gender;
    int age;
    char level[20];
};

int main()
{
    struct member members[5] = {{"hong",'M',49,"사장"},
    {"lee",'F',25,"사원"},{"lim",'M',39,"부장"},{"U",'F',28,"과장"},
    {"kim",'M',43,"이사"}};

    int i;
    char name[20];

    for (i=0;i<5;i++){
        if (members[i].gender == 'M' && members[i].age >= 40){
            printf("name : %s \n직위 : %s\n",members[i].name,members[i].level);
        }
    }
    printf("\n");
    printf("input name : ");
    scanf("%s",name);
    for (i=0; i<5; i++){
        if (strcmp(members[i].name,name) == 0){
            printf("name : %s \nlevel : %s\nage : %d\n",members[i].name,members[i].level,
            members[i].age);
        }
    }
}
