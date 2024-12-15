#include <stdio.h>
#include <stdlib.h>

struct student{
    char *name;
    int korean,english;
};

int main()
{
    struct student *sp;
    sp = (struct student*)malloc(sizeof(struct student));
    sp -> name = "HongGilDong";
    sp -> korean = 90;
    sp -> english = 95;
    printf("name: %s, korean: %d, english: %d\n", sp -> name, sp -> korean,
    sp -> english);
    free(sp);
}