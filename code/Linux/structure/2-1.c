#include <stdio.h>

struct student
{
    char *name;
    int korean;
    int english;
};

int main()
{
    struct student kdhong;
    struct student *sp = &kdhong;
    sp -> name = "HongGilDong";
    sp -> korean = 90;
    (*sp).english = 95;// 연산자 우선순위 때문에 ()

    printf("name: %s\n", sp-> name);
    printf("korean : %d\n", sp->korean);
    printf("english : %d\n", sp -> english);
}
