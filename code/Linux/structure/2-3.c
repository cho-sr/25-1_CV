#include <stdio.h>

struct student{
    char name[20];
    int korean, english;
};

void disp1(struct student aa){ //구조체 변수를 매개변수로 받음
    printf("%s %d %d\n", aa.name, aa.korean, aa.english);
}

void disp2(struct student *bb){ // 구조체 포인터를 매개변수로 받음
    printf("%s %d %d\n",bb -> name, bb -> korean, bb -> english);
}

int main()
{
    struct student hong ={"HongGilDong",90,80}; // 구조체 변수 선언

    disp1(hong); // 구조체 변수를 함수의 매개변수로 사용
    disp2(&hong); // 구조체 변수를 주소 (포인터)로 전달
}