#include <stdio.h>

typedef struct{
    unsigned short Grade:4;
    unsigned short Class:5;
    unsigned short Number:7;
} student;

int main()
{
    unsigned char x;
    student s;
    printf("grade(1~12): ");
    scanf("%u", &x);
    s.Grade = x;

    printf("class(1~31): ");
    scanf("%u", &x);
    s.Class = x; // 비트 필드는 개별적으로 주소가 없으므로
                    //다른 변수에 읽어서 다시 비트필드에 저장
                    // 즉, 비트필드는 직접 scnaf에 사용할 수 없음

    printf("number(1~127): ");
    scanf("%u", &x);
    s.Number = x;

    printf("%ugrade %uclass %dnumber\n", s.Grade,s.Class, s.Number);
    s.Grade = 10, s.Class = 15, s.Number = 20;
    printf("%ugrade %uclass %dnumber\n", s.Grade,s.Class, s.Number);
    printf("struct sizeof: %dByte\n",sizeof(student));
}