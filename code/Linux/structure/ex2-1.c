#include <stdio.h>
typedef struct { // 구조체 정의
char name[20]; // 이름
int ID_no, grade; // 학번, 학년
char tel[20]; // 전화번호
char email[30]; // 이-메일
} student; // 새로운 데이터 형 student 선언
int main()
{
student mem[3]= { // 새로운 데이터 형을 이용하여 포인터 배열 mem 선언
{"Kim", 20072356, 3, "010-2345-6789", "kdhong@naver.com"},
{"Lee", 20081234, 3, "011-9999-7727", "sllee@gmail.com"},
{"Ha", 20091557, 1, "010-9977-3345", "duha@nate.com"}};
student *m = mem; // 구조체 포인터 변수 m에 배열 대입
int i;
for (i=0; i<3; i++, m++) { // 포인터를 이용하여 배열의 내용 출력
printf("%s %d %d %s %s\n", m->name, m->ID_no,
m->grade, m->tel, m->email);
}
printf("\n");
}