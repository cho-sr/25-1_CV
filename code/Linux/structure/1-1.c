#include <stdio.h>

struct book {//책의 정보를 저장할 구조체 선언
    char title[50]; // 책 제목
    char author[35]; // 저자
    int pages; // 페이지 수
    int price; // 가격
};

int main()
{
    struct book book1; // 구조체 변수 선언
    printf(" book title(string) :");
    fgets(book1.title, sizeof(book1.title),stdin);
    printf(" author(string): ");
    fgets(book1.author, sizeof(book1.title),stdin);
    printf(" all pages(int) : ");
    scanf("%d",&book1.pages);// 책 페이지수 입력
    printf(" price(int) : ");
    scanf("%d", &book1.price); // 책값 입력
    //입력 받은 데이터 출력, 구조체의 멤버변수 액세스하여 출력
    printf("%s %s %dp, %dwon\n",book1.title, book1.author,book1.pages, book1.price);
}