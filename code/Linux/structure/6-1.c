#include <stdio.h>

typedef struct {
    unsigned int year: 12;
    unsigned int month: 4;
    unsigned int day: 5;
    unsigned int hour: 5;
    unsigned int min: 6;
} schedule;

int main()
{
    schedule my;
    my.year = 2001;
    my.month = 5;
    my.day = 28;
    my.hour = 12;
    my.min = 30;
    printf("%dyear %dmonth %dday %dhour %dmin\n",
    my.year,my.month,my.day,my.hour,my.min);
    printf("struct size: %dByte\n", sizeof(schedule));
}