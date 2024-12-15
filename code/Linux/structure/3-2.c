#include <stdio.h>

enum day {sun,mon,tue,wed,thu,fri, sat};
typedef enum day day;
char * day_name[] = {"일","월","화","수","목","금","토"};

day tomorrow(day d){
    return ((day)(((int)d+1)%7));
}

day yesterday(day d){
    return ((day)(((int)d+6)%7));
}

int main()
{
    day today=sun;
    printf("yesterday : %s\n",day_name[yesterday(today)]);
    printf("today : %s\n",day_name[(today)]);
    printf("tomorrow : %s ",day_name[tomorrow(today)]);
    
}