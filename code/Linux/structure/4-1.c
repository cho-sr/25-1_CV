#include <stdio.h>

typedef union int_or_float {
    int i;
    float f;
}number;

int main()
{
    number n;
    n.i = 1234;
    printf("i:%10d, f:%16.10e\n",n.i,n.f);
    n.f = 1234.0f;
    printf("i:%10d, f: %16.10e\n",n.i,n.f);
}