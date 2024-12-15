#include <stdio.h>
double circle(double, double);

int main()
{
    double base = 15.4, height = 20.8;
    printf("area= %f\n", circle(base,height));
}

double circle(double n1, double n2){
    return (n1 * n2 / 2.0);
}