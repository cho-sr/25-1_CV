#include <stdio.h>
#include <stdlib.h>
#include <time.h>
int main()
{
int *array, *ap, i, *max, *min;
array = (int *)malloc(400);
srand(time(NULL));

ap = array;
max = min = ap;
ap++;
for(i=1; i<100; i++, ap++){
    if(*ap > *max)
    max = ap;
    if(*ap < *min)
    min = ap;
}
printf("\nmax = %d, min = %d\n", *max, *min);
free(array);
}