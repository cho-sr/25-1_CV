#include <stdio.h>
#include <stdlib.h>
#include <time.h>
int main()
{
    int *array, *ap, i, *max, *min;
    array = (int *)malloc(400); //정수 100개
    srand(time(NULL));

    ap = array;
    for (i=0; i<100; i++, ap++){ // 포인터를 증가하면서
        printf("%d ", *ap = rand()); //100개의 난수 발생
    }

    ap = array;// ap포인터가 증가했기 떄문에 청므 주소값으로 대입
    max = min = ap; //max,min 초기값을 배열의 첫 요소로 설정
    ap++; // 포인터 증가
    for (i=1; i<100; i++,ap++){ // 1번 요소부터 끝까지 비교
        if(*ap > *max){
            max = ap; // max포인터 대입
        }
        if(*ap < *min){
            min = ap; //min 포인터 대입
        }
    }
    printf("\nmax = %d, min = %d\n",*max, *min);
    free(array); //할당된 메모리 해제
}