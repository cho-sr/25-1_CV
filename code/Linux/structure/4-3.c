#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct Flower {
    char name[30];
    enum {red,white,yellow,pink,etc} color;
};

struct Fruit{
    char name[30];
    int calories;
};

struct Vegetable {
    char name[30];
    int calories;
    int cooking_time;
};

union flower_fruit_vegetable {
    struct Flower flower;
    struct Fruit fruit;
    struct Vegetable vegetable;
};

typedef struct {
    char type;
    union flower_fruit_vegetable ffv;
}things;

enum type {flower, fruit, vegetable};
const char *color[] = {"red","white","yellow","pink","etc"};

int main()
{
    int i;
    things x[3];

    x[0].type = flower;
    strcpy(x[0].ffv.flower.name,"Sunflower");
    x[0].ffv.flower.color = yellow;

    x[1].type = fruit;
    strcpy(x[1].ffv.fruit.name,"Apple");
    x[1].ffv.fruit.calories = 52;

    x[2].type = vegetable;
    strcpy(x[2].ffv.vegetable.name,"Carrot");
    x[2].ffv.vegetable.calories = 41;
    x[2].ffv.vegetable.cooking_time = 5;

    for (i=0; i<3; i++){
        if(x[i].type == flower){
            printf("%s, %s\n", x[i].ffv.flower.name,color[x[i].ffv.flower.color]);
        }
        else if(x[i].type == fruit){
            printf("%s, %dKcal\n",x[i].ffv.fruit.name,
            x[i].ffv.fruit.calories);
        }
        else if(x[i].type == vegetable){
            printf("%s, %dKcal, %dMin\n",
            x[i].ffv.vegetable.name,
            x[i].ffv.vegetable.calories,
            x[i].ffv.vegetable.cooking_time);
        }
        
    }
}