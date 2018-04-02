#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>

#define E 0.00001

#define INSIDE(I, z, y, x) (x >= 0 && x < I.dx && y >= 0 && y < I.dy && z >= 0 && z < I.dz)

#define G(I, z, y, x)\
    *(I.data + (z)*I.dy*I.dx + (y)*I.dx + x) 

#define GS(I, z, y, x) \
    ((INSIDE(I, z, y, x)) ? *(I.data + (z)*I.dy*I.dx + (y)*I.dx + x) : 0.0)


typedef struct image {
	unsigned char *data;
	int dx, dy, dz;
    float sx, sy, sz;
} Image;

typedef struct d_image {
	double *data;
	int dx, dy, dz;
    float sx, sy, sz;
} Image_d;


Image perim(Image);

Image sum_bands(int, ...);

double calculate_H(Image_d, int, int, int);

void replicate(Image_d, Image_d);

Image_d smooth(Image, int);
