#include "smooth.h"


Image perim(Image image){
	Image out;
	int dx, dy, dz;
	int x,y,z,x_,y_,z_;
	dx = image.dx;
	dy = image.dy;
	dz = image.dz;
	out.dx = image.dx;
	out.dy = image.dy;
	out.dz = image.dz;
	out.data = (unsigned char *)malloc(dx*dy*dz*sizeof(unsigned char));
	for(z = 0; z < dz; z++){
		for(y = 0; y < dy; y++){
			for(x = 0; x < dx; x++){
				*(out.data + z*out.dy*dx + y*out.dx + x)= 0;
				for(z_=z-1; z_ <= z+1; z_++){
					for(y_=y-1; y_ <= y+1; y_++){
						for(x_=x-1; x_ <= x+1; x_++){
							if ((x_ >= 0) && (x_ < dx) && (y_ >= 0) && (y_ < dy) && (z_ >= 0) && (z_ < dz) \
									&& (G(image, z, y, x) != G(image, z_, y_, x_))){
								G(out, z, y, x) = 1;
							}
						}
					}
				}
			}
		}
	}

	return out;
}

Image sum_bands(int n, ...){
	int x, y, z, i;
	Image out, aux;
	va_list bands;
	
	va_start(bands, n);
	aux = va_arg(bands, Image);
	
	out.dx = aux.dx;
	out.dy = aux.dy;
	out.dz = aux.dz;
	out.data = (unsigned char *)malloc(aux.dx*aux.dy*aux.dz*sizeof(unsigned char));
	
	for(z=0; z < out.dz; z++){
		for(y=0; y < out.dy; y++){
			for(x=0; x < out.dx; x++){
				G(out, z, y, x) = G(aux, z, y, x);
			}
		}
	}

	for(i=1; i < n; i++){
		aux = va_arg(bands, Image);
		for(z=0; z < out.dz; z++){
			for(y=0; y < out.dy; y++){
				for(x=0; x < out.dx; x++){
					G(out, z, y, x) += G(aux, z, y, x);
				}
			}
		}
	}

	va_end(bands);
	return out;
}

double calculate_H(Image_d I, int z, int y, int x){
    double fx, fy, fz, fxx, fyy, fzz, fxy, fxz, fyz, H;
    int h, k, l;

    h = 1;
    k = 1;
    l = 1;

    fx = (GS(I, z, y, x + h) - GS(I, z, y, x - h)) / (2.0*h);

    fy = (GS(I, z, y + k, x) - GS(I, z, y - k, x)) / (2.0*k);

    fz = (GS(I, z + l, y, x) - GS(I, z - l, y, x)) / (2.0*l);

    fxx = (GS(I, z, y, x + h) - 2*GS(I, z, y, x) + GS(I, z, y, x - h)) / (h*h);

    fyy = (GS(I, z, y + k, x) - 2*GS(I, z, y, x) + GS(I, z, y - k, x)) / (k*k);

    fzz = (GS(I, z + l, y, x) - 2*GS(I, z, y, x) + GS(I, z - l, y, x)) / (l*l);

    fxy = (GS(I, z, y + k, x + h) - GS(I, z, y - k, x + h) \
            - GS(I, z, y + k, x - h) + GS(I, z, y - k, x - h)) \
            / (4.0*h*k);

    fxz = (GS(I, z + l, y, x + h) - GS(I, z + l, y, x - h) \
            - GS(I, z - l, y, x + h) + GS(I, z - l, y, x - h)) \
            / (4.0*h*l);

    fyz = (GS(I, z + l, y + k, x) - GS(I, z + l, y - k, x) \
            - GS(I, z - l, y + k, x) + GS(I, z - l, y - k, x)) \
            / (4.0*k*l);

    if (fx*fx + fy*fy + fz*fz > 0)
        H = ((fy*fy + fz*fz)*fxx + (fx*fx + fz*fz)*fyy \
                + (fx*fx + fy*fy)*fzz - 2*(fx*fy*fxy \
                + fx*fz*fxz + fy*fz*fyz)) \
                / (fx*fx + fy*fy + fz*fz);
	else
        H = 0.0;


    return H;
}

void replicate(Image_d source, Image_d dest){
	int x, y, z;
	for(z=0; z < source.dz; z++)
		for(y=0; y < source.dy; y++)
			for(x=0; x < source.dx; x++)
				G(dest, z, y, x) = G(source, z, y, x);
}

Image_d smooth(Image image, int n){
	int i, x, y, z, S;
	double H, diff=0, dt=1/6.0, v, cn;
	Image_d out, aux;

	Image A1 = perim(image);
	Image A2 = perim(A1);
	Image A3 = perim(A2);
	Image A4 = perim(A3);
	Image Band = sum_bands(4, A1, A2, A3, A4);
	free(A1.data);
	free(A2.data);
	free(A3.data);
	free(A4.data);

	out.data = (double *) malloc(image.dz*image.dy*image.dx*sizeof(double));
	out.dz = image.dz;
	out.dy = image.dy;
	out.dx = image.dx;
	aux.data = (double *) malloc(image.dz*image.dy*image.dx*sizeof(double));
	aux.dz = image.dz;
	aux.dy = image.dy;
	aux.dx = image.dx;
    
    out.sx = image.sx;
    out.sy = image.sy;
    out.sz = image.sz;

	S = 0;
	for(z=0; z < image.dz; z++){
		for(y=0; y < out.dy; y++){
			for(x=0; x < out.dx; x++){
				if (G(image, z, y, x))
					G(out, z, y, x) = 1;
				else
					G(out, z, y, x) = -1;
				if (G(Band, z, y, x))
					S += 1;
			}
		}
	}

	for(i=0; i < n; i++){
		replicate(out, aux);
		diff = 0.0;
		for(z=0; z < out.dz; z++){
			for(y=0; y < out.dy; y++){
				for(x=0; x < out.dx; x++){
					if (G(Band, z, y, x)){
						H = calculate_H(aux, z, y, x);
						v = G(aux, z, y, x) + dt*H;
						if(G(image, z, y, x)){
							G(out, z, y, x) = v > 0 ? v: 0;
						} else {
							G(out, z, y, x) = v < 0 ? v: 0;
						}
						diff += (G(out, z, y, x) - G(aux, z, y, x))*(G(out, z, y, x) - G(aux, z, y, x));
					}
				}
			}
		}
		cn = sqrt((1.0/S) * diff);
		printf("CN: %.28f - diff: %.28f\n", cn, diff);
		if (cn <= E)
			break;
	}
	return out;
}



