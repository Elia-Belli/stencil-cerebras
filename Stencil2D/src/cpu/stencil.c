#include <stdio.h>
#include <stdlib.h>

int main()
{
  int M = 8;  // rows
  int N = 8;  // cols
  int iterations = 10;

  float* A = (float*) malloc(M*N *sizeof(float));
  float* A_aux = (float*) calloc(M*N, sizeof(float));
  
  float x = 0;
  for(int i = 0; i < M; i++){
    for(int j = 0; j < N; j++){
      A[i*N + j] = x;
      x++;
    }
  }

  for(int iter = 0; iter < iterations; iter++){
    for(int i = 0; i < M; i++){
      for(int j = 0; j < N; j++){
        if(i-1 >= 0)  A_aux[i*N + j] += A[(i-1)*N + j];
        if(i+1 < M)   A_aux[i*N + j] += A[(i+1)*N + j];
        if(j-1 >= 0)  A_aux[i*N + j] += A[i*N + j-1];
        A_aux[i*N + j] -= 4.0 * A[i*N + j];
        if(j+1 < N)   A_aux[i*N + j] += A[i*N + j+1];
      }
    }

    float* temp = A;
    A = A_aux;
    A_aux = temp;
  }

  FILE* fp = fopen("./logs/c_result.txt","w+");
  for(int i = 0; i < M; i++){
    for(int j = 0; j < N; j++){
      printf("%f ", A[i*N +j]);
      fprintf(fp, "%.1f \n", A[i*N +j]);
    }
    printf("\n");
  }
  fclose(fp);
}