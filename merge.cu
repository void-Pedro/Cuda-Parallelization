%%writefile sort1.cu
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define MAX_STRING_SIZE 8 // 7 caracteres + terminador nulo

__device__ int compare(const char *str1, const char *str2, int size) {
    for (int i = 0; i < size; ++i)
    {
        char c1 = str1[i];
        char c2 = str2[i];

        if (c1 < c2)
            return -1;
        else if (c1 > c2)
            return 1;
    }
    return 0;
}

__device__ void merge(char *data, char *temp, int left, int mid, int right, int size) {
    int i = left;
    int j = mid + 1;
    int k = left;

    while (i <= mid && j <= right) {
        if (compare(data + i * size, data + j * size, size) <= 0) {
            memcpy(temp + k * size, data + i * size, size);
            i++;
        }
        else {
            memcpy(temp + k * size, data + j * size, size);
            j++;
        }
        k++;
    }

    while (i <= mid) {
        memcpy(temp + k * size, data + i * size, size);
        i++;
        k++;
    }

    while (j <= right) {
        memcpy(temp + k * size, data + j * size, size);
        j++;
        k++;
    }

    //for (int x = left; x <= right; x++)
    //    memcpy(data + x * size, temp + x * size, size);
}

__global__ void mergeKernel(char* arr, char* aux, int tamAtual, int num_items, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int low = idx * width;

    // evitar índices inválidos
    if(low >= num_items - tamAtual || low < 0)
      return;

    int mid = low + tamAtual - 1;
    int high = min(low + width - 1, num_items - 1); // evitar que o high seja maior que o limite superior do vetor

    merge(arr, aux, low, mid, high, MAX_STRING_SIZE);
}

int main(void) {
    int num_items = 0;
    char *h_data = NULL;
    char *d_data = NULL;
    char *auxArr = NULL;

    FILE *file = fopen("quicksort.in", "r");
    if (!file) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    fscanf(file, "%d", &num_items);

    h_data = (char *)malloc(num_items * MAX_STRING_SIZE);
    if (!h_data) {
        perror("Memory allocation error");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < num_items; i++)
        fscanf(file, "%s", h_data + i * MAX_STRING_SIZE);
    fclose(file);

    cudaMalloc((void **)&d_data, num_items * MAX_STRING_SIZE);
    cudaMalloc((void **)&auxArr, num_items * MAX_STRING_SIZE);
    cudaMemcpy(d_data, h_data, num_items * MAX_STRING_SIZE, cudaMemcpyHostToDevice);

    for(int tamAtual = 1; tamAtual < num_items; tamAtual *= 2) {
      int width = tamAtual*2;
      int numSorts = (num_items + width - 1)/width;

      int threadsPerBlock = 64;
      if(numSorts < 32) {
        threadsPerBlock = 2;
      }
      int blocksPerGrid = (numSorts + threadsPerBlock - 1) / threadsPerBlock;
      //printf("numSorts: %d, blocos: %d\n", numSorts, blocksPerGrid);

      //cudaMemcpy(auxArr, d_data, num_items * MAX_STRING_SIZE, cudaMemcpyHostToDevice);
      mergeKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, auxArr, tamAtual, num_items, width);

      //troca de ponteiros
      char* tmp = d_data;
      d_data = auxArr;
      auxArr = tmp;

      //cudaDeviceSynchronize();
    }

    char *results_h = (char *)malloc(num_items * MAX_STRING_SIZE);
    cudaMemcpy(results_h, d_data, num_items * MAX_STRING_SIZE, cudaMemcpyDeviceToHost);

    FILE *out_file = fopen("quicksort.out", "w");
    if (!out_file) {
        perror("Error opening output file");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < num_items; i++)
        fprintf(out_file, "%s\n", results_h + i * MAX_STRING_SIZE);

    fclose(out_file);

    free(h_data);
    free(results_h);
    cudaFree(d_data);
    cudaFree(auxArr);
    cudaDeviceReset();
    exit(EXIT_SUCCESS);
}