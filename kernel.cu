
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

__global__
void colorToGreyscaleConversion(unsigned char* Pout, unsigned char* Pin, int width, int height)
{
	int colIdx = blockIdx.x * blockDim.x + threadIdx.x;
	int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;
	if (colIdx < width && rowIdx < height)
	{
		int pixelIdx = rowIdx * width + colIdx;
		int rgbOffset = pixelIdx * 3; // 3 == CHANNELS
		unsigned char r = Pin[rgbOffset + 0];
		unsigned char g = Pin[rgbOffset + 1];
		unsigned char b = Pin[rgbOffset + 2];
		float greyScale = 0.21f * r + 0.71f * g + 0.07f * b;
		Pout[rgbOffset + 0] = (int)(greyScale);
		Pout[rgbOffset + 1] = (int)(greyScale);
		Pout[rgbOffset + 2] = (int)(greyScale);
	}
}

void GreyScaleConversion(unsigned char* Pout, unsigned char* Pin, int width, int height, int channel)
{
	unsigned char* d_Pin, *d_Pout;
	int size = width * height * channel * sizeof(unsigned char);
	
	cudaMalloc((void**)&d_Pin, size);
	cudaMemcpy(d_Pin, Pin, size, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&d_Pout, size);

	dim3 dimGrid(ceil(width / 16.0), ceil(height / 16.0), 1);
	dim3 dimBlock(16, 16, 1);

	colorToGreyscaleConversion <<<dimGrid, dimBlock>>> (d_Pout, d_Pin, width, height);

	cudaMemcpy(Pout, d_Pout, size, cudaMemcpyDeviceToHost);

	cudaFree(d_Pin);
	cudaFree(d_Pout);
}

int main()
{
	int w, h, n;
	unsigned char *data = stbi_load("rgba.png", &w, &h, &n, 0);
	unsigned char *oData = new unsigned char[w * h * n];

	GreyScaleConversion(oData, data, w, h, n);

	stbi_write_png("write.png", w, h, n, oData, 0);
	stbi_image_free(data);

    // Add vectors in parallel.
    // cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
	// cudaError_t cudaStatus;
    /*if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }*/


    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    /*cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }*/

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
/*
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}

*/
