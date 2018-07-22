#include <stdio.h>
#include <time.h>
#include <opencv2\opencv.hpp>
#include "device_launch_parameters.h"
#include "cuda_runtime.h"


using namespace cv;
using namespace std;

__global__ void uchar2Float(uchar* input, float* output, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int tid = ((gridDim.x * blockDim.x) * idy) + idx;
	if(tid < size)
		output[tid] = float(input[tid]);
}

void main_cuda(Mat img, int size, float* cuda_src)
{
	uchar* dev_c;
	float* deviceSrc;
	cudaMalloc((void**)&dev_c, size * sizeof(uchar));
	cudaMalloc((void**)&deviceSrc, size * sizeof(float));
	cudaMemcpy(dev_c, img.data, size * sizeof(uchar), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceSrc, 0, size * sizeof(float), cudaMemcpyHostToDevice);
	
	dim3 dimGrid(50, 50);
	dim3 blockDim(50, 20);
	uchar2Float << <dimGrid, blockDim >> > (dev_c, deviceSrc, size);

	cudaMemcpy(cuda_src, deviceSrc, size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(dev_c);
	cudaFree(deviceSrc);
}

void main_cpu(Mat img, int height, int width, int channels, int size, float* hostSrc)
{
	for (int k = 0; k < channels; k++)
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				hostSrc[k * height * width + i*width + j] = float(img.data[k * height * width + i * width + j]);
}

int main()
{
	Mat img = imread("lena.jpg");
	imshow("lena.jpg", img);
	int width, height, channels;
	width = img.size().width;
	height = img.size().height;
	channels = img.channels();
	printf("width = %d, height = %d, channels = %d \n", width, height, channels);
	int size = width * height * channels;

	float* cuda_src = new float[size];
	float* src = new float[size];
	
	clock_t start, finish;
	float costtime;

	start = clock();
	main_cuda(img, size, cuda_src);
	finish = clock();
	costtime = float(finish - start);
	printf("cuda_t = %.2f\n", costtime);

	start = clock();
	main_cpu(img, height, width, channels, size, src);
	finish = clock();
	costtime = float(finish - start);
	printf("cpu_t = %.2f\n", costtime);

	printf("src[1000] = %f, cuda_src[1000] = %f\n", src[2500], cuda_src[2500]);
	delete[] src;
	delete[] cuda_src;

	waitKey(0);
	return 0;
}