#include <stdio.h>
#include <opencv2\opencv.hpp>
#include "device_launch_parameters.h"
#include "cuda_runtime.h"

using namespace cv;

/*
*******************************************
* 函数功能： 图像卷积
* 函数输入：float* src             输入图像矩阵
					   float* dst             卷积后图像
					   int rows                图像行数
					   int cols                 图像列数
					   float* kernel        卷积核
					   int kernel_size     卷积核尺寸大小
*******************************************
*/
__global__ void conv2d(float* src, float* dst, int rows, int cols, float* kernel, int kernel_size)
{
	int row = blockIdx.x;
	if (row < 1 || row > rows - 1)
		return;
	int col = blockIdx.y;
	if (col < 1 || col > cols - 1)
		return;

	int dstIndex = col * rows + row;
	dst[dstIndex] = 0;
	int kerIndex = kernel_size * kernel_size - 1;
	for (int kc = -kernel_size / 2; kc < kernel_size / 2 + 1; kc++) {
		int srcIndex = (col + kc) * rows + row;
		for (int kr = -kernel_size / 2; kr < kernel_size / 2 + 1; kr++)
			dst[dstIndex] += kernel[kerIndex--] * src[srcIndex + kr];
	}
}

int main()
{
	Mat img = imread("lena.jpg", IMREAD_GRAYSCALE);
	int width, height, channels;
	width = img.size().width;
	height = img.size().height;
	channels = img.channels();
	printf("width = %d, height = %d, channels = %d\n", width, height, channels);
	int size = width * height * channels;

	float* hostSrc = new float[size];
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			hostSrc[i*width + j] = float(img.data[i*width + j]);

	float* deviceSrc, *deviceDst, *deviceKer;
	float* hostDst = new float[size];
	float kernel[9] = { 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1 };
	int kernel_size = 3;
	cudaMalloc((void**)&deviceSrc, size * sizeof(float));
	cudaMalloc((void**)&deviceDst, size * sizeof(float));
	cudaMalloc((void**)&deviceKer, size * sizeof(float));
	cudaMemcpy(deviceSrc, hostSrc, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceDst, 0, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceKer, kernel, kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice);
	dim3 dimGrid(height, width);
	conv2d << <dimGrid, 1 >> > (deviceSrc, deviceDst, height, width, deviceKer, kernel_size);
	cudaMemcpy(hostDst, deviceDst, size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(deviceSrc);
	cudaFree(deviceDst);
	cudaFree(deviceKer);

	Mat img1(height, width, img.type());
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			img1.data[i*width + j] = uchar(hostDst[i * width + j]);
	
	imshow("lena.jpg", img);
	imshow("lena_conv2d.jpg", img1);
	waitKey(0);
	return 0;
}