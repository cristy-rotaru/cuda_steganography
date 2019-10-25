#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/opencv.hpp>

int main(int argc, char **argv)
{
	cv::Mat image = cv::imread("image.jpg");
	cv::namedWindow("image", cv::WINDOW_NORMAL);
	cv::imshow("image", image);
	cv::waitKey(0);

	return 0;
}