#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/opencv.hpp>
#include <exception>
int main(int argc, char **argv)
{
	try
	{
		cv::Mat image = cv::imread("image.jpg");
		if (image.data == NULL)
		{
			std::cout << "NULL image\n";
			return 1;
		}
		cv::namedWindow("image", cv::WINDOW_NORMAL);
		cv::imshow("image", image);
		cv::waitKey(0);
	}
	catch (std::exception exc)
	{
		std::cout << exc.what();
	}
	return 0;
}