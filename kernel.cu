#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/opencv.hpp>
#include <exception>

//#define VERBOSE

cudaEvent_t start, stop, start1, stop1;
float elapsedTime;

#define TIME_EVENT(functionCall, text, start, stop)\
{cudaEventRecord(start, 0);\
functionCall;\
cudaEventRecord(stop, 0);\
cudaEventSynchronize(stop);\
cudaEventElapsedTime(&elapsedTime, start, stop);\
std::cout << text << elapsedTime<<'\n';\
}

#ifdef VERBOSE
#define VERBOSE_PRINT(code) code

#else
#define VERBOSE_PRINT(code)

#endif

#define MESSAGE_TOO_LARGE -1
#define MEMORY_ALLOCATION_ERROR -2
#define NULL_IMAGE_DATA -3
#define NULL_MESSAGE -4

//Steganography on the CPU
//@image - the image in which to hide the message, it will be modified on success
//@message - the message to be hidden
int CPU_basicSteganography(cv::Mat& image, std::string message)
{
	//Error checking
	if (!image.data)
	{
		return NULL_IMAGE_DATA;
	}
	if (message.length() == 0)
	{
		return NULL_MESSAGE;
	}
	if (message.length() * sizeof(message[0]) >= image.rows * image.cols * image.channels() * sizeof(image.data[0]))
	{
		return MESSAGE_TOO_LARGE;
	}

	//Steganography
	int pixelIndex = 0;
	int bitInPixel = 0;

	for (int i = 0; i <= message.length(); i++)
	{
		VERBOSE_PRINT(std::cout << "Hiding char: " << message[i] << '\n';)
		for (int j = 0; j < sizeof(message[i]) * 8; j++)
		{
			unsigned char isolatedBit = ((message[i] >> j) & 0x1);
			//Bits of 1 are hidden differently from bits of 0
			//std::cout << "j = " << j << '\n';
			VERBOSE_PRINT(std::cout<< "Current pixel : "<<(int)image.data[pixelIndex] << '\n';)
			if (isolatedBit)
			{
				//Hide a bit of 1
				VERBOSE_PRINT(std::cout << "Hiding bit of 1\n";);
				unsigned char bitMask = (0x01 << bitInPixel);
				image.data[pixelIndex] = (image.data[pixelIndex] | bitMask);
			}
			else
			{
				VERBOSE_PRINT(std::cout << "Hiding bit of 0\n";)
				//Hide a bit of 0
				unsigned char bitMask = ~(0x01 << bitInPixel);
				image.data[pixelIndex] = (image.data[pixelIndex] & bitMask);
			}
			VERBOSE_PRINT(std::cout << "Modified pixel: " << (int)image.data[pixelIndex] << '\n';)

			//Advance in the image
			pixelIndex++;
			if (pixelIndex == image.cols * image.rows * image.channels())
			{
				//Hit end of image, go to the next LSB and reset the pixel
				pixelIndex = 0;
				bitInPixel++;
			}
		}
		VERBOSE_PRINT(std::cout << '\n';)
	}
	return 0;
}

//Basic steganography kernel
//Every thread hides one bit from the message in one pixel
//@*image - pointer to the image data
//@*message - pointer to the message to be hidden
//@bit - bit in pixel where the message bit will be hidden
//@messageLength - the length of the @message
__global__ void cuda_basicSteganography(unsigned char* image, const unsigned char* message, unsigned char bit, int messageLength)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int byteInMessage = index / 8;
	if (byteInMessage < messageLength)
	{
		int bitInByte = index % 8;
		unsigned char isolatedBit = (message[byteInMessage] & (1 << bitInByte));
		unsigned char bitMask = isolatedBit ? (0x01 << bit) : ~(0x01 << bit);
		image[index] = isolatedBit ? (image[index] | bitMask) : (image[index] & bitMask);
		/*if (isolatedBit)
		{
			//bit is 1
			unsigned char bitMask = (0x01 << bit);
			image[index] = (image[index] | bitMask);
		}
		else
		{
			//bit is 0
			unsigned char bitMask = ~(0x01 << bit);
			image[index] = (image[index] & bitMask);
		}*/
	}
}

__global__ void testKernel()
{

}

// A function that does a basic steganography
// Hides a message in an image starting from the top left corner and going in a type writer manner
// It first hides in the Blue component
// If that is not enough, it hides in the Green component
// Lastly, it hides in the Red component
// It starts hiding from the LSB and goes up from there
// If there aren't enough bits to hide the message, a negative value is returned
// @image - image in which the message will be hidden
// @message - message to be hidden
// return value - the maximum number of bits used from each pixel
//				- a negative value if the message could not be hidden
int basicSteganography(cv::Mat& image, std::string message)
{
	int bitsUsed = 0;	//return value
	cudaError status;

	int imageNoOfBytes = image.cols * image.rows * image.channels();
	//Check if the message can be hidden in the image
	if (message.length() > (size_t)imageNoOfBytes)
	{
		return MESSAGE_TOO_LARGE;
	}
	
	cudaDeviceProp prop;
	status = cudaGetDeviceProperties(&prop, 0);
	if (status != cudaSuccess)
	{
		std::cout << "Error on getting device properties\n";
		goto Error;
	}
	std::cout << prop.maxThreadsPerBlock << '\n';
	int messageNoOfBits = message.length() * 8;
	bitsUsed = messageNoOfBits / imageNoOfBytes + (messageNoOfBits % imageNoOfBytes ? 1 : 0);
	
	unsigned char *d_message;
	unsigned char *d_image;

	status = cudaMalloc(&d_message, message.length() + 1);
	if (status != cudaSuccess)
	{
		std::cout << "Error on device message memory allocation\n";
		goto Error;
	}

	status = cudaMalloc(&d_image, imageNoOfBytes);
	if (status != cudaSuccess)
	{
		std::cout << "Error on device image memory allocation\n";
		goto Error;
	}

	status = cudaMemcpy(d_image, image.data, imageNoOfBytes, cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
	{
		std::cout << "Error on transfering the image to the device\n";
		goto Error;
	}

	status = cudaMemcpy(d_message, message.data(), message.length() + 1, cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
	{
		std::cout << "Error on transfering the message to the device\n";
		goto Error;
	}

	int noOfBlocks = (message.length() + 1) * 8 / prop.maxThreadsPerBlock + 1;

	//Done preparing the field for the GPU
	//Start calling kernels
	
	for (int currentBit = 0; currentBit < bitsUsed; currentBit++)
	{
		//cudaEventRecord(start1, 0);
		cuda_basicSteganography << <noOfBlocks, prop.maxThreadsPerBlock >> > (d_image, d_message, currentBit, message.length() + 1);
		status = cudaDeviceSynchronize();

		//cudaEventRecord(stop1, 0);
		//cudaEventSynchronize(stop1);

		//cudaEventElapsedTime(&elapsedTime, start1, stop1);
		//std::cout << "Elapsed time for kernels: " << elapsedTime << '\n';
		if (status != cudaSuccess)
		{
			std::cout << "Error on synchronizing the device\n";
			goto Error;
		}
	}
	
	status = cudaMemcpy(image.data, d_image, imageNoOfBytes, cudaMemcpyDeviceToHost);
	if (status != cudaSuccess)
	{
		std::cout << "Error on transfering the result to the host\n";
		goto Error;
	}

	return bitsUsed;

Error:
	cudaFree(d_image);
	cudaFree(d_message);
	return MEMORY_ALLOCATION_ERROR;
}

int reverseSteganography(cv::Mat image, std::string& decodedMessage)
{
	unsigned char currentChr = 0;
	int bitInCurrentChr = 0;
	unsigned char pixelBitMask = 0x01;
	for (int i = 0; ; i++, bitInCurrentChr = ((bitInCurrentChr + 1) % 8))
	{
		currentChr = (currentChr | ((image.data[i] & pixelBitMask) << bitInCurrentChr));
		VERBOSE_PRINT(std::cout << "Bit found: " << ((int)image.data[i] & pixelBitMask)<<'\n';)
		if (bitInCurrentChr == 7)
		{
			VERBOSE_PRINT(std::cout << currentChr << '\n';)
			decodedMessage.append(1, currentChr);
			if (currentChr == 0)
				break;
			currentChr = 0;
		}
		if (i == image.cols * image.rows * image.channels())
		{
			i = -1;
			pixelBitMask <<= 1;
		}
	}
	VERBOSE_PRINT('\n');
	return 0;
}

int main(int argc, char **argv)
{
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&start1);
	cudaEventCreate(&stop1);

	try
	{
		cv::Mat image = cv::imread("C:/Users/Sergiu/Desktop/cuda_steganography/x64/Debug/image.jpg");
		cv::Mat imageCopy = cv::imread("C:/Users/Sergiu/Desktop/cuda_steganography/x64/Debug/image.jpg");
		if (image.data == NULL)
		{
			std::cout << "NULL image\n";
			return 1;
		}
		/*for (int i = 0; i < 16; i++)
		{
			std::cout << (int)image.data[i] << ' ';
		}
		std::cout << '\n';*/
		std::string strToCode = "Am reusit oare sa fac sa mearga aceasta varianta super basic de steganografie? Am sa fac acest mesaj mult mai lung ca sa pot verifica mai bine cat de rapid ascunde GPU-ul. Ca momentan mesajul este minuscul.\
Oare de ce e asa de greu de gasit avantajele unui GPU? Pana acum mereu a fost mai rapid CPU-ul. Which is weird. Hai sa dau copy paste la asta. Hai sa dau copy paste la asta. Hai sa dau copy paste la asta. Hai sa dau copy paste la asta. Hai sa dau copy paste la asta. \
Hai sa dau copy paste la asta. Hai sa dau copy paste la asta. Hai sa dau copy paste la asta. Hai sa dau copy paste la asta. Hai sa dau copy paste la asta. Hai sa dau copy paste la asta. Hai sa dau copy paste la asta. Hai sa dau copy paste la asta. Hai sa dau copy paste la asta. Hai sa dau copy paste la asta. Hai sa dau copy paste la asta. Hai sa dau copy paste la asta. Hai sa dau copy paste la asta. \
Hai sa dau copy paste la asta. Hai sa dau copy paste la asta. Hai sa dau copy paste la asta. Hai sa dau copy paste la asta. Hai sa dau copy paste la asta. Hai sa dau copy paste la asta. Hai sa dau copy paste la asta. Hai sa dau copy paste la asta. Hai sa dau copy paste la asta. Hai sa dau copy paste la asta. Hai sa dau copy paste la asta. Hai sa dau copy paste la asta. Hai sa dau copy paste la asta. Hai sa dau copy paste la asta. Hai sa dau copy paste la asta. Hai sa dau copy paste la asta. Hai sa dau copy paste la asta. \
Hai sa dau copy paste la asta. Hai sa dau copy paste la asta. Hai sa dau copy paste la asta. Hai sa dau copy paste la asta. Hai sa dau copy paste la asta. Hai sa dau copy paste la asta. Hai sa dau copy paste la asta. Hai sa dau copy paste la asta. Hai sa dau copy paste la asta. Hai sa dau copy paste la asta. Hai sa dau copy paste la asta. Hai sa dau copy paste la asta. Hai sa dau copy paste la asta. Hai sa dau copy paste la asta. \
Hai sa dau copy paste la asta. Hai sa dau copy paste la asta. Hai sa dau copy paste la asta. Hai sa dau copy paste la asta. Hai sa dau copy paste la asta. Hai sa dau copy paste la asta. Hai sa dau copy paste la asta. Hai sa dau copy paste la asta. Hai sa dau copy paste la asta. Hai sa dau copy paste la asta. Hai sa dau copy paste la asta. Hai sa dau copy paste la asta. Hai sa dau copy paste la asta. Hai sa dau copy paste la asta.";

		std::cout << strToCode.length() << '\n';
		TIME_EVENT(basicSteganography(image, strToCode), "Elapsed time for GPU: ", start, stop);

		TIME_EVENT(CPU_basicSteganography(imageCopy, strToCode), "Elapsed time for CPU: ", start, stop);
		for (int i = 0; i < 16; i++)
		{
			VERBOSE_PRINT(std::cout << ((int)imageCopy.data[i] & 0x01) << ' ';)
		}
		VERBOSE_PRINT(std::cout << '\n';)

		std::string str;
		std::string str2;
		reverseSteganography(image, str);
		reverseSteganography(imageCopy, str2);

		std::cout << "Decode from CUDA: " << str << '\n';
		std::cout<< "Decode from CPU: " << str2 << '\n';
		cv::namedWindow("CUDAimage", cv::WINDOW_NORMAL);
		cv::imshow("CUDAimage", image);
		cv::namedWindow("CPU_image", cv::WINDOW_NORMAL);
		cv::imshow("CPU_image", imageCopy);
		cv::waitKey(0);
	}
	catch (std::exception exc)
	{
		std::cout << exc.what();
	}
	catch (...)
	{
		std::cout << "Unknown error was thrown\n";
	}
	return 0;
}