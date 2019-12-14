#include "steganographyOnCuda.cuh"
#include <exception>
#include <fstream>

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
	try
	{
		cv::Mat image = cv::imread("D:/Documents/Faculty/PGPU/Tema/cuda_steganography/test_images/landscape02.jpg");

		if (image.data == NULL)
		{
			std::cout << "NULL image\n";
			return 1;
		}
		uint32_t maxStreamSize = analyzeImage_CPU(image.data, image.size().height, image.size().width);
		std::cout << "Max stream size: " << maxStreamSize << std::endl;

		startImageAnalisys_GPU(image.data, image.size().height, image.size().width);

		system("pause");

		std::ifstream fileToHide("D:/Music/Back in Time.mp3", std::ios_base::in | std::ios_base::binary);
		if (!fileToHide)
		{
			std::cout << "Can't open file" << std::endl;
			return 2;
		}
		// computing file size
		std::streampos sPos = fileToHide.tellg();
		fileToHide.seekg(0, std::ios::end);
		sPos = fileToHide.tellg() - sPos;
		fileToHide.seekg(0);
		uint32_t streamSize = sPos;

		if (maxStreamSize < streamSize + 8)
		{
			std::cout << "The selected file does not fit inside the image" << std::endl;
			return 3;
		}

		uint8_t* streamToHide = new uint8_t[streamSize + 8]; // add 8 for header
		streamToHide[0] = 's';
		streamToHide[1] = 't';
		streamToHide[2] = 'e';
		streamToHide[3] = 'g';
		streamToHide[4] = streamSize >> 24;
		streamToHide[5] = streamSize >> 16;
		streamToHide[6] = streamSize >> 8;
		streamToHide[7] = streamSize;
		if (!fileToHide.read((char*)&streamToHide[8], streamSize));
		fileToHide.close();

		std::cout << "First 64 bytes of the stream data: ";
		for (int i = 0; i < 64; ++i)
		{
			std::cout << (int)streamToHide[i] << " ";
		}
		std::cout << std::endl;

		hide_CPU(streamToHide, streamSize + 8);
		delete[] streamToHide;

		std::cout << "First 64 bytes of the processed image: ";
		for (int i = 0; i < 64; ++i)
		{
			std::cout << (int)image.data[i] << " ";
		}
		std::cout << std::endl;

		cv::imwrite("D:/Documents/Faculty/PGPU/Tema/cuda_steganography/result.bmp", image);
		
		cv::Mat nextImage = cv::imread("D:/Documents/Faculty/PGPU/Tema/cuda_steganography/result.bmp");
		cleanUp_CPU();

		std::cout << "First 64 bytes of the image from disk: ";
		for (int i = 0; i < 64; ++i)
		{
			std::cout << (int)nextImage.data[i] << " ";
		}
		std::cout << std::endl;

		analyzeImage_CPU(nextImage.data, nextImage.size().height, nextImage.size().width);
		streamSize = extract_CPU(streamToHide) - 8;

		std::ofstream recoveredFile("D:/Documents/Faculty/PGPU/Tema/cuda_steganography/result.mp3", std::ios_base::out | std::ios_base::binary);
		recoveredFile.write((char*)&streamToHide[8], streamSize);
		recoveredFile.flush();
		recoveredFile.close();

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