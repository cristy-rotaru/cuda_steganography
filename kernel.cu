#include "steganographyOnCuda.cuh"
#include "OptionParser.h"
#include <exception>
#include <fstream>

void embedOnCPU(std::string inputImageFile, std::string fileToEmbed, std::string outputImageFile)
{
	cv::Mat image = cv::imread(inputImageFile);

	if (image.data == nullptr)
	{
		throw std::exception("Invalid image file!");
	}

	uint32_t maxStreamSize = analyzeImage_CPU(image.data, image.size().height, image.size().width);

	std::ifstream fileToHide(fileToEmbed, std::ios_base::in | std::ios_base::binary);

	if (!fileToHide)
	{
		throw std::exception("Unable to open file!");
	}
	// computing file size
	std::streampos sPos = fileToHide.tellg();
	fileToHide.seekg(0, std::ios::end);
	sPos = fileToHide.tellg() - sPos;
	fileToHide.seekg(0);
	size_t streamSize = sPos;

	if (maxStreamSize < streamSize + 8)
	{
		throw std::exception("The file does not fit inside the image!");
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
	fileToHide.read((char*)&streamToHide[8], streamSize);
	fileToHide.close();

	// will modify image
	hide_CPU(streamToHide, streamSize + 8);

	// the modified image is stored in image
	cv::imwrite(outputImageFile, image);

	cleanUp_CPU();
	delete[] streamToHide;
}

void extractOnCPU(std::string inputImageFile, std::string extractedFile)
{
	cv::Mat image = cv::imread(inputImageFile);

	if (image.data == nullptr)
	{
		throw std::exception("Invalid image file!");
	}

	analyzeImage_CPU(image.data, image.size().height, image.size().width);

	uint8_t* extractedStream;
	uint32_t streamSize = extract_CPU(extractedStream);
	if (streamSize == 0)
	{
		throw std::exception("The image does not contain anything!");
	}

	std::ofstream recoveredFile(extractedFile, std::ios_base::out | std::ios_base::binary);
	recoveredFile.write((char*)&extractedStream[8], streamSize);
	recoveredFile.flush();
	recoveredFile.close();

	cleanUp_CPU();
	delete[] extractedStream;
}

void embedOnGPU(std::string inputImageFile, std::string fileToEmbed, std::string outputImageFile)
{
	cv::Mat image = cv::imread(inputImageFile);

	if (image.data == nullptr)
	{
		throw std::exception("Invalid image file!");
	}

	startImageAnalisys_GPU(image.data, image.size().height, image.size().width);

	std::ifstream fileToHide(fileToEmbed, std::ios_base::in | std::ios_base::binary);

	if (!fileToHide)
	{
		throw std::exception("Unable to open file!");
	}
	// computing file size
	std::streampos sPos = fileToHide.tellg();
	fileToHide.seekg(0, std::ios::end);
	sPos = fileToHide.tellg() - sPos;
	fileToHide.seekg(0);
	size_t streamSize = sPos;

	uint8_t* streamToHide = new uint8_t[streamSize + 8]; // add 8 for header
	streamToHide[0] = 's';
	streamToHide[1] = 't';
	streamToHide[2] = 'e';
	streamToHide[3] = 'g';
	streamToHide[4] = streamSize >> 24;
	streamToHide[5] = streamSize >> 16;
	streamToHide[6] = streamSize >> 8;
	streamToHide[7] = streamSize;
	fileToHide.read((char*)&streamToHide[8], streamSize);
	fileToHide.close();

	initiateDataTransfer_GPU(streamToHide, streamSize + 8);
	uint32_t maxStreamSize = getImageCapacity_GPU();

	if (maxStreamSize < streamSize + 8)
	{
		throw std::exception("The file does not fit inside the image!");
	}

	// will modify image
	hide_GPU(image.data, streamSize + 8);
	cleanUp_GPU();

	delete[] streamToHide;

	// the modified image is stored in image
	cv::imwrite(outputImageFile, image);
}

void extractOnGPU(std::string inputImageFile, std::string extractedFile)
{
	cv::Mat image = cv::imread(inputImageFile);

	if (image.data == nullptr)
	{
		throw std::exception("Invalid image file!");
	}

	startImageAnalisys_GPU(image.data, image.size().height, image.size().width);
	getImageCapacity_GPU();

	uint8_t* extractedStream;
	uint32_t streamSize = extract_GPU(extractedStream);
	if (streamSize == 0)
	{
		throw std::exception("The image does not contain anything!");
	}

	cleanUp_GPU();

	std::ofstream recoveredFile(extractedFile, std::ios_base::out | std::ios_base::binary);
	recoveredFile.write((char*)&extractedStream[8], streamSize - 8);
	recoveredFile.flush();
	recoveredFile.close();

	delete[] extractedStream;
}

int main(int argc, char **argv)
{
	cudaDeviceReset();

	try
	{
		OptionParser parser;
		parser.parse(argc, argv);
		if (parser.getEncode())
		{
			if (parser.getCPUflag())
			{
				std::cout << "Embedding on CPU: ";

				embedOnCPU(parser.getInputFileName(), parser.getFileToHide(), parser.getOutputFileName());
				std::cout << "Done!\n";
			}
			// ------------------------------------------------------------------------------------------------------ //

			if (parser.getGPUflag())
			{
				std::cout << "Embedding on GPU: ";

				embedOnGPU(parser.getInputFileName(), parser.getFileToHide(), parser.getOutputFileName());
				std::cout << "Done!\n";
			}
			// ------------------------------------------------------------------------------------------------------ //

			if (parser.getPerformanceFlag())
			{
				//Do both, compare timings
				std::cout << "Embedding on CPU: ";

				std::chrono::high_resolution_clock::time_point startGPU = std::chrono::high_resolution_clock::now();
				embedOnCPU(parser.getInputFileName(), parser.getFileToHide(), parser.getOutputFileName());
				std::chrono::high_resolution_clock::time_point endGPU = std::chrono::high_resolution_clock::now();

				std::chrono::high_resolution_clock::duration elapsedGPU = endGPU - startGPU;

				std::cout << std::chrono::duration_cast<std::chrono::microseconds>(elapsedGPU).count() << "us" << std::endl;

				std::cout << "Embedding on GPU: ";

				std::chrono::high_resolution_clock::time_point startCPU = std::chrono::high_resolution_clock::now();
				embedOnGPU(parser.getInputFileName(), parser.getFileToHide(), parser.getOutputFileName());
				std::chrono::high_resolution_clock::time_point endCPU = std::chrono::high_resolution_clock::now();

				std::chrono::high_resolution_clock::duration elapsedCPU = endCPU - startCPU;

				std::cout << std::chrono::duration_cast<std::chrono::microseconds>(elapsedCPU).count() << "us" << std::endl;
			}
		}
		else
		{
			if (parser.getCPUflag())
			{
				std::cout << "Extracting on CPU: ";

				extractOnCPU(parser.getInputFileName(), parser.getOutputFileName());
				std::cout << "Done!\n";
				// ------------------------------------------------------------------------------------------------------ //
			}
			
			if (parser.getGPUflag())
			{
				std::cout << "Extracting on GPU: ";

				extractOnGPU(parser.getInputFileName(), parser.getOutputFileName());
				std::cout << "Done!\n";
			}

			if (parser.getPerformanceFlag())
			{
				std::cout << "Extracting on CPU: ";

				std::chrono::high_resolution_clock::time_point startGPU = std::chrono::high_resolution_clock::now();
				extractOnCPU(parser.getInputFileName(), parser.getOutputFileName());
				std::chrono::high_resolution_clock::time_point endGPU = std::chrono::high_resolution_clock::now();

				std::chrono::high_resolution_clock::duration elapsedGPU = endGPU - startGPU;

				std::cout << std::chrono::duration_cast<std::chrono::microseconds>(elapsedGPU).count() << "us" << std::endl;

				std::cout << "Extracting on GPU: ";

				std::chrono::high_resolution_clock::time_point startCPU = std::chrono::high_resolution_clock::now();
				extractOnGPU(parser.getInputFileName(), parser.getOutputFileName());
				std::chrono::high_resolution_clock::time_point endCPU = std::chrono::high_resolution_clock::now();

				std::chrono::high_resolution_clock::duration elapsedCPU = endCPU - startCPU;

				std::cout << std::chrono::duration_cast<std::chrono::microseconds>(elapsedCPU).count() << "us" << std::endl;

			}
		}
		system("pause");
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