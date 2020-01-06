#include <iostream>
#include "OptionParser.h"

OptionParser::OptionParser()
{
	CPUflag = false;
	GPUflag = true;
	performanceFlag = false;
	encodeActive = false;
	inputFileName = "";
	outputFileName = "";
	fileToHide = "";
}

bool OptionParser::getCPUflag()
{
	return CPUflag;
}

bool OptionParser::getGPUflag()
{
	return GPUflag;
}

bool OptionParser::getPerformanceFlag()
{
	return performanceFlag;
}

bool OptionParser::getEncode()
{
	return encodeActive;
}

std::string OptionParser::getInputFileName()
{
	return inputFileName;
}

std::string OptionParser::getOutputFileName()
{
	return outputFileName;
}

std::string OptionParser::getFileToHide()
{
	return fileToHide;
}

void OptionParser::printHelp()
{
	std::cout << "Available options:\n\t";
	std::cout << "-h - print a help message\n\t";
	std::cout << "-c - use CPU for processing\n\t";
	std::cout << "-g - use GPU for processing\n\t";
	std::cout << "-p - do a performance comparison between CPU and GPU\n\t";
	std::cout << "-e <inputImage> - encode a file in the given image\n\t";
	std::cout << "-d <inputImage> - decode a file hidden in the given image\n\t";
	std::cout << "-f <fileName> - only usable after -e option\n\t";
	std::cout << "              - specifies the file which will be encoded in the image\n\t";
	std::cout << "-o <outputImage/outputFile> - if used with -e: specifies the name of the output image, containing the encoded information\n\t";
	std::cout << "                            - if used with -d: specifies the name of the decoded file\n";
}

void OptionParser::parse(int argc, char** argv)
{
	bool optionArgument = false;
	encodeActive = false;
	for (int i = 1; i < argc; i++)
	{
		std::string str(argv[i]);
		//Work on the no argument options
		if (str == cpuOption)
		{
			CPUflag = true;
			performanceFlag = false;
			GPUflag = false;
		}
		else if (str == gpuOption)
		{
			GPUflag = true;
			CPUflag = false;
			performanceFlag = false;
		}
		else if (str == perfOption)
		{
			performanceFlag = true;
			GPUflag = false;
			CPUflag = false;
		}
		else if (str == helpOption)
		{
			printHelp();
		}
		//Work on the options that require an argument
		//Check if there is another argument
		//Check if the next argument is not an option as well (whether it starts with a '-' or not)
		else if (i + 1 < argc && argv[i+1][0] != '-')
		{
			//-o
			if (str == outputOption)
			{
				if (outputFileName == "")
					outputFileName = argv[i + 1];
				else
					throw std::exception("-o: Output file name option already given");
			}
			//-e
			if (str == encode)
			{
				if (inputFileName == "")
				{
					inputFileName = argv[i + 1];
					encodeActive = true;
				}
				else
				{
					if (encodeActive)
						throw std::exception("-e: Encode option already given");
					else
						throw std::exception("-e: Decode option already given");
				}
			}
			//-d
			if (str == decode)
			{
				if (inputFileName == "")
				{
					inputFileName = argv[i + 1];
				}
				else
				{
					if (encodeActive)
						throw std::exception("-d: Encode option already given");
					else
						throw std::exception("-d: Decode option already given");
				}
			}
			//-f
			if (str == fileToHideOption)
			{
				//Check if -e was parsed before
				//If not, raise an exception
				if (encodeActive)
				{
					if (fileToHide == "")
					{
						fileToHide = argv[i + 1];
					}
					else
					{
						throw std::exception("-f: Option already given");
					}
				}
				else
				{
					throw std::exception("-f: The -e option must be given before the -f option");
				}
			}
			i++;
		}
		else
		{
			//Argument option lacks a correct argument
			//Do something about this
			printHelp();
			throw std::exception("Wrong option input");
		}
	}
	if (inputFileName == "")
	{
		printHelp();
		throw std::exception("No input file given");
	}
	if (outputFileName == "")
	{
		printHelp();
		throw std::exception("No output file given");
	}
	if (encodeActive && fileToHide == "")
	{
		printHelp();
		throw std::exception("No file given to be encoded, although the -e option was used");
	}
}


OptionParser::~OptionParser()
{
}
