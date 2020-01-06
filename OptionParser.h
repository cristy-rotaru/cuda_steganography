#pragma once
#include <string>
//Option parser class for this project

class OptionParser
{
private:
	const std::string outputOption = "-o";
	const std::string encode = "-e";
	const std::string decode = "-d";
	const std::string fileToHideOption = "-f";
	const std::string cpuOption = "-c";
	const std::string gpuOption = "-g";
	const std::string perfOption = "-p";
	const std::string helpOption = "-h";
	bool encodeActive;
	bool CPUflag, GPUflag, performanceFlag;
	std::string inputFileName;
	std::string outputFileName;
	std::string fileToHide;

	void printHelp();
public:
	

	OptionParser();
	
	//returns true if the CPU is used
	bool getCPUflag();

	//returns true if the GPU is used
	bool getGPUflag();

	//returns true if a performance comparison is requested
	bool getPerformanceFlag();

	//returns true if encoding a file, false if decoding
	bool getEncode();

	//returns the name of the image file to be used for encoding/decoding
	std::string getInputFileName();

	//returns the name of the output file which will contain the encoded image/decoded file
	std::string getOutputFileName();

	//returns the name of the file to be hidden in case of encoding
	std::string getFileToHide();

	//parses the given arguments
	void parse(int argc, char** argv);
	~OptionParser();
};

