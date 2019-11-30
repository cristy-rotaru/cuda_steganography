#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/opencv.hpp>

#define getDelta(v1, v2) ((v1) > (v2)) ? ((((v1) & 0xF) >> 4) - (((v2) & 0xF) >> 4)) : ((((v2) & 0xF) >> 4) - (((v1) & 0xF) >> 4))

// ordinea in Mat::data este blue, breen, red
typedef struct
{
	uint8_t BLUE;
	uint8_t GREEN;
	uint8_t RED;
} color_t;

color_t* CPU_pixelWeight;
color_t* CPU_imageData;
size_t CPU_height, CPU_width;

// returns the number of bytes that can be encoded in the image
uint32_t analyzeImage_CPU(uint8_t* imageData, size_t height, size_t width)
{
	CPU_imageData = (color_t*)imageData;
	CPU_height = height;
	CPU_width = width;

	CPU_pixelWeight = new color_t[width * height];

	int bitCount = 0;
	for (size_t i = 0; i < height; ++i)
	{
		for (size_t j = 0; j < width; ++j)
		{
			size_t currentIndex = i * width + j;
			color_t currentColor = CPU_imageData[currentIndex];

			color_t maxDelta = { 0, 0, 0 };

			auto compareAndUpdateDelta = [currentColor, &maxDelta](size_t index)
			{
				color_t northColor = CPU_imageData[index];

				color_t delta;
				delta.RED = getDelta(currentColor.RED, northColor.RED);
				delta.GREEN = getDelta(currentColor.GREEN, northColor.GREEN);
				delta.BLUE = getDelta(currentColor.BLUE, northColor.BLUE);

				if (maxDelta.RED < delta.RED)
				{
					maxDelta.RED = delta.RED;
				}
				if (maxDelta.GREEN < delta.GREEN)
				{
					maxDelta.GREEN = delta.GREEN;
				}
				if (maxDelta.BLUE < delta.BLUE)
				{
					maxDelta.BLUE = delta.BLUE;
				}
			};

			//north
			if (i > 0)
			{
				compareAndUpdateDelta(currentIndex - width);
			}
			//south
			if (i < height - 1)
			{
				compareAndUpdateDelta(currentIndex + width);
			}
			//west
			if (j > 0)
			{
				compareAndUpdateDelta(currentIndex - 1);
			}
			//east
			if (j < width - 1)
			{
				compareAndUpdateDelta(currentIndex + 1);
			}

			maxDelta.RED = (maxDelta.RED >= 13) ? 4 : ((maxDelta.RED >= 6) ? 3 : ((maxDelta.RED >= 2) ? 2 : 1));
			maxDelta.GREEN = (maxDelta.GREEN >= 13) ? 4 : ((maxDelta.GREEN >= 6) ? 3 : ((maxDelta.GREEN >= 2) ? 2 : 1));
			maxDelta.BLUE = (maxDelta.BLUE >= 13) ? 4 : ((maxDelta.BLUE >= 6) ? 3 : ((maxDelta.BLUE >= 2) ? 2 : 1));

			CPU_pixelWeight[currentIndex] = maxDelta;

			bitCount += maxDelta.RED + maxDelta.GREEN + maxDelta.BLUE;
		}
	}

	return bitCount >> 3;
}

uint8_t* hide_CPU(uint8_t* stream, size_t streamSize)
{
	size_t bitsWritten = 0;

	for (size_t i = 0; i < CPU_height * CPU_width; ++i)
	{
		size_t currentIndex = i;
		color_t weight = CPU_pixelWeight[currentIndex];

		uint8_t maskRed = (1 << weight.RED) - 1;
		uint8_t maskGreen = (1 << weight.GREEN) - 1;
		uint8_t maskBlue = (1 << weight.BLUE) - 1;

		uint8_t writeRed = 0;
		uint8_t writeGreen = 0;
		uint8_t writeBlue = 0;

		size_t streamIndex = bitsWritten >> 3;
		uint8_t bitIndex = bitsWritten & 0x07;

		// red channel
		if (8 - bitIndex >= weight.RED)
		{// all bits can be taken from the current entry
			uint8_t shift = 8 - bitIndex - weight.RED;
			writeRed = (stream[streamIndex] & (maskRed << shift)) >> shift;
		}
		else
		{
			uint8_t shift = weight.RED + bitIndex - 8;
			writeRed = (stream[streamIndex] & (maskRed >> shift)) << shift;
			shift = 8 - shift;
			if (streamIndex < streamSize - 1)
			{
				writeRed |= (stream[streamIndex + 1] & (maskRed << shift)) >> shift;
				std::cout << " " << std::hex << (int)stream[streamIndex + 1];
			}
		}

		CPU_imageData[currentIndex].RED &= (~maskRed);
		CPU_imageData[currentIndex].RED |= maskRed & writeRed;
		bitsWritten += weight.RED;

		streamIndex = bitsWritten >> 3;
		bitIndex = bitsWritten & 0x07;

		if (streamIndex >= streamSize)
		{
			return (uint8_t*)CPU_imageData;
		}

		// green channel
		if (8 - bitIndex >= weight.GREEN)
		{// all bits can be taken from the current entry
			uint8_t shift = 8 - bitIndex - weight.GREEN;
			writeGreen = (stream[streamIndex] & (maskGreen << shift)) >> shift;
		}
		else
		{
			uint8_t shift = weight.GREEN + bitIndex - 8;
			writeGreen = (stream[streamIndex] & (maskGreen >> shift)) << shift;
			shift = 8 - shift;
			if (streamIndex < streamSize - 1)
			{
				writeGreen |= (stream[streamIndex + 1] & (maskGreen << shift)) >> shift;
				std::cout << " " << std::hex << (int)stream[streamIndex + 1];
			}
		}

		CPU_imageData[currentIndex].GREEN &= (~maskGreen);
		CPU_imageData[currentIndex].GREEN |= maskGreen & writeGreen;
		bitsWritten += weight.GREEN;

		streamIndex = bitsWritten >> 3;
		bitIndex = bitsWritten & 0x07;

		if (streamIndex >= streamSize)
		{
			return (uint8_t*)CPU_imageData;
		}

		// blue channel
		if (8 - bitIndex >= weight.BLUE)
		{// all bits can be taken from the current entry
			uint8_t shift = 8 - bitIndex - weight.BLUE;
			writeBlue = (stream[streamIndex] & (maskBlue << shift)) >> shift;
		}
		else
		{
			uint8_t shift = weight.BLUE + bitIndex - 8;
			writeBlue = (stream[streamIndex] & (maskBlue >> shift)) << shift;
			shift = 8 - shift;
			if (streamIndex < streamSize - 1)
			{
				writeBlue |= (stream[streamIndex + 1] & (maskBlue << shift)) >> shift;
				std::cout << " " << std::hex << (int)stream[streamIndex + 1];
			}
		}

		CPU_imageData[currentIndex].BLUE &= (~maskBlue);
		CPU_imageData[currentIndex].BLUE |= maskBlue & writeBlue;
		bitsWritten += weight.BLUE;

		streamIndex = bitsWritten >> 3;
		bitIndex = bitsWritten & 0x07;

		if (streamIndex >= streamSize)
		{
			return (uint8_t*)CPU_imageData;
		}
	}
}

// will return the size of the stream
// will also allocate the stream
uint32_t extract_CPU(uint8_t*& stream)
{
	uint8_t header[10];
	size_t bitsRead = 0;
	size_t imageIndex = 0;

	while (bitsRead < 64)
	{
		color_t pixelData = CPU_imageData[imageIndex];
		color_t weight = CPU_pixelWeight[imageIndex];

		size_t streamIndex = bitsRead >> 3;
		size_t bitIndex = bitsRead & 0x07;

		uint8_t maskRed = (1 << weight.RED) - 1;
		uint8_t maskGreen = (1 << weight.GREEN) - 1;
		uint8_t maskBlue = (1 << weight.BLUE) - 1;

		uint8_t readRed = pixelData.RED & maskRed;
		uint8_t readGreen = pixelData.GREEN & maskGreen;
		uint8_t readBlue = pixelData.BLUE & maskBlue;

		//red
		if (8 - bitIndex >= weight.RED)
		{
			uint8_t shift = 8 - bitIndex - weight.RED;
			header[streamIndex] &= ~(maskRed << shift); // ignore this warning | the compiler is retarded
			header[streamIndex] |= readRed << shift;
		}
		else
		{
			uint8_t shift = weight.RED + bitIndex - 8;
			header[streamIndex] &= ~(maskRed >> shift);
			header[streamIndex] |= readRed >> shift;
			shift = 8 - shift;
			header[streamIndex + 1] &= ~(maskRed << shift);
			header[streamIndex + 1] |= readRed << shift;
		}
		bitsRead += weight.RED;

		streamIndex = bitsRead >> 3;
		bitIndex = bitsRead & 0x07;

		//green
		if (8 - bitIndex >= weight.GREEN)
		{
			uint8_t shift = 8 - bitIndex - weight.GREEN;
			header[streamIndex] &= ~(maskGreen << shift); // ignore this warning | the compiler is retarded
			header[streamIndex] |= readGreen << shift;
		}
		else
		{
			uint8_t shift = weight.GREEN + bitIndex - 8;
			header[streamIndex] &= ~(maskGreen >> shift);
			header[streamIndex] |= readGreen >> shift;
			shift = 8 - shift;
			header[streamIndex + 1] &= ~(maskGreen << shift);
			header[streamIndex + 1] |= readGreen << shift;
		}
		bitsRead += weight.GREEN;

		streamIndex = bitsRead >> 3;
		bitIndex = bitsRead & 0x07;

		//green
		if (8 - bitIndex >= weight.BLUE)
		{
			uint8_t shift = 8 - bitIndex - weight.BLUE;
			header[streamIndex] &= ~(maskBlue << shift); // ignore this warning | the compiler is retarded
			header[streamIndex] |= readBlue << shift;
		}
		else
		{
			uint8_t shift = weight.BLUE + bitIndex - 8;
			header[streamIndex] &= ~(maskBlue >> shift);
			header[streamIndex] |= readBlue >> shift;
			shift = 8 - shift;
			header[streamIndex + 1] &= ~(maskBlue << shift);
			header[streamIndex + 1] |= readBlue << shift;
		}
		bitsRead += weight.BLUE;

		++imageIndex;
	}

	if (header[0] != 's' || header[1] != 't' || header[2] != 'e' || header[3] != 'g')
	{
		return 0;
	}

	uint32_t streamSize = 0;
	streamSize |= header[4] << 24;
	streamSize |= header[5] << 16;
	streamSize |= header[6] << 8;
	streamSize |= header[7];
	streamSize += 8;

	stream = new uint8_t[streamSize];
	memcpy(stream, header, 10);

	for (;;)
	{
		color_t pixelData = CPU_imageData[imageIndex];
		color_t weight = CPU_pixelWeight[imageIndex];

		size_t streamIndex = bitsRead >> 3;
		size_t bitIndex = bitsRead & 0x07;

		uint8_t maskRed = (1 << weight.RED) - 1;
		uint8_t maskGreen = (1 << weight.GREEN) - 1;
		uint8_t maskBlue = (1 << weight.BLUE) - 1;

		uint8_t readRed = pixelData.RED & maskRed;
		uint8_t readGreen = pixelData.GREEN & maskGreen;
		uint8_t readBlue = pixelData.BLUE & maskBlue;

		//red
		if (8 - bitIndex >= weight.RED)
		{
			uint8_t shift = 8 - bitIndex - weight.RED;
			stream[streamIndex] &= ~(maskRed << shift); // ignore this warning | the compiler is retarded
			stream[streamIndex] |= readRed << shift;
		}
		else
		{
			uint8_t shift = weight.RED + bitIndex - 8;
			stream[streamIndex] &= ~(maskRed >> shift);
			stream[streamIndex] |= readRed >> shift;
			shift = 8 - shift;
			if (streamIndex < streamSize - 1)
			{
				stream[streamIndex + 1] &= ~(maskRed << shift);
				stream[streamIndex + 1] |= readRed << shift;
			}
			else
			{
				return streamSize;
			}
		}
		bitsRead += weight.RED;

		streamIndex = bitsRead >> 3;
		bitIndex = bitsRead & 0x07;

		if (streamIndex >= streamSize)
		{
			return streamSize;
		}

		//green
		if (8 - bitIndex >= weight.GREEN)
		{
			uint8_t shift = 8 - bitIndex - weight.GREEN;
			stream[streamIndex] &= ~(maskGreen << shift); // ignore this warning | the compiler is retarded
			stream[streamIndex] |= readGreen << shift;
		}
		else
		{
			uint8_t shift = weight.GREEN + bitIndex - 8;
			stream[streamIndex] &= ~(maskGreen >> shift);
			stream[streamIndex] |= readGreen >> shift;
			shift = 8 - shift;
			if (streamIndex < streamSize - 1)
			{
				stream[streamIndex + 1] &= ~(maskGreen << shift);
				stream[streamIndex + 1] |= readGreen << shift;
			}
			else
			{
				return streamSize;
			}
		}
		bitsRead += weight.GREEN;

		streamIndex = bitsRead >> 3;
		bitIndex = bitsRead & 0x07;

		if (streamIndex >= streamSize)
		{
			return streamSize;
		}

		//green
		if (8 - bitIndex >= weight.BLUE)
		{
			uint8_t shift = 8 - bitIndex - weight.BLUE;
			stream[streamIndex] &= ~(maskBlue << shift); // ignore this warning | the compiler is retarded
			stream[streamIndex] |= readBlue << shift;
		}
		else
		{
			uint8_t shift = weight.BLUE + bitIndex - 8;
			stream[streamIndex] &= ~(maskBlue >> shift);
			stream[streamIndex] |= readBlue >> shift;
			shift = 8 - shift;
			if (streamIndex < streamSize - 1)
			{
				stream[streamIndex + 1] &= ~(maskBlue << shift);
				stream[streamIndex + 1] |= readBlue << shift;
			}
			else
			{
				return streamSize;
			}
		}
		bitsRead += weight.BLUE;

		streamIndex = bitsRead >> 3;
		bitIndex = bitsRead & 0x07;

		if (streamIndex >= streamSize)
		{
			return streamSize;
		}

		++imageIndex;
	}

	return 0;
}

void cleanUp_CPU()
{
	if (CPU_pixelWeight != nullptr)
	{
		delete[] CPU_pixelWeight;
	}
}