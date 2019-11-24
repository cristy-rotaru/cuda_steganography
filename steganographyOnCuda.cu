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

			maxDelta.RED = (maxDelta.RED >= 13) ? 4 : ((maxDelta.RED >= 6) ? 3 : 2);
			maxDelta.GREEN = (maxDelta.GREEN >= 13) ? 4 : ((maxDelta.GREEN >= 6) ? 3 : 2);
			maxDelta.BLUE = (maxDelta.BLUE >= 13) ? 4 : ((maxDelta.BLUE >= 6) ? 3 : 2);

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
			writeRed = (stream[streamIndex] & (maskRed << shift)) >> shift;
		}
		else
		{
			uint8_t shift = weight.GREEN + bitIndex - 8;
			writeRed = (stream[streamIndex] & (maskRed >> shift)) << shift;
			shift = 8 - shift;
			if (streamIndex < streamSize - 1)
			{
				writeRed |= (stream[streamIndex + 1] & (maskRed << shift)) >> shift;
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
			writeRed = (stream[streamIndex] & (maskRed << shift)) >> shift;
		}
		else
		{
			uint8_t shift = weight.BLUE + bitIndex - 8;
			writeRed = (stream[streamIndex] & (maskRed >> shift)) << shift;
			shift = 8 - shift;
			if (streamIndex < streamSize - 1)
			{
				writeRed |= (stream[streamIndex + 1] & (maskRed << shift)) >> shift;
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

		uint8_t dataRed = pixelData.RED & ((1 << weight.RED) - 1);
		uint8_t dataGreen = pixelData.GREEN & ((1 << weight.GREEN) - 1);
		uint8_t dataBlue = pixelData.BLUE & ((1 << weight.BLUE) - 1);
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