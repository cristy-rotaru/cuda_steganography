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
	uint8_t* imageData = (uint8_t*)CPU_imageData;
	uint8_t* pixelWeight = (uint8_t*)CPU_pixelWeight;

	for (size_t i = 0; i < CPU_height * CPU_width * 3; ++i)
	{
		size_t streamIndex = bitsWritten >> 3;
		uint8_t bitIndex = bitsWritten & 0x07;

		if (streamIndex >= streamSize)
		{
			return imageData;
		}

		uint8_t weight = pixelWeight[i];
		uint8_t mask = (1 << weight) - 1;
		uint8_t toWrite = 0;

		if (8 - bitIndex >= weight)
		{
			uint8_t shift = 8 - bitIndex - weight;
			toWrite = (stream[streamIndex] & (mask << shift)) >> shift;
		}
		else
		{
			uint8_t shift = weight + bitIndex - 8;
			toWrite = (stream[streamIndex] & (mask >> shift)) << shift;
			if (streamIndex < streamSize - 1)
			{
				shift = 8 - shift;
				toWrite |= (stream[streamIndex + 1] & (mask << shift)) >> shift;
			}
		}

		imageData[i] &= ~mask;
		imageData[i] |= mask & toWrite;

		bitsWritten += weight;
	}
}

// will return the size of the stream
// will also allocate the stream
uint32_t extract_CPU(uint8_t*& stream)
{
	uint8_t header[10];
	size_t bitsRead = 0;
	size_t imageIndex = 0;

	uint8_t* imageData = (uint8_t*)CPU_imageData;
	uint8_t* pixelWeight = (uint8_t*)CPU_pixelWeight;

	while (bitsRead < 64)
	{
		size_t streamIndex = bitsRead >> 3;
		uint8_t bitIndex = bitsRead & 0x07;

		uint8_t pixelData = imageData[imageIndex];
		uint8_t weight = pixelWeight[imageIndex];
		uint8_t mask = (1 << weight) - 1;
		uint8_t valuableData = pixelData & mask;

		if (8 - bitIndex >= weight)
		{
			uint8_t shift = 8 - bitIndex - weight;
			header[streamIndex] &= ~(mask << shift); // ignore this warning | the compiler is retarded
			header[streamIndex] |= valuableData << shift;
		}
		else
		{
			uint8_t shift = weight + bitIndex - 8;
			header[streamIndex] &= ~(mask >> shift);
			header[streamIndex] |= valuableData >> shift;
			shift = 8 - shift;
			header[streamIndex + 1] &= ~(mask << shift);
			header[streamIndex + 1] |= valuableData << shift;
		}

		bitsRead += weight;

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
		size_t streamIndex = bitsRead >> 3;
		size_t bitIndex = bitsRead & 0x07;

		if (streamIndex >= streamSize)
		{
			return streamSize;
		}

		uint8_t pixelData = imageData[imageIndex];
		uint8_t weight = pixelWeight[imageIndex];
		uint8_t mask = (1 << weight) - 1;
		uint8_t valuableData = pixelData & mask;

		if (8 - bitIndex >= weight)
		{
			uint8_t shift = 8 - bitIndex - weight;
			stream[streamIndex] &= ~(mask << shift); // ignore this warning | the compiler is retarded
			stream[streamIndex] |= valuableData << shift;
		}
		else
		{
			uint8_t shift = weight + bitIndex - 8;
			stream[streamIndex] &= ~(mask >> shift);
			stream[streamIndex] |= valuableData >> shift;

			if (streamIndex < streamSize - 1)
			{
				shift = 8 - shift;
				stream[streamIndex + 1] &= ~(mask << shift);
				stream[streamIndex + 1] |= valuableData << shift;
			}
			else
			{
				return streamSize;
			}
		}

		bitsRead += weight;

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