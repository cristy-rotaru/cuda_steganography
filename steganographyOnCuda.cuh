#ifndef STEGANOGRAPHYONCUDA_CUH_
#define STEGANOGRAPHYONCUDA_CUH_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/opencv.hpp>

uint32_t analyzeImage_CPU(uint8_t* imageData, size_t height, size_t width);
uint8_t* hide_CPU(uint8_t* stream, size_t streamSize);

void cleanUp_CPU();

#endif // !STEGANOGRAPHYONCUDA_CUH_
