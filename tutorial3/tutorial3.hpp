#pragma once
#include <stdio.h>
#include <sys/mman.h>
#include <unistd.h>
#include <pthread.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <cmath>

#define NUMWORKERS 4

typedef struct image_data {
    cv::Mat image;
    cv::Mat greyscale;
    cv::Mat sobel;
    int threadid;
} image_data;

void* process_image(void *arg);