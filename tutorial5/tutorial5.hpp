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
#include <arm_neon.h>
#include <perfmon/pfmlib.h>
#include <papi.h>

#define NUMWORKERS 4

typedef struct image_data {
    cv::Mat *image;
    cv::Mat *greyscale;
    cv::Mat *sobel;
} image_data;

typedef struct thread_data {
    image_data *data;
    int threadid;
} thread_data;

void* process_image(void *arg);