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

typedef struct image_data {
    cv::VideoCapture capture;
    pthread_t thread_table[];
    cv::Mat image;
    cv::Mat greyscale;
    cv::Mat sobel;


} image_data;