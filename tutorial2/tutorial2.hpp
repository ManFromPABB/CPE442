#pragma once
#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

static const signed char Gx_arr[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
static const signed char Gy_arr[] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

static const cv::Mat Gx(3, 3, CV_8S, (void *) Gx_arr);
static const cv::Mat Gy(3, 3, CV_8S, (void *) Gy_arr);