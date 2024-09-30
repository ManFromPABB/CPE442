#pragma once
#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <cmath>

typedef struct BGR {
    uchar B;
    uchar G;
    uchar R;
} BGR;

typedef struct BGRS {
    short B;
    short G;
    short R;
} BGRS;