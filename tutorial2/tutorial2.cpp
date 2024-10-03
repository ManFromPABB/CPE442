#include "tutorial2.hpp"

namespace filter {

    static inline int greyscale(int R, int G, int B);

    static const signed char Gx[3][3] = {{-1, 0, 1},
                                         {-2, 0, 2},
                                         {-1, 0, 1}};
    static const signed char Gy[3][3] = {{1, 2, 1},
                                         {0, 0, 0},
                                         {-1, -2, -1}};

    cv::Mat apply_greyscale(cv::Mat image) {
        cv::Mat grey_image = cv::Mat::zeros(image.size(), CV_8U);
        unsigned char *rgb = (unsigned char *) image.data;
        int channels = image.channels();
        for (int i = 0; i < image.rows; i++) {
            for (int j = 0; j < image.cols; j++) {
                int index = i * image.step + j * channels; // calculate data offset from current position/channel

                unsigned char b = rgb[index];
                unsigned char g = rgb[index + 1];
                unsigned char r = rgb[index + 2];

                unsigned char grey = greyscale(r, g, b); // calculate the grey value

                grey_image.ptr<uchar>(i)[j] = grey;
            }
        }
        return grey_image;
    }

    cv::Mat apply_convolution(cv::Mat image) {
        cv::Mat sobel = cv::Mat::zeros(image.size(), CV_8U); // initialize the output matrix as a 8U matrix
        for (int i = 1; i < image.rows - 1; i++) {
            for (int j = 1; j < image.cols - 1; j++) {

                short gradX = 0, gradY = 0, gradMag; // initialize intermediate calc values as shorts to prevent overflow

                for (int ki = -1; ki <= 1; ki++) {
                    for (int kj = -1; kj <= 1; kj++) {
                        uchar anchor = image.ptr<uchar>(i + ki)[j + kj]; // get BGR value for the pixel position + the convolution offset
                        gradX += anchor * filter::Gx[ki + 1][kj + 1]; // calculate the X gradient from the X kernel
                        gradY += anchor * filter::Gy[ki + 1][kj + 1]; // calculate the Y gradient from the Y kernel
                    }
                }

                gradMag = abs(gradX) + abs(gradY); // approximate gradient magnitude from absolute value of component sums

                if (gradMag > 255) gradMag = 255;

                sobel.ptr<uchar>(i)[j] = gradMag; // set BGR elements of pixel to corresponding filtered value
            }
        }
        return sobel;
    }

    static inline int greyscale(int R, int G, int B) {
        return 0.2126 * R + 0.7152 * G + 0.0722 * B; // apply CCIR 601 greyscaling
    }

}

int main(int argc, char *argv[]) {

    if (argc != 2) {
        throw std::runtime_error("tutorial2 requires one argument, the video to open...\n");
    }

    cv::VideoCapture capture(argv[1]);
    cv::Mat image, greyscale, sobel;

    if (!capture.isOpened()) {
        throw std::runtime_error("file cannot be opened...\n");
    }

    cv::namedWindow(argv[1], 1); // initialize display window
    int framesCreated = 0;

    while (true) {
        capture >> image; // get next image from video stream
        if (image.empty()) break; // exit if no more frames
        greyscale = filter::apply_greyscale(image); // apply greyscaling to the current image in-place
        sobel = filter::apply_convolution(greyscale); // apply the Sobel filter to the given frame
        framesCreated++; // increment frame counter
        printf("%d/%d frames computed...\n", framesCreated, (int) capture.get(cv::CAP_PROP_FRAME_COUNT));
        cv::imshow(argv[1], sobel); // display the image
	    cv::waitKey(1);
    }

    return 0;
}
