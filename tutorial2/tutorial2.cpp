#include "tutorial2.hpp"

namespace filter {

    static inline int greyscale(int R, int G, int B);

    static const signed char Gx[3][3] = {{-1, 0, 1},
                                         {-2, 0, 2},
                                         {-1, 0, 1}};
    static const signed char Gy[3][3] = {{1, 2, 1},
                                         {0, 0, 0},
                                         {-1, -2, -1}};

    void apply_greyscale(cv::Mat image) {
        unsigned char *rgb = (unsigned char *) image.data;
        int channels = image.channels();
        for (int i = 0; i < image.rows; i++) {
            for (int j = 0; j < image.cols; j++) {
                int index = i * image.step + j * channels; // calculate data offset from current position/channel

                char b = rgb[index];
                char g = rgb[index + 1];
                char r = rgb[index + 2];

                char grey = greyscale(r, g, b); // calculate the grey value

                rgb[index] = grey;
                rgb[index + 1] = grey;
                rgb[index + 2] = grey;
            }
        }
    }

    cv::Mat apply_convolution(cv::Mat image) {
        cv::Mat sobel = cv::Mat::zeros(image.size(), CV_16SC3); // initialize a 3-channel matrix with size equal to the image represented by signed shorts

        for (int i = 1; i < image.rows - 1; i++) {
            for (int j = 1; j < image.cols - 1; j++) {

                short gradX = 0, gradY = 0, gradMag;

                for (int ki = -1; ki <= 1; ki++) {
                    for (int kj = -1; kj <= 1; kj++) {
                        BGR &anchor = image.ptr<BGR>(i + ki)[j + kj]; // get BGR value for the pixel position + the convolution offset
                        gradX += anchor.B * filter::Gx[ki + 1][kj + 1]; // calculate the X gradient from the X kernel
                        gradY += anchor.B * filter::Gy[ki + 1][kj + 1]; // calculate the Y gradient from the Y kernel
                    }
                }

                gradMag = abs(gradX) + abs(gradY); // approximate gradient magnitude from absolute value of component sums

                BGRS sobel_color = {gradMag, gradMag, gradMag}; // create the struct to store pixel BGR value as a short

                sobel.ptr<BGRS>(i)[j] = sobel_color; // set BGR elements of pixel to corresponding filtered value
            }
        }

        normalize(sobel, sobel, 0, 255, cv::NORM_MINMAX, CV_8UC3); // convert each RGB element from a short to unsigned int to display properly

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
    cv::Mat image, sobel;

    if (!capture.isOpened()) {
        throw std::runtime_error("file cannot be opened...\n");
    }

    cv::namedWindow(argv[1], 1); // initialize display window
    int framesCreated = 0;

    while (true) {
        capture >> image; // get next image from video stream
        if (image.empty()) break; // exit if no more frames
        filter::apply_greyscale(image); // apply greyscaling to the current image in-place
        sobel = filter::apply_convolution(image); // apply the Sobel filter to the given frame
        framesCreated++; // increment frame counter
        printf("%d/%d frames computed...\n", framesCreated, (int) capture.get(cv::CAP_PROP_FRAME_COUNT));
        cv::imshow(argv[1], sobel); // display the image
	    cv::waitKey(1);
    }

    return 0;
}
