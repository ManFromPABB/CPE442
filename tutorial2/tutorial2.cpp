#include "tutorial2.hpp"

static inline int greyscale(int R, int G, int B);

namespace filter {

    void apply_greyscale(cv::Mat image) {
        unsigned char *rgb = (unsigned char *) image.data;
        int channels = image.channels();
        for (int i = 0; i < image.rows; i++) {
            for (int j = 0; j < image.cols; j++) {
                int index = i * image.step + j * channels;

                char b = rgb[index];
                char g = rgb[index + 1];
                char r = rgb[index + 2];

                char grey = greyscale(r, g, b);

                rgb[index] = grey;
                rgb[index + 1] = grey;
                rgb[index + 2] = grey;
            }
        }
    }
}

int main(int argc, char *argv[]) {

    if (argc != 2) {
        throw std::runtime_error("tutorial2 requires one argument, the video to open...\n");
    }

    cv::VideoCapture capture(argv[1]);
    cv::Mat image, greyscaled;

    cv::VideoWriter output("out.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(1080, 1920)); // testing, remove later

    if (!capture.isOpened()) {
        throw std::runtime_error("file cannot be opened...\n");
    }

    cv::namedWindow(argv[1], 1);
    while (true) {
        capture >> image;
        if (image.empty()) break;
        filter::apply_greyscale(image);
        output.write(image); // testing, remove later
        // todo: sobel
        cv::imshow(argv[1], 1);
    }

    output.release(); // testing, remove later

    return 0;
}