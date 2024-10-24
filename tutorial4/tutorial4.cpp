#include "tutorial3.hpp"

pthread_barrier_t workers_done, workers_ready;

namespace filter {

    static inline int greyscale(int R, int G, int B);

    static const signed char Gx[3][3] = {{-1, 0, 1},
                                         {-2, 0, 2},
                                         {-1, 0, 1}};
    static const signed char Gy[3][3] = {{1, 2, 1},
                                         {0, 0, 0},
                                         {-1, -2, -1}};

    void apply_greyscale(cv::Mat image, cv::Mat grey_image) {
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
    }

    void apply_convolution(cv::Mat image, cv::Mat sobel, int row) {
        for (int j = 1; j < image.cols - 1; j++) {

            short gradX = 0, gradY = 0, gradMag; // initialize intermediate calc values as shorts to prevent overflow

            for (int ki = -1; ki <= 1; ki++) {
                for (int kj = -1; kj <= 1; kj++) {
                    uchar anchor = image.ptr<uchar>(row + ki)[j + kj]; // get BGR value for the pixel position + the convolution offset
                    gradX += anchor * filter::Gx[ki + 1][kj + 1]; // calculate the X gradient from the X kernel
                    gradY += anchor * filter::Gy[ki + 1][kj + 1]; // calculate the Y gradient from the Y kernel
                }
            }

            gradMag = abs(gradX) + abs(gradY); // approximate gradient magnitude from absolute value of component sums

            if (gradMag > 255) gradMag = 255;

            sobel.ptr<uchar>(row)[j] = gradMag; // set BGR elements of pixel to corresponding filtered value
        }
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

    if (!capture.isOpened()) {
        throw std::runtime_error("file cannot be opened...\n");
    }

    cv::namedWindow("sobel", 1); // initialize display window

    image_data data;
    int numworkers = NUMWORKERS;
    cv::Mat greyscale, sobel;
    pthread_t *worker_threads = new pthread_t[numworkers];
    thread_data *thread_table = new thread_data[numworkers];
    pthread_barrier_init(&workers_done, NULL, numworkers + 1);
    pthread_barrier_init(&workers_ready, NULL, numworkers + 1);

    for (int i = 0; i < numworkers; i++) {
        thread_table[i].threadid = i;
        thread_table[i].data = &data;
        pthread_create(&worker_threads[i], NULL, process_image, &thread_table[i]);
    }

    cv::Mat image;
    capture >> image; // grab first frame to determine its size
    greyscale = cv::Mat::zeros(image.size(), CV_8U); // initialize the greyscale matrix as an 8U matrix
    sobel = cv::Mat::zeros(image.size(), CV_8U); // initialize the sobel matrix as an 8U matrix

    data.image = &image;
    data.greyscale = &greyscale;
    data.sobel = &sobel;

    while (true) {
        pthread_barrier_wait(&workers_ready);
        if (data.image->empty()) break; // exit if no more frames
        filter::apply_greyscale(*data.image, *data.greyscale);
        pthread_barrier_wait(&workers_done);
        cv::imshow("sobel", *data.sobel);
        cv::waitKey(1);
        capture >> *data.image;
    }

    for (int i = 0; i < numworkers; i++) {
        pthread_join(worker_threads[i], NULL);
    }

    pthread_barrier_destroy(&workers_done);
    pthread_barrier_destroy(&workers_ready);

    return 0;
}

void* process_image(void *arg) {
    thread_data *threaddata = (thread_data *) arg;
    image_data *data = threaddata->data;
    
    while (true) {
        pthread_barrier_wait(&workers_ready);

        if (data->image->empty()) return NULL;
        
        int chunk_size = data->greyscale->rows / NUMWORKERS;
        int start_index = std::max(1, threaddata->threadid * chunk_size);
        int end_index = std::min(data->greyscale->rows - 1, (threaddata->threadid == NUMWORKERS - 1) ? data->greyscale->rows : (start_index + chunk_size));

        for (int i = start_index; i < end_index; i++) {
            filter::apply_convolution(*data->greyscale, *data->sobel, i);
        }
        
        pthread_barrier_wait(&workers_done);
    }
    return NULL;
}
