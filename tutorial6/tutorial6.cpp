#include "tutorial6.hpp"

pthread_barrier_t workers_done, workers_ready;

namespace filter {

    void apply_greyscale(cv::Mat &image, cv::Mat &grey_image) {
        int channels = image.channels();
        for (int i = 0; i < image.rows; i++) {

            uchar *image_row = image.ptr<uchar>(i);
            uchar *grey_row = grey_image.ptr<uchar>(i);

            for (int j = 0; j < image.cols; j += 8) {
                uint8x8x3_t rgb_vec = vld3_u8(&image_row[j * channels]); // load 24 bytes of data representing 8 pixels into a uint8 8x3 matrix

                // split each matrix into uint8 vectors for each color
                uint16x8_t r_vec = vmovl_u8(rgb_vec.val[0]);
                uint16x8_t g_vec = vmovl_u8(rgb_vec.val[1]);
                uint16x8_t b_vec = vmovl_u8(rgb_vec.val[2]);
                
                float32x4_t r_low_f = vcvtq_f32_u32(vmovl_u16(vget_low_u16(r_vec))); // get lower 4 values from red vector and convert to float32
                float32x4_t r_high_f = vcvtq_f32_u32(vmovl_u16(vget_high_u16(r_vec))); // get high 4 values from red vector and convert to float32
                float32x4_t g_low_f = vcvtq_f32_u32(vmovl_u16(vget_low_u16(g_vec))); // get lower 4 values from green vector and convert to float32
                float32x4_t g_high_f = vcvtq_f32_u32(vmovl_u16(vget_high_u16(g_vec))); // get high 4 values from green vector and convert to float32
                float32x4_t b_low_f = vcvtq_f32_u32(vmovl_u16(vget_low_u16(b_vec))); // get lower 4 values from blue vector and convert to float32
                float32x4_t b_high_f = vcvtq_f32_u32(vmovl_u16(vget_high_u16(b_vec))); // get high 4 values from blue vector and convert to float32

                float32x4_t grey_low_f = vmlaq_n_f32(vmulq_n_f32(r_low_f, 0.2126f), g_low_f, 0.7152f); // do the multiplication on low red and green vectors and add 
                grey_low_f = vmlaq_n_f32(grey_low_f, b_low_f, 0.0722f); // finish the low grey calculations by multiplying the blue and accumulating

                float32x4_t grey_high_f = vmlaq_n_f32(vmulq_n_f32(r_high_f, 0.2126f), g_high_f, 0.7152f); // do the multiplication on high red and green vectors and add
                grey_high_f = vmlaq_n_f32(grey_high_f, b_high_f, 0.0722f); // finish the high grey calculations by multiplying the blue and accumulating

                // convert float32x4 to uint16x4
                uint16x4_t grey_low_u16 = vmovn_u32(vcvtq_u32_f32(grey_low_f));
                uint16x4_t grey_high_u16 = vmovn_u32(vcvtq_u32_f32(grey_high_f));

                uint8x8_t grey_u8 = vqmovn_u16(vcombine_u16(grey_low_u16, grey_high_u16)); // collapse the low/high uint16x4 vectors into a single uint8x8 vector

                vst1_u8(&grey_row[j], grey_u8); // store all 8 grey pixels into their respective locations
            }
        }
    }

    void apply_convolution(cv::Mat &image, cv::Mat &sobel, int row_start, int row_end) {
        for (int i = row_start; i < row_end; i++) {

            unsigned char *sobel_row = sobel.ptr<uchar>(i);

            for (int j = 1; j < image.cols - 1; j += 8) {

                int16x8_t gradX = vdupq_n_s16(0);
                int16x8_t gradY = vdupq_n_s16(0);

                // loop unroll to calculate each row's sobel equivalent explicitly
                const uchar *row_minus_1 = image.ptr<uchar>(i - 1); {
                    // get each vector of the top row of image values for 8 consecutive pixels
                    uint8x8_t vec_left = vld1_u8(&row_minus_1[j - 1]);
                    uint8x8_t vec_center = vld1_u8(&row_minus_1[j]);
                    uint8x8_t vec_right = vld1_u8(&row_minus_1[j + 1]);

                    // cast from u8 to i16 vector
                    int16x8_t vec_left_int = vreinterpretq_s16_u16(vmovl_u8(vec_left));
                    int16x8_t vec_center_int = vreinterpretq_s16_u16(vmovl_u8(vec_center));
                    int16x8_t vec_right_int = vreinterpretq_s16_u16(vmovl_u8(vec_right));

                    // multiply by kernel value and accumulate to gradient
                    gradX = vmlaq_n_s16(gradX, vec_left_int, -1);
                    gradX = vmlaq_n_s16(gradX, vec_right_int, 1);

                    gradY = vmlaq_n_s16(gradY, vec_left_int, 1);
                    gradY = vmlaq_n_s16(gradY, vec_center_int, 2);
                    gradY = vmlaq_n_s16(gradY, vec_right_int, 1);
                }

                const uchar *row_center = image.ptr<uchar>(i); {
                    // get each vector of the top row of image values for 8 consecutive pixels
                    uint8x8_t vec_left = vld1_u8(&row_center[j - 1]);
                    uint8x8_t vec_right = vld1_u8(&row_center[j + 1]);

                    // cast from u8 to i16 vector
                    int16x8_t vec_left_int = vreinterpretq_s16_u16(vmovl_u8(vec_left));
                    int16x8_t vec_right_int = vreinterpretq_s16_u16(vmovl_u8(vec_right));

                    // multiply by kernel value and accumulate to gradient
                    gradX = vmlaq_n_s16(gradX, vec_left_int, -2);
                    gradX = vmlaq_n_s16(gradX, vec_right_int, 2);

                    // no y gradient calculation needed for this row since it's all zero, this is where the speedup is
                }

                const uchar *row_plus_1 = image.ptr<uchar>(i + 1); {
                    // get each vector of the top row of image values for 8 consecutive pixels
                    uint8x8_t vec_left = vld1_u8(&row_plus_1[j - 1]);
                    uint8x8_t vec_center = vld1_u8(&row_plus_1[j]);
                    uint8x8_t vec_right = vld1_u8(&row_plus_1[j + 1]);

                    // cast from u8 to i16 vector
                    int16x8_t vec_left_int = vreinterpretq_s16_u16(vmovl_u8(vec_left));
                    int16x8_t vec_center_int = vreinterpretq_s16_u16(vmovl_u8(vec_center));
                    int16x8_t vec_right_int = vreinterpretq_s16_u16(vmovl_u8(vec_right));

                    // multiply by kernel value and accumulate to gradient
                    gradX = vmlaq_n_s16(gradX, vec_left_int, -1);
                    gradX = vmlaq_n_s16(gradX, vec_right_int, 1);

                    gradY = vmlaq_n_s16(gradY, vec_left_int, -1);
                    gradY = vmlaq_n_s16(gradY, vec_center_int, -2);
                    gradY = vmlaq_n_s16(gradY, vec_right_int, -1);
                }

                int16x8_t gradMag = vaddq_s16(vabsq_s16(gradX), vabsq_s16(gradY)); // approximate gradient magnitude from absolute value of component sums
                uint8x8_t result_vec = vqmovn_u16(vreinterpretq_u16_s16(gradMag)); // compute max on all vector elements such that they are less than 255
                
                vst1_u8(&sobel_row[j], result_vec); // set BGR elements of pixel to corresponding filtered value
            }
        }
    }
}

int main(int argc, char *argv[]) {

    if (argc != 2) {
        throw std::runtime_error("tutorial6 requires one argument, the video to open...\n");
    }

    // open the file specified by the input argument
    cv::VideoCapture capture(argv[1]);

    // setup PAPI library and make sure it's initialized
    if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT) {
        exit(1);
    }

    // declare 3 event counters for L1 data cache misses, L2 data cache misses, and total instructions executed
    int events[3] = {PAPI_L1_DCM, PAPI_L2_DCM, PAPI_TOT_INS};
    long long values[3];
    int event_set = PAPI_NULL;
    if (PAPI_create_eventset(&event_set) != PAPI_OK) { // create the event set
        exit(1);
    }

    // add each event counter to the set of measured events
    for (int i = 0; i < 3; i++) {
        if (PAPI_add_event(event_set, events[i]) != PAPI_OK) {
            exit(1);
        }
    }

    if (!capture.isOpened()) { // ensure video can be processed
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

    // create the worker threads and pass the Mat information across
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

    PAPI_start(event_set); // start the hardware event counters
    long long start_time = PAPI_get_real_nsec(); // start a hardware timer to measure execution time

    while (true) {
        pthread_barrier_wait(&workers_ready);
        if (data.image->empty()) break; // exit if no more frames
        filter::apply_greyscale(*data.image, *data.greyscale); // compute greyscale in main thread
        pthread_barrier_wait(&workers_done); // wait for worker threads to finish the sobel frame
        cv::imshow("sobel", *data.sobel); // display the new frame
        cv::waitKey(1);
        capture >> *data.image; // grab a new frame and pass to the worker threads
    }

    long long end_time = PAPI_get_real_nsec(); // stop the hardware timer

    // stop the cache/instruction counters
    if (PAPI_stop(event_set, values) != PAPI_OK) {
        fprintf(stderr, "Error stopping PAPI!\n");
        return 1;
    }

    double elapsed_time = (end_time - start_time) / 1e9; // calculate execution time in seconds

    // display the benchmark information
    printf("L1 Data Cache Misses: %lld\n", values[0]);
    printf("L2 Data Cache Misses: %lld\n", values[1]);
    printf("Total Instructions: %lld\n", values[2]);
    printf("Execution Time: %.9f seconds\n", elapsed_time);

    // terminate the PAPI event setup
    PAPI_shutdown();

    // recombine all the threads
    for (int i = 0; i < numworkers; i++) {
        pthread_join(worker_threads[i], NULL);
    }

    // delete the barriers
    pthread_barrier_destroy(&workers_done);
    pthread_barrier_destroy(&workers_ready);

    return 0;
}

void* process_image(void *arg) {
    thread_data *threaddata = (thread_data *) arg;
    image_data *data = threaddata->data;

    while (true) {
        pthread_barrier_wait(&workers_ready); // wait for the manager thread to coordinate a new frame

        if (data->image->empty()) return NULL;
        
        int chunk_size = data->greyscale->rows / NUMWORKERS; // determine the size of each thread's processing chunk
        int start_index = std::max(1, threaddata->threadid * chunk_size);
        int end_index = std::min(data->greyscale->rows - 1, (threaddata->threadid == NUMWORKERS - 1) ? data->greyscale->rows : (start_index + chunk_size)); // use the start index and the chunk size to determine the pixel to stop on

        filter::apply_convolution(*data->greyscale, *data->sobel, start_index, end_index); // do processing on the dataset
        
        pthread_barrier_wait(&workers_done); // wait for all workers to finish their chunks
    }
    return NULL;
}
