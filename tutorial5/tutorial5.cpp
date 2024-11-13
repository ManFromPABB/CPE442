#include "tutorial5.hpp"

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
        int channels = image.channels();
        for (int i = 0; i < image.rows; i++) {
            for (int j = 0; j < image.cols; j += 8) {
                int index = i * image.step + j * channels; // calculate data offset from current position/channel

                uint8x8x3_t rgb_vec = vld3_u8(&image.data[index]); // load 24 bytes of data representing 8 pixels into a uint8 8x3 matrix

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

                vst1_u8(&grey_image.ptr<uchar>(i)[j], grey_u8); // store all 8 grey pixels into their respective locations
            }
        }
    }

    void apply_convolution(cv::Mat image, cv::Mat sobel, int row) {
        for (int j = 1; j < image.cols - 1; j += 8) {
            
            int16x8_t gradX = vdupq_n_s16(0);
            int16x8_t gradY = vdupq_n_s16(0);

            for (int ki = -1; ki <= 1; ki++) {
                for (int kj = -1; kj <= 1; kj++) {
                    uint8x8_t anchor_vector = vld1_u8(&image.ptr<uchar>(row + ki)[j + kj]); // get BGR value for the pixel position + the convolution offset
                    int16x8_t anchor_vector_int = vreinterpretq_s16_u16(vmovl_u8(anchor_vector)); // cast uint8x8 to int16x8 in order for calculations to work
                    int16x8_t Gx_coeff = vdupq_n_s16(Gx[ki + 1][kj + 1]); // duplicate current Gx filter coefficient into all elements of int16x8 vector
                    int16x8_t Gy_coeff = vdupq_n_s16(Gy[ki + 1][kj + 1]); // duplicate current Gy filter coefficient into all elements of int16x8 vector
                    gradX = vmlaq_s16(gradX, anchor_vector_int, Gx_coeff); // vector multiply the pixels by Gx coefficients and accumulate to gradient vector
                    gradY = vmlaq_s16(gradY, anchor_vector_int, Gy_coeff); // vector multiply the pixels by Gy coefficients and accumulate to gradient vector
                }
            }

            int16x8_t absGradX = vabsq_s16(gradX); // find abs of gradient vector
            int16x8_t absGradY = vabsq_s16(gradY);
            int16x8_t gradMag = vaddq_s16(absGradX, absGradY); // approximate gradient magnitude from absolute value of component sums

            uint16x8_t gradMag_u = vreinterpretq_u16_s16(gradMag); // cast int16x8 to uint16x8
            uint8x8_t result_vec = vqmovn_u16(gradMag_u); // compute max on all vector elements such that they are less than 255

            vst1_u8(&sobel.ptr<uchar>(row)[j], result_vec); // set BGR elements of pixel to corresponding filtered value
        }
    }

    static inline int greyscale(int R, int G, int B) {
        return 0.2126 * R + 0.7152 * G + 0.0722 * B; // apply CCIR 601 greyscaling
    }

}

int main(int argc, char *argv[]) {

    if (argc != 2) {
        throw std::runtime_error("tutorial5 requires one argument, the video to open...\n");
    }

    // open the file specified by the input argument
    cv::VideoCapture capture(argv[1]);

    // setup PAPI library and make sure it's initialized
    if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT) {
        printf("PAPI Failed to Initialize\n");
        exit(1);
    }

    // declare 3 event counters for L1 data cache misses, L2 data cache misses, and total instructions executed
    int events[3];
    if (PAPI_event_name_to_code("perf::PERF_COUNT_HW_CACHE_L1D:MISS", &events[0]) != PAPI_OK)
    {
         printf("PAPI Failed to add L1 code\n");
    }
        if (PAPI_event_name_to_code("perf::PERF_COUNT_HW_CACHE_LL:MISS", &events[1]) != PAPI_OK)
    {
         printf("PAPI Failed to add L2 code\n");
    }
        if (PAPI_event_name_to_code("PERF_COUNT_HW_INSTRUCTIONS", &events[2]) != PAPI_OK)
    {
         printf("PAPI Failed to add INST CNT code\n");
    }
    long long values[3];
    int event_set = PAPI_NULL;
    if (PAPI_create_eventset(&event_set) != PAPI_OK) { // create the event set
        printf("PAPI Failed to create event set\n");
        exit(1);
    }

    // add each event counter to the set of measured events
    for (int i = 0; i < 3; i++) {
        if (PAPI_add_event(event_set, events[i]) != PAPI_OK) {
            printf("PAPI Failed to add event\n");
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

        for (int i = start_index; i < end_index; i++) {
            filter::apply_convolution(*data->greyscale, *data->sobel, i); // do processing on the dataset
        }
        
        pthread_barrier_wait(&workers_done); // wait for all workers to finish their chunks
    }
    return NULL;
}
