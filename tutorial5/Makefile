CC := g++ # name of the compiler to use
OUTPUT := sobel # name of the output file
LDFLAGS := -o $(OUTPUT) -lopencv_core -lopencv_highgui -lopencv_videoio -lpthread -L:./libpapi.so -lpapi # linker flags (-o is optional)
CFLAGS := -g -Wall -Wpedantic -std=c++11 -I/usr/include/opencv4 -o0 -march=armv8-a # build flags

FILES := # add additional build files here
MAIN_FILE := tutorial5.cpp # name of the file containing 'main' function

default: all # default build rule

all: # main build rule
	$(CC) $(CFLAGS) $(FILES) $(MAIN_FILE) $(LDFLAGS)

clean: # clears output files
	rm -rf *.o *.so $(OUTPUT)
