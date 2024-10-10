CC := g++ # name of the compiler to use
OUTPUT := sobel # name of the output file
LDFLAGS := -o $(OUTPUT) -lopencv_core -lopencv_highgui -lopencv_videoio -lpthread # linker flags (-o is optional)
CFLAGS := -g -Wall -Wpedantic -std=c++11 -I/usr/include/opencv4 -o0 # build flags

FILES := # add additional build files here
MAIN_FILE := tutorial3.cpp # name of the file containing 'main' function

default: all # default build rule

all: # main build rule
	$(CC) $(CFLAGS) $(FILES) $(MAIN_FILE) $(LDFLAGS)

clean: # clears output files
	rm -rf *.o *.so $(OUTPUT)
