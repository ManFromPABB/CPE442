CC := gcc # name of the compiler to use
OUTPUT := out.o # name of the output file
LDFLAGS := -o $(OUTPUT) # linker flags (-o is optional)
CFLAGS := -g -Wall -Wpedantic -std=c99 # build flags

FILES := # add additional build files here
MAIN_FILE := hello_world.c # name of the file containing 'main' function

default: all # default build rule

all: # main build rule
	$(CC) $(CFLAGS) $(FILES) $(MAIN_FILE) $(LDFLAGS)

clean: # clears output files
	rm -rf *.o *.so $(OUTPUT)
