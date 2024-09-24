This is a brief tutorial on how to use a makefile template to build your project
The template is designed to do a basic compilation for a hello world program, but the intent is for the makefile to be reusable and customizable for different projects

The below section will go over each line and elaborate on how to modify them for your needs.

CC: this variable is used to specify your preferred compiler. Most linux systems have gcc as a default and most macs use clang. C++ compilers such as g++ and clang++ may be used as well.

OUTPUT: This variable specifies the name of the output executable. In the case of the helloworld program, the name "out.out" is used. This can be changed to any name that's preferred

LDFLAGS: This variable specifies the linker flags used during compilation. Refer to your compiler's man page to see all available linker flags. The flag present is a placeholder

CFLAGS: This variable specifies the compilation flags. Refer to your compiler's man page to see all available compilation flags. The flags set enable additional warnings and tell the compiler to compile in c99 mode.

MAIN_FILE: This variable specifies the main c file name used in compilation. For the helloworld example, this is just "hello_world.c"

ADD_FILES: This variable specifies the additional c and header files used in compilation. The value is blank since the helloworld c program includes no additional files


Using the make file:

To compile your program, the command "make" or "make all" may be used. These both perform the same operation. When called, the makefile will instruct the your compiler to compile the primary file, alongside any additional headers and support c files with your specified compiler and linker flags to the output executable name specified.

If you'd like to recompile your program, the "make clean" command must be run prior. This command will delete the output executable, alongside any object or shared object files created during the compilation process.


